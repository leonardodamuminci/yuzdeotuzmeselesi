import os, json, time, math, requests, traceback
from datetime import datetime, timezone
from dateutil import tz
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# -------- Config --------
TIMEFRAME = "1d"          # sadece günlük mum
BARS      = 400           # geçmiş mum sayısı
TP_PCT    = 3             # bilgi amaçlı (mesajda gösteriyoruz)
SL_PCT    = 11
TIME_BARS = 15
LOW_TH    = 0.7           # RSI band katsayıları
HIGH_TH   = 1.3
RSI_LEN   = 14
SMA_LEN   = 14

# AI eşikleri (env'den)
PTH_BASE = float(os.getenv("PTH_BASE", "0.90"))
PTH_META = float(os.getenv("PTH_META","0.60"))
PTH_KNN  = float(os.getenv("PTH_KNN", "0.58"))
PTH_LSTM = float(os.getenv("PTH_LSTM","0.60"))
USE_LSTM = os.getenv("USE_LSTM","0") == "1"

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN","").strip()
TELEGRAM_CHATID = os.getenv("TELEGRAM_CHAT_ID","").strip()

IST = tz.gettz("Europe/Istanbul")

MODELS_DIR = Path("models")
BASE_M = MODELS_DIR / "base_model.pkl"
META_M = MODELS_DIR / "meta_model.pkl"
KNN_M  = MODELS_DIR / "knn_model.pkl"
LSTM_M = MODELS_DIR / "lstm_model.pkl"   # opsiyonel
SCALER = MODELS_DIR / "scaler.pkl"
FEATSJ = MODELS_DIR / "feat_config.json"

# -------- Utils --------
def rsi(series, length=14):
    s = series.astype(float)
    delta = s.diff()
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    roll_up   = pd.Series(gain, index=s.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(loss, index=s.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    r = 100 - (100 / (1 + rs))
    return r.fillna(method="bfill").fillna(50.0)

def fmt_ist(dt_utc):
    dt = dt_utc.astimezone(IST)
    return dt.strftime("%d/%m %H:%M") + " (UTC+03)"

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHATID:
        print("WARN: TELEGRAM_TOKEN/CHAT_ID yok; mesaj atlanıyor.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHATID, "text": text}
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception as e:
        print("Telegram error:", e)

def exchange_usdt_symbols():
    """Binance spot USDT sembollerini getir (TRADING)"""
    try:
        r = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=20)
        r.raise_for_status()
        data = r.json()
        out = []
        for s in data.get("symbols", []):
            if s.get("quoteAsset")=="USDT" and s.get("status")=="TRADING" and s.get("isSpotTradingAllowed", True):
                out.append(s["symbol"])
        # filtre: çok uzun/garip isimleri eleyelim
        out = [x for x in out if len(x) <= 12]
        out.sort()
        return out
    except Exception as e:
        print("ERR exchangeInfo:", e)
        # fallback: küçük örnek
        return ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT"]

def fetch_klines(symbol, interval="1d", limit=400):
    url = "https://api.binance.com/api/v3/klines"
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=p, timeout=20)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        raise RuntimeError("empty klines")
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(arr, columns=cols)
    df["open"]  = df["open"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"]= df["volume"].astype(float)
    df["time"]  = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["time","open","high","low","close","volume"]]

def build_features(df):
    # temel indikatörler
    rsi_c = rsi(df["close"], RSI_LEN)
    rsi_h = rsi(df["high"],  RSI_LEN)
    rsi_l = rsi(df["low"],   RSI_LEN)
    sma_r = rsi_c.rolling(SMA_LEN).mean().fillna(method="bfill")

    ret1 = df["close"].pct_change().fillna(0)
    vol  = df["volume"].rolling(20).mean().fillna(method="bfill")
    vstd = df["close"].pct_change().rolling(20).std().fillna(0)

    feats = pd.DataFrame({
        "rsi_c": rsi_c,
        "rsi_h": rsi_h,
        "rsi_l": rsi_l,
        "sma_r": sma_r,
        "ret1":  ret1,
        "vol_ma20": vol,
        "vstd20": vstd
    }, index=df.index)
    return feats

def detect_signals(df, feats):
    """wick-touch sinyali: aynı barda dokunuş kontrolü"""
    out = []
    rsi_h, rsi_l, sma = feats["rsi_h"], feats["rsi_l"], feats["sma_r"]
    thr_low  = sma * LOW_TH
    thr_high = sma * HIGH_TH

    # LONG: RSI_low <= sma*0.7
    long_mask  = (rsi_l <= thr_low)
    # SHORT: RSI_high >= sma*1.3
    short_mask = (rsi_h >= thr_high)

    for i in range(len(df)):
        if not pd.isfinite(feats["rsi_c"].iloc[i]): 
            continue
        if long_mask.iloc[i]:
            out.append(("LONG", i))
        if short_mask.iloc[i]:
            out.append(("SHORT", i))
    return out  # list of (side, idx)

def load_models():
    models = {}
    if BASE_M.exists():
        models["base"] = joblib.load(BASE_M)
    if META_M.exists():
        models["meta"] = joblib.load(META_M)
    if KNN_M.exists():
        models["knn"] = joblib.load(KNN_M)
    if USE_LSTM and LSTM_M.exists():
        models["lstm"] = joblib.load(LSTM_M)
    scaler=None
    if SCALER.exists():
        scaler = joblib.load(SCALER)
    if FEATSJ.exists():
        with open(FEATSJ,"r") as f:
            feat_cfg = json.load(f)
    else:
        feat_cfg = {"feats": ["rsi_c","rsi_h","rsi_l","sma_r","ret1","vol_ma20","vstd20"]}
    return models, scaler, feat_cfg["feats"]

def predict_probs(models, scaler, feats_row, feat_names):
    X = feats_row[feat_names].values.reshape(1,-1)
    if scaler is not None:
        X = scaler.transform(X)
    def p(mkey):
        m = models.get(mkey)
        if m is None: 
            return None
        # hem sklearn prob hem decision_function ihtimali
        if hasattr(m, "predict_proba"):
            # sınıf 1 (AL) olasılığı varsayalım
            pr = m.predict_proba(X)[0,1]
            return float(pr)
        elif hasattr(m, "decision_function"):
            z = m.decision_function(X)[0]
            # lojin benzeri sıkıştırma
            pr = 1/(1+math.exp(-z))
            return float(pr)
        else:
            y = m.predict(X)[0]
            return float(y)
    return {
        "p_base": p("base"),
        "p_meta": p("meta"),
        "p_knn":  p("knn"),
        "p_lstm": p("lstm"),
    }

def vote_decision(ps):
    """eşiklere göre AL / PAS"""
    ok = 0; total = 0
    if ps["p_base"] is not None: total+=1; ok += int(ps["p_base"]>=PTH_BASE)
    if ps["p_meta"] is not None: total+=1; ok += int(ps["p_meta"]>=PTH_META)
    if ps["p_knn"]  is not None: total+=1; ok += int(ps["p_knn"] >=PTH_KNN)
    if USE_LSTM and (ps["p_lstm"] is not None):
        total+=1; ok += int(ps["p_lstm"]>=PTH_LSTM)
    # kural: en az yarısı eşik üstü olsun
    if total==0:
        return "AL", "model yok (kural bazlı)"
    return ("AL","oylama: {}/{} eşik üstü".format(ok,total)) if ok*2>=total else ("PAS","oylama: {}/{} eşik altı".format(ok,total))

def volume_rank(df):
    # son 365 gün volüme toplamına göre basit sıralama değeri
    # (global sıralama için tüm semboller lazım; biz sembol içi normalize gösteriyoruz)
    last = df["volume"].tail(365)
    # kendi tarih aralığında gün sırası (yüksek hacim = küçük rank)
    # burada sadece “kaçıncı yüksek gün” bilgisini veririz
    ranks = last.rank(ascending=False, method="average")
    return int(ranks.iloc[-1])

def build_message(side, symbol, timeframe, scan_note, price, rsi_h, rsi_l, rsi_c,
                  p_base, p_meta, p_knn, p_lstm, decision, rule_txt,
                  tp_pct, sl_pct, time_bars, vol_rank_val, vol_total, ts_utc, sig_id):
    emoji = "🟢 LONG" if side=="LONG" else "🔴 SHORT"
    time_str = fmt_ist(ts_utc)
    p_base_s = f"{p_base:.2f}" if p_base is not None else "—"
    p_meta_s = f"{p_meta:.2f}" if p_meta is not None else "—"
    p_knn_s  = f"{p_knn:.2f}"  if p_knn  is not None else "—"
    p_lstm_s = f"{p_lstm:.2f}" if p_lstm is not None else "—"
    msg = (
f"⏱️ {time_str}\n"
f"{emoji} | {symbol} | {timeframe} (tarama: {scan_note})\n"
f"Fiyat: {price}\n"
f"RSI14(H/L/C): {rsi_h:.1f} / {rsi_l:.1f} / {rsi_c:.1f}\n"
f"p_base={p_base_s}  p_meta={p_meta_s}  p_knn={p_knn_s}  p_lstm={p_lstm_s}\n"
f"Karar: {decision}   {rule_txt}\n"
f"TP=+{tp_pct}%  SL=−{sl_pct}%  TIME={time_bars} bar\n"
f"Hacim sırası: {vol_rank_val}/{vol_total}\n"
f"ID: {sig_id}"
    )
    return msg

# -------- Main --------
def main():
    start_utc = datetime.now(timezone.utc)
    sym_list = exchange_usdt_symbols()
    vol_total_hint = len(sym_list)

    models, scaler, feat_names = load_models()

    header = (f"{start_utc.astimezone(IST).strftime('%Y-%m-%d %H:%M')} (UTC+03)\n"
              f"📡 Binance {TIMEFRAME.upper()} wick-touch (2 saatte 1)\n"
              f"⏳ RSI={RSI_LEN}, SMA={SMA_LEN} | Eşikler: {LOW_TH} / {HIGH_TH}")
    send_telegram(header)

    sent = 0
    for k, sym in enumerate(sym_list, 1):
        try:
            df = fetch_klines(sym, "1d", BARS)
            feats = build_features(df)
            sigs = detect_signals(df, feats)
            if not sigs:
                continue

            # sadece son bar sinyalini gönder (canlı kullanım)
            side, idx = sigs[-1]
            # sinyal zamanını bar kapanışı ile alıyoruz
            ts_utc = pd.to_datetime(df["time"].iloc[idx]).to_pydatetime().replace(tzinfo=timezone.utc)
            # fiyatı: sinyal barının close'unu göster
            price  = df["close"].iloc[idx]

            # AI olasılıkları
            ps = predict_probs(models, scaler, feats.iloc[idx], feat_names)
            decision, rule_txt = vote_decision(ps)

            # hacim "rank" (sembol-içi)
            vrank = volume_rank(df)

            msg = build_message(
                side=side, symbol=sym, timeframe="1D", scan_note="2s/1",
                price=price,
                rsi_h=feats["rsi_h"].iloc[idx],
                rsi_l=feats["rsi_l"].iloc[idx],
                rsi_c=feats["rsi_c"].iloc[idx],
                p_base=ps["p_base"], p_meta=ps["p_meta"], p_knn=ps["p_knn"], p_lstm=ps["p_lstm"],
                decision=("AL" if decision=="AL" else "PAS"),
                rule_txt=(f"(eşik: base≥{PTH_BASE:.2f}, meta≥{PTH_META:.2f}, knn≥{PTH_KNN:.2f}" + (f", lstm≥{PTH_LSTM:.2f}" if USE_LSTM else "") + ")"),
                tp_pct=TP_PCT, sl_pct=SL_PCT, time_bars=TIME_BARS,
                vol_rank_val=vrank, vol_total=vol_total_hint,
                ts_utc=ts_utc,
                sig_id=f"{sym}-{ts_utc.astimezone(IST).strftime('%Y%m%dT%H%M')}"
            )
            send_telegram(msg)
            sent += 1

            # çok uzun listeyi biraz yavaşlat
            if sent % 20 == 0:
                time.sleep(1)

        except requests.HTTPError as he:
            code = he.response.status_code if he.response is not None else "?"
            send_telegram(f"ERR {sym}: HTTP {code}")
        except Exception as e:
            print(sym, "ERR:", e)
            # debug amaçlı kısa log
            tb = "".join(traceback.format_exception_only(type(e), e)).strip()
            send_telegram(f"ERR {sym}: {tb}")

    if sent == 0:
        send_telegram("• İşlem yok")

if __name__ == "__main__":
    main()
