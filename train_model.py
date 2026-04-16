"""
Kronos — Improved Multi-Ticker Training Script
Improvements over v1:
  - RSI (14-period) feature
  - MACD signal line feature
  - Bollinger Band position feature
  - Volume spike feature
  - GradientBoostingRegressor (better directional accuracy than RandomForest)
  - Larger n_estimators + tuned hyperparams
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# TICKERS (50 Indian + 50 International)
# ══════════════════════════════════════════════════════════════
INDIAN_TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","HCLTECH.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","BAJFINANCE.NS",
    "NESTLEIND.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","JSWSTEEL.NS",
    "TATASTEEL.NS","ADANIENT.NS","ADANIPORTS.NS","BAJAJFINSV.NS",
    "DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","EICHERMOT.NS","HEROMOTOCO.NS",
    "BRITANNIA.NS","COALINDIA.NS","GRASIM.NS","INDUSINDBK.NS",
    "SBILIFE.NS","HDFCLIFE.NS","TECHM.NS","BPCL.NS","HINDALCO.NS",
    "APOLLOHOSP.NS","TATACONSUM.NS","PIDILITIND.NS","HAVELLS.NS",
    "DABUR.NS","MARICO.NS","BAJAJ-AUTO.NS",
]
INTL_TICKERS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","CRM",
    "CSCO","ADBE","QCOM","AMD","INTC","JPM","BAC","V","MA","GS","BRK-B",
    "UNH","JNJ","LLY","ABBV","MRK","PFE","WMT","KO","PEP","MCD","PG",
    "XOM","CVX","HD","COST","NKE","TSM","ASML","SAP","NVO","SHEL","BP",
    "UL","RIO","BHP","TM","SNY","BTI","NFLX",
]

ALL_TICKERS = INDIAN_TICKERS + INTL_TICKERS
START_DATE  = "2018-01-01"
END_DATE    = "2025-11-05"

# ══════════════════════════════════════════════════════════════
# IMPROVED FEATURES  (22 total vs 16 before)
# ══════════════════════════════════════════════════════════════
FEATURES = [
    # Price-normalised OHLCV
    "Open_norm","High_norm","Low_norm","Volume_norm",
    # Date
    "Day","Month","Year",
    # Returns
    "Return_1d","Return_5d","Return_20d",
    # Moving averages normalised
    "MA5_norm","MA20_norm","MA50_norm",
    # Volatility
    "Volatility_10d","Volatility_20d",
    # Momentum
    "Price_momentum",
    # NEW: RSI
    "RSI_14",
    # NEW: MACD signal
    "MACD_signal",
    # NEW: Bollinger Band position (0=at lower band, 1=at upper band)
    "BB_position",
    # NEW: Volume spike (volume / 50-day avg)
    "Volume_spike",
    # NEW: Average True Range normalised
    "ATR_norm",
]

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def download_and_engineer(ticker):
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         progress=False, auto_adjust=True)
    except Exception as e:
        return None
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "open":        col_map[c] = "Open"
        elif lc == "high":      col_map[c] = "High"
        elif lc == "low":       col_map[c] = "Low"
        elif lc == "close":     col_map[c] = "Close"
        elif "volume" in lc:    col_map[c] = "Volume"
    df = df.rename(columns=col_map)

    required = ["Open","High","Low","Close","Volume"]
    if not all(c in df.columns for c in required):
        return None
    df = df.dropna(subset=required).copy()
    if len(df) < 100:
        return None

    # ── Date ──
    df["Day"]   = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"]  = df["Date"].dt.year

    # ── Price normalised ──
    df["Open_norm"]  = df["Open"]  / df["Close"]
    df["High_norm"]  = df["High"]  / df["Close"]
    df["Low_norm"]   = df["Low"]   / df["Close"]

    # ── Moving averages ──
    df["MA5_norm"]   = df["Close"].rolling(5).mean()  / df["Close"]
    df["MA20_norm"]  = df["Close"].rolling(20).mean() / df["Close"]
    df["MA50_norm"]  = df["Close"].rolling(50).mean() / df["Close"]

    # ── Volume ──
    vol_ma20 = df["Volume"].rolling(20).mean().replace(0, 1)
    vol_ma50 = df["Volume"].rolling(50).mean().replace(0, 1)
    df["Volume_norm"]  = df["Volume"] / vol_ma20
    df["Volume_spike"] = df["Volume"] / vol_ma50     # NEW

    # ── Returns ──
    df["Return_1d"]  = df["Close"].pct_change(1)
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["Return_20d"] = df["Close"].pct_change(20)

    # ── Volatility ──
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    df["Volatility_20d"] = df["Return_1d"].rolling(20).std()

    # ── Price momentum (20-day high/low range) ──
    h20 = df["Close"].rolling(20).max()
    l20 = df["Close"].rolling(20).min()
    df["Price_momentum"] = (df["Close"] - l20) / (h20 - l20 + 1e-9).replace(0, 1)

    # ── RSI (14) ── NEW
    df["RSI_14"] = rsi(df["Close"], 14) / 100.0   # normalise to 0-1

    # ── MACD signal ── NEW
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    df["MACD_signal"] = (macd - signal_line) / (df["Close"] + 1e-9)  # normalised

    # ── Bollinger Band position ── NEW
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    df["BB_position"] = (df["Close"] - bb_dn) / ((bb_up - bb_dn) + 1e-9).replace(0, 1)

    # ── ATR normalised ── NEW
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR_norm"] = tr.rolling(14).mean() / (df["Close"] + 1e-9)

    # ── Target: next-day return ──
    df["Target_return"] = df["Return_1d"].shift(-1)

    df = df.dropna(subset=FEATURES + ["Target_return"])
    if len(df) < 60:
        return None
    df["Ticker"] = ticker
    return df

# ══════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("Kronos Training — Improved Model (22 features + GBM)")
print(f"Tickers: {len(ALL_TICKERS)} | Period: {START_DATE} → {END_DATE}")
print("=" * 65)

frames  = []
failed  = []
ok_i = ok_intl = 0

print("\n── Indian Tickers ──")
for t in INDIAN_TICKERS:
    print(f"  {t}...", end=" ", flush=True)
    f = download_and_engineer(t)
    if f is not None:
        frames.append(f); ok_i += 1
        print(f"✓ ({len(f)} rows)")
    else:
        failed.append(t); print("✗")

print("\n── International Tickers ──")
for t in INTL_TICKERS:
    print(f"  {t}...", end=" ", flush=True)
    f = download_and_engineer(t)
    if f is not None:
        frames.append(f); ok_intl += 1
        print(f"✓ ({len(f)} rows)")
    else:
        failed.append(t); print("✗")

print(f"\n── Summary: {ok_i} Indian + {ok_intl} International = {ok_i+ok_intl} tickers")
if failed:
    print(f"   Failed: {failed}")

if not frames:
    raise RuntimeError("No data downloaded.")

all_data = pd.concat(frames, ignore_index=True)
print(f"   Total rows: {len(all_data):,}")

# ══════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
X = all_data[FEATURES].values
y = all_data["Target_return"].values

split   = int(len(X) * 0.80)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
print(f"\n   Train: {len(X_tr):,} | Test: {len(X_te):,}")

# ══════════════════════════════════════════════════════════════
# TRAIN — GradientBoostingRegressor
# Outperforms RandomForest for directional accuracy on financial data
# because it minimises residuals sequentially, learning subtle patterns
# ══════════════════════════════════════════════════════════════
print("\n── Training GradientBoostingRegressor ──")
print("   (300 trees, max_depth=5 — may take 5-10 minutes)")

model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=10,
    max_features=0.7,
    random_state=42,
    verbose=0,
)
model.fit(X_tr, y_tr)
print("   Done.")

# ══════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════
y_pred   = model.predict(X_te)
mae      = mean_absolute_error(y_te, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_te, y_pred))
r2       = r2_score(y_te, y_pred)
dir_acc  = np.mean(np.sign(y_pred) == np.sign(y_te)) * 100

print(f"\n{'='*65}")
print("MODEL PERFORMANCE")
print(f"{'='*65}")
print(f"  MAE              : {mae*100:.4f}%")
print(f"  RMSE             : {rmse_val*100:.4f}%")
print(f"  R²               : {r2:.4f}")
print(f"  Directional Acc. : {dir_acc:.2f}%")

# Feature importance
imp = pd.Series(model.feature_importances_, index=FEATURES)
print(f"\nTop 10 features:")
for f, v in imp.sort_values(ascending=False).head(10).items():
    print(f"  {f:<22} {v:.4f}")

# Per-ticker dir acc
test_df = all_data.iloc[split:].copy()
test_df["y_pred"] = y_pred
test_df["y_true"] = y_te
test_df["correct"] = np.sign(test_df["y_pred"]) == np.sign(test_df["y_true"])
ticker_acc = test_df.groupby("Ticker")["correct"].mean().sort_values(ascending=False)*100
print(f"\nPer-ticker directional accuracy:")
for t, acc in ticker_acc.items():
    bar = "█" * int(acc/5)
    print(f"  {t:<22} {acc:5.1f}%  {bar}")

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
joblib.dump(model,    "model.joblib")
joblib.dump(FEATURES, "features.joblib")

print(f"\n{'='*65}")
print("✅ Saved model.joblib and features.joblib")
print(f"   Model: GradientBoostingRegressor (22 features)")
print(f"   Tickers: {ok_i+ok_intl} | Rows: {len(all_data):,}")
print(f"{'='*65}")