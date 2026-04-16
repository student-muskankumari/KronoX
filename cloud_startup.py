"""
cloud_startup.py
Runs automatically on Streamlit Cloud to:
  1. Set up Kronos model (clone repo + write path config)
  2. Train a lightweight ML model if model.joblib is missing

Import this at the top of app_hourly.py with:
    import cloud_startup
"""

import os, sys, json, subprocess, logging

log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ── 1. Kronos setup ───────────────────────────────────────────
def _setup_kronos_cloud():
    src_dir  = os.path.join(BASE, "src")
    kdir     = os.path.join(src_dir, "kronos")
    cfg_path = os.path.join(src_dir, "kronos_path.json")
    os.makedirs(src_dir, exist_ok=True)

    # Already configured?
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            info = json.load(f)
        if os.path.exists(os.path.join(
                info.get("kronos_root",""), "model", "__init__.py")):
            return  # already good

    # Clone if not present
    if not os.path.exists(os.path.join(kdir, "model", "__init__.py")):
        log.info("[startup] Cloning Kronos repo …")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/shiyu-coder/Kronos", kdir],
            check=False, capture_output=True
        )

    # Write config
    init_py = os.path.join(kdir, "model", "__init__.py")
    if os.path.exists(init_py):
        with open(cfg_path, "w") as f:
            json.dump({"kronos_root": kdir, "model_init": init_py}, f)
        log.info("[startup] kronos_path.json written")

# ── 2. Lightweight model training if model.joblib missing ────
def _train_if_missing():
    if os.path.exists(os.path.join(BASE, "model.joblib")):
        return
    log.info("[startup] model.joblib not found — training lightweight model …")
    try:
        import yfinance as yf
        import numpy as np
        import pandas as pd
        import joblib
        from sklearn.ensemble import GradientBoostingRegressor

        # Train on a small set of 5 tickers (fast, ~2 min on cloud)
        TICKERS = ["RELIANCE.NS", "TCS.NS", "AAPL", "MSFT", "GOOGL"]
        FEATURES = [
            "Open_norm","High_norm","Low_norm","Volume_norm",
            "Day","Month","Year",
            "Return_1d","Return_5d","Return_20d",
            "MA5_norm","MA20_norm","MA50_norm",
            "Volatility_10d","Volatility_20d","Price_momentum",
        ]
        frames = []
        for t in TICKERS:
            try:
                df = yf.download(t, start="2020-01-01",
                                 end="2024-12-31", progress=False,
                                 auto_adjust=True)
                if df is None or df.empty or len(df) < 100:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"])
                c = df["Close"]
                df["Open_norm"]      = df["Open"]  / c
                df["High_norm"]      = df["High"]  / c
                df["Low_norm"]       = df["Low"]   / c
                vm = df["Volume"].rolling(20).mean().replace(0,1)
                df["Volume_norm"]    = df["Volume"] / vm
                df["Day"]   = df["Date"].dt.day
                df["Month"] = df["Date"].dt.month
                df["Year"]  = df["Date"].dt.year
                df["Return_1d"]  = c.pct_change(1, fill_method=None)
                df["Return_5d"]  = c.pct_change(5, fill_method=None)
                df["Return_20d"] = c.pct_change(20, fill_method=None)
                df["MA5_norm"]   = c.rolling(5).mean()  / c
                df["MA20_norm"]  = c.rolling(20).mean() / c
                df["MA50_norm"]  = c.rolling(50).mean() / c
                df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
                df["Volatility_20d"] = df["Return_1d"].rolling(20).std()
                h20 = c.rolling(20).max(); l20 = c.rolling(20).min()
                df["Price_momentum"] = (c - l20) / ((h20 - l20).replace(0,1))
                df["Target"] = df["Return_1d"].shift(-1)
                df = df.dropna(subset=FEATURES + ["Target"])
                frames.append(df)
            except Exception as e:
                log.warning(f"[startup] {t} failed: {e}")

        if not frames:
            log.warning("[startup] No data downloaded — skipping training")
            return

        all_data = pd.concat(frames, ignore_index=True)
        X = all_data[FEATURES].values
        y = all_data["Target"].values
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42)
        model.fit(X, y)
        joblib.dump(model, os.path.join(BASE, "model.joblib"))
        joblib.dump(FEATURES, os.path.join(BASE, "features.joblib"))
        log.info("[startup] ✅ Lightweight model trained and saved")
    except Exception as e:
        log.warning(f"[startup] Training failed: {e}")


# ── Run on import ─────────────────────────────────────────────
_setup_kronos_cloud()
_train_if_missing()