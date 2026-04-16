#!/usr/bin/env python3
"""
Multi-ticker evaluator — compatible with the new return-based RF model.
Computes the same engineered features used in train_model.py.

Usage:
    python evaluate.py --tickers RELIANCE.NS TCS.NS AAPL MSFT NVDA --model model.joblib
"""
import argparse
import os
import math
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# Feature engineering — must match train_model.py exactly
# ══════════════════════════════════════════════
NEW_FEATURES = [
    "Open_norm", "High_norm", "Low_norm", "Volume_norm",
    "Day", "Month", "Year",
    "Return_1d", "Return_5d", "Return_20d",
    "MA5_norm", "MA20_norm", "MA50_norm",
    "Volatility_10d", "Volatility_20d",
    "Price_momentum",
]

# Legacy features (old single-ticker model)
LEGACY_FEATURES = ["Open", "High", "Low", "Volume", "Day", "Month", "Year"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to an OHLCV DataFrame."""
    d = df.copy()
    d["Open_norm"]    = d["Open"]  / d["Close"]
    d["High_norm"]    = d["High"]  / d["Close"]
    d["Low_norm"]     = d["Low"]   / d["Close"]

    vol_ma = d["Volume"].rolling(20).mean().replace(0, 1)
    d["Volume_norm"]  = d["Volume"] / vol_ma

    d["Day"]          = pd.to_datetime(d.index).day
    d["Month"]        = pd.to_datetime(d.index).month
    d["Year"]         = pd.to_datetime(d.index).year

    d["Return_1d"]    = d["Close"].pct_change(1)
    d["Return_5d"]    = d["Close"].pct_change(5)
    d["Return_20d"]   = d["Close"].pct_change(20)

    d["MA5_norm"]     = d["Close"].rolling(5).mean()  / d["Close"]
    d["MA20_norm"]    = d["Close"].rolling(20).mean() / d["Close"]
    d["MA50_norm"]    = d["Close"].rolling(50).mean() / d["Close"]

    d["Volatility_10d"] = d["Return_1d"].rolling(10).std()
    d["Volatility_20d"] = d["Return_1d"].rolling(20).std()

    high_20 = d["Close"].rolling(20).max()
    low_20  = d["Close"].rolling(20).min()
    rng     = (high_20 - low_20).replace(0, 1)
    d["Price_momentum"] = (d["Close"] - low_20) / rng

    return d


def safe_yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data, tuple):
        for part in data:
            if isinstance(part, pd.DataFrame):
                data = part
                break
    if not isinstance(data, pd.DataFrame):
        raise RuntimeError(f"yfinance returned unexpected type: {type(data)}")
    return data


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("open","o"):       rename_map[c] = "Open"
        elif lc in ("high","h"):     rename_map[c] = "High"
        elif lc in ("low","l"):      rename_map[c] = "Low"
        elif lc in ("close","c"):    rename_map[c] = "Close"
        elif "volume" in lc:         rename_map[c] = "Volume"
        elif lc in ("adj close","adjclose"): rename_map[c] = "Close"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            df.index = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
    df = df.sort_index()
    return df


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom  = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def detect_feature_type(model, feature_list):
    """Detect whether the model uses new engineered features or legacy raw features."""
    if feature_list is None:
        return "legacy"
    if any(f in feature_list for f in ["Open_norm", "Return_1d", "MA5_norm"]):
        return "new"
    return "legacy"


def evaluate_ticker(model, ticker, start, end, out_dir="eval_out", feature_list=None):
    model_type = detect_feature_type(model, feature_list)
    print(f"[INFO] Evaluating {ticker} from {start.date()} to {end.date()} "
          f"(model type: {model_type})...")

    df = safe_yf_download(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df.empty:
        print(f"[WARN] No data for {ticker}")
        return None

    df = standardize_df(df)
    if not all(c in df.columns for c in ("Open", "High", "Low", "Close", "Volume")):
        print(f"[WARN] {ticker} missing OHLCV columns, skipping.")
        return None

    if model_type == "new":
        # ── New return-based model ──
        df_feat = engineer_features(df)
        df_feat = df_feat.dropna(subset=NEW_FEATURES)

        if df_feat.empty:
            print(f"[WARN] No usable rows for {ticker} after feature engineering.")
            return None

        # Predict next-day returns
        X = df_feat[NEW_FEATURES].values
        try:
            pred_returns = model.predict(X)
        except Exception:
            pred_returns = model.predict(X)

        # Convert predicted returns → predicted prices
        actual_closes = df_feat["Close"].values
        # predicted price[i] = actual_close[i] * (1 + pred_return[i])
        pred_prices = actual_closes * (1 + pred_returns)

        # Align: we predict next-day close, so shift actual by 1
        y_true = actual_closes[1:]     # next-day actual
        y_pred = pred_prices[:-1]      # prediction made today for tomorrow
        idx    = df_feat.index[1:]

        n = min(len(y_true), len(y_pred))
        if n == 0:
            print(f"[WARN] No aligned rows for {ticker}.")
            return None

        y_true = y_true[:n]
        y_pred = y_pred[:n]
        idx    = idx[:n]

    else:
        # ── Legacy absolute-price model ──
        df["Day"]   = df.index.day
        df["Month"] = df.index.month
        df["Year"]  = df.index.year

        feat = feature_list if feature_list else LEGACY_FEATURES
        missing = [f for f in feat if f not in df.columns]
        if missing:
            print(f"[WARN] Missing features for {ticker}: {missing}. Skipping.")
            return None

        Xdf = df[feat].dropna()
        if Xdf.empty:
            return None

        y_series = df.loc[Xdf.index, "Close"]
        try:
            y_pred = model.predict(Xdf)
        except Exception:
            y_pred = model.predict(Xdf.values)

        y_pred = np.array(y_pred).ravel()
        y_true = np.array(y_series.values).ravel()
        n = min(len(y_pred), len(y_true))
        y_pred = y_pred[:n];  y_true = y_true[:n]
        idx = Xdf.index[:n]

    # ── Metrics ──
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    accuracy_pct = max(0.0, 100.0 - mape)

    prev  = np.concatenate(([np.nan], y_true[:-1]))[:n]
    mask  = ~np.isnan(prev)
    dir_acc = float(np.mean(
        np.sign(y_true[mask] - prev[mask]) == np.sign(y_pred[mask] - prev[mask])
    )) if mask.sum() > 0 else 0.0

    # ── Simple long-only backtest ──
    if n >= 2:
        returns = []
        for i in range(n - 1):
            pr = y_true[i]
            p  = y_pred[i]
            t  = y_true[i + 1]
            returns.append((t / pr - 1.0) if (p > pr and pr != 0) else 0.0)
        cum  = np.cumprod([1 + r for r in returns]) if returns else np.array([])
        if len(returns) > 1:
            mean_r = float(np.mean(returns))
            std_r  = float(np.std(returns, ddof=1))
            sharpe = (mean_r / std_r * math.sqrt(252.0)) if std_r > 0 else 0.0
            peak   = np.maximum.accumulate(cum) if len(cum) else np.array([])
            max_dd = float(np.min((cum - peak) / peak)) if len(peak) else 0.0
        else:
            sharpe = 0.0; max_dd = 0.0
    else:
        sharpe = 0.0; max_dd = 0.0

    # ── Save outputs ──
    os.makedirs(out_dir, exist_ok=True)
    pred_df = pd.DataFrame({"Date": idx, "Close_true": y_true, "Close_pred": y_pred}).set_index("Date")
    csv_path = os.path.join(out_dir, f"{ticker}_predictions.csv")
    pred_df.to_csv(csv_path)

    # Also save as predictions.csv for backtesting module
    pd.DataFrame({
        "date":            [str(d.date()) if hasattr(d, "date") else str(d) for d in idx],
        "actual_price":    y_true.tolist(),
        "predicted_price": y_pred.tolist(),
    }).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    # ── Per-ticker plot ──
    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(idx, y_true, label="True Close",      linewidth=1.6)
    plt.plot(idx, y_pred, label="Predicted Close", linestyle="--", linewidth=1.2)
    plt.title(f"{ticker} — Predicted vs True Close")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(); plt.grid(alpha=0.4)
    png_path = os.path.join(out_dir, f"{ticker}_pred_vs_true.png")
    plt.tight_layout(); plt.savefig(png_path)
    try: plt.close()
    except: pass

    result = {
        "Ticker": ticker, "Rows": int(n),
        "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2),
        "MAPE_pct": float(mape), "Accuracy_pct": float(accuracy_pct),
        "DirAcc": float(dir_acc), "Sharpe": float(sharpe), "MaxDD": float(max_dd),
        "csv": csv_path, "png": png_path,
    }
    print(f"[INFO] {ticker} done: MAE={mae:.4f}, RMSE={rmse:.4f}, "
          f"R2={r2:.4f}, DirAcc={dir_acc*100:.1f}%, Accuracy={accuracy_pct:.2f}%")
    return result


def main():
    p = argparse.ArgumentParser(description="Multi-ticker model evaluator")
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--model",   required=True, help="Path to joblib model file")
    p.add_argument("--days",    type=int, default=365 * 2, help="Lookback days (default 2 years)")
    p.add_argument("--out",     default="eval_out", help="Output folder")
    p.add_argument("--features",default="features.joblib", help="Optional saved feature list")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    model = joblib.load(args.model)
    print(f"[INFO] Loaded model from {args.model}")

    feature_list = None
    if args.features and os.path.exists(args.features):
        try:
            feature_list = joblib.load(args.features)
            print(f"[INFO] Loaded features from {args.features}: {feature_list}")
        except Exception as e:
            print(f"[WARN] Could not load features.joblib: {e}")

    end   = datetime.today()
    start = end - timedelta(days=args.days)

    results = []
    plt.figure(figsize=(14, 7))
    for ticker in args.tickers:
        res = evaluate_ticker(model, ticker, start, end, out_dir=args.out, feature_list=feature_list)
        if res:
            results.append(res)
            pred_df = pd.read_csv(res["csv"], index_col=0, parse_dates=True)
            tv = pred_df["Close_true"].values
            pv = pred_df["Close_pred"].values
            if len(tv) == 0 or tv[0] == 0 or pv[0] == 0:
                continue
            days = np.arange(len(tv))
            plt.plot(days, tv / tv[0], label=f"{ticker} True")
            plt.plot(days, pv / pv[0], linestyle="--", label=f"{ticker} Pred")

    plt.title("Multi-Ticker Predicted vs True (normalized)")
    plt.xlabel("Days (index)"); plt.ylabel("Normalized Price")
    plt.legend(ncol=2, fontsize="small"); plt.grid(alpha=0.4)
    combined_png = os.path.join(args.out, "multi_ticker_comparison.png")
    plt.tight_layout(); plt.savefig(combined_png, dpi=200)
    try: plt.show()
    except: pass

    if results:
        df_res = pd.DataFrame(results)
        summary_csv = os.path.join(args.out, "multi_ticker_summary.csv")
        df_res.to_csv(summary_csv, index=False)
        display_cols = ["Ticker","Rows","MAE","RMSE","R2","Accuracy_pct","MAPE_pct","DirAcc","Sharpe","MaxDD"]
        print(f"\n===== MODEL PERFORMANCE SUMMARY =====")
        print(df_res[display_cols].to_string(index=False, float_format="%.4f"))
        print(f"\n[INFO] Average Accuracy : {df_res['Accuracy_pct'].mean():.2f}%")
        print(f"[INFO] Avg Directional Acc: {df_res['DirAcc'].mean()*100:.2f}%")
        print(f"[INFO] Saved summary: {summary_csv}")
        print(f"[INFO] Saved chart:   {combined_png}")


if __name__ == "__main__":
    main()
