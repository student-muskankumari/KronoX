# src/data_loader.py
import pandas as pd
import yfinance as yf
import streamlit as st

@st.cache_data(show_spinner=False, ttl=60)   # refresh every 60 seconds
def fetch_yfinance(ticker, start=None, end=None, interval="1d"):
    """
    Fetch and normalize stock data from Yahoo Finance.
    Always returns columns: Date, Open, High, Low, Close, Volume
    TTL=60s so live mode gets fresh data each minute.
    """
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        group_by=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for '{ticker}' at interval '{interval}'.")

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns]

    df.columns = [str(c).strip().replace(" ", "_").capitalize() for c in df.columns]

    col_map = {}
    for col in df.columns:
        low = col.lower()
        if "open" in low:                          col_map[col] = "Open"
        elif "high" in low:                        col_map[col] = "High"
        elif "low" in low:                         col_map[col] = "Low"
        elif "close" in low and "adj" not in low:  col_map[col] = "Close"
        elif "volume" in low:                      col_map[col] = "Volume"
        elif "adj_close" in low:                   col_map[col] = "Close"

    df = df.rename(columns=col_map)

    if "Date" not in df.columns:
        possible_dates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if possible_dates:
            df.rename(columns={possible_dates[0]: "Date"}, inplace=True)
        else:
            raise ValueError(f"Could not find any 'Date' column in columns: {df.columns.tolist()}")

    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Yahoo returned data missing expected columns: {missing}\n"
            f"Columns found: {df.columns.tolist()}\n"
            f"Sample data:\n{df.head()}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, ttl=30)   # refresh every 30 seconds
def fetch_live_price(ticker: str) -> dict:
    """
    Fetch the current live/intraday price for a ticker.
    Returns dict with keys: price, change, change_pct, high, low, volume, time
    Falls back to last daily close if intraday unavailable.
    """
    try:
        tk   = yf.Ticker(ticker)
        info = tk.fast_info
        price = float(info.last_price) if hasattr(info, "last_price") and info.last_price else None

        if price is None or price == 0:
            # Fallback: last 2 days 1-minute data
            hist = tk.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        if price is None:
            return {"error": "Live price unavailable"}

        prev_close = float(info.previous_close) if hasattr(info, "previous_close") and info.previous_close else price
        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0.0

        day_high = float(info.day_high) if hasattr(info, "day_high") and info.day_high else price
        day_low  = float(info.day_low)  if hasattr(info, "day_low")  and info.day_low  else price
        volume   = int(info.shares) if hasattr(info, "shares") and info.shares else 0

        return {
            "price":      price,
            "change":     change,
            "change_pct": change_pct,
            "prev_close": prev_close,
            "day_high":   day_high,
            "day_low":    day_low,
            "time":       pd.Timestamp.now().strftime("%H:%M:%S"),
        }
    except Exception as e:
        return {"error": str(e)}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df