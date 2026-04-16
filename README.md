# 📊 Kronos — Live Stock Forecasting App

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GBM-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A real-time stock forecasting web application powered by the Kronos-mini foundation model and a custom-trained GradientBoosting ML model. Supports 100+ Indian (NSE/BSE) and International stocks with live price feeds, multi-step prediction, and backtesting.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Backtesting](#-backtesting) • [Model Training](#-model-training) • [Kronos Integration](#-kronos-integration) • [Tech Stack](#-tech-stack) • [FAQ](#-faq)

</div>

---

## 📸 Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│  Kronos – Live Stock Forecasting App                        │
│                                                             │
│  RELIANCE.NS  ₹1,344.10  ▲ +30.10 (+2.29%)                │
│                                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │  LIVE PRICE  │ │   PREDICTED  │ │     RMSE     │       │
│  │  ₹1,344.10  │ │  ₹1,321.83  │ │   27.3875    │       │
│  │ ▲ +2.29%    │ │  ▼ -1.66%   │ │  MAPE 1.76%  │       │
│  └──────────────┘ └──────────────┘ └──────────────┘       │
│                                                             │
│  [Live RELIANCE.NS Forecast Chart]                          │
│  Historical ──── Predicted - - - Confidence Band ░░        │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🔮 Prediction Engine
- **Three-tier prediction chain**: Kronos-mini → GBM/RF ML Model → Linear fallback
- **Multi-step forecasting**: 1 Week, 2 Weeks, 1 Month, 3 Months, 6 Months ahead
- **22 engineered features**: RSI, MACD, Bollinger Bands, ATR, Volume Spike + base OHLCV features
- **Confidence bands**: ±(RMSE × Margin) shown as shaded region around predictions

### 📡 Live Data
- **Real-time price feed** via yfinance fast_info
- **Live price bar**: Current price, change %, Day High/Low, Prev Close, timestamp
- **Auto-refresh**: Configurable refresh rate (5s to 600s) with live countdown timer
- **Market-aware**: Graceful fallback when market is closed

### 📈 Interactive Charts
- **Dark-themed Plotly charts** with zoom, pan, hover tooltips
- **Balanced view**: Shows 5× prediction horizon of history for clear comparison
- **Forecast divider**: Vertical annotation line marking where prediction starts
- **Equity curve**: Strategy vs real Buy & Hold with green/red shading

### 💰 Backtesting
- **Signal-based strategy**: ML model generates buy/sell signals on historical data
- **Configurable parameters**: Signal threshold, MA20 trend filter, 3% stop-loss
- **Full metrics**: Sharpe ratio, max drawdown, win rate, avg profit, avg hold duration
- **Trade log**: BUY/SELL/STOP trades only (HOLDs filtered out)
- **Signal distribution**: Donut chart showing BUY/SELL/HOLD signal breakdown

### 🌍 Market Coverage
- **50 Indian NSE stocks** (full Nifty 50 coverage)
- **48 International stocks** (S&P 500 top companies)
- **Intervals**: 1d, 1h, 1wk
- **History**: Up to 5 years (daily), 60 days (hourly)

---

## 🏗️ Architecture

```
kronos_patched/
│
├── app_hourly.py                  ← Main Streamlit app
├── train_model.py                 ← ML model training (run once)
├── evaluate.py                    ← Model evaluation script
├── setup_kronos.py                ← One-time Kronos setup
│
├── model.joblib                   ← Trained GBM model (22 features)
├── features.joblib                ← Feature names list
│
├── src/
│   ├── data_loader.py             ← yfinance data + live price
│   ├── forecast_utils.py          ← Business date generation
│   ├── model_handler_kronos.py    ← Kronos-mini integration
│   └── kronos_path.json           ← Auto-generated path config
│       kronos/                    ← Cloned Kronos repo
│           model/
│               __init__.py        ← KronosPredictor API
│               kronos.py          ← Transformer architecture
│               module.py          ← Attention layers
│
└── eval_out/
    └── predictions.csv            ← Saved evaluation predictions
```

### Data Flow

```
User Input (ticker, interval, horizon)
        │
        ▼
fetch_yfinance() ──────────────────── Historical OHLCV data
fetch_live_price() ─────────────────  Real-time price
        │
        ▼
do_forecast()
  ├─► predict_with_kronos()           Priority 1: Foundation model
  │         └─ _clean_df_for_kronos() 7-step NaN cleaning
  │
  ├─► ml_forecast()                   Priority 2: GBM/RF model
  │         └─ eng()                  22 feature engineering
  │
  └─► linear_fallback()               Priority 3: Last resort
        │
        ▼
backtest()                            Historical signal simulation
        │
        ▼
render()                              Charts + Metrics → Browser
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Git (for Kronos setup)
- 2 GB RAM minimum (4 GB recommended for Kronos-mini)

### Step 1 — Clone or Download the Project

```powershell
cd C:\Users\YourName\Desktop
# Download and extract the project folder
```

### Step 2 — Create Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```powershell
pip install streamlit plotly pandas numpy scikit-learn joblib yfinance
pip install einops rotary-embedding-torch huggingface-hub
```

Or install everything at once:

```powershell
pip install streamlit plotly pandas numpy scikit-learn joblib yfinance einops rotary-embedding-torch huggingface-hub
```

### Step 4 — Set Up Kronos Model (Optional but Recommended)

This downloads and configures the Kronos-mini foundation model for predictions.

```powershell
python setup_kronos.py
```

Expected output:
```
============================================================
Kronos Setup
============================================================
[1/4] Cloning Kronos repository …
   ✓  Cloned to src/kronos/
[2/4] Locating model package …
   ✓  Found model package at: src/kronos/model/
[3/4] Installing dependencies …
   ✓  Done
[4/4] Verifying import …
   ✓  model package verified
   ✓  Config saved to src/kronos_path.json
✅  Setup complete!
```

> **Note:** Model weights (~32 MB total) download from HuggingFace on first prediction run.

### Step 5 — Train the ML Model (Optional)

If `model.joblib` is not present, train it:

```powershell
python train_model.py
```

> **Takes 10–15 minutes.** Downloads price data for 98 tickers (2018–2025) and trains a GradientBoostingRegressor. Saves `model.joblib` and `features.joblib`.

### Step 6 — Launch the App

```powershell
streamlit run app_hourly.py
```

Open browser at: **http://localhost:8501**

---

## 🎮 Usage

### Sidebar Controls

| Control | Description | Options |
|---|---|---|
| **Ticker** | Yahoo Finance symbol | e.g. `RELIANCE.NS`, `TCS.NS`, `AAPL`, `MSFT` |
| **Interval** | Bar timeframe | `1d` (daily), `1h` (hourly), `1wk` (weekly) |
| **Date Range** | History to load | 1 Week → 5 Years (filtered by interval limits) |
| **Horizon** | Steps to predict | 1 Week → 6 Months (daily) / 12h → 48h (hourly) |
| **Margin** | Confidence band width | 0.5× to 5.0× RMSE |
| **Start** | Run forecast | Green button |
| **Stop** | Stop live mode | Red button |
| **Enable Live Mode** | Auto-refresh | Toggle |
| **Refresh every (s)** | Refresh interval | 5–600 seconds |

### Backtest Strategy Controls

| Control | Description | Default |
|---|---|---|
| **Signal Threshold (%)** | Min predicted return to trigger trade | 0.15% |
| **MA20 Trend Filter** | Only BUY when price > 20-day MA | ✅ On |
| **Stop Loss (3%)** | Exit if position drops 3% | ❌ Off |
| **Starting Capital** | Simulated capital | ₹1,00,000 |

### Ticker Formats

```
Indian NSE:    RELIANCE.NS   TCS.NS   HDFCBANK.NS   INFY.NS
Indian BSE:    RELIANCE.BO   TCS.BO
US Stocks:     AAPL          MSFT     NVDA          GOOGL
European:      ASML          SAP      NVO
Asian:         TSM           TM
Crypto:        BTC-USD       ETH-USD
```

### Live Mode Workflow

```
1. Enter ticker + settings
2. Click Start → initial forecast renders
3. Enable Live Mode checkbox
4. App auto-refreshes every N seconds:
   - Fetches fresh price data
   - Re-runs prediction
   - Updates live price bar
   - Overwrites chart in-place (no page reload)
```

---

## 💰 Backtesting

### How It Works

The backtest simulates trading on **historical data** using the ML model's 1-step signals:

```
For each historical bar:
  model.predict(features[bar]) → predicted_return

  if predicted_return > threshold AND price > MA20:
      → BUY  (spend all capital at current price)

  elif predicted_return < -threshold:
      → SELL (close position at next bar's price)

  elif price drops 3% from entry (if stop-loss enabled):
      → STOP (emergency exit)

  else:
      → HOLD (do nothing)
```

### Understanding the Metrics

| Metric | What it means | Good value |
|---|---|---|
| **Final Capital** | Ending portfolio value | > Starting Capital |
| **Strategy Return** | Total % gain/loss | Positive |
| **Buy & Hold** | Passive hold benchmark | Comparison only |
| **vs B&H badge** | Strategy vs passive | Positive = outperforming |
| **Max Drawdown** | Worst peak-to-trough drop | Closer to 0% |
| **Win Rate** | % of trades profitable | > 50% |
| **Sharpe Ratio** | Return per unit of risk | > 1.0 ideal |
| **Total Trades** | Number of BUY entries | 5–20 per year typical |
| **Avg Profit** | Average ₹ per closed trade | Positive |
| **Avg Hold** | Average bars held | — |

### Equity Curve Chart

```
Blue line  = Strategy cumulative return %
Grey dash  = Real Buy & Hold return %
Green fill = Periods where strategy beats B&H
Red fill   = Periods where strategy underperforms B&H
Flat line  = Strategy is in cash (no open position)
```

### Signal Distribution Donut

Shows what % of historical bars had each signal type. Typical values:
- **93% HOLD** — normal; model only trades when confident
- **5% BUY** — long positions triggered
- **2% SELL** — short signals / exits

### Tuning Tips

```
Too few trades?   → Lower Signal Threshold to 0.05–0.10%
                  → Disable MA20 Trend Filter
Too many losses?  → Raise Signal Threshold to 0.3–0.5%
                  → Enable Stop Loss
                  → Enable MA20 Filter
Market is falling?→ High HOLD% is correct — model avoiding bad trades
```

---

## 🤖 Model Training

### Training the Improved Model

```powershell
python train_model.py
```

### What Gets Trained

**Model:** `GradientBoostingRegressor` (300 trees, max_depth=5, lr=0.05)

**Target:** Next day's return % (supervised regression)

**Training Data:**
- 50 Indian NSE tickers (full Nifty 50)
- 48 International tickers (top S&P 500 companies)
- Period: 2018-01-01 to 2025-11-05
- ~186,000 training rows total

### Feature Engineering (22 Features)

| Category | Features |
|---|---|
| Price ratios | Open_norm, High_norm, Low_norm |
| Volume | Volume_norm, Volume_spike |
| Calendar | Day, Month, Year |
| Returns | Return_1d, Return_5d, Return_20d |
| Moving averages | MA5_norm, MA20_norm, MA50_norm |
| Volatility | Volatility_10d, Volatility_20d |
| Momentum | Price_momentum |
| Technical | RSI_14, MACD_signal, BB_position, ATR_norm |

### Expected Performance

| Metric | RandomForest (v1) | GradientBoosting (v2) |
|---|---|---|
| Directional Accuracy | ~52–53% | ~55–58% |
| MAE | ~1.23% | ~1.05% |
| Training Time | ~5 min | ~15 min |

### Why GradientBoosting?

RandomForest builds trees independently and averages them. GradientBoosting builds each tree to correct the errors of the previous ones — this iterative error correction gives better directional accuracy on financial time series.

---

## 🔭 Kronos Integration

Kronos is a decoder-only foundation model pre-trained on financial K-line (OHLCV) sequences — think GPT but for stock prices instead of text.

### How Kronos Works

```
OHLCV history (2048 bars max)
        │
        ▼
KronosTokenizer
  Converts continuous prices → discrete tokens
  (like text tokenization but for price levels)
        │
        ▼
Kronos Transformer (autoregressive decoder)
  Same architecture as language models
  Trained on massive financial price datasets
        │
        ▼
Predicted OHLCV for next N steps
        │
        ▼
KronosPredictor inverse-normalises → actual prices
```

### Models Available

| Model | Params | Context | Status |
|---|---|---|---|
| **Kronos-mini** | 4.1M | 2048 bars | ✅ Used in this app |
| Kronos-small | 24.7M | 512 bars | Available |
| Kronos-base | 102.3M | 512 bars | Available |
| Kronos-large | 499.2M | 512 bars | Not open-source |

### Prediction Priority Chain

```
1. Kronos-mini    ← Best: foundation model for financial markets
       ↓ fails (NaN / not installed)
2. GBM/RF Model   ← Good: trained on 98 tickers, 22 features
       ↓ fails (model.joblib missing)
3. Linear         ← Last resort: straight-line extrapolation
```

### NaN Cleaning Pipeline

yfinance sometimes returns NaN in recent candles (incomplete today's bar). The 7-step cleaner handles this:

```
Step 1: Drop last row if any price column is NaN (incomplete candle)
Step 2: Forward-fill then backward-fill prices (fill gaps)
Step 3: Fill volume NaN with 0 (volume optional for Kronos)
Step 4: Drop rows still NaN after filling
Step 5: Replace inf/-inf values
Step 6: Drop rows where close ≤ 0 (corrupt data guard)
Step 7: Align timestamps to match cleaned data length
```

---

## 🛠️ Tech Stack

### Backend / ML

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| scikit-learn | Latest | GradientBoostingRegressor, RandomForest |
| joblib | Latest | Model serialisation / loading |
| pandas | Latest | Data manipulation |
| numpy | Latest | Numerical computing |
| yfinance | Latest | Historical OHLCV + live price data |
| huggingface-hub | Latest | Download Kronos model weights |
| einops | Latest | Kronos dependency (tensor operations) |
| rotary-embedding-torch | Latest | Kronos dependency (positional encoding) |

### Frontend

| Technology | Purpose |
|---|---|
| Streamlit | App framework (Python → React web app) |
| Plotly (graph_objects) | Interactive dark-themed charts |
| Custom CSS | Dark theme, card components, button colours |
| Inline HTML | Live price bar, metric cards, banners |
| st.empty() slots | In-place live updates without page reload |

### Colour Palette

```python
Background:    #0f0f1a    # Deep navy black
Sidebar:       #13131f    # Slightly lighter
Card bg:       #1a1a2e    # Card surfaces
Text:          #e0e0f0    # Near white
Muted text:    #8888aa    # Purple-grey
Green accent:  #00c07a    # Start button, profit, up
Red accent:    #e53935    # Stop button, loss, down
Chart blue:    #4db8ff    # Historical price line
Chart orange:  #ff9933    # Predicted line
```

---

## 📋 File Reference

| File | Description |
|---|---|
| `app_hourly.py` | Main app — UI, orchestration, charts, live loop |
| `train_model.py` | Downloads 98 tickers, trains GBM, saves model |
| `evaluate.py` | Tests model accuracy on specific tickers |
| `setup_kronos.py` | Clones Kronos repo, installs deps, writes path config |
| `model.joblib` | Serialised trained GBM model |
| `features.joblib` | Ordered list of 22 feature names |
| `src/data_loader.py` | yfinance historical fetch + live price fetch |
| `src/forecast_utils.py` | Business date generation for predictions |
| `src/model_handler_kronos.py` | Kronos load, NaN cleaning, prediction |
| `src/kronos_path.json` | Auto-generated by setup_kronos.py |
| `src/kronos/model/` | Cloned Kronos transformer code |
| `eval_out/predictions.csv` | Saved prediction vs actual for analysis |

---

## ⚠️ Known Limitations

- **yfinance rate limiting**: Heavy usage may trigger Yahoo Finance's rate limits. If data fails to load, wait 60 seconds and retry.

- **Hourly data limit**: Yahoo Finance only provides hourly data for the past 60 days. Longer periods are automatically capped.

- **Kronos on CPU**: Kronos-mini runs on CPU in this setup. Prediction takes 5–15 seconds depending on hardware. Enable GPU by changing `device="cpu"` to `device="cuda"` in `model_handler_kronos.py` (requires CUDA-enabled GPU + PyTorch CUDA).

- **Model accuracy**: The GBM model achieves ~55–58% directional accuracy — better than random (50%) but not perfect. Use predictions as one signal among many, not as definitive buy/sell advice.

- **Backtesting limitations**: Past performance does not predict future results. The backtest does not account for transaction costs, slippage, or market impact.

---

## 🔧 Troubleshooting

### `ModuleNotFoundError: No module named 'joblib'`
```powershell
pip install joblib scikit-learn
```

### `[Kronos] prediction failed: Input DataFrame contains NaN values`
This means the NaN cleaner failed. Check `src/model_handler_kronos.py` is the latest version. Also try:
```powershell
# Delete config and re-run setup
del src\kronos_path.json
python setup_kronos.py
```

### `No data for 'TICKER' at '1h'`
Hourly data is only available for the last 60 days. Switch Date Range to "1 Month" or use `1d` interval.

### `RMSE shows "—"` (dash instead of number)
This happens when all predictions are NaN (Kronos failed and ML fallback also failed). Ensure `model.joblib` exists by running `python train_model.py`.

### Stop button appears green instead of red
Your Streamlit version may handle CSS differently. This is cosmetic only and does not affect functionality.

### Streamlit `use_container_width` warnings in terminal
These are deprecation warnings from Streamlit — harmless. They will be resolved in a future Streamlit update.

---

## 📈 How to Improve Prediction Accuracy

| Action | Expected Improvement |
|---|---|
| Shorten horizon to 1 Week | ~40% less MAPE |
| Use 1d interval (not 1h) | ~25% less MAPE |
| Retrain with `train_model.py` | ~15% less MAPE |
| Kronos-mini (once working) | Foundation model baseline |
| Increase training data to 5 years | ~10% less MAPE |

---

## 🗂️ Trained Tickers

### Indian NSE (50)
`RELIANCE.NS` `TCS.NS` `HDFCBANK.NS` `INFY.NS` `ICICIBANK.NS` `HINDUNILVR.NS` `SBIN.NS` `BHARTIARTL.NS` `ITC.NS` `KOTAKBANK.NS` `LT.NS` `AXISBANK.NS` `ASIANPAINT.NS` `MARUTI.NS` `HCLTECH.NS` `SUNPHARMA.NS` `TITAN.NS` `ULTRACEMCO.NS` `WIPRO.NS` `BAJFINANCE.NS` `NESTLEIND.NS` `POWERGRID.NS` `NTPC.NS` `ONGC.NS` `JSWSTEEL.NS` `TATASTEEL.NS` `ADANIENT.NS` `ADANIPORTS.NS` `BAJAJFINSV.NS` `DRREDDY.NS` `CIPLA.NS` `DIVISLAB.NS` `EICHERMOT.NS` `HEROMOTOCO.NS` `BRITANNIA.NS` `COALINDIA.NS` `GRASIM.NS` `INDUSINDBK.NS` `SBILIFE.NS` `HDFCLIFE.NS` `TECHM.NS` `BPCL.NS` `HINDALCO.NS` `APOLLOHOSP.NS` `TATACONSUM.NS` `PIDILITIND.NS` `HAVELLS.NS` `DABUR.NS` `MARICO.NS` `BAJAJ-AUTO.NS`

### International (48)
`AAPL` `MSFT` `NVDA` `GOOGL` `AMZN` `META` `TSLA` `AVGO` `ORCL` `CRM` `CSCO` `ADBE` `QCOM` `AMD` `INTC` `JPM` `BAC` `V` `MA` `GS` `BRK-B` `UNH` `JNJ` `LLY` `ABBV` `MRK` `PFE` `WMT` `KO` `PEP` `MCD` `PG` `XOM` `CVX` `HD` `COST` `NKE` `TSM` `ASML` `SAP` `NVO` `SHEL` `BP` `UL` `RIO` `BHP` `TM` `SNY` `BTI` `NFLX`

---

## 📜 License

This project is released under the **MIT License** — free for personal and educational use.

---

## 👩‍💻 AUTHOR

**Muskan Kumari**
AI Engineer | Business Analyst | Software Engineer

* Passionate about **AI-driven decision systems & data analytics**
* Focused on building **real-world ML + business impact solutions**
* Experience in **stock forecasting, automation, and intelligent dashboards**

🔗 GitHub: https://github.com/student-muskankumari

---

## 🙏 Credits

- **[Kronos Foundation Model](https://github.com/shiyu-coder/Kronos)** — Yu Shi et al., 2025. Pre-trained financial time-series transformer.
- **[yfinance](https://github.com/ranaroussi/yfinance)** — Unofficial Yahoo Finance Python API.
- **[Streamlit](https://streamlit.io/)** — Python web app framework.
- **[Plotly](https://plotly.com/python/)** — Interactive charting library.
- **[scikit-learn](https://scikit-learn.org/)** — Machine learning library.

---

## ⚡ Quick Start (TL;DR)

```powershell
# 1. Activate venv
venv\Scripts\activate

# 2. Install deps
pip install streamlit plotly pandas numpy scikit-learn joblib yfinance einops rotary-embedding-torch

# 3. Setup Kronos (optional)
python setup_kronos.py

# 4. Train ML model (optional, skip if model.joblib exists)
python train_model.py

# 5. Run app
streamlit run app_hourly.py
```

> **Disclaimer:** This application is for educational and research purposes only. It is not financial advice. Always do your own research before making investment decisions.