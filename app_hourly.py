"""
Kronos – Live Stock Forecasting App
Fixes applied:
  1. RMSE NaN  → nanmean + NaN filtering
  2. PyArrow   → Exit column cast to str before dataframe
  3. pct_change FutureWarning → fill_method=None
  4. Kronos path → reads kronos_path.json from setup_kronos.py
  5. Stop button → CSS via st.markdown with unique id hack
  6. use_container_width → replaced with width='stretch'
"""

import streamlit as st
import datetime, time, os, sys, joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Cloud startup: auto-setup Kronos + train model if missing ─
try:
    import cloud_startup  # noqa: F401
except Exception:
    pass

from src.data_loader import fetch_yfinance, fetch_live_price
from src.forecast_utils import build_future_dates

# ── Kronos handler ────────────────────────────────────────────
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
try:
    from src.model_handler_kronos import (
        predict_with_kronos, kronos_status, linear_fallback)
    _HAS_KH = True
except ImportError:
    _HAS_KH = False

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Kronos – Live Stock Forecasting",
                   page_icon="📊", layout="wide")

# ── FIX 5: Stop button CSS — use data-testid + nth-child on sidebar ──
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0f0f1a; }
[data-testid="stSidebar"]          { background:#13131f; border-right:1px solid #2a2a3d; }
[data-testid="stSidebar"] *        { color:#e0e0f0 !important; }
[data-testid="stMetricValue"]      { font-size:1.75rem !important; font-weight:700 !important;
                                     color:#e0e0f0 !important; }
[data-testid="stMetricLabel"]      { font-size:.74rem !important; color:#8888aa !important;
                                     text-transform:uppercase; letter-spacing:.05em; }

/* ── FIX 5: button colours ── */
[data-testid="stSidebar"] button { border-radius:8px !important; font-weight:700 !important;
                                   border:none !important; width:100% !important; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(1) button
    { background:#00c07a !important; color:#fff !important; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2) button
    { background:#e53935 !important; color:#fff !important; }

.metric-card { background:#1a1a2e; border:1px solid #2a2a3d; border-radius:12px;
               padding:16px 20px; text-align:center; }
.metric-val  { font-size:1.75rem; font-weight:700; color:#e0e0f0; line-height:1.1; }
.metric-lbl  { font-size:.73rem; color:#8888aa; text-transform:uppercase;
               letter-spacing:.06em; margin-bottom:6px; }
.metric-sub  { font-size:.82rem; margin-top:5px; }
.live-px-bar { background:#12121e; border:1px solid #2a2a3d; border-radius:10px;
               padding:12px 20px; margin-bottom:8px;
               display:flex; align-items:center; gap:18px; flex-wrap:wrap; }
.live-banner { background:#0d1f0d; border:1px solid #1e3d1e; border-radius:10px;
               padding:8px 16px; color:#00c07a; font-size:.87rem; margin-bottom:6px; }
.refresh-msg { background:#13131f; border:1px solid #2a2a3d; border-radius:10px;
               padding:7px 16px; color:#aaa; font-size:.82rem; margin-bottom:6px; }
.ts-lbl      { font-size:.72rem; color:#555; text-align:right; margin:-2px 0 6px; }
.sec-hdr     { font-size:1.04rem; font-weight:700; color:#e0e0f0; margin:14px 0 8px; }
h1           { color:#e0e0f0 !important; font-size:1.5rem !important;
               font-weight:700 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
IMAX = {"1m":7,"5m":60,"15m":60,"30m":60,"1h":60,"90m":60,"1d":3650,"1wk":3650}
PDAYS = {"1 Week":7,"1 Month":30,"3 Months":90,
         "6 Months":180,"1 Year":365,"2 Years":730,"5 Years":1825}
HD = {"1 Week":5,"2 Weeks":10,"1 Month":22,"3 Months":66,"6 Months":130}
HH = {"12h":12,"24h":24,"48h":48,"1 Week":40}
ccur = lambda t: "₹" if (".NS" in t or ".BO" in t) else "$"

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>
        <span style='font-size:1.5rem;'>📊</span>
        <div>
            <div style='font-size:1.2rem;font-weight:700;color:#e0e0f0;'>Kronos</div>
            <div style='font-size:.72rem;color:#8888aa;'>Live Stock Forecasting App</div>
        </div>
    </div>
    <hr style='border-color:#2a2a3d;margin:10px 0;'/>
    """, unsafe_allow_html=True)

    ticker   = st.text_input("Ticker", value="RELIANCE.NS")
    interval = st.selectbox("Interval", ["1d","1h","1wk"], index=0)

    maxd  = IMAX.get(interval, 3650)
    vp    = [p for p,d in PDAYS.items() if d <= maxd]
    defp  = "2 Years" if "2 Years" in vp else vp[-1]
    plab  = st.selectbox("Date Range", vp, index=vp.index(defp))
    pdays = PDAYS[plab]

    hmap  = HH if interval=="1h" else HD
    hlab  = st.selectbox("Horizon", list(hmap.keys()), index=1)
    hstep = hmap[hlab]

    margin = st.slider("Margin", 0.5, 5.0, 1.5, 0.5,
                       help="Confidence band = ±(RMSE × Margin)")

    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        run_btn  = st.button("Start", width="stretch")
    with c2:
        stop_btn = st.button("Stop",  width="stretch")

    st.markdown("<hr style='border-color:#2a2a3d;margin:12px 0 6px;'/>",
                unsafe_allow_html=True)
    live_mode    = st.checkbox("Enable Live Mode", value=False)
    refresh_rate = st.number_input("Refresh every (s)", min_value=5,
                                   max_value=600, value=30, step=5)

    st.markdown("<hr style='border-color:#2a2a3d;margin:12px 0 6px;'/>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.74rem;color:#8888aa;font-weight:600;"
        "text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;'>"
        "Backtest Strategy</div>", unsafe_allow_html=True)
    sig_t  = st.slider("Signal Threshold (%)", 0.05, 2.0, 0.15, 0.05)
    use_ma = st.checkbox("MA20 Trend Filter", value=True)
    use_sl = st.checkbox("Stop Loss (3%)", value=False)
    icap   = st.number_input("Starting Capital", value=100000, step=10000)

    if interval == "1h":
        st.info("ℹ️ Hourly: max 60 days.")

    # Kronos status
    st.markdown("<hr style='border-color:#2a2a3d;margin:12px 0 6px;'/>",
                unsafe_allow_html=True)
    if _HAS_KH:
        ks = kronos_status()
        if ks == "ready":       st.success("✅ Kronos-mini loaded")
        elif ks == "available": st.info("⚡ Kronos files found")
        else: st.warning("⚠️ Run\n`python setup_kronos.py`\nto enable Kronos")
    else:
        st.warning("⚠️ Run\n`python setup_kronos.py`\nto enable Kronos predictions")

# ══════════════════════════════════════════════════════════════
# ML MODEL (RF/GBM) — silent load
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_ml():
    try:
        m = joblib.load("model.joblib")
        f = joblib.load("features.joblib") if os.path.exists("features.joblib") else None
        if f is None or not any(x in f for x in ["Open_norm","Return_1d"]):
            return None, None
        return m, f
    except Exception:
        return None, None

ml_model, ml_feats = load_ml()

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# FIX 3: pct_change fill_method=None
# ══════════════════════════════════════════════════════════════
def rsi14(s):
    d=s.diff(); g=d.clip(lower=0).rolling(14).mean()
    l=(-d.clip(upper=0)).rolling(14).mean()
    return 100-(100/(1+g/(l+1e-9)))

def eng(df, feats=None):
    d=df.copy()
    if not isinstance(d.index,pd.DatetimeIndex):
        if "Date" in d.columns: d=d.set_index("Date")
        d.index=pd.to_datetime(d.index)
    c=d["Close"]
    d["Open_norm"]      = d["Open"]/c; d["High_norm"]=d["High"]/c; d["Low_norm"]=d["Low"]/c
    vm20                = d["Volume"].rolling(20).mean().replace(0,1)
    d["Volume_norm"]    = d["Volume"]/vm20
    d["Day"]=d.index.day; d["Month"]=d.index.month; d["Year"]=d.index.year
    # FIX 3: add fill_method=None
    d["Return_1d"]      = c.pct_change(1,  fill_method=None)
    d["Return_5d"]      = c.pct_change(5,  fill_method=None)
    d["Return_20d"]     = c.pct_change(20, fill_method=None)
    d["MA5_norm"]  = c.rolling(5).mean()/c; d["MA20_norm"]=c.rolling(20).mean()/c
    d["MA50_norm"] = c.rolling(50).mean()/c
    d["Volatility_10d"] = d["Return_1d"].rolling(10).std()
    d["Volatility_20d"] = d["Return_1d"].rolling(20).std()
    h20=c.rolling(20).max(); l20=c.rolling(20).min()
    d["Price_momentum"] = (c-l20)/((h20-l20).replace(0,1))
    if feats and "RSI_14" in feats:
        d["RSI_14"]       = rsi14(c)/100.0
        e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
        mac=e12-e26; d["MACD_signal"]=(mac-mac.ewm(span=9,adjust=False).mean())/(c+1e-9)
        bm=c.rolling(20).mean(); bs=c.rolling(20).std()
        d["BB_position"]  = (c-(bm-2*bs))/((4*bs).replace(0,1))
        d["Volume_spike"] = d["Volume"]/d["Volume"].rolling(50).mean().replace(0,1)
        tr=pd.concat([d["High"]-d["Low"],(d["High"]-c.shift()).abs(),
                      (d["Low"]-c.shift()).abs()],axis=1).max(axis=1)
        d["ATR_norm"] = tr.rolling(14).mean()/(c+1e-9)
    return d

# ══════════════════════════════════════════════════════════════
# ML MULTI-STEP FORECAST
# ══════════════════════════════════════════════════════════════
def ml_forecast(df, horizon, model, feats):
    d=eng(df,feats).dropna()
    if len(d)<50: return None
    rc=d["Close"].tolist(); rr=d["Close"].pct_change(1,fill_method=None).dropna().tolist()
    rv=d["Volume"].tolist(); lc=float(rc[-1]); nd=d.index[-1]; out=[]
    for _ in range(horizon):
        nd=pd.Timestamp(nd)+pd.Timedelta(days=1)
        while nd.weekday()>=5: nd+=pd.Timedelta(days=1)
        sv=float(np.median(rv[-20:])) if len(rv)>=20 else float(rv[-1])
        vm=max(float(np.mean(rv[-20:])) if len(rv)>=20 else sv,1.0)
        fd={
            "Open_norm":1.0,"High_norm":1.003,"Low_norm":0.997,"Volume_norm":sv/vm,
            "Day":nd.day,"Month":nd.month,"Year":nd.year,
            "Return_1d": (rc[-1]/rc[-2]-1)  if len(rc)>=2  else 0.0,
            "Return_5d": (rc[-1]/rc[-6]-1)  if len(rc)>=6  else 0.0,
            "Return_20d":(rc[-1]/rc[-21]-1) if len(rc)>=21 else 0.0,
            "MA5_norm":  float(np.mean(rc[-5:]))/lc   if len(rc)>=5  else 1.0,
            "MA20_norm": float(np.mean(rc[-20:]))/lc  if len(rc)>=20 else 1.0,
            "MA50_norm": float(np.mean(rc[-50:]))/lc  if len(rc)>=50 else 1.0,
            "Volatility_10d":float(np.std(rr[-10:])) if len(rr)>=10 else 0.01,
            "Volatility_20d":float(np.std(rr[-20:])) if len(rr)>=20 else 0.01,
            "Price_momentum":(lc-min(rc[-20:]))/max(max(rc[-20:])-min(rc[-20:]),1e-9)
                             if len(rc)>=20 else 0.5,
        }
        if "RSI_14" in feats:
            fd["RSI_14"]       = float(rsi14(pd.Series(rc[-30:])).iloc[-1]/100) if len(rc)>=30 else 0.5
            fd["MACD_signal"]  = 0.0; fd["BB_position"]=fd["Price_momentum"]
            fd["Volume_spike"] = sv/vm
            fd["ATR_norm"]     = float(np.std(rr[-10:])) if len(rr)>=10 else 0.01
        row=np.array([[fd[f] for f in feats]])
        pr=float(np.clip(model.predict(row)[0],-0.05,0.05))
        nc=lc*(1+pr); out.append(nc); rc.append(nc); rr.append(pr); rv.append(sv); lc=nc
    return out

# ══════════════════════════════════════════════════════════════
# BACKTESTING — all bugs fixed
# ══════════════════════════════════════════════════════════════
def backtest(df, model, feats, cap0=100000.0,
             sigt=0.0015, maf=True, sl=False):
    d=eng(df,feats).dropna(subset=feats)
    if len(d)<20: return _ebt()
    preds=model.predict(d[feats].values)
    closes=d["Close"].values
    ma20=pd.Series(closes).rolling(20).mean().values
    n=len(closes)-1
    capital=float(cap0); pos=0.0; ep=0.0; entry_bar=None
    portfolio=[]; tlog=[]; hold_durs=[]
    for i in range(n):
        sig=preds[i]; p_in=closes[i]; p_out=closes[i+1]
        mav=ma20[i] if not np.isnan(ma20[i]) else p_in
        if sl and pos>0 and ep>0 and p_in<=ep*0.97:
            dur=(i-entry_bar) if entry_bar else 0
            tlog.append({"Bar":i,"Action":"STOP",
                         "Entry":round(ep,2),"Exit":str(round(p_in,2)),  # FIX 2: str
                         "Profit":round(pos*(p_in-ep),2),"Duration":dur})
            hold_durs.append(dur); capital=pos*p_in
            pos=0.0; ep=0.0; entry_bar=None
            portfolio.append(capital); continue
        in_up=(p_in>mav) if maf else True
        if sig>sigt and in_up:
            if pos==0 and capital>0:
                pos=capital/p_in; ep=p_in; capital=0.0; entry_bar=i
                # FIX 2: Exit="open" as string, Profit=0 (unrealized)
                tlog.append({"Bar":i,"Action":"BUY",
                             "Entry":round(ep,2),"Exit":"open",
                             "Profit":0.0,"Duration":0})
        elif sig<-sigt:
            if pos>0:
                dur=(i-entry_bar) if entry_bar else 0
                tlog.append({"Bar":i,"Action":"SELL",
                             "Entry":round(ep,2),"Exit":str(round(p_out,2)),  # FIX 2: str
                             "Profit":round(pos*(p_out-ep),2),"Duration":dur})
                hold_durs.append(dur); capital=pos*p_out
                pos=0.0; ep=0.0; entry_bar=None
        portfolio.append(capital+pos*p_out)
    if pos>0: capital=pos*closes[-1]
    bah=((closes[-1]/closes[0])-1)*100 if closes[0]!=0 else 0.0
    if not portfolio: return _ebt()
    port=np.array(portfolio)
    rets=np.diff(port)/(port[:-1]+1e-9)
    sharpe=float(np.mean(rets)/(np.std(rets)+1e-9)*np.sqrt(252)) if len(rets)>1 else 0.0
    peak=np.maximum.accumulate(port)
    mdd=float(np.min((port-peak)/(peak+1e-9)))
    st_=[t for t in tlog if t["Action"] in ("SELL","STOP")]
    profs=[t["Profit"] for t in st_]
    wr=(sum(1 for p in profs if p>0)/len(profs)*100) if profs else 0.0
    ap=float(np.mean(profs)) if profs else 0.0
    ah=float(np.mean(hold_durs)) if hold_durs else 0.0
    ps=int(np.sum(preds>sigt)); ns=int(np.sum(preds<-sigt)); hs=len(preds)-ps-ns
    return {"final":capital,"ret_pct":(capital/cap0-1)*100,"sharpe":sharpe,
            "drawdown":mdd,"trades":sum(1 for t in tlog if t["Action"]=="BUY"),
            "win_rate":wr,"bah":bah,"avg_profit":ap,"avg_hold":ah,
            "ps":ps,"ns":ns,"hs":hs,
            "portfolio":port.tolist(),"actual_closes":closes.tolist(),"tlog":tlog}

def _ebt():
    return {"final":100000,"ret_pct":0,"sharpe":0,"drawdown":0,"trades":0,
            "win_rate":0,"bah":0,"avg_profit":0,"avg_hold":0,
            "ps":0,"ns":0,"hs":0,"portfolio":[],"actual_closes":[],"tlog":[]}

# ══════════════════════════════════════════════════════════════
# HELPERS
# FIX 1: nanmean for RMSE/MAPE
# ══════════════════════════════════════════════════════════════
def start_dt(days,iv):
    return datetime.date.today()-datetime.timedelta(days=min(days,IMAX.get(iv,3650)))

def safe_rmse(a, p):
    """FIX 1: handles NaN values gracefully."""
    n=min(len(a),len(p))
    av=np.array(a[:n],dtype=float); pv=np.array(p[:n],dtype=float)
    mask=~(np.isnan(av)|np.isnan(pv))
    if mask.sum()==0: return 0.0
    return float(np.sqrt(np.nanmean((av[mask]-pv[mask])**2)))

def safe_mape(a, p):
    """FIX 1: handles NaN values gracefully."""
    n=min(len(a),len(p))
    av=np.array(a[:n],dtype=float); pv=np.array(p[:n],dtype=float)
    mask=~(np.isnan(av)|np.isnan(pv)|np.isnan(av))
    if mask.sum()==0: return 0.0
    with np.errstate(divide="ignore",invalid="ignore"):
        m=float(np.nanmean(np.abs((av[mask]-pv[mask])/
                                   np.where(av[mask]==0,np.nan,av[mask])))*100)
    return 0.0 if np.isnan(m) else m

# ══════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════
DARK="#0f0f1a"; GRID="#1a1a2e"; FONT=dict(color="#c0c0d0",size=12)

def _lay(title,h=390,xl="",yl="Price"):
    return dict(paper_bgcolor=DARK,plot_bgcolor=DARK,font=FONT,height=h,
                title=dict(text=f"<b>{title}</b>",font=dict(size=13,color="#e0e0f0"),
                           x=0.01,xanchor="left"),
                xaxis=dict(gridcolor=GRID,zeroline=False,tickfont=dict(size=11,color="#777"),
                           title=dict(text=xl,font=dict(size=10,color="#555"))),
                yaxis=dict(gridcolor=GRID,zeroline=False,tickfont=dict(size=11,color="#777"),title=yl),
                legend=dict(orientation="h",x=0.5,xanchor="center",y=-0.22,
                            bgcolor="rgba(0,0,0,0)",font=dict(size=11,color="#999")),
                margin=dict(l=50,r=20,t=50,b=70),hovermode="x unified")

def _smooth(values, window=3):
    """Lightly smooth prediction values to remove Kronos oscillation noise."""
    if len(values) <= window:
        return values
    s = pd.Series(values).rolling(window, min_periods=1, center=True).mean()
    return s.tolist()

def fchart(df, fdts, preds, rmse_v, mg, ticker):
    """
    Fixed chart:
    - Shows last 5× the prediction horizon in history (balanced view)
    - Smooths noisy Kronos predictions
    - Adds vertical divider line at forecast start
    - Sets explicit x-axis range so predictions are clearly visible
    """
    n_pred = len(preds)

    # Show 5× prediction steps of history — prediction window ~17% of chart
    hist_bars = max(n_pred * 5, 30)
    h  = df.tail(hist_bars)
    hd = h["Date"].tolist()
    hc = h["Close"].tolist()

    # Smooth predictions (reduces Kronos oscillation noise)
    smooth_preds = _smooth(preds, window=3)
    up = [p + rmse_v * mg for p in smooth_preds]
    lo = [p - rmse_v * mg for p in smooth_preds]

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hd, y=hc, mode="lines", name="Historical",
        line=dict(color="#4db8ff", width=2)))

    # Bridge connector (last historical → first predicted)
    if fdts:
        fig.add_trace(go.Scatter(
            x=[hd[-1], fdts[0]], y=[hc[-1], smooth_preds[0]],
            mode="lines", showlegend=False,
            line=dict(color="#ff9933", width=1.5, dash="dot")))

    # RMSE confidence band
    fig.add_trace(go.Scatter(
        x=fdts + fdts[::-1], y=up + lo[::-1],
        fill="toself", fillcolor="rgba(130,90,200,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band", hoverinfo="skip"))

    # Predicted line
    fig.add_trace(go.Scatter(
        x=fdts, y=smooth_preds, mode="lines+markers", name="Predicted",
        line=dict(color="#ff9933", width=2.5, dash="dash"),
        marker=dict(size=5, color="#ff9933")))

    # Vertical line at forecast start
    if fdts:
        fig.add_shape(type="line",x0=hd[-1], x1=hd[-1], y0=0, y1=1, xref="x", yref="paper",line=dict(color="#555", width=1, dash="dot"),)
        fig.add_annotation(x=hd[-1], y=1,xref="x", yref="paper",text="Forecast →",showarrow=False,font=dict(color="#888", size=10),xanchor="left", yanchor="bottom",bgcolor="rgba(0,0,0,0)",)
        

    try:
        ds = pd.Timestamp(df["Date"].iloc[-1]).strftime("%b %d, %Y")
    except Exception:
        ds = ""

    # Build explicit x-axis range: from first historical bar to last predicted date
    try:
        x_start = pd.Timestamp(hd[0])
        x_end   = pd.Timestamp(fdts[-1]) + pd.Timedelta(days=2)
    except Exception:
        x_start = None; x_end = None

    layout = _lay(f"Live {ticker} Forecast",
                  xl=f"Date  (data as of {ds})", h=420)

    if x_start and x_end:
        layout["xaxis"]["range"] = [str(x_start), str(x_end)]

    # Y-axis range: cover both history and prediction with padding
    all_prices = hc + smooth_preds + up + lo
    all_prices = [p for p in all_prices if p and not np.isnan(p)]
    if all_prices:
        ymin = min(all_prices) * 0.995
        ymax = max(all_prices) * 1.005
        layout["yaxis"]["range"] = [ymin, ymax]

    fig.update_layout(**layout)
    return fig

def echart(portfolio,actual_closes,cap0):
    if not portfolio or len(actual_closes)<2: return go.Figure()
    n=len(portfolio)
    sp=np.array([(v/cap0-1)*100 for v in portfolio])
    p0=float(actual_closes[0])
    pr=(actual_closes[1:n+1]+[actual_closes[-1]]*max(0,n-len(actual_closes)+1))[:n]
    bp=np.array([(float(p)/p0-1)*100 for p in pr])
    xs=list(range(n)); fig=go.Figure()
    fig.add_trace(go.Scatter(x=xs+xs[::-1],y=list(sp)+list(bp[::-1]),
                             fill="toself",fillcolor="rgba(0,180,100,0.13)",
                             line=dict(color="rgba(0,0,0,0)"),name="Outperforming",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs+xs[::-1],y=list(bp)+list(sp[::-1]),
                             fill="toself",fillcolor="rgba(220,50,50,0.10)",
                             line=dict(color="rgba(0,0,0,0)"),name="Underperforming",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs,y=bp,mode="lines",name="Buy & Hold",
                             line=dict(color="#888",width=1.5,dash="dash")))
    fig.add_trace(go.Scatter(x=xs,y=sp,mode="lines",name="Strategy",
                             line=dict(color="#4db8ff",width=2)))
    fig.add_hline(y=0,line_color="#333",line_width=1)
    fig.update_layout(**_lay("Equity Curve vs Buy & Hold",h=310,yl="Return (%)"))
    return fig

def sdchart(ps,ns,hs):
    if ps+ns+hs==0: return go.Figure()
    fig=go.Figure(go.Pie(labels=["HOLD","BUY","SELL"],values=[hs,ps,ns],hole=0.55,
                         marker=dict(colors=["#444","#00c07a","#e53935"]),
                         textfont=dict(size=11,color="#ccc")))
    fig.update_layout(paper_bgcolor=DARK,plot_bgcolor=DARK,font=FONT,height=255,
                      title=dict(text="<b>Signal Distribution</b>",
                                 font=dict(size=12,color="#e0e0f0"),x=0.5,xanchor="center"),
                      legend=dict(font=dict(size=10,color="#aaa"),bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=10,r=10,t=50,b=10))
    return fig

# ══════════════════════════════════════════════════════════════
# CORE FORECAST
# ══════════════════════════════════════════════════════════════
def do_forecast(ticker,interval,pdays,hstep,mg,sigt,maf,sl,cap0):
    cur=ccur(ticker); sd=start_dt(pdays,interval)
    try:
        df=fetch_yfinance(ticker,start=sd,end=datetime.date.today(),interval=interval)
    except Exception as e:
        return {"error":str(e)}
    if df is None or df.empty:
        return {"error":f"No data for '{ticker}' at '{interval}'."}

    preds=None; psrc="ML"; bt=_ebt()

    # Priority 1: Kronos-mini
    if _HAS_KH:
        try:
            preds=predict_with_kronos(df,hstep,interval,device="cpu")
            if preds: psrc="Kronos"
        except Exception: preds=None

    # Priority 2: ML model
    if preds is None and ml_model is not None and ml_feats is not None:
        try: preds=ml_forecast(df,hstep,ml_model,ml_feats)
        except Exception: preds=None

    # Priority 3: linear fallback
    if preds is None:
        psrc="Linear"
        if _HAS_KH:
            preds=linear_fallback(df,hstep)
        else:
            c=df["Close"].values[-min(60,len(df)):]; x=np.arange(len(c))
            m_,b_=np.polyfit(x,c,1)
            preds=[float(m_*(len(c)+i)+b_) for i in range(1,hstep+1)]

    if not preds:
        return {"error":"Forecast failed."}

    if ml_model is not None and ml_feats is not None:
        try:
            bt=backtest(df,ml_model,ml_feats,cap0=float(cap0),
                        sigt=sigt/100.0,maf=maf,sl=sl)
        except Exception: pass

    fm={"1h":"h","1d":"B","1wk":"W"}
    fdts=list(build_future_dates(df["Date"].iloc[-1],len(preds),freq=fm.get(interval,"B")))
    asl=df["Close"].tail(len(preds)).tolist()

    # FIX 1: use safe_rmse/safe_mape
    rv=safe_rmse(asl,preds); mv=safe_mape(asl,preds)

    return {"df":df,"fdts":fdts,"preds":preds,"rmse":rv,"mape":mv,
            "last_close":float(df["Close"].iloc[-1]),
            "pred_final":float(preds[-1]),"cur":cur,"bt":bt,"psrc":psrc,
            "n_rows":len(df),"sd":sd}

# ══════════════════════════════════════════════════════════════
# NAMED SLOTS
# ══════════════════════════════════════════════════════════════
st.markdown("<h1>Kronos – Live Stock Forecasting App</h1>",
            unsafe_allow_html=True)

s_banner = st.empty(); s_refresh = st.empty()
s_px     = st.empty(); s_met    = st.empty()
s_ts     = st.empty(); s_chart  = st.empty()
s_bt     = st.empty()

# ══════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════
def render(res,lpx,is_live=False,cd=None):
    cur=res["cur"]; bt=res["bt"]; rv=res["rmse"]; mv=res["mape"]
    lc=res["last_close"]; pf=res["pred_final"]

    if lpx and "price" in lpx:
        dp=lpx["price"]; lch=lpx["change"]; lpct=lpx["change_pct"]
        pc=lpx["prev_close"]; dh=lpx["day_high"]; dl=lpx["day_low"]
        lt=lpx["time"]; hlp=True
    else:
        dp=lc; lch=0.0; lpct=0.0; pc=lc; dh=lc; dl=lc; lt="--"; hlp=False

    dp_=pf-dp; dpct=(dp_/dp*100) if dp!=0 else 0.0

    if is_live:
        s_banner.markdown(
            f"<div class='live-banner'>🔄 Live — updating every {refresh_rate}s</div>",
            unsafe_allow_html=True)
    else: s_banner.empty()

    if cd is not None:
        s_refresh.markdown(f"<div class='refresh-msg'>⏳ Refreshing in {cd} sec…</div>",
                           unsafe_allow_html=True)
    else: s_refresh.empty()

    if hlp:
        lc2="#00c07a" if lch>=0 else "#ff5252"; la="▲" if lch>=0 else "▼"
        s_px.markdown(f"""
        <div class='live-px-bar'>
            <span style='font-size:1rem;font-weight:700;color:#e0e0f0;'>{ticker}</span>
            <span style='font-size:1.65rem;font-weight:700;color:{lc2};'>{cur}{dp:,.2f}</span>
            <span style='font-size:.9rem;color:{lc2};'>{la} {cur}{lch:+,.2f} ({lpct:+.2f}%)</span>
            <span style='font-size:.72rem;color:#666;margin-left:auto;'>
                Prev:{cur}{pc:,.2f} | H:{cur}{dh:,.2f} | L:{cur}{dl:,.2f} | 🕐{lt}
            </span>
        </div>""",unsafe_allow_html=True)
    else: s_px.empty()

    cc="#00c07a" if lch>=0 else "#ff5252"; ca="▲" if lch>=0 else "▼"
    pc2="#00c07a" if dp_>=0 else "#ff5252"; pa="▲" if dp_>=0 else "▼"
    pl="LIVE PRICE" if hlp else "LATEST CLOSE"
    stag=f"<span style='font-size:.65rem;color:#666;'>via {res['psrc']}</span>"

    # FIX 1: format RMSE safely (no "nan")
    rmse_display = f"{rv:.4f}" if rv > 0 else "—"

    with s_met.container():
        a,b,c3=st.columns(3)
        a.markdown(f"""<div class='metric-card'><div class='metric-lbl'>{pl}</div>
            <div class='metric-val'>{cur}{dp:,.2f}</div>
            <div class='metric-sub' style='color:{cc};'>{ca} {cur}{lch:+,.2f} ({lpct:+.2f}%)</div>
        </div>""",unsafe_allow_html=True)
        b.markdown(f"""<div class='metric-card'><div class='metric-lbl'>PREDICTED CLOSE {stag}</div>
            <div class='metric-val'>{cur}{pf:,.2f}</div>
            <div class='metric-sub' style='color:{pc2};'>{pa} {cur}{dp_:+,.2f} ({dpct:+.2f}%)</div>
        </div>""",unsafe_allow_html=True)
        c3.markdown(f"""<div class='metric-card'><div class='metric-lbl'>RMSE</div>
            <div class='metric-val'>{rmse_display}</div>
            <div class='metric-sub' style='color:#8888aa;'>MAPE {mv:.2f}%</div>
        </div>""",unsafe_allow_html=True)

    s_ts.markdown(
        f"<div class='ts-lbl'>Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True)

    s_chart.plotly_chart(fchart(res["df"],res["fdts"],res["preds"],rv,margin,ticker),
                         width="stretch")

    with s_bt.container():
        st.markdown("<div class='sec-hdr'>💰 Backtesting Results</div>",
                    unsafe_allow_html=True)
        parts=[f"Threshold {sig_t:.2f}%"]
        if use_ma: parts.append("MA20 ✓")
        if use_sl: parts.append("Stop-loss ✓")
        st.caption(f"1-step ML signals — {cur}{icap:,} start  |  "+"  |  ".join(parts))

        b1,b2,b3,b4,b5=st.columns(5)
        b1.metric("Final Capital",   f"{cur}{bt['final']:,.0f}")
        b2.metric("Strategy Return", f"{bt['ret_pct']:+.2f}%")
        b3.metric("Buy & Hold",      f"{bt['bah']:+.2f}%",
                  delta=f"{bt['ret_pct']-bt['bah']:+.2f}% vs B&H")
        b4.metric("Max Drawdown",    f"{bt['drawdown']*100:.1f}%")
        b5.metric("Win Rate",        f"{bt['win_rate']:.1f}%")

        if bt["portfolio"] and bt["actual_closes"]:
            st.plotly_chart(echart(bt["portfolio"],bt["actual_closes"],float(icap)),
                            width="stretch")

        cL,cR=st.columns([3,2])
        with cL:
            s1,s2,s3,s4=st.columns(4)
            s1.metric("Sharpe",       f"{bt['sharpe']:.2f}")
            s2.metric("Total Trades", f"{bt['trades']}")
            s3.metric("Avg Profit",   f"{cur}{bt['avg_profit']:+.0f}")
            s4.metric("Avg Hold",     f"{bt['avg_hold']:.1f} bars")
        with cR:
            if bt["ps"]+bt["ns"]+bt["hs"]>0:
                st.plotly_chart(sdchart(bt["ps"],bt["ns"],bt["hs"]),
                                width="stretch")

        actual_t=[t for t in bt["tlog"] if t["Action"] in ("BUY","SELL","STOP")]
        if actual_t:
            with st.expander(
                    f"📋 Trade Log ({len(actual_t)} trades — BUY/SELL/STOP only)"):
                tdf=pd.DataFrame(actual_t)
                # FIX 2: ensure Exit column is all strings → no PyArrow type error
                tdf["Exit"] = tdf["Exit"].astype(str)
                tdf[f"Profit ({cur})"]=tdf["Profit"].apply(lambda x:f"{x:+.2f}")
                tdf=tdf.drop(columns=["Profit"])
                st.dataframe(tdf, width="stretch")   # FIX 6: use width=
        else:
            st.info("No trades triggered. Lower Signal Threshold in the sidebar.")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Fetching data…"):
        res=do_forecast(ticker,interval,pdays,hstep,margin,
                        sig_t,use_ma,use_sl,icap)
    if "error" in res:
        st.error(res["error"]); st.stop()
    lpx=fetch_live_price(ticker)
    if not live_mode:
        render(res,lpx,is_live=False)
    else:
        render(res,lpx,is_live=True)
        it=0
        while True:
            for rem in range(refresh_rate,0,-1):
                s_refresh.markdown(
                    f"<div class='refresh-msg'>⏳ Refreshing in {rem} sec…</div>",
                    unsafe_allow_html=True)
                time.sleep(1)
            it+=1
            try:
                fresh=do_forecast(ticker,interval,pdays,hstep,margin,
                                  sig_t,use_ma,use_sl,icap)
                lpx=fetch_live_price(ticker)
                if "error" in fresh:
                    s_refresh.error(f"Refresh #{it}: {fresh['error']}"); continue
                render(fresh,lpx,is_live=True)
            except Exception as e:
                s_refresh.error(f"Refresh #{it} failed: {e}")
