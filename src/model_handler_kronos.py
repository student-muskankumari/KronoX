"""
Kronos model handler.
FIX: Clean NaN values from DataFrame before passing to Kronos predictor.
"""

import os, sys, json, logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG  = os.path.join(_HERE, "kronos_path.json")

_predictor = None
_loaded    = False
_attempted = False


def _read_config():
    if not os.path.exists(_CFG):
        return None
    try:
        with open(_CFG) as f:
            return json.load(f)
    except Exception:
        return None


def kronos_available():
    cfg = _read_config()
    if not cfg:
        return False
    root = cfg.get("kronos_root", "")
    return bool(root) and os.path.exists(
        os.path.join(root, "model", "__init__.py"))


def _import_classes():
    cfg = _read_config()
    if not cfg:
        return None
    kronos_root = cfg.get("kronos_root", "")
    if not kronos_root or not os.path.exists(
            os.path.join(kronos_root, "model", "__init__.py")):
        return None

    if kronos_root not in sys.path:
        sys.path.insert(0, kronos_root)

    # Direct import (same as Kronos examples)
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        return Kronos, KronosTokenizer, KronosPredictor
    except ImportError as e:
        log.warning(f"[Kronos] direct import failed: {e}")

    # Importlib fallback
    try:
        import importlib.util
        init_path = os.path.join(kronos_root, "model", "__init__.py")
        spec = importlib.util.spec_from_file_location(
            "model", init_path,
            submodule_search_locations=[os.path.join(kronos_root, "model")])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["model"] = mod
        spec.loader.exec_module(mod)
        return (getattr(mod, "Kronos", None),
                getattr(mod, "KronosTokenizer", None),
                getattr(mod, "KronosPredictor", None))
    except Exception as e:
        log.warning(f"[Kronos] importlib fallback failed: {e}")
        return None


def load_kronos(device="cpu"):
    global _predictor, _loaded, _attempted
    if _attempted:
        return _predictor
    _attempted = True

    classes = _import_classes()
    if not classes or None in classes:
        return None

    Kronos, KronosTokenizer, KronosPredictor = classes
    try:
        log.info("[Kronos] loading tokenizer …")
        tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
        log.info("[Kronos] loading model …")
        mdl = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
        mdl.to(device); mdl.eval()
        _predictor = KronosPredictor(mdl, tok, max_context=2048)
        _loaded = True
        log.info("[Kronos] ✅ ready")
        return _predictor
    except Exception as e:
        log.warning(f"[Kronos] load failed: {e}")
        return None


def _clean_df_for_kronos(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX: Kronos fails on any NaN in price/volume columns.
    Steps:
      1. Forward-fill then backward-fill price columns (handles gaps mid-series)
      2. Fill remaining NaN volume with 0
      3. Drop any rows still containing NaN
      4. Reset index
    """
    d = df.copy()

    price_cols = [c for c in d.columns if c in ("open","high","low","close")]
    vol_cols   = [c for c in d.columns if c == "volume"]

    # Forward + backward fill prices
    for c in price_cols:
        d[c] = d[c].ffill().bfill()

    # Fill volume with 0 if missing
    for c in vol_cols:
        d[c] = d[c].fillna(0)

    # Drop any residual NaN rows
    d = d.dropna(subset=price_cols).reset_index(drop=True)

    return d


def predict_with_kronos(df: pd.DataFrame, horizon_steps: int,
                        interval: str = "1d",
                        device: str = "cpu") -> list | None:
    """
    Predict future close prices using Kronos-mini.
    Returns list of floats or None on failure.
    """
    predictor = load_kronos(device)
    if predictor is None:
        return None
    try:
        # Build Kronos input DataFrame
        kdf = pd.DataFrame({
            "open":  df["Open"].astype(float).values,
            "high":  df["High"].astype(float).values,
            "low":   df["Low"].astype(float).values,
            "close": df["Close"].astype(float).values,
        })
        if "Volume" in df.columns:
            kdf["volume"] = df["Volume"].astype(float).fillna(0).values

        # ── FIX: clean NaN before passing to Kronos ───────────
        kdf = _clean_df_for_kronos(kdf)

        if len(kdf) < 10:
            log.warning("[Kronos] too few rows after cleaning NaN")
            return None

        # Build timestamps aligned to cleaned kdf
        dates_clean = pd.to_datetime(df["Date"]).reset_index(drop=True)
        if len(dates_clean) > len(kdf):
            # dates_clean may be longer if rows were dropped
            dates_clean = dates_clean.iloc[-len(kdf):].reset_index(drop=True)
        x_ts = dates_clean

        # Future timestamps
        freq = {"1d": "B", "1h": "h", "1wk": "W-FRI"}.get(interval, "B")
        last = pd.Timestamp(x_ts.iloc[-1])
        y_ts = pd.Series(
            pd.date_range(last, periods=horizon_steps + 1, freq=freq)[1:]
        ).reset_index(drop=True)

        pred_df = predictor.predict(
            df          = kdf,
            x_timestamp = x_ts,
            y_timestamp = y_ts,
            pred_len    = horizon_steps,
            T           = 1.0,
            top_p       = 0.9,
            sample_count= 1,
        )

        col = "close" if "close" in pred_df.columns else pred_df.columns[0]
        out = pred_df[col].tolist()

        # Sanity check: predictions must be positive numbers close to last price
        last_px = float(kdf["close"].iloc[-1])
        out_clean = [p for p in out
                     if isinstance(p, (int, float))
                     and not np.isnan(p)
                     and 0 < p < last_px * 5]
        if not out_clean:
            log.warning("[Kronos] predictions failed sanity check")
            return None

        return out_clean

    except Exception as e:
        log.warning(f"[Kronos] prediction failed: {e}")
        return None


def linear_fallback(df: pd.DataFrame, steps: int) -> list:
    c = df["Close"].dropna().values[-min(60, len(df)):]
    x = np.arange(len(c)); m, b = np.polyfit(x, c, 1)
    return [float(m*(len(c)+i)+b) for i in range(1, steps+1)]


def kronos_status() -> str:
    if _loaded:            return "ready"
    if kronos_available(): return "available"
    return "not_installed"