"""Tests for the Kronos input-cleaning step (src/model_handler_kronos.py).

Kronos rejects any NaN in its price/volume inputs, so these tests pin down
what _clean_df_for_kronos must guarantee before data reaches the model.
Run from the repo root:  python -m pytest -q
"""
import numpy as np
import pandas as pd

from src.model_handler_kronos import _clean_df_for_kronos


def _sample_df():
    return pd.DataFrame({
        "open":   [np.nan, 101.0, np.nan, 103.0],
        "high":   [np.nan, 102.0, np.nan, 104.0],
        "low":    [np.nan, 100.0, np.nan, 102.0],
        "close":  [np.nan, 101.5, np.nan, 103.5],
        "volume": [1000.0, np.nan, 1200.0, np.nan],
    })


def test_price_gaps_are_filled_and_no_nan_remains():
    out = _clean_df_for_kronos(_sample_df())

    assert not out.isna().any().any()
    # mid-series gap is forward-filled from the previous row
    assert out.loc[2, "close"] == 101.5
    # leading gap is back-filled from the first valid row
    assert out.loc[0, "close"] == 101.5


def test_missing_volume_becomes_zero():
    out = _clean_df_for_kronos(_sample_df())

    assert out.loc[1, "volume"] == 0
    assert out.loc[3, "volume"] == 0
    assert out.loc[0, "volume"] == 1000.0


def test_input_is_not_mutated_and_index_is_reset():
    df = _sample_df()
    df.index = [10, 11, 12, 13]
    before = df.copy()

    out = _clean_df_for_kronos(df)

    pd.testing.assert_frame_equal(df, before)  # original left untouched
    assert list(out.index) == [0, 1, 2, 3]
    assert len(out) == len(df)
