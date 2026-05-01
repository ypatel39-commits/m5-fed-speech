"""Tests for backtest math (no network)."""
import numpy as np
import pandas as pd

from m5_fed_speech.backtest import compute_yield_changes, metrics


def _yields() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    # linear ramp 1.0 -> 2.0 over the year
    y = np.linspace(1.0, 2.0, len(dates))
    return pd.DataFrame({"date": dates, "yield": y})


def _speeches() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2020-02-01", "2020-06-01", "2020-09-01"],
        "title": ["a", "b", "c"],
        "url": ["a", "b", "c"],
        "score": [5.0, -3.0, 2.0],
    })


def test_compute_yield_changes_aligns_dates():
    out = compute_yield_changes(_speeches(), _yields(), horizon_days=30)
    assert "yield_change" in out.columns
    assert (out["yield_change"] > 0).all()  # ramp -> always positive


def test_metrics_returns_finite_floats():
    out = compute_yield_changes(_speeches(), _yields(), horizon_days=30)
    res = metrics(out)
    assert res.n_speeches == 3
    for v in [res.pearson_corr, res.spearman_corr, res.hit_rate, res.sharpe_like]:
        assert isinstance(v, float)
        assert not np.isnan(v)
