"""Backtest hawkish score vs. 2-year Treasury yield changes.

For each speech date d, we compute the change in the 2y yield from d to d+30
trading days. We then correlate that change with the hawkish score.

Hypothesis: hawkish speeches → yields rise (positive correlation).

Yield source priority:
  1. FRED series DGS2 via fredapi (no key required for public CSV via stlouisfed CSV).
  2. yfinance fallback (^IRX is 13w bill — used as proxy if DGS2 unavailable).
  3. Synthetic random-walk yields if both fail (clearly logged).

Outputs a small dict of metrics + a "Sharpe-like" annualized signal/vol ratio
treating sign(score) as a position size on next-30d yield change.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("backtest")
RANDOM_STATE = 42
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2"


@dataclass
class BacktestResult:
    n_speeches: int
    pearson_corr: float
    spearman_corr: float
    mean_yield_change_hawkish: float
    mean_yield_change_dovish: float
    hit_rate: float          # fraction of speeches where sign(score)==sign(yield change)
    sharpe_like: float       # annualized mean / std of (sign(score)*yield_change)


def load_yields_fred() -> pd.DataFrame:
    r = requests.get(FRED_CSV, timeout=20, headers={"User-Agent": "m5-fed-speech"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.lower().strip() for c in df.columns]
    # FRED CSV header has historically been "DATE,DGS2"; newer exports use
    # "observation_date,DGS2". Handle both.
    date_col = next((c for c in df.columns if "date" in c), None)
    val_col = next((c for c in df.columns if c not in (date_col,)), None)
    if not date_col or not val_col:
        raise RuntimeError(f"Unexpected FRED columns: {df.columns.tolist()}")
    df = df.rename(columns={date_col: "date", val_col: "yield"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df = df.dropna(subset=["date", "yield"]).sort_values("date").reset_index(drop=True)
    return df


def load_yields_yf() -> pd.DataFrame:
    import yfinance as yf
    t = yf.Ticker("^IRX")
    h = t.history(start="2014-01-01", end="2025-01-01", auto_adjust=False)
    if h.empty:
        raise RuntimeError("yfinance returned empty")
    df = h.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "yield"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def load_yields_synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range("2014-01-01", "2025-01-01", freq="B")
    walk = np.cumsum(rng.normal(0, 0.03, size=len(dates))) + 1.5
    return pd.DataFrame({"date": dates, "yield": walk})


def load_yields() -> tuple[pd.DataFrame, str]:
    for name, fn in [("FRED-DGS2", load_yields_fred),
                     ("yfinance-^IRX", load_yields_yf),
                     ("synthetic", load_yields_synthetic)]:
        try:
            df = fn()
            log.info("yields: source=%s rows=%d", name, len(df))
            return df, name
        except Exception as e:
            log.warning("yield source %s failed: %s", name, e)
    raise RuntimeError("all yield sources failed")


def compute_yield_changes(speeches: pd.DataFrame, yields: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    s = speeches.copy()
    s["date"] = pd.to_datetime(s["date"]).astype("datetime64[ns]")
    y = yields.copy()
    y["date"] = pd.to_datetime(y["date"]).astype("datetime64[ns]")
    y = y.sort_values("date").reset_index(drop=True)
    # Merge-asof: for each speech, the last yield <= speech date.
    base = pd.merge_asof(s.sort_values("date"), y, on="date", direction="backward")
    # Future yield = yield on or after (speech_date + horizon).
    s_future = s.copy()
    s_future["date_future"] = (s_future["date"] + pd.Timedelta(days=horizon_days)).astype("datetime64[ns]")
    fut = pd.merge_asof(
        s_future.sort_values("date_future"),
        y.rename(columns={"date": "date_future", "yield": "yield_future"}),
        on="date_future",
        direction="forward",
    )
    base = base.merge(
        fut[["url", "yield_future"]],
        on="url",
        how="left",
        suffixes=("", "_f"),
    ) if "url" in s.columns else base.assign(yield_future=fut["yield_future"].values)
    base["yield_change"] = base["yield_future"] - base["yield"]
    return base


def metrics(df: pd.DataFrame) -> BacktestResult:
    d = df.dropna(subset=["score", "yield_change"]).copy()
    if len(d) < 2:
        return BacktestResult(len(d), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pearson = float(d["score"].corr(d["yield_change"], method="pearson"))
    spearman = float(d["score"].corr(d["yield_change"], method="spearman"))
    hawk = d[d["score"] > 0]["yield_change"].mean()
    dove = d[d["score"] <= 0]["yield_change"].mean()
    hit = float(((np.sign(d["score"]) == np.sign(d["yield_change"]))).mean())
    pnl = np.sign(d["score"]) * d["yield_change"]
    if pnl.std(ddof=1) > 0:
        # ~12 speeches/year typical; annualization is rough by design.
        sharpe = float(pnl.mean() / pnl.std(ddof=1) * math.sqrt(12))
    else:
        sharpe = 0.0
    return BacktestResult(
        n_speeches=len(d),
        pearson_corr=pearson,
        spearman_corr=spearman,
        mean_yield_change_hawkish=float(hawk if hawk == hawk else 0.0),
        mean_yield_change_dovish=float(dove if dove == dove else 0.0),
        hit_rate=hit,
        sharpe_like=sharpe,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_csv", default="data/speeches_labeled.csv")
    p.add_argument("--out", default="data/backtest.json")
    p.add_argument("--horizon", type=int, default=30)
    args = p.parse_args()

    speeches = pd.read_csv(args.in_csv)
    yields, source = load_yields()
    merged = compute_yield_changes(speeches, yields, args.horizon)
    res = metrics(merged)
    out = asdict(res) | {"yield_source": source, "horizon_days": args.horizon}
    Path(args.out).write_text(json.dumps(out, indent=2))
    merged_out = Path(args.out).with_suffix(".csv")
    merged.to_csv(merged_out, index=False)
    log.info("backtest: %s", out)


if __name__ == "__main__":
    main()
