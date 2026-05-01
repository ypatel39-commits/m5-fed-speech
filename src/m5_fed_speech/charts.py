"""Charts: hawkish/dovish histogram and backtest scatter.

Saves PNGs into docs/. Used by the demo notebook and scripts/run_pipeline.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def histogram_hawk_dove(df: pd.DataFrame, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["score"].dropna(), bins=30, color="#2b6cb0", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="hawk/dove cutoff")
    ax.set_title("Fed speech hawkish/dovish score distribution")
    ax.set_xlabel("score = (hawk - dove) / total_words * 1000")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def scatter_score_vs_yield(df: pd.DataFrame, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    d = df.dropna(subset=["score", "yield_change"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d["score"], d["yield_change"], alpha=0.5, s=18, color="#2b6cb0")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    if len(d) >= 2:
        m, b = np.polyfit(d["score"].values, d["yield_change"].values, 1)
        xs = np.linspace(d["score"].min(), d["score"].max(), 50)
        ax.plot(xs, m * xs + b, color="red", linewidth=1, label=f"fit: y={m:.4f}x+{b:.4f}")
        ax.legend()
    ax.set_title("Hawkish score vs. next-30d 2y yield change")
    ax.set_xlabel("hawkish score")
    ax.set_ylabel("yield change (pp)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labeled", default="data/speeches_labeled.csv")
    p.add_argument("--backtest", default="data/backtest.csv")
    p.add_argument("--docs", default="docs")
    args = p.parse_args()
    docs = Path(args.docs)
    histogram_hawk_dove(pd.read_csv(args.labeled), docs / "hist_hawk_dove.png")
    scatter_score_vs_yield(pd.read_csv(args.backtest), docs / "scatter_score_yield.png")


if __name__ == "__main__":
    main()
