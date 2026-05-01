"""Run the full M5 pipeline end-to-end.

  python scripts/run_pipeline.py [--synthetic]

Steps:
  1. Scrape (or synthetic) -> data/speeches.csv
  2. Train classifier      -> data/model.joblib + data/speeches_labeled.csv
  3. Backtest vs 2y yield  -> data/backtest.json
  4. Charts                -> docs/hist_hawk_dove.png + docs/scatter_score_yield.png
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    logging.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    py = sys.executable

    scrape_cmd = [py, "-m", "m5_fed_speech.scrape", "--out", "data/speeches.csv"]
    if args.synthetic:
        scrape_cmd.append("--synthetic")
    run(scrape_cmd)
    run([py, "-m", "m5_fed_speech.classify",
         "--in", "data/speeches.csv",
         "--out-model", "data/model.joblib",
         "--out-labeled", "data/speeches_labeled.csv",
         "--out-metrics", "data/metrics.json"])
    run([py, "-m", "m5_fed_speech.backtest",
         "--in", "data/speeches_labeled.csv",
         "--out", "data/backtest.json"])
    run([py, "-m", "m5_fed_speech.charts",
         "--labeled", "data/speeches_labeled.csv",
         "--backtest", "data/backtest.csv",
         "--docs", "docs"])


if __name__ == "__main__":
    main()
