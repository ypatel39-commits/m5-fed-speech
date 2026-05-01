# M5 Fed Speech — STATE

## Status: v0.1.0 complete (2026-04-30)

## Done
- [x] Dependencies declared in `pyproject.toml` (requests, bs4, lxml, pandas,
      numpy, scikit-learn, yfinance, matplotlib, fredapi; `[bert]` optional)
- [x] `src/m5_fed_speech/lexicon.py` — Apel-Grimaldi-style word lists + scoring
- [x] `src/m5_fed_speech/scrape.py` — federalreserve.gov scraper, 1 req/sec,
      synthetic fallback
- [x] `src/m5_fed_speech/classify.py` — TF-IDF + LogisticRegression
      (random_state=42); BERT path documented in README
- [x] `src/m5_fed_speech/backtest.py` — FRED → yfinance → synthetic yield
      cascade; merge-asof yield-change computation; Sharpe-like metric
- [x] `src/m5_fed_speech/charts.py` — histogram + scatter PNG renderers
- [x] `scripts/run_pipeline.py` — 4-step end-to-end runner
- [x] `notebooks/01_demo.ipynb` — full pipeline with inline charts
- [x] 9 pytest tests passing (lexicon, classify, backtest, smoke)
- [x] `docs/hist_hawk_dove.png`, `docs/scatter_score_yield.png` rendered from
      live data
- [x] README with problem / data / method / results / charts / run / limits

## Latest run (live scrape, 2022-2024, 120 speeches)
- Held-out accuracy: 0.7083
- Pearson r (score vs Δyield_30d): +0.336
- Spearman r: +0.415
- Mean Δyield hawkish: +0.198 pp; dovish: +0.121 pp
- Sharpe-like: 1.11
- Yield source: yfinance `^IRX` (FRED timed out from this network)

## Each source file ≤ 200 lines (verified)

## Future
- [ ] Expand lexicon and validate against a human-labeled subset
- [ ] Optional `[bert]` distilbert fine-tune path on GPU
- [ ] Full 2015-2024 corpus + per-FOMC-cycle backtest
- [ ] Replace `^IRX` proxy with clean FRED `DGS2` once a stable network path is in CI
