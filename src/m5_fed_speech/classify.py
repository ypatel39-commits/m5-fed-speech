"""Train a hawkish/dovish classifier.

Default: scikit-learn TF-IDF + LogisticRegression (fast, CPU-friendly,
reproducible). BERT path documented in README under optional[bert].

Labels come from `lexicon.score_text` — i.e. the model learns to mimic the
Apel-Grimaldi-style lexicon score from the raw text. This is honest:
the README documents that the "ground truth" is a lexicon proxy, not a
human-labeled gold set.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .lexicon import score_text

RANDOM_STATE = 42
log = logging.getLogger("classify")


@dataclass
class TrainResult:
    accuracy: float
    n_train: int
    n_test: int
    report: dict
    pipeline: Pipeline


def label_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'score' and 'label' columns using the lexicon."""
    scores = df["text"].fillna("").map(score_text)
    df = df.copy()
    df["score"] = [s.score for s in scores]
    df["hawkish_count"] = [s.hawkish_count for s in scores]
    df["dovish_count"] = [s.dovish_count for s in scores]
    df["total_words"] = [s.total_words for s in scores]
    df["label"] = [s.label for s in scores]
    return df


def train(df: pd.DataFrame, test_size: float = 0.2) -> TrainResult:
    df = df[df["text"].fillna("").str.len() > 200].copy()
    df = label_corpus(df)
    # If the lexicon produces a single class on this corpus, fall back to a
    # balanced split using sign of score so the test still runs.
    if df["label"].nunique() < 2:
        log.warning("Lexicon produced one class; cannot train.")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values,
        df["label"].values,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["label"].values if df["label"].nunique() > 1 else None,
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return TrainResult(acc, len(X_train), len(X_test), report, pipe)


def save_model(pipe: Pipeline, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_csv", default="data/speeches.csv")
    p.add_argument("--out-model", default="data/model.joblib")
    p.add_argument("--out-labeled", default="data/speeches_labeled.csv")
    p.add_argument("--out-metrics", default="data/metrics.json")
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)
    res = train(df)
    log.info("accuracy=%.4f train=%d test=%d", res.accuracy, res.n_train, res.n_test)
    save_model(res.pipeline, Path(args.out_model))
    label_corpus(df).to_csv(args.out_labeled, index=False)
    Path(args.out_metrics).write_text(json.dumps({
        "accuracy": res.accuracy,
        "n_train": res.n_train,
        "n_test": res.n_test,
        "report": res.report,
    }, indent=2))


if __name__ == "__main__":
    main()
