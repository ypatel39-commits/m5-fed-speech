"""Tests for the classifier on a tiny synthetic corpus."""
import pandas as pd

from m5_fed_speech.classify import label_corpus, train


def _toy_df() -> pd.DataFrame:
    hawk = (
        "Inflation risk remains elevated and the Committee may need to "
        "tighten further. We are prepared to raise rates if the economy "
        "shows signs of overheating. Restrictive policy will continue."
    ) * 5
    dove = (
        "Downside risk has increased. The Committee is patient and stands "
        "ready to ease policy and lower rates to support the recovery. "
        "Accommodation will remain in place."
    ) * 5
    rows = []
    for i in range(12):
        rows.append({"date": f"2020-01-{i+1:02d}", "title": f"h{i}", "url": f"u{i}", "text": hawk, "source": "synth"})
    for i in range(12):
        rows.append({"date": f"2020-02-{i+1:02d}", "title": f"d{i}", "url": f"u_d{i}", "text": dove, "source": "synth"})
    return pd.DataFrame(rows)


def test_label_corpus_assigns_both_classes():
    df = label_corpus(_toy_df())
    assert set(df["label"].unique()) == {"hawkish", "dovish"}


def test_train_produces_reasonable_accuracy():
    res = train(_toy_df(), test_size=0.34)
    assert 0.0 <= res.accuracy <= 1.0
    # On a clean synthetic corpus, accuracy should be high.
    assert res.accuracy >= 0.7
