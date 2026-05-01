"""Tests for the hawkish/dovish lexicon."""
from m5_fed_speech.lexicon import score_text, word_count


def test_hawkish_text_scores_positive():
    text = (
        "Inflation risk remains elevated and we may need to tighten policy "
        "further. The Committee is prepared to raise rates if conditions warrant."
    )
    s = score_text(text)
    assert s.hawkish_count >= 2
    assert s.score > 0
    assert s.label == "hawkish"


def test_dovish_text_scores_negative():
    text = (
        "Downside risk has increased; the Committee remains patient and "
        "stands ready to ease policy and lower rates to support the recovery."
    )
    s = score_text(text)
    assert s.dovish_count >= 2
    assert s.score < 0
    assert s.label == "dovish"


def test_empty_text_safe():
    s = score_text("")
    assert s.total_words == 0
    assert s.score == 0.0
    assert s.label == "dovish"


def test_word_count_basic():
    assert word_count("Hello, world! It's a test.") == 5
