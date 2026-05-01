"""Hawkish/Dovish lexicon and scoring.

Starter list inspired by Apel & Grimaldi (2012) and Picault & Renault (2017).
Kept intentionally small and transparent — the README documents this and
notes that production work would expand and test the list.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HAWKISH_WORDS: list[str] = [
    "tighten", "tightening", "tightened",
    "raise", "raising", "raised",
    "hike", "hiking", "hiked",
    "inflation risk", "inflationary",
    "overheating", "overheat",
    "elevated", "restrictive",
    "above target", "above our target",
    "withdraw accommodation",
    "upside risk",
]

DOVISH_WORDS: list[str] = [
    "accommodate", "accommodative", "accommodation",
    "ease", "easing", "eased",
    "lower", "lowering", "lowered",
    "cut", "cuts", "cutting",
    "stimulus", "stimulative",
    "downside risk",
    "below target", "below our target",
    "patient", "patience",
    "supportive", "support the recovery",
    "subdued",
]

_TOKEN_RE = re.compile(r"[A-Za-z']+")


@dataclass
class SpeechScore:
    hawkish_count: int
    dovish_count: int
    total_words: int
    score: float          # (h - d) / total_words * 1000
    label: str            # "hawkish" or "dovish"


def _count_phrases(text: str, phrases: list[str]) -> int:
    """Case-insensitive count of each phrase in text. Phrases can be multi-word."""
    lowered = text.lower()
    total = 0
    for p in phrases:
        # word-boundary on edges so "raise" doesn't match "raisefoo"
        pattern = r"\b" + re.escape(p.lower()) + r"\b"
        total += len(re.findall(pattern, lowered))
    return total


def word_count(text: str) -> int:
    """Cheap word tokenizer; good enough for a per-1000-words rate."""
    return len(_TOKEN_RE.findall(text))


def score_text(
    text: str,
    hawkish: list[str] | None = None,
    dovish: list[str] | None = None,
) -> SpeechScore:
    """Score one speech. Returns SpeechScore.

    score = (hawk - dove) / total_words * 1000
    label = 'hawkish' if score > 0 else 'dovish' (ties → dovish, conservative).
    """
    hawkish = hawkish or HAWKISH_WORDS
    dovish = dovish or DOVISH_WORDS
    if not isinstance(text, str) or not text.strip():
        return SpeechScore(0, 0, 0, 0.0, "dovish")
    h = _count_phrases(text, hawkish)
    d = _count_phrases(text, dovish)
    total = word_count(text)
    score = ((h - d) / total * 1000) if total > 0 else 0.0
    label = "hawkish" if score > 0 else "dovish"
    return SpeechScore(h, d, total, score, label)
