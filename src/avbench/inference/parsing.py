"""Extract a parsed answer, MCQ letter, and verbal confidence from raw text."""

import re
from typing import Optional

from avbench.schema import PromptFormat, Sample

_CONF_PATTERNS = [
    re.compile(r"confidence[^0-9]{0,15}(\d{1,3})\s*%?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%\s*confiden", re.IGNORECASE),
]
_ANSWER_LINE = re.compile(r"(?:final\s*)?answer\s*[:\-]\s*(.+)", re.IGNORECASE)
_MCQ_LETTER = re.compile(r"\b([A-E])\b")

_ABSTAIN_MARKERS = (
    "cannot determine",
    "can't determine",
    "insufficient",
    "not enough information",
    "unable to determine",
)


def extract_confidence(text: str) -> Optional[float]:
    for pat in _CONF_PATTERNS:
        m = pat.search(text)
        if m:
            val = float(m.group(1))
            return max(0.0, min(1.0, val / 100.0))
    return None


def is_abstention(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ABSTAIN_MARKERS)


def extract_answer(text: str, sample: Sample) -> str:
    """Pull the answer span; for MCQ normalize to a single letter when possible."""
    m = _ANSWER_LINE.search(text)
    span = m.group(1).strip() if m else text.strip()
    span = span.splitlines()[0].strip() if span else span

    if sample.prompt_format == PromptFormat.MCQ:
        lm = _MCQ_LETTER.search(span)
        if lm:
            return lm.group(1).upper()
    return span
