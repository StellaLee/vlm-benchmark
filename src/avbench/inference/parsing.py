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
# glm-4.1v-thinking wraps its final answer in these markers after the reasoning.
_BOX = re.compile(r"<\|begin_of_box\|>\s*(.*?)\s*<\|end_of_box\|>", re.DOTALL)

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


def _norm_opt(s: str) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation — for matching
    an answer's quoted option text against the choice list."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def _letter_from_options(text: str, options) -> Optional[str]:
    """Recover an MCQ letter when the model quotes an option's text but no bare
    letter (e.g. Gemini's prose '...status is "Going ahead."'). Returns the letter
    only if exactly one option's text appears, so ambiguous answers aren't guessed."""
    if not options:
        return None
    norm_text = _norm_opt(text)
    hits = []
    for i, opt in enumerate(options):
        opt_n = _norm_opt(opt)
        if opt_n and opt_n in norm_text:
            hits.append(chr(ord("A") + i))
    return hits[0] if len(hits) == 1 else None


def extract_answer(text: str, sample: Sample) -> str:
    """Pull the answer span; for MCQ normalize to a single letter when possible."""
    # A boxed span (glm-4.1v-thinking) is the model's own final-answer marker, so
    # it takes precedence over the "Answer:" line and over reasoning text.
    box = _BOX.search(text or "")
    if box:
        span = box.group(1).strip()
    else:
        m = _ANSWER_LINE.search(text)
        span = m.group(1).strip() if m else text.strip()
    span = span.splitlines()[0].strip() if span else span

    if sample.prompt_format == PromptFormat.MCQ:
        lm = _MCQ_LETTER.search(span)
        if lm:
            return lm.group(1).upper()
        # No bare letter: fall back to matching the quoted option text. Search the
        # full text (not just the first line) so verbose answers still resolve.
        letter = _letter_from_options(text, sample.options)
        if letter:
            return letter
    return span
