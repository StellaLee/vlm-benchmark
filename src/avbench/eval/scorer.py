"""Correctness scorers: turn an open-ended (prediction, reference) pair into a
binary correct/incorrect label that ECE/AUROC can consume.

Design goal — minimal discretion, maximal transparency (see the brainstorm in the
project notes). Layer 1, implemented here, is a *deterministic, reference-grounded*
scorer: no model, so no self-preference / verbosity / leniency bias by
construction. Its only bias is rigidity (it can under-credit valid paraphrases the
lexicon misses), which is conservative — it under-credits rather than rewarding
fluent-but-ungrounded answers.

Future layers plug in behind the same `Scorer` interface and registry:
  - Layer 2: NLI claim-entailment (deterministic discriminative model)
  - Layer 3: constrained, confidence-blind, cross-family LLM judge
Add a new file with `@register_scorer("nli")` / `@register_scorer("llm_judge")`;
nothing downstream changes. `evaluate.py --scorer <name>` selects one.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, Field

# Tiny, fixed, *documented* resources. Extend deliberately — every entry is a
# transparent modeling choice, not a learned/opaque one.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "at",
    "and", "or", "there", "this", "that", "these", "those", "it", "its", "be",
    "as", "for", "with", "by", "ego", "vehicle", "object", "objects", "scene",
    "should", "will", "can", "could", "be", "based", "which", "what", "id", "ids",
}
_SYNONYMS = {
    "sedan": "car", "suv": "car", "auto": "car", "automobile": "car",
    "ped": "pedestrian", "person": "pedestrian", "people": "pedestrian",
    "lorry": "truck", "bike": "bicycle", "cyclist": "bicycle",
    "stationary": "stopped", "parked": "stopped", "still": "stopped",
    "moving": "moving", "ahead": "forward", "straight": "forward",
}

_LETTER = re.compile(r"\b([A-E])\b")
_OBJ_REF = re.compile(r"<c\d+,[^>]*>")


class ScoreResult(BaseModel):
    score: float  # 0..1 graded consistency with the reference
    correct: int  # binarized via the scorer's threshold (0/1)
    method: str
    details: Dict[str, Any] = Field(default_factory=dict)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).strip()


def content_tokens(text: str) -> set:
    toks = set()
    for t in normalize(text).split():
        if t in _STOPWORDS or len(t) <= 1:
            continue
        toks.add(_SYNONYMS.get(t, t))
    return toks


def _looks_like_mcq(gold: str) -> bool:
    return bool(re.fullmatch(r"[A-E]", (gold or "").strip(), re.IGNORECASE))


def set_f1(pred: set, gold: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(gold)
    return 2 * prec * rec / (prec + rec)


class Scorer(ABC):
    name: str
    threshold: float = 0.5

    @abstractmethod
    def score(self, pred: Optional[str], gold: str, sample: Optional[dict] = None) -> ScoreResult:
        ...

    def is_correct(self, pred: Optional[str], gold: str, sample: Optional[dict] = None) -> int:
        return self.score(pred, gold, sample).correct


_SCORERS: Dict[str, Type[Scorer]] = {}


def register_scorer(name: str) -> Callable[[Type[Scorer]], Type[Scorer]]:
    def deco(cls: Type[Scorer]) -> Type[Scorer]:
        cls.name = name
        _SCORERS[name] = cls
        return cls

    return deco


def get_scorer(name: str, **kwargs) -> Scorer:
    if name not in _SCORERS:
        raise KeyError("Unknown scorer '{}'. Registered: {}".format(name, sorted(_SCORERS)))
    return _SCORERS[name](**kwargs)


@register_scorer("exact")
class ExactScorer(Scorer):
    """Normalized exact match (MCQ letter or whole-string). The strictest, most
    conservative baseline."""

    def score(self, pred, gold, sample=None) -> ScoreResult:
        if _looks_like_mcq(gold):
            lm = _LETTER.search(pred or "")
            got = lm.group(1).upper() if lm else normalize(pred)
            c = int(got == gold.strip().upper())
        else:
            c = int(normalize(pred) == normalize(gold))
        return ScoreResult(score=float(c), correct=c, method=self.name)


@register_scorer("structured")
class StructuredScorer(Scorer):
    """Deterministic, reference-grounded Layer-1 scorer.

    MCQ -> exact letter. Open-ended -> set-F1 over content tokens (stopword-
    filtered, synonym-canonicalized), with a bonus requirement that referenced
    object IDs (<c1,...>) overlap when the reference names any. `correct` is
    `score >= threshold`. F1 balances recall (did it capture the reference facts?)
    with precision (did it hallucinate extra claims?) — the latter matters for a
    self-knowledge benchmark.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, pred, gold, sample=None) -> ScoreResult:
        if _looks_like_mcq(gold):
            lm = _LETTER.search(pred or "")
            got = lm.group(1).upper() if lm else ""
            c = int(got == gold.strip().upper())
            return ScoreResult(score=float(c), correct=c, method=self.name,
                               details={"mode": "mcq", "got": got, "gold": gold.strip().upper()})

        p_tok, g_tok = content_tokens(pred or ""), content_tokens(gold)
        f1 = set_f1(p_tok, g_tok)

        # If the reference cites object IDs, require some overlap (grounding gate).
        g_refs, p_refs = set(_OBJ_REF.findall(gold or "")), set(_OBJ_REF.findall(pred or ""))
        ref_ok = True
        if g_refs:
            ref_ok = bool(g_refs & p_refs)

        score = f1 if ref_ok else min(f1, self.threshold - 1e-6)  # fail the gate
        return ScoreResult(
            score=score,
            correct=int(score >= self.threshold),
            method=self.name,
            details={"mode": "open", "f1": round(f1, 3),
                     "overlap": sorted(p_tok & g_tok), "ref_gate": ref_ok},
        )
