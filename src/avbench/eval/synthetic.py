"""Synthetic-control harness — validating a scorer without human labels.

We can't measure a scorer against humans yet, so we manufacture pseudo-ground
truth we *control*:
  - known-CORRECT: the reference itself, a meaning-preserving paraphrase, and a
    verbose-padded reference.
  - known-INCORRECT: a field-corrupted reference, a verbose-padded corruption
    (verbosity-bias probe), and another sample's reference (easy negative).

Running a scorer over these yields precision/recall/accuracy plus targeted bias
probes:
  - paraphrase flip-rate  : how often it wrongly rejects valid paraphrases (over-
    strictness / lexical bias) — lower is better.
  - corrupt catch-rate    : how often it correctly rejects corruptions — higher
    is better. Subtle single-word flips are the hard case.
  - verbosity-bias delta  : does padding a *wrong* answer rescue it? (catch-rate
    on corrupt vs corrupt_verbose) — near zero is good.

This won't match the distribution of real model errors, but it bounds the
scorer's discriminative power and exposes specific failure modes — and it's the
same harness you'd reuse to vet an NLI or LLM judge later.
"""

import random
import re
from collections import defaultdict
from typing import Dict, List, Optional

_FILLER = ("Based on a careful and detailed analysis of all six surround camera "
           "views and the full driving context, ")

# Meaning-PRESERVING rewrites (for positives).
_PARAPHRASE = [
    ("keep going", "continue"), ("the ego vehicle", "the ego car"),
    ("stationary", "not moving"), ("going ahead", "moving forward"),
    ("at the same speed", "without changing speed"), ("decelerate", "slow down"),
    ("back of", "rear of"), ("there is", "we can see"), ("high", "large"),
]

# Meaning-CHANGING edits (for negatives). Each is an antonym/identity flip.
_FLIPS = [
    ("stationary", "moving"), ("moving", "stationary"),
    ("keep going", "stop"), ("going ahead", "turning around"),
    ("left", "right"), ("right", "left"), ("accelerate", "decelerate"),
    ("front", "back"), ("high", "low"), ("not moving", "moving fast"),
]
_NOUN_SWAP = [("car", "pedestrian"), ("sedan", "truck"), ("truck", "bicycle"),
              ("pedestrian", "car")]


def paraphrase(text: str) -> str:
    out = text
    for a, b in _PARAPHRASE:
        out = re.sub(re.escape(a), b, out, flags=re.IGNORECASE)
    return out if out != text else _FILLER + text  # ensure it differs


def corrupt(text: str, rng: random.Random) -> Optional[str]:
    """Return a meaning-changed version, or None if no edit applies."""
    t = (text or "").strip()
    if re.fullmatch(r"[A-E]", t, re.IGNORECASE):  # MCQ letter -> different letter
        opts = [c for c in "ABCD" if c != t.upper()]
        return rng.choice(opts)
    for a, b in _FLIPS + _NOUN_SWAP:
        if re.search(re.escape(a), t, re.IGNORECASE):
            return re.sub(re.escape(a), b, t, count=1, flags=re.IGNORECASE)
    return None  # caller falls back to a mismatch negative


def make_controls(samples: List[dict], seed: int = 0) -> List[dict]:
    rng = random.Random(seed)
    golds = [s["answer"] for s in samples]
    controls: List[dict] = []
    for i, s in enumerate(samples):
        gold = s["answer"]
        meta = {"sample_id": s["sample_id"], "task": s.get("task_type"),
                "gold": gold, "prompt_format": s.get("prompt_format")}

        def add(kind, pred, label):
            controls.append(dict(meta, kind=kind, pred=pred, label=label))

        add("identity", gold, 1)
        add("paraphrase", paraphrase(gold), 1)
        add("verbose", _FILLER + gold, 1)

        bad = corrupt(gold, rng)
        if bad is not None:
            add("corrupt", bad, 0)
            add("corrupt_verbose", _FILLER + bad, 0)
        # easy negative: a different sample's reference
        others = [g for j, g in enumerate(golds) if j != i and g.strip() != gold.strip()]
        if others:
            add("mismatch", rng.choice(others), 0)
    return controls


def evaluate_scorer(scorer, controls: List[dict]) -> Dict:
    tp = fp = tn = fn = 0
    by_kind = defaultdict(lambda: [0, 0])  # kind -> [n, n_judged_correct]
    for c in controls:
        pred_correct = scorer.is_correct(c["pred"], c["gold"], c)
        by_kind[c["kind"]][0] += 1
        by_kind[c["kind"]][1] += pred_correct
        if c["label"] == 1 and pred_correct:
            tp += 1
        elif c["label"] == 1 and not pred_correct:
            fn += 1
        elif c["label"] == 0 and pred_correct:
            fp += 1
        else:
            tn += 1

    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    acc = (tp + tn) / n if n else float("nan")

    def rate(kind):  # fraction judged correct
        d = by_kind.get(kind)
        return (d[1] / d[0]) if d and d[0] else float("nan")

    catch = lambda k: (1 - rate(k)) if rate(k) == rate(k) else float("nan")
    return {
        "n_controls": n,
        "precision": prec, "recall": rec, "accuracy": acc,
        "paraphrase_flip_rate": (1 - rate("paraphrase")) if rate("paraphrase") == rate("paraphrase") else float("nan"),
        "verbose_flip_rate": (1 - rate("verbose")) if rate("verbose") == rate("verbose") else float("nan"),
        "corrupt_catch_rate": catch("corrupt"),
        "corrupt_verbose_catch_rate": catch("corrupt_verbose"),
        "mismatch_catch_rate": catch("mismatch"),
        "by_kind_n": {k: v[0] for k, v in by_kind.items()},
    }
