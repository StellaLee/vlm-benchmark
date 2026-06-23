"""Calibration metrics for the O2 slice.

Given joined (correctness, confidence) pairs:
  - accuracy
  - ECE (expected calibration error, equal-width bins)
  - AUROC (can confidence separate correct from incorrect answers?)

MCQ correctness is exact-match on the option letter. Open-ended QA/CAP need
fuzzier scoring (BLEU / GPT-score) — out of scope for this prototype, which is
why we lead with MCQ.
"""

from typing import List, Optional

import numpy as np


def exact_match(pred: Optional[str], gold: str) -> int:
    if pred is None:
        return 0
    return int(pred.strip().lower() == gold.strip().lower())


def accuracy(correct: List[int]) -> float:
    return float(np.mean(correct)) if correct else float("nan")


def expected_calibration_error(correct: List[int], conf: List[float], n_bins: int = 10) -> float:
    if not correct:
        return float("nan")
    correct_a = np.asarray(correct, dtype=float)
    conf_a = np.asarray(conf, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf_a, bins[1:-1], right=False)
    ece = 0.0
    n = len(correct_a)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        acc = correct_a[mask].mean()
        avg_conf = conf_a[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def auroc(correct: List[int], conf: List[float]) -> float:
    """Probability a correct answer is ranked above an incorrect one by confidence."""
    if len(set(correct)) < 2:
        return float("nan")  # undefined when all correct or all wrong
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(correct, conf))
    except Exception:
        return float("nan")
