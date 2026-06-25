"""Calibration + discrimination metrics for the O2 slice.

Calibration (given joined (correctness, confidence) pairs):
  - accuracy
  - ECE (expected calibration error, equal-width bins)
  - AUROC (can confidence separate correct from incorrect answers?)

Discrimination (given joined (gold_label, pred_label) pairs, categorical answers):
  - confusion_matrix / recall_per_class / precision_per_class / balanced_accuracy
These exist because plain accuracy is a base-rate trap on imbalanced categorical
tasks (e.g. yes/no identification is ~97% "No": a constant-"No" model scores ~97%
but has zero positive-class recall). Balanced accuracy + per-class recall expose
that; AUROC above is calibration, not discrimination — keep the two separate.

NOTE on free-form answers: every metric here consumes either a binary `correct` or
a categorical label, never raw text. For open-ended answers that binary/label comes
from a Scorer (eval/scorer.py), and the current `structured` scorer reduces text to
correctness via thresholded token-F1 — which is lossy: it misses subtle single-token
corruptions (corrupt-catch ~0.56) and binarizes partial matches. So on free-form
tasks these numbers are only as trustworthy as that approximation. We need to handle
free-form more gracefully before reporting open-ended calibration as headline:
Layer 2 (NLI claim-entailment) and Layer 3 (cross-family LLM judge) per TODO.md, and
likely a *soft* correctness signal (use the continuous F1/entailment score, not just
correct>=threshold) so a near-miss isn't scored identically to a confident hallucination.
Confusion/recall below only make sense for categorical answers; callers must gate on a
small label space (open-ended free-form has ~unique labels and should be skipped).
"""

from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

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
    """Probability a correct answer is ranked above an incorrect one by confidence.

    This is CALIBRATION AUROC (label = correctness, score = confidence): does the
    model know when it is right? It is task-agnostic — works for yes/no, MCQ, and
    open-ended alike, since it only needs (correct, confidence). Do not confuse it
    with class-discrimination AUROC (label = gold class); that does not generalize
    past binary tasks and is not what this benchmark reports. For discrimination on
    categorical tasks use balanced_accuracy / recall_per_class below."""
    if len(set(correct)) < 2:
        return float("nan")  # undefined when all correct or all wrong
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(correct, conf))
    except Exception:
        return float("nan")


# --- discrimination (categorical answers) ---------------------------------------
# Plain accuracy hides positive-class failure under class imbalance. These operate
# on categorical labels (gold_label, pred_label) — caller derives those from text
# (scorer.answer_label) and must gate on a small label space.


def confusion_matrix(
    gold: Sequence[str], pred: Sequence[str], labels: Optional[Sequence[str]] = None
) -> Tuple[List[str], List[List[int]]]:
    """Counts[g][p] = #(gold==g, pred==p). Returns (labels, matrix). Labels default
    to the sorted union of gold+pred so unseen predictions still get a column."""
    if labels is None:
        labels = sorted(set(gold) | set(pred))
    labels = list(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        if g in index and p in index:
            m[index[g]][index[p]] += 1
    return labels, m


def recall_per_class(
    gold: Sequence[str], pred: Sequence[str], labels: Optional[Sequence[str]] = None
) -> "OrderedDict[str, float]":
    """Per-class recall = TP / support. NaN for a class with no gold support."""
    if labels is None:
        labels = sorted(set(gold) | set(pred))
    out: "OrderedDict[str, float]" = OrderedDict()
    for lab in labels:
        support = sum(1 for g in gold if g == lab)
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        out[lab] = (tp / support) if support else float("nan")
    return out


def precision_per_class(
    gold: Sequence[str], pred: Sequence[str], labels: Optional[Sequence[str]] = None
) -> "OrderedDict[str, float]":
    """Per-class precision = TP / #predicted. NaN for a class never predicted."""
    if labels is None:
        labels = sorted(set(gold) | set(pred))
    out: "OrderedDict[str, float]" = OrderedDict()
    for lab in labels:
        predicted = sum(1 for p in pred if p == lab)
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        out[lab] = (tp / predicted) if predicted else float("nan")
    return out


def balanced_accuracy(
    gold: Sequence[str], pred: Sequence[str], labels: Optional[Sequence[str]] = None
) -> float:
    """Macro-average of per-class recall — the imbalance-robust accuracy. On the
    yes/no base-rate trap, a constant-"No" model gets balanced_accuracy 0.5 (not
    ~0.97), exposing zero positive-class recall."""
    rec = recall_per_class(gold, pred, labels)
    vals = [v for v in rec.values() if v == v]  # drop NaN (classes with no support)
    return float(np.mean(vals)) if vals else float("nan")
