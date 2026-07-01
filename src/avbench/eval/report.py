"""Join predictions with gold and assign each a selective-prediction outcome.

Every prediction is one of three things:
  - answered  — the model committed to an answer; it gets scored (correct/incorrect)
                and, if it reported a confidence, enters the calibration table.
  - abstained — the model declined ("I cannot determine ..."); a refusal is neither
                right nor wrong, so it is kept OUT of accuracy/ECE/AUROC and the
                confusion matrix, and lowers *coverage* instead.
  - error / unknown-sample — dropped (counted separately).

Keeping this join/partition here (not in scripts/evaluate.py) makes the abstention
accounting unit-testable and leaves the script as pure formatting. Selective
accuracy is confidence-independent — an answered item with no confidence still
counts toward coverage and accuracy; it is simply absent from the calibration table.
"""

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Tuple

from avbench.eval.scorer import Scorer, answer_label


@dataclass
class Row:
    group: str
    abstained: bool
    correct: Optional[int]        # None iff abstained
    confidence: Optional[float]   # None if abstained or the model gave no confidence
    gold_label: Optional[str]     # None iff abstained (excluded from confusion matrix)
    pred_label: Optional[str]


def build_rows(preds: Iterable[dict], gold: Mapping[str, dict], scorer: Scorer,
               by: str = "task") -> Tuple[List[Row], int]:
    """Score each prediction against its gold sample. Returns (rows, n_err).

    Abstentions become Row(abstained=True, correct=None, ...) — not scored wrong,
    not counted as missing-confidence. Predictions with an `error` field or no
    matching gold sample are skipped (errors counted into n_err)."""
    rows: List[Row] = []
    n_err = 0
    for p in preds:
        g = gold.get(p["sample_id"])
        if g is None:
            continue
        if p.get("error"):
            n_err += 1
            continue
        group = g["task_type"] if by == "task" else str((p.get("condition") or {}).get(by, "?"))
        if p.get("abstained"):
            rows.append(Row(group=group, abstained=True, correct=None,
                            confidence=None, gold_label=None, pred_label=None))
            continue
        conf = p.get("verbal_confidence")
        rows.append(Row(
            group=group,
            abstained=False,
            correct=scorer.is_correct(p.get("answer"), g["answer"], g),
            confidence=None if conf is None else float(conf),
            gold_label=answer_label(g["answer"], g["answer"]),
            pred_label=answer_label(p.get("answer"), g["answer"]),
        ))
    return rows, n_err


def coverage(rows: List[Row]) -> float:
    """Fraction of predictions the model answered rather than abstained on."""
    if not rows:
        return float("nan")
    answered = sum(1 for r in rows if not r.abstained)
    return answered / len(rows)
