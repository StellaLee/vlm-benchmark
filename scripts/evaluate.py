#!/usr/bin/env python3
"""Join curated Samples with Predictions and report calibration metrics.

Reports accuracy / ECE / AUROC overall and stratified by task_type — the O2-KR2
hypothesis is that VLMs are well-calibrated on scene-level understanding but poor
on localized/fine-grained tasks.

Example:
  python scripts/evaluate.py --curated data/curated/drive_v1.jsonl \
      --pred runs/mock_vc.jsonl
"""

import argparse
from collections import defaultdict

import _bootstrap  # noqa: F401

from avbench.eval.metrics import (
    accuracy,
    auroc,
    balanced_accuracy,
    confusion_matrix,
    expected_calibration_error,
    precision_per_class,
    recall_per_class,
)
from avbench.eval.scorer import answer_label, get_scorer
from avbench.io_utils import read_jsonl

# Show the confusion matrix / per-class recall only when answers are categorical with
# a small label space (yes/no, MCQ). Open-ended free-form has ~unique labels per row,
# where a confusion matrix is meaningless, so we skip it past this many classes.
CATEGORICAL_MAX_LABELS = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--scorer", default="exact",
                    help="correctness scorer: exact | structured (for open-ended)")
    ap.add_argument("--threshold", type=float, default=0.5, help="structured-scorer cutoff")
    ap.add_argument("--by", default="task",
                    help="stratify by: task | a condition key (e.g. marker_grounding)")
    args = ap.parse_args()

    kw = {"threshold": args.threshold} if args.scorer == "structured" else {}
    scorer = get_scorer(args.scorer, **kw)

    gold = {r["sample_id"]: r for r in read_jsonl(args.curated)}
    rows = []  # (group, correct, confidence, gold_label, pred_label)
    n_missing_conf = 0
    n_err = 0
    for p in read_jsonl(args.pred):
        g = gold.get(p["sample_id"])
        if g is None:
            continue
        if p.get("error"):
            n_err += 1
            continue
        conf = p.get("verbal_confidence")
        if conf is None:
            n_missing_conf += 1
            continue
        correct = scorer.is_correct(p.get("answer"), g["answer"], g)
        if args.by == "task":
            group = g["task_type"]
        else:
            group = str((p.get("condition") or {}).get(args.by, "?"))
        gold_label = answer_label(g["answer"], g["answer"])
        pred_label = answer_label(p.get("answer"), g["answer"])
        rows.append((group, correct, float(conf), gold_label, pred_label))

    if not rows:
        print("No scorable rows (errors={}, missing-confidence={}).".format(n_err, n_missing_conf))
        return

    by_task = defaultdict(lambda: ([], []))
    allc, allf = [], []
    for task, c, f, _gl, _pl in rows:
        by_task[task][0].append(c)
        by_task[task][1].append(f)
        allc.append(c)
        allf.append(f)

    print("\nCalibration (confidence vs correctness):")
    print("{:<14} {:>6} {:>8} {:>8} {:>8}".format(args.by, "n", "acc", "ECE", "AUROC"))
    print("-" * 48)
    for task in sorted(by_task):
        c, f = by_task[task]
        print("{:<14} {:>6} {:>8.3f} {:>8.3f} {:>8}".format(
            task, len(c), accuracy(c),
            expected_calibration_error(c, f, args.bins),
            _fmt(auroc(c, f)),
        ))
    print("-" * 48)
    print("{:<14} {:>6} {:>8.3f} {:>8.3f} {:>8}".format(
        "OVERALL", len(allc), accuracy(allc),
        expected_calibration_error(allc, allf, args.bins),
        _fmt(auroc(allc, allf)),
    ))

    _report_discrimination(rows, args.by)

    if n_err or n_missing_conf:
        print("\nskipped: errors={}, missing-confidence={}".format(n_err, n_missing_conf))


def _report_discrimination(rows, by: str) -> None:
    """Confusion matrix + per-class recall/precision + balanced accuracy, per group
    and overall. Only for categorical answers (small label space) — accuracy is a
    base-rate trap under class imbalance, this exposes positive-class recall."""
    labels = sorted({gl for _, _, _, gl, _ in rows} | {pl for _, _, _, _, pl in rows})
    if len(labels) > CATEGORICAL_MAX_LABELS:
        return  # open-ended free-form: confusion matrix is meaningless

    groups = sorted({g for g, *_ in rows})
    blocks = [(g, [(gl, pl) for grp, _, _, gl, pl in rows if grp == g]) for g in groups]
    if len(groups) > 1:  # add an overall block when stratified
        blocks.append(("OVERALL", [(gl, pl) for _, _, _, gl, pl in rows]))

    print("\nDiscrimination (categorical answers; '{}' groups):".format(by))
    for name, pairs in blocks:
        gold = [gl for gl, _ in pairs]
        pred = [pl for _, pl in pairs]
        labs, mat = confusion_matrix(gold, pred, labels)
        rec = recall_per_class(gold, pred, labs)
        prec = precision_per_class(gold, pred, labs)
        bal = balanced_accuracy(gold, pred, labs)
        print("\n  [{}]  n={}  balanced_acc={}".format(name, len(pairs), _fmt(bal)))
        head = "    {:>10} |".format("gold\\pred")
        head += "".join("{:>8}".format(_short(l)) for l in labs) + "  | recall  support"
        print(head)
        print("    " + "-" * (len(head) - 4))
        for i, l in enumerate(labs):
            line = "    {:>10} |".format(_short(l))
            line += "".join("{:>8}".format(mat[i][j]) for j in range(len(labs)))
            line += "  | {:>6}  {:>7}".format(_fmt(rec[l]), sum(mat[i]))  # support = row sum
            print(line)
        print("    precision: " + "  ".join("{}={}".format(_short(l), _fmt(prec[l])) for l in labs))


def _short(label: str, width: int = 8) -> str:
    return label if len(label) <= width else label[: width - 1] + "…"


def _fmt(x: float) -> str:
    return "n/a" if x != x else "{:.3f}".format(x)  # x!=x -> NaN


if __name__ == "__main__":
    main()
