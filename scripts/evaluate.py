#!/usr/bin/env python3
"""Join curated Samples with Predictions and report calibration metrics.

Reports accuracy / ECE / AUROC overall and stratified by task_type — the O2-KR2
hypothesis is that VLMs are well-calibrated on scene-level understanding but poor
on localized/fine-grained tasks.

Abstention-aware: when a prediction declines ("I cannot determine ...") it is a
refusal, not an error. Abstentions are reported as reduced *coverage* (a
selective-prediction table) and kept out of accuracy / ECE / AUROC / the confusion
matrix, which are computed over the *answered* items only.

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
from avbench.eval.report import build_rows, coverage
from avbench.eval.scorer import get_scorer
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
                    help="stratify by: task | a condition key (e.g. layout, marker_grounding)")
    args = ap.parse_args()

    kw = {"threshold": args.threshold} if args.scorer == "structured" else {}
    scorer = get_scorer(args.scorer, **kw)

    gold = {r["sample_id"]: r for r in read_jsonl(args.curated)}
    rows, n_err = build_rows(read_jsonl(args.pred), gold, scorer, by=args.by)

    if not rows:
        print("No scorable rows (errors={}).".format(n_err))
        return

    answered = [r for r in rows if not r.abstained]
    n_abstained = len(rows) - len(answered)
    n_missing_conf = sum(1 for r in answered if r.confidence is None)

    if n_abstained:
        _report_coverage(rows, args.by)

    _report_calibration([r for r in answered if r.confidence is not None], args.by, args.bins)
    _report_discrimination(answered, args.by)

    notes = []
    if n_err:
        notes.append("errors={}".format(n_err))
    if n_abstained:
        notes.append("abstained={} (excluded from accuracy/calibration)".format(n_abstained))
    if n_missing_conf:
        notes.append("answered-without-confidence={} (in accuracy, not calibration)".format(n_missing_conf))
    if notes:
        print("\nnotes: " + ", ".join(notes))


def _report_coverage(rows, by: str) -> None:
    """Selective-prediction view: how often the model answered vs abstained, and its
    accuracy on the answered slice. Only shown when some prediction abstained."""
    groups = defaultdict(list)
    for r in rows:
        groups[r.group].append(r)
    print("\nSelective prediction (abstention-aware):")
    print("{:<14} {:>6} {:>9} {:>9} {:>9}".format(by, "n", "answered", "coverage", "sel_acc"))
    print("-" * 50)

    def line(name, group_rows):
        ans = [r for r in group_rows if not r.abstained]
        sel = accuracy([r.correct for r in ans]) if ans else float("nan")
        print("{:<14} {:>6} {:>9} {:>9.3f} {:>9}".format(
            name, len(group_rows), len(ans), coverage(group_rows), _fmt(sel)))

    for g in sorted(groups):
        line(g, groups[g])
    print("-" * 50)
    line("OVERALL", rows)


def _report_calibration(rows, by: str, bins: int) -> None:
    """Accuracy / ECE / AUROC over answered items that reported a confidence."""
    print("\nCalibration (confidence vs correctness; answered items):")
    print("{:<14} {:>6} {:>8} {:>8} {:>8}".format(by, "n", "acc", "ECE", "AUROC"))
    print("-" * 48)
    if not rows:
        print("(no answered items with a confidence value)")
        return

    by_group = defaultdict(lambda: ([], []))
    allc, allf = [], []
    for r in rows:
        by_group[r.group][0].append(r.correct)
        by_group[r.group][1].append(r.confidence)
        allc.append(r.correct)
        allf.append(r.confidence)

    for g in sorted(by_group):
        c, f = by_group[g]
        print("{:<14} {:>6} {:>8.3f} {:>8.3f} {:>8}".format(
            g, len(c), accuracy(c), expected_calibration_error(c, f, bins), _fmt(auroc(c, f))))
    print("-" * 48)
    print("{:<14} {:>6} {:>8.3f} {:>8.3f} {:>8}".format(
        "OVERALL", len(allc), accuracy(allc),
        expected_calibration_error(allc, allf, bins), _fmt(auroc(allc, allf))))


def _report_discrimination(rows, by: str) -> None:
    """Confusion matrix + per-class recall/precision + balanced accuracy, per group
    and overall. Only for categorical answers (small label space) — accuracy is a
    base-rate trap under class imbalance, this exposes positive-class recall.
    Consumes answered rows only; abstentions carry no (gold, pred) label pair."""
    pairs_all = [(r.gold_label, r.pred_label) for r in rows]
    labels = sorted({gl for gl, _ in pairs_all} | {pl for _, pl in pairs_all})
    if len(labels) > CATEGORICAL_MAX_LABELS:
        return  # open-ended free-form: confusion matrix is meaningless

    groups = sorted({r.group for r in rows})
    blocks = [(g, [(r.gold_label, r.pred_label) for r in rows if r.group == g]) for g in groups]
    if len(groups) > 1:  # add an overall block when stratified
        blocks.append(("OVERALL", pairs_all))

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
