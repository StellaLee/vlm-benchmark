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

from avbench.eval.metrics import accuracy, auroc, expected_calibration_error
from avbench.eval.scorer import get_scorer
from avbench.io_utils import read_jsonl


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
    rows = []  # (task_type, correct, confidence)
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
        rows.append((group, correct, float(conf)))

    if not rows:
        print("No scorable rows (errors={}, missing-confidence={}).".format(n_err, n_missing_conf))
        return

    by_task = defaultdict(lambda: ([], []))
    allc, allf = [], []
    for task, c, f in rows:
        by_task[task][0].append(c)
        by_task[task][1].append(f)
        allc.append(c)
        allf.append(f)

    print("\n{:<14} {:>6} {:>8} {:>8} {:>8}".format(args.by, "n", "acc", "ECE", "AUROC"))
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
    if n_err or n_missing_conf:
        print("\nskipped: errors={}, missing-confidence={}".format(n_err, n_missing_conf))


def _fmt(x: float) -> str:
    return "n/a" if x != x else "{:.3f}".format(x)  # x!=x -> NaN


if __name__ == "__main__":
    main()
