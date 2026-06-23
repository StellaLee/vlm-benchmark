#!/usr/bin/env python3
"""Validate a correctness scorer without human labels, using the synthetic-control
harness (manufactured correct/incorrect pairs).

  python scripts/validate_scorer.py --curated data/curated/drivebench_real.jsonl \
      --scorer structured --threshold 0.5
"""

import argparse

import _bootstrap  # noqa: F401

from avbench.eval.scorer import get_scorer
from avbench.eval.synthetic import evaluate_scorer, make_controls
from avbench.io_utils import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", required=True)
    ap.add_argument("--scorer", default="structured", help="exact | structured")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = list(read_jsonl(args.curated))
    kw = {"threshold": args.threshold} if args.scorer == "structured" else {}
    scorer = get_scorer(args.scorer, **kw)
    controls = make_controls(samples, seed=args.seed)
    rep = evaluate_scorer(scorer, controls)

    print("scorer = {}  (threshold={})".format(args.scorer, args.threshold))
    print("controls: {}  from {} curated answers  {}".format(
        rep["n_controls"], len(samples), rep["by_kind_n"]))
    print("-" * 52)
    print("  accuracy            {:.3f}".format(rep["accuracy"]))
    print("  precision           {:.3f}   (judged-correct that truly are)".format(rep["precision"]))
    print("  recall              {:.3f}   (true-correct it accepts)".format(rep["recall"]))
    print("-" * 52)
    print("  paraphrase flip     {:.3f}   (lower=better: rejects valid paraphrases)".format(rep["paraphrase_flip_rate"]))
    print("  verbose flip        {:.3f}   (lower=better: padding breaks a correct)".format(rep["verbose_flip_rate"]))
    print("  corrupt catch       {:.3f}   (higher=better: rejects corruptions)".format(rep["corrupt_catch_rate"]))
    print("  corrupt+pad catch   {:.3f}   (vs above: verbosity-bias probe)".format(rep["corrupt_verbose_catch_rate"]))
    print("  mismatch catch      {:.3f}   (higher=better: rejects other answers)".format(rep["mismatch_catch_rate"]))


if __name__ == "__main__":
    main()
