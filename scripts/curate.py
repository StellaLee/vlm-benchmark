#!/usr/bin/env python3
"""Curate one or more benchmarks into a unified Sample JSONL.

Example:
  python scripts/curate.py --dataset drivebench \
      --qa-file data/sample/drivebench_sample.json --split clean --formats mcq \
      --per-task 50 --out data/curated/drive_v1.jsonl
"""

import argparse

import _bootstrap  # noqa: F401

from avbench.curation.adapters import drivebench  # noqa: F401  register
from avbench.curation.base import get_adapter
from avbench.curation.build import build, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="drivebench")
    ap.add_argument("--qa-file", required=True, help="Native benchmark QA JSON")
    ap.add_argument("--data-root", default="", help="Root for resolving image paths")
    ap.add_argument("--split", default="clean", help="'clean', a corruption name, or 'all'")
    ap.add_argument("--formats", nargs="*", default=None,
                    help="Filter to mcq/qa/cap; default keeps all (arena is qa)")
    ap.add_argument("--per-task", type=int, default=None, help="Cap samples per task_type")
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--no-require-images", action="store_true",
                    help="Keep samples even if image files are missing (smoke tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    adapter_cls = get_adapter(args.dataset)
    adapter = adapter_cls(
        qa_file=args.qa_file,
        data_root=args.data_root,
        split=None if args.split == "all" else args.split,
        formats=args.formats or None,
        require_images=not args.no_require_images,
    )
    samples = build(
        [adapter], args.out, per_task=args.per_task, total=args.total
    )
    print(summarize(samples))
    print("Wrote -> {}".format(args.out))


if __name__ == "__main__":
    main()
