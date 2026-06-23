#!/usr/bin/env python3
"""Run a VLM over a curated Sample JSONL, writing Prediction JSONL.

Examples:
  # Offline smoke test (no API key):
  python scripts/infer.py --data data/curated/drive_v1.jsonl \
      --backend mock --strategy verbal_confidence --out runs/mock_vc.jsonl

  # Real Gemini (free tier; needs GEMINI_API_KEY):
  python scripts/infer.py --data data/curated/drive_v1.jsonl \
      --backend gemini --model gemini-2.0-flash \
      --strategy verbal_confidence --out runs/gemini_vc.jsonl
"""

import argparse
import asyncio
import sys

import _bootstrap  # noqa: F401

from avbench.inference.client import get_backend
from avbench.inference.runner import run_inference
from avbench.inference.strategies.base import get_strategy
from avbench.io_utils import read_jsonl
from avbench.schema import Sample


def load_samples(path, limit=None, task=None):
    out = []
    for rec in read_jsonl(path):
        s = Sample.model_validate(rec)
        if task and s.task_type.value != task:
            continue
        out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Curated Sample JSONL")
    ap.add_argument("--backend", default="gemini", help="gemini | mock")
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--strategy", default="verbal_confidence")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--task", default=None, help="Filter to one task_type")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    samples = load_samples(args.data, limit=args.limit, task=args.task)
    print("Loaded {} samples".format(len(samples)))

    try:
        client = get_backend(args.backend)(model=args.model)
    except RuntimeError as e:
        print("error: {}".format(e), file=sys.stderr)
        raise SystemExit(1)
    strategy = get_strategy(args.strategy)()

    ok = asyncio.run(
        run_inference(
            samples, client, strategy, args.out,
            concurrency=args.concurrency, resume=not args.no_resume,
        )
    )
    print("Done: {} predictions -> {}".format(ok, args.out))


if __name__ == "__main__":
    main()
