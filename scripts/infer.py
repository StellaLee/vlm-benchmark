#!/usr/bin/env python3
"""Run a VLM over a curated Sample JSONL, writing Prediction JSONL.

Settings come from a YAML config and/or CLI flags (flags override config):
  # From a run config:
  python scripts/infer.py --config configs/runs/gemini_verbal_conf.yaml
  # Offline smoke test (no API key):
  python scripts/infer.py --data data/curated/drive_v1.jsonl \
      --backend mock --strategy verbal_confidence --out runs/mock_vc.jsonl
  # Real Gemini (needs GEMINI_API_KEY in .env):
  python scripts/infer.py --data data/curated/drive_v1.jsonl \
      --backend gemini --model gemini-2.5-flash-lite --out runs/gemini_vc.jsonl
"""

import argparse
import asyncio
import sys

import _bootstrap  # noqa: F401

from avbench.config import apply_config, load_config
from avbench.inference.client import get_backend
from avbench.inference.runner import run_inference
from avbench.inference.strategies.base import get_strategy
from avbench.io_utils import read_jsonl
from avbench.schema import Sample

DEFAULTS = {
    "data": None,           # required
    "backend": "gemini",
    "model": "gemini-2.5-flash-lite",
    "strategy": "verbal_confidence",
    "out": None,            # required
    "concurrency": 4,
    "limit": None,
    "task": None,
    "resume": True,
}


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
    ap.add_argument("--config", default=None, help="YAML config; CLI flags override it")
    ap.add_argument("--data", default=None, help="Curated Sample JSONL")
    ap.add_argument("--backend", default=None, help="gemini | mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--task", default=None, help="Filter to one task_type")
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=None)
    args = ap.parse_args()

    apply_config(args, load_config(args.config), DEFAULTS)

    missing = [n for n in ("data", "out") if not getattr(args, n)]
    if missing:
        ap.error("missing required setting(s): {} (pass via --{} or in --config)".format(
            ", ".join(missing), "/--".join(missing)))

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
            concurrency=args.concurrency, resume=args.resume,
        )
    )
    print("Done: {} predictions -> {}".format(ok, args.out))


if __name__ == "__main__":
    main()
