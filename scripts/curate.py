#!/usr/bin/env python3
"""Curate one or more benchmarks into a unified Sample JSONL.

Settings come from a YAML config and/or CLI flags (flags override config):
  python scripts/curate.py --config configs/datasets/drivebench.yaml
  python scripts/curate.py --config configs/datasets/drivebench.yaml --per-task 5
  python scripts/curate.py --dataset drivebench --qa-file ... --out ...
"""

import argparse

import _bootstrap  # noqa: F401

from avbench.config import apply_config, load_config
from avbench.curation.adapters import drivebench  # noqa: F401  register
from avbench.curation.base import get_adapter
from avbench.curation.build import build, summarize

# name -> built-in default, used when neither CLI nor config supplies a value.
DEFAULTS = {
    "dataset": "drivebench",
    "qa_file": None,        # required
    "data_root": "",
    "split": "clean",
    "formats": None,        # None = keep all formats
    "per_task": None,
    "total": None,
    "require_images": True,
    "out": None,            # required
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config; CLI flags override it")
    # Config-backed options default to None so we can tell "user passed it" apart
    # from "use config/default".
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--qa-file", dest="qa_file", default=None, help="Native benchmark QA JSON")
    ap.add_argument("--data-root", dest="data_root", default=None, help="Root for resolving image paths")
    ap.add_argument("--split", default=None, help="'clean', a corruption name, or 'all'")
    ap.add_argument("--formats", nargs="*", default=None,
                    help="Filter to mcq/qa/cap; default keeps all (arena is qa)")
    ap.add_argument("--per-task", dest="per_task", type=int, default=None, help="Cap samples per task_type")
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--no-require-images", dest="require_images", action="store_false", default=None,
                    help="Keep samples even if image files are missing (smoke tests)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    apply_config(args, load_config(args.config), DEFAULTS)

    missing = [n for n in ("qa_file", "out") if not getattr(args, n)]
    if missing:
        ap.error("missing required setting(s): {} (pass via --{} or in --config)".format(
            ", ".join(missing), "/--".join(m.replace("_", "-") for m in missing)))

    adapter_cls = get_adapter(args.dataset)
    adapter = adapter_cls(
        qa_file=args.qa_file,
        data_root=args.data_root,
        split=None if args.split == "all" else args.split,
        formats=args.formats or None,
        require_images=args.require_images,
    )
    samples = build([adapter], args.out, per_task=args.per_task, total=args.total)
    print(summarize(samples))
    print("Wrote -> {}".format(args.out))


if __name__ == "__main__":
    main()
