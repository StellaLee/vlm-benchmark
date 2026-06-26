#!/usr/bin/env python3
"""Render one sample's View output (marked/stitched images + the decorated prompt)
to a folder so you can eyeball exactly what the model is sent — handy when a specific
example looks wrong, without keeping a bulk image cache around.

  python scripts/preview_view.py --data data/curated/yes_subset.jsonl \
      --layout stitch --marker-grounding --out experiments/outputs/preview
  python scripts/preview_view.py --data data/curated/yes_subset.jsonl \
      --sample-id drivebench/672026f4...127 --layout stitch
"""

import argparse
import os

import _bootstrap  # noqa: F401

from avbench.inference.strategies.base import get_strategy
from avbench.inference.view import Layout, View
from avbench.io_utils import read_jsonl
from avbench.schema import Sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Curated Sample JSONL")
    ap.add_argument("--sample-id", default=None, help="default: first sample in the file")
    ap.add_argument("--layout", choices=[m.value for m in Layout], default="stitch")
    ap.add_argument("--marker-grounding", dest="marker_grounding", action="store_true")
    ap.add_argument("--strategy", default="verbal_confidence",
                    help="strategy whose decorated prompt to show")
    ap.add_argument("--out", default="experiments/outputs/preview",
                    help="folder to render the images into (so you can open them)")
    args = ap.parse_args()

    samples = [Sample.model_validate(r) for r in read_jsonl(args.data)]
    if args.sample_id:
        samples = [s for s in samples if s.sample_id == args.sample_id]
        if not samples:
            raise SystemExit("sample_id not found: {}".format(args.sample_id))
    if not samples:
        raise SystemExit("no samples in {}".format(args.data))
    sample = samples[0]

    os.makedirs(args.out, exist_ok=True)
    view = View(layout=Layout(args.layout), marker_grounding=args.marker_grounding,
                cache_dir=args.out)  # render straight into the inspectable folder
    strategy = get_strategy(args.strategy)()
    strategy.view = view

    images = view.images_for(sample)
    prompt = strategy.prompt_for(sample)

    print("sample_id:  {}".format(sample.sample_id))
    print("question:   {}".format(sample.question))
    print("gold:       {}".format(sample.answer))
    print("layout={}  marker_grounding={}".format(args.layout, args.marker_grounding))
    print("\n--- images sent ({}) ---".format(len(images)))
    for im in images:
        print("  {:16} {}".format(im.camera or "?", im.path))
    print("\n--- prompt sent ---\n{}".format(prompt))
    print("\nrendered into: {}".format(os.path.abspath(args.out)))


if __name__ == "__main__":
    main()
