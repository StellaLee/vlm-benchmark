#!/usr/bin/env python3
"""Preview the exact prompt — image *and* text — a strategy sends for one sample.

Builds the prompt the chosen --strategy would send under the given --layout /
--marker-grounding, prints the text, and renders the images into a folder so you
can open them. Handy for eyeballing a specific example without running inference or
keeping a bulk image cache around.

  python scripts/preview_prompt.py --data data/curated/yes_subset.jsonl \
      --strategy verbal_confidence --layout stitch --marker-grounding
  python scripts/preview_prompt.py --data data/curated/yes_subset.jsonl \
      --sample-id drivebench/672026f4...127 --strategy self_reflection --layout stitch
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
                    help="strategy whose prompt to preview (e.g. verbal_confidence, "
                         "self_reflection, consistency)")
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
    print("gold:       {}".format(sample.answer))
    print("strategy={}  layout={}  marker_grounding={}".format(
        args.strategy, args.layout, args.marker_grounding))
    print("\n--- prompt text ---\n{}".format(prompt))
    print("\n--- prompt images ({}) — open these to eyeball ---".format(len(images)))
    for im in images:
        print("  {:16} {}".format(im.camera or "?", im.path))
    print("\nrendered into: {}".format(os.path.abspath(args.out)))


if __name__ == "__main__":
    main()
