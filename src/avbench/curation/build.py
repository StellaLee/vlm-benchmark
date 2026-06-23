"""Curation entrypoint: adapters -> normalized JSONL of `Sample`s."""

import os
from collections import Counter
from typing import List, Optional

from avbench.curation.base import DatasetAdapter
from avbench.curation.stratify import stratified_subset
from avbench.io_utils import append_jsonl
from avbench.schema import Sample


def build(
    adapters: List[DatasetAdapter],
    out_path: str,
    per_task: Optional[int] = None,
    total: Optional[int] = None,
    seed: int = 0,
) -> List[Sample]:
    samples: List[Sample] = []
    for adapter in adapters:
        samples.extend(adapter.iter_samples())

    if per_task is not None or total is not None:
        samples = stratified_subset(samples, per_task=per_task, total=total, seed=seed)

    if os.path.exists(out_path):
        os.remove(out_path)  # curation is idempotent; rewrite from scratch
    for s in samples:
        append_jsonl(out_path, s.model_dump_json())

    return samples


def summarize(samples: List[Sample]) -> str:
    by_task = Counter(s.task_type.value for s in samples)
    by_fmt = Counter(s.prompt_format.value for s in samples)
    lines = ["Curated {} samples".format(len(samples)),
             "  by task:   " + dict_str(by_task),
             "  by format: " + dict_str(by_fmt)]
    return "\n".join(lines)


def dict_str(counter: Counter) -> str:
    return ", ".join("{}={}".format(k, v) for k, v in sorted(counter.items()))
