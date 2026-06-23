"""Task-type stratified sampling so perception/prediction/planning/behavior are
balanced in the curated set (O2 runs stratified calibration analysis)."""

import random
from collections import defaultdict
from typing import Iterable, List, Optional

from avbench.schema import Sample


def stratified_subset(
    samples: Iterable[Sample],
    per_task: Optional[int] = None,
    total: Optional[int] = None,
    seed: int = 0,
) -> List[Sample]:
    """Cap samples per task_type (and optionally an overall total)."""
    buckets = defaultdict(list)
    for s in samples:
        buckets[s.task_type].append(s)

    rng = random.Random(seed)
    out: List[Sample] = []
    for task, items in buckets.items():
        rng.shuffle(items)
        out.extend(items[:per_task] if per_task else items)

    rng.shuffle(out)
    if total is not None:
        out = out[:total]
    return out
