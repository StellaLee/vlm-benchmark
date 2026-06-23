"""Small JSONL helpers shared by curation and inference."""

import json
import os
from typing import Dict, Iterator, Set


def append_jsonl(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def read_jsonl(path: str) -> Iterator[Dict]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_completed_ids(path: str, key: str = "sample_id") -> Set[str]:
    """Read already-written records so a crashed run can resume."""
    done: Set[str] = set()
    for rec in read_jsonl(path):
        if key in rec:
            done.add(rec[key])
    return done
