"""Async inference runner: bounded concurrency, resume-on-crash, error capture.

Concurrency is semaphore-bounded so we stay under Gemini free-tier rate limits.
Each completed prediction is appended to JSONL immediately, and a re-run skips
sample_ids already present, so an interrupted job resumes cheaply.
"""

import asyncio
import sys
from typing import List

from avbench.inference.client import VLMClient
from avbench.inference.strategies.base import PromptStrategy
from avbench.io_utils import append_jsonl, load_completed_ids
from avbench.schema import Prediction, Sample


async def run_inference(
    samples: List[Sample],
    client: VLMClient,
    strategy: PromptStrategy,
    out_path: str,
    concurrency: int = 4,
    resume: bool = True,
) -> int:
    done = load_completed_ids(out_path) if resume else set()
    todo = [s for s in samples if s.sample_id not in done]
    if done:
        print("Resuming: {} done, {} to go".format(len(done), len(todo)), file=sys.stderr)

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    counter = {"ok": 0, "err": 0}

    async def worker(sample: Sample) -> None:
        async with sem:
            try:
                pred = await strategy.run(sample, client)
            except Exception as e:  # noqa: BLE001 — record, don't abort the batch
                pred = Prediction(
                    sample_id=sample.sample_id,
                    model=client.model,
                    strategy=strategy.name,
                    raw_text="",
                    error="{}: {}".format(type(e).__name__, e),
                )
            async with lock:
                append_jsonl(out_path, pred.model_dump_json())
                key = "err" if pred.error else "ok"
                counter[key] += 1
                n = counter["ok"] + counter["err"]
                if n % 10 == 0 or n == len(todo):
                    print("  {}/{} (ok={}, err={})".format(n, len(todo), counter["ok"], counter["err"]),
                          file=sys.stderr)

    await asyncio.gather(*[worker(s) for s in todo])
    return counter["ok"]
