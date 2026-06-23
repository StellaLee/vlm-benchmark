"""Offline mock backend.

Returns deterministic pseudo-answers so the full curate -> infer -> evaluate
pipeline runs without an API key or network. Useful for CI and for validating the
plumbing before spending Gemini quota.
"""

import hashlib
import re
from typing import List

from avbench.inference.client import GenResult, VLMClient, register_backend
from avbench.schema import ImageRef


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


@register_backend("mock")
class MockClient(VLMClient):
    def __init__(self, model: str = "mock", **_):
        self.model = model

    async def generate(
        self,
        prompt: str,
        images: List[ImageRef],
        n: int = 1,
        temperature: float = 0.0,
        logprobs: bool = False,
    ) -> List[GenResult]:
        seed = _seed(prompt)
        # If the prompt lists MCQ options "A. ... B. ...", pick one pseudo-randomly.
        letters = re.findall(r"\b([A-E])\.", prompt)
        if letters:
            choice = letters[seed % len(letters)]
            answer = "Answer: {}".format(choice)
        else:
            answer = "Answer: object is moving"
        conf = 50 + (seed % 50)  # 50..99
        text = "{}\nConfidence: {}".format(answer, conf)
        usage = {"backend": "mock", "model": self.model, "latency_s": 0.0}
        # Vary slightly across n samples to exercise consistency logic.
        out = []
        for i in range(n):
            t = text if i == 0 else text.replace("Confidence", "Conf").rstrip() + " "
            out.append(GenResult(text=t, avg_logprob=-0.3 - (seed % 7) / 10.0, usage=usage))
        return out
