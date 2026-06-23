"""Gemini backend (Google AI Studio free tier).

Uses the `google-genai` SDK async client. API key is read from GEMINI_API_KEY or
GOOGLE_API_KEY. Free-tier models (e.g. gemini-2.0-flash, gemini-2.5-flash) are
rate-limited, so we add bounded retry/backoff on errors and loop for n>1 rather
than relying on candidate_count (the free tier restricts it).

Logprobs: most flash models on the free tier do not return token logprobs; when
unavailable we leave `avg_logprob=None` and the token-probability strategy
records a capability gap instead of failing.
"""

import asyncio
import os
import time
from typing import List, Optional

from avbench.inference.client import GenResult, VLMClient, register_backend
from avbench.schema import ImageRef

_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _mime_for(path: str) -> str:
    return _MIME.get(os.path.splitext(path)[1].lower(), "image/jpeg")


@register_backend("gemini")
class GeminiClient(VLMClient):
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_retries: int = 4,
        request_logprobs: bool = False,
    ):
        from google import genai  # imported lazily so curation has no hard dep

        self.model = model
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key. Set GEMINI_API_KEY (https://aistudio.google.com/apikey)."
            )
        self._client = genai.Client(api_key=key)
        self._genai = genai
        self.max_retries = max_retries
        self.request_logprobs = request_logprobs

    def _build_contents(self, prompt: str, images: List[ImageRef]):
        from google.genai import types

        parts = []
        for img in images:
            try:
                with open(img.path, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            label = img.camera or os.path.basename(img.path)
            parts.append(types.Part.from_text(text="[{}]".format(label)))
            parts.append(types.Part.from_bytes(data=data, mime_type=_mime_for(img.path)))
        parts.append(types.Part.from_text(text=prompt))
        return parts

    async def _one_call(self, contents, temperature: float, logprobs: bool) -> GenResult:
        from google.genai import types

        cfg_kwargs = {"temperature": temperature, "candidate_count": 1}
        if logprobs:
            # Best-effort; ignored/erroring models fall back below.
            cfg_kwargs["response_logprobs"] = True
        config = types.GenerateContentConfig(**cfg_kwargs)

        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = await self._client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                return self._to_result(resp, time.time() - t0)
            except Exception as e:  # noqa: BLE001 — surface after retries
                last_err = e
                msg = str(e).lower()
                # Retry on rate limit / transient; drop logprobs if unsupported.
                if logprobs and ("logprob" in msg or "response_logprobs" in msg):
                    config = types.GenerateContentConfig(
                        temperature=temperature, candidate_count=1
                    )
                    logprobs = False
                    continue
                if "429" in msg or "rate" in msg or "503" in msg or "unavailable" in msg:
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                break
        raise RuntimeError("Gemini call failed: {}".format(last_err))

    def _to_result(self, resp, latency: float) -> GenResult:
        text = getattr(resp, "text", None) or ""
        avg_lp = None
        try:
            cand = resp.candidates[0]
            avg_lp = getattr(cand, "avg_logprobs", None)
        except (AttributeError, IndexError):
            pass
        usage = {"latency_s": round(latency, 3), "backend": "gemini", "model": self.model}
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage.update(
                prompt_tokens=getattr(um, "prompt_token_count", None),
                output_tokens=getattr(um, "candidates_token_count", None),
                total_tokens=getattr(um, "total_token_count", None),
            )
        return GenResult(text=text, avg_logprob=avg_lp, usage=usage)

    async def generate(
        self,
        prompt: str,
        images: List[ImageRef],
        n: int = 1,
        temperature: float = 0.0,
        logprobs: bool = False,
    ) -> List[GenResult]:
        contents = self._build_contents(prompt, images)
        want_lp = logprobs or self.request_logprobs
        if n == 1:
            return [await self._one_call(contents, temperature, want_lp)]
        # Free tier: loop instead of candidate_count>1.
        results = await asyncio.gather(
            *[self._one_call(contents, temperature, want_lp) for _ in range(n)]
        )
        return list(results)
