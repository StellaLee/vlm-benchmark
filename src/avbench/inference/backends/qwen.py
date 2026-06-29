"""Qwen-VL backend (Alibaba DashScope, OpenAI-compatible mode).

DashScope exposes an OpenAI-compatible endpoint (chat/completions with an
image_url content part), so we call it directly over httpx, mirroring the GLM
backend. API key is read from QWEN_API_KEY (or DASHSCOPE_API_KEY); the base URL
defaults to the mainland DashScope platform and is overridable via QWEN_BASE_URL
(e.g. https://dashscope-intl.aliyuncs.com/compatible-mode/v1 for the
international platform).

Images are sent inline as base64 data URLs, one labelled part per camera, mirroring
the GLM/Gemini backends. Logprobs are not requested, so avg_logprob stays None and
the token-probability strategy records a capability gap instead of failing.
"""

import asyncio
import base64
import os
from typing import List, Optional

from avbench.inference.client import GenResult, VLMClient, register_backend
from avbench.schema import ImageRef

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _mime_for(path: str) -> str:
    return _MIME.get(os.path.splitext(path)[1].lower(), "image/jpeg")


@register_backend("qwen")
class QwenClient(VLMClient):
    def __init__(
        self,
        model: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 4,
        timeout: float = 120.0,
    ):
        import httpx  # imported lazily so curation has no hard dep

        self.model = model
        key = api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key. Set QWEN_API_KEY (https://dashscope.console.aliyun.com/apiKey)."
            )
        self._key = key
        self._base_url = (base_url or os.environ.get("QWEN_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self.max_retries = max_retries
        self._httpx = httpx
        self._timeout = timeout

    def _build_messages(self, prompt: str, images: List[ImageRef]):
        content = []
        for img in images:
            try:
                with open(img.path, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            label = img.camera or os.path.basename(img.path)
            b64 = base64.b64encode(data).decode("ascii")
            url = "data:{};base64,{}".format(_mime_for(img.path), b64)
            content.append({"type": "text", "text": "[{}]".format(label)})
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    async def _one_call(self, messages, temperature: float) -> GenResult:
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": "Bearer {}".format(self._key)}
        url = "{}/chat/completions".format(self._base_url)

        last_err = None
        async with self._httpx.AsyncClient(timeout=self._timeout) as http:
            for attempt in range(self.max_retries):
                try:
                    resp = await http.post(url, json=payload, headers=headers)
                    if resp.status_code >= 400:
                        # Surface DashScope's error body (it explains the 400 reason).
                        last_err = "HTTP {}: {}".format(resp.status_code, resp.text[:500])
                        if resp.status_code in (429, 500, 502, 503):
                            await asyncio.sleep(min(2 ** attempt, 30))
                            continue
                        break  # 4xx other than 429 won't fix on retry
                    return self._to_result(resp.json())
                except Exception as e:  # noqa: BLE001 — surface after retries
                    last_err = e
                    msg = str(e).lower()
                    if "timeout" in msg or "connect" in msg:
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    break
        raise RuntimeError("Qwen call failed: {}".format(last_err))

    def _to_result(self, data: dict) -> GenResult:
        text = ""
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            pass
        usage = {"backend": "qwen", "model": self.model}
        um = data.get("usage") or {}
        usage.update(
            prompt_tokens=um.get("prompt_tokens"),
            output_tokens=um.get("completion_tokens"),
            total_tokens=um.get("total_tokens"),
        )
        return GenResult(text=text, avg_logprob=None, usage=usage)

    async def generate(
        self,
        prompt: str,
        images: List[ImageRef],
        n: int = 1,
        temperature: float = 0.0,
        logprobs: bool = False,
    ) -> List[GenResult]:
        messages = self._build_messages(prompt, images)
        if n == 1:
            return [await self._one_call(messages, temperature)]
        results = await asyncio.gather(
            *[self._one_call(messages, temperature) for _ in range(n)]
        )
        return list(results)
