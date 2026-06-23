"""Provider-agnostic VLM client interface + backend registry.

Strategies talk only to `VLMClient.generate`; swapping Gemini for a local vLLM
backend (O2/O3) is a one-line config change. `GenResult.logprobs` is None for
backends that don't expose them (e.g. Gemini free tier) — strategies degrade
gracefully rather than crash.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from avbench.schema import ImageRef


class GenResult(BaseModel):
    text: str
    avg_logprob: Optional[float] = None  # mean token logprob if available
    logprobs: Optional[List[Any]] = None  # per-token, if backend exposes
    usage: Dict[str, Any] = Field(default_factory=dict)


class VLMClient(ABC):
    model: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        images: List[ImageRef],
        n: int = 1,
        temperature: float = 0.0,
        logprobs: bool = False,
    ) -> List[GenResult]:
        ...


_BACKENDS: Dict[str, Type[VLMClient]] = {}


def register_backend(name: str) -> Callable[[Type[VLMClient]], Type[VLMClient]]:
    def deco(cls: Type[VLMClient]) -> Type[VLMClient]:
        _BACKENDS[name] = cls
        return cls

    return deco


def get_backend(name: str) -> Type[VLMClient]:
    # Import for side-effect registration.
    from avbench.inference.backends import gemini, mock  # noqa: F401

    if name not in _BACKENDS:
        raise KeyError(f"Unknown backend '{name}'. Registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]
