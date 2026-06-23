"""Dataset adapter contract + a tiny registry.

An adapter's only job: parse one benchmark's native format, resolve image/sensor
paths, tag task_type / prompt_format, and yield unified `Sample`s. Adding a
dataset = one new adapter, nothing downstream changes.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, Type

from avbench.schema import Sample


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def iter_samples(self) -> Iterator[Sample]:
        ...


_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}


def register(name: str) -> Callable[[Type[DatasetAdapter]], Type[DatasetAdapter]]:
    def deco(cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_adapter(name: str) -> Type[DatasetAdapter]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
