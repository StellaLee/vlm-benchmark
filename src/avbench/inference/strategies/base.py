"""Prompting-strategy contract + registry.

Each strategy encapsulates one O2-KR1 confidence-elicitation method. It builds a
prompt from a `Sample`, calls the `VLMClient`, and returns a `Prediction` with the
parsed answer and whatever confidence signal it elicited.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

from avbench.inference.client import VLMClient
from avbench.schema import ImageRef, PromptFormat, Prediction, Sample


class PromptStrategy(ABC):
    name: str
    # Run-level option, set by infer.py per run. (Abstention is a prompt
    # formulation, so it's a strategy, not a flag; markers modify the image.)
    marker_grounding: bool = False

    @abstractmethod
    def build_prompt(self, sample: Sample) -> str:
        ...

    @abstractmethod
    async def run(self, sample: Sample, client: VLMClient) -> Prediction:
        ...

    # --- shared plumbing for the run-level options -------------------------
    def images_for(self, sample: Sample) -> List[ImageRef]:
        """sample.images, optionally with a marker drawn at each object_ref's
        (camera, x, y) when --marker-grounding is set."""
        if self.marker_grounding and sample.object_refs:
            from avbench.inference.grounding import render_markers

            return render_markers(sample)
        return sample.images

    def condition(self) -> Dict[str, Any]:
        """The active ablation condition, recorded on the Prediction so
        evaluate.py can stratify metrics by it."""
        return {"marker_grounding": bool(self.marker_grounding)}


def render_question(sample: Sample) -> str:
    """Shared prompt body: question text plus MCQ options when present."""
    lines = [sample.question]
    if sample.prompt_format == PromptFormat.MCQ and sample.options:
        lines.append("")
        for i, opt in enumerate(sample.options):
            letter = chr(ord("A") + i)
            # Avoid double-lettering if options already start with "A."
            opt_s = str(opt)
            if opt_s[:2].upper() in ("{}.".format(letter), "{} ".format(letter)):
                lines.append(opt_s)
            else:
                lines.append("{}. {}".format(letter, opt_s))
    return "\n".join(lines)


_STRATEGIES: Dict[str, Type[PromptStrategy]] = {}


def register(name: str) -> Callable[[Type[PromptStrategy]], Type[PromptStrategy]]:
    def deco(cls: Type[PromptStrategy]) -> Type[PromptStrategy]:
        cls.name = name
        _STRATEGIES[name] = cls
        return cls

    return deco


def get_strategy(name: str) -> Type[PromptStrategy]:
    from avbench.inference import strategies  # noqa: F401  ensure registration

    if name not in _STRATEGIES:
        raise KeyError(f"Unknown strategy '{name}'. Registered: {sorted(_STRATEGIES)}")
    return _STRATEGIES[name]
