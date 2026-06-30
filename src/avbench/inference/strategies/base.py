"""Prompting-strategy contract + registry.

Each strategy encapsulates one O2-KR1 confidence-elicitation method. It builds a
prompt from a `Sample`, calls the `VLMClient`, and returns a `Prediction` with the
parsed answer and whatever confidence signal it elicited.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type

from avbench.inference.client import VLMClient
from avbench.inference.view import View
from avbench.schema import ImageRef, PromptFormat, Prediction, Sample

class PromptStrategy(ABC):
    name: str
    # How the sample's images + prompt are presented (layout / annotations), set by
    # infer.py per run. (Abstention is a prompt formulation, so it's a strategy, not
    # a View knob.) Lazily per-instance so there's no shared mutable default.
    _view: Optional[View] = None

    @property
    def view(self) -> View:
        if self._view is None:
            self._view = View()
        return self._view

    @view.setter
    def view(self, value: View) -> None:
        self._view = value

    @abstractmethod
    def build_prompt(self, sample: Sample) -> str:
        ...

    @abstractmethod
    async def run(self, sample: Sample, client: VLMClient) -> Prediction:
        ...

    # --- shared plumbing: delegate presentation to the active View ----------
    def images_for(self, sample: Sample) -> List[ImageRef]:
        return self.view.images_for(sample)

    def prompt_for(self, sample: Sample) -> str:
        """build_prompt + the View's layout-dependent prompt decoration. Strategies
        feed this (not build_prompt) to the client so e.g. a stitched view tells the
        model it's a camera grid."""
        return self.view.decorate_prompt(sample, self.build_prompt(sample))

    def condition(self) -> Dict[str, object]:
        """The active ablation condition, recorded on the Prediction so evaluate.py
        can stratify metrics by it (e.g. --by layout)."""
        return self.view.condition()


def answer_instruction(sample: Sample) -> str:
    """How the model should shape its answer, given the question type (set by the
    dataset adapter at curation time — inference stays dataset-agnostic)."""
    if sample.options:
        return "Choose the single best option and give its letter."
    if sample.prompt_format == PromptFormat.YESNO:
        return "Answer Yes or No."
    return "Give a concise answer."


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
