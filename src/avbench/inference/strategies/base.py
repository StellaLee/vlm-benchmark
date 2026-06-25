"""Prompting-strategy contract + registry.

Each strategy encapsulates one O2-KR1 confidence-elicitation method. It builds a
prompt from a `Sample`, calls the `VLMClient`, and returns a `Prediction` with the
parsed answer and whatever confidence signal it elicited.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

from avbench.inference.client import VLMClient
from avbench.schema import ImageRef, PromptFormat, Prediction, Sample

# DriveLM identification questions ("Is <c1,CAM_BACK,0.54,0.48> a traffic sign or
# a road barrier?") are polar yes/no despite the "a A or a B?" surface form — the
# gold is Yes/No, not one of the two nouns. The phrasing misleads models into
# answering with a noun, so we detect them and tell the model the answer space.
# Match the structural form (Is <obj> a/an ... or a/an ...?), not a loose " or ".
_YESNO_Q = re.compile(r"^\s*Is\s+<c\d+[^>]*>\s+an?\b.*\bor\s+an?\b.*\?\s*$", re.IGNORECASE)


class PromptStrategy(ABC):
    name: str
    # Run-level options, set by infer.py per run. (Abstention is a prompt
    # formulation, so it's a strategy, not a flag; markers modify the image.)
    marker_grounding: bool = False
    # Send only the referenced camera when the question names exactly one. Other
    # cameras are distractors for single-camera identification questions, and
    # dropping them removes the "which of the 6 cameras" half of grounding.
    single_camera: bool = False

    @abstractmethod
    def build_prompt(self, sample: Sample) -> str:
        ...

    @abstractmethod
    async def run(self, sample: Sample, client: VLMClient) -> Prediction:
        ...

    # --- shared plumbing for the run-level options -------------------------
    def images_for(self, sample: Sample) -> List[ImageRef]:
        """sample.images, modified by the active run-level options:
        - --marker-grounding: draw a marker at each object_ref's (camera, x, y).
        - --single-camera: keep only the referenced camera when the question
          names exactly one (otherwise pass through unchanged — multi/no-ref
          questions still need the full surround view)."""
        images = sample.images
        if self.marker_grounding and sample.object_refs:
            from avbench.inference.grounding import render_markers

            images = render_markers(sample)
        if self.single_camera:
            from avbench.inference.grounding import referenced_cameras

            cams = referenced_cameras(sample)
            if len(cams) == 1:
                only = [im for im in images if im.camera == cams[0]]
                if only:
                    images = only
        return images

    def condition(self) -> Dict[str, Any]:
        """The active ablation condition, recorded on the Prediction so
        evaluate.py can stratify metrics by it."""
        return {
            "marker_grounding": bool(self.marker_grounding),
            "single_camera": bool(self.single_camera),
        }


def is_yes_no_question(question: str) -> bool:
    return bool(_YESNO_Q.match(question or ""))


def answer_instruction(sample: Sample) -> str:
    """How the model should shape its answer, given the question type."""
    if sample.options:
        return "Choose the single best option and give its letter."
    if is_yes_no_question(sample.question):
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
