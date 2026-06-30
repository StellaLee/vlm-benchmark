"""Unified data contract.

Curation produces `Sample` (the x + ground-truth y); inference consumes `Sample`
and produces `Prediction` (the y_hat + uncertainty signals). Everything else in
the system depends only on these two models, so adding a dataset or a prompting
strategy never ripples downstream.

Typed with `typing` (not PEP 604 `X | None`) for Python 3.9 compatibility.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    PERCEPTION = "perception"
    PREDICTION = "prediction"
    PLANNING = "planning"
    BEHAVIOR = "behavior"  # DriveBench has a behavior split
    UNKNOWN = "unknown"


class PromptFormat(str, Enum):
    MCQ = "mcq"  # multiple choice — clean binary correctness
    YESNO = "yesno"  # polar Yes/No (e.g. DriveLM "Is <c> a sign or a barrier?")
    QA = "qa"  # open-ended short answer
    CAP = "cap"  # caption / description


class ImageRef(BaseModel):
    path: str  # resolved local path to the image
    camera: Optional[str] = None  # CAM_FRONT, CAM_BACK, ...
    frame_idx: int = 0  # for multi-frame / temporal prompts


class Sample(BaseModel):
    """The model input (x) plus ground-truth answer (y)."""

    sample_id: str  # globally unique, e.g. "drivebench/<native_id>"
    dataset: str
    task_type: TaskType
    prompt_format: PromptFormat
    question: str
    images: List[ImageRef] = Field(default_factory=list)
    answer: str  # ground truth y
    options: Optional[List[str]] = None  # MCQ choices (letters resolved in text)
    object_refs: List[str] = Field(default_factory=list)  # <c1,CAM_BACK,...>
    split: Optional[str] = None  # e.g. "clean" vs a corruption setting
    extras: Dict[str, Any] = Field(default_factory=dict)  # scene_token, map_path


class Prediction(BaseModel):
    """The model output (y_hat) plus uncertainty signals for calibration."""

    sample_id: str
    model: str
    strategy: str
    raw_text: str
    answer: Optional[str] = None  # parsed y_hat
    verbal_confidence: Optional[float] = None  # 0..1 from the prompt
    token_logprob: Optional[float] = None  # seq/answer logprob if backend exposes
    samples: List[str] = Field(default_factory=list)  # consistency runs
    abstained: bool = False
    usage: Dict[str, Any] = Field(default_factory=dict)  # tokens, latency, cost
    condition: Dict[str, Any] = Field(default_factory=dict)  # ablation flags active
    error: Optional[str] = None  # populated if the call failed
