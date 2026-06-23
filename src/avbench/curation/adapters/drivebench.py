"""DriveBench adapter — the first/prototype dataset.

DriveBench is built on DriveLM-nuScenes and ships three prompt formats
(MCQ / QA / CAP) plus a clean split and several corruption settings. We lead with
the **clean split, MCQ format** because MCQ yields a binary `is y_hat correct?`
label for free, which de-risks the ECE/AUROC calibration pipeline before we take
on noisy open-ended scoring (BLEU / GPT-score).

The released JSON has drifted across versions, so this adapter is intentionally
tolerant: it reads a list (or id-keyed dict) of records and maps common field
names. Assumed per-record fields (all best-effort)::

    {
      "id" | "question_id" | "token": str,
      "question": str,
      "answer" | "gt_answer": str,
      "task" | "category": "perception|prediction|planning|behavior",
      "format" | "type": "mcq|qa|cap",      # else inferred from `options`
      "options" | "choices": [str, ...],     # MCQ only
      "images": {"CAM_FRONT": "rel/path.jpg", ...} | [paths] | "single.jpg",
      "corruption" | "setting" | "split": "clean" | "<corruption-name>"
    }
"""

import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional

from avbench.curation.base import DatasetAdapter, register
from avbench.curation.sensor import order_images, resolve_image
from avbench.schema import ImageRef, PromptFormat, Sample, TaskType

_OBJECT_REF = re.compile(r"<c\d+,\s*CAM_[A-Z_]+[^>]*>")

_TASK_MAP = {
    "perception": TaskType.PERCEPTION,
    "prediction": TaskType.PREDICTION,
    "planning": TaskType.PLANNING,
    "behavior": TaskType.BEHAVIOR,
}

_FORMAT_MAP = {
    "mcq": PromptFormat.MCQ,
    "multiple_choice": PromptFormat.MCQ,
    "qa": PromptFormat.QA,
    "vqa": PromptFormat.QA,
    "open": PromptFormat.QA,
    "cap": PromptFormat.CAP,
    "caption": PromptFormat.CAP,
}


def _first(rec: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return default


def _coerce_images(raw, data_root: str, file_index: Optional[Dict[str, str]] = None) -> List[ImageRef]:
    imgs: List[ImageRef] = []
    if raw is None:
        return imgs
    if isinstance(raw, str):
        imgs.append(_resolve(raw, None, 0, data_root, file_index))
    elif isinstance(raw, dict):  # {camera: path} — the arena schema
        for cam, path in raw.items():
            if path:
                imgs.append(_resolve(path, cam, 0, data_root, file_index))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                imgs.append(
                    _resolve(
                        item.get("path") or item.get("image"),
                        item.get("camera"),
                        int(item.get("frame_idx", 0)),
                        data_root,
                        file_index,
                    )
                )
            elif item:
                imgs.append(_resolve(item, None, 0, data_root, file_index))
    return order_images([im for im in imgs if im is not None])


def _resolve(path, camera, frame_idx, data_root, file_index) -> Optional[ImageRef]:
    """Resolve a referenced image path, falling back to a basename lookup so the
    curation works regardless of the image archive's internal folder layout
    (e.g. DriveLM's val zip vs DriveBench's documented data/nuscenes/ layout)."""
    if not path:
        return None
    ref = resolve_image(path, data_root, camera=camera, frame_idx=frame_idx)
    if os.path.exists(ref.path) or file_index is None:
        return ref
    hit = file_index.get(os.path.basename(path))
    if hit:
        return ImageRef(path=hit, camera=camera, frame_idx=frame_idx)
    return ref  # leave unresolved; require_images filter drops it later


@register("drivebench")
class DriveBenchAdapter(DatasetAdapter):
    def __init__(
        self,
        qa_file: str,
        data_root: str = "",
        split: Optional[str] = "clean",
        formats: Optional[List[str]] = None,
        require_images: bool = True,
    ):
        self.qa_file = qa_file
        self.data_root = data_root or os.path.dirname(qa_file)
        self.split = split  # None = keep all settings
        self.formats = set(formats) if formats else None  # None = all formats
        self.require_images = require_images
        self._file_index = self._index_images(self.data_root)

    @staticmethod
    def _index_images(root: str) -> Dict[str, str]:
        """Map image basename -> absolute path, for layout-agnostic resolution."""
        index: Dict[str, str] = {}
        if not root or not os.path.isdir(root):
            return index
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    index.setdefault(fn, os.path.join(dirpath, fn))
        return index

    def _load_records(self) -> List[Dict[str, Any]]:
        with open(self.qa_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # id-keyed dict -> inject the key as id
            out = []
            for k, v in data.items():
                v = dict(v)
                v.setdefault("id", k)
                out.append(v)
            return out
        return data

    def iter_samples(self) -> Iterator[Sample]:
        for i, rec in enumerate(self._load_records()):
            sample = self._to_sample(rec, i)
            if sample is None:
                continue
            yield sample

    def _to_sample(self, rec: Dict[str, Any], idx: int) -> Optional[Sample]:
        setting = str(_first(rec, "corruption", "setting", "split", default="clean"))
        if self.split is not None and setting != self.split:
            return None

        fmt_raw = str(_first(rec, "format", "type", default="")).lower()
        options = _first(rec, "options", "choices")
        fmt = _FORMAT_MAP.get(fmt_raw)
        if fmt is None:  # infer
            fmt = PromptFormat.MCQ if options else PromptFormat.QA
        if self.formats is not None and fmt.value not in self.formats:
            return None

        question = _first(rec, "question", "prompt", default="")
        answer = _first(rec, "answer", "gt_answer", "final_answer", default="")
        if not question or answer in (None, ""):
            return None

        task_raw = str(_first(rec, "question_type", "task", "category", "task_type", default="")).lower()
        task = _TASK_MAP.get(task_raw, TaskType.UNKNOWN)

        images = _coerce_images(
            _first(rec, "image_path", "images", "image", "img"),
            self.data_root,
            self._file_index,
        )
        if self.require_images:
            images = [im for im in images if os.path.exists(im.path)]
            if not images:
                return None

        # arena has no per-question id; key on frame_token+index for traceability.
        frame = _first(rec, "frame_token", "sample_token")
        native_id = "{}_{}".format(frame, idx) if frame else str(
            _first(rec, "id", "question_id", "token", default=idx)
        )
        return Sample(
            sample_id="drivebench/{}".format(native_id),
            dataset="drivebench",
            task_type=task,
            prompt_format=fmt,
            question=question,
            images=images,
            answer=str(answer),
            options=[str(o) for o in options] if options else None,
            object_refs=_OBJECT_REF.findall(question),
            split=setting,
            extras={
                k: rec[k]
                for k in ("scene_token", "frame_token", "sample_token")
                if k in rec
            },
        )
