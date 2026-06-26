"""Run-level presentation: how a sample's images + prompt are shown to the model.

Two independent axes, kept separate on purpose:

  - layout (Layout): how the surround cameras are *packaged* — a single mutually
    exclusive choice. SEPARATE (today's default: 6 images in sequence), STITCH
    (one composite grid image), SINGLE (only the referenced camera).
  - annotations: pixel overlays that compose with any layout — currently just
    marker_grounding; a future camera-label overlay would slot in the same way.

A future feature like "label the camera in the image *and* point to it in the
prompt text" spans both image and prompt, so the View owns the image pipeline
(`images_for`) *and* the prompt hook (`decorate_prompt`) — adding such a knob is a
change to this one file, not to every strategy. `condition()` records the active
choice so evaluate.py can stratify with `--by layout`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from avbench.inference.grounding import (
    referenced_cameras,
    render_markers,
    stitch_surround,
    surround_caption,
)
from avbench.schema import ImageRef, Sample


class Layout(str, Enum):
    SEPARATE = "separate"  # 6 cameras as separate images in sequence (default)
    STITCH = "stitch"      # 6 cameras composited into one grid image
    SINGLE = "single"      # only the referenced camera, when the question names one


@dataclass
class View:
    layout: Layout = Layout.SEPARATE
    marker_grounding: bool = False  # orthogonal image annotation, applied before layout
    # Where rendered (marked/stitched) images are written. None = the default
    # persistent cache; infer.py's --no-image-cache points this at an ephemeral
    # temp dir (cleaned at exit) so bulk runs leave nothing behind.
    cache_dir: Optional[str] = None

    def images_for(self, sample: Sample) -> List[ImageRef]:
        images = sample.images
        if self.marker_grounding and sample.object_refs:
            images = render_markers(sample, cache_dir=self.cache_dir)
        if self.layout is Layout.SINGLE:
            cams = referenced_cameras(sample)
            if len(cams) == 1:
                only = [im for im in images if im.camera == cams[0]]
                if only:
                    images = only
        elif self.layout is Layout.STITCH:
            images = [stitch_surround(images, sample.sample_id, cache_dir=self.cache_dir)]
        return images

    def decorate_prompt(self, sample: Sample, prompt: str) -> str:
        """Prompt-side counterpart of the layout. The stitched view collapses 6
        images into one, so the model needs to be told it's a camera grid; other
        layouts pass the prompt through unchanged. This is the extension point for
        future image-pointer text."""
        if self.layout is Layout.STITCH:
            return surround_caption(sample.images) + "\n\n" + prompt
        return prompt

    def condition(self) -> dict:
        return {"layout": self.layout.value, "marker_grounding": bool(self.marker_grounding)}
