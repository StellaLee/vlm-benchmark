"""Sensor-data linking.

For nuScenes-derived benchmarks (DriveBench, DriveLM, nuScenes-QA) the QA file
references camera images by relative path or token. This module resolves those to
absolute local paths against a configured data root. The real nuScenes token ->
file lookup (via the `nuscenes-devkit`) plugs in here later; for the prototype we
resolve relative paths and verify existence.
"""

import os
from typing import List

from avbench.schema import ImageRef

# nuScenes camera channels, front-to-back, used to order multi-view inputs.
CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def resolve_image(rel_path: str, data_root: str, camera: str = None, frame_idx: int = 0) -> ImageRef:
    path = rel_path if os.path.isabs(rel_path) else os.path.join(data_root, rel_path)
    return ImageRef(path=os.path.normpath(path), camera=camera, frame_idx=frame_idx)


def order_images(images: List[ImageRef]) -> List[ImageRef]:
    """Stable front-to-back ordering so prompts present views consistently."""
    def key(img: ImageRef):
        cam = img.camera or ""
        rank = CAMERA_ORDER.index(cam) if cam in CAMERA_ORDER else len(CAMERA_ORDER)
        return (img.frame_idx, rank)

    return sorted(images, key=key)
