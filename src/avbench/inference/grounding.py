"""Visual marker grounding for object-reference questions.

DriveLM/DriveBench reference objects positionally, e.g. `<c1,CAM_BACK,0.88,0.61>`
= the object at ~(88%, 61%) in the rear camera. The coordinates reach the model
only as text, so it must localize them itself — conflating a localization failure
with a reasoning one. `--marker-grounding` draws a marker at the (x, y) on the
named camera to remove that confound.

Marked images are cached and reused, so repeated runs don't re-render.
"""

import os
import re
from typing import List, Optional, Tuple

from avbench.schema import ImageRef, Sample

CACHE_DIR = "data/cache/markers"

# <c1,CAM_BACK,0.5073,0.5778>  ->  (camera, x, y)
_REF = re.compile(r"<\s*c\d+\s*,\s*(CAM_[A-Z_]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)")


def parse_refs(text: str) -> List[Tuple[str, float, float]]:
    """Extract (camera, x, y) triples from object-reference tags in `text`."""
    out: List[Tuple[str, float, float]] = []
    for cam, sx, sy in _REF.findall(text or ""):
        try:
            out.append((cam, float(sx), float(sy)))
        except ValueError:
            continue
    return out


def _safe(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sample_id)


def _draw(src_path: str, points: List[Tuple[float, float]], dst_path: str) -> Optional[str]:
    """Draw crosshair markers at the given normalized points; return dst_path."""
    from PIL import Image, ImageDraw

    try:
        img = Image.open(src_path).convert("RGB")
    except OSError:
        return None
    w, h = img.size
    d = ImageDraw.Draw(img)
    r = max(8, int(0.02 * max(w, h)))  # marker radius scales with image size
    for nx, ny in points:
        cx, cy = int(nx * w), int(ny * h)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=3)
        d.line([cx - r, cy, cx + r, cy], fill=(255, 0, 0), width=2)
        d.line([cx, cy - r, cx, cy + r], fill=(255, 0, 0), width=2)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    img.save(dst_path, quality=90)
    return dst_path


def render_markers(sample: Sample, cache_dir: str = CACHE_DIR) -> List[ImageRef]:
    """Return sample.images with markers drawn on cameras referenced by the
    question's object tags. Non-referenced cameras pass through unchanged."""
    refs = parse_refs(" ".join(sample.object_refs) or sample.question)
    if not refs:
        return sample.images

    by_cam = {}  # camera -> [(x, y), ...]
    for cam, x, y in refs:
        by_cam.setdefault(cam, []).append((x, y))

    out: List[ImageRef] = []
    for im in sample.images:
        pts = by_cam.get(im.camera)
        if not pts:
            out.append(im)
            continue
        dst = os.path.join(cache_dir, "{}__{}.jpg".format(_safe(sample.sample_id), im.camera))
        marked = dst if os.path.exists(dst) else _draw(im.path, pts, dst)
        out.append(ImageRef(path=marked or im.path, camera=im.camera, frame_idx=im.frame_idx))
    return out
