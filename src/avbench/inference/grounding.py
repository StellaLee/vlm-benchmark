"""Visual marker grounding for object-reference questions.

DriveLM/DriveBench reference objects positionally, e.g. `<c1,CAM_BACK,0.88,0.61>`
= the object at ~(88%, 61%) in the rear camera. The coordinates reach the model
only as text, so it must localize them itself — conflating a localization failure
with a reasoning one. `--marker-grounding` draws a marker at the (x, y) on the
named camera to remove that confound.

Marked images are cached and reused, so repeated runs don't re-render.
"""

import hashlib
import os
import re
from typing import List, Optional, Tuple

from avbench.schema import ImageRef, Sample

CACHE_DIR = "data/cache/markers"
STITCH_CACHE_DIR = "data/cache/stitch"
MASK_CACHE_DIR = "data/cache/mask"

# Fill color for the masked/blind ablation: a mid-gray canvas carries no scene
# content while staying a valid, same-sized image (isolates the language prior).
_MASK_FILL = (128, 128, 128)

# Canonical nuScenes surround order: front row left-to-right, then back row.
SURROUND_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                  "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

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


def referenced_cameras(sample: Sample) -> List[str]:
    """Distinct cameras named by the sample's object references, first-seen order.

    Uses the same ref source as render_markers so the two stay in lockstep. A
    question that references exactly one camera only needs that camera's image."""
    refs = parse_refs(" ".join(sample.object_refs) or sample.question)
    seen: List[str] = []
    for cam, _x, _y in refs:
        if cam not in seen:
            seen.append(cam)
    return seen


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


def render_markers(sample: Sample, cache_dir: Optional[str] = None) -> List[ImageRef]:
    """Return sample.images with markers drawn on cameras referenced by the
    question's object tags. Non-referenced cameras pass through unchanged."""
    cache_dir = cache_dir or CACHE_DIR
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


def mask_images(images: List[ImageRef], sample_id: str,
                cache_dir: Optional[str] = None) -> List[ImageRef]:
    """Return blank same-sized canvases in place of each image, preserving count and
    per-camera label. Removes all scene content so a run measures the language prior
    only (the blind/masked hallucination probe); keeps the image *slots* so token
    cost and prompt structure stay comparable to the full-image run."""
    from PIL import Image

    cache_dir = cache_dir or MASK_CACHE_DIR
    out: List[ImageRef] = []
    for im in images:
        try:
            with Image.open(im.path) as src:
                size = src.size
        except OSError:
            continue
        # Key on size so different-resolution cameras don't collide on one canvas.
        dst = os.path.join(cache_dir, "mask_{}x{}.jpg".format(*size))
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
            Image.new("RGB", size, _MASK_FILL).save(dst, quality=90)
        out.append(ImageRef(path=dst, camera=im.camera, frame_idx=im.frame_idx))
    return out


def surround_sorted(images: List[ImageRef]) -> List[ImageRef]:
    """Images in canonical surround order (front row, then back row); cameras not in
    SURROUND_ORDER keep their relative position at the end (stable sort)."""
    rank = {c: i for i, c in enumerate(SURROUND_ORDER)}
    return sorted(images, key=lambda im: rank.get(im.camera or "", len(SURROUND_ORDER)))


def surround_caption(images: List[ImageRef]) -> str:
    """One-line description of a stitched composite, so the model knows the single
    image is a camera grid and which view is in which cell. Cells are named with the
    full CAM_ prefix so they match the <c1,CAM_...> object-ref tags in the question
    verbatim (the model can map a referenced camera to its cell without translation)."""
    cams = [im.camera or "?" for im in surround_sorted(images)]
    return ("This single image is a {}-camera surround grid, in reading order "
            "(left to right, top to bottom): {}.".format(len(cams), ", ".join(cams)))


def stitch_surround(images: List[ImageRef], sample_id: str, cols: int = 3,
                    cache_dir: Optional[str] = None) -> ImageRef:
    """Composite the per-camera images into one grid image (canonical surround order),
    cached by sample_id. Returns a single ImageRef pointing at the composite."""
    from PIL import Image

    cache_dir = cache_dir or STITCH_CACHE_DIR
    ordered = surround_sorted(images)
    # Key on the source paths, not just sample_id: the composite depends on the input
    # (e.g. marked vs unmarked frames), so they must not share a cache entry.
    key = hashlib.md5("|".join(im.path for im in ordered).encode()).hexdigest()[:10]
    dst = os.path.join(cache_dir, "{}__{}.jpg".format(_safe(sample_id), key))
    if os.path.exists(dst):
        return ImageRef(path=dst, camera="SURROUND", frame_idx=ordered[0].frame_idx)

    tiles = [Image.open(im.path).convert("RGB") for im in ordered]
    cell_w = max(t.width for t in tiles)
    cell_h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    for i, t in enumerate(tiles):
        if t.size != (cell_w, cell_h):
            t = t.resize((cell_w, cell_h))
        r, c = divmod(i, cols)
        canvas.paste(t, (c * cell_w, r * cell_h))
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    canvas.save(dst, quality=90)
    return ImageRef(path=dst, camera="SURROUND", frame_idx=ordered[0].frame_idx)
