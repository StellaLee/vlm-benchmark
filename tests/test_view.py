"""The View is the run-level presentation knob: how a sample's images + prompt are
shown to the model. Layout (separate / stitch / single) is a single mutually-exclusive
choice; marker_grounding is an orthogonal image annotation that composes with it. These
pin (a) the layout pipeline produces the right image set, (b) markers apply before
layout, (c) the stitched view tells the model what the combined image is, and (d) the
condition is recorded the way evaluate.py stratifies on (--by layout)."""

import pytest
from PIL import Image

from avbench.inference.view import Layout, View
from avbench.schema import ImageRef, PromptFormat, Sample, TaskType


@pytest.fixture(autouse=True)
def _isolate_caches(tmp_path, monkeypatch):
    # Per-test cache dirs: keep rendered images out of the repo and prevent tests
    # reusing a sample_id from reading each other's stale composite.
    monkeypatch.setattr("avbench.inference.grounding.STITCH_CACHE_DIR",
                        str(tmp_path / "stitch_cache"))
    monkeypatch.setattr("avbench.inference.grounding.CACHE_DIR",
                        str(tmp_path / "marker_cache"))


def _img(path, w=64, h=48, color=(10, 20, 30)):
    Image.new("RGB", (w, h), color).save(path, "JPEG")
    return str(path)


def _sample(tmp_path, cams, object_refs=()):
    imgs = [ImageRef(path=_img(tmp_path / (c + ".jpg")), camera=c) for c in cams]
    return Sample(sample_id="d/x", dataset="d", task_type=TaskType.PERCEPTION,
                  prompt_format=PromptFormat.QA, question="q", answer="a",
                  images=imgs, object_refs=list(object_refs))


def test_separate_layout_keeps_all_images_as_separate(tmp_path):
    s = _sample(tmp_path, ["CAM_FRONT", "CAM_BACK"])
    out = View(layout=Layout.SEPARATE).images_for(s)
    assert [im.camera for im in out] == ["CAM_FRONT", "CAM_BACK"]


def test_single_layout_keeps_only_referenced_camera(tmp_path):
    s = _sample(tmp_path, ["CAM_FRONT", "CAM_BACK"], ["<c1,CAM_BACK,0.5,0.5>"])
    out = View(layout=Layout.SINGLE).images_for(s)
    assert [im.camera for im in out] == ["CAM_BACK"]


def test_single_layout_passes_through_when_not_exactly_one_camera(tmp_path):
    s = _sample(tmp_path, ["CAM_FRONT", "CAM_BACK"],
                ["<c1,CAM_FRONT,0.5,0.5>", "<c2,CAM_BACK,0.5,0.5>"])
    out = View(layout=Layout.SINGLE).images_for(s)
    assert {im.camera for im in out} == {"CAM_FRONT", "CAM_BACK"}  # can't reduce


def test_stitch_layout_returns_a_single_combined_image(tmp_path):
    s = _sample(tmp_path, ["CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT"])
    out = View(layout=Layout.STITCH).images_for(s)
    assert len(out) == 1
    assert Image.open(out[0].path).size  # one openable composite


def test_marker_applies_before_stitch(tmp_path):
    # With markers on, the single stitched image must be built from the *marked*
    # frames, not the originals — i.e. the marker pipeline runs first.
    s = _sample(tmp_path, ["CAM_BACK", "CAM_FRONT"], ["<c1,CAM_BACK,0.5,0.5>"])
    out = View(layout=Layout.STITCH, marker_grounding=True).images_for(s)
    assert len(out) == 1
    # the composite path is freshly rendered (not a passthrough of an original)
    assert out[0].path not in {im.path for im in s.images}


def test_stitch_orders_cameras_canonically(tmp_path):
    # Shuffled input; each camera must land in its canonical surround cell so the
    # prompt caption (left-to-right) matches the pixels. Distinct colors per cell.
    cells = [("CAM_FRONT_LEFT", (255, 0, 0)),
             ("CAM_FRONT", (0, 255, 0)),
             ("CAM_FRONT_RIGHT", (0, 0, 255))]
    shuffled = [cells[1], cells[2], cells[0]]
    imgs = [ImageRef(path=_img(tmp_path / (c + ".jpg"), color=col), camera=c)
            for c, col in shuffled]
    s = Sample(sample_id="d/x", dataset="d", task_type=TaskType.PERCEPTION,
               prompt_format=PromptFormat.QA, question="q", answer="a", images=imgs)
    out = View(layout=Layout.STITCH).images_for(s)
    im = Image.open(out[0].path).convert("RGB")
    w, h = im.size
    cell = w // 3
    assert im.getpixel((cell // 2, h // 2))[0] > 200            # cell 0 red (FRONT_LEFT)
    assert im.getpixel((cell + cell // 2, h // 2))[1] > 200     # cell 1 green (FRONT)
    assert im.getpixel((2 * cell + cell // 2, h // 2))[2] > 200  # cell 2 blue (FRONT_RIGHT)


def test_stitch_layout_prompt_describes_the_grid(tmp_path):
    s = _sample(tmp_path, ["CAM_FRONT", "CAM_BACK"])
    out = View(layout=Layout.STITCH).decorate_prompt(s, "What do you see?")
    assert "What do you see?" in out
    assert "grid" in out.lower()
    # Cells are named with the full CAM_ prefix so they match the <c1,CAM_...> tags
    # in the question verbatim (no FRONT vs CAM_FRONT mapping for the model to do).
    assert "CAM_FRONT" in out and "CAM_BACK" in out


def test_separate_layout_prompt_unchanged(tmp_path):
    s = _sample(tmp_path, ["CAM_FRONT", "CAM_BACK"])
    assert View(layout=Layout.SEPARATE).decorate_prompt(s, "Q?") == "Q?"


def test_stitch_cache_distinguishes_marked_from_unmarked_input(tmp_path):
    # The composite depends on whether markers were drawn first, but the cache must
    # not collide them on one file — else `--layout stitch --marker-grounding` would
    # silently reuse the plain stitch (or vice versa).
    s = _sample(tmp_path, ["CAM_BACK", "CAM_FRONT"], ["<c1,CAM_BACK,0.5,0.5>"])
    plain = View(layout=Layout.STITCH).images_for(s)[0].path
    marked = View(layout=Layout.STITCH, marker_grounding=True).images_for(s)[0].path
    assert plain != marked


def test_stitch_feeds_full_resolution(tmp_path):
    # The composite must NOT be downsized — the model sees full per-camera resolution.
    # 3 tiles of 64x48 in one row -> 192-wide composite, no shrink.
    s = _sample(tmp_path, ["CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT"])
    out = View(layout=Layout.STITCH).images_for(s)[0]
    assert Image.open(out.path).size == (192, 48)


def test_cache_dir_keeps_renders_out_of_the_default_cache(tmp_path):
    # `--no-image-cache` points the View at an ephemeral dir, so a bulk run leaves
    # nothing in the persistent cache.
    import os

    import avbench.inference.grounding as g

    s = _sample(tmp_path, ["CAM_BACK", "CAM_FRONT"], ["<c1,CAM_BACK,0.5,0.5>"])
    dest = tmp_path / "ephemeral"
    out = View(layout=Layout.STITCH, marker_grounding=True, cache_dir=str(dest)).images_for(s)
    assert out[0].path.startswith(str(dest))
    assert not os.path.exists(g.STITCH_CACHE_DIR)  # default cache untouched
    assert not os.path.exists(g.CACHE_DIR)


def test_condition_records_layout_and_marker():
    assert View().condition() == {"layout": "separate", "marker_grounding": False}
    assert View(layout=Layout.STITCH, marker_grounding=True).condition() == {
        "layout": "stitch", "marker_grounding": True}
