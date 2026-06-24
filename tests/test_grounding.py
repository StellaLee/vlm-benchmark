"""Marker grounding changes what the model sees, so a bug here would silently
shift the headline numbers per condition. These pin the coordinate parse
(markers must land at the right pixel), that marking is scoped to the referenced
camera + current frame only, and that the condition is recorded the way
evaluate.py stratifies on."""

from PIL import Image

from avbench.inference.grounding import parse_refs, render_markers
from avbench.inference.strategies.direct import DirectAnswer
from avbench.schema import ImageRef, PromptFormat, Sample, TaskType


def _img(path, w=64, h=48, color=(10, 20, 30)):
    Image.new("RGB", (w, h), color).save(path, "JPEG")
    return str(path)


def _sample(tmp_path, object_refs):
    imgs = [ImageRef(path=_img(tmp_path / "back.jpg"), camera="CAM_BACK"),
            ImageRef(path=_img(tmp_path / "front.jpg"), camera="CAM_FRONT")]
    return Sample(sample_id="d/x", dataset="d", task_type=TaskType.PERCEPTION,
                  prompt_format=PromptFormat.QA, question="q", answer="a",
                  images=imgs, object_refs=object_refs)


def test_parse_refs():
    assert parse_refs("<c1,CAM_BACK,0.8781,0.6120>") == [("CAM_BACK", 0.8781, 0.612)]
    assert parse_refs("no refs here") == []


def test_marker_only_on_referenced_camera(tmp_path):
    s = _sample(tmp_path, ["<c1,CAM_BACK,0.5,0.5>"])
    out = render_markers(s, cache_dir=str(tmp_path / "cache"))
    by_cam = {im.camera: im for im in out}
    assert "cache" in by_cam["CAM_BACK"].path      # rewritten -> marked
    assert by_cam["CAM_FRONT"].path == s.images[1].path  # untouched


def test_no_refs_returns_images_unchanged(tmp_path):
    s = _sample(tmp_path, [])
    assert render_markers(s, cache_dir=str(tmp_path / "cache")) is s.images


def test_condition_records_marker_flag():
    strat = DirectAnswer()
    assert strat.condition() == {"marker_grounding": False}
    strat.marker_grounding = True
    assert strat.condition() == {"marker_grounding": True}
