"""Marker grounding changes what the model sees, so a bug here would silently
shift the headline numbers per condition. These pin the coordinate parse
(markers must land at the right pixel), that marking is scoped to the referenced
camera + current frame only, and that the condition is recorded the way
evaluate.py stratifies on."""

from PIL import Image

from avbench.inference.grounding import parse_refs, render_markers
from avbench.inference.strategies.base import answer_instruction, is_yes_no_question
from avbench.inference.strategies.direct import DirectAnswer
from avbench.schema import ImageRef, PromptFormat, Sample, TaskType


def _q(question, options=None):
    return Sample(sample_id="d/x", dataset="d", task_type=TaskType.PERCEPTION,
                  prompt_format=PromptFormat.MCQ if options else PromptFormat.QA,
                  question=question, answer="a", images=[], options=options)


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


def test_condition_records_flags():
    strat = DirectAnswer()
    assert strat.condition() == {"marker_grounding": False, "single_camera": False}
    strat.marker_grounding = True
    strat.single_camera = True
    assert strat.condition() == {"marker_grounding": True, "single_camera": True}


def test_single_camera_keeps_only_referenced_camera(tmp_path):
    s = _sample(tmp_path, ["<c1,CAM_FRONT,0.5,0.5>"])  # 2 imgs: CAM_BACK, CAM_FRONT
    strat = DirectAnswer()
    assert {im.camera for im in strat.images_for(s)} == {"CAM_BACK", "CAM_FRONT"}
    strat.single_camera = True
    out = strat.images_for(s)
    assert [im.camera for im in out] == ["CAM_FRONT"]


def test_single_camera_passes_through_when_multiple_cameras(tmp_path):
    s = _sample(tmp_path, ["<c1,CAM_FRONT,0.5,0.5>", "<c2,CAM_BACK,0.5,0.5>"])
    strat = DirectAnswer()
    strat.single_camera = True
    # Two referenced cameras -> can't reduce; keep the full surround view.
    assert {im.camera for im in strat.images_for(s)} == {"CAM_BACK", "CAM_FRONT"}


def test_yes_no_question_detection():
    assert is_yes_no_question("Is <c1,CAM_BACK,0.5,0.5> a traffic sign or a road barrier?")
    assert not is_yes_no_question("What is the moving status of <c1,CAM_BACK,0.5,0.5>?")
    assert not is_yes_no_question("What are the important objects in the scene?")


def test_answer_instruction_by_question_type():
    yn = _q("Is <c1,CAM_BACK,0.5,0.5> a traffic sign or a road barrier?")
    assert answer_instruction(yn) == "Answer Yes or No."
    assert answer_instruction(_q("What is X?")) == "Give a concise answer."
    assert "letter" in answer_instruction(_q("Pick one", options=["A", "B"]))
