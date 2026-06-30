"""Marker grounding changes what the model sees, so a bug here would silently
shift the headline numbers per condition. These pin the coordinate parse
(markers must land at the right pixel), that marking is scoped to the referenced
camera + current frame only, and that the condition is recorded the way
evaluate.py stratifies on."""

from PIL import Image

from avbench.inference.grounding import parse_refs, render_markers
from avbench.inference.strategies.base import answer_instruction
from avbench.inference.strategies.direct import DirectAnswer
from avbench.inference.view import Layout, View
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


def test_strategy_delegates_presentation_to_its_view(tmp_path):
    # The strategy is just a thin pass-through to its View (pipeline behavior itself
    # is pinned in test_view.py); here we check the wiring + the default.
    strat = DirectAnswer()
    assert strat.condition() == {"layout": "separate", "marker_grounding": False}
    s = _sample(tmp_path, ["<c1,CAM_FRONT,0.5,0.5>"])  # 2 imgs: CAM_BACK, CAM_FRONT
    assert {im.camera for im in strat.images_for(s)} == {"CAM_BACK", "CAM_FRONT"}
    strat.view = View(layout=Layout.SINGLE)
    assert [im.camera for im in strat.images_for(s)] == ["CAM_FRONT"]
    assert strat.condition() == {"layout": "single", "marker_grounding": False}


def test_answer_instruction_by_question_type():
    # keys off the typed prompt_format (set during curation), not a regex on the text.
    yn = Sample(sample_id="d/x", dataset="d", task_type=TaskType.PERCEPTION,
                prompt_format=PromptFormat.YESNO, question="Is <c1> a sign or a barrier?",
                answer="No.", images=[])
    assert answer_instruction(yn) == "Answer Yes or No."
    assert answer_instruction(_q("What is X?")) == "Give a concise answer."
    assert "letter" in answer_instruction(_q("Pick one", options=["A", "B"]))
