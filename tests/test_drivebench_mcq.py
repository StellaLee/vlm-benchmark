"""DriveBench inline-MCQ parsing (pure, no file/network).

The released DriveBench test JSON embeds the choices inside the question text
("Please select the correct answer from the following options: A. ... B. ...")
with no separate `options` field, so the adapter must recognise that pattern,
lift the lettered options into `Sample.options`, and mark the sample as MCQ —
otherwise `--formats mcq` can never select them (they'd all infer as QA)."""

from avbench.curation.adapters.drivebench import DriveBenchAdapter
from avbench.schema import PromptFormat


def _adapter(formats=None):
    # data_root="" + require_images=False keeps this offline (no image index, no
    # existence filter); only the question/answer parsing is exercised.
    return DriveBenchAdapter(qa_file="dummy.json", data_root="",
                             formats=formats, require_images=False)


def _rec(question, answer="A", qtype="perception"):
    return {"question": question, "answer": answer, "question_type": qtype,
            "frame_token": "tok"}


def test_inline_options_parsed_as_mcq():
    rec = _rec(
        "What is the moving status of object <c1,CAM_BACK,0.5,0.5>? "
        "Please select the correct answer from the following options: "
        "A. Going ahead. B. Turn left. C. Turn right.")
    s = _adapter()._to_sample(rec, 0)
    assert s.prompt_format == PromptFormat.MCQ
    assert s.options == ["Going ahead.", "Turn left.", "Turn right."]
    # the "Please select..." scaffolding is stripped from the stem so the prompt
    # renderer (which re-emits the options) doesn't double-print them.
    assert "Please select" not in s.question
    assert s.question == "What is the moving status of object <c1,CAM_BACK,0.5,0.5>?"
    assert s.answer == "A"


def test_multi_sentence_options_not_split_on_period():
    # behavior options contain internal periods; split must key on the letter markers.
    rec = _rec(
        "Predict the behavior of the ego vehicle. "
        "Please select the correct answer from the following options: "
        "A. The ego vehicle is going straight. The ego vehicle is not moving. "
        "B. The ego vehicle is steering to the right. The ego vehicle is driving fast.",
        answer="B", qtype="behavior")
    s = _adapter()._to_sample(rec, 0)
    assert s.options == [
        "The ego vehicle is going straight. The ego vehicle is not moving.",
        "The ego vehicle is steering to the right. The ego vehicle is driving fast.",
    ]


def test_plain_qa_without_marker_stays_qa():
    rec = _rec("What are the important objects in the current scene?",
               answer="There is a gray sedan.")
    s = _adapter()._to_sample(rec, 0)
    assert s.prompt_format == PromptFormat.QA
    assert s.options is None


def test_formats_mcq_filter_selects_inline_mcq():
    mcq = _rec("Status of <c1,CAM_BACK,0.5,0.5>? Please select the correct answer "
               "from the following options: A. Going ahead. B. Turn left.")
    qa = _rec("What are the important objects?", answer="A sedan.")
    a = _adapter(formats=["mcq"])
    assert a._to_sample(mcq, 0) is not None
    assert a._to_sample(qa, 1) is None


# Yes/No identification questions ("Is <c> a A or a B?") are polar — gold is Yes/No,
# not a noun. Classifying them at curation (from the gold answer) lets inference shape
# the prompt off a typed field instead of re-parsing the question text downstream.
def test_yesno_gold_classified_as_yesno():
    q = "Is <c1,CAM_BACK,0.5,0.5> a traffic sign or a road barrier?"
    for ans in ("Yes.", "No.", "yes", "NO"):
        s = _adapter()._to_sample(_rec(q, answer=ans), 0)
        assert s.prompt_format == PromptFormat.YESNO, ans
        assert s.options is None  # not MCQ — no lettered choices


def test_mcq_letter_gold_not_misread_as_yesno():
    s = _adapter()._to_sample(_rec(
        "Status? Please select the correct answer from the following options: "
        "A. Going ahead. B. Turn left.", answer="A"), 0)
    assert s.prompt_format == PromptFormat.MCQ
