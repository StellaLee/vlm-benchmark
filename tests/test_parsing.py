"""Parsing is the first place a silent bug poisons every downstream number:
a mis-extracted answer or a confidence that leaks into the answer would corrupt
both correctness and calibration without raising an error."""

import pytest

from avbench.inference.parsing import extract_answer, extract_confidence, is_abstention
from avbench.schema import PromptFormat, Sample, TaskType


def _sample(fmt=PromptFormat.QA, answer="x", options=None):
    return Sample(sample_id="s", dataset="d", task_type=TaskType.PLANNING,
                  prompt_format=fmt, question="q", answer=answer, images=[],
                  options=options)


# ---- confidence -------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Answer: A\nConfidence: 80", 0.8),
    ("Answer: A\nConfidence: 100", 1.0),
    ("I am 95% confident", 0.95),
    ("confidence 0", 0.0),
    ("Answer: foo", None),          # no confidence stated
    ("no numbers here", None),
])
def test_extract_confidence(text, expected):
    assert extract_confidence(text) == expected


# ---- answer extraction + no confidence leakage ------------------------------

def test_mcq_answer_is_letter_only():
    s = _sample(PromptFormat.MCQ)
    assert extract_answer("Answer: C\nConfidence: 90", s) == "C"


_OPTS = ["Turn left.", "Going ahead.", "Turn right."]


def test_mcq_recovers_letter_from_quoted_option():
    # A verbose answer that names the option text but no bare letter still maps to
    # the right choice (recovers task competence from a format-noncompliant answer).
    s = _sample(PromptFormat.MCQ, answer="B", options=_OPTS)
    text = ('The silver car is facing forward and in motion, so the most '
            'appropriate status is "Going ahead."')
    assert extract_answer(text, s) == "B"


def test_mcq_bare_letter_wins_over_option_text():
    s = _sample(PromptFormat.MCQ, answer="A", options=_OPTS)
    assert extract_answer("A. Turn left.", s) == "A"


def test_mcq_ambiguous_multiple_options_not_recovered():
    # If several option texts appear, don't guess — leave it for strict scoring.
    s = _sample(PromptFormat.MCQ, answer="A", options=_OPTS)
    text = "It could be going ahead or turn right; hard to say."
    assert extract_answer(text, s) != "B"
    assert extract_answer(text, s) != "C"


def test_glm_boxed_final_answer_after_reasoning():
    # glm-4.1v-thinking reasons first, then marks its final answer in a box at the
    # end. The boxed span is authoritative — not the first stray letter/line.
    s = _sample(PromptFormat.MCQ, answer="D", options=_OPTS)
    text = ("Option A looks plausible but is wrong.\n"
            "Thus the best match is <|begin_of_box|>D<|end_of_box|>.")
    assert extract_answer(text, s) == "D"


def test_glm_box_used_for_open_ended_too():
    s = _sample(PromptFormat.QA)
    text = "Reasoning here.\n<|begin_of_box|>keep going at the same speed<|end_of_box|>"
    assert extract_answer(text, s) == "keep going at the same speed"


def test_open_ended_answer_excludes_confidence_line():
    s = _sample(PromptFormat.QA)
    ans = extract_answer("Answer: keep going at the same speed.\nConfidence: 80", s)
    assert ans == "keep going at the same speed."
    assert "80" not in ans and "Confidence" not in ans


def test_answer_without_label_falls_back_to_text():
    s = _sample(PromptFormat.QA)
    assert extract_answer("the car is stationary", s) == "the car is stationary"


# ---- abstention -------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("I cannot determine this from the available visual input.", True),
    ("There is not enough information to answer.", True),
    ("The ego vehicle should keep going.", False),
])
def test_is_abstention(text, expected):
    assert is_abstention(text) is expected
