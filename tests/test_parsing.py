"""Parsing is the first place a silent bug poisons every downstream number:
a mis-extracted answer or a confidence that leaks into the answer would corrupt
both correctness and calibration without raising an error."""

import pytest

from avbench.inference.parsing import extract_answer, extract_confidence, is_abstention
from avbench.schema import PromptFormat, Sample, TaskType


def _sample(fmt=PromptFormat.QA, answer="x"):
    return Sample(sample_id="s", dataset="d", task_type=TaskType.PLANNING,
                  prompt_format=fmt, question="q", answer=answer, images=[])


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
