"""`build_rows` joins predictions with gold and assigns each a selective-prediction
outcome: answered (scored) or abstained. These tests pin the abstention handling —
an abstention is a refusal, so it must not be scored as wrong nor dropped as
missing-confidence, and it lowers coverage rather than accuracy."""

from avbench.eval.report import build_rows, coverage
from avbench.eval.scorer import get_scorer

GOLD = {
    "s1": {"sample_id": "s1", "task_type": "perception", "answer": "A"},
    "s2": {"sample_id": "s2", "task_type": "perception", "answer": "B"},
    "s3": {"sample_id": "s3", "task_type": "behavior", "answer": "C"},
}
SC = get_scorer("exact")


def test_build_rows_scores_answered_predictions():
    preds = [{"sample_id": "s1", "answer": "A", "verbal_confidence": 0.9}]
    rows, n_err = build_rows(preds, GOLD, SC)
    assert n_err == 0
    assert len(rows) == 1
    r = rows[0]
    assert r.abstained is False
    assert r.correct == 1
    assert r.confidence == 0.9
    assert r.pred_label == "A"


def test_abstention_is_not_scored_wrong_and_not_missing_confidence():
    # The model declined: no confidence line -> verbal_confidence None, but this is a
    # refusal, not missing data. It must be recorded as abstained (correct=None), not
    # counted as a wrong answer and not dropped as missing-confidence.
    preds = [{"sample_id": "s1", "answer": "I cannot determine this.",
              "abstained": True, "verbal_confidence": None}]
    rows, n_err = build_rows(preds, GOLD, SC)
    assert n_err == 0
    assert len(rows) == 1
    r = rows[0]
    assert r.abstained is True
    assert r.correct is None
    assert r.pred_label is None  # kept out of the confusion matrix


def test_answered_without_confidence_is_scored_not_dropped():
    # An answered item lacking a confidence value is still a real answer: it counts
    # toward coverage/accuracy (confidence-independent), only its calibration is absent.
    preds = [{"sample_id": "s2", "answer": "B", "verbal_confidence": None}]
    rows, _ = build_rows(preds, GOLD, SC)
    assert len(rows) == 1
    assert rows[0].abstained is False
    assert rows[0].correct == 1
    assert rows[0].confidence is None


def test_coverage_is_answered_over_total():
    rows, _ = build_rows(
        [
            {"sample_id": "s1", "answer": "A", "verbal_confidence": 0.9},
            {"sample_id": "s2", "answer": "I cannot determine.", "abstained": True},
            {"sample_id": "s3", "answer": "I cannot determine.", "abstained": True},
        ],
        GOLD, SC)
    assert coverage(rows) == 1 / 3
    # selective accuracy is over answered items only (the one answered item is correct)
    answered = [r for r in rows if not r.abstained]
    assert sum(r.correct for r in answered) / len(answered) == 1.0
