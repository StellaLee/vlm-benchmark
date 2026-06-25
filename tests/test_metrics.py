"""ECE/AUROC are the headline numbers. Test them against closed-form cases where
the right answer is obvious, so a miscomputation can't slip into a draft."""

import math

from avbench.eval.metrics import (
    accuracy,
    auroc,
    balanced_accuracy,
    confusion_matrix,
    exact_match,
    expected_calibration_error,
    precision_per_class,
    recall_per_class,
)


def test_accuracy():
    assert accuracy([1, 1, 0, 0]) == 0.5
    assert accuracy([1, 1, 1]) == 1.0


def test_exact_match_is_case_insensitive():
    assert exact_match("A", "a") == 1
    assert exact_match("B", "C") == 0
    assert exact_match(None, "x") == 0


def test_ece_zero_when_perfectly_calibrated():
    # correct@conf=1.0 and wrong@conf=0.0 -> accuracy equals confidence in every bin.
    correct = [1, 1, 0, 0]
    conf = [1.0, 1.0, 0.0, 0.0]
    assert expected_calibration_error(correct, conf) == 0.0


def test_ece_max_when_confidently_wrong():
    # all wrong but fully confident -> |acc - conf| = 1 in the single occupied bin.
    assert expected_calibration_error([0, 0, 0], [1.0, 1.0, 1.0]) == 1.0


def test_ece_known_intermediate():
    # 10 samples, all conf=0.8, half correct -> |0.5 - 0.8| = 0.3
    correct = [1] * 5 + [0] * 5
    conf = [0.8] * 10
    assert math.isclose(expected_calibration_error(correct, conf), 0.3, abs_tol=1e-9)


def test_auroc_perfect_and_reversed():
    correct = [1, 1, 0, 0]
    assert auroc(correct, [0.9, 0.8, 0.2, 0.1]) == 1.0   # confidence ranks correct above wrong
    assert auroc(correct, [0.1, 0.2, 0.8, 0.9]) == 0.0   # perfectly reversed


def test_auroc_undefined_when_single_class():
    # No incorrect answers -> AUROC undefined -> NaN, not a crash.
    assert math.isnan(auroc([1, 1, 1], [0.9, 0.5, 0.7]))


# --- discrimination metrics ---------------------------------------------------
# The yes/no base-rate trap: 9 "no" + 6 "yes" gold, a model that always says "no".
_GOLD = ["no"] * 9 + ["yes"] * 6
_CONST_NO = ["no"] * 15


def test_confusion_matrix_counts():
    labels, m = confusion_matrix(_GOLD, _CONST_NO, labels=["no", "yes"])
    assert labels == ["no", "yes"]
    # rows=gold, cols=pred: all 9 no->no, all 6 yes->no, nothing predicted yes.
    assert m == [[9, 0], [6, 0]]


def test_recall_exposes_zero_positive_class():
    rec = recall_per_class(_GOLD, _CONST_NO, labels=["no", "yes"])
    assert rec["no"] == 1.0      # every "no" caught
    assert rec["yes"] == 0.0     # zero positive-class recall, hidden by raw accuracy


def test_balanced_accuracy_defeats_base_rate_trap():
    # Raw accuracy of constant-"no" = 9/15 = 0.6; balanced accuracy = mean(1.0, 0.0).
    assert accuracy([1] * 9 + [0] * 6) == 0.6
    assert balanced_accuracy(_GOLD, _CONST_NO, labels=["no", "yes"]) == 0.5


def test_precision_nan_for_never_predicted_class():
    prec = precision_per_class(_GOLD, _CONST_NO, labels=["no", "yes"])
    assert prec["no"] == 9 / 15
    assert math.isnan(prec["yes"])  # "yes" never predicted -> precision undefined
