"""ECE/AUROC are the headline numbers. Test them against closed-form cases where
the right answer is obvious, so a miscomputation can't slip into a draft."""

import math

from avbench.eval.metrics import accuracy, auroc, exact_match, expected_calibration_error


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
