"""The scorer assigns the binary correctness label that calibration is computed
against. These tests pin the two behaviors that matter: MCQ exactness, and that
`structured` credits paraphrases `exact` would reject without crediting genuinely
wrong answers."""

from avbench.eval.scorer import answer_label, get_scorer


def test_answer_label_buckets_yes_no_with_trailing_period():
    # Gold answers in the DriveBench yes/no identification split are "Yes."/"No.";
    # the model's parsed answer is "Yes"/"No". Both must map to the same class so the
    # confusion matrix / recall treat yes/no as a 2-class problem, not just MCQ.
    assert answer_label("Yes.", "Yes.") == "yes"
    assert answer_label("No.", "No.") == "no"
    assert answer_label("Yes", "Yes.") == "yes"      # parsed pred matches gold class
    assert answer_label("no", "No.") == "no"          # case-insensitive


def test_answer_label_mcq_uses_letter():
    assert answer_label("A", "A") == "A"
    assert answer_label("The answer is B.", "A") == "B"  # extracts the letter


def test_mcq_letter_match():
    for name in ("exact", "structured"):
        sc = get_scorer(name)
        assert sc.is_correct("C", "C") == 1
        assert sc.is_correct("A", "C") == 0


def test_exact_rejects_valid_paraphrase():
    # "a gray car" is a correct paraphrase of "there is a gray sedan" but exact
    # string match misses it (motivating the structured scorer).
    assert get_scorer("exact").is_correct("a gray car", "there is a gray sedan") == 0


def test_structured_credits_paraphrase():
    sc = get_scorer("structured", threshold=0.5)
    assert sc.is_correct("a gray car", "there is a gray sedan") == 1


def test_structured_rejects_wrong_answer():
    sc = get_scorer("structured", threshold=0.5)
    assert sc.is_correct("turn left and stop", "keep going at the same speed") == 0


def test_identity_is_always_correct():
    # Invariant the synthetic harness relies on: a gold answer scores itself right.
    sc = get_scorer("structured")
    for gold in ["C", "the ego vehicle should keep going", "there is a gray sedan"]:
        assert sc.is_correct(gold, gold) == 1


def test_score_is_bounded():
    res = get_scorer("structured").score("a gray car", "there is a gray sedan")
    assert 0.0 <= res.score <= 1.0
    assert res.correct in (0, 1)
