from metrics import classification_metrics, hallucination_rate, rate


def test_perfect_predictions_score_perfectly():
    y_true = ["a", "b", "a", "c", "b"]
    y_pred = ["a", "b", "a", "c", "b"]
    m = classification_metrics(y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.cohen_kappa == 1.0
    assert m.n == 5


def test_all_wrong_predictions_score_zero_accuracy():
    y_true = ["a", "a", "a"]
    y_pred = ["b", "b", "b"]
    m = classification_metrics(y_true, y_pred)
    assert m.accuracy == 0.0


def test_macro_f1_penalizes_ignoring_a_rare_class():
    # model never predicts the rare class "c" at all
    y_true = ["a", "a", "a", "a", "b", "b", "b", "b", "c"]
    y_pred = ["a", "a", "a", "a", "b", "b", "b", "b", "a"]
    m = classification_metrics(y_true, y_pred)
    assert m.accuracy > 0.85          # only one mistake out of nine
    assert m.macro_f1 < m.accuracy    # but macro F1 catches the ignored class


def test_cohen_kappa_below_one_when_imperfect():
    y_true = ["a", "b", "a", "b", "a", "b"]
    y_pred = ["a", "b", "b", "a", "a", "b"]
    m = classification_metrics(y_true, y_pred)
    assert 0.0 <= m.cohen_kappa < 1.0


def test_hallucination_rate_basic():
    assert hallucination_rate([0, 0, 1, 0]) == 0.25
    assert hallucination_rate([1, 1, 1]) == 1.0
    assert hallucination_rate([]) == 0.0


def test_rate_helper():
    assert rate(3, 10) == 0.3
    assert rate(0, 0) == 0.0


def test_mismatched_lengths_raise():
    try:
        classification_metrics(["a", "b"], ["a"])
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_empty_lists_raise():
    try:
        classification_metrics([], [])
        assert False, "expected AssertionError"
    except AssertionError:
        pass
