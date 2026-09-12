from golden_set import GoldenCase
from baselines import majority_class_baseline, tfidf_logreg_baseline, run_all_baselines, _make_folds


def _case(i, msg, intent, action="auto_handle"):
    return GoldenCase(
        id=f"gold_{i:03d}", customer_msg=msg, ground_truth_intent=intent,
        ground_truth_resolution="res", difficulty="easy", expected_action=action,
    )


def _separable_dataset(per_class=8):
    """Text that clearly correlates with label, repeated with light variation
    so a bag-of-words classifier has an easy, real signal to find."""
    templates = {
        "billing_refund_dispute": [
            "I was charged twice for my subscription this month",
            "There is a duplicate charge on my credit card statement",
            "Please refund the extra billing charge I was hit with",
            "My invoice shows an incorrect amount charged to my card",
        ],
        "account_login_auth": [
            "I forgot my password and cannot log in",
            "My login keeps failing even after a password reset",
            "Two factor authentication code never arrives by text",
            "I am locked out of my account after changing my email",
        ],
        "product_issue_or_bug": [
            "The app crashes every time I open it on my phone",
            "Playback keeps stuttering and the app freezes constantly",
            "The app closes immediately after the latest update",
            "There is a bug causing the screen to go completely blank",
        ],
    }
    cases = []
    i = 0
    for intent, msgs in templates.items():
        for _ in range(per_class):
            for msg in msgs:
                i += 1
                cases.append(_case(i, msg, intent))
    return cases


def test_majority_class_baseline_predicts_the_most_common_label():
    cases = (
        [_case(i, "x", "product_issue_or_bug") for i in range(10)]
        + [_case(i, "y", "billing_refund_dispute") for i in range(10, 13)]
    )
    result = majority_class_baseline(cases, n_splits=3, seed=1)
    # majority class is heavily dominant (10 vs 3), so folds should mostly predict it,
    # driving accuracy well above chance for a 2-class problem
    assert result.name == "Majority Class"
    assert result.metrics.n == 13
    assert result.metrics.accuracy > 0.5


def test_tfidf_logreg_beats_majority_class_on_separable_text():
    cases = _separable_dataset(per_class=6)
    majority = majority_class_baseline(cases, n_splits=5, seed=1)
    tfidf = tfidf_logreg_baseline(cases, n_splits=5, seed=1)

    assert tfidf.metrics.accuracy > majority.metrics.accuracy
    assert tfidf.metrics.macro_f1 > majority.metrics.macro_f1
    # 3 balanced classes -> majority-class accuracy should sit near 1/3
    assert 0.2 < majority.metrics.accuracy < 0.5


def test_run_all_baselines_returns_both_in_order():
    cases = _separable_dataset(per_class=4)
    results = run_all_baselines(cases, n_splits=3, seed=1)
    assert [r.name for r in results] == ["Majority Class", "TF-IDF + Logistic Regression"]


def test_handles_class_with_single_example_without_crashing():
    cases = (
        [_case(i, f"billing message variant {i}", "billing_refund_dispute") for i in range(6)]
        + [_case(99, "a totally unique one-off message", "legal_regulatory_threat")]
    )
    # should not raise, and should fall back to unstratified KFold
    majority = majority_class_baseline(cases, n_splits=5, seed=1)
    tfidf = tfidf_logreg_baseline(cases, n_splits=5, seed=1)
    assert majority.stratified is False
    assert tfidf.stratified is False
    assert majority.metrics.n == 7
    assert tfidf.metrics.n == 7


def test_make_folds_caps_n_splits_to_rarest_class():
    labels = ["a"] * 10 + ["b"] * 2
    folds, n_splits, stratified = _make_folds(labels, requested_n_splits=5, seed=1)
    assert stratified is True
    assert n_splits == 2  # capped to the rarest class's count
    assert len(folds) == 2


def test_baseline_result_reports_n_splits_used():
    cases = _separable_dataset(per_class=3)  # small -> may cap below requested 10
    result = tfidf_logreg_baseline(cases, n_splits=10, seed=1)
    assert result.n_splits <= 10
    assert result.n_splits >= 2
