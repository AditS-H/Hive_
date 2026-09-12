import os
import tempfile

from golden_set import GoldenCase, save_golden_set, load_golden_set, validate_golden_set, sample_for_labeling


TAXONOMY = {"billing_refund_dispute": {}, "how_to_or_feature_question": {}}


def _case(**overrides):
    base = dict(
        id="gold_001", customer_msg="I was charged twice", ground_truth_intent="billing_refund_dispute",
        ground_truth_resolution="Direct to receipts page", difficulty="easy",
        expected_action="auto_handle", risk_factors=[],
    )
    base.update(overrides)
    return GoldenCase(**base)


def test_round_trip_through_jsonl():
    cases = [_case(id="gold_001"), _case(id="gold_002", difficulty="hard", risk_factors=["financial"])]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "golden.jsonl")
        save_golden_set(cases, path)
        loaded = load_golden_set(path)
    assert loaded == cases


def test_validate_clean_set_has_no_problems():
    cases = [_case()]
    assert validate_golden_set(cases, TAXONOMY) == []


def test_validate_catches_unknown_intent():
    cases = [_case(ground_truth_intent="not_a_real_intent")]
    problems = validate_golden_set(cases, TAXONOMY)
    assert len(problems) == 1
    assert "not_a_real_intent" in problems[0]


def test_validate_catches_bad_difficulty_and_action():
    cases = [_case(difficulty="impossible", expected_action="maybe")]
    problems = validate_golden_set(cases, TAXONOMY)
    assert any("difficulty" in p for p in problems)
    assert any("expected_action" in p for p in problems)


def test_validate_catches_duplicate_ids():
    cases = [_case(id="dup"), _case(id="dup")]
    problems = validate_golden_set(cases, TAXONOMY)
    assert any("duplicate id" in p for p in problems)


def test_validate_catches_empty_resolution():
    cases = [_case(ground_truth_resolution="")]
    problems = validate_golden_set(cases, TAXONOMY)
    assert any("empty ground_truth_resolution" in p for p in problems)


def test_sample_for_labeling_returns_n_unique_empty_templates():
    messages = [f"message number {i} with some words" for i in range(30)]
    sampled = sample_for_labeling(messages, n=9, seed=1)
    assert len(sampled) == 9
    ids = [s["id"] for s in sampled]
    assert len(set(ids)) == 9
    for s in sampled:
        assert s["ground_truth_intent"] == ""
        assert s["expected_action"] == ""


def test_sample_for_labeling_deduplicates_input():
    messages = ["same message"] * 10 + ["different message"]
    sampled = sample_for_labeling(messages, n=5, seed=1)
    # only 2 unique inputs exist, so we can never get more than 2 back
    assert len(sampled) <= 2


def test_sample_for_labeling_is_deterministic_given_seed():
    messages = [f"msg {i} " + "x" * i for i in range(50)]
    a = sample_for_labeling(messages, n=10, seed=7)
    b = sample_for_labeling(messages, n=10, seed=7)
    assert [c["customer_msg"] for c in a] == [c["customer_msg"] for c in b]


def test_sample_for_labeling_handles_fewer_messages_than_requested():
    messages = ["only one message here"]
    sampled = sample_for_labeling(messages, n=10, seed=1)
    assert len(sampled) == 1
