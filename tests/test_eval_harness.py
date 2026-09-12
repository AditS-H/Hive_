import os
import json
import tempfile

from golden_set import GoldenCase
from eval_harness import run_eval, format_comparison_table, save_report


class FakeRetriever:
    def query(self, message, k=3, min_similarity=0.0):
        return [{"customer_msg": "similar past case", "resolution": "we fixed it", "similarity": 0.9}]


def fake_classify(message, taxonomy):
    msg = message.lower()
    if "charged" in msg or "billing" in msg:
        intent = "billing_refund_dispute"
    elif "sue" in msg or "lawyer" in msg:
        intent = "legal_regulatory_threat"
    elif "crash" in msg:
        intent = "product_issue_or_bug"
    else:
        intent = "how_to_or_feature_question"  # deliberately wrong for the "freezing" case below
    is_furious = "sue" in msg
    return {
        "intent": intent, "confidence": 0.9, "reasoning": "fake",
        "sentiment": "furious" if is_furious else "neutral",
        "urgency": "critical" if is_furious else "low",
    }


def fake_draft(message, intent, precedents, brand_name="", tone_description="", signoff=""):
    return f"reply {signoff}".strip()


_call_count = {"n": 0}


def fake_judge(message, intent, draft, precedents):
    # flag hallucination only on the 2nd call (the billing case), everything else clean
    _call_count["n"] += 1
    hallucination = 1 if _call_count["n"] == 2 else 0
    return {"groundedness": 4, "tone_match": 5, "resolution_quality": 4,
            "hallucination_flag": hallucination, "notes": "fake judge note"}


GOLDEN = [
    GoldenCase(id="g1", customer_msg="my app crashes constantly", ground_truth_intent="product_issue_or_bug",
               ground_truth_resolution="reinstall", difficulty="easy", expected_action="auto_handle"),
    GoldenCase(id="g2", customer_msg="I was charged twice this month", ground_truth_intent="billing_refund_dispute",
               ground_truth_resolution="refund", difficulty="easy", expected_action="auto_handle"),
    GoldenCase(id="g3", customer_msg="I will sue you and call my lawyer", ground_truth_intent="legal_regulatory_threat",
               ground_truth_resolution="escalate to legal", difficulty="adversarial", expected_action="escalate"),
    GoldenCase(id="g4", customer_msg="the app keeps freezing", ground_truth_intent="product_issue_or_bug",
               ground_truth_resolution="reinstall", difficulty="medium", expected_action="auto_handle"),
]


def _run():
    _call_count["n"] = 0
    return run_eval(
        GOLDEN, FakeRetriever(),
        classify_fn=fake_classify, draft_fn=fake_draft, judge_fn=fake_judge,
        include_baselines=True, baseline_n_splits=5, baseline_seed=1,
    )


def test_sample_size_and_case_count():
    report = _run()
    assert report.sample_size == 4
    assert len(report.case_results) == 4


def test_intent_accuracy_reflects_one_misclassification():
    report = _run()
    # g4 gets misclassified as how_to_or_feature_question by fake_classify
    assert report.classification.accuracy == 0.75
    wrong = [cr for cr in report.case_results if not cr.is_correct_intent]
    assert len(wrong) == 1
    assert wrong[0].test_case.id == "g4"


def test_escalation_decision_and_accuracy():
    report = _run()
    decisions = {cr.test_case.id: cr.pipeline_result.escalation.decision for cr in report.case_results}
    assert decisions["g1"] == "auto_handle"
    assert decisions["g2"] == "auto_handle"
    assert decisions["g3"] == "escalate"   # legal intent + sue/lawyer keywords + furious sentiment
    assert decisions["g4"] == "auto_handle"
    assert report.auto_handle_rate == 0.75
    assert report.escalation_rate == 0.25
    # escalation decision matches expected_action on all 4, even though g4's
    # intent label was wrong -- intent correctness and escalation correctness
    # are independent measurements.
    assert report.escalation_accuracy == 1.0


def test_hallucination_rate_from_judge_flags():
    report = _run()
    assert report.hallucination_rate == 0.25  # only g2's judge call was flagged


def test_avg_judge_scores():
    report = _run()
    assert report.avg_groundedness == 4.0
    assert report.avg_tone_match == 5.0
    assert report.avg_resolution_quality == 4.0


def test_baselines_included_by_default_and_excludable():
    with_baselines = _run()
    assert len(with_baselines.baselines) == 2

    without = run_eval(
        GOLDEN, FakeRetriever(), classify_fn=fake_classify, draft_fn=fake_draft,
        judge_fn=fake_judge, include_baselines=False,
    )
    assert without.baselines == []


def test_empty_golden_set_raises():
    try:
        run_eval([], FakeRetriever(), classify_fn=fake_classify, draft_fn=fake_draft, judge_fn=fake_judge)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_format_comparison_table_contains_all_three_rows():
    report = _run()
    table = format_comparison_table(report)
    assert "Majority Class" in table
    assert "TF-IDF + Logistic Regression" in table
    assert "Gemini Grounded Agent Pipeline" in table
    assert "75.0%" in table  # production accuracy


def test_save_report_round_trips_as_json():
    report = _run()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.json")
        save_report(report, path)
        with open(path) as f:
            data = json.load(f)
    assert data["sample_size"] == 4
    assert data["hallucination_rate"] == 0.25
    assert len(data["case_results"]) == 4
    assert data["case_results"][0]["test_case"]["id"] == "g1"
    assert "escalation" in data["case_results"][0]["pipeline_result"]
