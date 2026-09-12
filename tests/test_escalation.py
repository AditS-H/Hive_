"""
Pure-logic tests for escalation.py. No network, no API keys -- every case
here should run in milliseconds.
"""
from config import EscalationConfig
from escalation import evaluate_escalation_rules


def _classification(intent="how_to_or_feature_question", confidence=0.9,
                     sentiment="neutral", urgency="low"):
    return {
        "intent": intent, "confidence": confidence, "reasoning": "test",
        "sentiment": sentiment, "urgency": urgency,
    }


def _retrieval(top_similarity=0.8, precedents=None):
    if precedents is None:
        precedents = [{"customer_msg": "x", "resolution": "y", "similarity": top_similarity}]
    return {"precedents": precedents, "top_similarity": top_similarity}


def test_all_rules_pass_auto_handles():
    result = evaluate_escalation_rules(
        "How do I cancel my plan?",
        _classification(confidence=0.95),
        _retrieval(top_similarity=0.9),
        "Here's how to cancel: go to Settings > Plan > Cancel.",
    )
    assert result.decision == "auto_handle"
    assert all(r.passed for r in result.rules)
    assert result.routed_queue is None
    assert len(result.rules) == 6


def test_low_confidence_escalates():
    result = evaluate_escalation_rules(
        "vague message",
        _classification(confidence=0.5),
        _retrieval(top_similarity=0.9),
        "A generic reply.",
    )
    assert result.decision == "escalate"
    rule = next(r for r in result.rules if r.id == "rule_confidence")
    assert rule.passed is False
    assert "50.0%" in rule.actual_value


def test_weak_grounding_escalates():
    result = evaluate_escalation_rules(
        "message",
        _classification(confidence=0.95),
        _retrieval(top_similarity=0.2),
        "reply",
    )
    assert result.decision == "escalate"
    rule = next(r for r in result.rules if r.id == "rule_grounding")
    assert rule.passed is False


def test_empty_precedents_fails_grounding_even_with_similarity_zero_default():
    # No precedents at all -> grounding must fail regardless of top_similarity.
    result = evaluate_escalation_rules(
        "message",
        _classification(confidence=0.95),
        {"precedents": [], "top_similarity": 0.0},
        "reply",
    )
    rule = next(r for r in result.rules if r.id == "rule_grounding")
    assert rule.passed is False


def test_legal_keyword_triggers_legal_queue():
    result = evaluate_escalation_rules(
        "I will call my lawyer and sue you for this.",
        _classification(confidence=0.95, sentiment="furious", urgency="high"),
        _retrieval(top_similarity=0.9),
        "reply",
    )
    assert result.decision == "escalate"
    kw_rule = next(r for r in result.rules if r.id == "rule_risk_keywords")
    assert kw_rule.passed is False
    assert "lawyer" in kw_rule.actual_value
    assert "sue" in kw_rule.actual_value
    assert result.routed_queue == "Legal & Regulatory Triage"


def test_keyword_scan_does_not_false_positive_on_substrings():
    # "court" as a substring of "courtesy" must NOT match \bcourt\b.
    result = evaluate_escalation_rules(
        "Thanks for the courtesy discount!",
        _classification(confidence=0.95),
        _retrieval(top_similarity=0.9),
        "reply",
    )
    kw_rule = next(r for r in result.rules if r.id == "rule_risk_keywords")
    assert kw_rule.passed is True


def test_furious_sentiment_routes_to_retention():
    result = evaluate_escalation_rules(
        "This is unacceptable, worst service ever!!",
        _classification(confidence=0.95, sentiment="furious", urgency="critical"),
        _retrieval(top_similarity=0.9),
        "reply",
    )
    assert result.decision == "escalate"
    sentiment_rule = next(r for r in result.rules if r.id == "rule_sentiment")
    assert sentiment_rule.passed is False
    assert sentiment_rule.severity == "warning"
    assert result.routed_queue == "Priority Retention Team"


def test_mandatory_intent_routes_to_trust_and_safety():
    result = evaluate_escalation_rules(
        "someone logged into my account from another country",
        _classification(intent="account_security_breach", confidence=0.95),
        _retrieval(top_similarity=0.9),
        "reply",
    )
    assert result.decision == "escalate"
    intent_rule = next(r for r in result.rules if r.id == "rule_intent_restriction")
    assert intent_rule.passed is False
    assert result.routed_queue == "Tier-2 Trust & Safety"


def test_legal_intent_without_keyword_still_routes_legal_via_general_queue():
    # legal_regulatory_threat intent with no matched keyword and non-furious
    # sentiment should still escalate (mandatory intent rule) and, absent a
    # keyword/security/furious match, fall through to the general queue.
    result = evaluate_escalation_rules(
        "I am filing a formal complaint with the regulator about this.",
        _classification(intent="legal_regulatory_threat", confidence=0.95),
        _retrieval(top_similarity=0.9),
        "reply",
    )
    assert result.decision == "escalate"
    assert result.routed_queue == "Legal & Regulatory Triage"  # "regulator" is a sensitive keyword


def test_hallucinated_dollar_amount_escalates():
    result = evaluate_escalation_rules(
        "I want a refund",
        _classification(confidence=0.95),
        _retrieval(top_similarity=0.9, precedents=[
            {"customer_msg": "x", "resolution": "Refunds take 3-5 days.", "similarity": 0.9}
        ]),
        "We'll refund you $50 immediately.",
    )
    rule = next(r for r in result.rules if r.id == "rule_hallucination_guard")
    assert rule.passed is False
    assert result.decision == "escalate"


def test_dollar_amount_present_in_precedent_does_not_flag():
    result = evaluate_escalation_rules(
        "I want a refund",
        _classification(confidence=0.95),
        _retrieval(top_similarity=0.9, precedents=[
            {"customer_msg": "x", "resolution": "We refunded $50 to your card.", "similarity": 0.9}
        ]),
        "As shown in similar cases, we'll refund $50 to your card.",
    )
    rule = next(r for r in result.rules if r.id == "rule_hallucination_guard")
    assert rule.passed is True


def test_custom_config_thresholds_are_respected():
    strict_config = EscalationConfig(confidence_threshold=0.99, min_precedent_similarity=0.99)
    result = evaluate_escalation_rules(
        "message",
        _classification(confidence=0.9),
        _retrieval(top_similarity=0.9),
        "reply",
        config=strict_config,
    )
    assert result.decision == "escalate"
    assert not next(r for r in result.rules if r.id == "rule_confidence").passed
    assert not next(r for r in result.rules if r.id == "rule_grounding").passed


def test_audit_explanation_present_for_both_outcomes():
    passing = evaluate_escalation_rules(
        "How do I do X", _classification(confidence=0.95), _retrieval(top_similarity=0.9), "reply",
    )
    assert "approved" in passing.audit_explanation.lower()

    failing = evaluate_escalation_rules(
        "I'll sue", _classification(confidence=0.95), _retrieval(top_similarity=0.9), "reply",
    )
    assert "escalation triggered" in failing.audit_explanation.lower()
