from config import INTENT_TAXONOMY
from demo_fakes import fake_classify_intent, fake_draft_reply, fake_judge_reply


def test_classify_matches_billing_keywords():
    result = fake_classify_intent("I was billed twice this month", INTENT_TAXONOMY)
    assert result["intent"] == "billing_refund_dispute"
    assert result["confidence"] >= 0.76


def test_classify_matches_past_tense_crash():
    # regression check: "crashed" must match, not just bare "crash"
    result = fake_classify_intent("the app crashed again this morning", INTENT_TAXONOMY)
    assert result["intent"] == "product_issue_or_bug"


def test_classify_falls_back_to_how_to_with_lower_confidence():
    result = fake_classify_intent("what a nice day today", INTENT_TAXONOMY)
    assert result["intent"] == "how_to_or_feature_question"
    assert result["confidence"] < 0.76  # low enough to fail the escalation confidence rule


def test_classify_detects_furious_sentiment():
    result = fake_classify_intent("this is unacceptable, worst service ever!!", INTENT_TAXONOMY)
    assert result["sentiment"] == "furious"
    assert result["urgency"] == "critical"


def test_legal_keywords_take_priority_over_billing():
    result = fake_classify_intent("I'm suing over these charges, my lawyer will call you", INTENT_TAXONOMY)
    assert result["intent"] == "legal_regulatory_threat"


def test_draft_uses_top_precedent_resolution():
    precedents = [{"customer_msg": "x", "resolution": "Here is the fix.", "similarity": 0.9}]
    draft = fake_draft_reply("message", "product_issue_or_bug", precedents, signoff="^Demo")
    assert draft == "Here is the fix. ^Demo"


def test_draft_does_not_duplicate_signoff_already_in_resolution():
    precedents = [{"customer_msg": "x", "resolution": "Here is the fix. ^Demo", "similarity": 0.9}]
    draft = fake_draft_reply("message", "product_issue_or_bug", precedents, signoff="^Demo")
    assert draft.count("^Demo") == 1


def test_draft_handles_no_precedents():
    draft = fake_draft_reply("message", "how_to_or_feature_question", [], signoff="^Demo")
    assert "^Demo" in draft
    assert len(draft) > 0


def test_judge_scores_higher_with_grounding():
    grounded = fake_judge_reply("m", "i", "d", [{"customer_msg": "x", "resolution": "y", "similarity": 0.9}])
    ungrounded = fake_judge_reply("m", "i", "d", [])
    assert grounded["groundedness"] > ungrounded["groundedness"]
    assert grounded["hallucination_flag"] == 0
