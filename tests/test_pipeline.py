"""
Tests the orchestration logic in pipeline.py -- step order, data threading
between steps, and the assembled PipelineExecution record -- using fake
classify/draft functions and a fake retriever. None of this touches a real
LLM API or a real embedding model, so it runs anywhere in milliseconds.
"""
from config import Brand, EscalationConfig
from pipeline import run_pipeline


class FakeRetriever:
    """Returns a fixed, pre-sorted list of precedents regardless of the query."""
    def __init__(self, precedents):
        self._precedents = precedents

    def query(self, message, k=3, min_similarity=0.0):
        return self._precedents[:k]


def fake_classify_confident(message, taxonomy):
    return {
        "intent": "product_issue_or_bug", "confidence": 0.95,
        "reasoning": "fake", "sentiment": "neutral", "urgency": "low",
    }


def fake_classify_uncertain(message, taxonomy):
    return {
        "intent": "how_to_or_feature_question", "confidence": 0.4,
        "reasoning": "fake", "sentiment": "neutral", "urgency": "low",
    }


def fake_classify_legal(message, taxonomy):
    return {
        "intent": "legal_regulatory_threat", "confidence": 0.9,
        "reasoning": "fake", "sentiment": "furious", "urgency": "critical",
    }


def fake_draft(message, intent, precedents, brand_name="", tone_description="", signoff=""):
    return f"Thanks for reaching out. {signoff}".strip()


TEST_BRAND = Brand(id="t", name="TestBrand", handle="@TestBrand", tone_description="terse", agent_signoff="^TB")


def test_auto_handles_when_confident_and_grounded():
    retriever = FakeRetriever([{"customer_msg": "similar", "resolution": "fixed it", "similarity": 0.9}])
    exec_ = run_pipeline(
        "My app keeps crashing",
        retriever,
        brand=TEST_BRAND,
        classify_fn=fake_classify_confident,
        draft_fn=fake_draft,
    )
    assert exec_.classification["intent"] == "product_issue_or_bug"
    assert exec_.retrieval["top_similarity"] == 0.9
    assert exec_.draft["text"] == "Thanks for reaching out. ^TB"
    assert exec_.draft["signoff_used"] == "^TB"
    assert exec_.draft["grounding_source_count"] == 1
    assert exec_.escalation.decision == "auto_handle"
    assert exec_.total_latency_ms >= 0
    assert exec_.id.startswith("exec_")


def test_escalates_when_classification_uncertain():
    retriever = FakeRetriever([{"customer_msg": "similar", "resolution": "fixed it", "similarity": 0.9}])
    exec_ = run_pipeline(
        "vague thing",
        retriever,
        brand=TEST_BRAND,
        classify_fn=fake_classify_uncertain,
        draft_fn=fake_draft,
    )
    assert exec_.escalation.decision == "escalate"


def test_escalates_and_routes_for_legal_intent():
    retriever = FakeRetriever([])
    exec_ = run_pipeline(
        "I'm calling my attorney about this",
        retriever,
        brand=TEST_BRAND,
        classify_fn=fake_classify_legal,
        draft_fn=fake_draft,
    )
    assert exec_.escalation.decision == "escalate"
    assert exec_.escalation.routed_queue == "Legal & Regulatory Triage"
    assert exec_.retrieval["top_similarity"] == 0.0
    assert exec_.retrieval["precedents"] == []


def test_respects_retrieval_k():
    precedents = [
        {"customer_msg": f"msg{i}", "resolution": f"res{i}", "similarity": 0.9 - i * 0.1}
        for i in range(5)
    ]
    retriever = FakeRetriever(precedents)
    exec_ = run_pipeline(
        "message", retriever, brand=TEST_BRAND,
        classify_fn=fake_classify_confident, draft_fn=fake_draft, retrieval_k=2,
    )
    assert len(exec_.retrieval["precedents"]) == 2
    assert exec_.draft["grounding_source_count"] == 2


def test_custom_escalation_config_is_threaded_through():
    strict = EscalationConfig(confidence_threshold=0.99)
    retriever = FakeRetriever([{"customer_msg": "x", "resolution": "y", "similarity": 0.9}])
    exec_ = run_pipeline(
        "message", retriever, brand=TEST_BRAND,
        classify_fn=fake_classify_confident, draft_fn=fake_draft,
        escalation_config=strict,
    )
    # fake_classify_confident returns confidence=0.95, which fails a 0.99 bar
    assert exec_.escalation.decision == "escalate"


def test_message_and_author_are_preserved_in_the_record():
    retriever = FakeRetriever([{"customer_msg": "x", "resolution": "y", "similarity": 0.9}])
    exec_ = run_pipeline(
        "exact customer text", retriever, author="@some_user", brand=TEST_BRAND,
        classify_fn=fake_classify_confident, draft_fn=fake_draft,
    )
    assert exec_.message_text == "exact customer text"
    assert exec_.message_author == "@some_user"
