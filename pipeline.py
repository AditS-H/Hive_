"""
The five-step, per-message pipeline: classify -> retrieve -> draft ->
escalate -> log. `run_pipeline()` is a single function meant to be read top
to bottom in a code review -- no agent framework, no hidden control flow
(see llm_clients.py's module docstring for why that's a deliberate choice,
not an oversight).

Deliberately decoupled from embed_utils: `retriever` only needs a
`.query(message, k, min_similarity) -> list[dict]` method. That's
embed_utils.RetrievalIndex's real signature, but tests can hand run_pipeline
any object shaped like that -- see tests/test_pipeline.py, which never loads
a real sentence-transformers model or calls a real LLM API.
"""
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from config import INTENT_TAXONOMY, DEFAULT_ESCALATION_CONFIG, DEMO_BRAND, Brand, EscalationConfig
from escalation import evaluate_escalation_rules, EscalationResult
from llm_clients import classify_intent, draft_reply


class Retriever(Protocol):
    def query(self, message: str, k: int = 3, min_similarity: float = 0.0) -> list[dict]: ...


@dataclass
class PipelineExecution:
    id: str
    timestamp: str
    message_text: str
    message_author: str
    classification: dict
    retrieval: dict
    draft: dict
    escalation: EscalationResult
    total_latency_ms: int


def run_pipeline(
    message_text: str,
    retriever: Retriever,
    author: str = "@customer",
    taxonomy: dict = INTENT_TAXONOMY,
    brand: Brand = DEMO_BRAND,
    escalation_config: EscalationConfig = DEFAULT_ESCALATION_CONFIG,
    retrieval_k: int = 3,
    classify_fn: Callable[[str, dict], dict] = classify_intent,
    draft_fn: Callable[..., str] = draft_reply,
) -> PipelineExecution:
    """
    classify_fn/draft_fn default to the real llm_clients functions but are
    swappable -- tests inject deterministic fakes so the orchestration logic
    (not the LLM call itself) is what gets verified without network access.
    """
    t_start = time.perf_counter()

    # 1. Classify intent (+ sentiment, urgency -- escalation rule 4 needs both)
    t0 = time.perf_counter()
    classification = classify_fn(message_text, taxonomy)
    classification.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000))

    # 2. Retrieve precedents (local, offline, no LLM call)
    t0 = time.perf_counter()
    precedents = retriever.query(message_text, k=retrieval_k, min_similarity=0.0)
    # min_similarity=0.0 here deliberately: we want the top-k regardless of
    # quality so escalation rule 2 can see and reject weak grounding itself,
    # rather than retrieval silently hiding a low-similarity match.
    top_similarity = precedents[0]["similarity"] if precedents else 0.0
    retrieval = {
        "precedents": precedents,
        "top_similarity": top_similarity,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }

    # 3. Draft a grounded reply
    t0 = time.perf_counter()
    draft_text = draft_fn(
        message_text,
        classification["intent"],
        precedents,
        brand_name=brand.name,
        tone_description=brand.tone_description,
        signoff=brand.agent_signoff,
    )
    draft = {
        "text": draft_text,
        "grounding_source_count": len(precedents),
        "signoff_used": brand.agent_signoff,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }

    # 4. Evaluate the deterministic escalation gate
    escalation = evaluate_escalation_rules(
        message_text, classification, retrieval, draft["text"], escalation_config
    )

    # 5. Assemble the auditable record -- this is the artifact a human (or
    # the eval harness) actually inspects, so every upstream field is kept,
    # not just the final decision.
    total_latency_ms = round((time.perf_counter() - t_start) * 1000)
    return PipelineExecution(
        id=f"exec_{uuid.uuid4().hex[:12]}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        message_text=message_text,
        message_author=author,
        classification=classification,
        retrieval=retrieval,
        draft=draft,
        escalation=escalation,
        total_latency_ms=total_latency_ms,
    )
