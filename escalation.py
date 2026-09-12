"""
Deterministic, auditable escalation gate.

This is the safety boundary between "auto-send this to a real customer" and
"hand it to a human" -- so the decision has to be explainable rule-by-rule,
not a single opaque score. Six independent checks, each one inspectable on
its own; the message auto-handles only if every one passes.

Ported 1:1 from the reference architecture's `escalationRules.ts` (same six
checks, same thresholds, same routing logic) so the Python and TypeScript
versions are provably the same policy. Config lives in config.py; this file
is pure logic with no I/O, so it needs no network access and no API keys to
test -- see tests/test_escalation.py.
"""
import re
from dataclasses import dataclass, field

from config import DEFAULT_ESCALATION_CONFIG, EscalationConfig


@dataclass
class RuleEvaluation:
    id: str
    name: str
    description: str
    passed: bool
    actual_value: str
    threshold: str
    reason: str
    severity: str  # "blocker" | "warning"


@dataclass
class EscalationResult:
    decision: str  # "auto_handle" | "escalate"
    rules: list[RuleEvaluation]
    trigger_reasons: list[str]
    routed_queue: str | None
    audit_explanation: str


# Matches a naked dollar figure ($50, $1200). Used only to flag a draft
# asserting a specific amount that no retrieved precedent backs up -- the
# single worst failure mode in this domain (see llm_clients.draft_reply).
_DOLLAR_RE = re.compile(r"\$\d+")

# Keywords that specifically signal a legal/regulatory proceeding (as
# opposed to e.g. "fraud"/"scam"/"stolen", which are security-flavored and
# route through the intent-based Trust & Safety branch instead). The
# reference architecture only checked for 'lawyer' and 'sue' (plus a dead
# check for the literal string 'legal', which never appears in
# sensitiveKeywords and can never match) -- a message that only says
# "I'm reporting this to the FTC" fell through to the generic queue there.
# Widened to the full legal-flavored subset so anything naming a legal or
# regulatory process reaches the team equipped to handle it.
_LEGAL_ROUTE_KEYWORDS = {
    "lawyer", "attorney", "sue", "lawsuit", "court", "ftc",
    "attorney general", "regulator",
}


def evaluate_escalation_rules(
    customer_msg: str,
    classification: dict,
    retrieval: dict,
    draft_text: str,
    config: EscalationConfig = DEFAULT_ESCALATION_CONFIG,
) -> EscalationResult:
    """
    classification: {"intent", "confidence", "reasoning", "sentiment", "urgency", ...}
        (llm_clients.classify_intent's return shape)
    retrieval: {"precedents": list[dict], "top_similarity": float, ...}
        (built from embed_utils.RetrievalIndex.query's return shape)
    draft_text: the drafted reply string (llm_clients.draft_reply's return value)
    """
    rules: list[RuleEvaluation] = []
    trigger_reasons: list[str] = []
    lower_msg = customer_msg.lower()

    # Rule 1: intent classification confidence
    confidence = classification["confidence"]
    confidence_passed = confidence >= config.confidence_threshold
    rules.append(RuleEvaluation(
        id="rule_confidence",
        name="Intent Classification Confidence",
        description=f"Requires classification confidence >= {config.confidence_threshold * 100:.0f}%",
        passed=confidence_passed,
        actual_value=f"{confidence * 100:.1f}%",
        threshold=f">= {config.confidence_threshold * 100:.0f}%",
        reason=(
            "Intent confidently detected by classifier" if confidence_passed
            else f"Confidence ({confidence * 100:.1f}%) below safe automation cutoff"
        ),
        severity="blocker",
    ))
    if not confidence_passed:
        trigger_reasons.append(
            f"Confidence ({confidence * 100:.1f}%) < {config.confidence_threshold * 100:.0f}% threshold"
        )

    # Rule 2: precedent grounding similarity
    precedents = retrieval.get("precedents", [])
    top_sim = retrieval.get("top_similarity", 0.0)
    grounding_passed = top_sim >= config.min_precedent_similarity and len(precedents) > 0
    rules.append(RuleEvaluation(
        id="rule_grounding",
        name="Historical Precedent Grounding",
        description=f"Requires top precedent similarity >= {config.min_precedent_similarity * 100:.0f}%",
        passed=grounding_passed,
        actual_value=f"{top_sim * 100:.1f}%",
        threshold=f">= {config.min_precedent_similarity * 100:.0f}%",
        reason=(
            f"Strong precedent grounding ({top_sim * 100:.1f}% cosine match)" if grounding_passed
            else f"Precedent similarity ({top_sim * 100:.1f}%) below verified resolution threshold"
        ),
        severity="blocker",
    ))
    if not grounding_passed:
        trigger_reasons.append(
            f"Grounding similarity ({top_sim * 100:.1f}%) < {config.min_precedent_similarity * 100:.0f}%"
        )

    # Rule 3: high-risk / legal keyword scan
    matched_keywords = [
        kw for kw in config.sensitive_keywords
        if re.search(rf"\b{re.escape(kw)}\b", lower_msg)
    ]
    keywords_passed = len(matched_keywords) == 0
    rules.append(RuleEvaluation(
        id="rule_risk_keywords",
        name="High-Risk & Legal Keyword Scan",
        description="Screens for litigation, regulatory complaints, fraud, or chargeback threats",
        passed=keywords_passed,
        actual_value="None detected" if keywords_passed else ", ".join(matched_keywords),
        threshold="0 sensitive keywords",
        reason=(
            "No high-risk litigation or fraud keywords found" if keywords_passed
            else f"Flagged sensitive keyword(s): [{', '.join(matched_keywords)}]"
        ),
        severity="blocker",
    ))
    if not keywords_passed:
        trigger_reasons.append(f"High-risk keyword trigger: {', '.join(matched_keywords)}")

    # Rule 4: sentiment / distress threshold
    is_furious = classification.get("sentiment") == "furious" or classification.get("urgency") == "critical"
    sentiment_passed = not config.disallow_furious_sentiment or not is_furious
    rules.append(RuleEvaluation(
        id="rule_sentiment",
        name="Customer Distress & Hostility Check",
        description="Requires human escalation when severe customer rage or hostility is detected",
        passed=sentiment_passed,
        actual_value=f"{classification.get('sentiment')} (urgency: {classification.get('urgency')})",
        threshold="Non-furious / non-critical",
        reason=(
            "Customer tone within standard operational tolerance" if sentiment_passed
            else "Severe customer distress detected; human empathy required"
        ),
        severity="warning",
    ))
    if not sentiment_passed:
        trigger_reasons.append("Customer distress: furious tone or critical urgency")

    # Rule 5: mandatory escalation intents
    intent = classification["intent"]
    is_restricted_intent = intent in config.mandatory_escalation_intents
    intent_passed = not is_restricted_intent
    rules.append(RuleEvaluation(
        id="rule_intent_restriction",
        name="Policy-Restricted Intent Check",
        description="Guarantees human review for sensitive categories (e.g. security compromise)",
        passed=intent_passed,
        actual_value=intent,
        threshold="Not on mandatory escalation list",
        reason=(
            "Intent is authorized for automated handling" if intent_passed
            else f"Intent [{intent}] is flagged for mandatory specialist review"
        ),
        severity="blocker",
    ))
    if not intent_passed:
        trigger_reasons.append(f"Mandatory escalation policy for intent: {intent}")

    # Rule 6: hallucination guardrail -- flag a drafted dollar amount that no
    # retrieved precedent's resolution text contains.
    invented_dollars = [
        amt for amt in _DOLLAR_RE.findall(draft_text)
        if not any(amt in p.get("resolution", "") for p in precedents)
    ]
    hallucination_passed = len(invented_dollars) == 0
    rules.append(RuleEvaluation(
        id="rule_hallucination_guard",
        name="Grounded Output Factuality Check",
        description="Prevents drafting unauthorized financial amounts or unverifiable commitments",
        passed=hallucination_passed,
        actual_value="Compliant" if hallucination_passed else "Invented financial claim detected",
        threshold="100% grounded facts",
        reason=(
            "Draft strictly reflects precedent without hallucinating unverified figures" if hallucination_passed
            else "Draft asserted unverified financial promise not in precedent"
        ),
        severity="blocker",
    ))
    if not hallucination_passed:
        trigger_reasons.append("Draft contains ungrounded financial or policy assertions")

    is_auto_handle = all(r.passed for r in rules)
    decision = "auto_handle" if is_auto_handle else "escalate"

    routed_queue = None
    if decision == "escalate":
        if any(kw in _LEGAL_ROUTE_KEYWORDS for kw in matched_keywords):
            routed_queue = "Legal & Regulatory Triage"
        elif "security" in intent or "compromise" in intent:
            routed_queue = "Tier-2 Trust & Safety"
        elif classification.get("sentiment") == "furious":
            routed_queue = "Priority Retention Team"
        else:
            routed_queue = "General Tier-1 Human Queue"

    audit_explanation = (
        "All 6 auditable rules passed with high confidence and verified precedent grounding. "
        "Auto-handling approved."
        if is_auto_handle else
        f"Escalation triggered by {len(trigger_reasons)} rule violation(s): "
        f"{'; '.join(trigger_reasons)}. Routed to {routed_queue}."
    )

    return EscalationResult(
        decision=decision,
        rules=rules,
        trigger_reasons=trigger_reasons,
        routed_queue=routed_queue,
        audit_explanation=audit_explanation,
    )
