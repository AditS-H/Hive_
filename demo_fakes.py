"""
Deterministic stand-ins for llm_clients.classify_intent / draft_reply /
judge_reply. These are NOT a quality approximation of Gemini or Groq --
they're keyword rules, nothing more -- and exist only to exercise the
pipeline's wiring (data flowing correctly step to step, escalation rules
firing on the right inputs, metrics computing correctly end to end) without
spending API quota or requiring GEMINI_API_KEY/GROQ_API_KEY to be set.

Swap in with --fake-llm on scripts/05_run_pipeline.py and
scripts/06_run_eval.py. Any real evaluation number always requires the real
classify_intent/draft_reply/judge_reply -- see README "Status" for what's
still gated on that.
"""
import re

from config import INTENT_TAXONOMY

_KEYWORD_RULES = [
    (re.compile(r"\b(lawyer|attorney|sue|lawsuit|court|ftc|regulator)\b", re.I), "legal_regulatory_threat"),
    (re.compile(r"\b(hack(ed)?|compromise|unauthorized|stolen|takeover)\b", re.I), "account_security_breach"),
    (re.compile(r"\b(charge(d)?|refund(ed)?|bill(ed|ing)?|invoice)\b", re.I), "billing_refund_dispute"),
    (re.compile(r"\b(password|log ?in|2fa|locked out|authentication)\b", re.I), "account_login_auth"),
    (re.compile(r"\b(cancel|subscription|upgrade|downgrade|plan)\b", re.I), "subscription_or_plan_change"),
    (re.compile(r"\b(order|package|shipment|delivery|delivered|ride|booking)\b", re.I), "order_or_delivery_status"),
    (re.compile(r"\b(crash(ed|es|ing)?|bug|broken|broke|freeze(s|ing)?|glitch)\b", re.I), "product_issue_or_bug"),
]

_FURIOUS_RE = re.compile(r"\b(worst|trash|terrible|unacceptable|furious|scam)\b|!!", re.I)


def fake_classify_intent(message: str, taxonomy: dict = INTENT_TAXONOMY) -> dict:
    intent = "how_to_or_feature_question"
    for pattern, matched_intent in _KEYWORD_RULES:
        if pattern.search(message):
            intent = matched_intent
            break

    is_furious = bool(_FURIOUS_RE.search(message))
    return {
        "intent": intent,
        "confidence": 0.9 if intent != "how_to_or_feature_question" else 0.6,
        "reasoning": f"[fake-llm] keyword rule matched intent={intent}",
        "sentiment": "furious" if is_furious else "neutral",
        "urgency": "critical" if is_furious else "low",
    }


def fake_draft_reply(
    message: str,
    intent: str,
    retrieved_cases: list[dict],
    brand_name: str = "the brand",
    tone_description: str = "",
    signoff: str = "",
) -> str:
    if retrieved_cases:
        base = retrieved_cases[0]["resolution"]
        # historical resolution text already carries its own real sign-off
        # (that's realistic -- past replies really did end that way); only
        # append if this one doesn't already have it, so a naive fake
        # doesn't double it up the way dumb string concatenation would.
        if signoff and signoff not in base:
            return f"{base} {signoff}".strip()
        return base.strip()
    return f"Thanks for reaching out -- could you share a bit more detail so we can help? {signoff}".strip()


def fake_judge_reply(message: str, intent: str, draft: str, retrieved_cases: list[dict]) -> dict:
    has_grounding = len(retrieved_cases) > 0
    return {
        "groundedness": 4 if has_grounding else 2,
        "tone_match": 4,
        "resolution_quality": 4 if has_grounding else 2,
        "hallucination_flag": 0,
        "notes": "[fake-llm] heuristic score based on precedent availability only.",
    }
