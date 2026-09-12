"""
Single source of truth for the two things that change behaviour without
changing code: the intent taxonomy and the escalation policy. Every other
module imports from here instead of hardcoding either one.

INTENT TAXONOMY
----------------
Generalized across the candidate brands in `01_discover_brands.py`'s top-10
(SpotifyCares / AppleSupport / AmazonHelp / Delta / Uber_Support, etc.) rather
than tuned to one brand's product surface, because brand selection is still
open (see README "Status"). Four categories are brand-agnostic by nature and
drive escalation directly: billing, login/auth, security breach, legal
threat. The remaining four are deliberately generic containers
(product_issue_or_bug, subscription_or_plan_change, order_or_delivery_status,
how_to_or_feature_question) that read naturally for a media app, an
e-commerce brand, an airline, or a rideshare app alike.

Once brand selection is final: open `subsample_twcs.csv` for the winning
brand, read ~50 real customer messages, and rename/split 1-2 of the generic
categories to match that brand's actual vocabulary (e.g.
product_issue_or_bug -> flight_delay_or_baggage for Delta). That is a change
in this one file; nothing downstream (classify_intent, escalation, eval)
needs to know the category names changed.

ESCALATION POLICY
------------------
Mirrors the reference architecture's `escalationRules.ts` `DEFAULT_ESCALATION_CONFIG`
value-for-value, so the Python and TypeScript versions are provably the same
policy. See escalation.py for the six rules that read this config.

One deliberate deviation from the reference: its `mandatoryEscalationIntents`
lists `vip_press_inquiry` and `billing_chargeback`, neither of which is a
defined entry in its own INTENT_TAXONOMY -- those two strings could never
match a real classification and were dead config. `billing_chargeback` is
redundant anyway: "chargeback" is already in `sensitive_keywords`, so that
signal is caught either way. Dropped both rather than carry over a rule that
can never fire.
"""
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

INTENT_TAXONOMY: dict[str, dict] = {
    "billing_refund_dispute": {
        "description": "Double charge, unrecognized charge or renewal, refund request, or invoice discrepancy.",
        "risk_level": "medium",
        "requires_human_review": False,
    },
    "account_login_auth": {
        "description": "Password reset, 2FA issues, unable to log in, or email/identity update on the account.",
        "risk_level": "medium",
        "requires_human_review": False,
    },
    "product_issue_or_bug": {
        "description": "App crash, broken feature, sync failure, or any other defect in the product itself.",
        "risk_level": "low",
        "requires_human_review": False,
    },
    "subscription_or_plan_change": {
        "description": "Cancel, upgrade, downgrade, pause, or otherwise change a subscription or plan tier.",
        "risk_level": "low",
        "requires_human_review": False,
    },
    "order_or_delivery_status": {
        "description": "Where an order, shipment, booking, or ride is, or a complaint that it did not arrive as expected.",
        "risk_level": "low",
        "requires_human_review": False,
    },
    "how_to_or_feature_question": {
        "description": "Customer wants to know how to use a feature that already exists and works as designed.",
        "risk_level": "low",
        "requires_human_review": False,
    },
    "account_security_breach": {
        "description": "Suspicious logins, unauthorized email/password change, or other signs of account takeover.",
        "risk_level": "high",
        "requires_human_review": True,
    },
    "legal_regulatory_threat": {
        "description": "Lawyer involvement, regulatory complaint (FTC/state AG/similar), or formal legal demand.",
        "risk_level": "high",
        "requires_human_review": True,
    },
}


# ---------------------------------------------------------------------------
# Escalation policy
# ---------------------------------------------------------------------------

@dataclass
class EscalationConfig:
    confidence_threshold: float = 0.76
    min_precedent_similarity: float = 0.58
    disallow_furious_sentiment: bool = True
    sensitive_keywords: list[str] = field(default_factory=lambda: [
        "lawyer", "attorney", "sue", "lawsuit", "chargeback", "fraud",
        "scam", "police", "stolen", "regulator", "court", "ftc",
        "attorney general", "illegal",
    ])
    mandatory_escalation_intents: list[str] = field(default_factory=lambda: [
        "account_security_breach", "legal_regulatory_threat",
    ])


DEFAULT_ESCALATION_CONFIG = EscalationConfig()


# ---------------------------------------------------------------------------
# Brand profile
# ---------------------------------------------------------------------------
# `01_discover_brands.py` gives quantitative selection evidence (reply
# volume, boilerplate-DM rate, followup rate) but NOT voice/tone -- that's a
# qualitative read of real replies a human has to do once. DEMO_BRAND below
# is a clearly-labeled placeholder so the pipeline is runnable today; swap in
# the real values after brand selection (see data/README.md step 2).

@dataclass
class Brand:
    id: str
    name: str
    handle: str
    tone_description: str
    agent_signoff: str


DEMO_BRAND = Brand(
    id="demo_brand",
    name="Demo Brand Support",
    handle="@DemoBrandCares",
    tone_description="Friendly, concise, one concrete next step per reply.",
    agent_signoff="^Demo",
)
