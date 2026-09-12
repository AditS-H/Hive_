"""
Two providers, two jobs. No abstraction beyond what's needed to swap a model id.

  - GEMINI  -> classification + reply drafting (the agent's own outputs)
  - GROQ    -> LLM-as-judge only (deliberately a different model family, so the
               judge isn't scoring a model's own style/blind-spots as if they
               were quality)

Both are free-tier, no-credit-card APIs as of Sept 2026. Check current rate
limits before a big run:
  - Gemini: https://ai.google.dev/gemini-api/docs/rate-limits
  - Groq:   https://console.groq.com/docs/rate-limits

If either free tier gets stingier by the time you run this, drop the model id
in ONE place below -- nothing downstream needs to change. Both ids below were
verified live against each provider's own docs on 2026-09-12:
  - gemini-2.5-flash -> gemini-3.8-flash: 2.5 isn't gone, but 3.8 is the
    current GA Flash model (ai.google.dev/gemini-api/docs/latest-model) and
    the one the reference architecture already assumes.
  - llama-3.3-70b-versatile -> openai/gpt-oss-120b: the old id was announced
    deprecated by Groq on 2026-06-17 and fully decommissioned 2026-08-16
    (console.groq.com/docs/deprecations) -- calls to it now fail outright.
    gpt-oss-120b is Groq's own suggested replacement and keeps the
    different-family property intact (OpenAI open-weights vs. Google Gemini).
"""
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-3.8-flash"          # GA as of 2026-09-02; check ai.google.dev/gemini-api/docs/models before a big run
GROQ_JUDGE_MODEL = "openai/gpt-oss-120b"   # different family than Gemini; llama-3.3-70b-versatile (former default) is decommissioned

_gemini_client = None
_groq_client = None


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def _groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def _retry(fn, tries=4, base_delay=2.0):
    """Free tiers rate-limit hard. Back off instead of failing a whole batch run."""
    last_err = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))
    raise last_err


def classify_intent(message: str, taxonomy: dict) -> dict:
    """
    taxonomy: {"billing_refund_dispute": {"description": "...", ...}, ...}
              (config.INTENT_TAXONOMY's shape -- only "description" is used
              here; risk_level/requires_human_review are read by escalation.py)
    Returns {"intent": str, "confidence": float, "reasoning": str,
             "sentiment": "positive"|"neutral"|"frustrated"|"furious",
             "urgency": "low"|"medium"|"high"|"critical"}

    sentiment and urgency are not decoration -- escalation.py's distress rule
    (rule 4) reads classification["sentiment"] and ["urgency"] directly, so
    both have to come back on every call, not just intent/confidence.
    """
    labels = list(taxonomy.keys())
    descriptions = {k: v["description"] if isinstance(v, dict) else v for k, v in taxonomy.items()}
    prompt = f"""Classify this customer support message into exactly one intent.
Also assess the customer's sentiment and the urgency of their message.

Intents:
{json.dumps(descriptions, indent=2)}

Message: "{message}"

Return your best-guess confidence honestly -- do not default to a high number."""

    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": labels},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "frustrated", "furious"]},
            "urgency": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        },
        "required": ["intent", "confidence", "reasoning", "sentiment", "urgency"],
    }

    def call():
        resp = _gemini().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0,
            },
        )
        return json.loads(resp.text)

    return _retry(call)


def draft_reply(
    message: str,
    intent: str,
    retrieved_cases: list[dict],
    brand_name: str = "the brand",
    tone_description: str = "",
    signoff: str = "",
) -> str:
    """
    retrieved_cases: [{"customer_msg": ..., "resolution": ..., "similarity": 0.83}, ...]
    brand_name/tone_description/signoff: optional brand voice (config.Brand);
        omit them and you get a generic-but-still-grounded reply, which is
        why they default to "" rather than being required.
    Grounding is explicit: the model is told not to assert anything not present
    in the retrieved precedent, because invented specifics (refund amounts,
    policy claims) are the worst failure mode in this domain.
    """
    context = "\n\n".join(
        f"Precedent {i+1} (similarity {c['similarity']:.2f}):\n"
        f"Customer: {c['customer_msg']}\nBrand resolution: {c['resolution']}"
        for i, c in enumerate(retrieved_cases)
    ) or "No sufficiently similar precedent was found."

    voice_lines = []
    if tone_description:
        voice_lines.append(f"- Match this brand's voice: {tone_description}")
    if signoff:
        voice_lines.append(f'- End the reply with: "{signoff}"')
    voice_block = ("\n" + "\n".join(voice_lines)) if voice_lines else ""

    prompt = f"""You are drafting a Twitter/X reply on behalf of {brand_name}'s support account, in their voice.

Customer intent: {intent}
Customer message: "{message}"

Historical precedent for similar cases:
{context}

Rules:
- Match the tone and structure of the precedent replies.{voice_block}
- Do NOT state specific facts (amounts, dates, policy terms, order status) unless
  they appear in the precedent above. If you don't have the specific, ask for it
  or give a generic next step instead of inventing one.
- If no precedent is relevant, write a short reply that asks a clarifying
  question or routes to the standard next step -- don't improvise a resolution.
- Keep it under 280 characters if you reasonably can; never pad length for its own sake."""

    def call():
        resp = _gemini().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.3},
        )
        return resp.text.strip()

    return _retry(call)


def judge_reply(message: str, intent: str, draft: str, retrieved_cases: list[dict]) -> dict:
    """
    Runs on a DIFFERENT model family (Groq/Llama) than the generator (Gemini),
    on purpose -- see module docstring. Only called on the golden set / dev
    diagnostics, never in the production per-message path, so free-tier volume
    is not a concern.
    Returns groundedness / tone / resolution / hallucination scores (1-5) + notes.
    """
    context = "\n".join(f"- {c['customer_msg']} -> {c['resolution']}" for c in retrieved_cases) or "(none)"

    prompt = f"""Score this drafted support reply. Be strict, not generous.

Customer message: "{message}"
Intent: {intent}
Available precedent:
{context}

Drafted reply: "{draft}"

Score each 1-5 (5 = best):
- groundedness: does it avoid asserting facts not in the precedent?
- tone_match: does it read like real brand support (not generic AI voice)?
- resolution_quality: does it actually move the customer's issue forward?
- hallucination_flag: 1 if it invents a specific fact not in precedent, else 0

Return ONLY JSON: {{"groundedness": int, "tone_match": int, "resolution_quality": int, "hallucination_flag": 0 or 1, "notes": "one sentence"}}"""

    def call():
        resp = _groq().chat.completions.create(
            model=GROQ_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    return _retry(call)
