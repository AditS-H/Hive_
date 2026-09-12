"""
The golden set is a held-out, HUMAN-labeled benchmark (target: 150-250
cases, per the reference architecture's "Audit Verified" golden_set node) --
that's not something this pipeline can generate for itself, by design: if
the same process that drafts replies also invented its own ground truth,
the eval would just measure self-agreement. This file provides the
infrastructure around that human step, not a substitute for it:

  - GoldenCase: the schema one labeled case has to match.
  - load/save: JSONL in, JSONL out.
  - validate: catch labeling typos (an intent that doesn't exist in the
    taxonomy, a stray difficulty value) before they silently break a report.
  - sample_for_labeling: turn raw candidate customer messages (from
    subsample_twcs.csv) into an empty-labeled template a human fills in --
    stratified by message length so the sample isn't accidentally all short,
    easy tweets.
"""
import json
import random
from dataclasses import dataclass, asdict, field

VALID_DIFFICULTIES = {"easy", "medium", "hard", "adversarial"}
VALID_ACTIONS = {"auto_handle", "escalate"}


@dataclass
class GoldenCase:
    id: str
    customer_msg: str
    ground_truth_intent: str
    ground_truth_resolution: str
    difficulty: str  # one of VALID_DIFFICULTIES
    expected_action: str  # one of VALID_ACTIONS
    risk_factors: list[str] = field(default_factory=list)


def save_golden_set(cases: list[GoldenCase], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")


def load_golden_set(path: str) -> list[GoldenCase]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(GoldenCase(**json.loads(line)))
    return cases


def validate_golden_set(cases: list[GoldenCase], taxonomy: dict) -> list[str]:
    """Returns a list of human-readable problems; empty list means clean."""
    problems = []
    seen_ids = set()
    valid_intents = set(taxonomy.keys())

    for case in cases:
        if case.id in seen_ids:
            problems.append(f"{case.id}: duplicate id")
        seen_ids.add(case.id)

        if case.ground_truth_intent not in valid_intents:
            problems.append(f"{case.id}: ground_truth_intent '{case.ground_truth_intent}' is not in the taxonomy")
        if case.difficulty not in VALID_DIFFICULTIES:
            problems.append(f"{case.id}: difficulty '{case.difficulty}' not in {sorted(VALID_DIFFICULTIES)}")
        if case.expected_action not in VALID_ACTIONS:
            problems.append(f"{case.id}: expected_action '{case.expected_action}' not in {sorted(VALID_ACTIONS)}")
        if not case.customer_msg.strip():
            problems.append(f"{case.id}: empty customer_msg")
        if not case.ground_truth_resolution.strip():
            problems.append(f"{case.id}: empty ground_truth_resolution")

    return problems


def sample_for_labeling(messages: list[str], n: int, seed: int = 42) -> list[dict]:
    """
    Stratified by message-length tercile (short/medium/long) as a cheap,
    content-free proxy for conversational complexity -- guards against an
    accidentally skewed sample (e.g. all short "thanks!" replies) without
    presuming to know the taxonomy or outcome ahead of the human labeling
    pass. Returns plain dicts (not GoldenCase) with the label fields empty,
    ready to write to JSONL/CSV for a human to fill in.
    """
    unique = list(dict.fromkeys(m.strip() for m in messages if m and m.strip()))
    if not unique:
        return []

    rng = random.Random(seed)
    by_length = sorted(unique, key=len)
    thirds = len(by_length) // 3 or 1
    buckets = [
        by_length[:thirds],
        by_length[thirds:2 * thirds],
        by_length[2 * thirds:],
    ]
    buckets = [b for b in buckets if b]

    per_bucket = max(1, n // len(buckets))
    sampled: list[str] = []
    for bucket in buckets:
        take = min(per_bucket, len(bucket))
        sampled.extend(rng.sample(bucket, take))

    # top up from the full pool (minus what's already picked) if buckets
    # were uneven and we're still short of n
    remaining_pool = [m for m in unique if m not in sampled]
    if len(sampled) < n and remaining_pool:
        top_up = min(n - len(sampled), len(remaining_pool))
        sampled.extend(rng.sample(remaining_pool, top_up))

    rng.shuffle(sampled)
    sampled = sampled[:n]

    return [
        {
            "id": f"gold_{i + 1:03d}",
            "customer_msg": msg,
            "ground_truth_intent": "",
            "ground_truth_resolution": "",
            "difficulty": "",
            "expected_action": "",
            "risk_factors": [],
        }
        for i, msg in enumerate(sampled)
    ]
