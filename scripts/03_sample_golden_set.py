"""
subsample_twcs.csv -> golden_set_template.jsonl (empty-labeled, ready for a
human to fill in). See golden_set.py's module docstring for why this has to
be a human step.

Usage:
    python scripts/03_sample_golden_set.py --input subsample_twcs.csv --n 200 --out golden_set_template.jsonl

After running: open the output file (one JSON object per line) and fill in,
per case:
  - ground_truth_intent      (must be a key in config.INTENT_TAXONOMY)
  - ground_truth_resolution  (what a correct reply should actually say/do)
  - difficulty                easy | medium | hard | adversarial
  - expected_action           auto_handle | escalate
  - risk_factors               free-form list of tags, can stay empty

Then validate before the first eval run:
    python -c "from golden_set import load_golden_set, validate_golden_set; \\
               from config import INTENT_TAXONOMY; \\
               p = validate_golden_set(load_golden_set('golden_set_template.jsonl'), INTENT_TAXONOMY); \\
               print(p or 'clean')"
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from golden_set import sample_for_labeling, save_golden_set, GoldenCase
from config import INTENT_TAXONOMY


def main(input_path, out_path, n):
    df = pd.read_csv(input_path, dtype={"tweet_id": "Int64"})
    df["inbound"] = df["inbound"].astype(bool)
    customer_messages = df[df["inbound"]]["text"].dropna().tolist()
    print(f"Loaded {len(customer_messages)} raw customer messages from {input_path}")

    templates = sample_for_labeling(customer_messages, n=n)

    # write directly as JSONL (not via save_golden_set, since these are
    # intentionally incomplete dicts, not valid GoldenCase instances yet)
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        for row in templates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(templates)} cases to {out_path} -- open it and fill in the label fields per case.")
    print(f"Target size per the architecture spec is 150-250; you asked for {n}.")
    print("Valid intents:", ", ".join(INTENT_TAXONOMY.keys()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="subsample_twcs.csv from scripts/01_discover_brands.py")
    ap.add_argument("--out", default="golden_set_template.jsonl")
    ap.add_argument("--n", type=int, default=200, help="target golden set size (architecture target: 150-250)")
    args = ap.parse_args()
    main(args.input, args.out, args.n)
