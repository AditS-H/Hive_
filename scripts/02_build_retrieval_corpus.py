"""
subsample_twcs.csv (raw tweet thread rows, both sides) -> retrieval_corpus.csv
((customer_msg, resolution) pairs -- the shape embed_utils.RetrievalIndex needs).

Usage:
    python scripts/02_build_retrieval_corpus.py \\
        --input subsample_twcs.csv --brand SpotifyCares --out retrieval_corpus.csv

Pairs each brand reply with the customer tweet it responded to
(in_response_to_tweet_id -> tweet_id), then drops pairs where the reply is
itself a boilerplate DM-redirect (same DM_PATTERNS regex as
01_discover_brands.py) -- a reply that just says "please DM us" never
discloses the actual resolution, so keeping it would poison retrieval with
"grounding" that isn't actually grounded in anything.
"""
import argparse
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

DM_PATTERNS = re.compile(
    r"\b(?:dm|direct message)\b.{0,40}\b(?:us|me)\b|"
    r"\bplease\s+dm\b|\bsend\s+(?:us\s+)?a\s+dm\b|\bfollow.{0,20}dm\b",
    re.IGNORECASE,
)  # non-capturing groups -- see the same pattern in 01_discover_brands.py


def build_corpus(df: "pd.DataFrame", brand: str) -> "pd.DataFrame":
    df = df.copy()
    df["inbound"] = df["inbound"].astype(bool)
    tweets_by_id = df.set_index("tweet_id")["text"].to_dict()
    brand_replies = df[(~df["inbound"]) & (df["author_id"] == brand)]

    pairs = []
    for _, reply in brand_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        if pd.isna(parent_id):
            continue
        customer_text = tweets_by_id.get(parent_id)
        if not customer_text or not isinstance(customer_text, str):
            continue
        if DM_PATTERNS.search(reply["text"]):
            continue  # boilerplate redirect, not a real resolution
        pairs.append({"customer_msg": customer_text, "resolution": reply["text"]})

    return pd.DataFrame(pairs, columns=["customer_msg", "resolution"]).drop_duplicates()


def main(input_path, brand, out_path):
    df = pd.read_csv(
        input_path,
        dtype={"tweet_id": "Int64", "response_tweet_id": "string", "in_response_to_tweet_id": "Int64"},
    )
    out_df = build_corpus(df, brand)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} (customer_msg, resolution) pairs for {brand} to {out_path}")
    if len(out_df) == 0:
        print("WARNING: 0 pairs -- check --brand matches an author_id exactly (case-sensitive).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="subsample_twcs.csv from scripts/01_discover_brands.py")
    ap.add_argument("--brand", required=True, help="author_id of the selected brand, e.g. SpotifyCares")
    ap.add_argument("--out", default="retrieval_corpus.csv")
    args = ap.parse_args()
    main(args.input, args.brand, args.out)
