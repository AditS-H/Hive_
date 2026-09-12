"""
Run locally (Claude's sandbox can't reach kaggle.com):

    pip install kaggle pandas
    kaggle datasets download -d thoughtvector/customer-support-on-twitter
    unzip customer-support-on-twitter.zip -d data_raw/
    python scripts/01_discover_brands.py --input data_raw/twcs.csv --out_dir .

Produces two files to upload back:
  - brand_candidates_summary.csv  (few KB)  -> the brand-selection evidence
  - subsample_twcs.csv            (few MB)  -> capped at 20k rows/brand, top-10 brands only

Why these two metrics, not just volume:
  - boilerplate_dm_rate: brand replies that just redirect to DM ("please DM your
    order #") never show the actual resolution in-thread. High rate = bad
    grounding material, even if the brand has huge tweet volume.
  - followup_rate: fraction of brand replies that get another customer reply
    in the same thread. High rate can mean the first reply didn't close the
    loop. Not disqualifying alone, but combined with high DM-rate it's a
    brand where "resolution" mostly happens off-platform -- exactly what we
    can't ground an LLM reply on.

Lower on both = more real, in-thread, resolvable conversations for this brand.
"""
import argparse
import re
import os
import pandas as pd

DM_PATTERNS = re.compile(
    r"\b(?:dm|direct message)\b.{0,40}\b(?:us|me)\b|"
    r"\bplease\s+dm\b|\bsend\s+(?:us\s+)?a\s+dm\b|\bfollow.{0,20}dm\b",
    re.IGNORECASE,
)  # non-capturing groups: pandas' .str.contains() below warns on capturing
   # groups in a match-only (not extract) context -- these are grouped only
   # for alternation, so (?:...) removes the warning with no behavior change


def main(input_path: str, out_dir: str, top_n_brands: int, cap_per_brand: int):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading {input_path} ...")
    df = pd.read_csv(
        input_path,
        dtype={"tweet_id": "Int64", "response_tweet_id": "string", "in_response_to_tweet_id": "Int64"},
    )
    df["inbound"] = df["inbound"].astype(bool)

    company_replies = df[~df["inbound"]]
    top_brands = company_replies["author_id"].value_counts().head(top_n_brands).index.tolist()
    print(f"Top {top_n_brands} brands by reply volume: {top_brands}")

    # customer tweet_id -> set of tweet_ids that reply to it (used for followup check)
    replies_to = df.dropna(subset=["in_response_to_tweet_id"]).groupby("in_response_to_tweet_id")["tweet_id"].apply(set)

    rows = []
    for brand in top_brands:
        brand_replies = company_replies[company_replies["author_id"] == brand]
        n = len(brand_replies)

        dm_hits = brand_replies["text"].str.contains(DM_PATTERNS, na=False).sum()
        boilerplate_dm_rate = dm_hits / n if n else 0.0

        # a brand reply gets a "followup" if the customer tweet it replied to
        # has ANOTHER response besides this one -> the loop wasn't closed here
        followups = 0
        checked = 0
        for _, r in brand_replies.iterrows():
            parent = r["in_response_to_tweet_id"]
            if pd.isna(parent):
                continue
            checked += 1
            siblings = replies_to.get(parent, set())
            if len(siblings) > 1:
                followups += 1
        followup_rate = followups / checked if checked else 0.0

        composite_score = boilerplate_dm_rate + followup_rate  # lower = better grounding material
        rows.append({
            "brand": brand,
            "total_replies": n,
            "boilerplate_dm_rate": round(boilerplate_dm_rate, 3),
            "followup_rate": round(followup_rate, 3),
            "composite_score": round(composite_score, 3),
        })

    summary = pd.DataFrame(rows).sort_values("composite_score")
    summary_path = f"{out_dir}/brand_candidates_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}:\n{summary.to_string(index=False)}")

    # full threads (both sides) for top brands, capped per brand for a manageable upload
    capped_frames = []
    for brand in top_brands:
        brand_rows = company_replies[company_replies["author_id"] == brand].head(cap_per_brand)
        parent_ids = brand_rows["in_response_to_tweet_id"].dropna().unique()
        customer_rows = df[df["tweet_id"].isin(parent_ids)]
        capped_frames.append(brand_rows)
        capped_frames.append(customer_rows)
    subsample = pd.concat(capped_frames).drop_duplicates(subset="tweet_id")

    subsample_path = f"{out_dir}/subsample_twcs.csv"
    subsample.to_csv(subsample_path, index=False)
    print(f"Wrote {subsample_path}: {len(subsample)} rows across {len(top_brands)} brands")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--top_n_brands", type=int, default=10)
    ap.add_argument("--cap_per_brand", type=int, default=20000)
    args = ap.parse_args()
    main(args.input, args.out_dir, args.top_n_brands, args.cap_per_brand)
