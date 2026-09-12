"""
Run the full offline evaluation: the production pipeline + LLM judge across
the golden set, plus both baselines, into one report -- printed as the
comparison table and saved as JSON.

Usage:
    # real Gemini/Groq calls (needs both API keys in .env)
    python scripts/06_run_eval.py --corpus retrieval_corpus.csv --golden_set golden_set.jsonl --out eval_report.json

    # smoke-test the wiring with no API keys and no network
    python scripts/06_run_eval.py --corpus retrieval_corpus.csv --golden_set golden_set.jsonl --fake-llm
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from embed_utils import RetrievalIndex
from golden_set import load_golden_set, validate_golden_set
from eval_harness import run_eval, format_comparison_table, save_report
from config import INTENT_TAXONOMY, DEMO_BRAND


def main(corpus_path, golden_set_path, out_path, sample_limit, fake_llm):
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    corpus_df = pd.read_csv(corpus_path)
    retriever = RetrievalIndex(corpus_df["customer_msg"].tolist(), corpus_df["resolution"].tolist())

    cases = load_golden_set(golden_set_path)
    problems = validate_golden_set(cases, INTENT_TAXONOMY)
    if problems:
        print(f"Golden set has {len(problems)} problem(s) -- fix these before running eval:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)

    if sample_limit:
        cases = cases[:sample_limit]

    kwargs = {}
    if fake_llm:
        from demo_fakes import fake_classify_intent, fake_draft_reply, fake_judge_reply
        kwargs = {"classify_fn": fake_classify_intent, "draft_fn": fake_draft_reply, "judge_fn": fake_judge_reply}

    print(f"Running pipeline + judge over {len(cases)} golden cases (brand: {DEMO_BRAND.name})"
          f"{' [fake-llm]' if fake_llm else ''}...")
    report = run_eval(cases, retriever, brand=DEMO_BRAND, **kwargs)

    print()
    print(format_comparison_table(report))
    print()
    print(f"Hallucination rate: {report.hallucination_rate * 100:.1f}%   "
          f"Auto-handle rate: {report.auto_handle_rate * 100:.1f}%   "
          f"Escalation accuracy: {report.escalation_accuracy * 100:.1f}%   "
          f"Avg latency: {report.avg_latency_ms:.0f}ms")

    save_report(report, out_path)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--golden_set", required=True)
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument("--sample_limit", type=int, default=None,
                     help="evaluate only the first N cases (useful for a quick check)")
    ap.add_argument("--fake-llm", action="store_true",
                     help="use demo_fakes instead of real Gemini/Groq calls (no API key or network needed)")
    args = ap.parse_args()
    main(args.corpus, args.golden_set, args.out, args.sample_limit, args.fake_llm)
