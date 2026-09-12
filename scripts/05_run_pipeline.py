"""
Run the five-step pipeline (classify -> retrieve -> draft -> escalate ->
log) against one message, or interactively against a stream of them. The
CLI equivalent of the reference architecture's "Live Playground" tab.

Usage:
    # one message, real Gemini/Groq calls (needs GEMINI_API_KEY in .env)
    python scripts/05_run_pipeline.py --corpus retrieval_corpus.csv --message "My app keeps crashing"

    # interactive REPL, real calls
    python scripts/05_run_pipeline.py --corpus retrieval_corpus.csv --interactive

    # smoke-test the wiring with no API keys and no network
    python scripts/05_run_pipeline.py --corpus retrieval_corpus.csv --message "..." --fake-llm
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from embed_utils import RetrievalIndex
from pipeline import run_pipeline
from config import DEMO_BRAND


def _load_retriever(corpus_path):
    df = pd.read_csv(corpus_path)
    return RetrievalIndex(df["customer_msg"].tolist(), df["resolution"].tolist())


def _print_execution(exec_):
    print(json.dumps({
        "classification": exec_.classification,
        "retrieval": {
            "top_similarity": round(exec_.retrieval["top_similarity"], 3),
            "latency_ms": exec_.retrieval["latency_ms"],
            "precedents": [
                {"similarity": round(p["similarity"], 3), "customer_msg": p["customer_msg"][:80]}
                for p in exec_.retrieval["precedents"]
            ],
        },
        "draft": exec_.draft,
        "escalation": {
            "decision": exec_.escalation.decision,
            "routed_queue": exec_.escalation.routed_queue,
            "trigger_reasons": exec_.escalation.trigger_reasons,
            "audit_explanation": exec_.escalation.audit_explanation,
        },
        "total_latency_ms": exec_.total_latency_ms,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="retrieval_corpus.csv from scripts/02_build_retrieval_corpus.py")
    ap.add_argument("--message", help="a single message to run through the pipeline")
    ap.add_argument("--author", default="@customer")
    ap.add_argument("--interactive", action="store_true", help="read messages from stdin, one per line, until EOF")
    ap.add_argument("--fake-llm", action="store_true",
                     help="use demo_fakes instead of real Gemini/Groq calls (no API key or network needed)")
    args = ap.parse_args()

    retriever = _load_retriever(args.corpus)

    kwargs = {}
    if args.fake_llm:
        from demo_fakes import fake_classify_intent, fake_draft_reply
        kwargs["classify_fn"] = fake_classify_intent
        kwargs["draft_fn"] = fake_draft_reply

    if args.interactive:
        print("Type a customer message and press enter (Ctrl-D to quit).")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            exec_ = run_pipeline(line, retriever, author=args.author, brand=DEMO_BRAND, **kwargs)
            _print_execution(exec_)
    elif args.message:
        exec_ = run_pipeline(args.message, retriever, author=args.author, brand=DEMO_BRAND, **kwargs)
        _print_execution(exec_)
    else:
        ap.error('pass --message "..." or --interactive')


if __name__ == "__main__":
    main()
