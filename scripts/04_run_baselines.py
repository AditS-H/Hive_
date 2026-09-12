"""
Run the two comparison baselines (majority-class, TF-IDF+LogReg) against a
labeled golden set via stratified k-fold CV. See baselines.py for why CV
over the golden set, not a separate training corpus.

Usage:
    python scripts/04_run_baselines.py --golden_set golden_set.jsonl
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from golden_set import load_golden_set, validate_golden_set
from baselines import run_all_baselines
from config import INTENT_TAXONOMY


def main(golden_set_path, n_splits, seed):
    cases = load_golden_set(golden_set_path)
    problems = validate_golden_set(cases, INTENT_TAXONOMY)
    if problems:
        print(f"Golden set has {len(problems)} problem(s) -- fix these first:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)

    print(f"Loaded {len(cases)} golden cases from {golden_set_path}\n")
    for result in run_all_baselines(cases, n_splits=n_splits, seed=seed):
        m = result.metrics
        fold_kind = "stratified" if result.stratified else "unstratified (a class had < n_splits examples)"
        print(f"{result.name}  [{fold_kind}, {result.n_splits}-fold CV, n={m.n}]")
        print(f"  accuracy={m.accuracy:.3f}  macro_f1={m.macro_f1:.3f}  "
              f"precision={m.precision:.3f}  recall={m.recall:.3f}  cohen_kappa={m.cohen_kappa:.3f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden_set", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.golden_set, args.n_splits, args.seed)
