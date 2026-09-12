# Report: Customer Support AI Agent Pipeline

## Status

| Deliverable | Status |
|---|---|
| Brand selection | Script ready (`scripts/01_discover_brands.py`). Blocked on the real dataset -- see below. |
| Intent taxonomy | Done (`config.py`) |
| Baselines (majority-class, TF-IDF+LogReg) | Done (`baselines.py`) |
| Agent pipeline (classify -> retrieve -> draft -> escalate) | Done (`pipeline.py`, `escalation.py`) |
| Golden set + judge + eval harness | Infrastructure done (`golden_set.py`, `eval_harness.py`). Real 150-250 case labeled set blocked on you -- see below. |
| Report + decision log | This file + `decision_log.md` |

**Two things this sandbox genuinely cannot do**, both explained in full in
`decision_log.md`: reach `kaggle.com` for the real dataset, and reach
`generativelanguage.googleapis.com` / `console.groq.com` for real model
calls (also no API keys were provided). Every module is built and tested
against that reality -- see "Smoke test" below for what *was* verified,
and "Next steps" for exactly what turns this into a real evaluation.

## Architecture

```
OFFLINE PREP (once per brand)                    PER-MESSAGE PIPELINE (live)
  brand subsample                                   message
  -> retrieval corpus                               -> classify (Gemini, +sentiment/urgency)
  -> golden set (human-labeled, 150-250)             -> retrieve (local cosine, top-3, >=0.58 sim)
                                                      -> draft (Gemini, grounded, brand voice)
OFFLINE EVAL (golden set only)                       -> escalate (6 deterministic rules)
  golden-set drafts (same pipeline)                  -> auto_handle | escalate + routed queue
  -> judge (Groq, cross-family)
  -> metrics (accuracy, macro F1, P/R, kappa,
              hallucination rate) vs. both baselines
```

Ported from the reference architecture's own diagram
(`src/components/ArchitectureDiagram.tsx`) 1:1 in structure; every box
above is a real Python module, not a UI mockup box. See `decision_log.md`
for where the Python version corrects or extends the reference rather than
copying it verbatim, and why.

## Escalation policy

Six rules, all-must-pass for auto-handling: intent confidence >= 76%,
precedent grounding >= 58% similarity, no high-risk/legal keywords, no
severe distress (furious sentiment or critical urgency), intent not on the
mandatory-human-review list, no invented dollar figures not present in
precedent. Every decision is traceable to which specific rule(s) fired --
see any `case_results[i].pipeline_result.escalation.rules` entry in
`eval_report_demo.json` for the full audit trail format.

## Smoke test (synthetic data, today's run -- NOT the real evaluation)

Ran the actual, unmodified pipeline end to end against fabricated data for
two fictional brands (`data/demo/`), with `--fake-llm` swapping in
deterministic keyword-rule stand-ins for `classify_intent`/`draft_reply`/
`judge_reply` (no API keys, no network calls). This proves the wiring --
every module actually calling the next one correctly, in the right order,
with the right data shapes -- not model quality. Reproduce with:

```
python scripts/01_discover_brands.py --input data/demo/subsample_demo.csv --out_dir /tmp/demo --top_n_brands 2
python scripts/06_run_eval.py --corpus data/demo/retrieval_corpus_demo.csv --golden_set data/demo/golden_set_demo.jsonl --fake-llm
```

Brand selection on the synthetic data correctly separated the two
fictional brands: `DemoBrandCares` (composite score 0.0 -- no boilerplate
DM-redirects, real in-thread resolutions) vs. `OtherBrandHelp` (composite
score 1.0 -- 100% boilerplate). That's the mechanism working, not a
finding about real brands.

Actual output from the eval run against the 12-case synthetic golden set:

| Model | Accuracy | Macro F1 | Precision | Recall | Cohen's kappa | Hallucination rate |
|---|---|---|---|---|---|---|
| Majority Class | 16.7% | 0.056 | 0.042 | 0.083 | -0.111 | N/A |
| TF-IDF + Logistic Regression | 16.7% | 0.062 | 0.050 | 0.083 | -0.017 | N/A |
| Fake-LLM Agent Pipeline | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 | 0.0% |
| Auto-handle rate | 16.7% | | Escalation accuracy | 41.7% | | |

**Read this table as "the harness computes real, correct metrics," not
"the agent is perfect."** Three specific artifacts, all explained in full
in `decision_log.md`:

1. The two baselines score near chance because 12 examples spread across 8
   categories leaves almost nothing for 5-fold CV to learn from per fold.
   The real golden set (150-250 cases) will have a few dozen examples per
   category -- meaningfully more signal.
2. 100% agent accuracy is a deterministic keyword classifier scoring
   keyword-matchable text I wrote myself -- expected to be near-perfect by
   construction, and *not* a claim about real Gemini's accuracy on real,
   messy customer tweets.
3. The 16.7% auto-handle rate and 41.7% escalation accuracy are lower than
   the golden set's own designed 9-auto/3-escalate split because this
   sandbox can't install real `sentence-transformers` (no route to
   huggingface.co, insufficient disk for torch) -- a crude word-hash
   stand-in encoder was used for retrieval, which underestimates true
   semantic similarity vs. the real MiniLM model. This affects only *my*
   verification environment: the identical `--fake-llm` command on a real
   machine (real `sentence-transformers` per `requirements.txt`) uses real
   embeddings for retrieval, since `--fake-llm` only swaps
   classify/draft/judge, never the retriever.

Full machine-readable output: `eval_report_demo.json` alongside this file.

## Results (real evaluation -- fill in after the steps below)

*(Not filled in with placeholder numbers on purpose -- see `decision_log.md`
on why fabricated-looking numbers in a report are worse than an honest gap.)*

| Model | Accuracy | Macro F1 | Precision | Recall | Cohen's kappa | Hallucination rate |
|---|---|---|---|---|---|---|
| Majority Class | | | | | | N/A |
| TF-IDF + Logistic Regression | | | | | | N/A |
| Gemini Grounded Agent Pipeline | | | | | | |

Auto-handle rate: __   Escalation accuracy: __   Avg latency: __ ms

## Next steps to get real numbers

1. `python scripts/01_discover_brands.py --input <real twcs.csv> --out_dir data/` (needs the real Kaggle dataset -- `data/README.md` has the download commands)
2. Read `data/brand_candidates_summary.csv`, pick the winning brand, fill in a real `config.Brand` (currently `DEMO_BRAND`)
3. `python scripts/02_build_retrieval_corpus.py --input data/subsample_twcs.csv --brand <BRAND>`
4. `python scripts/03_sample_golden_set.py --input data/subsample_twcs.csv --n 200`, then hand-label 150-250 cases
5. Add `GEMINI_API_KEY` / `GROQ_API_KEY` to `.env`
6. `python scripts/04_run_baselines.py --golden_set data/golden_set.jsonl`
7. `python scripts/06_run_eval.py --corpus data/retrieval_corpus.csv --golden_set data/golden_set.jsonl --out eval_report.json` (no `--fake-llm`)
8. Copy the printed comparison table into the "Results" section above
