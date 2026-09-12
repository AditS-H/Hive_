# Hiver take-home — support agent

## Setup (once)
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed to run tests
cp .env.example .env   # fill in GEMINI_API_KEY and GROQ_API_KEY (both free, no card)
```
- Gemini key: https://aistudio.google.com/apikey
- Groq key: https://console.groq.com/keys

## Status
- [x] Repo scaffold
- [x] LLM layer (`llm_clients.py`) — Gemini for classify/draft, Groq for judge
- [x] Retrieval layer (`embed_utils.py`) — local sentence-transformers, no vector DB
- [x] Intent taxonomy (`config.py`)
- [x] Baselines: majority-class, TF-IDF+LogReg (`baselines.py`)
- [x] Agent pipeline: classify → retrieve → draft → escalate (`pipeline.py`, `escalation.py`)
- [x] Golden set schema + judge + eval harness (`golden_set.py`, `eval_harness.py`)
- [x] Report + decision log (`REPORT.md`, `decision_log.md`)
- [x] 66 tests, all passing (`tests/`) — see "Testing" below for how, given no network access was available while building this
- [ ] Brand selection — **blocked on you**: run the commands below and upload the two outputs
- [ ] Real 150-250 case golden set — **blocked on you**: needs the real brand's data, see `data/README.md`
- [ ] Real eval numbers — **blocked on you**: needs the above plus your own API keys; `REPORT.md` has a synthetic-data smoke test in the meantime

## Next step
```bash
pip install kaggle pandas
kaggle datasets download -d thoughtvector/customer-support-on-twitter
unzip customer-support-on-twitter.zip -d data_raw/
python scripts/01_discover_brands.py --input data_raw/twcs.csv --out_dir data/
```
Full path from there (corpus building, golden-set labeling, running eval) is in `data/README.md`.

## Project structure
```
config.py          intent taxonomy + escalation policy + brand profile (single source of truth)
llm_clients.py      Gemini (classify, draft) + Groq (judge) -- the only network calls in the project
embed_utils.py      local sentence-transformers retrieval index (unmodified from the original scaffold)
escalation.py       six-rule deterministic escalation gate
pipeline.py         classify -> retrieve -> draft -> escalate -> log, one function, five steps
baselines.py        majority-class + TF-IDF/LogReg, cross-validated over the golden set
metrics.py          one metrics implementation, shared by baselines.py and eval_harness.py
golden_set.py       golden-case schema, JSONL I/O, validation, labeling-sample helper
eval_harness.py     runs the pipeline + judge across the golden set, assembles the final report
demo_fakes.py       deterministic keyword-rule stand-ins for --fake-llm (no API keys needed)
scripts/            thin CLI wrappers around the above, numbered in pipeline order
tests/              66 tests covering every module above
data/demo/          synthetic fixtures so every script runs today, with no Kaggle download or API keys
```

## Quick start (no API keys, no real dataset needed)
```bash
python scripts/01_discover_brands.py --input data/demo/subsample_demo.csv --out_dir /tmp/demo --top_n_brands 2
python scripts/05_run_pipeline.py --corpus data/demo/retrieval_corpus_demo.csv --message "my app keeps crashing" --fake-llm
python scripts/06_run_eval.py --corpus data/demo/retrieval_corpus_demo.csv --golden_set data/demo/golden_set_demo.jsonl --fake-llm
```
`--fake-llm` swaps `classify_intent`/`draft_reply`/`judge_reply` for
deterministic keyword rules (`demo_fakes.py`) -- proves every module wires
together correctly without spending API quota. Drop the flag (and add real
keys to `.env`) once you have a real corpus and golden set. See `REPORT.md`
for what this actually produced and exactly which parts are real vs.
sandbox-limited.

## Testing
```bash
pytest tests/ -v
```
66 tests, no API keys or network access needed. `escalation.py` /
`metrics.py` / `baselines.py` / `golden_set.py` are pure logic or local
sklearn, tested directly. `pipeline.py` / `eval_harness.py` take
`classify_fn`/`draft_fn`/`judge_fn` as injectable parameters (default to
the real `llm_clients` functions) so tests pass deterministic fakes
instead of calling a real API. `embed_utils.py` is tested by stubbing
`sentence_transformers` in `sys.modules` with a small deterministic
encoder before import -- the real, unmodified retrieval code gets
exercised, just without downloading real model weights.

## Why these choices (full reasoning in `decision_log.md`)
- **Two LLM providers, not one**: the judge must not share blind spots with the
  generator, or "the judge agrees with the generator 95% of the time" measures
  nothing. Gemini drafts and classifies; Groq (different model family) judges.
- **Local embeddings, not an API**: retrieval doesn't need generation — a small
  offline model is free, instant, and keeps LLM API quota for calls that
  actually need an LLM.
- **No vector DB**: the corpus is a few thousand vectors at most. Numpy cosine
  similarity is exact and effectively instant at that size.
- **No LangChain / agent framework**: the pipeline is five linear steps a
  human can read top to bottom in the live code-review. A framework would
  hide exactly the auditability this assignment is grading.
- **Baselines cross-validated over the golden set, not a separate corpus**:
  the golden set is the only intent-labeled data that exists in this
  project; k-fold CV avoids training and testing a baseline on the same
  rows. Full reasoning in `decision_log.md`.
- **One metrics implementation, shared** (`metrics.py`) by baselines and the
  eval harness, so every row of the final comparison table is computed the
  same way — not three different approximations of "accuracy."
