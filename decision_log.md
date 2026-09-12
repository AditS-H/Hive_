# Decision log

Every choice here is one a reviewer would reasonably ask "why?" about.
Organized by area, newest reasoning within each area last.

## Scope: what changed and why

Given the architecture mockup (a Google AI Studio React/Express demo with
synthetic data) and the existing Python scaffold (`llm_clients.py`,
`embed_utils.py`, `scripts/01_discover_brands.py`), the task was to build
everything the scaffold's own README marked unchecked -- intent taxonomy,
baselines, agent pipeline, golden set + judge + eval harness, report -- so
the Python side matches what the architecture already specifies, faithfully
where the architecture is right and *not* faithfully where it cuts corners
a real implementation shouldn't. Both are called out below.

Two files that already worked were left alone: `embed_utils.py` is
untouched. `01_discover_brands.py` moved into `scripts/` (matching its own
documented invocation path) with one cosmetic fix (see "Non-capturing
groups" below) and no logic changes.

## Two real bugs, fixed via live verification, not guesses

**`llm_clients.GROQ_JUDGE_MODEL`**: was `llama-3.3-70b-versatile`. Groq
announced this deprecated 2026-06-17 and fully decommissioned it
2026-08-16 (`console.groq.com/docs/deprecations`) -- calls to it fail
outright as of this session (2026-09-12). Replaced with
`openai/gpt-oss-120b`, Groq's own suggested replacement for that model,
which also preserves the "different model family than the generator"
property the module docstring calls out (OpenAI open-weights vs. Google
Gemini) -- the whole reason the judge runs on Groq instead of also being a
Gemini call.

**`llm_clients.GEMINI_MODEL`**: was `gemini-2.5-flash`. Confirmed via
`ai.google.dev`'s own docs (changelog, latest-model page) that
`gemini-3.8-flash` is the current GA Flash model (shipped 2026-09-02,
free-tier eligible via AI Studio) -- and it's what the reference
architecture's `server.ts` already assumed, so this also closes a
Python/architecture mismatch.

Both verified live (web search + fetch against `ai.google.dev` and
`console.groq.com`), not carried over from training-data assumptions, since
this is exactly the kind of fast-moving detail the module's own docstring
already warned would go stale.

## Intent taxonomy (`config.py`)

Generalized to 8 categories instead of copying the architecture's 7
verbatim, because its taxonomy is implicitly Spotify-shaped
(`technical_playback_bug`) while brand selection is still open --
`01_discover_brands.py`'s candidates span a media app, e-commerce, an
airline, and a rideshare app. Four categories are brand-agnostic by
construction and drive escalation directly (billing, login/auth, security
breach, legal threat); the rest are deliberately generic containers
(`product_issue_or_bug`, `subscription_or_plan_change`,
`order_or_delivery_status`, `how_to_or_feature_question`) that read
naturally regardless of which brand wins. Once brand selection is final,
review ~50 real messages from that brand's `subsample_twcs.csv` and rename
1-2 categories to match its actual vocabulary -- a single-file edit, by
design (nothing downstream references category names directly).

## Escalation policy (`escalation.py`, `config.py`)

Ported the six rules 1:1 from `escalationRules.ts`, same thresholds
(`confidence_threshold=0.76`, `min_precedent_similarity=0.58`), so the
Python and TypeScript versions are provably the same policy. Three
deliberate deviations, each because the reference version had a real gap:

1. **Dead mandatory-escalation entries.** The reference's
   `mandatoryEscalationIntents` lists `vip_press_inquiry` and
   `billing_chargeback` -- neither is a key in its own `INTENT_TAXONOMY`,
   so neither could ever match a real classification. Dropped both.
   `billing_chargeback` was redundant anyway: "chargeback" is already in
   `sensitive_keywords`, so that signal isn't lost.

2. **Dead legal-queue keyword.** The reference's routing checks
   `matchedKeywords.includes('legal')`, but `'legal'` is never in
   `sensitiveKeywords` -- that branch could never fire from that check
   alone. It only ever routed to the legal queue via `'lawyer'` or
   `'sue'`.

3. **Narrow legal-queue trigger (caught by a failing test, not inspection).**
   Widening the routing to more legal-flavored keywords (`attorney`,
   `lawsuit`, `court`, `ftc`, `attorney general`, `regulator`) was
   originally meant to just replace the dead `'legal'` check with a real
   one. Writing `tests/test_escalation.py` surfaced that this was too
   narrow either way: a message mentioning only "regulator" or "FTC" (no
   "lawyer"/"sue") escalated correctly but landed in the generic queue
   instead of Legal & Regulatory Triage, in both the original and my first
   pass. Widened `_LEGAL_ROUTE_KEYWORDS` to the full legal-flavored subset
   of `sensitive_keywords`, leaving the fraud/security-flavored ones
   (`chargeback`, `fraud`, `scam`, `stolen`, `illegal`, `police`) to route
   through the existing intent- and sentiment-based branches, since the
   architecture's queue set only has four destinations and doesn't
   subdivide further than that.

## `classify_intent` / `draft_reply` (`llm_clients.py`)

**Added `sentiment` and `urgency` to `classify_intent`'s schema, prompt,
and return contract.** The original returned only
`{intent, confidence, reasoning}`. Escalation rule 4 (Customer Distress &
Hostility Check) needs `classification["sentiment"]` and `["urgency"]`
directly -- without this, that rule could never be implemented against
real classifier output, only against hand-constructed test dicts.

**Added optional `brand_name` / `tone_description` / `signoff` to
`draft_reply`**, defaulting to `""` / generic phrasing so existing callers
still work unchanged. The architecture's draft prompt threads brand voice
and a mandated sign-off into every draft; the original Python prompt had
no such parameters at all. Kept the original's stronger anti-hallucination
rules and its explicit handling of the zero-precedent case (asks a
clarifying question instead of improvising) -- the architecture's own
draft prompt doesn't handle that case as carefully, so that part was
*not* copied over.

**Judge, retrieval: no changes.** `judge_reply`'s return shape already
matches the architecture's `JudgeEvaluation` fields; `overallScore` and
`testCaseId` are the harness's job to attach (computed from data already
available), not the LLM call's. `embed_utils.RetrievalIndex.query(message,
k, min_similarity)` already matches the architecture's retrieval call
signature exactly.

## Golden set: why this can't be auto-generated (`golden_set.py`)

The architecture's `golden_set` node is explicitly "Audit Verified" --
human-labeled. That's not a formality: the only labeled data in this
project *is* the golden set. `subsample_twcs.csv` has real customer/brand
text but no ground-truth intent labels. If the same pipeline that drafts
replies also invented its own ground truth, evaluation would just measure
self-agreement, not quality. `golden_set.py` builds everything around that
human step (schema, JSONL I/O, validation that catches labeling typos
before a wasted eval run, a length-stratified sampler that turns raw
candidate messages into an empty-labeled template) but doesn't try to
remove the human from the loop.

**Held out from the retrieval corpus, not a subset of it** -- worth
stating explicitly because it's easy to get backwards even in a synthetic
smoke test. `data/demo/golden_set_demo.jsonl`'s 12 cases are paraphrases of
`data/demo/retrieval_corpus_demo.csv`'s 10 pairs, not the literal same
rows. Golden cases that are literally corpus entries would retrieve
themselves at ~100% similarity, which would look like excellent grounding
for a reason that has nothing to do with the pipeline actually working.

## Baselines: cross-validated over the golden set, not a separate corpus (`baselines.py`)

Majority-class and TF-IDF+LogReg need labeled training data, and -- see
above -- the golden set is the only labeled data that exists. Training and
testing a baseline on the same rows would be leakage, so both baselines
run via stratified k-fold CV: each fold's held-out predictions are the
ones scored. Falls back to plain shuffled `KFold` when a class has fewer
examples than the fold count (true of the 12-case demo set spread across
8 categories; shouldn't matter once the real 150-250 case set exists with
a few dozen examples per category), and falls back to predicting the
sole class seen when a fold's training split is single-class, rather than
letting `LogisticRegression.fit` raise. `class_weight="balanced"` because
support intents are naturally imbalanced and unweighted accuracy would
reward ignoring the rare, high-stakes categories.

## One metrics implementation, shared (`metrics.py`)

`baselines.py` and `eval_harness.py` both call
`metrics.classification_metrics()`, so every row of the final comparison
table is computed the same way. This matters because the reference
architecture's own `/api/eval/run` doesn't clear this bar: it derives
`macroF1`/`precision`/`recall` as `accuracy * 0.95` / `* 0.96` / `* 0.94`
-- fixed multipliers on accuracy, not independent metrics. Fine for a UI
mockup with synthetic numbers; not something to carry into a report anyone
might cite. Real sklearn throughout: `f1_score(average="macro")`,
`cohen_kappa_score`, etc.

## Testing without network access or model weights

This sandbox has no route to `huggingface.co` (model weights),
`generativelanguage.googleapis.com` / `console.groq.com` (live API calls),
or `kaggle.com` (the real dataset), and not enough disk for a torch
install. None of that is a reason to leave the new code unverified:

- **Escalation, metrics, baselines, golden set**: pure logic / local
  sklearn computation, no stubbing needed. 13 + 8 + 6 + 10 tests.
- **`pipeline.py` / `eval_harness.py`**: `classify_fn` / `draft_fn` /
  `judge_fn` are injectable parameters (default to the real
  `llm_clients` functions). Tests pass deterministic fakes instead of
  hitting a real API -- this is dependency injection via default
  arguments, not a framework, and it's the reason `run_pipeline()` and
  `run_eval()` are testable at all without keys. 6 + 9 tests.
- **`embed_utils.py`** (unmodified): imports `sentence_transformers` at
  module load time. Tests stub that module in `sys.modules` with a small
  deterministic hash-based encoder *before* importing `embed_utils`, so
  the real, shipped code gets exercised -- top-k ordering, the
  `min_similarity` cutoff, cosine-via-dot-product on normalized vectors --
  not a reimplementation of it. 5 tests.
- **`demo_fakes.py`**: keyword-rule regressions (e.g. "crashed" matching
  where a narrower regex wouldn't have). 9 tests.

66 tests total, all passing together (no shared-state interference) as of
this session.

**The end-to-end demo run used the same sentence-transformers stub**,
injected via `PYTHONPATH` pointing at a stub package kept *outside* this
project's directory -- it is not part of the deliverable and the user's
own `pip install -r requirements.txt` never touches it. One consequence
worth flagging: that stub is a crude word-hash encoder, much weaker than
real MiniLM at paraphrase similarity, so the demo run's grounding
similarity scores (and therefore its auto-handle rate) are an artifact of
*my verification environment*, not of the shipped code or the checked-in
demo data. Running the identical `--fake-llm` command on a real machine
(real `sentence-transformers` installed per `requirements.txt`) will use
real embeddings for retrieval even in `--fake-llm` mode, since that flag
only swaps `classify_fn`/`draft_fn`/`judge_fn`, never the retriever -- see
`REPORT.md` for the actual numbers this produced and exactly which parts
of them are real vs. sandbox-limited.

## Non-capturing groups (`01_discover_brands.py`, `02_build_retrieval_corpus.py`)

`DM_PATTERNS`'s groups exist only for alternation, not extraction, but
pandas' `.str.contains()` warns on any capturing group in a match-only
context. Changed to `(?:...)` in both files -- zero behavior change,
warning gone (confirmed by re-running against the demo data before and
after).

## What's still blocked on you, and exactly what to run

1. **Real dataset.** This sandbox can't reach kaggle.com. Run
   `scripts/01_discover_brands.py` locally (see `data/README.md`).
2. **Brand voice.** After picking a brand from step 1's output, skim ~50
   real replies and fill in a real `config.Brand` (name/handle/tone/
   signoff) to replace `DEMO_BRAND`.
3. **Golden set.** Run `scripts/03_sample_golden_set.py`, hand-label
   150-250 cases (target per the architecture spec), save as
   `data/golden_set.jsonl`.
4. **API keys.** Add `GEMINI_API_KEY` and `GROQ_API_KEY` to `.env`, then
   `scripts/06_run_eval.py` (no `--fake-llm`) produces the real numbers
   that belong in `REPORT.md`'s "Results" section.
