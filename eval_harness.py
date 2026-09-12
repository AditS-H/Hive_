"""
Runs the production pipeline across the golden set -- the same pipeline
code a live message would hit, batch-executed over held-out cases so there
is no train/test leakage -- and scores each draft with an independent LLM
judge (Groq, a different model family than the Gemini generator; see
llm_clients.py). Rolls the result into one EvalReport alongside the two
baselines from baselines.py, computed through the same metrics.py functions,
so the final comparison table is genuinely apples-to-apples.
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable

from baselines import run_all_baselines, BaselineResult
from config import INTENT_TAXONOMY, DEFAULT_ESCALATION_CONFIG, DEMO_BRAND, Brand, EscalationConfig
from golden_set import GoldenCase
from llm_clients import judge_reply, classify_intent, draft_reply
from metrics import classification_metrics, ClassificationMetrics, hallucination_rate, rate
from pipeline import run_pipeline, PipelineExecution, Retriever


@dataclass
class JudgeScore:
    test_case_id: str
    groundedness: int
    tone_match: int
    resolution_quality: int
    hallucination_flag: int
    notes: str
    overall_score: float


@dataclass
class CaseResult:
    test_case: GoldenCase
    pipeline_result: PipelineExecution
    judge: JudgeScore
    is_correct_intent: bool
    escalation_matches_expected: bool


@dataclass
class EvalReport:
    timestamp: str
    sample_size: int
    classification: ClassificationMetrics
    hallucination_rate: float
    avg_groundedness: float
    avg_tone_match: float
    avg_resolution_quality: float
    auto_handle_rate: float
    escalation_rate: float
    escalation_accuracy: float  # how often the auto_handle/escalate decision matched expected_action
    avg_latency_ms: float
    baselines: list[BaselineResult]
    case_results: list[CaseResult]


def run_eval(
    golden_cases: list[GoldenCase],
    retriever: Retriever,
    taxonomy: dict = INTENT_TAXONOMY,
    brand: Brand = DEMO_BRAND,
    escalation_config: EscalationConfig = DEFAULT_ESCALATION_CONFIG,
    classify_fn: Callable[[str, dict], dict] = classify_intent,
    draft_fn: Callable[..., str] = draft_reply,
    judge_fn: Callable[..., dict] = judge_reply,
    include_baselines: bool = True,
    baseline_n_splits: int = 5,
    baseline_seed: int = 42,
) -> EvalReport:
    if not golden_cases:
        raise ValueError("golden_cases is empty -- nothing to evaluate")

    case_results: list[CaseResult] = []
    correct_intents = 0
    escalation_correct = 0
    auto_handled = 0
    hallucinations: list[int] = []
    groundedness_scores: list[int] = []
    tone_scores: list[int] = []
    resolution_scores: list[int] = []
    latencies: list[int] = []

    for case in golden_cases:
        exec_ = run_pipeline(
            case.customer_msg,
            retriever,
            author="@golden_set",
            taxonomy=taxonomy,
            brand=brand,
            escalation_config=escalation_config,
            classify_fn=classify_fn,
            draft_fn=draft_fn,
        )

        judge_raw = judge_fn(
            case.customer_msg, exec_.classification["intent"], exec_.draft["text"], exec_.retrieval["precedents"]
        )
        overall = round(
            (judge_raw["groundedness"] + judge_raw["tone_match"] + judge_raw["resolution_quality"]) / 3, 2
        )
        judge = JudgeScore(
            test_case_id=case.id,
            groundedness=judge_raw["groundedness"],
            tone_match=judge_raw["tone_match"],
            resolution_quality=judge_raw["resolution_quality"],
            hallucination_flag=judge_raw["hallucination_flag"],
            notes=judge_raw.get("notes", ""),
            overall_score=overall,
        )

        is_correct = exec_.classification["intent"] == case.ground_truth_intent
        escalation_ok = exec_.escalation.decision == case.expected_action

        correct_intents += int(is_correct)
        escalation_correct += int(escalation_ok)
        auto_handled += int(exec_.escalation.decision == "auto_handle")
        hallucinations.append(judge.hallucination_flag)
        groundedness_scores.append(judge.groundedness)
        tone_scores.append(judge.tone_match)
        resolution_scores.append(judge.resolution_quality)
        latencies.append(exec_.total_latency_ms)

        case_results.append(CaseResult(
            test_case=case,
            pipeline_result=exec_,
            judge=judge,
            is_correct_intent=is_correct,
            escalation_matches_expected=escalation_ok,
        ))

    n = len(golden_cases)
    y_true = [c.ground_truth_intent for c in golden_cases]
    y_pred = [cr.pipeline_result.classification["intent"] for cr in case_results]

    baselines = run_all_baselines(golden_cases, n_splits=baseline_n_splits, seed=baseline_seed) if include_baselines else []

    return EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sample_size=n,
        classification=classification_metrics(y_true, y_pred),
        hallucination_rate=hallucination_rate(hallucinations),
        avg_groundedness=round(sum(groundedness_scores) / n, 2),
        avg_tone_match=round(sum(tone_scores) / n, 2),
        avg_resolution_quality=round(sum(resolution_scores) / n, 2),
        auto_handle_rate=rate(auto_handled, n),
        escalation_rate=rate(n - auto_handled, n),
        escalation_accuracy=rate(escalation_correct, n),
        avg_latency_ms=round(sum(latencies) / n, 1),
        baselines=baselines,
        case_results=case_results,
    )


def format_comparison_table(report: EvalReport) -> str:
    """Plain-text render of the Baseline Comparison Matrix (majority-class /
    TF-IDF+LogReg / production pipeline), matching the reference
    architecture's GoldenSetEvalSuite table -- but with every number
    actually computed, not hardcoded."""
    rows = [(b.name, b.metrics.accuracy, b.metrics.macro_f1, b.metrics.precision,
             b.metrics.recall, b.metrics.cohen_kappa, None) for b in report.baselines]
    rows.append((
        "Gemini Grounded Agent Pipeline",
        report.classification.accuracy, report.classification.macro_f1,
        report.classification.precision, report.classification.recall,
        report.classification.cohen_kappa, report.hallucination_rate,
    ))

    header = f"{'Model':<32}{'Acc':>8}{'MacroF1':>10}{'Prec':>8}{'Recall':>8}{'Kappa':>8}{'Halluc':>10}"
    lines = [header, "-" * len(header)]
    for name, acc, f1, prec, rec, kappa, halluc in rows:
        halluc_str = "N/A" if halluc is None else f"{halluc * 100:.1f}%"
        lines.append(
            f"{name:<32}{acc * 100:>7.1f}%{f1:>10.3f}{prec:>8.3f}{rec:>8.3f}{kappa:>8.3f}{halluc_str:>10}"
        )
    return "\n".join(lines)


def save_report(report: EvalReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)
