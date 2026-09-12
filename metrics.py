"""
One metrics implementation, shared by baselines.py and eval_harness.py, so
every row of the final "Baseline Comparison Matrix" (majority-class /
TF-IDF+LogReg / Gemini agent pipeline) is computed by the exact same code
against the exact same held-out labels. Compute accuracy, macro F1, and
precision/recall three different ways for three different rows and the
comparison stops meaning anything -- this file exists so that can't happen.

(The reference architecture's own eval endpoint doesn't clear this bar: its
`/api/eval/run` derives macroF1/precision/recall as `accuracy * 0.95`,
`accuracy * 0.96`, `accuracy * 0.94` -- fixed multipliers on accuracy, not
independent metrics. Fine for a UI mockup with synthetic data; not something
to carry into a real report.)

Real implementations, via sklearn, no hand-rolled math.
"""
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    precision: float
    recall: float
    cohen_kappa: float
    n: int


def classification_metrics(y_true: list[str], y_pred: list[str]) -> ClassificationMetrics:
    """
    Macro-averaged precision/recall/F1 (every intent counts equally,
    regardless of how common it is in the golden set -- a rare but
    high-stakes intent like account_security_breach shouldn't get
    swamped by how_to_or_feature_question just because it's rarer).

    zero_division=0: a class the model never predicts scores 0 precision
    for that class rather than raising or silently dropping it, so a
    classifier that ignores a whole intent gets penalized instead of
    hidden by macro-averaging over fewer classes.
    """
    assert len(y_true) == len(y_pred) and len(y_true) > 0, "need equal-length, non-empty label lists"
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        cohen_kappa=cohen_kappa_score(y_true, y_pred),
        n=len(y_true),
    )


def hallucination_rate(flags: list[int]) -> float:
    """flags: list of 0/1 hallucination_flag values from the judge."""
    if not flags:
        return 0.0
    return sum(flags) / len(flags)


def rate(count: int, total: int) -> float:
    """Small helper so callers don't hand-roll `x / n if n else 0.0` everywhere."""
    return count / total if total else 0.0
