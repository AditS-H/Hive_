"""
Two baselines the production Gemini pipeline has to beat, or an LLM-based
approach isn't earning its extra cost and latency over something far
simpler.

Both are evaluated via k-fold cross-validation OVER THE GOLDEN SET, not a
separate labeled training corpus -- because the golden set is the only
ground-truth-labeled data in this project. `subsample_twcs.csv` has real
customer/brand text but no intent labels; the golden set is where labels
come from at all (see golden_set.py). CV avoids training and testing a
baseline on the same rows: each fold's held-out predictions are the ones
scored, via the same metrics.classification_metrics() the production
pipeline's own report uses, so every row of the final comparison table
is apples-to-apples.

Falls back from StratifiedKFold to plain shuffled KFold when some intent
has fewer examples than the fold count -- true for a small smoke-test
golden set (see data/demo/), not expected to matter once the real 150-250
case set exists with a few dozen examples per category.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold

from golden_set import GoldenCase
from metrics import classification_metrics, ClassificationMetrics


@dataclass
class BaselineResult:
    name: str
    metrics: ClassificationMetrics
    n_splits: int
    stratified: bool


def _make_folds(labels: list[str], requested_n_splits: int, seed: int):
    """
    Returns (folds, n_splits_used, was_stratified). Stratifies when every
    class has enough members to appear in each fold; otherwise falls back
    to plain KFold rather than letting sklearn raise on a rare class.
    """
    n = len(labels)
    min_class_count = min(Counter(labels).values())

    if min_class_count >= 2:
        n_splits = max(2, min(requested_n_splits, min_class_count))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(n), labels)), n_splits, True

    n_splits = max(2, min(requested_n_splits, n))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(n))), n_splits, False


def majority_class_baseline(cases: list[GoldenCase], n_splits: int = 5, seed: int = 42) -> BaselineResult:
    """
    The floor: predict whatever intent was most common in the training
    fold, ignoring the message text entirely. Any real classifier that
    can't clear this isn't classifying anything.
    """
    labels = [c.ground_truth_intent for c in cases]
    folds, n_splits, stratified = _make_folds(labels, n_splits, seed)

    preds: list[str] = [""] * len(labels)
    for train_idx, test_idx in folds:
        train_labels = [labels[i] for i in train_idx]
        majority_label = Counter(train_labels).most_common(1)[0][0]
        for i in test_idx:
            preds[i] = majority_label

    return BaselineResult(
        name="Majority Class",
        metrics=classification_metrics(labels, preds),
        n_splits=n_splits,
        stratified=stratified,
    )


def tfidf_logreg_baseline(cases: list[GoldenCase], n_splits: int = 5, seed: int = 42) -> BaselineResult:
    """
    A real (if simple) text classifier: TF-IDF unigrams+bigrams into
    multinomial logistic regression. class_weight="balanced" because
    support intents are naturally imbalanced (far more how-to questions
    than legal threats) and unweighted accuracy would reward ignoring the
    rare, high-stakes categories.
    """
    texts = [c.customer_msg for c in cases]
    labels = [c.ground_truth_intent for c in cases]
    folds, n_splits, stratified = _make_folds(labels, n_splits, seed)

    preds: list[str] = [""] * len(labels)
    for train_idx, test_idx in folds:
        train_labels = [labels[i] for i in train_idx]

        if len(set(train_labels)) < 2:
            # A fold whose training split only ever saw one class can't fit
            # a real decision boundary -- predicting that one class is the
            # honest best a classifier could do here, same as the majority
            # baseline would, rather than letting sklearn raise.
            only_label = train_labels[0]
            for i in test_idx:
                preds[i] = only_label
            continue

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, stop_words="english")
        X_train = vectorizer.fit_transform([texts[i] for i in train_idx])
        X_test = vectorizer.transform([texts[i] for i in test_idx])

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, train_labels)
        fold_preds = clf.predict(X_test)
        for i, p in zip(test_idx, fold_preds):
            preds[i] = p

    return BaselineResult(
        name="TF-IDF + Logistic Regression",
        metrics=classification_metrics(labels, preds),
        n_splits=n_splits,
        stratified=stratified,
    )


def run_all_baselines(cases: list[GoldenCase], n_splits: int = 5, seed: int = 42) -> list[BaselineResult]:
    return [
        majority_class_baseline(cases, n_splits, seed),
        tfidf_logreg_baseline(cases, n_splits, seed),
    ]
