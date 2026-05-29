from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Sequence

from project.esg.models import DOMAINS, DeliberationRecord, DomainScore, PatternRunResult


def score_to_label(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def compute_coverage(total_tokens: int, retrieved_tokens: int, total_chunks: int, retrieved_chunks: int) -> dict[str, float]:
    weighted_binary = min(1.0, retrieved_chunks / max(1, total_chunks))
    partial = min(1.0, retrieved_tokens / max(1, total_tokens))
    return {
        "weighted_binary": round(weighted_binary, 4),
        "partial": round(partial, 4),
    }


def compute_classification_metrics(predictions: Sequence[str], truths: Sequence[str]) -> dict[str, float]:
    labels = sorted(set(predictions) | set(truths) | {"low", "medium", "high"})
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for pred, truth in zip(predictions, truths) if pred == label and truth == label)
        fp = sum(1 for pred, truth in zip(predictions, truths) if pred == label and truth != label)
        fn = sum(1 for pred, truth in zip(predictions, truths) if pred != label and truth == label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "precision_macro": round(mean(precisions), 4),
        "recall_macro": round(mean(recalls), 4),
        "f1_macro": round(mean(f1s), 4),
    }


def compute_mae(predictions: Sequence[float], truths: Sequence[float]) -> float:
    if not predictions or not truths:
        return 0.0
    return round(mean(abs(pred - truth) for pred, truth in zip(predictions, truths)), 4)


def judge_score(domain_score: DomainScore) -> float:
    total_claims = domain_score.supported_claims + domain_score.unsupported_claims + domain_score.partially_supported_claims
    if total_claims == 0:
        return 0.0
    value = (domain_score.supported_claims + 0.5 * domain_score.partially_supported_claims) / total_claims
    return round(value, 4)


def text_similarity(left: str, right: str) -> float:
    left_tokens = set(left.lower().split())
    right_tokens = set(right.lower().split())
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def output_similarity(left: PatternRunResult, right: PatternRunResult) -> float:
    score_similarity = 1 - min(1.0, abs(left.total_score - right.total_score) / 100.0)
    rationale_similarity = mean(
        text_similarity(left.domain_scores[domain].rationale, right.domain_scores[domain].rationale)
        for domain in DOMAINS
    )
    return round((score_similarity + rationale_similarity) / 2, 4)


def consistency_score(results: Sequence[PatternRunResult]) -> float:
    if len(results) < 2:
        return 1.0
    sims: list[float] = []
    for index, left in enumerate(results[:-1]):
        for right in results[index + 1 :]:
            sims.append(output_similarity(left, right))
    return round(mean(sims), 4) if sims else 1.0


def fleiss_kappa(label_matrix: Sequence[Sequence[str]]) -> float:
    if not label_matrix:
        return 0.0
    categories = sorted({label for row in label_matrix for label in row})
    n_items = len(label_matrix)
    n_raters = len(label_matrix[0])
    p_j = []
    for category in categories:
        count = sum(label == category for row in label_matrix for label in row)
        p_j.append(count / max(1, n_items * n_raters))
    p_e = sum(value * value for value in p_j)
    p_i = []
    for row in label_matrix:
        counts = Counter(row)
        agreement = sum(count * (count - 1) for count in counts.values()) / max(1, n_raters * (n_raters - 1))
        p_i.append(agreement)
    p_bar = mean(p_i)
    if math.isclose(1 - p_e, 0.0):
        return 1.0
    return round((p_bar - p_e) / (1 - p_e), 4)


def pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or not values_a:
        return 0.0
    mean_a = mean(values_a)
    mean_b = mean(values_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    denominator_a = math.sqrt(sum((a - mean_a) ** 2 for a in values_a))
    denominator_b = math.sqrt(sum((b - mean_b) ** 2 for b in values_b))
    denominator = denominator_a * denominator_b
    if math.isclose(denominator, 0.0):
        return 0.0
    return round(numerator / denominator, 4)


def inter_agent_agreement(domain_scores: dict[str, DomainScore]) -> dict[str, float]:
    labels = [[score_to_label(domain_scores[domain].estimated_score) for domain in DOMAINS]]
    continuous_pairs = []
    values = [domain_scores[domain].estimated_score for domain in DOMAINS]
    for index, left in enumerate(values[:-1]):
        for right in values[index + 1 :]:
            continuous_pairs.append(pearson([left, 50.0], [right, 50.0]))
    return {
        "fleiss_kappa": fleiss_kappa(labels),
        "pairwise_pearson": round(mean(continuous_pairs), 4) if continuous_pairs else 0.0,
    }


def hallucination_rates(domain_scores: dict[str, DomainScore]) -> dict[str, float]:
    supported = sum(score.supported_claims for score in domain_scores.values())
    unsupported = sum(score.unsupported_claims for score in domain_scores.values())
    partial = sum(score.partially_supported_claims for score in domain_scores.values())
    total = supported + unsupported + partial
    return {
        "unsupported_rate": round(unsupported / max(1, total), 4),
        "partially_supported_rate": round(partial / max(1, total), 4),
    }


def deliberation_quality(deliberations: Sequence[DeliberationRecord], actual_total: float | None, pre_total: float, post_total: float) -> dict[str, float]:
    if not deliberations:
        return {"cdr": 0.0, "rq_or_rc": 0.0, "dr": 1.0, "composite": 0.0}
    genuine_conflicts = sum(1 for item in deliberations if "gap" in item.issue or "bias" in item.issue)
    cdr = genuine_conflicts / max(1, len(deliberations))
    resolved = sum(1 for item in deliberations if item.approved)
    rq_or_rc = resolved / max(1, len(deliberations))
    if actual_total is None:
        dominance_matches = sum(1 for item in deliberations if math.isclose(item.initial_score, item.final_score, abs_tol=0.25))
    else:
        before_gap = abs(pre_total - actual_total)
        after_gap = abs(post_total - actual_total)
        dominance_matches = len(deliberations) if after_gap >= before_gap else max(0, len(deliberations) - 1)
    dr = dominance_matches / max(1, len(deliberations))
    composite = (cdr + rq_or_rc + (1 - dr)) / 3
    return {
        "cdr": round(cdr, 4),
        "rq_or_rc": round(rq_or_rc, 4),
        "dr": round(dr, 4),
        "composite": round(composite, 4),
    }
