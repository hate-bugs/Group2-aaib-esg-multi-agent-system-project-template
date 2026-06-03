from __future__ import annotations

import math
from statistics import mean
from time import perf_counter
from typing import Any

from project.esg_framework_sustainalytics.models_sustainalytics import DomainScore, RetrievalEvent

SUPPORTED_MARKERS = ("Detected", "Used", "[llm]", "[FALLBACK]", "Critique applied")
PARTIAL_MARKER = "heuristic"
TOTAL_SCORE_MAX = 100.0


class Timer:
    def __init__(self) -> None:
        self._checkpoints: dict[str, float] = {}
        self._durations: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._checkpoints[name] = perf_counter()

    def stop(self, name: str) -> None:
        if name in self._checkpoints:
            self._durations[name] = perf_counter() - self._checkpoints[name]

    @property
    def durations(self) -> dict[str, float]:
        return {key: round(value, 6) for key, value in self._durations.items()}


def coverage_metrics(
    all_chunk_ids: list[str],
    chunk_weights: dict[str, float],
    chunk_token_counts: dict[str, int],
    trace: list[RetrievalEvent],
) -> dict[str, float]:
    retrieved_ids: set[str] = set()
    used_ids: set[str] = set()
    for event in trace:
        retrieved_ids.update(event.retrieved_chunk_ids)
        used_ids.update(event.used_chunk_ids)

    total_weight = sum(chunk_weights.get(cid, 1.0) for cid in all_chunk_ids) or 1.0
    covered_weight = sum(chunk_weights.get(cid, 1.0) for cid in retrieved_ids if cid in chunk_weights)
    weighted = covered_weight / total_weight

    total_tokens = sum(chunk_token_counts.get(cid, 0) for cid in all_chunk_ids) or 1
    partial_tokens = sum(chunk_token_counts.get(cid, 0) for cid in used_ids)
    partial = partial_tokens / total_tokens

    return {
        "coverage_weighted": round(weighted, 4),
        "coverage_partial": round(partial, 4),
    }


def mae(prediction: float, truth: float) -> float:
    return round(abs(prediction - truth), 4)


def bias_score(prediction: float, truth: float) -> float:
    denominator = max(abs(truth), 1.0)
    return round((prediction - truth) / denominator, 4)


def accuracy_from_mae(mae_value: float, max_score: float = TOTAL_SCORE_MAX) -> float:
    if max_score <= 0:
        return 0.0
    normalized_error = min(1.0, max(0.0, mae_value / max_score))
    return round(1.0 - normalized_error, 4)


def judge_accuracy_stub(scores: dict[str, DomainScore]) -> float:
    return round(mean(min(1.0, s.confidence + 0.1) for s in scores.values()), 4)


def pair_similarity(a: dict[str, DomainScore], b: dict[str, DomainScore]) -> float:
    domains = sorted(set(a.keys()) & set(b.keys()))
    if not domains:
        return 0.0
    diffs = [abs(a[d].estimated_score - b[d].estimated_score) / 100.0 for d in domains]
    sim = 1.0 - mean(diffs)
    return round(max(0.0, min(1.0, sim)), 4)


def consistency_metric(trial_outputs: list[dict[str, DomainScore]]) -> float:
    if len(trial_outputs) < 2:
        return 1.0
    sims: list[float] = []
    for i in range(len(trial_outputs) - 1):
        for j in range(i + 1, len(trial_outputs)):
            sims.append(pair_similarity(trial_outputs[i], trial_outputs[j]))
    return round(mean(sims), 4) if sims else 1.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    if den == 0:
        return 0.0
    return round(num / den, 4)


def fleiss_kappa(label_matrix: list[list[str]], categories: list[str]) -> float:
    n_items = len(label_matrix)
    if n_items == 0:
        return 0.0
    n_raters = len(label_matrix[0]) if label_matrix[0] else 0
    if n_raters < 2:
        return 0.0

    p_j: dict[str, float] = {c: 0.0 for c in categories}
    p_i_vals: list[float] = []

    for row in label_matrix:
        counts = {c: 0 for c in categories}
        for vote in row:
            if vote in counts:
                counts[vote] += 1
        row_agreement = sum(v * (v - 1) for v in counts.values()) / (n_raters * (n_raters - 1))
        p_i_vals.append(row_agreement)
        for category in categories:
            p_j[category] += counts[category]

    p_bar = mean(p_i_vals)
    p_e = sum((p_j[c] / (n_items * n_raters)) ** 2 for c in categories)
    if p_e == 1.0:
        return 1.0
    return round((p_bar - p_e) / (1 - p_e), 4)


def inter_agent_agreement(results: dict[str, DomainScore]) -> dict[str, float]:
    labels = [score.label for score in results.values()]
    label_matrix = [labels]
    kappa = fleiss_kappa(label_matrix, ["low", "medium", "high"])

    vectors = [
        [
            score.estimated_score,
            score.confidence,
            float(len(score.used_chunk_ids)),
        ]
        for score in results.values()
    ]
    pairwise = []
    for i in range(len(vectors) - 1):
        for j in range(i + 1, len(vectors)):
            pairwise.append(_pearson(vectors[i], vectors[j]))

    return {
        "fleiss_kappa": round(kappa, 4),
        "pairwise_pearson": round(mean(pairwise), 4) if pairwise else 0.0,
    }


def hallucination_rate(outputs: dict[str, DomainScore]) -> dict[str, float]:
    total_claims = 0
    unsupported = 0
    partial = 0
    for score in outputs.values():
        claims = [chunk.strip() for chunk in score.rationale.split(".") if chunk.strip()]
        total_claims += len(claims)
        for claim in claims:
            if not any(marker in claim for marker in SUPPORTED_MARKERS):
                unsupported += 1
            elif PARTIAL_MARKER in claim and "Detected" not in claim:
                partial += 1

    denom = max(1, total_claims)
    return {
        "unsupported_claim_rate": round(unsupported / denom, 4),
        "partial_support_rate": round(partial / denom, 4),
    }


def deliberation_quality(
    conflicts_flagged: int,
    disagreements: int,
    resolved_with_trace: int,
    dominance_ratio: float,
) -> dict[str, float]:
    overlap = max(1, disagreements)
    cdr = conflicts_flagged / overlap
    rq = resolved_with_trace / overlap
    dq = (cdr + rq + (1 - dominance_ratio)) / 3
    return {
        "conflict_detection_rate": round(cdr, 4),
        "resolution_quality": round(rq, 4),
        "dominance_ratio": round(dominance_ratio, 4),
        "deliberation_quality": round(dq, 4),
    }


def latency_efficiency(
    durations: dict[str, float],
    coverage: float,
    total_agent_calls: int,
    total_tokens: int,
) -> dict[str, float]:
    total_latency = sum(durations.values())
    scoring_paths = [value for key, value in durations.items() if key.startswith("score_")]
    critical_path = max(scoring_paths) if scoring_paths else total_latency
    return {
        "latency_total": round(total_latency, 6),
        "latency_critical_path": round(critical_path, 6),
        "coverage_per_call": round(coverage / max(1, total_agent_calls), 6),
        "token_efficiency": round(coverage / max(1, total_tokens), 8),
    }


def aggregate_pattern_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}

    summary_keys = [
        "coverage_weighted",
        "coverage_partial",
        "mae_total",
        "accuracy",
        "bias_total",
        "predicted_total_avg",
        "actual_total_avg",
        "judge_accuracy",
        "consistency_quantitative",
        "latency_total",
        "hallucination_unsupported",
        "deliberation_quality",
    ]
    out: dict[str, Any] = {}
    for key in summary_keys:
        vals = [row.get(key, 0.0) for row in results]
        out[key] = round(mean(vals), 4)
    return out


def _ground_truth_weighted_total(report: Any, weights: dict[str, float] | None = None) -> float:
    """Helper to compute weighted ground truth total from a report record."""
    from project.esg_framework_sustainalytics.heuristics_sustainalytics import ALL_DOMAINS
    weights = weights or {domain: 1.0 for domain in ALL_DOMAINS}
    numerator = 0.0
    denominator = 0.0
    for domain in ALL_DOMAINS:
        w = weights.get(domain, 1.0)
        numerator += report.ground_truth.get(domain, 0.0) * w
        denominator += w
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 2)


def _compare_to_ground_truth(total_score: float, report: Any) -> dict[str, float | str]:
    """Helper to compare predicted score to ground truth."""
    from project.esg_framework_sustainalytics.heuristics_sustainalytics import ALL_DOMAINS
    actual = _ground_truth_weighted_total(report)
    err = abs(total_score - actual)
    pct = (err / actual) * 100 if actual else 0.0
    direction = "over" if total_score > actual else "under"
    return {
        "actual_total": round(actual, 2),
        "actual_total_dataset": round(report.ground_truth.get("total", 0.0), 2),
        "absolute_error": round(err, 4),
        "percentage_error": round(pct, 4),
        "direction": direction,
    }


def _metric_bundle(
    report: Any,
    domain_scores: dict[str, DomainScore],
    retrieval_trace: list[RetrievalEvent],
    chunks,
    timer: Timer,
    total_agent_calls: int,
    deliberation_stats: dict[str, float],
) -> dict[str, float]:
    """Helper to compute the full metric bundle for a report run."""
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    chunk_weights = {chunk.chunk_id: chunk.weight for chunk in chunks}
    chunk_tokens = {chunk.chunk_id: chunk.token_count for chunk in chunks}
    coverage = coverage_metrics(chunk_ids, chunk_weights, chunk_tokens, retrieval_trace)

    hallucination = hallucination_rate(domain_scores)
    agreement = inter_agent_agreement(domain_scores)
    latency = latency_efficiency(
        timer.durations,
        coverage=coverage["coverage_weighted"],
        total_agent_calls=total_agent_calls,
        total_tokens=sum(chunk_tokens.values()),
    )

    predicted_total = sum(s.estimated_score for s in domain_scores.values()) / len(domain_scores) if domain_scores else 0.0
    actual_total = _ground_truth_weighted_total(report)
    mae_total = mae(predicted_total, actual_total)
    bias_total = bias_score(predicted_total, actual_total)
    return {
        **coverage,
        "mae_total": mae_total,
        "accuracy": accuracy_from_mae(mae_total),
        "bias_total": bias_total,
        "predicted_total_avg": round(predicted_total, 4),
        "actual_total_avg": round(actual_total, 4),
        "judge_accuracy": judge_accuracy_stub(domain_scores),
        "agreement_fleiss_kappa": agreement["fleiss_kappa"],
        "agreement_pairwise_pearson": agreement["pairwise_pearson"],
        "hallucination_unsupported": hallucination["unsupported_claim_rate"],
        "hallucination_partial": hallucination["partial_support_rate"],
        **latency,
        **deliberation_stats,
    }
