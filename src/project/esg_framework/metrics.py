from __future__ import annotations

import math
from statistics import mean
from time import perf_counter
from typing import Any

from project.esg_framework.models import DomainScore, RetrievalEvent

# Markers expected in rationale templates for lightweight grounding checks.
# Limitation: this is a format-dependent heuristic, not semantic fact verification.
# Claims without these markers are treated as likely unsupported in local hallucination estimates.
# Includes [llm] prefix used by LLM scoring and critique patterns.
SUPPORTED_MARKERS = ("Detected", "Used", "[llm]", "[FALLBACK]", "Critique applied")
PARTIAL_MARKER = "heuristic"
TOTAL_SCORE_MAX = 20.0


class Timer:
    """Lightweight timer utility for measuring execution durations of named operations.
    
    Tracks checkpoints and computes elapsed time between start/stop calls.
    Used to measure latency of various pattern execution phases.
    """

    def __init__(self) -> None:
        """Initialize empty checkpoint and duration dictionaries."""
        self._checkpoints: dict[str, float] = {}
        self._durations: dict[str, float] = {}

    def start(self, name: str) -> None:
        """Record the current time as a checkpoint for the given operation name.
        
        Args:
            name: Identifier for the timing operation (e.g., 'parse_report', 'score_environmental').
        """
        self._checkpoints[name] = perf_counter()

    def stop(self, name: str) -> None:
        """Compute and store the duration for a previously started checkpoint.
        
        Args:
            name: Identifier matching a previous start() call.
        """
        if name in self._checkpoints:
            self._durations[name] = perf_counter() - self._checkpoints[name]

    @property
    def durations(self) -> dict[str, float]:
        """Get all recorded durations as a dictionary of {name: seconds}.
        
        Returns:
            Dictionary with operation names as keys and rounded durations (6 decimal places) as values.
        """
        return {key: round(value, 6) for key, value in self._durations.items()}


def coverage_metrics(
    all_chunk_ids: list[str],
    chunk_weights: dict[str, float],
    chunk_token_counts: dict[str, int],
    trace: list[RetrievalEvent],
) -> dict[str, float]:
    """Compute evidence coverage metrics based on retrieved and used chunks.
    
    Measures what proportion of available evidence (chunks) was retrieved and actually used
    by the scoring agents. Helps assess whether patterns are making decisions with
    sufficient evidence or missing critical information.
    
    Args:
        all_chunk_ids: List of all chunk IDs from the report.
        chunk_weights: Mapping of chunk ID to its importance weight.
        chunk_token_counts: Mapping of chunk ID to its token count.
        trace: List of RetrievalEvent objects recording which chunks were retrieved/used.
    
    Returns:
        Dictionary with:
        - coverage_weighted: Weighted coverage ratio (retrieved weight / total weight).
        - coverage_partial: Token-based coverage ratio (used tokens / total tokens).
    """
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
    """Compute Mean Absolute Error between prediction and ground truth.
    
    Simple absolute difference metric. Higher values indicate worse performance.
    
    Args:
        prediction: The predicted/estimated score.
        truth: The ground truth score.
    
    Returns:
        Absolute difference rounded to 4 decimal places.
    """
    return round(abs(prediction - truth), 4)


def bias_score(prediction: float, truth: float) -> float:
    """Compute relative bias: (prediction - truth) / truth.
    
    Measures systematic overestimation (positive) or underestimation (negative).
    A value of 0.1 means 10% overestimation. Helps identify if a pattern
    consistently scores too high or too low.
    
    Args:
        prediction: The predicted/estimated score.
        truth: The ground truth score.
    
    Returns:
        Relative bias ratio rounded to 4 decimal places. Positive = overestimation.
    """
    denominator = max(abs(truth), 1.0)
    return round((prediction - truth) / denominator, 4)


def accuracy_from_mae(mae_value: float, max_score: float = TOTAL_SCORE_MAX) -> float:
    """Convert MAE to an accuracy-like metric normalized by max possible score.
    
    Transforms absolute error into a 0-1 accuracy scale where 1.0 = perfect
    (MAE=0) and 0.0 = worst (MAE >= max_score).
    
    Args:
        mae_value: The Mean Absolute Error to convert.
        max_score: The maximum possible score (default 20.0).
    
    Returns:
        Accuracy value between 0.0 and 1.0, rounded to 4 decimal places.
    """
    if max_score <= 0:
        return 0.0
    normalized_error = min(1.0, max(0.0, mae_value / max_score))
    return round(1.0 - normalized_error, 4)


def judge_accuracy_stub(scores: dict[str, DomainScore]) -> float:
    """Proxy judge accuracy metric for local runs without a dedicated judge model.
    
    Estimates judge agreement by averaging domain confidence scores with a small
    upward adjustment (+0.1). This is a standalone approximation used when
    external judging is not available.
    
    Args:
        scores: Dictionary of domain names to DomainScore objects.
    
    Returns:
        Mean of (confidence + 0.1) capped at 1.0, rounded to 4 decimal places.
    """
    # Proxy judge mode for local runs without extra judge model.
    return round(mean(min(1.0, s.confidence + 0.1) for s in scores.values()), 4)


def pair_similarity(a: dict[str, DomainScore], b: dict[str, DomainScore]) -> float:
    """Compute similarity between two sets of domain scores.
    
    Compares estimated scores for shared domains, normalizing differences by the
    max score (20.0). Returns a similarity score where 1.0 = identical, 0.0 = maximally different.
    
    Args:
        a: First set of domain scores (domain name -> DomainScore).
        b: Second set of domain scores (domain name -> DomainScore).
    
    Returns:
        Similarity score between 0.0 and 1.0, rounded to 4 decimal places.
    """
    domains = sorted(set(a.keys()) & set(b.keys()))
    if not domains:
        return 0.0
    diffs = [abs(a[d].estimated_score - b[d].estimated_score) / 20.0 for d in domains]
    sim = 1.0 - mean(diffs)
    return round(max(0.0, min(1.0, sim)), 4)


def consistency_metric(trial_outputs: list[dict[str, DomainScore]]) -> float:
    """Measure consistency across multiple trial runs.
    
    Computes pairwise similarity between all pairs of trial outputs using
    pair_similarity(). A value of 1.0 means all trials produced identical scores.
    Values below 0.85 indicate moderate variance; below 0.7 indicate high variance.
    
    Args:
        trial_outputs: List of domain score dictionaries, one per trial.
    
    Returns:
        Mean pairwise similarity across all trial pairs, rounded to 4 decimal places.
        Returns 1.0 if fewer than 2 trials.
    """
    if len(trial_outputs) < 2:
        return 1.0
    sims: list[float] = []
    for i in range(len(trial_outputs) - 1):
        for j in range(i + 1, len(trial_outputs)):
            sims.append(pair_similarity(trial_outputs[i], trial_outputs[j]))
    return round(mean(sims), 4) if sims else 1.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient between two vectors.
    
    Measures linear correlation between two variables. Values range from -1.0
    (perfect negative correlation) to +1.0 (perfect positive correlation).
    
    Args:
        xs: First vector of numerical values.
        ys: Second vector of numerical values (must be same length as xs).
    
    Returns:
        Pearson correlation coefficient rounded to 4 decimal places.
        Returns 0.0 if vectors are too short or have zero variance.
    """
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
    """Compute Fleiss' kappa statistic for inter-rater agreement.
    
    Measures agreement among multiple raters assigning categorical labels to items.
    Values range from -1 (no agreement) to +1 (perfect agreement), with 0 indicating
    agreement expected by chance.
    
    Note: This implementation currently creates a single-row matrix from a flat
    list of labels, which is a simplified approach for the multi-agent context.
    
    Args:
        label_matrix: 2D list where each row is an item and each column is a rater's label.
        categories: List of possible label categories (e.g., ["low", "medium", "high"]).
    
    Returns:
        Fleiss' kappa value rounded to 4 decimal places.
        Returns 0.0 if there are no items or fewer than 2 raters.
    """
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
    """Compute agreement metrics between domain scoring agents.
    
    Uses two approaches:
    1. Fleiss' kappa on categorical labels (low/medium/high).
    2. Pairwise Pearson correlation on continuous dimensions (score, confidence, chunk count).
    
    Provides a comprehensive view of whether different domain agents are producing
    consistent results.
    
    Args:
        results: Dictionary of domain names to DomainScore objects.
    
    Returns:
        Dictionary with:
        - fleiss_kappa: Categorical agreement on score labels.
        - pairwise_pearson: Mean Pearson correlation across continuous dimensions.
    """
    labels = [score.label for score in results.values()]
    label_matrix = [labels]
    kappa = fleiss_kappa(label_matrix, ["low", "medium", "high"])

    # Local proxy: compare each agent over shared continuous dimensions.
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
    """Estimate hallucination rates by analyzing rationale text for support markers.
    
    This is a lightweight, format-dependent heuristic that checks if claims in the
    rationale are backed by evidence markers. It does NOT perform semantic fact verification.
    
    A claim is considered:
    - "unsupported" if it contains none of the SUPPORTED_MARKERS.
    - "partial" if it contains PARTIAL_MARKER but not "Detected".
    
    Note: This metric is sensitive to rationale formatting and the presence of
    specific marker strings. It should be interpreted as a proxy, not an absolute
    measure of factual accuracy.
    
    Args:
        outputs: Dictionary of domain names to DomainScore objects with rationales.
    
    Returns:
        Dictionary with:
        - unsupported_claim_rate: Fraction of claims without any support markers.
        - partial_support_rate: Fraction of claims with only partial support.
    """
    # This local approximation depends on scorer rationale templates containing support markers.
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
    """Compute deliberation quality metrics for patterns with multi-agent interaction.
    
    Combines three dimensions into an overall deliberation quality score:
    1. Conflict detection rate: How often disagreements were flagged.
    2. Resolution quality: How often flagged conflicts were resolved with trace.
    3. Dominance ratio: How concentrated decision-making was (lower = more distributed).
    
    The overall deliberation_quality is the mean of these three components.
    
    Args:
        conflicts_flagged: Number of conflicts detected during deliberation.
        disagreements: Total number of disagreements (used as denominator).
        resolved_with_trace: Number of disagreements that were resolved with trace.
        dominance_ratio: Ratio of most dominant agent's contributions (0-1).
    
    Returns:
        Dictionary with:
        - conflict_detection_rate: conflicts_flagged / disagreements.
        - resolution_quality: resolved_with_trace / disagreements.
        - dominance_ratio: The input dominance ratio.
        - deliberation_quality: Mean of the three component scores.
    """
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
    """Compute latency and efficiency metrics for pattern execution.
    
    Measures both raw speed and efficiency in terms of evidence coverage
    per computational resource used.
    
    Args:
        durations: Dictionary of operation names to their elapsed times (seconds).
        coverage: The coverage_weighted metric value.
        total_agent_calls: Number of LLM/agent calls made during execution.
        total_tokens: Total number of tokens processed.
    
    Returns:
        Dictionary with:
        - latency_total: Sum of all operation durations.
        - latency_critical_path: Longest single scoring operation (bottleneck).
        - coverage_per_call: Coverage achieved per agent call.
        - token_efficiency: Coverage achieved per token processed.
    """
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
    """Aggregate metrics across multiple reports to produce pattern-level summary statistics.
    
    For each metric in the summary_keys list, computes the mean value across all
    provided result dictionaries. This is used to produce the final summary table
    comparing different orchestration patterns.
    
    Args:
        results: List of metric dictionaries (one per report/run).
    
    Returns:
        Dictionary with the same keys as summary_keys, where each value is the
        mean of that metric across all input results, rounded to 4 decimal places.
        Returns empty dict if input is empty.
    """
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
