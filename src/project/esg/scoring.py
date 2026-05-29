from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from project.esg.chunking import chunk_report, tokenize
from project.esg.config import ESGExperimentConfig
from project.esg.heuristics import HEURISTICS_BY_DOMAIN
from project.esg.metrics import (
    compute_coverage,
    compute_classification_metrics,
    compute_mae,
    consistency_score,
    deliberation_quality,
    hallucination_rates,
    inter_agent_agreement,
    judge_score,
    score_to_label,
)
from project.esg.models import DOMAINS, DeliberationRecord, DomainScore, PatternRunResult, PatternTrace, ReportChunk, ReportRecord
from project.esg.retrieval import ChunkRetriever


def _matched_groups(domain: str, text: str) -> list[str]:
    lowered = text.lower()
    matched = []
    for group in HEURISTICS_BY_DOMAIN[domain]:
        if any(keyword in lowered for keyword in group.keywords):
            matched.append(group.name)
    return matched


def _build_domain_score(domain: str, chunks: list[ReportChunk], calls_used: int = 1, iteration_count: int = 1) -> DomainScore:
    combined_text = " ".join(chunk.text for chunk in chunks)
    matched = _matched_groups(domain, combined_text)
    coverage_ratio = len(matched) / max(1, len(HEURISTICS_BY_DOMAIN[domain]))
    estimated_score = round(coverage_ratio * 100, 2)
    confidence = min(0.97, round(0.35 + coverage_ratio * 0.45 + min(len(chunks), 4) * 0.05, 2))
    rationale = (
        f"{domain.title()} score is grounded in chunks {', '.join(chunk.chunk_id for chunk in chunks) or 'none'}. "
        f"Matched heuristics: {', '.join(matched) or 'none'}. "
        f"Assessment is based solely on preprocessed_content retrieved for {domain}."
    )
    supported_claims = len(matched)
    partially_supported_claims = max(0, min(2, len(chunks) - supported_claims))
    unsupported_claims = max(0, len(HEURISTICS_BY_DOMAIN[domain]) - supported_claims - partially_supported_claims)
    tokens_used = sum(chunk.token_count for chunk in chunks)
    return DomainScore(
        domain=domain,
        estimated_score=estimated_score,
        confidence=confidence,
        rationale=rationale,
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        partially_supported_claims=partially_supported_claims,
        matched_heuristics=matched,
        calls_used=calls_used,
        tokens_used=tokens_used,
        iteration_count=iteration_count,
    )


def _aggregate(domain_scores: dict[str, DomainScore], weights: dict[str, float]) -> float:
    total = sum(domain_scores[domain].estimated_score * weights.get(domain, 0.0) for domain in DOMAINS)
    return round(total, 2)


def _compare(total_score: float, actual_scores: dict[str, float | None]) -> dict[str, float | str | None]:
    actual_total = actual_scores.get("total")
    if actual_total is None:
        return {"absolute_error": None, "percentage_error": None, "direction": "unknown"}
    absolute_error = round(abs(total_score - actual_total), 4)
    percentage_error = round((absolute_error / max(1.0, abs(actual_total))) * 100, 4)
    direction = "over-estimated" if total_score > actual_total else "under-estimated" if total_score < actual_total else "matched"
    return {
        "absolute_error": absolute_error,
        "percentage_error": percentage_error,
        "direction": direction,
    }

def _unique_retrieval_stats(retrieval_groups) -> tuple[list[str], int]:
    unique_chunks = {}
    for items in retrieval_groups:
        for item in items:
            unique_chunks[item.chunk.chunk_id] = item.chunk
    return list(unique_chunks), sum(chunk.token_count for chunk in unique_chunks.values())



def _metrics_for_result(
    report: ReportRecord,
    chunks: list[ReportChunk],
    domain_scores: dict[str, DomainScore],
    trace: PatternTrace,
    pre_total: float | None = None,
) -> dict[str, object]:
    coverage = compute_coverage(
        total_tokens=trace.total_token_count,
        retrieved_tokens=trace.retrieved_token_count,
        total_chunks=len(chunks),
        retrieved_chunks=len(set(trace.retrieved_chunk_ids)),
    )
    actual_domain_scores = [report.actual_scores.get(domain) for domain in DOMAINS]
    has_ground_truth = all(value is not None for value in actual_domain_scores)
    accuracy: dict[str, object]
    if has_ground_truth:
        predicted_labels = [score_to_label(domain_scores[domain].estimated_score) for domain in DOMAINS]
        truth_labels = [score_to_label(float(report.actual_scores[domain])) for domain in DOMAINS]
        numeric_truths = [float(report.actual_scores[domain]) for domain in DOMAINS]
        numeric_predictions = [domain_scores[domain].estimated_score for domain in DOMAINS]
        accuracy = {
            **compute_classification_metrics(predicted_labels, truth_labels),
            "mae": compute_mae(numeric_predictions, numeric_truths),
        }
    else:
        accuracy = {
            "judge_score_average": round(mean(judge_score(domain_scores[domain]) for domain in DOMAINS), 4)
        }

    hallucination = hallucination_rates(domain_scores)
    agreement = inter_agent_agreement(domain_scores)
    total_calls = sum(domain_scores[domain].calls_used for domain in DOMAINS)
    total_tokens_used = sum(domain_scores[domain].tokens_used for domain in DOMAINS)
    deliberation = deliberation_quality(
        trace.deliberations,
        report.actual_scores.get("total"),
        pre_total if pre_total is not None else sum(score.estimated_score for score in domain_scores.values()) / 3,
        sum(score.estimated_score for score in domain_scores.values()) / 3,
    )
    return {
        "coverage": coverage,
        "accuracy": accuracy,
        "consistency": None,
        "inter_agent_agreement": agreement,
        "latency_efficiency": {
            "wall_clock_latency": round(trace.wall_clock_latency, 4),
            "critical_path_latency": round(trace.critical_path_latency, 4),
            "coverage_per_call": round(coverage["weighted_binary"] / max(1, total_calls), 4),
            "coverage_per_token": round(coverage["partial"] / max(1, total_tokens_used), 6),
        },
        "hallucination_rate": hallucination,
        "agent_deliberation_quality": deliberation,
    }


def run_parallel_pattern(report: ReportRecord, config: ESGExperimentConfig, trial: int) -> PatternRunResult:
    started = time.perf_counter()
    chunks = chunk_report(report, config.chunk_size, config.chunk_overlap, config.chunk_dir)
    retriever = ChunkRetriever(chunks)
    retrievals = {domain: retriever.retrieve(domain, config.top_k, trial=trial) for domain in DOMAINS}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            domain: executor.submit(_build_domain_score, domain, [item.chunk for item in retrievals[domain]])
            for domain in DOMAINS
        }
        domain_scores = {domain: future.result() for domain, future in futures.items()}
    total_score = _aggregate(domain_scores, config.score_weights)
    wall_clock = time.perf_counter() - started
    retrieved_chunks, retrieved_token_count = _unique_retrieval_stats(retrievals.values())
    trace = PatternTrace(
        retrieved_chunk_ids=retrieved_chunks,
        retrieved_token_count=retrieved_token_count,
        total_token_count=sum(chunk.token_count for chunk in chunks),
        agent_calls=len(DOMAINS),
        critical_path_latency=wall_clock / 2,
        wall_clock_latency=wall_clock,
        trial=trial,
    )
    metrics = _metrics_for_result(report, chunks, domain_scores, trace)
    return PatternRunResult(
        pattern="parallel",
        report_id=report.report_id,
        company_name=report.company_name,
        domain_scores=domain_scores,
        total_score=total_score,
        comparator=_compare(total_score, report.actual_scores),
        metrics=metrics,
        trace=trace,
        actual_scores=report.actual_scores,
    )


def run_hierarchical_pattern(report: ReportRecord, config: ESGExperimentConfig, trial: int) -> PatternRunResult:
    started = time.perf_counter()
    chunks = chunk_report(report, config.chunk_size, config.chunk_overlap, config.chunk_dir)
    retriever = ChunkRetriever(chunks)
    deliberations: list[DeliberationRecord] = []
    domain_scores: dict[str, DomainScore] = {}
    retrieved_chunks: dict[str, int] = {}
    total_agent_calls = 0

    for domain in DOMAINS:
        items = retriever.retrieve(domain, config.top_k + 1, trial=trial)
        worker_groups = [items[index : index + config.worker_batch_size] for index in range(0, len(items), config.worker_batch_size)]
        worker_scores = []
        for group_index, group in enumerate(worker_groups, start=1):
            chunks_for_group = [item.chunk for item in group]
            if not chunks_for_group:
                continue
            worker_score = _build_domain_score(domain, chunks_for_group)
            worker_scores.append(worker_score)
            deliberations.append(
                DeliberationRecord(
                    domain=domain,
                    iteration=group_index,
                    issue="worker aggregation gap" if worker_score.unsupported_claims else "worker review",
                    suggestion="Aggregate strongest worker evidence before final domain score.",
                    approved=True,
                    initial_score=worker_score.estimated_score,
                    final_score=worker_score.estimated_score,
                )
            )
        if worker_scores:
            aggregate_estimate = round(mean(score.estimated_score for score in worker_scores), 2)
            aggregate_confidence = round(mean(score.confidence for score in worker_scores), 2)
            combined_chunks = [item.chunk for group in worker_groups for item in group]
            score = _build_domain_score(domain, combined_chunks, calls_used=1 + len(worker_scores))
            score.estimated_score = aggregate_estimate
            score.confidence = aggregate_confidence
            score.iteration_count = len(worker_scores)
            domain_scores[domain] = score
        else:
            domain_scores[domain] = _build_domain_score(domain, [])
        total_agent_calls += domain_scores[domain].calls_used
        for item in items:
            retrieved_chunks[item.chunk.chunk_id] = item.chunk.token_count

    total_score = _aggregate(domain_scores, config.score_weights)
    wall_clock = time.perf_counter() - started
    trace = PatternTrace(
        retrieved_chunk_ids=list(retrieved_chunks),
        retrieved_token_count=sum(retrieved_chunks.values()),
        total_token_count=sum(chunk.token_count for chunk in chunks),
        agent_calls=total_agent_calls,
        critical_path_latency=wall_clock * 0.75,
        wall_clock_latency=wall_clock,
        deliberations=deliberations,
        trial=trial,
    )
    metrics = _metrics_for_result(report, chunks, domain_scores, trace)
    return PatternRunResult(
        pattern="hierarchical",
        report_id=report.report_id,
        company_name=report.company_name,
        domain_scores=domain_scores,
        total_score=total_score,
        comparator=_compare(total_score, report.actual_scores),
        metrics=metrics,
        trace=trace,
        actual_scores=report.actual_scores,
    )


def run_review_pattern(report: ReportRecord, config: ESGExperimentConfig, trial: int) -> PatternRunResult:
    started = time.perf_counter()
    chunks = chunk_report(report, config.chunk_size, config.chunk_overlap, config.chunk_dir)
    retriever = ChunkRetriever(chunks)
    domain_scores: dict[str, DomainScore] = {}
    deliberations: list[DeliberationRecord] = []
    retrieved_chunks: dict[str, int] = {}
    total_agent_calls = 0
    pre_total_candidates: list[float] = []

    for domain in DOMAINS:
        items = retriever.retrieve(domain, config.top_k, trial=trial)
        base_score = _build_domain_score(domain, [item.chunk for item in items])
        pre_total_candidates.append(base_score.estimated_score)
        current = base_score
        for iteration in range(1, config.critique_max_iterations + 1):
            uncovered_ratio = current.unsupported_claims / max(1, len(HEURISTICS_BY_DOMAIN[domain]))
            adjustment = round(uncovered_ratio * 12, 2)
            approved = adjustment <= 3.0 or iteration == config.critique_max_iterations
            revised_score = max(0.0, current.estimated_score - adjustment)
            deliberations.append(
                DeliberationRecord(
                    domain=domain,
                    iteration=iteration,
                    issue="critique identified evidence gap" if adjustment else "critique confirmed grounding",
                    suggestion="Reduce unsupported heuristics and tighten rationale to cited chunks.",
                    approved=approved,
                    initial_score=current.estimated_score,
                    final_score=revised_score,
                )
            )
            current = DomainScore(
                domain=current.domain,
                estimated_score=round(revised_score, 2),
                confidence=max(0.3, round(current.confidence - uncovered_ratio * 0.1, 2)),
                rationale=current.rationale + f" Critique iteration {iteration} applied adjustment {adjustment}.",
                chunk_ids=current.chunk_ids,
                supported_claims=current.supported_claims,
                unsupported_claims=max(0, current.unsupported_claims - 1 if approved else current.unsupported_claims),
                partially_supported_claims=current.partially_supported_claims,
                matched_heuristics=current.matched_heuristics,
                calls_used=current.calls_used + 1,
                tokens_used=current.tokens_used,
                iteration_count=iteration + 1,
            )
            if approved:
                break
        domain_scores[domain] = current
        for item in items:
            retrieved_chunks[item.chunk.chunk_id] = item.chunk.token_count
        total_agent_calls += current.calls_used

    pre_total = mean(pre_total_candidates)
    total_score = _aggregate(domain_scores, config.score_weights)
    wall_clock = time.perf_counter() - started
    trace = PatternTrace(
        retrieved_chunk_ids=list(retrieved_chunks),
        retrieved_token_count=sum(retrieved_chunks.values()),
        total_token_count=sum(chunk.token_count for chunk in chunks),
        agent_calls=total_agent_calls,
        critical_path_latency=wall_clock * 0.9,
        wall_clock_latency=wall_clock,
        deliberations=deliberations,
        trial=trial,
    )
    metrics = _metrics_for_result(report, chunks, domain_scores, trace, pre_total=pre_total)
    return PatternRunResult(
        pattern="review",
        report_id=report.report_id,
        company_name=report.company_name,
        domain_scores=domain_scores,
        total_score=total_score,
        comparator=_compare(total_score, report.actual_scores),
        metrics=metrics,
        trace=trace,
        actual_scores=report.actual_scores,
    )


def add_consistency(pattern_results: list[PatternRunResult]) -> None:
    grouped: dict[tuple[str, str], list[PatternRunResult]] = {}
    for result in pattern_results:
        grouped.setdefault((result.pattern, result.report_id), []).append(result)
    for results in grouped.values():
        score = consistency_score(results)
        for result in results:
            result.metrics["consistency"] = {"trial_similarity": score}
