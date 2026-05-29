from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Callable

from project.esg_framework.chunking import split_report_to_chunks
from project.esg_framework.heuristics import ALL_DOMAINS
from project.esg_framework.metrics import (
    Timer,
    coverage_metrics,
    deliberation_quality,
    hallucination_rate,
    inter_agent_agreement,
    judge_accuracy_stub,
    latency_efficiency,
    mae,
)
from project.esg_framework.models import DomainScore, ReportRecord, ReportRunResult, RetrievalEvent
from project.esg_framework.retrieval import ChunkStore, retrieve_for_domain
from project.esg_framework.scoring import aggregate_confidence, aggregate_total_score, critique_and_adjust, estimate_domain_score, score_label


DomainScorer = Callable[[str], DomainScore]


def _compare_to_ground_truth(total_score: float, report: ReportRecord) -> dict[str, float | str]:
    actual = report.ground_truth.get("total", 0.0)
    err = abs(total_score - actual)
    pct = (err / actual) * 100 if actual else 0.0
    direction = "over" if total_score > actual else "under"
    return {
        "actual_total": round(actual, 2),
        "absolute_error": round(err, 4),
        "percentage_error": round(pct, 4),
        "direction": direction,
    }


def _metric_bundle(
    report: ReportRecord,
    domain_scores: dict[str, DomainScore],
    retrieval_trace: list[RetrievalEvent],
    chunks,
    timer: Timer,
    total_agent_calls: int,
    deliberation_stats: dict[str, float],
) -> dict[str, float]:
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

    return {
        **coverage,
        "mae_total": mae(aggregate_total_score(domain_scores), report.ground_truth.get("total", 0.0)),
        "judge_accuracy": judge_accuracy_stub(domain_scores),
        "agreement_fleiss_kappa": agreement["fleiss_kappa"],
        "agreement_pairwise_pearson": agreement["pairwise_pearson"],
        "hallucination_unsupported": hallucination["unsupported_claim_rate"],
        "hallucination_partial": hallucination["partial_support_rate"],
        **latency,
        **deliberation_stats,
    }


def run_parallel_pattern(report: ReportRecord, chunk_store: ChunkStore) -> ReportRunResult:
    timer = Timer()
    timer.start("parse_report")
    chunks = split_report_to_chunks(report)
    chunk_store.put(report.report_id, chunks)
    timer.stop("parse_report")

    trace: list[RetrievalEvent] = []

    def _score(domain: str) -> DomainScore:
        timer.start(f"score_{domain}")
        selected = retrieve_for_domain(chunks, domain)
        score = estimate_domain_score(domain, selected)
        score.retrieved_chunk_ids = [chunk.chunk_id for chunk in selected]
        score.used_chunk_ids = [chunk.chunk_id for chunk in selected]
        trace.append(
            RetrievalEvent(
                agent_name=f"{domain}_analyst",
                domain=domain,
                query=domain,
                retrieved_chunk_ids=score.retrieved_chunk_ids,
                used_chunk_ids=score.used_chunk_ids,
            )
        )
        timer.stop(f"score_{domain}")
        return score

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_score, domain): domain for domain in ALL_DOMAINS}
        domain_scores = {}
        for fut in as_completed(futures):
            domain_scores[futures[fut]] = fut.result()

    total_score = aggregate_total_score(domain_scores)
    comparison = _compare_to_ground_truth(total_score, report)
    metrics = _metric_bundle(
        report,
        domain_scores,
        trace,
        chunks,
        timer,
        total_agent_calls=3,
        deliberation_stats=deliberation_quality(0, 1, 0, 1.0),
    )

    return ReportRunResult(
        report_id=report.report_id,
        pattern="parallel_concurrent",
        domain_scores=domain_scores,
        total_score=total_score,
        confidence=aggregate_confidence(domain_scores),
        retrieval_trace=trace,
        comparison=comparison,
        metrics=metrics,
    )


def run_handoff_pattern(report: ReportRecord, chunk_store: ChunkStore) -> ReportRunResult:
    timer = Timer()
    timer.start("parse_report")
    chunks = split_report_to_chunks(report)
    chunk_store.put(report.report_id, chunks)
    timer.stop("parse_report")

    trace: list[RetrievalEvent] = []
    domain_scores: dict[str, DomainScore] = {}

    for domain in ALL_DOMAINS:
        timer.start(f"score_{domain}")
        selected = retrieve_for_domain(chunks, domain, max_chunks=10)
        worker_groups = [selected[i::2] for i in range(2)]

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(estimate_domain_score, domain, group) for group in worker_groups if group]
            worker_outputs = [future.result() for future in as_completed(futures)]

        if worker_outputs:
            merged_score = round(mean(item.estimated_score for item in worker_outputs), 2)
            merged_conf = round(mean(item.confidence for item in worker_outputs), 3)
            merged_rationale = " ".join(item.rationale for item in worker_outputs)
        else:
            fallback = estimate_domain_score(domain, selected)
            merged_score = fallback.estimated_score
            merged_conf = fallback.confidence
            merged_rationale = fallback.rationale

        out = DomainScore(
            estimated_score=merged_score,
            confidence=merged_conf,
            rationale=f"Hierarchical worker aggregation. {merged_rationale}",
            label=score_label(merged_score),
            retrieved_chunk_ids=[chunk.chunk_id for chunk in selected],
            used_chunk_ids=[chunk.chunk_id for group in worker_groups for chunk in group],
        )

        trace.append(
            RetrievalEvent(
                agent_name=f"{domain}_analyst_handoff",
                domain=domain,
                query=f"{domain} handoff",
                retrieved_chunk_ids=out.retrieved_chunk_ids,
                used_chunk_ids=out.used_chunk_ids,
            )
        )
        domain_scores[domain] = out
        timer.stop(f"score_{domain}")

    total_score = aggregate_total_score(domain_scores)
    comparison = _compare_to_ground_truth(total_score, report)
    metrics = _metric_bundle(
        report,
        domain_scores,
        trace,
        chunks,
        timer,
        total_agent_calls=9,
        deliberation_stats=deliberation_quality(0, 1, 0, 1.0),
    )

    return ReportRunResult(
        report_id=report.report_id,
        pattern="handoff_hierarchical",
        domain_scores=domain_scores,
        total_score=total_score,
        confidence=aggregate_confidence(domain_scores),
        retrieval_trace=trace,
        comparison=comparison,
        metrics=metrics,
    )


def run_review_critique_pattern(report: ReportRecord, chunk_store: ChunkStore, max_rounds: int = 3) -> ReportRunResult:
    # max_rounds controls the maximum critique-adjust iterations per ESG domain scorer.
    timer = Timer()
    timer.start("parse_report")
    chunks = split_report_to_chunks(report)
    chunk_store.put(report.report_id, chunks)
    timer.stop("parse_report")

    trace: list[RetrievalEvent] = []
    domain_scores: dict[str, DomainScore] = {}
    conflicts = 0
    resolved = 0
    dominance_counter: dict[str, int] = {}

    for domain in ALL_DOMAINS:
        timer.start(f"score_{domain}")
        selected = retrieve_for_domain(chunks, domain, max_chunks=8)
        current = estimate_domain_score(domain, selected)
        current.retrieved_chunk_ids = [chunk.chunk_id for chunk in selected]
        current.used_chunk_ids = [chunk.chunk_id for chunk in selected]
        rounds = 0

        while rounds < max_rounds:
            rounds += 1
            critique_chunks = retrieve_for_domain(chunks, domain, max_chunks=5)
            revised, stats = critique_and_adjust(domain, current, critique_chunks)
            if stats["adjusted"]:
                conflicts += 1
                resolved += 1
                current = revised
            else:
                break

        dominance_counter[domain] = 1 if rounds == 1 else 0
        trace.append(
            RetrievalEvent(
                agent_name=f"{domain}_analyst_with_critique",
                domain=domain,
                query=f"{domain} critique",
                retrieved_chunk_ids=current.retrieved_chunk_ids,
                used_chunk_ids=current.used_chunk_ids,
            )
        )
        domain_scores[domain] = current
        timer.stop(f"score_{domain}")

    dominance_ratio = (max(dominance_counter.values()) / len(dominance_counter)) if dominance_counter else 0.0
    deliberation = deliberation_quality(conflicts, len(ALL_DOMAINS), resolved, dominance_ratio)

    total_score = aggregate_total_score(domain_scores)
    comparison = _compare_to_ground_truth(total_score, report)
    metrics = _metric_bundle(
        report,
        domain_scores,
        trace,
        chunks,
        timer,
        total_agent_calls=6,
        deliberation_stats=deliberation,
    )

    return ReportRunResult(
        report_id=report.report_id,
        pattern="review_critique",
        domain_scores=domain_scores,
        total_score=total_score,
        confidence=aggregate_confidence(domain_scores),
        retrieval_trace=trace,
        comparison=comparison,
        metrics=metrics,
        metadata={"conflicts": conflicts, "resolved": resolved},
    )
