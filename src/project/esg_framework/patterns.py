from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from project.esg_framework.chunking import split_report_to_chunks
from project.esg_framework.heuristics import ALL_DOMAINS, DOMAIN_SCORE_PRIORS
from project.esg_framework.metrics import (
    Timer,
    accuracy_from_mae,
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
import logging

# Configure a module-level logger. If the application already configures
# logging this will be a no-op; otherwise we provide a sensible default so
# logs are visible during development.
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)

# PATTERN_PARALLEL_MAX_WORKERS: threadpool size for the parallel/concurrent pattern
PARALLEL_MAX_WORKERS = int(os.getenv("PATTERN_PARALLEL_MAX_WORKERS", "3"))
# PATTERN_PARALLEL_MAX_CHUNKS: max chunks retrieved per domain in parallel pattern
PARALLEL_MAX_CHUNKS = int(os.getenv("PATTERN_PARALLEL_MAX_CHUNKS", "6"))

# PATTERN_HANDOFF_MAX_WORKERS: threadpool size for hierarchical handoff worker groups
HANDOFF_MAX_WORKERS = int(os.getenv("PATTERN_HANDOFF_MAX_WORKERS", "2"))
# PATTERN_HANDOFF_WORKER_GROUPS: number of worker groups to split work into (hierarchy)
HANDOFF_WORKER_GROUPS = int(os.getenv("PATTERN_HANDOFF_WORKER_GROUPS", "2"))
# PATTERN_HANDOFF_MAX_CHUNKS: max chunks retrieved per domain in handoff pattern
HANDOFF_MAX_CHUNKS = int(os.getenv("PATTERN_HANDOFF_MAX_CHUNKS", "8"))

# PATTERN_REVIEW_MAX_CHUNKS: initial max chunks retrieved per domain in review/critique
REVIEW_MAX_CHUNKS = int(os.getenv("PATTERN_REVIEW_MAX_CHUNKS", "6"))
# PATTERN_REVIEW_CRITIQUE_CHUNKS: chunks used specifically for critique adjustments
REVIEW_CRITIQUE_CHUNKS = int(os.getenv("PATTERN_REVIEW_CRITIQUE_CHUNKS", "4"))
# PATTERN_REVIEW_MAX_ROUNDS: maximum critique-adjust rounds per domain
REVIEW_MAX_ROUNDS = int(os.getenv("PATTERN_REVIEW_MAX_ROUNDS", "3"))

# Total agent call estimates used for latency/efficiency metrics.
# Computed programmatically from the number of domains and pattern-specific config.
NUM_DOMAINS = len(ALL_DOMAINS)  # 3: environmental, social, governance
PARALLEL_TOTAL_AGENT_CALLS = NUM_DOMAINS * 1  # 1 call per domain
HANDOFF_TOTAL_AGENT_CALLS = NUM_DOMAINS * HANDOFF_WORKER_GROUPS  # N workers per domain
REVIEW_TOTAL_AGENT_CALLS = NUM_DOMAINS * (1 + REVIEW_MAX_ROUNDS)  # 1 initial + N critique rounds per domain


def _ground_truth_weighted_total(report: ReportRecord, weights: dict[str, float] | None = None) -> float:
    weights = weights or {"environmental": 1.0, "social": 1.0, "governance": 1.0}
    numerator = 0.0
    denominator = 0.0
    for domain in ALL_DOMAINS:
        w = weights.get(domain, 1.0)
        numerator += report.ground_truth.get(domain, 0.0) * w
        denominator += w
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 2)


def _compare_to_ground_truth(total_score: float, report: ReportRecord) -> dict[str, float | str]:
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

    predicted_total = aggregate_total_score(domain_scores)
    actual_total = _ground_truth_weighted_total(report)
    mae_total = mae(predicted_total, actual_total)
    return {
        **coverage,
        "mae_total": mae_total,
        "accuracy": accuracy_from_mae(mae_total),
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


def run_parallel_pattern(report: ReportRecord, chunk_store: ChunkStore) -> ReportRunResult:
    timer = Timer()
    timer.start("parse_report")
    chunks = split_report_to_chunks(report)
    chunk_store.put(report.report_id, chunks)
    timer.stop("parse_report")

    trace: list[RetrievalEvent] = []

    def _score(domain: str) -> DomainScore:
        timer.start(f"score_{domain}")
        logger.info("%s: analyst starting scoring", f"{domain}_analyst")
        # retrieve a bounded number of chunks relevant to the domain. The
        # PARALLEL_MAX_CHUNKS constant controls the breadth of retrieval and
        # therefore the amount of evidence available to the domain scorer.
        selected = retrieve_for_domain(chunks, domain, max_chunks=PARALLEL_MAX_CHUNKS)
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
        logger.info("%s: analyst finished scoring (used %d chunks). result=%.2f", f"{domain}_analyst", len(selected), score.estimated_score)
        return score

    # Parallelize domain scoring to simulate concurrent agents working at once.
    # PARALLEL_MAX_WORKERS controls how many threads run simultaneously.
    logger.info("Submitting parallel scoring tasks for domains: %s", ",".join(ALL_DOMAINS))
    with ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as pool:
        futures = {pool.submit(_score, domain): domain for domain in ALL_DOMAINS}
        domain_scores = {}
        for fut in as_completed(futures):
            domain_scores[futures[fut]] = fut.result()
    logger.info("Parallel scoring complete for domains: %s", ",".join(domain_scores.keys()))

    total_score = aggregate_total_score(domain_scores)
    comparison = _compare_to_ground_truth(total_score, report)
    metrics = _metric_bundle(
        report,
        domain_scores,
        trace,
        chunks,
        timer,
        total_agent_calls=PARALLEL_TOTAL_AGENT_CALLS,
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

    # Define small agent-like classes to represent manager and workers.
    # These are lightweight in-process agents that encapsulate a name and
    # a run() method so the handoff pattern uses explicit agents rather
    # than anonymous thread tasks. They call the existing
    # `estimate_domain_score` logic for scoring.
    class WorkerAgent:
        def __init__(self, name: str):
            self.name = name

        def run(self, domain: str, chunks_subset) -> DomainScore:
            logger.info("%s: worker starting processing %d chunks for domain %s", self.name, len(chunks_subset), domain)
            out = estimate_domain_score(domain, chunks_subset)
            # annotate rationale with worker identity to preserve auditability
            out.rationale = f"Processed by {self.name}. " + out.rationale
            out.retrieved_chunk_ids = [c.chunk_id for c in chunks_subset]
            out.used_chunk_ids = [c.chunk_id for c in chunks_subset]
            logger.info("%s: worker finished processing for domain %s (score=%.2f)", self.name, domain, out.estimated_score)
            return out

    class ManagerAgent:
        def __init__(self, name: str):
            self.name = name

        def delegate(self, workers: list[WorkerAgent], domain: str, groups: list[list]) -> list[tuple["WorkerAgent", list, DomainScore]]:
            # Submit worker work in parallel and return their DomainScore outputs.
            outputs: list[tuple[WorkerAgent, list, DomainScore]] = []
            logger.info("%s: delegating %d groups to workers: %s", self.name, len(groups), ",".join(w.name for w in workers))
            with ThreadPoolExecutor(max_workers=HANDOFF_MAX_WORKERS) as pool:
                futures = {pool.submit(w.run, domain, grp): (w, grp) for w, grp in zip(workers, groups)}
                for fut in as_completed(futures):
                    w, grp = futures[fut]
                    try:
                        res = fut.result()
                        outputs.append((w, grp, res))
                        logger.info("%s: worker %s completed delegation task for domain %s", self.name, w.name, domain)
                    except Exception as e:
                        logger.exception("%s: worker %s failed while processing domain %s: %s", self.name, w.name, domain, e)
            return outputs

    for domain in ALL_DOMAINS:
        timer.start(f"score_{domain}")
        # retrieve a slightly larger set of chunks for handoff, since the
        # hierarchical aggregation spreads the evidence across worker groups.
        selected = retrieve_for_domain(chunks, domain, max_chunks=HANDOFF_MAX_CHUNKS)

        # Split the selected chunks evenly into HANDOFF_WORKER_GROUPS groups.
        # The slice idiom selected[i::n] distributes items in a round-robin way
        # to each worker group which emulates dividing responsibilities.
        worker_groups = [selected[i::HANDOFF_WORKER_GROUPS] for i in range(HANDOFF_WORKER_GROUPS)]

        # Create explicit manager and worker agents and delegate work.
        manager_name = f"{domain}_manager"
        manager_agent = ManagerAgent(manager_name)

        # Create WorkerAgent instances for each non-empty group.
        workers = [WorkerAgent(f"{manager_name}_worker_{i}") for i in range(len(worker_groups)) if worker_groups[i]]
        groups = [group for group in worker_groups if group]

        # Manager delegates the groups to the worker agents and collects outputs.
        worker_outputs = manager_agent.delegate(workers, domain, groups)

        # Record a RetrievalEvent per worker so the trace shows which worker
        # handled which chunks, as well as a manager-level handoff event.
        for w, grp, outp in worker_outputs:
            trace.append(
                RetrievalEvent(
                    agent_name=w.name,
                    domain=domain,
                    query=f"{domain} worker processing",
                    retrieved_chunk_ids=[c.chunk_id for c in grp],
                    used_chunk_ids=outp.used_chunk_ids,
                )
            )

        # Merge worker group outputs by averaging their numeric estimates and
        # concatenating rationales. This is a simple aggregation strategy that
        # approximates hierarchical consensus.
        if worker_outputs:
            weighted_parts = []
            for _, grp, item in worker_outputs:
                weighted_parts.append((max(0.1, item.confidence) * max(1, len(grp)), item))
            total_weight = sum(weight for weight, _ in weighted_parts) or 1.0
            merged_score = round(sum(weight * item.estimated_score for weight, item in weighted_parts) / total_weight, 2)
            merged_conf = round(sum(weight * item.confidence for weight, item in weighted_parts) / total_weight, 3)
            merged_conf = round(min(0.95, merged_conf), 3)
            merged_rationale = " ".join(item.rationale for _, _, item in worker_outputs)
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
                agent_name=manager_name,
                domain=domain,
                query=f"{domain} handoff (delegated to {len(worker_groups)} workers)",
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
        total_agent_calls=HANDOFF_TOTAL_AGENT_CALLS,
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


def run_review_critique_pattern(report: ReportRecord, chunk_store: ChunkStore, max_rounds: int = REVIEW_MAX_ROUNDS) -> ReportRunResult:
    """
    Run the review & critique pattern where:
    1. Domain agent analyzes chunks and produces initial score with rationale
    2. Review & critique agent evaluates the rationale and provides feedback
    3. Domain agent's score is adjusted based on the critique feedback
    
    This pattern uses an LLM-based critique agent that:
    - Reviews the candidate's score and rationale
    - Examines additional evidence chunks
    - Identifies gaps, overstatements, and biases
    - Recommends an adjusted score based on ALL evidence
    """
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
    total_critique_feedback: list[dict] = []

    for domain in ALL_DOMAINS:
        timer.start(f"score_{domain}")
        # Initial retrieval for the review-and-critique pattern. We fetch a
        # bounded set of chunks for the initial estimate and then separately
        # fetch smaller sets used specifically for critique rounds.
        selected = retrieve_for_domain(chunks, domain, max_chunks=REVIEW_MAX_CHUNKS)
        current = estimate_domain_score(domain, selected)
        current.retrieved_chunk_ids = [chunk.chunk_id for chunk in selected]
        current.used_chunk_ids = [chunk.chunk_id for chunk in selected]
        rounds = 0
        domain_feedback: list[dict] = []

        while rounds < max_rounds:
            rounds += 1
            # For each critique round we retrieve additional, focused evidence
            # (REVIEW_CRITIQUE_CHUNKS) and ask the critique/adjust routine to
            # propose a revised estimate. If the scorer adjusts its estimate we
            # consider that a conflict+resolution; otherwise we stop early.
            already_used = set(current.used_chunk_ids)
            critique_chunks = retrieve_for_domain(
                chunks,
                domain,
                max_chunks=REVIEW_CRITIQUE_CHUNKS,
                exclude_chunk_ids=already_used,
                min_score=0.8,
            )
            if not critique_chunks:
                break
            
            revised, stats = critique_and_adjust(domain, current, critique_chunks)
            
            # Record feedback from this critique round
            domain_feedback.append({
                "round": rounds,
                "adjusted": stats.get("adjusted", False),
                "gap": stats.get("gap", 0.0),
                "feedback": stats.get("feedback", ""),
                "adjustment_direction": stats.get("adjustment_direction", "no_change"),
                "missing_evidence": stats.get("missing_evidence", []),
                "overstated_claims": stats.get("overstated_claims", []),
            })
            
            if stats["adjusted"]:
                conflicts += 1
                resolved += 1
                current = revised
            else:
                break

        total_critique_feedback.append({
            "domain": domain,
            "rounds": rounds,
            "feedback": domain_feedback,
        })
        
        dominance_counter[domain] = 1 if rounds == 1 else 0
        trace.append(
            RetrievalEvent(
                agent_name=f"{domain}_analyst_with_critique",
                domain=domain,
                query=f"{domain} critique (rounds={rounds})",
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
        total_agent_calls=REVIEW_TOTAL_AGENT_CALLS,
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
        metadata={
            "conflicts": conflicts,
            "resolved": resolved,
            "critique_feedback": total_critique_feedback,
        },
    )
