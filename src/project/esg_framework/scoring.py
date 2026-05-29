from __future__ import annotations

from statistics import mean

from project.esg_framework.heuristics import DOMAIN_HEURISTICS, DOMAIN_KEYWORDS
from project.esg_framework.models import Chunk, DomainScore


def score_label(score: float) -> str:
    if score < 8:
        return "low"
    if score < 16:
        return "medium"
    return "high"


def _normalize_score(raw_hits: int, token_count: int) -> float:
    density = raw_hits / max(token_count, 1)
    calibrated = min(20.0, max(0.0, density * 160.0))
    return round(calibrated, 2)


def estimate_domain_score(domain: str, chunks: list[Chunk]) -> DomainScore:
    keywords = DOMAIN_KEYWORDS[domain]
    token_total = sum(chunk.token_count for chunk in chunks)
    hits = 0
    for chunk in chunks:
        lower = chunk.text.lower()
        hits += sum(lower.count(keyword) for keyword in keywords)

    score = _normalize_score(hits, token_total)
    coverage_hint = min(1.0, len(chunks) / 8)
    confidence = round(min(0.95, 0.35 + (hits / max(20, token_total)) + 0.4 * coverage_hint), 3)
    rationale = (
        f"Used {len(chunks)} retrieved chunks ({token_total} tokens). "
        f"Detected {hits} domain-keyword hits based on heuristic: {DOMAIN_HEURISTICS[domain]}"
    )

    return DomainScore(
        estimated_score=score,
        confidence=confidence,
        rationale=rationale,
        label=score_label(score),
    )


def aggregate_total_score(domain_scores: dict[str, DomainScore], weights: dict[str, float] | None = None) -> float:
    weights = weights or {"environmental": 1.0, "social": 1.0, "governance": 1.0}
    numerator = 0.0
    denominator = 0.0
    for domain, output in domain_scores.items():
        w = weights.get(domain, 1.0)
        numerator += output.estimated_score * w
        denominator += w
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 2)


def aggregate_confidence(domain_scores: dict[str, DomainScore]) -> float:
    if not domain_scores:
        return 0.0
    return round(mean(score.confidence for score in domain_scores.values()), 3)


def critique_and_adjust(domain: str, candidate: DomainScore, critique_chunks: list[Chunk]) -> tuple[DomainScore, dict[str, float | bool]]:
    critique = estimate_domain_score(domain, critique_chunks)
    gap = abs(candidate.estimated_score - critique.estimated_score)
    needs_adjustment = gap > 2.5 and critique.confidence >= candidate.confidence
    if not needs_adjustment:
        return candidate, {"adjusted": False, "gap": round(gap, 2)}

    merged_score = round((candidate.estimated_score + critique.estimated_score) / 2.0, 2)
    merged_conf = round(max(candidate.confidence, critique.confidence), 3)
    updated = DomainScore(
        estimated_score=merged_score,
        confidence=merged_conf,
        rationale=(
            f"Initial score revised after critique. Original rationale: {candidate.rationale} "
            f"Critique rationale: {critique.rationale}"
        ),
        label=score_label(merged_score),
        retrieved_chunk_ids=list(dict.fromkeys(candidate.retrieved_chunk_ids + critique.retrieved_chunk_ids)),
        used_chunk_ids=list(dict.fromkeys(candidate.used_chunk_ids + critique.used_chunk_ids)),
    )
    return updated, {"adjusted": True, "gap": round(gap, 2)}
