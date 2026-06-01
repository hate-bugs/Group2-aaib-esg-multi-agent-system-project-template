from __future__ import annotations

import json
import re
from statistics import mean

MAX_SCORE = 20.0
BASE_CONFIDENCE = 0.35
MAX_CONFIDENCE = 0.95
CONFIDENCE_COVERAGE_WEIGHT = 0.4
MIN_TOKEN_NORMALIZER = 20
MAX_LLM_CHUNK_CHARS = 2500

from project.esg_framework.heuristics import (
    DOMAIN_HEURISTICS,
    DOMAIN_KEYWORDS,
    DOMAIN_SCORE_PRIORS,
    DOMAIN_SIGNAL_KEYWORDS,
    SCORING_RUBRIC,
)
from project.esg_framework.models import Chunk, DomainScore


def score_label(score: float) -> str:
    if score < 8:
        return "low"
    if score < 16:
        return "medium"
    return "high"


def _normalize_score(raw_hits: int, token_count: int) -> float:
    density = raw_hits / max(token_count, 1)
    calibrated = min(MAX_SCORE, max(0.0, density * 120.0))
    return round(calibrated, 2)


def _heuristic_domain_score(domain: str, chunks: list[Chunk]) -> DomainScore:
    keywords = DOMAIN_KEYWORDS[domain]
    token_total = sum(chunk.token_count for chunk in chunks)
    hits = 0
    for chunk in chunks:
        lower = chunk.text.lower()
        hits += sum(lower.count(keyword) for keyword in keywords)

    score = _normalize_score(hits, token_total)
    coverage_hint = min(1.0, len(chunks) / 8)
    keyword_density_boost = hits / max(MIN_TOKEN_NORMALIZER, token_total)
    coverage_boost = CONFIDENCE_COVERAGE_WEIGHT * coverage_hint
    confidence_raw = BASE_CONFIDENCE + keyword_density_boost + coverage_boost
    confidence = round(min(MAX_CONFIDENCE, confidence_raw), 3)
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


def _extract_first_json_dict(raw_text: str) -> dict | None:
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        pass

    match = re.search(r"\{.*}", raw_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _build_domain_prompt(domain: str, chunks: list[Chunk]) -> str:
    evidence = []
    for chunk in chunks:
        text = chunk.text.strip().replace("\n", " ")
        evidence.append(f"[{chunk.chunk_id}] {text[:MAX_LLM_CHUNK_CHARS]}")
    evidence_block = "\n\n".join(evidence) if evidence else "[NO_EVIDENCE]"

    rubric = "\n".join(f"- {band}: {desc}" for band, desc in SCORING_RUBRIC.items())
    return (
        "You are an ESG scoring assistant.\n"
        f"Domain: {domain}\n"
        "Estimate a score from 0 to 20 using ONLY the evidence chunks.\n"
        "Use only preprocessed_content evidence. Do not assume facts that are not explicitly stated.\n"
        f"Apply this domain heuristic checklist: {DOMAIN_HEURISTICS[domain]}\n"
        "Use this scoring rubric exactly:\n"
        f"{rubric}\n"
        "Scoring discipline:\n"
        "1) Start near a neutral prior and move up only with explicit evidence.\n"
        "2) Keep score <= 8 if evidence is generic or lacks measurable targets/results.\n"
        "3) Keep score <= 12 unless evidence shows clear governance ownership plus risk methods and monitoring outcomes.\n"
        "4) Use > 16 only for exceptional, comprehensive, and quantified evidence across strategy, governance, and risk management.\n"
        "Return strict JSON with keys: estimated_score (number), confidence (number 0..1), rationale (string).\n"
        "Do not include markdown.\n\n"
        f"Evidence:\n{evidence_block}"
    )


def _evidence_strength(domain: str, chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    keywords = DOMAIN_KEYWORDS[domain]
    token_total = sum(chunk.token_count for chunk in chunks)
    text = " ".join(chunk.text.lower() for chunk in chunks)
    hits = sum(text.count(token) for token in keywords)
    diversity = sum(1 for token in keywords if token in text)
    density = hits / max(token_total, 1)
    diversity_ratio = diversity / max(len(keywords), 1)
    coverage_ratio = min(1.0, len(chunks) / 10)
    quality = _evidence_quality_score(chunks)
    strength = (
        (0.25 * min(1.0, density * 35.0))
        + (0.25 * diversity_ratio)
        + (0.1 * coverage_ratio)
        + (0.4 * quality)
    )
    return max(0.0, min(1.0, strength))


def _evidence_quality_score(chunks: list[Chunk]) -> float:
    if not chunks:
        return 0.0
    text = " ".join(chunk.text.lower() for chunk in chunks)
    if not text.strip():
        return 0.0

    category_scores: list[float] = []
    for keywords in DOMAIN_SIGNAL_KEYWORDS.values():
        hits = sum(1 for keyword in keywords if keyword in text)
        category_scores.append(hits / max(len(keywords), 1))
    return max(0.0, min(1.0, mean(category_scores)))


def _calibrate_domain_score(
    domain: str,
    base_score: DomainScore,
    heuristic_score: DomainScore,
    chunks: list[Chunk],
) -> DomainScore:
    prior = DOMAIN_SCORE_PRIORS.get(domain, 8.0)
    strength = _evidence_strength(domain, chunks)
    quality = _evidence_quality_score(chunks)

    anchor = (0.5 * heuristic_score.estimated_score) + (0.5 * prior)
    confidence_factor = max(0.0, min(1.0, base_score.confidence))
    llm_delta = base_score.estimated_score - anchor
    blend_alpha = 0.22 + (0.5 * strength) + (0.18 * confidence_factor)
    if llm_delta > 0 and quality < 0.35:
        blend_alpha *= 0.65
    blend_alpha = max(0.14, min(0.78, blend_alpha))
    calibrated = anchor + (blend_alpha * llm_delta)

    # Keep optimistic tails under control unless evidence quality is genuinely strong.
    evidence_cap = min(MAX_SCORE, prior + 4.0 + (11.0 * quality))
    if calibrated > evidence_cap:
        calibrated = evidence_cap

    calibrated_score = round(max(0.0, min(MAX_SCORE, calibrated)), 2)

    calibrated_confidence = min(
        MAX_CONFIDENCE,
        0.2 + (0.45 * confidence_factor) + (0.2 * strength) + (0.1 * quality),
    )

    rationale = (
        f"Used {len(chunks)} retrieved chunks. Detected evidence strength={strength:.2f}, quality={quality:.2f}. "
        f"Detected calibration anchor from heuristic and healthcare prior ({prior:.2f}) with rubric gating. "
        f"Used base rationale: {base_score.rationale}"
    )

    return DomainScore(
        estimated_score=calibrated_score,
        confidence=round(calibrated_confidence, 3),
        rationale=rationale,
        label=score_label(calibrated_score),
    )


def _llm_domain_score(domain: str, chunks: list[Chunk]) -> DomainScore | None:
    try:
        from project.llm_config import llm
    except Exception:
        return None

    prompt = _build_domain_prompt(domain, chunks)
    response = None
    for method_name in ("call", "invoke", "predict"):
        method = getattr(llm, method_name, None)
        if callable(method):
            try:
                response = method(prompt)
                break
            except Exception:
                continue
    if response is None:
        return None

    if isinstance(response, dict):
        parsed = response
    else:
        parsed = _extract_first_json_dict(str(response))
    if not isinstance(parsed, dict):
        return None

    try:
        estimated_score = max(0.0, min(MAX_SCORE, float(parsed.get("estimated_score", 0.0))))
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", BASE_CONFIDENCE))))
    except (TypeError, ValueError):
        return None
    rationale = str(parsed.get("rationale", "")).strip()
    if not rationale:
        rationale = f"LLM estimated {domain} score from retrieved evidence chunks."

    return DomainScore(
        estimated_score=round(estimated_score, 2),
        confidence=round(confidence, 3),
        rationale=rationale,
        label=score_label(estimated_score),
    )


def estimate_domain_score(domain: str, chunks: list[Chunk]) -> DomainScore:
    heuristic = _heuristic_domain_score(domain, chunks)
    llm_score = _llm_domain_score(domain, chunks)
    if llm_score is not None:
        llm_score.rationale = f"[llm] {llm_score.rationale}"
        calibrated = _calibrate_domain_score(domain, llm_score, heuristic, chunks)
        calibrated.retrieved_chunk_ids = llm_score.retrieved_chunk_ids
        calibrated.used_chunk_ids = llm_score.used_chunk_ids
        return calibrated

    heuristic.rationale = f"[heuristic_fallback] {heuristic.rationale}"
    calibrated = _calibrate_domain_score(domain, heuristic, heuristic, chunks)
    calibrated.retrieved_chunk_ids = heuristic.retrieved_chunk_ids
    calibrated.used_chunk_ids = heuristic.used_chunk_ids
    return calibrated


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
            f"Detected critique gap={gap:.2f}. Used initial rationale: {candidate.rationale} "
            f"Used critique rationale: {critique.rationale}"
        ),
        label=score_label(merged_score),
        retrieved_chunk_ids=list(dict.fromkeys(candidate.retrieved_chunk_ids + critique.retrieved_chunk_ids)),
        used_chunk_ids=list(dict.fromkeys(candidate.used_chunk_ids + critique.used_chunk_ids)),
    )
    return updated, {"adjusted": True, "gap": round(gap, 2)}
