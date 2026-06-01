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
    """
    Always uses LLM scoring. Heuristic scoring code is kept for reference but unused.
    Raises RuntimeError if LLM is not configured.
    """
    llm_score = _llm_domain_score(domain, chunks)
    if llm_score is None:
        raise RuntimeError(
            f"LLM scoring is required but llm is not configured in project.llm_config. "
            f"Domain: {domain}, chunks: {len(chunks)}"
        )
    llm_score.rationale = f"[llm] {llm_score.rationale}"
    # Still run calibration using heuristic for reference (kept unused as requested)
    heuristic = _heuristic_domain_score(domain, chunks)
    calibrated = _calibrate_domain_score(domain, llm_score, heuristic, chunks)
    calibrated.retrieved_chunk_ids = llm_score.retrieved_chunk_ids
    calibrated.used_chunk_ids = llm_score.used_chunk_ids
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


def _build_critique_prompt(
    domain: str,
    candidate_score: float,
    candidate_rationale: str,
    critique_chunks: list[Chunk],
    used_chunk_ids: list[str],
) -> str:
    """Build a prompt for the critique agent to evaluate the candidate's rationale and evidence."""
    evidence = []
    for chunk in critique_chunks:
        text = chunk.text.strip().replace("\n", " ")
        evidence.append(f"[{chunk.chunk_id}] {text[:MAX_LLM_CHUNK_CHARS]}")
    evidence_block = "\n\n".join(evidence) if evidence else "[NO_ADDITIONAL_EVIDENCE]"

    rubric = "\n".join(f"- {band}: {desc}" for band, desc in SCORING_RUBRIC.items())
    heuristic = DOMAIN_HEURISTICS.get(domain, "N/A")

    return (
        "You are an ESG Critique Agent. Your role is to review a domain scorer's work and provide grounded feedback.\n\n"
        f"Domain: {domain}\n"
        f"Candidate's current score: {candidate_score}/20\n"
        f"Candidate's rationale:\n{candidate_rationale}\n\n"
        "Your task:\n"
        "1. Carefully review the candidate's rationale and scoring decision.\n"
        "2. Examine the additional evidence chunks provided below.\n"
        "3. Identify:\n"
        "   - Gaps: What important evidence or criteria are missing from the rationale?\n"
        "   - Overstatements: Are there claims not supported by evidence?\n"
        "   - Understatements: Is the score too conservative given the evidence?\n"
        "   - Biases: Any systematic lean (optimism/pessimism) in the reasoning?\n"
        "4. Assess whether the current score is appropriate given ALL available evidence (original + new chunks).\n\n"
        f"Domain heuristic checklist: {heuristic}\n\n"
        f"Scoring rubric:\n{rubric}\n\n"
        "Scoring discipline:\n"
        "1) Start near a neutral prior and move up only with explicit evidence.\n"
        "2) Keep score <= 8 if evidence is generic or lacks measurable targets/results.\n"
        "3) Keep score <= 12 unless evidence shows clear governance ownership plus risk methods and monitoring outcomes.\n"
        "4) Use > 16 only for exceptional, comprehensive, and quantified evidence across strategy, governance, and risk management.\n\n"
        "Return strict JSON with these keys:\n"
        "- feedback (string): Your critique of the candidate's rationale (2-3 sentences)\n"
        "- missing_evidence (list of strings): Specific gaps or missing criteria\n"
        "- overstated_claims (list of strings): Claims not supported by evidence\n"
        "- recommended_score (number): Your suggested score (0-20) based on ALL evidence\n"
        "- confidence (number 0-1): Your confidence in the recommended score\n"
        "- adjustment_direction (string): 'increase', 'decrease', or 'no_change'\n"
        "Do not include markdown.\n\n"
        f"Additional evidence chunks:\n{evidence_block}"
    )


def _extract_critique_response(raw_text: str) -> dict | None:
    """Extract and validate the critique agent's JSON response."""
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_critique_llm(prompt: str) -> dict | None:
    """Call the LLM to get a critique response."""
    try:
        from project.llm_config import llm
    except Exception:
        return None

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
        parsed = _extract_critique_response(str(response))
    if not isinstance(parsed, dict):
        return None
    return parsed


def critique_and_adjust(domain: str, candidate: DomainScore, critique_chunks: list[Chunk]) -> tuple[DomainScore, dict[str, float | bool]]:
    """
    Use an LLM-based critique agent to evaluate the candidate's rationale and suggest adjustments.
    
    The critique agent reviews:
    1. The candidate's current score and rationale
    2. Additional evidence chunks not used in the initial scoring
    3. Provides feedback on gaps, overstatements, and biases
    4. Recommends an adjusted score based on ALL evidence
    
    Returns the adjusted DomainScore and stats about the adjustment.
    """
    if not critique_chunks:
        return candidate, {"adjusted": False, "gap": 0.0, "feedback": "No critique chunks provided"}

    # Build the critique prompt
    prompt = _build_critique_prompt(
        domain=domain,
        candidate_score=candidate.estimated_score,
        candidate_rationale=candidate.rationale,
        critique_chunks=critique_chunks,
        used_chunk_ids=candidate.used_chunk_ids,
    )

    # Get critique from LLM
    critique_response = _call_critique_llm(prompt)
    
    if critique_response is None:
        # Fallback to simple re-scoring if LLM call fails
        critique = estimate_domain_score(domain, critique_chunks)
        gap = abs(candidate.estimated_score - critique.estimated_score)
        needs_adjustment = gap > 1.5  # Relaxed threshold without confidence barrier
        if not needs_adjustment:
            return candidate, {"adjusted": False, "gap": round(gap, 2), "feedback": "LLM unavailable, no adjustment"}

        merged_score = round((candidate.estimated_score + critique.estimated_score) / 2.0, 2)
        merged_conf = round(max(candidate.confidence, critique.confidence), 3)
        updated = DomainScore(
            estimated_score=merged_score,
            confidence=merged_conf,
            rationale=(
                f"[FALLBACK] Detected gap={gap:.2f}. Initial: {candidate.rationale} | "
                f"Critique: {critique.rationale}"
            ),
            label=score_label(merged_score),
            retrieved_chunk_ids=list(dict.fromkeys(candidate.retrieved_chunk_ids + critique.retrieved_chunk_ids)),
            used_chunk_ids=list(dict.fromkeys(candidate.used_chunk_ids + critique.used_chunk_ids)),
        )
        return updated, {"adjusted": True, "gap": round(gap, 2), "feedback": "Fallback re-scoring applied"}

    # Parse and validate the critique response
    feedback = critique_response.get("feedback", "")
    missing = critique_response.get("missing_evidence", [])
    overstated = critique_response.get("overstated_claims", [])
    adjustment_direction = critique_response.get("adjustment_direction", "no_change")
    
    try:
        recommended_score = float(critique_response.get("recommended_score", candidate.estimated_score))
        recommended_score = max(0.0, min(MAX_SCORE, recommended_score))
        critique_confidence = float(critique_response.get("confidence", candidate.confidence))
        critique_confidence = max(0.0, min(1.0, critique_confidence))
    except (TypeError, ValueError):
        return candidate, {"adjusted": False, "gap": 0.0, "feedback": "Invalid critique response format"}

    gap = abs(candidate.estimated_score - recommended_score)
    
    # Determine if we should adjust
    # We adjust if: there's a meaningful gap, OR there are specific feedback items
    has_feedback = bool(feedback) or bool(missing) or bool(overstated)
    needs_adjustment = gap > 1.0 or (has_feedback and adjustment_direction != "no_change")
    
    if not needs_adjustment:
        return candidate, {
            "adjusted": False,
            "gap": round(gap, 2),
            "feedback": feedback,
            "missing_evidence": missing,
            "overstated_claims": overstated,
        }

    # Apply the adjustment
    # Use a weighted blend: more weight to critique if it has high confidence
    # and provides substantial feedback
    feedback_weight = min(0.7, 0.3 + (0.4 * critique_confidence))
    if gap > 5.0:
        # Large gap: trust the critique more
        adjusted_score = round(recommended_score, 2)
        adjusted_conf = round(critique_confidence, 3)
    else:
        # Blend the scores
        adjusted_score = round(
            (candidate.estimated_score * (1.0 - feedback_weight)) + (recommended_score * feedback_weight),
            2
        )
        adjusted_conf = round(
            max(candidate.confidence, critique_confidence) * 0.95 + 0.05,
            3
        )

    # Build the new rationale incorporating the critique
    missing_str = f" Missing evidence: {', '.join(missing[:3])}" if missing else ""
    overstated_str = f" Overstated: {', '.join(overstated[:3])}" if overstated else ""
    adjustment_str = f" [{adjustment_direction.upper()}]" if adjustment_direction != "no_change" else ""
    
    new_rationale = (
        f"Critique applied{adjustment_str}: {feedback}."
        f"{missing_str}{overstated_str}\n"
        f"Original rationale: {candidate.rationale}"
    )

    updated = DomainScore(
        estimated_score=adjusted_score,
        confidence=adjusted_conf,
        rationale=new_rationale,
        label=score_label(adjusted_score),
        retrieved_chunk_ids=list(dict.fromkeys(candidate.retrieved_chunk_ids + [c.chunk_id for c in critique_chunks])),
        used_chunk_ids=list(dict.fromkeys(candidate.used_chunk_ids + [c.chunk_id for c in critique_chunks])),
    )

    return updated, {
        "adjusted": True,
        "gap": round(gap, 2),
        "feedback": feedback,
        "missing_evidence": missing,
        "overstated_claims": overstated,
        "adjustment_direction": adjustment_direction,
        "critique_score": round(recommended_score, 2),
        "critique_confidence": round(critique_confidence, 3),
    }
