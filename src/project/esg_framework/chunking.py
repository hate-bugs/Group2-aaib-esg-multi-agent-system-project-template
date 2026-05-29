from __future__ import annotations

from project.esg_framework.heuristics import DOMAIN_KEYWORDS
from project.esg_framework.models import Chunk, ReportRecord


def _tokenize(text: str) -> list[str]:
    return [token for token in text.replace("\n", " ").split(" ") if token]


def _score_domain(text: str, domain: str) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in DOMAIN_KEYWORDS[domain])


def split_report_to_chunks(
    report: ReportRecord,
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    tokens = _tokenize(report.preprocessed_content)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    for idx, start in enumerate(range(0, len(tokens), step)):
        end = min(start + chunk_size, len(tokens))
        text = " ".join(tokens[start:end]).strip()
        if not text:
            continue

        domain_scores = {domain: _score_domain(text, domain) for domain in DOMAIN_KEYWORDS}
        max_score = max(domain_scores.values()) if domain_scores else 0
        tags = [domain for domain, score in domain_scores.items() if score == max_score and score > 0] or ["general"]
        weight = 1.0
        if "environmental" in tags:
            weight = 1.2
        elif "social" in tags:
            weight = 1.1
        elif "governance" in tags:
            weight = 1.15

        chunks.append(
            Chunk(
                chunk_id=f"{report.report_id}-{idx}",
                report_id=report.report_id,
                text=text,
                token_count=len(text.split()),
                tags=tags,
                weight=weight,
            )
        )

        if end >= len(tokens):
            break
    return chunks
