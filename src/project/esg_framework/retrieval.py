from __future__ import annotations

import json
from pathlib import Path

from project.esg_framework.heuristics import DOMAIN_KEYWORDS
from project.esg_framework.models import Chunk

BOILERPLATE_TERMS = {
    "style guide",
    "pantone",
    "logo",
    "copyright",
    "table of content",
    "photo",
    "image",
    "contact us",
    "this page intentionally",
}


class ChunkStore:
    def __init__(self) -> None:
        self._chunks_by_report: dict[str, list[Chunk]] = {}

    def put(self, report_id: str, chunks: list[Chunk]) -> None:
        self._chunks_by_report[report_id] = chunks

    def get(self, report_id: str) -> list[Chunk]:
        return self._chunks_by_report.get(report_id, [])

    def persist_json(self, report_id: str, path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        chunks = self.get(report_id)
        payload = [
            {
                "chunk_id": c.chunk_id,
                "report_id": c.report_id,
                "text": c.text,
                "token_count": c.token_count,
                "tags": c.tags,
                "weight": c.weight,
            }
            for c in chunks
        ]
        path_obj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def retrieve_for_domain(
    chunks: list[Chunk],
    domain: str,
    max_chunks: int = 8,
) -> list[Chunk]:
    keywords = DOMAIN_KEYWORDS.get(domain, set())

    scored: list[tuple[float, float, Chunk]] = []
    for chunk in chunks:
        text = chunk.text.lower()
        # Score both total hits and keyword diversity to avoid over-valuing repeated single-term mentions.
        keyword_hits = sum(min(text.count(token), 3) for token in keywords)
        diversity_hits = sum(1 for token in keywords if token in text)
        token_count = max(chunk.token_count, 1)
        density = (keyword_hits / token_count) * 100
        domain_bonus = 2.0 if domain in chunk.tags else 0.0
        boilerplate_penalty = sum(1 for token in BOILERPLATE_TERMS if token in text)
        score = (0.6 * keyword_hits) + (1.1 * diversity_hits) + (0.8 * min(density, 5.0)) + domain_bonus - (1.5 * boilerplate_penalty)
        scored.append((score, chunk.weight, chunk))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in scored if item[0] > 1.2][:max_chunks]
    if not selected:
        # Fallback keeps deterministic order while still preferring larger, more informative chunks.
        selected = sorted(chunks, key=lambda c: c.token_count, reverse=True)[:max_chunks]
    return selected
