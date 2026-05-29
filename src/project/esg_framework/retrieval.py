from __future__ import annotations

import json
from pathlib import Path

from project.esg_framework.heuristics import DOMAIN_KEYWORDS
from project.esg_framework.models import Chunk


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

    scored: list[tuple[int, float, Chunk]] = []
    for chunk in chunks:
        text = chunk.text.lower()
        keyword_hits = sum(text.count(token) for token in keywords)
        domain_bonus = 2 if domain in chunk.tags else 0
        scored.append((keyword_hits + domain_bonus, chunk.weight, chunk))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in scored if item[0] > 0][:max_chunks]
    if not selected:
        selected = chunks[:max_chunks]
    return selected
