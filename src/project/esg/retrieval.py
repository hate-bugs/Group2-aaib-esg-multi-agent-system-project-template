from __future__ import annotations

from dataclasses import dataclass

from project.esg.chunking import tokenize
from project.esg.heuristics import HEURISTICS_BY_DOMAIN, domain_keywords
from project.esg.models import Domain, ReportChunk


@dataclass(slots=True)
class RetrievedChunk:
    chunk: ReportChunk
    score: float


class ChunkRetriever:
    def __init__(self, chunks: list[ReportChunk]):
        self.chunks = chunks

    def retrieve(self, domain: Domain, top_k: int, *, trial: int = 1) -> list[RetrievedChunk]:
        domain_terms = domain_keywords(domain)
        heuristic_terms = {
            token
            for group in HEURISTICS_BY_DOMAIN[domain]
            for keyword in group.keywords
            for token in keyword.lower().split()
        }
        scored: list[RetrievedChunk] = []
        for chunk in self.chunks:
            tokens = set(tokenize(chunk.text))
            lexical = len(tokens & domain_terms)
            heuristic = len(tokens & heuristic_terms)
            label_bonus = chunk.label_scores.get(domain, 0.0) * 10
            total = lexical + heuristic * 1.5 + label_bonus
            if domain in chunk.labels:
                total += 1.0
            if total > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=total))
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        if len(scored) <= top_k:
            return scored
        rotation = (trial - 1) % max(1, min(top_k, len(scored)))
        return scored[rotation : rotation + top_k]
