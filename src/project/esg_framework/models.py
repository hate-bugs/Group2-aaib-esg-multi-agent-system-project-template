from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportRecord:
    report_id: str
    filename: str
    ticker: str
    year: str
    preprocessed_content: str
    ground_truth: dict[str, float]
    sector: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    report_id: str
    text: str
    token_count: int
    tags: list[str]
    weight: float = 1.0


@dataclass
class RetrievalEvent:
    agent_name: str
    domain: str
    query: str
    retrieved_chunk_ids: list[str]
    used_chunk_ids: list[str]


@dataclass
class DomainScore:
    estimated_score: float
    confidence: float
    rationale: str
    label: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    used_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ReportRunResult:
    report_id: str
    pattern: str
    domain_scores: dict[str, DomainScore]
    total_score: float
    confidence: float
    retrieval_trace: list[RetrievalEvent]
    comparison: dict[str, Any]
    metrics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
