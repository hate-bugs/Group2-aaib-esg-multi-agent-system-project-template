from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Domain = Literal["environmental", "social", "governance"]
PATTERNS = ("parallel", "hierarchical", "review")
DOMAINS: tuple[Domain, ...] = ("environmental", "social", "governance")


@dataclass(slots=True)
class ReportRecord:
    report_id: str
    company_name: str
    preprocessed_content: str
    actual_scores: dict[str, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReportChunk:
    chunk_id: str
    report_id: str
    text: str
    token_count: int
    labels: list[Domain]
    label_scores: dict[str, float]


@dataclass(slots=True)
class DomainScore:
    domain: Domain
    estimated_score: float
    confidence: float
    rationale: str
    chunk_ids: list[str]
    supported_claims: int
    unsupported_claims: int
    partially_supported_claims: int
    matched_heuristics: list[str]
    calls_used: int = 1
    tokens_used: int = 0
    iteration_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DeliberationRecord:
    domain: Domain
    iteration: int
    issue: str
    suggestion: str
    approved: bool
    initial_score: float
    final_score: float


@dataclass(slots=True)
class PatternTrace:
    retrieved_chunk_ids: list[str]
    retrieved_token_count: int
    total_token_count: int
    agent_calls: int
    critical_path_latency: float
    wall_clock_latency: float
    deliberations: list[DeliberationRecord] = field(default_factory=list)
    trial: int = 1


@dataclass(slots=True)
class PatternRunResult:
    pattern: str
    report_id: str
    company_name: str
    domain_scores: dict[str, DomainScore]
    total_score: float
    comparator: dict[str, Any]
    metrics: dict[str, Any]
    trace: PatternTrace
    actual_scores: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "report_id": self.report_id,
            "company_name": self.company_name,
            "domain_scores": {key: value.to_dict() for key, value in self.domain_scores.items()},
            "total_score": self.total_score,
            "comparator": self.comparator,
            "metrics": self.metrics,
            "trace": {
                **asdict(self.trace),
                "deliberations": [asdict(item) for item in self.trace.deliberations],
            },
            "actual_scores": self.actual_scores,
        }
