from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from project.esg.models import Domain


@dataclass(frozen=True, slots=True)
class HeuristicGroup:
    name: str
    keywords: tuple[str, ...]


ENVIRONMENTAL_HEURISTICS: tuple[HeuristicGroup, ...] = (
    HeuristicGroup("strategy_financial_planning", ("environmental strategy", "financial planning", "decarbonisation", "climate strategy", "transition plan")),
    HeuristicGroup("targets_limits", ("target", "objective", "limit", "net zero", "reduction")),
    HeuristicGroup("eu_taxonomy", ("eu taxonomy", "taxonomy aligned", "green investment")),
    HeuristicGroup("counterparty_engagement", ("supplier engagement", "counterparty", "engagement", "environmental risk mitigation")),
    HeuristicGroup("governance_responsibility", ("management body", "board", "committee", "reporting line", "remuneration")),
    HeuristicGroup("risk_methodology", ("risk framework", "identify", "measure", "monitor", "mitigation", "stress test")),
    HeuristicGroup("data_quality_limits", ("data quality", "risk limits", "escalation", "trigger")),
    HeuristicGroup("risk_transmission", ("credit risk", "liquidity", "market risk", "operational risk", "reputational risk")),
)

SOCIAL_HEURISTICS: tuple[HeuristicGroup, ...] = (
    HeuristicGroup("strategy_financial_planning", ("social strategy", "financial planning", "community", "employee wellbeing", "human rights")),
    HeuristicGroup("targets_limits", ("target", "objective", "limit", "safety target", "training target")),
    HeuristicGroup("counterparty_engagement", ("supplier code", "counterparty", "responsible sourcing", "harmful social activities")),
    HeuristicGroup("governance_responsibility", ("management body", "board", "community", "employee", "customer", "human rights")),
    HeuristicGroup("governance_integration", ("organizational structure", "reporting line", "reporting frequency", "remuneration")),
    HeuristicGroup("risk_methodology", ("social risk", "identify", "measure", "monitor", "mitigation", "grievance")),
    HeuristicGroup("limits_escalation", ("risk limit", "escalation", "trigger", "tolerance")),
    HeuristicGroup("risk_transmission", ("credit risk", "liquidity", "market risk", "operational risk", "reputational risk")),
)

GOVERNANCE_HEURISTICS: tuple[HeuristicGroup, ...] = (
    HeuristicGroup("governance_arrangements", ("governance arrangement", "board oversight", "counterparty governance")),
    HeuristicGroup("non_financial_reporting", ("non-financial reporting", "sustainability committee", "audit committee")),
    HeuristicGroup("ethics_transparency", ("ethical", "strategy", "inclusive", "transparency", "speak up")),
    HeuristicGroup("conflict_management", ("conflict of interest", "whistleblowing", "internal communication", "compliance")),
    HeuristicGroup("risk_management", ("risk management", "governance performance", "internal control", "oversight")),
)

HEURISTICS_BY_DOMAIN: dict[Domain, tuple[HeuristicGroup, ...]] = {
    "environmental": ENVIRONMENTAL_HEURISTICS,
    "social": SOCIAL_HEURISTICS,
    "governance": GOVERNANCE_HEURISTICS,
}


def domain_keywords(domain: Domain) -> set[str]:
    keywords: set[str] = set()
    for group in HEURISTICS_BY_DOMAIN[domain]:
        for keyword in group.keywords:
            keywords.update(keyword.lower().split())
        keywords.update(group.name.lower().split("_"))
    keywords.add(domain)
    return keywords


def all_domain_keywords(domains: Iterable[Domain]) -> dict[Domain, set[str]]:
    return {domain: domain_keywords(domain) for domain in domains}
