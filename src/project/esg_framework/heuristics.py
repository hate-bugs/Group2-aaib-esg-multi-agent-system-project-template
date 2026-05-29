from __future__ import annotations

DOMAIN_HEURISTICS = {
    "environmental": (
        "Evaluate integration of environmental factors in strategy, objectives and limits, EU Taxonomy alignment, "
        "engagement policy, management responsibility, governance and remuneration linkage, risk framework integration, "
        "methods, measurement and monitoring, mitigation tools, data quality, escalation limits, and risk transmission channels."
    ),
    "social": (
        "Evaluate integration of social factors in strategy, objectives and limits, engagement on harmful social activity, "
        "management responsibility across community, employees, customers, and human rights, governance and remuneration linkage, "
        "risk framework integration, methods, monitoring, mitigation and tools, escalation limits, and risk transmission channels."
    ),
    "governance": (
        "Evaluate counterparty governance integration in governance arrangements and risk management, highest governance body role "
        "in non-financial reporting, ethical conduct, strategy, inclusiveness, transparency, conflict-of-interest handling, and internal communication."
    ),
}

DOMAIN_KEYWORDS = {
    "environmental": {
        "environment", "emission", "carbon", "climate", "taxonomy", "energy", "waste", "pollution", "biodiversity", "water", "renewable"
    },
    "social": {
        "social", "employee", "community", "human rights", "safety", "labor", "training", "customer", "diversity", "inclusion", "wellbeing"
    },
    "governance": {
        "governance", "board", "audit", "ethic", "compliance", "transparency", "remuneration", "oversight", "risk committee", "conflict"
    },
}

ALL_DOMAINS = ("environmental", "social", "governance")
