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
        "environment", "emission", "carbon", "climate", "taxonomy", "energy", "waste", "pollution", "biodiversity", "water", "renewable",
        "decarbon", "net zero", "scope 1", "scope 2", "scope 3", "ghg", "recycling", "efficiency", "tailings", "rehabilitation"
    },
    "social": {
        "social", "employee", "community", "human rights", "safety", "labor", "training", "customer", "diversity", "inclusion", "wellbeing",
        "health", "injury", "incident", "engagement", "workforce", "indigenous", "culture", "retention", "well-being", "mental"
    },
    "governance": {
        "governance", "board", "audit", "ethic", "compliance", "transparency", "remuneration", "oversight", "risk committee", "conflict",
        "policy", "internal control", "whistleblower", "anti-corruption", "independence", "committee", "accountability", "assurance"
    },
}

ALL_DOMAINS = ("environmental", "social", "governance")


DOMAIN_CHUNK_WEIGHTS = {
    "environmental": 1.2,
    "social": 1.1,
    "governance": 1.15,
    "general": 1.0,
}

# Healthcare priors estimated from the full healthcare subset in the provided benchmark dataset.
# They reduce optimistic drift when evidence is weak, while still allowing high scores with strong support.
DOMAIN_SCORE_PRIORS = {
    "environmental": 5.74,
    "social": 10.44,
    "governance": 7.45,
}

