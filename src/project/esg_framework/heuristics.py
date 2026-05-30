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

SCORING_RUBRIC = {
    "0-4": "Very weak or absent disclosure. Mentions are generic, non-committal, or marketing-focused.",
    "5-8": "Basic policy intent. Some governance/process description but limited measurable targets or monitoring outcomes.",
    "9-12": "Moderate maturity. Clear ownership and risk/process integration with at least some quantified targets or implementation evidence.",
    "13-16": "Strong maturity. Structured strategy, governance, and risk-management disclosures with concrete targets, limits, and progress indicators.",
    "17-20": "Leading practice. Comprehensive and consistent evidence across all heuristic areas, including quantified outcomes and clear accountability.",
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

DOMAIN_SIGNAL_KEYWORDS = {
    "structured_policy": {
        "policy",
        "framework",
        "oversight",
        "governance",
        "committee",
        "responsibility",
    },
    "quantified_targets": {
        "target",
        "kpi",
        "threshold",
        "limit",
        "baseline",
        "2025",
        "2030",
        "%",
    },
    "monitoring_results": {
        "progress",
        "result",
        "achieved",
        "reduced",
        "improved",
        "measured",
        "trend",
    },
    "risk_methods": {
        "risk management",
        "methodology",
        "scenario",
        "stress test",
        "escalation",
        "data quality",
        "controls",
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

