from __future__ import annotations

# Morningstar Sustainalytics ESG Risk Ratings Methodology
# Five distinct thematic areas for Beta Indicators (used to refine exposure assessments)

DOMAIN_HEURISTICS = {
    "product_production": (
        "Evaluate risks related to a company's products, services, and production processes. "
        "Focus on product environmental impact, carbon emissions, resource efficiency, waste management, "
        "product safety, and production-related ESG risks."
    ),
    "financials": (
        "Assess financial risks tied to ESG factors. "
        "Evaluate solvency, financial flexibility, asset performance, ESG-related financial disclosures, "
        "capital allocation, and financial resilience to ESG shocks."
    ),
    "events": (
        "Capture risks from controversial or high-impact events. "
        "Identify environmental incidents, corruption scandals, human rights violations, "
        "major controversies, and event-driven ESG risks that can significantly impact risk exposure."
    ),
    "geographic": (
        "Evaluate risks based on the company's geographic exposure. "
        "Assess water risk in regions with scarcity, regional corruption levels, political stability, "
        "regional regulatory environments, and location-specific ESG risks."
    ),
    "governance": (
        "Evaluate Corporate Governance and Stakeholder Governance baseline issues. "
        "Assess board independence, stakeholder engagement, transparency, "
        "executive compensation alignment, shareholder rights, and governance structure."
    ),
}

SCORING_RUBRIC = {
    "0-20": "Very weak or absent disclosure. Mentions are generic, non-committal, or marketing-focused.",
    "21-40": "Basic policy intent. Some governance/process description but limited measurable targets or monitoring outcomes.",
    "41-60": "Moderate maturity. Clear ownership and risk/process integration with at least some quantified targets or implementation evidence.",
    "61-80": "Strong maturity. Structured strategy, governance, and risk-management disclosures with concrete targets, limits, and progress indicators.",
    "81-100": "Leading practice. Comprehensive and consistent evidence across all heuristic areas, including quantified outcomes and clear accountability.",
}

# Keywords for each Sustainalytics MEI thematic domain
DOMAIN_KEYWORDS = {
    "product_production": {
        "product", "production", "manufacturing", "emission", "carbon", "environmental impact",
        "resource", "efficiency", "waste", "pollution", "biodiversity", "water", "energy",
        "renewable", "decarbon", "net zero", "scope 1", "scope 2", "scope 3", "ghg",
        "recycling", "tailings", "rehabilitation", "supply chain", "sourcing",
    },
    "financials": {
        "financial", "solvency", "flexibility", "capital", "asset", "performance",
        "liquidity", "debt", "equity", "investment", "financing", "cost of capital",
        "profitability", "revenue", "expense", "budget", "allocation", "resilience",
        "risk management", "insurance", "hedging", "exposure", "financial stability",
    },
    "events": {
        "event", "incident", "controversy", "scandal", "violation", "breach",
        "accident", "spill", "leak", "fire", "explosion", "corruption",
        "fraud", "litigation", "lawsuit", "penalty", "fine", "sanction",
        "human rights", "labor dispute", "strike", "protest", "boycott",
        "environmental incident", "safety incident", "compliance violation",
    },
    "geographic": {
        "geographic", "geography", "region", "location", "country", "territory",
        "water risk", "scarcity", "abundance", "climate zone", "regulatory environment",
        "political stability", "corruption", "bribery", "tax regime", "trade restriction",
        "local community", "indigenous rights", "land use", "resource nationalism",
        "cross-border", "jurisdiction", "geopolitical", "regional risk",
    },
    "governance": {
        "governance", "board", "director", "audit", "ethics", "ethical", "compliance",
        "transparency", "disclosure", "remuneration", "compensation", "incentive",
        "oversight", "committee", "independence", "conflict of interest", "policy",
        "internal control", "whistleblower", "anti-corruption", "anti-bribery",
        "stakeholder engagement", "shareholder rights", "voting", "accountability",
        "assurance", "reporting", "code of conduct", "corporate governance",
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

# Five Sustainalytics MEI thematic areas
ALL_DOMAINS = ("product_production", "financials", "events", "geographic", "governance")


DOMAIN_CHUNK_WEIGHTS = {
    "product_production": 1.25,
    "financials": 1.2,
    "events": 1.3,
    "geographic": 1.15,
    "governance": 1.1,
    "general": 1.0,
}

# Sustainalytics-specific priors based on the 5 thematic areas (0-100 scale)
# These are neutral starting points for the Management Indicator
DOMAIN_SCORE_PRIORS = {
    "product_production": 50.0,
    "financials": 50.0,
    "events": 40.0,
    "geographic": 45.0,
    "governance": 60.0,
}
