from __future__ import annotations

import json
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ChunkLookupInput(BaseModel):
    report_id: str = Field(..., description="Report identifier matching the persisted chunk index file")
    domain: str = Field(..., description="ESG domain to inspect: environmental, social, or governance")


class ChunkLookupTool(BaseTool):
    name: str = "chunk_lookup_tool"
    description: str = "Loads persisted report chunks from the local JSON index so agents can inspect RAG-ready evidence."
    args_schema: Type[BaseModel] = ChunkLookupInput

    def _run(self, report_id: str, domain: str) -> str:
        root = Path(__file__).resolve().parents[3]
        chunk_file = root / "outputs" / "indexes" / f"{report_id}.json"
        if not chunk_file.exists():
            return f"No chunk index found for report {report_id}. Run the parser flow first."
        payload = json.loads(chunk_file.read_text(encoding="utf-8"))
        filtered = [item for item in payload if domain in item.get("labels", [])]
        return json.dumps(filtered[:5], indent=2)


class WeightedAverageInput(BaseModel):
    environmental_score: float
    social_score: float
    governance_score: float
    environmental_weight: float = 1 / 3
    social_weight: float = 1 / 3
    governance_weight: float = 1 / 3


class WeightedAverageTool(BaseTool):
    name: str = "weighted_average_tool"
    description: str = "Computes a configurable weighted ESG total score from E, S, and G inputs."
    args_schema: Type[BaseModel] = WeightedAverageInput

    def _run(
        self,
        environmental_score: float,
        social_score: float,
        governance_score: float,
        environmental_weight: float = 1 / 3,
        social_weight: float = 1 / 3,
        governance_weight: float = 1 / 3,
    ) -> str:
        total = (
            environmental_score * environmental_weight
            + social_score * social_weight
            + governance_score * governance_weight
        )
        return f"{total:.2f}"
