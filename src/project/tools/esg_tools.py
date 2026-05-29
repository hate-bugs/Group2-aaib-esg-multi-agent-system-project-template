from __future__ import annotations

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Simple numeric expression using +,-,*,/")


class CalculatorTool(BaseTool):
    name: str = "score_calculator"
    description: str = "Safely evaluate simple arithmetic expressions for score aggregation."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        allowed = set("0123456789.+-*/() ")
        if not set(expression) <= allowed:
            return "Invalid expression"
        try:
            return str(round(float(eval(expression, {"__builtins__": {}}, {})), 4))
        except Exception:
            return "Invalid expression"


class RetrieverInput(BaseModel):
    query: str = Field(..., description="Domain query for retrieval tracing")


class RetrievalTraceTool(BaseTool):
    name: str = "retrieval_trace"
    description: str = "Records retrieval queries for chunk coverage analysis."
    args_schema: Type[BaseModel] = RetrieverInput

    def _run(self, query: str) -> str:
        return f"retrieval_query:{query}"
