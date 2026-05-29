from __future__ import annotations

from typing import Type
import ast
import operator

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Simple numeric expression using +,-,*,/")


class CalculatorTool(BaseTool):
    name: str = "score_calculator"
    description: str = "Safely evaluate simple arithmetic expressions for score aggregation."
    args_schema: Type[BaseModel] = CalculatorInput

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in self._OPS:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._OPS[type(node.op)](left, right)
        raise ValueError("Unsupported expression")

    def _run(self, expression: str) -> str:
        try:
            parsed = ast.parse(expression, mode="eval")
            value = self._eval_node(parsed.body)
            return str(round(float(value), 4))
        except SyntaxError:
            return "Invalid expression syntax"
        except ZeroDivisionError:
            return "Division by zero"
        except ValueError:
            return "Unsupported expression"


class RetrieverInput(BaseModel):
    query: str = Field(..., description="Domain query for retrieval tracing")


class RetrievalTraceTool(BaseTool):
    name: str = "retrieval_trace"
    description: str = "Records retrieval queries for chunk coverage analysis."
    args_schema: Type[BaseModel] = RetrieverInput

    def _run(self, query: str) -> str:
        return f"retrieval_query:{query}"
