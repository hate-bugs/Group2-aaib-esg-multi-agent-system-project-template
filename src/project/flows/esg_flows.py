from __future__ import annotations

from pydantic import BaseModel, Field
from crewai.flow.flow import Flow, listen, start

from project.esg.config import ESGExperimentConfig
from project.esg.models import PatternRunResult
from project.esg.runner import PATTERN_RUNNERS


class ESGFlowState(BaseModel):
    report: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    trial: int = 1
    chunks_ready: bool = False
    domain_scores: dict = Field(default_factory=dict)
    final_result: dict = Field(default_factory=dict)


class BaseESGFlow(Flow[ESGFlowState]):
    pattern_name: str = "parallel"

    @start()
    def parse_report(self):
        self.state.chunks_ready = True
        return self.state.report

    @listen(parse_report)
    def score_report(self, _previous_result):
        config = ESGExperimentConfig(**self.state.config)
        runner = PATTERN_RUNNERS[self.pattern_name]
        result: PatternRunResult = runner(self.state.report["record"], config, self.state.trial)
        self.state.domain_scores = {key: value.to_dict() for key, value in result.domain_scores.items()}
        self.state.final_result = result.to_dict()
        return self.state.final_result

    @listen(score_report)
    def evaluate_report(self, result):
        return result


class ParallelESGFlow(BaseESGFlow):
    pattern_name = "parallel"


class HierarchicalESGFlow(BaseESGFlow):
    pattern_name = "hierarchical"


class ReviewCritiqueESGFlow(BaseESGFlow):
    pattern_name = "review"
