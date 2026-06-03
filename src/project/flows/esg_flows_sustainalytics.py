from __future__ import annotations

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

from project.esg_framework_sustainalytics.models_sustainalytics import ReportRecord
from project.esg_framework_sustainalytics.patterns_sustainalytics import (
    run_handoff_pattern,
    run_parallel_pattern,
    run_review_critique_pattern,
)
from project.esg_framework_sustainalytics.retrieval_sustainalytics import ChunkStore


class ESGFlowStateSustainalytics(BaseModel):
    report_id: str = ""
    pattern: str = ""
    done: bool = False


class BaseESGPatternFlowSustainalytics(Flow[ESGFlowStateSustainalytics]):
    def __init__(self, report: ReportRecord):
        super().__init__()
        self.report = report
        self.chunk_store = ChunkStore()
        self.result = None


class ParallelConcurrentESGFlowSustainalytics(BaseESGPatternFlowSustainalytics):
    @start()
    def run_pattern(self):
        self.state.report_id = self.report.report_id
        self.state.pattern = "parallel_concurrent"
        self.result = run_parallel_pattern(self.report, self.chunk_store)
        return self.result

    @listen(run_pattern)
    def finalize(self, _):
        self.state.done = True
        return self.result


class HandoffHierarchicalESGFlowSustainalytics(BaseESGPatternFlowSustainalytics):
    @start()
    def run_pattern(self):
        self.state.report_id = self.report.report_id
        self.state.pattern = "handoff_hierarchical"
        self.result = run_handoff_pattern(self.report, self.chunk_store)
        return self.result

    @listen(run_pattern)
    def finalize(self, _):
        self.state.done = True
        return self.result


class ReviewCritiqueESGFlowSustainalytics(BaseESGPatternFlowSustainalytics):
    @start()
    def run_pattern(self):
        self.state.report_id = self.report.report_id
        self.state.pattern = "review_critique"
        self.result = run_review_critique_pattern(self.report, self.chunk_store)
        return self.result

    @listen(run_pattern)
    def finalize(self, _):
        self.state.done = True
        return self.result
