from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from project.llm_config import llm, llm_thinking
from project.tools.esg_tools import ChunkLookupTool, WeightedAverageTool


@CrewBase
class ESGEvaluationCrew:
    agents: list[BaseAgent]
    tasks: list[Task]

    def _agent(self, key: str, *, thinking: bool = False, tools: list | None = None) -> Agent:
        kwargs = {
            "config": self.agents_config[key],  # type: ignore[index]
            "verbose": True,
            "tools": tools or [],
        }
        selected_llm = llm_thinking if thinking else llm
        if selected_llm is not None:
            kwargs["llm"] = selected_llm
        return Agent(**kwargs)

    @agent
    def report_parser(self) -> Agent:
        return self._agent("report_parser", tools=[ChunkLookupTool()])

    @agent
    def environmental_analyst(self) -> Agent:
        return self._agent("environmental_analyst", thinking=True, tools=[ChunkLookupTool()])

    @agent
    def social_analyst(self) -> Agent:
        return self._agent("social_analyst", thinking=True, tools=[ChunkLookupTool()])

    @agent
    def governance_analyst(self) -> Agent:
        return self._agent("governance_analyst", thinking=True, tools=[ChunkLookupTool()])

    @agent
    def score_aggregator(self) -> Agent:
        return self._agent("score_aggregator", tools=[WeightedAverageTool()])

    @agent
    def performance_comparator(self) -> Agent:
        return self._agent("performance_comparator")

    @agent
    def metrics_evaluator(self) -> Agent:
        return self._agent("metrics_evaluator")

    @agent
    def critique_agent(self) -> Agent:
        return self._agent("critique_agent", thinking=True, tools=[ChunkLookupTool()])

    @task
    def parse_report_task(self) -> Task:
        return Task(config=self.tasks_config["parse_report_task"])  # type: ignore[index]

    @task
    def environmental_scoring_task(self) -> Task:
        return Task(config=self.tasks_config["environmental_scoring_task"])  # type: ignore[index]

    @task
    def social_scoring_task(self) -> Task:
        return Task(config=self.tasks_config["social_scoring_task"])  # type: ignore[index]

    @task
    def governance_scoring_task(self) -> Task:
        return Task(config=self.tasks_config["governance_scoring_task"])  # type: ignore[index]

    @task
    def aggregate_scores_task(self) -> Task:
        return Task(config=self.tasks_config["aggregate_scores_task"])  # type: ignore[index]

    @task
    def compare_scores_task(self) -> Task:
        return Task(config=self.tasks_config["compare_scores_task"])  # type: ignore[index]

    @task
    def evaluate_metrics_task(self) -> Task:
        return Task(config=self.tasks_config["evaluate_metrics_task"])  # type: ignore[index]

    @task
    def critique_task(self) -> Task:
        return Task(config=self.tasks_config["critique_task"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=True,
        )
