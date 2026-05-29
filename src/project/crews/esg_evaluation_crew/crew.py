from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from project.llm_config import llm
from project.tools import CalculatorTool, RetrievalTraceTool


@CrewBase
class ESGEvaluationCrew:
    agents: list[BaseAgent]
    tasks: list[Task]

    def _tools(self):
        return {
            "calc": CalculatorTool(),
            "trace": RetrievalTraceTool(),
        }

    @agent
    def report_parser(self) -> Agent:
        tools = self._tools()
        return Agent(config=self.agents_config["report_parser"], tools=[tools["trace"]], llm=llm, verbose=True)

    @agent
    def environmental_analyst(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["environmental_analyst"],
            tools=[tools["trace"], tools["calc"]],
            allow_delegation=True,
            llm=llm,
            verbose=True,
        )

    @agent
    def social_analyst(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["social_analyst"],
            tools=[tools["trace"], tools["calc"]],
            allow_delegation=True,
            llm=llm,
            verbose=True,
        )

    @agent
    def governance_analyst(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["governance_analyst"],
            tools=[tools["trace"], tools["calc"]],
            allow_delegation=True,
            llm=llm,
            verbose=True,
        )

    @agent
    def score_aggregator(self) -> Agent:
        tools = self._tools()
        return Agent(config=self.agents_config["score_aggregator"], tools=[tools["calc"]], llm=llm, verbose=True)

    @agent
    def performance_comparator(self) -> Agent:
        tools = self._tools()
        return Agent(config=self.agents_config["performance_comparator"], tools=[tools["calc"]], llm=llm, verbose=True)

    @agent
    def metrics_evaluator(self) -> Agent:
        tools = self._tools()
        return Agent(config=self.agents_config["metrics_evaluator"], tools=[tools["calc"]], llm=llm, verbose=True)

    @agent
    def critique_agent(self) -> Agent:
        tools = self._tools()
        return Agent(config=self.agents_config["critique_agent"], tools=[tools["trace"], tools["calc"]], llm=llm, verbose=True)

    @task
    def parse_report_task(self) -> Task:
        return Task(config=self.tasks_config["parse_report_task"])

    @task
    def environmental_scoring_task(self) -> Task:
        return Task(config=self.tasks_config["environmental_scoring_task"])

    @task
    def social_scoring_task(self) -> Task:
        return Task(config=self.tasks_config["social_scoring_task"])

    @task
    def governance_scoring_task(self) -> Task:
        return Task(config=self.tasks_config["governance_scoring_task"])

    @task
    def aggregate_scores_task(self) -> Task:
        return Task(config=self.tasks_config["aggregate_scores_task"])

    @task
    def compare_performance_task(self) -> Task:
        return Task(config=self.tasks_config["compare_performance_task"])

    @task
    def evaluate_metrics_task(self) -> Task:
        return Task(config=self.tasks_config["evaluate_metrics_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            tracing=True,
            verbose=True,
        )
