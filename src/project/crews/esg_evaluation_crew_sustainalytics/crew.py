from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from project.llm_config import llm
from project.tools import CalculatorTool, RetrievalTraceTool


@CrewBase
class ESGEvaluationCrewSustainalytics:
    """
    ESG Evaluation Crew following the Morningstar Sustainalytics ESG Risk Ratings Methodology.
    
    This crew implements the comprehensive Sustainalytics framework with 8 specialized agents:
    1. Material ESG Issues Analyst - Identifies relevant MEIs and subindustry exposure
    2. Corporate and Stakeholder Governance Analyst - Evaluates baseline governance MEIs
    3. Systemic and Idiosyncratic Issues Analyst - Assesses sea change and company-specific events
    4. Exposure Dimension Analyst - Calculates company-specific exposure using Beta indicators
    5. Management Dimension Analyst - Evaluates management quality and unmanaged risk
    6. ESG Risk Score Calculator - Computes final risk rating and category
    7. Data Quality and Validation Analyst - Validates data sources and quality
    8. Report Aggregator and Finalizer - Compiles comprehensive ESG Risk Rating report
    """
    agents: list[BaseAgent]
    tasks: list[Task]

    def _tools(self):
        return {
            "calc": CalculatorTool(),
            "trace": RetrievalTraceTool(),
        }

    @agent
    def material_esg_analyst(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["material_esg_analyst"],
            tools=[tools["trace"], tools["calc"]],
            llm=llm,
            verbose=True
        )

    @agent
    def governance_analyst(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["governance_analyst"],
            tools=[tools["trace"], tools["calc"]],
            llm=llm,
            verbose=True
        )

    @agent
    def systemic_idiosyncratic_analyst(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["systemic_idiosyncratic_analyst"],
            tools=[tools["trace"], tools["calc"]],
            llm=llm,
            verbose=True
        )

    @agent
    def exposure_analyst(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["exposure_analyst"],
            tools=[tools["trace"], tools["calc"]],
            llm=llm,
            verbose=True
        )

    @agent
    def management_analyst(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["management_analyst"],
            tools=[tools["trace"], tools["calc"]],
            llm=llm,
            verbose=True
        )

    @agent
    def esg_score_calculator(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["esg_score_calculator"],
            tools=[tools["calc"]],
            llm=llm,
            verbose=True
        )

    @agent
    def data_quality_analyst(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["data_quality_analyst"],
            tools=[tools["trace"]],
            llm=llm,
            verbose=True
        )

    @agent
    def report_aggregator(self) -> Agent:
        tools = self._tools()
        return Agent(
            config=self.agents_config["report_aggregator"],
            tools=[tools["trace"], tools["calc"]],
            allow_delegation=True,
            llm=llm,
            verbose=True
        )

    @task
    def identify_material_esg_issues_task(self) -> Task:
        return Task(config=self.tasks_config["identify_material_esg_issues_task"])

    @task
    def assess_governance_task(self) -> Task:
        return Task(config=self.tasks_config["assess_governance_task"])

    @task
    def identify_systemic_idiosyncratic_issues_task(self) -> Task:
        return Task(config=self.tasks_config["identify_systemic_idiosyncratic_issues_task"])

    @task
    def calculate_exposure_task(self) -> Task:
        return Task(config=self.tasks_config["calculate_exposure_task"])

    @task
    def assess_management_task(self) -> Task:
        return Task(config=self.tasks_config["assess_management_task"])

    @task
    def calculate_esg_risk_score_task(self) -> Task:
        return Task(config=self.tasks_config["calculate_esg_risk_score_task"])

    @task
    def validate_data_quality_task(self) -> Task:
        return Task(config=self.tasks_config["validate_data_quality_task"])

    @task
    def compile_final_report_task(self) -> Task:
        return Task(config=self.tasks_config["compile_final_report_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            tracing=True,
            verbose=True,
        )
