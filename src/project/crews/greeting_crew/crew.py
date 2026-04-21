from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from project.llm_config import llm


@CrewBase
class GreetingCrew():
    agents: list[BaseAgent]
    tasks: list[Task]
    
    @agent
    def greeting_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['greeting_agent'], # type: ignore[index]
            verbose=True,
            llm=llm
        )
    
    @task
    def greeting_task(self) -> Task:
        return Task(
            config=self.tasks_config['greeting_task'], # type: ignore[index],
            output_file='outputs/greeting_crew/greeting.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            name="Greetring Crew",
            streaming=True,
            tracing=True
        )
