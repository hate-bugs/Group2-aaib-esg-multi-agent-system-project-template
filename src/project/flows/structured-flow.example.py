from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Define your state model
class UserPreferences(BaseModel):
    theme: str = "light"
    language: str = "English"

class AppState(BaseModel):
    user_name: str = ""
    preferences: UserPreferences = UserPreferences()
    items: List[str] = []
    processed: bool = False
    completion_percentage: float = 0.0

# Create a flow with typed state
class StructuredStateFlow(Flow[AppState]):
    @start()
    def initialize_data(self):
        print("Initializing flow data")
        # Set state values (type-checked)
        self.state.user_name = "Taylor"
        self.state.preferences.theme = "dark"

        # The ID field is automatically available
        print(f"Flow ID: {self.state.id}")

        return "Initialized"

    @listen(initialize_data)
    def process_data(self, previous_result):
        print(f"Processing data for {self.state.user_name}")

        # Modify state (with type checking)
        self.state.items.append("item1")
        self.state.items.append("item2")
        self.state.processed = True
        self.state.completion_percentage = 50.0

        return "Processed"

    @listen(process_data)
    def generate_summary(self, previous_result):
        # Access state (with autocompletion)
        summary = f"User {self.state.user_name} has {len(self.state.items)} items "
        summary += f"with {self.state.preferences.theme} theme. "
        summary += "Data is processed." if self.state.processed else "Data is not processed."
        summary += f" Completion: {self.state.completion_percentage}%"

        return summary

# Run the flow
# flow = StructuredStateFlow()
# result = flow.kickoff()
# print(f"Final result: {result}")
# print(f"Final state: {flow.state}")