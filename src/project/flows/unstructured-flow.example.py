from crewai.flow.flow import Flow, listen, start

class UnstructuredStateFlow(Flow):
    @start()
    def initialize_data(self):
        print("Initializing flow data")
        # Add key-value pairs to state
        self.state["user_name"] = "Anton"
        self.state["preferences"] = {
            "theme": "dark",
            "language": "English"
        }
        self.state["items"] = []

        # The flow state automatically gets a unique ID
        print(f"Flow ID: {self.state['id']}")

        return "Initialized"

    @listen(initialize_data)
    def process_data(self, previous_result):
        print(f"Previous step returned: {previous_result}")

        # Access and modify state
        user = self.state["user_name"]
        print(f"Processing data for {user}")

        # Add items to a list in state
        self.state["items"].append("item1")
        self.state["items"].append("item2")

        # Add a new key-value pair
        self.state["processed"] = True

        return "Processed"

    @listen(process_data)
    def generate_summary(self, previous_result):
        # Access multiple state values
        user = self.state["user_name"]
        theme = self.state["preferences"]["theme"]
        items = self.state["items"]
        processed = self.state.get("processed", False)

        summary = f"User {user} has {len(items)} items with {theme} theme. "
        summary += "Data is processed." if processed else "Data is not processed."

        return summary

# Run the flow
# flow = UnstructuredStateFlow()
# result = flow.kickoff()
# print(f"Final result: {result}")
# print(f"Final state: {flow.state}")