# Multi-Agent Collaboration Project

A multi-agent AI system built with CrewAI. Multiple specialised agents work together to analyse Financial Sustainability Reports.

---

## What This Project Does

Single LLMs struggle when asked to do many different things at once. This project solves that by splitting work across multiple agents, each scoped to one task. The agents communicate through shared memory and a Flow that coordinates execution order.

**Use case:** Financial Sustainability Reports — documents that contain financial data, ESG metrics, and qualitative statements that must be read and interpreted together.

---

## Project Structure

```
agent_collaboration_project/
├── .env                        ← your credentials and model config (never commit this)
├── .gitignore
├── pyproject.toml              ← project metadata and dependencies
├── README.md
└── src/
    └── project/
        ├── __init__.py
        ├── main.py             ← entry point, runs crews or the full Flow
        ├── llm_config.py       ← LLM setup, reads from .env
        ├── tools/              ← custom tools agents can use
        │   ├── __init__.py
        │   └── custom_tool.py
        ├── flows/              ← flow examples and templates
        │   ├── __init__.py
        │   ├── structured-flow.example.py      ← Flow with typed Pydantic state
        │   └── unstructured-flow.example.py    ← Flow with plain dict state
        └── crews/              ← one folder per crew
            └── greeting_crew/
                ├── __init__.py
                ├── greeting_crew.py
                └── config/
                    ├── agents.yaml     ← agent role, goal, backstory
                    └── tasks.yaml      ← task descriptions and expected outputs
```

### Key concepts

| Concept / Abstraction | Single Sentence Explanation|
|---|---|
| **Agent** | An LLM with a role, a goal, and optional tools |
| **Task** | A specific instruction given to an agent |
| **Crew** | A group of agents + tasks that run together |
| **Flow** | Orchestrates multiple crews, manages shared state |
| **Tool** | A Python function an agent can call (e.g. read a PDF, make a calculation) |

### Structured vs Unstructured Flows

In CrewAI, **Flows** allow you to coordinate multiple tasks and agents. Choosing between a Structured or Unstructured state determines how data is passed and validated between these steps.

| Type | State | Use when |
|---|---|---|
| Structured | Pydantic `BaseModel` | You know the fields upfront. Type-safe. Recommended. |
| Unstructured | Plain `dict` | You need dynamic or unknown fields at runtime. |

See `flows/structured-flow.example.py` and `flows/unstructured-flow.example.py` for templates.

---

## Environment Variables

Create a `.env` file at the project root:

```bash
LLM_MODEL_NAME=your-model-name
LLM_API_KEY=your-key
LLM_BASE_URL=your-model-url
```

These are read by `llm_config.py` file at startup.
`llm_config.py` is the file where you can configure your LLM settings.
It is one central file, as that makes it easy to configure all settings for your LLM accross all agents and crews.
 <b><u>Never commit `.env`</u></b>

---

## UV — Package Manager

`uv` manages dependencies and virtual environments.

### Install

**Mac / Linux using curl:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Mac / Linux using wget:**
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After running your install command you should in top shape to use `uv`. Howerver that may not always be the case, so to be on the safe side, make sure to restart your terminal after installing.

### UV commands that you will use most often

The following table contains a list of the most commonly used commands that you can expect to use throughout your project.

| Command | What it does |
|---|---|
| `uv sync` | Install all dependencies from `pyproject.toml`. Creates the `.venv`  folder for you |
| `uv add <package>` | Add a new dependency and install it |
| `uv run <file.py>` | Run a Python file inside the project environment |
| `uv run <script>` | Run a named script from `pyproject.toml` |

In case there are dependency issues, or you need to get funcky with them, a rule of thumb is to recreate the `.venv` folder

#### What is the `.venv` folder?
<u>The .venv folder is a private, isolated sandbox that stores all the specific tools and libraries your project needs to run.</u> By keeping these files inside your project folder rather than mixing them with your computer’s main system, `uv` ensures that one project's requirements never interfere with another's. <u>Think of it as a custom "tool kit" where the Python version and every package you install are neatly tucked away, ensuring your code works exactly the same every time you run it.</u>

One of the best things about `uv` is that it manages this folder with incredible speed and efficiency. Instead of wasting disk space by downloading the same library multiple times for different projects, uv uses a central cache to link files into the `.venv` folder instantly. You can delete the folder at any time to "reset" your environment and recreate it in seconds using `uv sync`. <u>Just remember to keep it out of your Git repository.</u>

---

## Running the Project

To run the project itself and the scripts defined in your project, you use the following command. This ensures the script runs within your isolated `.venv` environment with all the correct dependencies automatically loaded:

```bash
uv run <project-script>
```

To run a specific function from `main.py` directly:

```bash
# Mac / Linux
uv run src/project/main.py

# Windows
uv run src\project\main.py
```

To add your own scripts, you simply list them in your `pyproject.toml` file under a `[project.scripts]` section. You map the command name you want to use to the specific function in your Python code, see example `pyproject.toml` file below:

```toml
[project]
.....

[project.scripts]
project = "project.main:run"
greeting_crew = "project.main:run"
train = "project.main:train"
replay = "project.main:replay"
test = "project.main:test"
run_with_trigger = "project.main:run_with_trigger"

[build-system]
......

[tool.uv]
.....

[tool.crewai]
....

```

---

## Adding a New Crew
To expand your project with a new specialized crew, follow these steps to ensure it integrates correctly with the CrewAI framework and your `uv` environment:

1.  **Create the Directory Structure** Create a new folder at `src/project/crews/<my_new_crew>/`. Keeping each crew in its own directory ensures that configurations and custom logic remain modular and easy to debug.

2.  **Initialize Required Files** Inside your new folder, add the following boilerplate files:
    * `__init__.py`: Marks the directory as a Python package.
    * `my_crew.py`: This is the engine room where you define your class and logic.
    * `config/agents.yaml`: A clean YAML file to define agent roles, goals, and backstories.
    * `config/tasks.yaml`: A YAML file to define the specific assignments and expected outputs for your agents.

3.  **Implement the `@CrewBase` Pattern** Open `my_crew.py` and mirror the structure of an existing crew. Use the `@CrewBase` decorator to let CrewAI handle the automatic loading of your YAML configurations. This pattern connects your Python methods (decorated with `@agent` and `@task`) directly to the settings defined in your YAML files.


4.  **Register and Invoke in `main.py`** To make your crew executable, import the class into your entry point (`src/project/main.py`). Wrap it in a function—for example, `run_my_new_crew()`—that instantiates the crew and calls `.kickoff()`.

---

### Adding a New Flow

A **Flow** is the "brain" that connects your different crews. While a crew handles specific tasks, the Flow decides the order in which those crews run and how they share information.

* **1. Start with a Template**
    Copy `flows/structured-flow.example.py` or `flows/unstructured-flow.example.py`. These files contain the boilerplate code needed to make the Flow talk to CrewAI.
* **2. Name Your Logic**
    Rename the class and the "State" model. The State is just a shared container (like a backpack) that holds the data as it moves from the first crew to the last.
* **3. Map the Sequence**
    Use `@start()` to pick the first action and `@listen()` to tell the next action to wait for the one before it. This "wires" your crews together in a specific chain.
* **4. Activate in Main**
    Go to `main.py`, import your new Flow, and create a function that calls `.kickoff()`. This is the "start button" that puts the entire sequence into motion.
---

## ESG Multi-Agent Evaluation Framework

This repository now includes a modular CrewAI-based ESG evaluation framework focused on sustainability report analysis with RAG-style chunk retrieval and tracing.

### Included orchestration patterns

1. **Parallel + concurrent (baseline)**
2. **Handoff / hierarchical** (domain scorers aggregate delegated worker subsets)
3. **Review and critique** (domain scorer + critique loop)

Each flow processes **one report per run**:

- `ParallelConcurrentESGFlow`
- `HandoffHierarchicalESGFlow`
- `ReviewCritiqueESGFlow`

Implementation modules live under `src/project/esg_framework/` and `src/project/flows/esg_flows.py`.

### Agent + task definitions

A CrewBase-style crew matching the repository convention is included in:

- `src/project/crews/esg_evaluation_crew/crew.py`
- `src/project/crews/esg_evaluation_crew/config/agents.yaml`
- `src/project/crews/esg_evaluation_crew/config/tasks.yaml`

Agent definitions include role, goal, backstory, tools, allow_delegation, and verbose fields.

### Data scope

- Dataset: `knowledge/sustainability-reports-preprocessed.csv`
- Focus: healthcare companies (sector when available, keyword fallback otherwise)
- Sample mode: first `N` healthcare reports (default 10)

### Run local comparison (first 10 reports)

```bash
uv run esg_experiments --sample-size 10 --trials 3 --output /tmp/esg_experiment_results.json
```

Without `uv`, run directly:

```bash
PYTHONPATH=src python -m project.esg_experiments --sample-size 10 --trials 3 --output /tmp/esg_experiment_results.json
```

Outputs include per-pattern summary and comparative metrics for:

- Coverage (weighted + partial)
- Accuracy (MAE + normalized accuracy + local judge-mode proxy abstraction)
- Predicted and actual total score averages
- Consistency (qualitative + quantitative)
- Inter-agent agreement (Fleiss' Kappa + pairwise Pearson)
- Latency and efficiency
- Hallucination rates
- Deliberation quality

Retrieval traces and chunk stores are serialized (default `/tmp/esg_chunk_store`) to support coverage analysis and reproducibility.

### Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
