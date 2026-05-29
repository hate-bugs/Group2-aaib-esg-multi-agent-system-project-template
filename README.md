# ESG Multi-Agent Evaluation Framework

This repository now includes a modular ESG evaluation framework built around CrewAI-style agents, tasks, tools, and flows for comparing multi-agent orchestration patterns on healthcare sustainability reports.

## What it does

The framework evaluates the first `N` reports from `sustainability-reports-2026-05-29.csv` and compares three orchestration patterns for ESG score prediction:

1. **Parallel + concurrent**
2. **Handoff / hierarchical**
3. **Review and critique**

Each report is parsed into chunks, labeled for ESG relevance, retrieved in a RAG-style evidence pass, scored for E/S/G, aggregated into a total score, and evaluated with shared metrics.

## Project structure

```text
src/project/
├── crews/
│   ├── esg_evaluation_crew/
│   │   ├── config/agents.yaml
│   │   ├── config/tasks.yaml
│   │   └── crew.py
│   └── greeting_crew/
├── esg/
│   ├── config.py
│   ├── data.py
│   ├── chunking.py
│   ├── retrieval.py
│   ├── heuristics.py
│   ├── metrics.py
│   ├── scoring.py
│   └── runner.py
├── flows/
│   └── esg_flows.py
├── outputs/sample/first10_metrics_summary.json
├── tools/
│   ├── custom_tool.py
│   └── esg_tools.py
├── llm_config.py
└── main.py
```

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Optional environment variables for live LLM-backed CrewAI use

Create `.env` in the repository root if you want live LLM-backed CrewAI agents:

```bash
LLM_MODEL_NAME=your-model
LLM_BASE_URL=https://your-endpoint
LLM_API_KEY=your-key
```

If these are not set, the repository still runs the deterministic experiment framework and tests without crashing.

## Dataset

Default dataset path:

```text
/tmp/workspace/hate-bugs/Group2-aaib-esg-multi-agent-system-project-template/knowledge/sustainability-reports-2026-05-29.csv
```

The loader supports either:

- preprocessed report text (`preprocessed_content` and aliases), or
- report PDF URLs (`Report Link` and aliases), which are downloaded and parsed for RAG chunking.

Supported field aliases include:

- `report_id`, `id`, `company_id`, `isin`
- `company_name`, `company`, `issuer_name`, `name`
- `preprocessed_content`, `processed_content`, `content`, `report_text`, `text`
- `report_link`, `Report Link`, `report url`, `pdf_url`, `url`
- `environmental_score`, `social_score`, `governance_score`, `esg_score`

## How to run

### Run all three patterns on the first 10 reports

```bash
uv run esg_experiments --sample-size 10
```

### Run a single pattern

```bash
uv run esg_experiments --pattern parallel
uv run esg_experiments --pattern hierarchical
uv run esg_experiments --pattern review
```

### Use a different dataset path

```bash
uv run esg_experiments --dataset-path /absolute/path/to/sustainability-reports-2026-05-29.csv
```

### Change hyperparameters

```bash
uv run esg_experiments \
  --sample-size 10 \
  --chunk-size 220 \
  --chunk-overlap 40 \
  --top-k 5 \
  --trials 4 \
  --critique-max-iterations 4
```

## Outputs

Generated artifacts:

- `outputs/indexes/<report_id>.json` — persisted report chunks with ESG labels
- `outputs/experiments/first10_metrics_summary.json` — latest experiment output
- `src/project/outputs/sample/first10_metrics_summary.json` — committed sample output artifact

## Orchestration patterns

### 1. Parallel + concurrent

Pipeline:

```text
loader -> parser -> E/S/G scorers in parallel -> aggregator -> comparator -> metrics
```

### 2. Handoff / hierarchical

Pipeline:

```text
loader -> parser -> domain scorers -> worker chunk splits -> domain aggregation -> aggregator -> comparator -> metrics
```

### 3. Review and critique

Pipeline:

```text
loader -> parser -> scorer <-> critique loop -> aggregator -> comparator -> metrics
```

The critique loop is bounded by `critique_max_iterations`.

## Agent definitions

CrewAI-style agents are defined in `src/project/crews/esg_evaluation_crew/config/agents.yaml` and include:

- Report Parser
- Environmental Analyst
- Social Analyst
- Governance Analyst
- Score Aggregator
- Performance Comparator
- Metrics Evaluator
- Critique Agent

Each agent definition includes:

- `role`
- `goal`
- `backstory`
- `tools`
- `allow_delegation`
- `verbose`

## Scoring contract

Every ESG domain scorer returns:

- `estimated_score`
- `confidence`
- `rationale`

The rationale is grounded in retrieved chunk IDs and only uses `preprocessed_content`.

## Retrieval and indexing

- Reports are chunked by configurable token-like word windows.
- Each chunk is labeled for environmental, social, and/or governance relevance.
- Chunk indexes are persisted as JSON artifacts for local retrieval.
- Retrieval uses lexical/domain heuristic overlap to approximate RAG behavior.

## Metrics implemented

The framework reports operational versions of the requested metrics:

1. **Coverage**
   - weighted binary coverage
   - partial token coverage
2. **Accuracy**
   - with GT: macro precision/recall/F1 and MAE
   - without GT: judge score average approximation
3. **Consistency**
   - repeated-trial similarity across trials
4. **Inter-Agent Agreement**
   - Fleiss' kappa over score bands
   - pairwise Pearson-style continuous agreement approximation
5. **Latency and Efficiency**
   - wall-clock latency
   - critical path latency
   - coverage-per-call
   - coverage-per-token
6. **Hallucination Rate**
   - unsupported claim rate
   - partially supported claim rate
7. **Agent Deliberation Quality**
   - conflict detection / resolution quality / dominance ratio composite

## Metric assumptions

Some requested formulas require LLM judging or external ground truth not guaranteed to exist in every CSV. This implementation makes pragmatic, inspectable approximations:

- chunk retrieval coverage is based on retrieved chunk IDs and token counts
- judge score uses grounded vs unsupported heuristic claims in the generated rationale
- consistency compares repeated-trial score distance plus rationale token overlap
- agreement maps continuous scores into `low`, `medium`, `high` bands for kappa
- deliberation quality uses critique and worker review traces as the deliberation record
- the report `preprocessed_content` is treated as the grounding source of truth for support checks

## Validation

Run focused tests with:

```bash
uv run python -m unittest discover -s tests
```

Run a compile sanity check with:

```bash
uv run python -m compileall src
```

## Known limitations

- The deterministic experiment engine is designed to be reproducible and testable without requiring external LLM credentials.
- CrewAI agents/tasks/flows are defined in-repo, but the committed sample artifact is generated from the deterministic execution path for offline validation.
- Retrieval uses lightweight lexical heuristics rather than an embedding database.
- If your real CSV uses different column names, pass the expected aliases or update the loader candidates in `src/project/esg/data.py`.
