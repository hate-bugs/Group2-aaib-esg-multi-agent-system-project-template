#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from project.crews.greeting_crew.crew import GreetingCrew
from project.esg.config import ESGExperimentConfig
from project.esg.runner import run_experiments

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run() -> None:
    """Run the greeting crew example."""
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year),
    }
    try:
        GreetingCrew().crew().kickoff(inputs=inputs)
    except Exception as exc:  # pragma: no cover - passthrough for manual use
        raise Exception(f"An error occurred while running the crew: {exc}") from exc


def train() -> None:
    """Train the greeting crew for a given number of iterations."""
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year),
    }
    try:
        GreetingCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as exc:  # pragma: no cover - passthrough for manual use
        raise Exception(f"An error occurred while training the crew: {exc}") from exc


def replay() -> None:
    """Replay the greeting crew from a specific task."""
    try:
        GreetingCrew().crew().replay(task_id=sys.argv[1])
    except Exception as exc:  # pragma: no cover - passthrough for manual use
        raise Exception(f"An error occurred while replaying the crew: {exc}") from exc


def test() -> None:
    """Test the greeting crew."""
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year),
    }
    try:
        GreetingCrew().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except Exception as exc:  # pragma: no cover - passthrough for manual use
        raise Exception(f"An error occurred while testing the crew: {exc}") from exc


def run_with_trigger():
    """Run the greeting crew using a JSON trigger payload."""
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        raise Exception("Invalid JSON payload provided as argument") from exc

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": "",
    }

    try:
        return GreetingCrew().crew().kickoff(inputs=inputs)
    except Exception as exc:  # pragma: no cover - passthrough for manual use
        raise Exception(f"An error occurred while running the crew with trigger: {exc}") from exc


def run_esg_experiments() -> None:
    """Run the ESG experiment framework across one or more orchestration patterns."""
    parser = argparse.ArgumentParser(description="Run ESG multi-agent experiments")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--pattern", choices=["parallel", "hierarchical", "review"], default=None)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--chunk-overlap", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--critique-max-iterations", type=int, default=3)
    args = parser.parse_args(sys.argv[1:])

    config = ESGExperimentConfig(
        dataset_path=args.dataset_path,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        trials=args.trials,
        critique_max_iterations=args.critique_max_iterations,
    )
    output = run_experiments(config, pattern=args.pattern)
    print(f"Saved metrics summary to {Path(output['config']['output_dir']) / 'first10_metrics_summary.json'}")
