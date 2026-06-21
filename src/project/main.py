#!/usr/bin/env python
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from project.esg_experiments import main as run_esg_cli
from project.esg_experiments_sustainalytics import main as run_esg_cli_sustainalytics

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run_esg_experiments():
    """
    Run ESG orchestration experiments on preprocessed sustainability reports.
    """
    try:
        run_esg_cli(sys.argv[1:])
    except Exception as e:
        raise Exception(f"An error occurred while running ESG experiments: {e}")


def run_esg_experiments_sustainalytics():
    """
    Run ESG orchestration experiments with Sustainalytics methodology on preprocessed sustainability reports.
    """
    try:
        run_esg_cli_sustainalytics(sys.argv[1:])
    except Exception as e:
        raise Exception(f"An error occurred while running ESG Sustainalytics experiments: {e}")


def _run_single_hyperparameter_config(config, base_args):
    """
    Run a single hyperparameter configuration.
    This is a helper function designed to be called from a process pool.
    """
    # Store original env vars for this process
    original_env = {}
    for key in config["env"].keys():
        original_env[key] = os.environ.get(key)

    try:
        print(f"\n{'='*60}")
        print(f"Running Sustainalytics experiment with configuration: {config['name']}")
        print(f"{'='*60}")

        # Apply hyperparameters
        for key, value in config["env"].items():
            os.environ[key] = value

        # Generate unique output path for this configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/esg_experiment_sustainalytics_{config['name']}_{timestamp}.json"
        chunk_store = f"output/tmp/esg_chunk_store_sustainalytics_{config['name']}_{timestamp}"

        args = base_args + ["--output", output_path, "--chunk-store-dir", chunk_store]
        run_esg_cli_sustainalytics(args)
        
        print(f"\n{'='*60}")
        print(f"Completed configuration: {config['name']}")
        print(f"{'='*60}")
        
    finally:
        # Restore original environment for this process
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_esg_experiments_sustainalytics_hyperparameter():
    """
    Run ESG Sustainalytics experiments with n=30, 2 trials, and multiple hyperparameter configurations.
    Uses multiprocessing for config-level parallelism and threading for report-level parallelism.
    """
    # Base configuration
    base_args = [
        "--sample-size", "30",
        "--trials", "2",
    ]

    # Define hyperparameter configurations
    # Each configuration varies at least one parameter per pattern (parallel, handoff, review)
    # to enable cross-pattern comparison since patterns run independently
    control_base = {
        "PATTERN_PARALLEL_MAX_WORKERS": "3",
        "PATTERN_PARALLEL_MAX_CHUNKS": "6",
        "PATTERN_HANDOFF_MAX_WORKERS": "2",
        "PATTERN_HANDOFF_WORKER_GROUPS": "2",
        "PATTERN_HANDOFF_MAX_CHUNKS": "8",
        "PATTERN_REVIEW_MAX_CHUNKS": "6",
        "PATTERN_REVIEW_CRITIQUE_CHUNKS": "4",
        "PATTERN_REVIEW_MAX_ROUNDS": "3",
    }

    configs = [
        {
            "name": "control",
            "env": control_base.copy()
        },
        {
            "name": "no_critique",
            "env": {
                **control_base.copy(),
                "PATTERN_PARALLEL_MAX_CHUNKS": "8",
                "PATTERN_HANDOFF_WORKER_GROUPS": "3",
                "PATTERN_REVIEW_MAX_ROUNDS": "0",
            }
        },
        {
            "name": "high_parallel",
            "env": {
                **control_base.copy(),
                "PATTERN_PARALLEL_MAX_WORKERS": "5",
                "PATTERN_PARALLEL_MAX_CHUNKS": "10",
                "PATTERN_HANDOFF_MAX_WORKERS": "1",
                "PATTERN_HANDOFF_MAX_CHUNKS": "6",
                "PATTERN_REVIEW_MAX_CHUNKS": "8",
            }
        },
        {
            "name": "high_handoff",
            "env": {
                **control_base.copy(),
                "PATTERN_PARALLEL_MAX_WORKERS": "2",
                "PATTERN_PARALLEL_MAX_CHUNKS": "4",
                "PATTERN_HANDOFF_MAX_WORKERS": "4",
                "PATTERN_HANDOFF_WORKER_GROUPS": "4",
                "PATTERN_HANDOFF_MAX_CHUNKS": "12",
                "PATTERN_REVIEW_CRITIQUE_CHUNKS": "6",
            }
        },
        {
            "name": "high_review",
            "env": {
                **control_base.copy(),
                "PATTERN_PARALLEL_MAX_WORKERS": "2",
                "PATTERN_HANDOFF_WORKER_GROUPS": "1",
                "PATTERN_REVIEW_MAX_CHUNKS": "10",
                "PATTERN_REVIEW_CRITIQUE_CHUNKS": "6",
                "PATTERN_REVIEW_MAX_ROUNDS": "5",
            }
        },
    ]

    # Determine number of workers for process pool
    num_processes = min(len(configs), os.cpu_count() or 4)
    print(f"\n[PARALLEL] Using {num_processes} processes for {len(configs)} configurations")
    print(f"[PARALLEL] Each process will use thread-level parallelism for reports")

    try:
        # Use ProcessPoolExecutor for config-level parallelism
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all config tasks
            futures = {
                executor.submit(_run_single_hyperparameter_config, config, base_args): config["name"]
                for config in configs
            }
            
            # Wait for all to complete
            for future in as_completed(futures):
                config_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Configuration '{config_name}' failed: {e}")
                    raise

    except Exception as e:
        raise Exception(f"An error occurred while running ESG Sustainalytics hyperparameter experiments: {e}")
