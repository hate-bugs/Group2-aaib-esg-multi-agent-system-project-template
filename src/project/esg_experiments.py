from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from project.esg_framework.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ESG multi-agent orchestration comparison experiments")
    # Environment-overridable defaults (each can be set via an environment variable):
    # ESG_DATASET_PATH: path to the preprocessed CSV of reports
    default_dataset = os.getenv("ESG_DATASET_PATH", "knowledge/sustainability-reports-preprocessed.csv")
    # ESG_SAMPLE_SIZE: number of reports to sample for experiments
    default_sample_size = int(os.getenv("ESG_SAMPLE_SIZE", "10"))
    # ESG_TRIALS: number of trials per pattern/report to run
    default_trials = int(os.getenv("ESG_TRIALS", "3"))
    # ESG_OUTPUT_PATH: where to write the experiment results JSON
    default_output = os.getenv("ESG_OUTPUT_PATH", "output/esg_experiment_results.json")
    # ESG_CHUNK_STORE_DIR: directory to store serialized chunk stores used for retrieval tracing
    default_chunk_store = os.getenv("ESG_CHUNK_STORE_DIR", "output/tmp/esg_chunk_store")

    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help="Path to preprocessed report CSV",
    )
    parser.add_argument("--sample-size", type=int, default=default_sample_size, help="Healthcare report sample size")
    parser.add_argument("--trials", type=int, default=default_trials, help="Trials per report/pattern for consistency")
    parser.add_argument(
        "--output",
        default=default_output,
        help="Output JSON path for detailed results",
    )
    parser.add_argument(
        "--chunk-store-dir",
        default=default_chunk_store,
        help="Directory for serialized chunk stores used for retrieval tracing",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Normalize any system /tmp paths to the repository-local `output/` folder so
    # temporary files are created inside the project rather than global /tmp.
    repo_root = Path(__file__).resolve().parents[2]

    def _is_system_tmp(p: str) -> bool:
        if not p:
            return False
        s = str(p)
        return s == "/tmp" or s == "/private/tmp" or s.startswith("/tmp/") or s.startswith("/private/tmp/")

    if _is_system_tmp(args.output):
        args.output = str(repo_root / "output" / Path(args.output).name)

    if _is_system_tmp(args.chunk_store_dir):
        # keep same directory name but place under repo_root/output/tmp
        args.chunk_store_dir = str(repo_root / "output" / "tmp" / Path(args.chunk_store_dir).name)

    payload = run_experiment(
        dataset_path=args.dataset,
        sample_size=args.sample_size,
        trials=args.trials,
        output_path=args.output,
        chunk_store_dir=args.chunk_store_dir,
    )

    summary_view = {
        "sample_size": payload["sample_size"],
        "healthcare_pool_size": payload["healthcare_pool_size"],
        "patterns": payload["patterns"],
        "summary": payload["summary"],
        "output_file": str(Path(args.output).resolve()),
    }
    print(json.dumps(summary_view, indent=2, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
