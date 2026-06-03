from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from project.esg_framework_sustainalytics.runner_sustainalytics import run_experiment_sustainalytics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ESG multi-agent orchestration comparison experiments with Sustainalytics methodology"
    )
    # Environment-overridable defaults
    default_dataset = os.getenv("ESG_DATASET_PATH_SUSTAINALYTICS", "knowledge/sustainability-reports-preprocessed.csv")
    default_sample_size = int(os.getenv("ESG_SAMPLE_SIZE_SUSTAINALYTICS", "10"))
    default_trials = int(os.getenv("ESG_TRIALS_SUSTAINALYTICS", "3"))
    default_output = os.getenv("ESG_OUTPUT_PATH_SUSTAINALYTICS", "output/esg_experiment_results_sustainalytics.json")
    default_chunk_store = os.getenv("ESG_CHUNK_STORE_DIR_SUSTAINALYTICS", "output/tmp/esg_chunk_store_sustainalytics")

    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help="Path to preprocessed report CSV for Sustainalytics",
    )
    parser.add_argument("--sample-size", type=int, default=default_sample_size, help="Report sample size")
    parser.add_argument("--trials", type=int, default=default_trials, help="Trials per report/pattern")
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

    # Normalize any system /tmp paths to the repository-local `output/` folder
    repo_root = Path(__file__).resolve().parents[2]

    def _is_system_tmp(p: str) -> bool:
        if not p:
            return False
        s = str(p)
        return s == "/tmp" or s == "/private/tmp" or s.startswith("/tmp/") or s.startswith("/private/tmp/")

    if _is_system_tmp(args.output):
        args.output = str(repo_root / "output" / Path(args.output).name)

    if _is_system_tmp(args.chunk_store_dir):
        args.chunk_store_dir = str(repo_root / "output" / "tmp" / Path(args.chunk_store_dir).name)

    payload = run_experiment_sustainalytics(
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
