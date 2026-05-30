from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.esg_framework.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ESG multi-agent orchestration comparison experiments")
    parser.add_argument(
        "--dataset",
        default="knowledge/sustainability-reports-preprocessed.csv",
        help="Path to preprocessed report CSV",
    )
    parser.add_argument("--sample-size", type=int, default=10, help="Healthcare report sample size")
    parser.add_argument("--trials", type=int, default=3, help="Trials per report/pattern for consistency")
    parser.add_argument(
        "--output",
        default="/tmp/esg_experiment_results.json",
        help="Output JSON path for detailed results",
    )
    parser.add_argument(
        "--chunk-store-dir",
        default="/tmp/esg_chunk_store",
        help="Directory for serialized chunk stores used for retrieval tracing",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)

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
