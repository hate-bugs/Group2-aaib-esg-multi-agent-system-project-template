from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from project.esg.config import ESGExperimentConfig
from project.esg.data import load_reports
from project.esg.models import PATTERNS, PatternRunResult
from project.esg.scoring import add_consistency, run_hierarchical_pattern, run_parallel_pattern, run_review_pattern

PATTERN_RUNNERS = {
    "parallel": run_parallel_pattern,
    "hierarchical": run_hierarchical_pattern,
    "review": run_review_pattern,
}


def _summarize(results: list[PatternRunResult]) -> dict[str, object]:
    summary: dict[str, object] = {"patterns": {}}
    for pattern in PATTERNS:
        pattern_results = [result for result in results if result.pattern == pattern and result.trace.trial == 1]
        if not pattern_results:
            continue
        summary["patterns"][pattern] = {
            "reports": len(pattern_results),
            "average_total_score": round(sum(item.total_score for item in pattern_results) / len(pattern_results), 4),
            "average_coverage": round(
                sum(float(item.metrics["coverage"]["weighted_binary"]) for item in pattern_results) / len(pattern_results),
                4,
            ),
            "average_partial_coverage": round(
                sum(float(item.metrics["coverage"]["partial"]) for item in pattern_results) / len(pattern_results),
                4,
            ),
            "average_wall_clock_latency": round(
                sum(float(item.metrics["latency_efficiency"]["wall_clock_latency"]) for item in pattern_results) / len(pattern_results),
                4,
            ),
            "average_judge_or_mae": round(
                sum(
                    float(item.metrics["accuracy"].get("judge_score_average") or item.metrics["accuracy"].get("mae") or 0.0)
                    for item in pattern_results
                )
                / len(pattern_results),
                4,
            ),
            "average_consistency": round(
                sum(float(item.metrics["consistency"]["trial_similarity"]) for item in pattern_results) / len(pattern_results),
                4,
            ),
        }
    return summary


def run_experiments(config: ESGExperimentConfig, *, pattern: str | None = None) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    config = config.with_root(root)
    reports = load_reports(config.dataset_path, config.sample_size)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.sample_output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_patterns = [pattern] if pattern else list(PATTERN_RUNNERS)
    results: list[PatternRunResult] = []
    for report in reports:
        for pattern_name in selected_patterns:
            runner = PATTERN_RUNNERS[pattern_name]
            for trial in range(1, config.trials + 1):
                results.append(runner(report, config, trial))

    add_consistency(results)
    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "results": [result.to_dict() for result in results if result.trace.trial == 1],
        "trial_results": [result.to_dict() for result in results],
        "summary": _summarize(results),
    }
    output_path = config.output_dir / "first10_metrics_summary.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    config.sample_output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
