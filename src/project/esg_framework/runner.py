from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from project.esg_framework.data import filter_healthcare_reports, load_reports, select_sample
from project.esg_framework.metrics import aggregate_pattern_metrics, consistency_metric
from project.esg_framework.models import ReportRunResult
from project.esg_framework.patterns import (
    run_handoff_pattern,
    run_parallel_pattern,
    run_review_critique_pattern,
)
from project.esg_framework.retrieval import ChunkStore

def _safe_path_fragment(value: str, max_len: int = 64) -> str:
    clean = "".join(ch for ch in str(value) if ch.isalnum() or ch in ("-", "_"))
    return (clean[:max_len] if clean else "unknown")


PATTERN_FUNCTIONS = {
    "parallel_concurrent": run_parallel_pattern,
    "handoff_hierarchical": run_handoff_pattern,
    "review_critique": run_review_critique_pattern,
}


def _serialize_report_result(result: ReportRunResult) -> dict[str, Any]:
    return {
        "report_id": result.report_id,
        "pattern": result.pattern,
        "total_score": result.total_score,
        "confidence": result.confidence,
        "domain_scores": {
            name: {
                "estimated_score": output.estimated_score,
                "confidence": output.confidence,
                "rationale": output.rationale,
                "label": output.label,
                "retrieved_chunk_ids": output.retrieved_chunk_ids,
                "used_chunk_ids": output.used_chunk_ids,
            }
            for name, output in result.domain_scores.items()
        },
        "comparison": result.comparison,
        "metrics": result.metrics,
        "metadata": result.metadata,
    }


def run_experiment(
    dataset_path: str | Path,
    sample_size: int = 10,
    trials: int = 3,
    output_path: str | Path | None = None,
    chunk_store_dir: str | Path = "/tmp/esg_chunk_store",
) -> dict[str, Any]:
    records = load_reports(dataset_path)
    healthcare = filter_healthcare_reports(records)
    sample = select_sample(healthcare, sample_size)

    chunk_store = ChunkStore()
    all_results: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern in PATTERN_FUNCTIONS}
    detailed: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern in PATTERN_FUNCTIONS}

    for record in sample:
        for pattern_name, fn in PATTERN_FUNCTIONS.items():
            trials_results: list[ReportRunResult] = []
            for _ in range(max(1, trials)):
                result = fn(record, chunk_store)
                trials_results.append(result)

            consistency = consistency_metric([result.domain_scores for result in trials_results])
            representative = trials_results[0]
            representative.metrics["consistency_quantitative"] = consistency
            representative.metrics["consistency_qualitative"] = (
                "Stable" if consistency >= 0.85 else "Moderate variance" if consistency >= 0.7 else "High variance"
            )

            all_results[pattern_name].append(representative.metrics)
            detailed[pattern_name].append(_serialize_report_result(representative))

            safe_id = _safe_path_fragment(record.report_id)
            chunk_file = Path(chunk_store_dir) / f"report_{safe_id}_{pattern_name}.json"
            chunk_store.persist_json(record.report_id, chunk_file)

    summary = {pattern: aggregate_pattern_metrics(metrics) for pattern, metrics in all_results.items()}

    comparison_table = [
        {
            "pattern": pattern,
            **stats,
        }
        for pattern, stats in summary.items()
    ]

    payload = {
        "dataset": str(dataset_path),
        "sample_size": len(sample),
        "trials": max(1, trials),
        "healthcare_pool_size": len(healthcare),
        "patterns": list(PATTERN_FUNCTIONS.keys()),
        "summary": summary,
        "comparison": comparison_table,
        "results": detailed,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return payload
