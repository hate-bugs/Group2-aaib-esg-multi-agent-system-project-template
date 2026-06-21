from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from project.esg_framework_sustainalytics.data_sustainalytics import filter_healthcare_reports, load_reports, select_sample
from project.esg_framework_sustainalytics.metrics_sustainalytics import aggregate_pattern_metrics, consistency_metric
from project.esg_framework_sustainalytics.models_sustainalytics import ReportRunResult
from project.esg_framework_sustainalytics.patterns_sustainalytics import (
    run_handoff_pattern,
    run_parallel_pattern,
    run_review_critique_pattern,
)
from project.esg_framework_sustainalytics.retrieval_sustainalytics import ChunkStore


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


def run_experiment_sustainalytics(
    dataset_path: str | Path,
    sample_size: int = 10,
    trials: int = 3,
    output_path: str | Path | None = None,
    chunk_store_dir: str | Path = "output/tmp/esg_chunk_store_sustainalytics",
) -> dict[str, Any]:
    """Run the Sustainalytics ESG framework experiment with the same patterns as the original framework."""
    def _is_system_tmp(p: str | Path) -> bool:
        if not p:
            return False
        s = str(p)
        return s == "/tmp" or s == "/private/tmp" or s.startswith("/tmp/") or s.startswith("/private/tmp/")

    if _is_system_tmp(chunk_store_dir):
        repo_root = Path(__file__).resolve().parents[3]
        chunk_store_dir = str(repo_root / "output" / "tmp" / Path(chunk_store_dir).name)
    
    print(f"[VERBOSE] Loading reports from: {dataset_path}")
    records = load_reports(dataset_path)
    print(f"[VERBOSE] Loaded {len(records)} total records")
    sample = select_sample(records, sample_size)
    print(f"[VERBOSE] Selected {len(sample)} reports for sampling")
    
    # Pre-warm Crew Agent on the main thread so worker threads can reuse the cached instance.
    from project.esg_framework_sustainalytics.scoring_sustainalytics import _get_cached_management_agent

    cached_agent = _get_cached_management_agent()
    if cached_agent is None:
        raise RuntimeError(
            "Failed to pre-warm Crew management_analyst Agent on main thread. "
            "Verify Crew config at src/project/crews/esg_evaluation_crew_sustainalytics/config/agents.yaml "
            "and LLM credentials in project.llm_config."
        )
    print("[VERBOSE] Pre-warmed management_analyst Crew Agent")
    print(f"[VERBOSE] Patterns to run: {list(PATTERN_FUNCTIONS.keys())}")
    print(f"[VERBOSE] Trials per pattern: {max(1, trials)}")
    
    # Always use thread-level parallelism for report processing
    num_report_workers = min(len(sample), int(os.getenv("ESG_REPORT_WORKERS", "4")))
    print(f"[PARALLEL] Using {num_report_workers} threads for report-level parallelism")

    chunk_store = ChunkStore()
    all_results: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern in PATTERN_FUNCTIONS}
    detailed: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern in PATTERN_FUNCTIONS}

    def _process_report(record, pattern_functions, chunk_store_ref, chunk_store_dir_ref, trials_count, report_index, total_reports):
        """Process a single report through all patterns."""
        from project.esg_framework_sustainalytics.retrieval_sustainalytics import ChunkStore
        
        # Create a local chunk store for this report to avoid thread conflicts
        local_chunk_store = ChunkStore()
        
        print(f"[VERBOSE] Processing report {report_index+1}/{total_reports}: {record.report_id}")
        
        report_pattern_results = {}
        report_pattern_detailed = {}
        
        for pattern_name, fn in pattern_functions.items():
            print(f"[VERBOSE]  Running pattern: {pattern_name}")
            trials_results: list[ReportRunResult] = []
            for trial in range(max(1, trials_count)):
                print(f"[VERBOSE]    Trial {trial+1}/{max(1, trials_count)}")
                result = fn(record, local_chunk_store)
                trials_results.append(result)

            consistency = consistency_metric([result.domain_scores for result in trials_results])
            representative = trials_results[0]
            representative.metrics["consistency_quantitative"] = consistency
            representative.metrics["consistency_qualitative"] = (
                "Stable" if consistency >= 0.85 else "Moderate variance" if consistency >= 0.7 else "High variance"
            )

            report_pattern_results[pattern_name] = representative.metrics
            report_pattern_detailed[pattern_name] = _serialize_report_result(representative)
            print(f"[VERBOSE]    Pattern {pattern_name} completed for report {record.report_id}")

            safe_id = _safe_path_fragment(record.report_id)
            chunk_file = Path(chunk_store_dir_ref) / f"report_{safe_id}_{pattern_name}.json"
            local_chunk_store.persist_json(record.report_id, chunk_file)
            print(f"[VERBOSE]    Chunk store saved to: {chunk_file}")
        
        return report_pattern_results, report_pattern_detailed

    if num_report_workers > 1:
        # Use thread pool for parallel report processing
        with ThreadPoolExecutor(max_workers=num_report_workers) as executor:
            futures = {
                executor.submit(
                    _process_report,
                    record,
                    PATTERN_FUNCTIONS,
                    chunk_store,
                    chunk_store_dir,
                    trials,
                    i,
                    len(sample)
                ): i for i, record in enumerate(sample)
            }
            
            for future in as_completed(futures):
                i = futures[future]
                pattern_results, pattern_detailed = future.result()
                
                # Aggregate results from this report
                for pattern_name in PATTERN_FUNCTIONS:
                    all_results[pattern_name].append(pattern_results[pattern_name])
                    detailed[pattern_name].append(pattern_detailed[pattern_name])
    else:
        # Fallback to sequential processing if only 1 worker
        for i, record in enumerate(sample):
            pattern_results, pattern_detailed = _process_report(
                record, PATTERN_FUNCTIONS, chunk_store, chunk_store_dir, trials, i, len(sample)
            )
            for pattern_name in PATTERN_FUNCTIONS:
                all_results[pattern_name].append(pattern_results[pattern_name])
                detailed[pattern_name].append(pattern_detailed[pattern_name])

    print(f"[VERBOSE] Aggregating metrics for all patterns...")
    summary = {pattern: aggregate_pattern_metrics(metrics) for pattern, metrics in all_results.items()}

    comparison_table = [
        {
            "pattern": pattern,
            **stats,
        }
        for pattern, stats in summary.items()
    ]

    print(f"[VERBOSE] Experiment complete! Summary:")
    for pattern, stats in summary.items():
        print(f"[VERBOSE]  {pattern}: {stats}")
    print(f"[VERBOSE] Saving results to: {output_path}")

    payload = {
        "dataset": str(dataset_path),
        "sample_size": len(sample),
        "trials": max(1, trials),
        "healthcare_pool_size": len(records),
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
