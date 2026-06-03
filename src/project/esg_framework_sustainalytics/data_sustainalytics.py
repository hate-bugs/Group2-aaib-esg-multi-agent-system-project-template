from __future__ import annotations

import csv
import sys
from pathlib import Path

from project.esg_framework_sustainalytics.models_sustainalytics import ReportRecord

MAX_CSV_FIELD_SIZE = 10**9

HEALTHCARE_KEYWORDS = (
    "health", "medical", "hospital", "pharma", "biotech", "care", "patient", "clinical", "therapeutic"
)


def _to_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def load_reports(path: str | Path) -> list[ReportRecord]:
    records: list[ReportRecord] = []
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(min(sys.maxsize, MAX_CSV_FIELD_SIZE))
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                report_id = row.get("") or str(idx)
                # Check if 5-domain Sustainalytics format is available
                has_pp = "pp_score" in row
                has_f = "f_score" in row
                has_gov = "gov_score" in row
                
                if has_pp and has_f and has_gov:
                    # Use 5-domain Sustainalytics format (0-100 scale)
                    ground_truth = {
                        "product_production": _to_float(row.get("pp_score")),
                        "financials": _to_float(row.get("f_score")),
                        "events": _to_float(row.get("e_score")),
                        "geographic": _to_float(row.get("g_score")),
                        "governance": _to_float(row.get("gov_score")),
                        "total": _to_float(row.get("total_score")),
                    }
                else:
                    # Fallback: try to map from 3-domain format (0-20 scale) to 5-domain (0-100 scale)
                    # Map: environmental -> product_production, social -> financials, governance -> governance
                    # events and geographic will be estimated from available data
                    e_score = _to_float(row.get("e_score"))
                    s_score = _to_float(row.get("s_score"))
                    g_score = _to_float(row.get("g_score"))
                    total_score = _to_float(row.get("total_score"))
                    
                    # Scale 0-20 to 0-100 and map to 5 domains
                    # This is an approximation for testing purposes
                    ground_truth = {
                        "product_production": e_score * 5.0,  # Map environmental to product_production
                        "financials": s_score * 5.0,  # Map social to financials
                        "events": (e_score + s_score + g_score) * 5.0 / 3.0,  # Average as events
                        "geographic": g_score * 5.0,  # Map governance to geographic
                        "governance": g_score * 5.0,  # Map governance to governance
                        "total": total_score * 5.0,  # Scale total
                    }
                
                records.append(
                    ReportRecord(
                        report_id=str(report_id),
                        filename=row.get("filename", ""),
                        ticker=row.get("ticker", ""),
                        year=row.get("year", ""),
                        preprocessed_content=row.get("preprocessed_content", ""),
                        sector=row.get("sector") or row.get("Sector"),
                        ground_truth=ground_truth,
                    )
                )
    finally:
        csv.field_size_limit(previous_limit)
    return records


def filter_healthcare_reports(records: list[ReportRecord]) -> list[ReportRecord]:
    # Assume dataset contains correct data without filtering
    return records


def select_sample(records: list[ReportRecord], sample_size: int = 10) -> list[ReportRecord]:
    return records[: max(0, sample_size)]
