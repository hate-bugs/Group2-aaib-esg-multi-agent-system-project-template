from __future__ import annotations

import csv
import sys
from pathlib import Path

from project.esg_framework.models import ReportRecord

MAX_CSV_FIELD_SIZE = 10**9  # Supports unusually large preprocessed report text columns without csv field truncation.

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
    # Reports may contain very large preprocessed text fields; temporarily raise CSV field limit while loading.
    csv.field_size_limit(min(sys.maxsize, MAX_CSV_FIELD_SIZE))
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                report_id = row.get("") or str(idx)
                records.append(
                    ReportRecord(
                        report_id=str(report_id),
                        filename=row.get("filename", ""),
                        ticker=row.get("ticker", ""),
                        year=row.get("year", ""),
                        preprocessed_content=row.get("preprocessed_content", ""),
                        sector=row.get("sector") or row.get("Sector"),
                        ground_truth={
                            "environmental": _to_float(row.get("e_score")),
                            "social": _to_float(row.get("s_score")),
                            "governance": _to_float(row.get("g_score")),
                            "total": _to_float(row.get("total_score")),
                        },
                    )
                )
    finally:
        csv.field_size_limit(previous_limit)
    return records


def filter_healthcare_reports(records: list[ReportRecord]) -> list[ReportRecord]:
    exact = [
        item
        for item in records
        if item.sector and "health" in item.sector.lower()
    ]
    if exact:
        return exact

    keyword_based: list[ReportRecord] = []
    for item in records:
        content = f"{item.filename} {item.preprocessed_content[:2000]}".lower()
        if any(keyword in content for keyword in HEALTHCARE_KEYWORDS):
            keyword_based.append(item)
    if keyword_based:
        return keyword_based

    return records


def select_sample(records: list[ReportRecord], sample_size: int = 10) -> list[ReportRecord]:
    return records[: max(0, sample_size)]
