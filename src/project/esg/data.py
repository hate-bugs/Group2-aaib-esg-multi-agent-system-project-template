from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from project.esg.models import ReportRecord

CONTENT_FIELDS = (
    "preprocessed_content",
    "processed_content",
    "content",
    "report_text",
    "text",
)
REPORT_ID_FIELDS = ("report_id", "id", "company_id", "isin")
NAME_FIELDS = ("company_name", "company", "issuer_name", "name")
ACTUAL_SCORE_FIELDS = {
    "environmental": ("environmental_score", "e_score", "env_score"),
    "social": ("social_score", "s_score"),
    "governance": ("governance_score", "g_score", "gov_score"),
    "total": ("esg_score", "total_score", "score"),
}


def _resolve_field(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_reports(dataset_path: Path, sample_size: int) -> list[ReportRecord]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Place sustainability-reports-2026-05-29.csv there or pass --dataset-path."
        )

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Dataset is missing header row")

        content_field = _resolve_field(reader.fieldnames, CONTENT_FIELDS)
        if not content_field:
            raise ValueError(
                "Dataset must include a preprocessed report text column such as preprocessed_content"
            )

        report_id_field = _resolve_field(reader.fieldnames, REPORT_ID_FIELDS)
        company_name_field = _resolve_field(reader.fieldnames, NAME_FIELDS)
        score_fields = {
            key: _resolve_field(reader.fieldnames, candidates)
            for key, candidates in ACTUAL_SCORE_FIELDS.items()
        }

        reports: list[ReportRecord] = []
        for index, row in enumerate(reader, start=1):
            content = (row.get(content_field) or "").strip()
            if not content:
                continue
            report_id = (row.get(report_id_field) or f"report-{index}").strip() if report_id_field else f"report-{index}"
            company_name = (row.get(company_name_field) or report_id).strip() if company_name_field else report_id
            actual_scores = {
                domain: _parse_float(row.get(field)) if field else None
                for domain, field in score_fields.items()
            }
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {content_field, report_id_field, company_name_field, *[field for field in score_fields.values() if field]}
            }
            reports.append(
                ReportRecord(
                    report_id=report_id,
                    company_name=company_name,
                    preprocessed_content=content,
                    actual_scores=actual_scores,
                    metadata=metadata,
                )
            )
            if len(reports) >= sample_size:
                break

    if not reports:
        raise ValueError("No usable reports were loaded from the dataset")
    return reports
