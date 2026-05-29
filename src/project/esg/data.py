from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
REPORT_LINK_FIELDS = ("report_link", "report link", "report url", "report", "pdf_url", "url")
ACTUAL_SCORE_FIELDS = {
    "environmental": ("environmental_score", "e_score", "env_score"),
    "social": ("social_score", "s_score"),
    "governance": ("governance_score", "g_score", "gov_score"),
    "total": ("esg_score", "total_score", "score"),
}


def _is_remote_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"}


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


def _extract_text_from_pdf_url(url: str, *, timeout: int = 60) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("pypdf is required to extract report text from PDF links") from exc

    request = Request(url=url, headers={"User-Agent": "Mozilla/5.0 ESG-RAG-Agent"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except URLError as exc:
        raise RuntimeError(f"Failed to download PDF report: {url}") from exc

    reader = PdfReader(io.BytesIO(payload))
    pages = [page.extract_text() or "" for page in reader.pages]
    return " ".join(page.strip() for page in pages if page.strip())


def load_reports(
    dataset_path: Path,
    sample_size: int,
    *,
    pdf_extractor: Callable[[str], str] | None = None,
) -> list[ReportRecord]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Place sustainability-reports-2026-05-29.csv there or pass --dataset-path."
        )

    extractor = pdf_extractor or _extract_text_from_pdf_url
    pdf_text_cache: dict[str, str] = {}

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Dataset is missing header row")

        content_field = _resolve_field(reader.fieldnames, CONTENT_FIELDS)
        report_link_field = _resolve_field(reader.fieldnames, REPORT_LINK_FIELDS)
        if not content_field and not report_link_field:
            raise ValueError(
                "Dataset must include either a preprocessed content column (e.g. preprocessed_content) or a report PDF link column (e.g. Report Link)"
            )

        report_id_field = _resolve_field(reader.fieldnames, REPORT_ID_FIELDS)
        company_name_field = _resolve_field(reader.fieldnames, NAME_FIELDS)
        score_fields = {
            key: _resolve_field(reader.fieldnames, candidates)
            for key, candidates in ACTUAL_SCORE_FIELDS.items()
        }

        reports: list[ReportRecord] = []
        for index, row in enumerate(reader, start=1):
            content = (row.get(content_field) or "").strip() if content_field else ""
            report_link = (row.get(report_link_field) or "").strip() if report_link_field else ""
            if not content and report_link and _is_remote_url(report_link):
                if report_link not in pdf_text_cache:
                    try:
                        pdf_text_cache[report_link] = extractor(report_link).strip()
                    except Exception:
                        pdf_text_cache[report_link] = ""
                content = pdf_text_cache[report_link]
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
