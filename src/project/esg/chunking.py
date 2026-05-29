from __future__ import annotations

import json
import re
from pathlib import Path

from project.esg.heuristics import all_domain_keywords
from project.esg.models import DOMAINS, ReportChunk, ReportRecord

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
DOMAIN_KEYWORDS = all_domain_keywords(DOMAINS)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _windowed(words: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    if chunk_size <= overlap:
        overlap = max(0, chunk_size // 4)
    step = max(1, chunk_size - overlap)
    return [words[start : start + chunk_size] for start in range(0, len(words), step) if words[start : start + chunk_size]]


def label_chunk(text: str) -> tuple[list[str], dict[str, float]]:
    tokens = tokenize(text)
    token_set = set(tokens)
    scores = {
        domain: round(len(token_set & keywords) / max(1, len(keywords)), 4)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best = max(scores.values(), default=0.0)
    if best == 0.0:
        labels = list(DOMAINS)
    else:
        labels = [domain for domain, score in scores.items() if score >= max(0.05, best * 0.6)]
    return labels, scores


def chunk_report(report: ReportRecord, chunk_size: int, overlap: int, chunk_dir: Path) -> list[ReportChunk]:
    words = report.preprocessed_content.split()
    windows = _windowed(words, chunk_size=chunk_size, overlap=overlap)
    chunks: list[ReportChunk] = []
    for index, window in enumerate(windows, start=1):
        text = " ".join(window)
        labels, label_scores = label_chunk(text)
        chunks.append(
            ReportChunk(
                chunk_id=f"{report.report_id}-chunk-{index}",
                report_id=report.report_id,
                text=text,
                token_count=len(window),
                labels=labels,
                label_scores=label_scores,
            )
        )

    chunk_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "report_id": chunk.report_id,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "labels": chunk.labels,
            "label_scores": chunk.label_scores,
        }
        for chunk in chunks
    ]
    (chunk_dir / f"{report.report_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return chunks
