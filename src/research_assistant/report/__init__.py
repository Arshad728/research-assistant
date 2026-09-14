"""Turning verified findings into a cited report (Phase 4)."""

from .models import (
    Citation,
    ReportSection,
    ResearchReport,
    build_citations,
    convert_finding_markers,
)
from .render_pdf import render_report_pdf
from .synthesis import (
    WRITER_SYSTEM_PROMPT,
    build_writer_prompt,
    format_findings,
    parse_report_reply,
)
from .validate import (
    ReportIssue,
    ReportValidation,
    evidence_numbers,
    significant_numbers,
    validate_report,
)

__all__ = [
    "Citation",
    "ReportIssue",
    "ReportSection",
    "ReportValidation",
    "ResearchReport",
    "WRITER_SYSTEM_PROMPT",
    "build_citations",
    "build_writer_prompt",
    "convert_finding_markers",
    "evidence_numbers",
    "format_findings",
    "parse_report_reply",
    "render_report_pdf",
    "significant_numbers",
    "validate_report",
]
