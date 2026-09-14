"""Rendering a report to PDF (Phase 4.3).

Both output formats are produced from the ``ResearchReport`` object rather
than one being converted from the other, so the Markdown and the PDF can
never drift apart. It also means no Markdown parser is needed here: the
structure is already known, and only the parts of it that exist get drawn.

reportlab is used rather than a HTML-to-PDF converter because it is pure
Python and installs anywhere, which matters for a project someone else is
meant to be able to clone and run. Per reportlab's own limitations, the
markup below uses its ``<super>`` tag rather than Unicode superscript
characters, which its built-in fonts do not carry and would render as solid
black boxes.
"""
from __future__ import annotations

import html
import re
from typing import List

from .models import CITATION_MARKER, ResearchReport

_CITATION_IN_TEXT = re.compile(r"\[(\d+)\]")


def _escape(text: str) -> str:
    """Escape for reportlab's mini-markup, then restore citation superscripts."""
    escaped = html.escape(text, quote=False)
    return _CITATION_IN_TEXT.sub(lambda m: f"<super>[{m.group(1)}]</super>", escaped)


def render_report_pdf(report: ResearchReport, path: str) -> str:
    """Write the report to ``path`` as a PDF and return the path."""
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=base["Title"], fontSize=18, leading=23, spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=base["Heading2"], fontSize=13, leading=16,
        spaceBefore=16, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=base["BodyText"], fontSize=10.5, leading=15.5,
        alignment=TA_JUSTIFY, spaceAfter=9,
    )
    reference_style = ParagraphStyle(
        "Reference", parent=body_style, fontSize=9.5, leading=13, alignment=0, spaceAfter=4,
    )
    caveat_style = ParagraphStyle(
        "Caveat", parent=reference_style, textColor="#8a3a00",
    )

    story: List = [Paragraph(_escape(report.question), title_style)]

    if report.introduction:
        story.append(Paragraph(_escape(report.introduction), body_style))

    for section in report.sections:
        story.append(Paragraph(_escape(section.heading), heading_style))
        for paragraph in section.paragraphs:
            story.append(Paragraph(_escape(paragraph), body_style))

    if report.conclusion:
        story.append(Paragraph("Conclusion", heading_style))
        story.append(Paragraph(_escape(report.conclusion), body_style))

    if report.citations:
        story.append(Paragraph("References", heading_style))
        for citation in report.citations:
            label = html.escape(citation.title or citation.source_url, quote=False)
            kind = f" ({html.escape(citation.source_type, quote=False)})" if citation.source_type else ""
            url = html.escape(citation.source_url, quote=False)
            story.append(
                Paragraph(
                    f"[{citation.number}] {label}{kind}.<br/>"
                    f'<font color="#2F5496"><link href="{url}">{url}</link></font>',
                    reference_style,
                )
            )

    if report.caveats:
        story.append(Paragraph("Verification notes", heading_style))
        story.append(
            ListFlowable(
                [ListItem(Paragraph(_escape(caveat), caveat_style)) for caveat in report.caveats],
                bulletType="bullet",
                start="-",
                leftIndent=14,
            )
        )

    document = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title=report.question[:120],
        author="Multi-Agent Research Assistant",
    )
    document.build(story)
    return path
