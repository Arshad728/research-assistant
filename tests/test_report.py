"""Tests for report structure, citation numbering and validation (Phase 4.1, 4.3)."""
import pytest

from research_assistant.report.models import (
    Citation,
    ReportSection,
    ResearchReport,
    build_citations,
    convert_finding_markers,
)
from research_assistant.report.validate import (
    evidence_numbers,
    significant_numbers,
    validate_report,
)
from research_assistant.schemas import CandidateSource, VerifiedFinding

TRIAL_URL = "https://example.org/four-day-week-trial-2025"
SURVEY_URL = "https://example.org/workplace-survey-2025"

FINDINGS = [
    VerifiedFinding(
        claim="Output per hour rose by 8% while total weekly output was unchanged.",
        supporting_snippet="hourly productivity increased by 8%, while total weekly output "
        "remained statistically unchanged",
        source_url=TRIAL_URL,
    ),
    VerifiedFinding(
        claim="The trial covered 61 organisations across four countries.",
        supporting_snippet="Participating organisations numbered 61 in total, spread across "
        "four countries.",
        source_url=TRIAL_URL,
    ),
    VerifiedFinding(
        claim="Survey respondents reported higher satisfaction under the shorter schedule.",
        supporting_snippet="Respondents reported markedly higher satisfaction scores under the "
        "shorter working schedule than under their previous arrangements.",
        source_url=SURVEY_URL,
    ),
]

SOURCES = [
    CandidateSource(
        title="The Four-Day Week: Assessing Global Trials",
        source_url=TRIAL_URL,
        source_type="journal article",
        relevance_note="Controlled trial.",
    ),
    CandidateSource(
        title="Employee Outcomes Under Reduced-Hour Schedules",
        source_url=SURVEY_URL,
        source_type="web page",
        relevance_note="Survey data.",
    ),
]


def good_report() -> ResearchReport:
    citations, _ = build_citations(FINDINGS, SOURCES)
    return ResearchReport(
        question="What is the effect of a four-day work week on productivity?",
        introduction="This report draws on two sources covering a multi-site trial and a survey.",
        sections=[
            ReportSection(
                heading="Measured output",
                paragraphs=[
                    "Output per hour rose by 8% while total weekly output held level [1].",
                    "The trial covered 61 organisations across four countries [1].",
                ],
            ),
            ReportSection(
                heading="Self-reported experience",
                paragraphs=["Respondents reported higher satisfaction [2]."],
            ),
        ],
        conclusion="The evidence suggests output did not fall, though it is limited [1][2].",
        citations=citations,
    )


class TestBuildCitations:
    def test_findings_from_one_source_share_a_number(self):
        citations, mapping = build_citations(FINDINGS, SOURCES)
        assert len(citations) == 2
        assert mapping[1] == mapping[2] == 1
        assert mapping[3] == 2

    def test_citation_carries_the_source_title_and_type(self):
        citations, _ = build_citations(FINDINGS, SOURCES)
        assert citations[0].title == "The Four-Day Week: Assessing Global Trials"
        assert citations[0].source_type == "journal article"

    def test_works_without_source_metadata(self):
        citations, _ = build_citations(FINDINGS)
        assert citations[0].title == ""
        assert citations[0].source_url == TRIAL_URL


class TestConvertFindingMarkers:
    def test_rewrites_finding_numbers_as_citation_numbers(self):
        _, mapping = build_citations(FINDINGS, SOURCES)
        text, unknown = convert_finding_markers("Output rose [F1] and satisfaction rose [F3].", mapping)
        assert text == "Output rose [1] and satisfaction rose [2]."
        assert unknown == []

    def test_an_unknown_finding_marker_is_left_in_place_and_reported(self):
        _, mapping = build_citations(FINDINGS, SOURCES)
        text, unknown = convert_finding_markers("A claim from nowhere [F9].", mapping)
        assert "[F9]" in text
        assert unknown == [9]


class TestMarkdown:
    def test_includes_every_part_of_the_report(self):
        markdown = good_report().to_markdown()
        assert markdown.startswith("# What is the effect")
        assert "## Measured output" in markdown
        assert "## Conclusion" in markdown
        assert "## References" in markdown
        assert f"<{TRIAL_URL}>" in markdown

    def test_omits_sections_that_are_empty(self):
        markdown = ResearchReport(question="Q", introduction="Intro only.").to_markdown()
        assert "## References" not in markdown
        assert "## Conclusion" not in markdown

    def test_caveats_appear_under_verification_notes(self):
        report = good_report()
        report.caveats = ["unsupported_figures: the report states 99."]
        assert "## Verification notes" in report.to_markdown()
        assert "the report states 99" in report.to_markdown()


class TestSignificantNumbers:
    def test_ignores_small_counting_numbers(self):
        assert significant_numbers("three themes emerged across two studies") == set()

    def test_keeps_percentages_however_small(self):
        assert significant_numbers("output rose 8%") == {8.0}

    def test_keeps_larger_numbers_and_decimals(self):
        assert significant_numbers("61 sites over 1.5 years") == {61.0, 1.5}

    def test_evidence_pool_covers_claims_and_snippets(self):
        pool = evidence_numbers(FINDINGS)
        assert 8.0 in pool
        assert 61.0 in pool

    def test_the_pool_genuinely_reads_the_snippets_not_only_the_claims(self):
        # A mutation test found the snippet half of this could be deleted
        # without any test noticing, because every fixture happened to repeat
        # its figures in both places.
        from research_assistant.schemas import VerifiedFinding

        finding = VerifiedFinding(
            claim="Satisfaction improved under the shorter schedule.",
            supporting_snippet="Satisfaction scores rose by 37 points under the shorter "
            "working schedule compared with the previous arrangement.",
            source_url=SURVEY_URL,
        )
        assert 37.0 in evidence_numbers([finding])


class TestValidation:
    def test_a_sound_report_passes(self):
        result = validate_report(good_report(), FINDINGS)
        assert result.passed is True
        assert result.errors == []

    def test_a_dangling_citation_marker_is_an_error(self):
        report = good_report()
        report.sections[0].paragraphs[0] = "Output rose by 8% [7]."
        result = validate_report(report, FINDINGS)

        assert result.passed is False
        assert any(i.check == "dangling_citations" for i in result.errors)

    def test_an_unresolved_finding_marker_is_an_error(self):
        report = good_report()
        report.sections[0].paragraphs[0] = "Output rose by 8% [F9]."
        result = validate_report(report, FINDINGS)

        assert any(i.check == "unresolved_markers" for i in result.errors)

    def test_a_citation_to_a_source_with_no_findings_is_an_error(self):
        report = good_report()
        report.citations.append(
            Citation(number=3, source_url="https://example.net/never-verified", title="Ghost")
        )
        report.sections[1].paragraphs.append("An extra claim [3].")
        result = validate_report(report, FINDINGS)

        assert any(i.check == "ungrounded_citations" for i in result.errors)

    def test_a_section_with_no_citations_is_an_error(self):
        report = good_report()
        report.sections.append(
            ReportSection(heading="Background", paragraphs=["Some general context."])
        )
        result = validate_report(report, FINDINGS)

        assert any(i.check == "uncited_section" for i in result.errors)

    def test_a_figure_with_no_citation_is_an_error(self):
        report = good_report()
        report.sections[0].paragraphs.append("A further 45% reported improvements.")
        result = validate_report(report, FINDINGS)

        assert any(i.check == "uncited_figure" for i in result.errors)

    def test_a_figure_absent_from_the_evidence_is_an_error(self):
        report = good_report()
        report.sections[0].paragraphs[0] = "Output per hour rose by 23% [1]."
        result = validate_report(report, FINDINGS)

        assert result.passed is False
        issue = next(i for i in result.errors if i.check == "unsupported_figures")
        assert "23" in issue.detail

    def test_unused_evidence_is_only_a_warning(self):
        report = good_report()
        report.sections = report.sections[:1]  # drops the section citing source 2
        report.conclusion = "Output did not fall [1]."
        result = validate_report(report, FINDINGS)

        assert result.passed is True
        assert any(i.check == "unused_evidence" for i in result.warnings)

    def test_a_report_with_no_sections_is_an_error(self):
        result = validate_report(ResearchReport(question="Q"), FINDINGS)
        assert any(i.check == "empty_body" for i in result.errors)

    def test_feedback_names_each_problem_for_the_model(self):
        report = good_report()
        report.sections[0].paragraphs[0] = "Output per hour rose by 23% [7]."
        feedback = validate_report(report, FINDINGS).feedback_for_model()

        assert "23" in feedback
        assert "[7]" in feedback
        assert "Do not invent new evidence" in feedback

    def test_feedback_is_empty_when_nothing_is_wrong(self):
        assert validate_report(good_report(), FINDINGS).feedback_for_model() == ""


class TestPdfRendering:
    def test_writes_a_readable_pdf(self, tmp_path):
        from research_assistant.report.render_pdf import render_report_pdf

        report = good_report()
        report.caveats = ["unused_evidence: one source was never cited."]
        path = render_report_pdf(report, str(tmp_path / "report.pdf"))

        import pymupdf

        document = pymupdf.open(path)
        text = "\n".join(page.get_text("text") for page in document)
        page_count = document.page_count
        document.close()

        assert page_count >= 1
        assert "Measured output" in text
        assert "References" in text
        assert "Verification notes" in text
        assert TRIAL_URL in text

    def test_escapes_characters_that_would_break_the_markup(self, tmp_path):
        from research_assistant.report.render_pdf import render_report_pdf

        report = good_report()
        report.sections[0].paragraphs[0] = "Output rose <b>8%</b> & held level [1]."
        path = render_report_pdf(report, str(tmp_path / "escaped.pdf"))

        import pymupdf

        document = pymupdf.open(path)
        text = "\n".join(page.get_text("text") for page in document)
        document.close()

        assert "<b>" in text  # rendered literally, not interpreted as bold
        assert "&" in text
