"""Tests for the Writer Agent (Phase 4.2), including the revision loop.

The model is scripted, so a draft that invents a statistic can be produced
on demand and the system's response to it observed. That is the behaviour
worth pinning down: not that the writer writes well, but that a writer
writing badly cannot get a bad report past the checks unnoticed.
"""
import json

import pytest

from research_assistant.agents.writer_agent import WriterAgent, save_report
from research_assistant.report.synthesis import build_writer_prompt, format_findings, parse_report_reply

from .test_report import FINDINGS, SOURCES, SURVEY_URL, TRIAL_URL

QUESTION = "What is the effect of a four-day work week on productivity?"

GOOD_DRAFT = {
    "introduction": "This report draws on a multi-site trial and a survey of participants.",
    "sections": [
        {
            "heading": "Measured output",
            "paragraphs": [
                "Output per hour rose by 8% while total weekly output held level [F1].",
                "The trial covered 61 organisations across four countries [F2].",
            ],
        },
        {
            "heading": "Self-reported experience",
            "paragraphs": ["Participants reported higher satisfaction [F3]."],
        },
    ],
    "conclusion": "Output did not fall, though the evidence base is small [F1][F3].",
}

DRAFT_WITH_INVENTED_FIGURE = {
    "introduction": "This report draws on a multi-site trial and a survey.",
    "sections": [
        {
            "heading": "Measured output",
            "paragraphs": ["Output per hour rose by 23% across the trial [F1]."],
        },
        {
            "heading": "Self-reported experience",
            "paragraphs": ["Participants reported higher satisfaction [F3]."],
        },
    ],
    "conclusion": "The shorter week improved output [F1].",
}


def scripted(*replies):
    """A completion function that returns each scripted reply in turn."""
    payloads = [r if isinstance(r, str) else json.dumps(r) for r in replies]

    async def _complete(system_prompt: str, user_prompt: str) -> str:
        _complete.calls.append((system_prompt, user_prompt))
        index = min(len(_complete.calls) - 1, len(payloads) - 1)
        return payloads[index]

    _complete.calls = []
    return _complete


class TestPrompt:
    def test_findings_are_numbered_for_citation(self):
        rendered = format_findings(FINDINGS)
        assert "[F1]" in rendered and "[F3]" in rendered

    def test_the_writer_never_sees_a_url(self):
        prompt = build_writer_prompt(QUESTION, FINDINGS)
        assert TRIAL_URL not in prompt
        assert SURVEY_URL not in prompt

    def test_the_prompt_carries_the_question_and_the_evidence(self):
        prompt = build_writer_prompt(QUESTION, FINDINGS)
        assert QUESTION in prompt
        assert "hourly productivity increased by 8%" in prompt


class TestParseReply:
    def test_reads_a_well_formed_draft(self):
        parsed = parse_report_reply(json.dumps(GOOD_DRAFT))
        assert len(parsed["sections"]) == 2
        assert parsed["conclusion"].startswith("Output did not fall")

    def test_accepts_a_single_paragraph_given_as_a_string(self):
        parsed = parse_report_reply(
            json.dumps({"introduction": "i", "sections": [{"heading": "H", "paragraphs": "one"}],
                        "conclusion": "c"})
        )
        assert parsed["sections"][0]["paragraphs"] == ["one"]

    def test_drops_malformed_sections(self):
        parsed = parse_report_reply(
            json.dumps({"sections": [{"heading": "", "paragraphs": ["x"]}, {"heading": "OK",
                                                                           "paragraphs": ["y"]}]})
        )
        assert [s["heading"] for s in parsed["sections"]] == ["OK"]

    def test_unreadable_reply_is_none(self):
        assert parse_report_reply("I cannot do that.") is None


class TestWriting:
    async def test_a_sound_draft_passes_first_time(self):
        agent = WriterAgent(complete=scripted(GOOD_DRAFT))
        outcome = await agent.write_report(QUESTION, FINDINGS, SOURCES)

        assert outcome.passed_validation is True
        assert outcome.revisions_used == 0
        assert outcome.report.caveats == []

    async def test_finding_markers_become_citation_numbers(self):
        outcome = await WriterAgent(complete=scripted(GOOD_DRAFT)).write_report(
            QUESTION, FINDINGS, SOURCES
        )
        body = outcome.report.full_text()

        assert "[F1]" not in body
        assert "[1]" in body and "[2]" in body
        # F1 and F2 share a source, so they share citation number 1.
        assert outcome.report.cited_numbers() == [1, 2]

    async def test_the_reference_list_carries_real_titles_and_urls(self):
        outcome = await WriterAgent(complete=scripted(GOOD_DRAFT)).write_report(
            QUESTION, FINDINGS, SOURCES
        )
        first = outcome.report.citations[0]

        assert first.title == "The Four-Day Week: Assessing Global Trials"
        assert first.source_url == TRIAL_URL

    async def test_an_invented_figure_triggers_a_revision(self):
        complete = scripted(DRAFT_WITH_INVENTED_FIGURE, GOOD_DRAFT)
        outcome = await WriterAgent(complete=complete, max_revisions=1).write_report(
            QUESTION, FINDINGS, SOURCES
        )

        assert outcome.revisions_used == 1
        assert outcome.passed_validation is True
        # The revision prompt told the model precisely what was wrong.
        second_prompt = complete.calls[1][1]
        assert "23" in second_prompt
        assert "Do not invent new evidence" in second_prompt

    async def test_a_writer_that_keeps_inventing_is_not_allowed_to_hide_it(self):
        agent = WriterAgent(
            complete=scripted(DRAFT_WITH_INVENTED_FIGURE, DRAFT_WITH_INVENTED_FIGURE),
            max_revisions=1,
        )
        outcome = await agent.write_report(QUESTION, FINDINGS, SOURCES)

        assert outcome.passed_validation is False
        assert outcome.revisions_used == 1
        # The unresolved problem is printed on the report itself.
        assert any("23" in caveat for caveat in outcome.report.caveats)
        assert "Verification notes" in outcome.report.to_markdown()

    async def test_an_unreadable_reply_is_retried_then_reported(self):
        agent = WriterAgent(complete=scripted("nonsense", "still nonsense"), max_revisions=1)
        outcome = await agent.write_report(QUESTION, FINDINGS, SOURCES)

        assert outcome.draft_failed is True
        assert outcome.revisions_used == 1
        assert "could not be read" in " ".join(outcome.report.caveats)

    async def test_an_unreadable_first_reply_can_still_recover(self):
        agent = WriterAgent(complete=scripted("nonsense", GOOD_DRAFT), max_revisions=1)
        outcome = await agent.write_report(QUESTION, FINDINGS, SOURCES)

        assert outcome.draft_failed is False
        assert outcome.passed_validation is True

    async def test_no_findings_produces_an_honest_empty_report_not_an_essay(self):
        complete = scripted(GOOD_DRAFT)
        outcome = await WriterAgent(complete=complete).write_report(QUESTION, [], SOURCES)

        assert outcome.draft_failed is True
        assert outcome.report.sections == []
        assert "No verified findings" in outcome.report.introduction
        assert complete.calls == []  # the model is never asked to write from nothing

    async def test_references_never_cited_are_dropped_from_the_list(self):
        draft = json.loads(json.dumps(GOOD_DRAFT))
        draft["sections"] = draft["sections"][:1]
        draft["conclusion"] = "Output did not fall [F1]."
        outcome = await WriterAgent(complete=scripted(draft)).write_report(
            QUESTION, FINDINGS, SOURCES
        )

        assert [c.number for c in outcome.report.citations] == [1]
        assert any(i.check == "unused_evidence" for i in outcome.validation.warnings)


class TestSaving:
    async def test_saves_markdown_and_pdf(self, tmp_path):
        outcome = await WriterAgent(complete=scripted(GOOD_DRAFT)).write_report(
            QUESTION, FINDINGS, SOURCES
        )
        markdown_path, pdf_path = save_report(outcome.report, str(tmp_path / "report"))

        assert markdown_path.endswith(".md")
        assert pdf_path.endswith(".pdf")

        with open(markdown_path, encoding="utf-8") as handle:
            markdown = handle.read()
        assert "## Measured output" in markdown
        assert TRIAL_URL in markdown

        import pymupdf

        document = pymupdf.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in document)
        document.close()
        assert "Measured output" in text
