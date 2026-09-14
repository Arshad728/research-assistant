"""Tests for the Extraction and Verification Agent (Phases 3.2 and 3.3).

The model call is injected, so these run the real extraction-and-verification
path end to end with a scripted model reply -- including replies a careless
or overconfident model would actually produce. That is more useful than a
live run here, because it lets each failure mode be triggered on purpose.
"""
import json

import httpx
import pytest
import respx

from research_assistant.agents.extraction_agent import ExtractionAgent, summarise_reports
from research_assistant.extraction.documents import SourceDocument
from research_assistant.extraction.extract import parse_extraction_reply
from research_assistant.schemas import CandidateSource

from .test_parsers import build_pdf
from .test_verify import GOOD_SNIPPET, SOURCE_TEXT

QUESTION = "What is the effect of a four-day work week on productivity?"

DOCUMENT = SourceDocument(
    source_url="https://example.org/four-day-week-trial-2025",
    title="The Four-Day Week: Assessing Global Trials",
    text=SOURCE_TEXT,
    content_kind="pdf",
    page_count=12,
)


def scripted(reply):
    """Build a completion function that always returns one scripted reply."""
    text = reply if isinstance(reply, str) else json.dumps(reply)

    async def _complete(system_prompt: str, user_prompt: str) -> str:
        _complete.calls.append((system_prompt, user_prompt))
        return text

    _complete.calls = []
    return _complete


class TestParseExtractionReply:
    def test_reads_a_plain_array(self):
        parsed = parse_extraction_reply('[{"claim": "c", "supporting_snippet": "s"}]')
        assert parsed == [{"claim": "c", "supporting_snippet": "s"}]

    def test_accepts_alternative_key_names(self):
        parsed = parse_extraction_reply('[{"claim": "c", "quote": "s"}]')
        assert parsed == [{"claim": "c", "supporting_snippet": "s"}]

    def test_accepts_a_wrapped_object(self):
        parsed = parse_extraction_reply('{"claims": [{"claim": "c", "supporting_snippet": "s"}]}')
        assert len(parsed) == 1

    def test_an_empty_array_is_not_the_same_as_an_unreadable_reply(self):
        assert parse_extraction_reply("[]") == []
        assert parse_extraction_reply("I could not do that.") is None

    def test_entries_missing_a_snippet_are_dropped(self):
        parsed = parse_extraction_reply('[{"claim": "c"}, {"claim": "d", "supporting_snippet": "s"}]')
        assert parsed == [{"claim": "d", "supporting_snippet": "s"}]


class TestExtraction:
    async def test_a_well_behaved_reply_produces_verified_findings(self):
        agent = ExtractionAgent(complete=scripted([
            {
                "claim": "Output per hour rose by 8% while total weekly output was unchanged.",
                "supporting_snippet": GOOD_SNIPPET,
            }
        ]))

        findings, report = await agent.extract_from_document(DOCUMENT, QUESTION)

        assert report.status == "ok"
        assert report.candidates_proposed == 1
        assert report.findings_verified == 1
        assert report.rejected == []
        assert findings[0].source_url == DOCUMENT.source_url
        assert findings[0].verified is True

    async def test_a_fabricated_quotation_is_rejected_and_reported(self):
        agent = ExtractionAgent(complete=scripted([
            {
                "claim": "The four-day week doubled output at every organisation.",
                "supporting_snippet": "Researchers concluded that output doubled at every "
                "participating organisation without exception during the trial.",
            }
        ]))

        findings, report = await agent.extract_from_document(DOCUMENT, QUESTION)

        assert findings == []
        assert report.status == "no_claims"
        assert len(report.rejected) == 1
        assert "snippet_grounded" in report.rejected[0].failed_checks
        assert report.rejection_rate == 1.0

    async def test_a_real_quotation_with_an_invented_statistic_is_rejected(self):
        agent = ExtractionAgent(complete=scripted([
            {
                "claim": "Output per hour rose by 8% across 61 organisations.",
                "supporting_snippet": GOOD_SNIPPET,
            }
        ]))

        _, report = await agent.extract_from_document(DOCUMENT, QUESTION)

        assert report.findings_verified == 0
        assert "numeric_support" in report.rejected[0].failed_checks
        assert "61" in report.rejected[0].detail

    async def test_good_and_bad_claims_in_one_reply_are_separated(self):
        agent = ExtractionAgent(complete=scripted([
            {
                "claim": "Output per hour rose by 8% while total weekly output was unchanged.",
                "supporting_snippet": GOOD_SNIPPET,
            },
            {
                "claim": "Every organisation reported higher morale.",
                "supporting_snippet": "All participating organisations reported substantially "
                "higher morale scores at the end of the trial period.",
            },
        ]))

        findings, report = await agent.extract_from_document(DOCUMENT, QUESTION)

        assert report.candidates_proposed == 2
        assert report.findings_verified == 1
        assert len(report.rejected) == 1
        assert len(findings) == 1
        assert report.status == "ok"

    async def test_a_source_with_nothing_relevant_says_so_without_stretching(self):
        agent = ExtractionAgent(complete=scripted([]))

        findings, report = await agent.extract_from_document(DOCUMENT, QUESTION)

        assert findings == []
        assert report.status == "no_claims"
        assert "does not address" in report.detail
        assert report.rejected == []

    async def test_an_unreadable_reply_is_distinguished_from_an_empty_one(self):
        agent = ExtractionAgent(complete=scripted("Sorry, I was unable to comply."))

        _, report = await agent.extract_from_document(DOCUMENT, QUESTION)

        assert report.status == "no_claims"
        assert "could not be read" in report.detail

    async def test_a_document_with_almost_no_text_is_not_sent_to_the_model(self):
        scanned = SourceDocument(
            source_url="https://example.org/scan", text="Figure 1.", content_kind="pdf"
        )
        complete = scripted([])
        agent = ExtractionAgent(complete=complete)

        _, report = await agent.extract_from_document(scanned, QUESTION)

        assert report.status == "unreadable"
        assert "scanned image" in report.detail
        assert complete.calls == []  # no pointless model call

    async def test_the_prompt_carries_the_question_and_the_source_text(self):
        complete = scripted([])
        await ExtractionAgent(complete=complete).extract_from_document(DOCUMENT, QUESTION)

        system_prompt, user_prompt = complete.calls[0]
        assert "Extraction and Verification Agent" in system_prompt
        assert QUESTION in user_prompt
        assert "hourly productivity increased by 8%" in user_prompt

    async def test_long_documents_are_truncated_for_the_model_but_not_for_verification(self):
        padding = "Filler sentence about unrelated topics. " * 400
        long_document = SourceDocument(
            source_url="https://example.org/long",
            text=padding + SOURCE_TEXT,
            content_kind="pdf",
        )
        complete = scripted([
            {
                "claim": "Output per hour rose by 8% while total weekly output was unchanged.",
                "supporting_snippet": GOOD_SNIPPET,
            }
        ])
        # A limit small enough that the evidence sits beyond it.
        agent = ExtractionAgent(complete=complete, text_limit=1000)

        findings, report = await agent.extract_from_document(long_document, QUESTION)

        assert report.text_truncated_for_model is True
        assert len(complete.calls[0][1]) < len(long_document.text)
        # Verification searched the whole document, so the claim still passes.
        assert report.findings_verified == 1
        assert findings[0].verified is True


class TestFetchingSources:
    @respx.mock
    async def test_a_source_that_cannot_be_fetched_is_reported_not_raised(self):
        respx.get("https://example.org/gone").mock(return_value=httpx.Response(404))
        source = CandidateSource(
            title="Missing",
            source_url="https://example.org/gone",
            source_type="web page",
            relevance_note="Looked relevant.",
        )

        findings, report = await ExtractionAgent(complete=scripted([])).extract_from_source(
            source, QUESTION
        )

        assert findings == []
        assert report.status == "fetch_failed"
        assert "404" in report.detail

    @respx.mock
    async def test_a_batch_keeps_going_after_one_source_fails(self):
        data = build_pdf([SOURCE_TEXT])
        respx.get("https://example.org/broken").mock(return_value=httpx.Response(500))
        respx.get("https://example.org/good").mock(
            return_value=httpx.Response(
                200, content=data, headers={"content-type": "application/pdf"}
            )
        )

        sources = [
            CandidateSource(
                title="Broken",
                source_url="https://example.org/broken",
                source_type="web page",
                relevance_note="n",
            ),
            CandidateSource(
                title="Good",
                source_url="https://example.org/good",
                source_type="preprint",
                relevance_note="n",
            ),
        ]
        agent = ExtractionAgent(complete=scripted([
            {
                "claim": "Output per hour rose by 8% while total weekly output was unchanged.",
                "supporting_snippet": GOOD_SNIPPET,
            }
        ]))

        findings, reports = await agent.extract_from_sources(sources, QUESTION)

        assert len(reports) == 2
        assert reports[0].status == "fetch_failed"
        assert reports[1].findings_verified == 1
        assert len(findings) == 1


class TestSummary:
    def test_summarises_a_mixed_batch(self):
        from research_assistant.extraction.extract import ExtractionReport, RejectedClaim

        reports = [
            ExtractionReport(source_url="https://a.org", candidates_proposed=3, findings_verified=2,
                             rejected=[RejectedClaim(claim="x", failed_checks=["numeric_support"])]),
            ExtractionReport(source_url="https://b.org", status="fetch_failed", detail="404"),
        ]
        summary = summarise_reports(reports)

        assert "2 source(s) processed" in summary
        assert "1 produced usable findings" in summary
        assert "2 of 3 proposed claim(s) verified" in summary
        assert "1 source(s) could not be read" in summary

    def test_handles_an_empty_batch(self):
        assert summarise_reports([]) == "No sources were processed."
