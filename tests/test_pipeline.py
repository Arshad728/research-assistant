"""End-to-end tests for the Orchestrator (Phases 5.1 and 5.3).

These run the real pipeline: real orchestration, real extraction, real
verification, real report assembly and validation. Only two things are
stood in for -- the Search Agent, which needs a live model and the open
internet, and the model calls inside extraction and writing, which are
scripted.

The scripted models are written to behave like real ones rather than like
oracles. The extraction stand-in quotes the document it was given, by
pulling a sentence out of the prompt, so verification has something genuine
to check. A second stand-in fabricates instead, which is how the rejection
path gets exercised end to end.
"""
import json
import re
from typing import Dict, List

import httpx
import pytest
import respx

from research_assistant.agents.extraction_agent import ExtractionAgent
from research_assistant.agents.writer_agent import WriterAgent
from research_assistant.orchestration import ResearchPipeline, StoppingRule
from research_assistant.schemas import CandidateSource

QUESTION = "What is the effect of a four-day work week on productivity?"

PAGES: Dict[str, str] = {
    "https://example.org/trial": """<html><head><title>Multi-Site Trial</title></head><body>
      <article>
        <p>Across the trial period, hourly productivity increased by 8%, while total weekly
        output remained statistically unchanged across the participating organisations.</p>
        <p>Participating organisations numbered 61 in total, spread across four countries and
        covering sectors from software to manufacturing and professional services.</p>
      </article></body></html>""",
    "https://example.org/survey": """<html><head><title>Participant Survey</title></head><body>
      <article>
        <p>Respondents reported markedly higher satisfaction scores under the shorter working
        schedule than under their previous arrangements, with most citing reduced fatigue.</p>
        <p>The survey covered the same programme as the trial and relies entirely on
        self-reported measures rather than on observed output.</p>
      </article></body></html>""",
    "https://example.org/critique": """<html><head><title>A Critical Read</title></head><body>
      <article>
        <p>The absence of a control group means the observed changes cannot be attributed to
        the schedule change with confidence, a limitation common to workplace trials.</p>
        <p>Several participating organisations had already begun other efficiency programmes
        before the trial started, which complicates attribution further.</p>
      </article></body></html>""",
}

SOURCES = [
    CandidateSource(
        title="Multi-Site Trial",
        source_url="https://example.org/trial",
        source_type="journal article",
        relevance_note="Measures output before and after.",
    ),
    CandidateSource(
        title="Participant Survey",
        source_url="https://example.org/survey",
        source_type="web page",
        relevance_note="Self-reported outcomes.",
    ),
    CandidateSource(
        title="A Critical Read",
        source_url="https://example.org/critique",
        source_type="web page",
        relevance_note="Notes methodological limits.",
    ),
]


class StubSearchAgent:
    """Returns scripted sources, and records which angles it was asked for."""

    def __init__(self, by_round: List[List[CandidateSource]]):
        self.by_round = by_round
        self.angles: List[str] = []

    async def find_sources(self, query: str):
        from research_assistant.agents.search_agent import SearchOutcome

        index = min(len(self.angles), len(self.by_round) - 1)
        self.angles.append(query)
        return SearchOutcome(query=query, sources=list(self.by_round[index]))


def honest_extractor(system_prompt: str, user_prompt: str) -> str:
    """A stand-in extractor that quotes the source it was actually given."""
    body = user_prompt.split("<<<SOURCE_START>>>")[-1].split("<<<SOURCE_END>>>")[0]
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", body) if len(s.strip()) > 80]
    claims = [
        {"claim": f"The source reports: {sentence[:90].rstrip('.')}.", "supporting_snippet": sentence}
        for sentence in sentences[:2]
    ]
    return json.dumps(claims)


def fabricating_extractor(system_prompt: str, user_prompt: str) -> str:
    """A stand-in extractor that invents its evidence."""
    return json.dumps(
        [
            {
                "claim": "The four-day week doubled output at every participating organisation.",
                "supporting_snippet": "Researchers concluded that output doubled at every "
                "participating organisation without exception during the trial period.",
            }
        ]
    )


def compliant_writer(system_prompt: str, user_prompt: str) -> str:
    """A stand-in writer that cites every finding and adds nothing of its own."""
    findings = re.findall(r"\[F(\d+)\]\n  claim: (.+)", user_prompt)
    if not findings:
        return json.dumps({"introduction": "No findings.", "sections": [], "conclusion": ""})

    paragraphs = [f"{claim.strip()} [F{number}]" for number, claim in findings]
    half = max(1, len(paragraphs) // 2)
    return json.dumps(
        {
            "introduction": "This report summarises the verified evidence gathered.",
            "sections": [
                {"heading": "What the evidence shows", "paragraphs": paragraphs[:half]},
                {"heading": "Further detail", "paragraphs": paragraphs[half:]},
            ]
            if len(paragraphs) > 1
            else [{"heading": "What the evidence shows", "paragraphs": paragraphs}],
            "conclusion": f"The evidence rests on {len(findings)} verified finding(s) "
            f"[F{findings[0][0]}].",
        }
    )


def planner_reply(system_prompt: str, user_prompt: str) -> str:
    return json.dumps(
        ["four day week productivity trial", "reduced hours output evidence", "four day week criticism"]
    )


def make_completion(*, extractor, writer, planner=planner_reply):
    """Route each model call to the right stand-in by its system prompt."""

    async def _complete(system_prompt: str, user_prompt: str) -> str:
        if "Orchestrator" in system_prompt:
            return planner(system_prompt, user_prompt)
        if "Extraction and Verification Agent" in system_prompt:
            return extractor(system_prompt, user_prompt)
        return writer(system_prompt, user_prompt)

    return _complete


@pytest.fixture
def served_pages():
    with respx.mock:
        for url, html in PAGES.items():
            respx.get(url).mock(
                return_value=httpx.Response(
                    200, text=html, headers={"content-type": "text/html; charset=utf-8"}
                )
            )
        yield


def build_pipeline(*, search, extractor=honest_extractor, writer=compliant_writer, rule=None):
    complete = make_completion(extractor=extractor, writer=writer)
    return ResearchPipeline(
        search_agent=search,
        extraction_agent=ExtractionAgent(complete=complete),
        writer_agent=WriterAgent(complete=complete, max_revisions=1),
        stopping_rule=rule or StoppingRule(),
        complete=complete,
    )


class TestPlanning:
    async def test_the_model_plan_is_used_when_it_is_usable(self, served_pages):
        search = StubSearchAgent([SOURCES])
        run = await build_pipeline(search=search).run(QUESTION)

        assert run.used_fallback_plan is False
        assert len(run.angles_planned) == 3
        assert search.angles[0] == "four day week productivity trial"

    async def test_an_unusable_plan_falls_back_without_failing_the_run(self, served_pages):
        def broken_planner(system_prompt, user_prompt):
            return "I am not going to answer in JSON."

        complete = make_completion(
            extractor=honest_extractor, writer=compliant_writer, planner=broken_planner
        )
        search = StubSearchAgent([SOURCES])
        pipeline = ResearchPipeline(
            search_agent=search,
            extraction_agent=ExtractionAgent(complete=complete),
            writer_agent=WriterAgent(complete=complete),
            complete=complete,
        )
        run = await pipeline.run(QUESTION)

        assert run.used_fallback_plan is True
        assert run.angles_planned[0] == QUESTION
        assert run.succeeded is True

    async def test_duplicate_angles_are_collapsed(self):
        from research_assistant.orchestration.planning import parse_planning_reply

        angles = parse_planning_reply(
            json.dumps(["four day week productivity", "productivity four-day week", "cost impact"])
        )
        assert len(angles) == 2


class TestEndToEnd:
    async def test_one_question_produces_a_complete_cited_report(self, served_pages):
        run = await build_pipeline(search=StubSearchAgent([SOURCES])).run(QUESTION)

        assert run.succeeded is True
        assert run.findings, "no findings survived verification"
        assert run.report is not None
        assert run.report.sections
        assert run.report.citations
        # Every citation points at a source that really produced a finding.
        finding_urls = {f.source_url for f in run.findings}
        assert all(c.source_url in finding_urls for c in run.report.citations)
        assert run.writer.passed_validation is True

    async def test_the_run_records_how_it_reached_its_answer(self, served_pages):
        run = await build_pipeline(search=StubSearchAgent([SOURCES])).run(QUESTION)

        assert run.rounds
        first = run.rounds[0]
        assert first.angle
        assert first.sources_found == 3
        assert first.sources_new == 3
        assert first.findings_added > 0
        assert run.stop_decision is not None
        assert run.stop_decision.reason
        assert "Verified findings" in run.summary()

    async def test_a_source_seen_twice_is_only_read_once(self, served_pages):
        # Both rounds return the same sources; the second should add nothing.
        search = StubSearchAgent([SOURCES[:1], SOURCES[:1]])
        run = await build_pipeline(
            search=search, rule=StoppingRule(min_findings=99, max_rounds=3)
        ).run(QUESTION)

        assert run.rounds[0].sources_new == 1
        assert run.rounds[1].sources_new == 0
        assert run.rounds[1].sources_read == 0

    async def test_a_stagnant_run_stops_instead_of_burning_rounds(self, served_pages):
        search = StubSearchAgent([SOURCES[:1], SOURCES[:1], SOURCES[:1]])
        run = await build_pipeline(
            search=search, rule=StoppingRule(min_findings=99, max_rounds=5)
        ).run(QUESTION)

        assert run.stop_decision.cause == "no_new_sources"
        assert len(run.rounds) == 2

    async def test_thin_evidence_is_declared_on_the_report(self, served_pages):
        search = StubSearchAgent([SOURCES[:1]])
        run = await build_pipeline(
            search=search, rule=StoppingRule(min_findings=99, max_rounds=1)
        ).run(QUESTION)

        assert run.stop_decision.cause == "round_limit"
        assert any("evidence_gathering" in caveat for caveat in run.report.caveats)

    async def test_a_fabricating_extractor_produces_no_report_rather_than_a_false_one(
        self, served_pages
    ):
        run = await build_pipeline(
            search=StubSearchAgent([SOURCES]), extractor=fabricating_extractor
        ).run(QUESTION)

        assert run.findings == []
        assert run.succeeded is False
        assert run.stop_decision.cause == "no_evidence"
        assert "No verified findings" in run.report.introduction
        # Every fabricated claim was rejected with a reason.
        rejected = sum(len(r.rejected) for r in run.extraction_reports)
        assert rejected > 0
        # And the count reaches the run record, which is what the CLI and the
        # web interface show. A mutation test found this could be hard-coded
        # to zero without any test noticing.
        assert sum(r.claims_rejected for r in run.rounds) == rejected
        assert "Claims rejected in verification: 3" in run.summary()

    async def test_unreachable_sources_do_not_stop_the_run(self):
        with respx.mock:
            respx.get("https://example.org/trial").mock(
                return_value=httpx.Response(200, text=PAGES["https://example.org/trial"],
                                            headers={"content-type": "text/html"})
            )
            respx.get("https://example.org/survey").mock(return_value=httpx.Response(404))
            respx.get("https://example.org/critique").mock(return_value=httpx.Response(500))

            run = await build_pipeline(search=StubSearchAgent([SOURCES])).run(QUESTION)

        assert run.rounds[0].unreadable_sources == 2
        assert run.findings, "the one reachable source should still have produced findings"
        assert run.report.sections

    async def test_a_run_with_no_sources_at_all_ends_honestly(self):
        search = StubSearchAgent([[]])
        run = await build_pipeline(search=search, rule=StoppingRule(max_rounds=2)).run(QUESTION)

        assert run.findings == []
        assert run.succeeded is False
        assert run.report.sections == []
        assert run.stop_decision.cause in ("no_new_sources", "no_evidence")

    async def test_reading_is_capped_per_round(self, served_pages):
        pipeline = build_pipeline(search=StubSearchAgent([SOURCES]))
        pipeline.max_sources_per_round = 1
        run = await pipeline.run(QUESTION)

        assert run.rounds[0].sources_new == 3
        assert run.rounds[0].sources_read == 1
