"""Phase 2.3, live half: retrieval against the real APIs.

These are opt-in. A plain ``pytest`` run skips them entirely, because they
need outbound internet access and, for two of the three tools, an API key.
Run them deliberately with::

    pytest -m integration

Why they are separated rather than simply written as normal tests: the
environment this project was developed in cannot reach arxiv.org,
api.semanticscholar.org or api.tavily.com at all (see the Build Log,
Phase 1.2). Tests that silently fail for environmental reasons train you to
ignore failures. Tests that announce "not run, and here is what is missing"
do not.

What these check is different from the offline tests. The offline suite
proves the parsing and planning logic is right. These prove the assumptions
about each API are still right: that the endpoints exist, the auth header
names are correct, and the response shapes have not changed underneath us.
"""
import os
import re

import pytest

from research_assistant.config import get_settings
from research_assistant.retrieval import search_arxiv, search_semantic_scholar, search_web
from research_assistant.schemas import CandidateSource
from research_assistant.search.planner import SearchPlanner

pytestmark = pytest.mark.integration

SAMPLE_QUERY = "four day work week productivity trial"


def _check(sources, *, minimum=1):
    assert len(sources) >= minimum, "the API responded but returned nothing usable"
    for source in sources:
        assert isinstance(source, CandidateSource)
        assert source.source_url.startswith(("http://", "https://"))
        assert source.title.strip()
        assert source.relevance_note.strip()


async def test_arxiv_live():
    """arXiv needs no key -- the only requirement is outbound network access."""
    _check(await search_arxiv(SAMPLE_QUERY, max_results=5))


async def test_semantic_scholar_live():
    if not get_settings().semantic_scholar_api_key:
        pytest.skip(
            "No SEMANTIC_SCHOLAR_API_KEY set. The keyless pool throttles /paper/search "
            "heavily, so this test would be flaky rather than informative."
        )
    _check(await search_semantic_scholar(SAMPLE_QUERY, max_results=5))


async def test_web_search_live():
    if get_settings().search_provider == "none":
        pytest.skip("No TAVILY_API_KEY or SERPER_API_KEY set.")
    _check(await search_web(SAMPLE_QUERY, max_results=5))


async def test_combined_results_are_judged_adequate():
    """The real check Chapter 6.3 asks for: are real results actually usable?"""
    sources = await search_arxiv(SAMPLE_QUERY, max_results=8)
    if get_settings().search_provider != "none":
        sources.extend(await search_web(SAMPLE_QUERY, max_results=8))

    planner = SearchPlanner(original_query=SAMPLE_QUERY)
    assessment = planner.register_attempt(SAMPLE_QUERY, "live", sources)

    assert assessment.verdict != "off_topic", (
        f"Live results looked off-topic: {assessment.reason}. "
        "Either the query planning or the relevance heuristic needs attention."
    )


async def test_fetching_and_parsing_a_real_paper():
    """Phase 3.1 against reality: does a real arXiv PDF actually come out as text?

    Needs no API key. This is the test most likely to catch a parsing
    problem that saved fixtures cannot, because real papers are messier than
    anything written by hand for a test.
    """
    from research_assistant.extraction import fetch_source

    document = await fetch_source("https://arxiv.org/abs/1706.03762")  # a stable, famous paper

    assert document.content_kind == "pdf"
    assert document.is_usable, f"only {document.char_count} characters extracted"
    assert document.page_count and document.page_count > 1
    # De-hyphenation should have joined words broken across lines. Check the
    # specific pattern it targets (a hyphen still stuck to a word character
    # with nothing rejoined) rather than a blanket search for any dash before
    # a newline -- a real paper can legitimately contain a standalone dash
    # (e.g. punctuation in a figure) that was never a split word to begin with.
    assert not re.search(r"\w-\s*\n\s*\w", document.text)


async def test_verification_finds_a_real_quotation_in_a_real_paper():
    """Phase 3.3 against reality: can a genuine quote be located in a real PDF?

    Saved fixtures are tidy. Real PDFs have column breaks, ligatures and
    stray spacing, which is exactly what the snippet matcher has to survive.
    """
    from research_assistant.extraction import fetch_source, locate_snippet

    document = await fetch_source("https://arxiv.org/abs/1706.03762")

    # Take a real sentence out of the middle of the extracted text and feed
    # it back in, the way an honest extraction would quote it.
    words = document.text.split()
    quotation = " ".join(words[len(words) // 2 : len(words) // 2 + 25])

    match = locate_snippet(quotation, document.text)
    assert match.found is True, f"a quotation taken from the document itself was not found: {quotation!r}"


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No ANTHROPIC_API_KEY set; running the agent itself costs money.",
)
async def test_extraction_agent_on_a_real_paper():
    """The Phase 3 deliverable end to end: a real source in, verified findings out."""
    from research_assistant.agents import ExtractionAgent
    from research_assistant.extraction import fetch_source

    document = await fetch_source("https://arxiv.org/abs/1706.03762")
    findings, report = await ExtractionAgent().extract_from_document(
        document, "What architecture does this paper propose and how does it perform?"
    )

    assert report.status in ("ok", "no_claims")
    for finding in findings:
        assert finding.verified is True
        assert finding.supporting_snippet.strip()
    # Every rejection should carry a reason a person can act on.
    for rejection in report.rejected:
        assert rejection.failed_checks
        assert rejection.detail.strip()


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No ANTHROPIC_API_KEY set; running the agent itself costs money.",
)
async def test_writer_agent_stays_inside_its_evidence():
    """Phase 4 against a real model: does the writer invent anything?

    The offline tests prove the checks catch a writer that strays. This one
    asks the different question those cannot: does a real model, given real
    instructions, actually stay put? A failure here is informative either
    way -- either the prompt needs work, or the checks are too strict.
    """
    from research_assistant.agents import WriterAgent
    from research_assistant.schemas import VerifiedFinding

    findings = [
        VerifiedFinding(
            claim="Output per hour rose by 8% while total weekly output was unchanged.",
            supporting_snippet="hourly productivity increased by 8%, while total weekly output "
            "remained statistically unchanged",
            source_url="https://example.org/trial-2025",
        ),
        VerifiedFinding(
            claim="Survey respondents reported higher satisfaction under the shorter schedule.",
            supporting_snippet="Respondents reported markedly higher satisfaction scores under "
            "the shorter working schedule than under their previous arrangements.",
            source_url="https://example.org/survey-2025",
        ),
    ]

    outcome = await WriterAgent(max_revisions=1).write_report(
        "What is the effect of a four-day work week on productivity?", findings
    )

    assert not outcome.draft_failed
    assert outcome.report.sections, "the writer produced no themed sections"
    assert outcome.passed_validation, (
        "the finished report failed validation: "
        + "; ".join(issue.detail for issue in outcome.validation.errors)
    )


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No ANTHROPIC_API_KEY set; running the agent itself costs money.",
)
async def test_full_pipeline_end_to_end():
    """The Phase 5.3 deliverable against reality: one question, one cited report.

    This is the most expensive test in the suite -- it runs every agent on
    live sources -- and the most informative, because it is the only one
    where the question "does the whole thing actually work" gets a real
    answer rather than a simulated one.
    """
    from research_assistant.orchestration import ResearchPipeline, StoppingRule

    run = await ResearchPipeline(stopping_rule=StoppingRule(max_rounds=2)).run(
        "What does current research say about the effect of a four-day work week on productivity?"
    )

    assert run.rounds, "the pipeline never completed a round"
    assert run.stop_decision is not None
    assert run.report is not None

    if not run.findings:
        pytest.fail(
            "The pipeline produced no verified findings. That is a legitimate outcome for a "
            "genuinely obscure question, but not for this one: "
            f"{run.stop_decision.reason}"
        )

    assert run.report.sections, "findings were verified but no report was written from them"
    assert run.report.citations
    finding_urls = {f.source_url for f in run.findings}
    assert all(c.source_url in finding_urls for c in run.report.citations), (
        "the report cites a source that produced no verified finding"
    )
    assert run.writer.passed_validation, (
        "the finished report failed validation: "
        + "; ".join(issue.detail for issue in run.writer.validation.errors)
    )


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No ANTHROPIC_API_KEY set; running the agent itself costs money.",
)
async def test_search_agent_end_to_end():
    """The full Phase 2 deliverable: a query in, a checked shortlist out."""
    from research_assistant.agents import SearchAgent

    outcome = await SearchAgent(max_rounds=2).find_sources(
        "What does current research say about the effect of a four-day work week on productivity?"
    )

    assert outcome.retrieved_count > 0, "the agent never successfully retrieved anything"
    _check(outcome.sources)
    assert not outcome.dropped_unknown_urls, (
        f"The agent proposed URLs that were never retrieved: {outcome.dropped_unknown_urls}. "
        "They were correctly discarded, but this is worth investigating."
    )
