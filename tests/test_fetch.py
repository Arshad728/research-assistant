"""Tests for fetching a source and deciding how to read it (Phase 3.1)."""
import httpx
import pytest
import respx

from research_assistant.extraction.fetch import FetchError, fetch_source, resolve_fetch_url
from research_assistant.schemas import CandidateSource

from .test_parsers import ARTICLE_HTML, build_pdf


class TestResolveFetchUrl:
    def test_arxiv_abstract_pages_are_rewritten_to_the_pdf(self):
        assert resolve_fetch_url("https://arxiv.org/abs/2401.01234") == (
            "https://arxiv.org/pdf/2401.01234"
        )

    def test_version_suffixes_are_preserved(self):
        assert resolve_fetch_url("http://arxiv.org/abs/2401.01234v2") == (
            "https://arxiv.org/pdf/2401.01234v2"
        )

    def test_other_urls_are_left_alone(self):
        url = "https://example.org/papers/four-day-week"
        assert resolve_fetch_url(url) == url


@respx.mock
async def test_fetches_and_parses_a_pdf_by_content_type():
    data = build_pdf(["Hourly productivity increased by 8% during the trial."])
    respx.get("https://example.org/paper").mock(
        return_value=httpx.Response(200, content=data, headers={"content-type": "application/pdf"})
    )

    document = await fetch_source("https://example.org/paper")

    assert document.content_kind == "pdf"
    assert document.page_count == 1
    assert "Hourly productivity increased by 8%" in document.text


@respx.mock
async def test_detects_a_pdf_even_when_the_content_type_lies():
    data = build_pdf(["Some measured findings about scheduling and output."])
    respx.get("https://example.org/download").mock(
        return_value=httpx.Response(200, content=data, headers={"content-type": "text/html"})
    )

    document = await fetch_source("https://example.org/download")
    assert document.content_kind == "pdf"


@respx.mock
async def test_fetches_and_parses_a_web_page():
    respx.get("https://example.org/trial").mock(
        return_value=httpx.Response(
            200, text=ARTICLE_HTML, headers={"content-type": "text/html; charset=utf-8"}
        )
    )

    document = await fetch_source("https://example.org/trial")

    assert document.content_kind == "html"
    assert "Four-Day Week Trial Results" in document.title
    assert "hourly productivity increased by 8%" in document.text.lower()


@respx.mock
async def test_an_arxiv_candidate_is_fetched_as_a_pdf_not_an_abstract_page():
    data = build_pdf(["Full paper text, not just the abstract."])
    route = respx.get("https://arxiv.org/pdf/2401.01234").mock(
        return_value=httpx.Response(200, content=data, headers={"content-type": "application/pdf"})
    )

    source = CandidateSource(
        title="Shorter Working Weeks",
        source_url="http://arxiv.org/abs/2401.01234",
        source_type="preprint",
        relevance_note="A multi-site trial.",
    )
    document = await fetch_source(source)

    assert route.called
    assert "Full paper text" in document.text
    # The recorded URL stays the one the Search Agent shortlisted, so the
    # citation still points where a reader would expect.
    assert document.source_url == "http://arxiv.org/abs/2401.01234"


@respx.mock
async def test_a_paywalled_or_missing_source_reports_why():
    respx.get("https://example.org/paywalled").mock(return_value=httpx.Response(403, text="nope"))

    with pytest.raises(FetchError, match="paywalled"):
        await fetch_source("https://example.org/paywalled")


@respx.mock
async def test_a_connection_failure_is_reported():
    respx.get("https://example.org/gone").mock(side_effect=httpx.ConnectError("no route"))

    with pytest.raises(FetchError, match="Could not download"):
        await fetch_source("https://example.org/gone")


@respx.mock
async def test_an_oversized_download_is_refused():
    from research_assistant.extraction import fetch as fetch_module

    payload = b"x" * (fetch_module.MAX_DOWNLOAD_BYTES + 1)
    respx.get("https://example.org/huge").mock(
        return_value=httpx.Response(200, content=payload, headers={"content-type": "text/html"})
    )

    with pytest.raises(FetchError, match="above the"):
        await fetch_source("https://example.org/huge")
