"""Tests for the pure parsing half of each retrieval client.

No network is touched here. Every test feeds a saved response shape to a
parser and checks what comes out, which is what makes this layer testable
in an environment that cannot reach the real APIs at all.
"""
import pytest

from research_assistant.retrieval.arxiv import parse_arxiv_atom
from research_assistant.retrieval.base import RetrievalError, condense
from research_assistant.retrieval.semantic_scholar import (
    parse_semantic_scholar_response,
)
from research_assistant.retrieval.web_search import (
    parse_serper_response,
    parse_tavily_response,
)
from research_assistant.schemas import CandidateSource

from .fixtures import (
    ARXIV_ATOM_RESPONSE,
    SEMANTIC_SCHOLAR_RESPONSE,
    SERPER_RESPONSE,
    TAVILY_RESPONSE,
)


class TestCondense:
    def test_collapses_whitespace_and_keeps_short_text_intact(self):
        assert condense("  hello   there\n  world ") == "hello there world"

    def test_truncates_on_a_word_boundary(self):
        text = "alpha beta gamma delta epsilon"
        result = condense(text, limit=14)
        assert result.endswith("...")
        assert " " not in result[: result.index("...")].split()[-1]
        assert len(result) <= 18

    def test_handles_missing_text(self):
        assert condense(None) == ""


class TestArxivParser:
    def test_parses_entries_into_candidate_sources(self):
        sources = parse_arxiv_atom(ARXIV_ATOM_RESPONSE)
        assert all(isinstance(s, CandidateSource) for s in sources)
        assert [s.source_url for s in sources] == [
            "http://arxiv.org/abs/2401.01234v1",
            "http://arxiv.org/abs/2402.05678v2",
        ]

    def test_skips_entries_without_a_usable_link(self):
        sources = parse_arxiv_atom(ARXIV_ATOM_RESPONSE)
        assert not any("Malformed" in s.title for s in sources)

    def test_collapses_multiline_titles(self):
        first = parse_arxiv_atom(ARXIV_ATOM_RESPONSE)[0]
        assert "\n" not in first.title
        assert first.title == "Shorter Working Weeks and Measured Output: A Multi-Site Trial"

    def test_labels_arxiv_results_as_preprints(self):
        assert {s.source_type for s in parse_arxiv_atom(ARXIV_ATOM_RESPONSE)} == {"preprint"}

    def test_relevance_note_carries_authors_year_and_category(self):
        first = parse_arxiv_atom(ARXIV_ATOM_RESPONSE)[0]
        assert "A. Researcher" in first.relevance_note
        assert "2024" in first.relevance_note
        assert "econ.GN" in first.relevance_note

    def test_abbreviates_long_author_lists(self):
        second = parse_arxiv_atom(ARXIV_ATOM_RESPONSE)[1]
        assert "et al." in second.relevance_note

    def test_respects_max_results(self):
        assert len(parse_arxiv_atom(ARXIV_ATOM_RESPONSE, max_results=1)) == 1

    def test_invalid_xml_raises_a_clear_error(self):
        with pytest.raises(RetrievalError, match="not valid XML"):
            parse_arxiv_atom("<feed><unclosed>")


class TestSemanticScholarParser:
    def test_parses_papers_and_skips_untitled_records(self):
        sources = parse_semantic_scholar_response(SEMANTIC_SCHOLAR_RESPONSE)
        assert len(sources) == 2
        assert sources[0].title == "The Four-Day Week: Assessing Global Trials"

    def test_falls_back_to_arxiv_link_when_url_is_missing(self):
        sources = parse_semantic_scholar_response(SEMANTIC_SCHOLAR_RESPONSE)
        assert sources[1].source_url == "https://arxiv.org/abs/2303.09999"

    def test_an_arxiv_link_is_preferred_over_the_semantic_scholar_page(self):
        # So that the same preprint found on arXiv and on Semantic Scholar
        # deduplicates to one source. An earlier version preferred the S2
        # landing page, which meant the same paper counted twice -- and the
        # stopping rule treats distinct sources as corroboration.
        from research_assistant.search.planner import deduplicate
        from research_assistant.schemas import CandidateSource

        payload = {
            "data": [
                {
                    "title": "Shorter Working Weeks and Measured Output",
                    "abstract": "A multi-site trial.",
                    "url": "https://www.semanticscholar.org/paper/xyz",
                    "externalIds": {"ArXiv": "2401.01234"},
                    "year": 2024,
                }
            ]
        }
        from_s2 = parse_semantic_scholar_response(payload)[0]
        assert from_s2.source_url == "https://arxiv.org/abs/2401.01234"

        from_arxiv = CandidateSource(
            title="Shorter Working Weeks and Measured Output",
            source_url="http://arxiv.org/abs/2401.01234v1",
            source_type="preprint",
            relevance_note="n",
        )
        assert len(deduplicate([from_arxiv, from_s2])) == 1

    def test_classifies_journal_articles_without_claiming_peer_review(self):
        sources = parse_semantic_scholar_response(SEMANTIC_SCHOLAR_RESPONSE)
        assert sources[0].source_type == "journal article"
        assert "peer-reviewed" not in sources[0].source_type

    def test_classifies_venueless_arxiv_records_as_preprints(self):
        sources = parse_semantic_scholar_response(SEMANTIC_SCHOLAR_RESPONSE)
        assert sources[1].source_type == "preprint"

    def test_relevance_note_includes_venue_year_and_citation_count(self):
        note = parse_semantic_scholar_response(SEMANTIC_SCHOLAR_RESPONSE)[0].relevance_note
        assert "Journal of Work Studies" in note
        assert "2025" in note
        assert "cited 143 times" in note

    def test_handles_a_paper_with_no_abstract(self):
        note = parse_semantic_scholar_response(SEMANTIC_SCHOLAR_RESPONSE)[1].relevance_note
        assert note  # never empty
        assert "cited 7 times" in note

    def test_unexpected_shape_raises_rather_than_returning_nothing(self):
        with pytest.raises(RetrievalError, match="did not contain a 'data' list"):
            parse_semantic_scholar_response({"papers": []})


class TestWebSearchParsers:
    def test_parses_tavily_results(self):
        sources = parse_tavily_response(TAVILY_RESPONSE)
        assert len(sources) == 2  # the linkless result is skipped
        assert sources[0].source_url == "https://example.org/four-day-week-report"
        assert sources[0].source_type == "web page"

    def test_parses_serper_results(self):
        sources = parse_serper_response(SERPER_RESPONSE)
        assert len(sources) == 1
        assert sources[0].source_url == "https://example.org/pilot-results"

    def test_tavily_and_serper_produce_the_same_shape(self):
        tavily = parse_tavily_response(TAVILY_RESPONSE)[0]
        serper = parse_serper_response(SERPER_RESPONSE)[0]
        assert set(tavily.model_dump()) == set(serper.model_dump())

    @pytest.mark.parametrize(
        "parser, payload, expected",
        [
            (parse_tavily_response, {"data": []}, "'results' list"),
            (parse_serper_response, {"results": []}, "'organic' list"),
        ],
    )
    def test_unexpected_shapes_raise_clear_errors(self, parser, payload, expected):
        with pytest.raises(RetrievalError, match=expected):
            parser(payload)
