"""Tests for the deterministic half of the Search Agent (Phase 2.2)."""
import pytest

from research_assistant.schemas import CandidateSource
from research_assistant.search.planner import (
    SearchPlanner,
    deduplicate,
    distinct_domains,
    keywords,
    normalize_url,
)


def source(url: str, title: str = "A study of four-day work week productivity", note: str = "") -> CandidateSource:
    return CandidateSource(
        title=title,
        source_url=url,
        source_type="journal article",
        relevance_note=note or "Measures productivity under a four-day working week schedule.",
    )


class TestNormalizeUrl:
    @pytest.mark.parametrize(
        "left, right",
        [
            ("https://example.org/paper", "http://example.org/paper"),
            ("https://www.example.org/paper", "https://example.org/paper"),
            ("https://example.org/paper/", "https://example.org/paper"),
            ("https://example.org/paper#section-2", "https://example.org/paper"),
            ("https://EXAMPLE.org/paper", "https://example.org/paper"),
        ],
    )
    def test_cosmetic_differences_collapse(self, left, right):
        assert normalize_url(left) == normalize_url(right)

    def test_arxiv_abs_pdf_and_versions_are_the_same_paper(self):
        forms = [
            "http://arxiv.org/abs/2401.01234v1",
            "https://arxiv.org/abs/2401.01234",
            "https://arxiv.org/pdf/2401.01234v3",
        ]
        assert len({normalize_url(url) for url in forms}) == 1

    def test_pre_2007_arxiv_identifiers_are_canonicalised_too(self):
        forms = [
            "https://arxiv.org/abs/cs/0301001",
            "https://arxiv.org/pdf/cs/0301001v2",
        ]
        assert len({normalize_url(url) for url in forms}) == 1
        assert normalize_url("https://arxiv.org/abs/cs/0301001") != normalize_url(
            "https://arxiv.org/abs/cs/0301002"
        )

    def test_different_papers_stay_different(self):
        assert normalize_url("https://arxiv.org/abs/2401.01234") != normalize_url(
            "https://arxiv.org/abs/2402.05678"
        )

    def test_query_strings_are_preserved(self):
        assert normalize_url("https://example.org/s?id=7") != normalize_url("https://example.org/s?id=8")


class TestDeduplicate:
    def test_removes_the_same_paper_reached_by_different_routes(self):
        sources = [
            source("https://arxiv.org/abs/2401.01234v1"),
            source("https://arxiv.org/pdf/2401.01234v2"),
            source("https://example.org/other"),
        ]
        assert len(deduplicate(sources)) == 2

    def test_keeps_first_occurrence_order(self):
        first = source("https://example.org/a", title="First")
        second = source("https://example.org/a/", title="Second")
        assert deduplicate([first, second])[0].title == "First"

    def test_counts_distinct_domains(self):
        sources = [
            source("https://arxiv.org/abs/2401.01234"),
            source("https://arxiv.org/abs/2402.05678"),
            source("https://example.org/x"),
        ]
        assert distinct_domains(sources) == 2


class TestKeywords:
    def test_drops_stopwords_and_short_words(self):
        assert keywords("What does the research say about a four day week") == {
            "four",
            "day",
            "week",
        }


class TestRepeatDetection:
    def test_rejects_the_same_query_reordered(self):
        planner = SearchPlanner(original_query="four day week productivity")
        planner.register_attempt("four day week productivity", "search_web", [])
        assert planner.is_repeat("productivity four-day week") is True

    def test_accepts_a_genuinely_different_query(self):
        planner = SearchPlanner(original_query="four day week productivity")
        planner.register_attempt("four day week productivity", "search_web", [])
        assert planner.is_repeat("reduced hours output manufacturing") is False

    def test_an_empty_query_counts_as_a_repeat(self):
        planner = SearchPlanner(original_query="four day week")
        assert planner.is_repeat("   the and of   ") is True


class TestAssessment:
    def test_no_results_is_thin(self):
        planner = SearchPlanner(original_query="four day week productivity")
        assessment = planner.assess([])
        assert assessment.verdict == "thin"
        assert "No results" in assessment.reason

    def test_too_few_results_is_thin(self):
        planner = SearchPlanner(original_query="four day week productivity")
        assessment = planner.assess([source("https://example.org/a")])
        assert assessment.verdict == "thin"
        assert "Only 1 result" in assessment.reason

    def test_enough_results_from_one_domain_is_still_thin(self):
        planner = SearchPlanner(original_query="four day week productivity")
        results = [source(f"https://example.org/{i}") for i in range(5)]
        assessment = planner.assess(results)
        assert assessment.verdict == "thin"
        assert "not corroboration" in assessment.reason

    def test_results_about_something_else_are_off_topic(self):
        planner = SearchPlanner(original_query="four day work week productivity")
        results = [
            CandidateSource(
                title="Photosynthesis in alpine lichen",
                source_url=f"https://example{i}.org/paper",
                source_type="journal article",
                relevance_note="Measures carbon fixation rates at altitude.",
            )
            for i in range(5)
        ]
        assessment = planner.assess(results)
        assert assessment.verdict == "off_topic"

    def test_varied_on_topic_results_are_adequate(self):
        planner = SearchPlanner(original_query="four day work week productivity")
        results = [source(f"https://example{i}.org/paper") for i in range(5)]
        assessment = planner.assess(results)
        assert assessment.verdict == "adequate"
        assert assessment.distinct_domains == 5


class TestLoopControl:
    def test_first_round_always_runs(self):
        planner = SearchPlanner(original_query="four day week productivity")
        assert planner.should_continue() is True

    def test_stops_once_results_are_adequate(self):
        planner = SearchPlanner(original_query="four day work week productivity")
        planner.register_attempt(
            "four day work week productivity",
            "search_web",
            [source(f"https://example{i}.org/paper") for i in range(5)],
        )
        assert planner.should_continue() is False
        assert "Enough material" in planner.stop_reason()

    def test_keeps_going_while_results_are_thin(self):
        planner = SearchPlanner(original_query="four day work week productivity")
        planner.register_attempt("four day work week productivity", "search_web", [])
        assert planner.should_continue() is True

    def test_respects_the_round_limit(self):
        planner = SearchPlanner(original_query="four day work week productivity", max_rounds=2)
        planner.register_attempt("first query", "search_web", [])
        planner.register_attempt("second entirely different query", "search_arxiv", [])
        assert planner.should_continue() is False
        assert "2-round limit" in planner.stop_reason()

    def test_results_accumulate_across_rounds_without_duplicates(self):
        planner = SearchPlanner(original_query="four day work week productivity")
        planner.register_attempt("q1", "search_arxiv", [source("https://arxiv.org/abs/2401.01234v1")])
        planner.register_attempt("q2 different words entirely", "search_web", [
            source("https://arxiv.org/pdf/2401.01234v2"),
            source("https://example.org/new"),
        ])
        assert len(planner.collected) == 2
        assert planner.rounds_used == 2


class TestFallbackReformulation:
    def test_broadening_drops_quoted_phrases_and_years(self):
        planner = SearchPlanner(original_query='"four day week" productivity 2024 trials')
        broadened = planner.broaden('"four day week" productivity 2024 trials')
        assert '"' not in broadened
        assert "2024" not in broadened

    def test_broadening_keeps_the_distinctive_terms_not_the_filler(self):
        # Regression: an earlier version kept the shortest words and turned
        # this query into "day four work", losing the actual subject.
        planner = SearchPlanner(original_query="effect of a four day work week on productivity")
        broadened = planner.broaden("effect of a four day work week on productivity")

        assert "productivity" in broadened
        assert len(broadened.split()) <= 3

    def test_broadening_preserves_word_order(self):
        planner = SearchPlanner(original_query="measured productivity outcomes shorter schedules")
        broadened = planner.broaden("measured productivity outcomes shorter schedules")
        words = broadened.split()
        assert words == sorted(words, key=lambda w: "measured productivity outcomes shorter schedules".index(w))

    def test_suggested_query_is_never_a_repeat(self):
        planner = SearchPlanner(original_query="four day work week productivity outcomes")
        planner.register_attempt("four day work week productivity outcomes", "search_web", [])
        suggestion = planner.suggest_next_query()
        assert suggestion is not None
        assert planner.is_repeat(suggestion) is False

    def test_returns_none_when_nothing_new_is_left(self):
        planner = SearchPlanner(original_query="productivity")
        planner.register_attempt("productivity", "search_web", [])
        assert planner.suggest_next_query() is None
