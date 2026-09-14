"""Tests for shortlist reconciliation (Phase 2.2).

The important case is the last one: a model that names a plausible-looking
URL no tool ever returned must not be able to get that URL into the
shortlist. That is the same failure Chapter 1.5 describes -- a fabricated
citation that looks exactly as convincing as a real one -- caught at the
earliest point it can appear.
"""
import json

from research_assistant.schemas import CandidateSource
from research_assistant.search.reconcile import extract_json, reconcile_shortlist

RETRIEVED = [
    CandidateSource(
        title="The Four-Day Week: Assessing Global Trials",
        source_url="https://example.org/four-day-week-trial-2025",
        source_type="journal article",
        relevance_note="Journal of Work Studies | 2025 | cited 143 times.",
    ),
    CandidateSource(
        title="Shorter Working Weeks and Measured Output",
        source_url="http://arxiv.org/abs/2401.01234v1",
        source_type="preprint",
        relevance_note="A. Researcher | 2024 | econ.GN.",
    ),
    CandidateSource(
        title="Opinion: the four-day week is a fad",
        source_url="https://example.com/opinion-column",
        source_type="web page",
        relevance_note="A columnist argues the idea will not last.",
    ),
]


class TestExtractJson:
    def test_reads_bare_json(self):
        assert extract_json('[{"a": 1}]') == [{"a": 1}]

    def test_reads_json_inside_a_fenced_block(self):
        text = 'Here is my shortlist:\n```json\n[{"a": 1}]\n```\nHope that helps.'
        assert extract_json(text) == [{"a": 1}]

    def test_reads_json_surrounded_by_prose(self):
        text = 'I picked two sources. [{"source_url": "https://x.org"}] Let me know.'
        assert extract_json(text) == [{"source_url": "https://x.org"}]

    def test_returns_none_when_there_is_no_json(self):
        assert extract_json("I could not find anything useful.") is None
        assert extract_json("") is None


class TestReconcile:
    def test_keeps_only_the_sources_the_model_selected(self):
        model_reply = json.dumps([
            {"source_url": "https://example.org/four-day-week-trial-2025",
             "relevance_note": "Controlled trial with before/after output measures."},
            {"source_url": "http://arxiv.org/abs/2401.01234v1",
             "relevance_note": "Multi-site trial, directly on point."},
        ])
        shortlist = reconcile_shortlist(model_reply, RETRIEVED)

        assert len(shortlist.sources) == 2
        assert not shortlist.dropped_unknown_urls
        assert "opinion-column" not in {s.source_url for s in shortlist.sources}

    def test_uses_the_models_note_but_the_retrieved_title_and_type(self):
        model_reply = json.dumps([
            {
                "source_url": "https://example.org/four-day-week-trial-2025",
                "title": "A Title The Model Made Up",
                "source_type": "peer-reviewed study",
                "relevance_note": "Controlled trial with before/after output measures.",
            }
        ])
        source = reconcile_shortlist(model_reply, RETRIEVED).sources[0]

        assert source.title == "The Four-Day Week: Assessing Global Trials"
        assert source.source_type == "journal article"
        assert source.relevance_note == "Controlled trial with before/after output measures."

    def test_falls_back_to_the_retrieved_note_when_the_model_gives_none(self):
        model_reply = json.dumps([{"source_url": "https://example.org/four-day-week-trial-2025"}])
        source = reconcile_shortlist(model_reply, RETRIEVED).sources[0]
        assert source.relevance_note.startswith("Journal of Work Studies")

    def test_matches_urls_that_differ_only_cosmetically(self):
        model_reply = json.dumps([{"source_url": "https://arxiv.org/pdf/2401.01234v3"}])
        shortlist = reconcile_shortlist(model_reply, RETRIEVED)

        assert len(shortlist.sources) == 1
        # the canonical retrieved URL is what gets passed on, not the model's variant
        assert shortlist.sources[0].source_url == "http://arxiv.org/abs/2401.01234v1"

    def test_drops_a_url_no_tool_ever_returned(self):
        model_reply = json.dumps([
            {"source_url": "https://example.org/four-day-week-trial-2025", "relevance_note": "real"},
            {"source_url": "https://journals.example.net/invented-study-2024",
             "relevance_note": "looks plausible, was never retrieved"},
        ])
        shortlist = reconcile_shortlist(model_reply, RETRIEVED)

        assert len(shortlist.sources) == 1
        assert shortlist.dropped_unknown_urls == ["https://journals.example.net/invented-study-2024"]

    def test_accepts_a_wrapped_object_as_well_as_a_bare_list(self):
        model_reply = json.dumps(
            {"sources": [{"source_url": "https://example.org/four-day-week-trial-2025"}]}
        )
        assert len(reconcile_shortlist(model_reply, RETRIEVED).sources) == 1

    def test_ignores_a_duplicate_selection(self):
        model_reply = json.dumps([
            {"source_url": "https://example.org/four-day-week-trial-2025"},
            {"source_url": "https://example.org/four-day-week-trial-2025/"},
        ])
        assert len(reconcile_shortlist(model_reply, RETRIEVED).sources) == 1

    def test_respects_max_items(self):
        model_reply = json.dumps([{"source_url": s.source_url} for s in RETRIEVED])
        assert len(reconcile_shortlist(model_reply, RETRIEVED, max_items=2).sources) == 2

    def test_unparseable_reply_degrades_to_everything_retrieved(self):
        shortlist = reconcile_shortlist("I could not decide, sorry.", RETRIEVED)

        assert shortlist.used_fallback is True
        assert len(shortlist.sources) == len(RETRIEVED)
        assert "unfiltered" in shortlist.parse_error

    def test_max_items_is_respected_even_on_the_fallback_path(self):
        # A mutation test found max_items was ignored when the model's reply
        # was unparseable, so a failed parse could flood the next agent with
        # every source retrieved.
        shortlist = reconcile_shortlist("not json at all", RETRIEVED, max_items=2)

        assert shortlist.used_fallback is True
        assert len(shortlist.sources) == 2

    def test_empty_selection_is_respected_not_treated_as_failure(self):
        shortlist = reconcile_shortlist("[]", RETRIEVED)
        assert shortlist.sources == []
        assert shortlist.used_fallback is False
