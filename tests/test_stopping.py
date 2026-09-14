"""Tests for the stopping rule (Phase 5.2).

Chapter 4.2 calls stopping one of the harder problems in the system, and
Decision Record 0002 argues that it belongs in code precisely so it can be
tested against every combination of evidence and round count rather than
merely sampled.
"""
import pytest

from research_assistant.orchestration.stopping import StoppingRule
from research_assistant.schemas import SharedState, VerifiedFinding


def finding(source: str, claim: str = "A claim.") -> VerifiedFinding:
    return VerifiedFinding(
        claim=claim,
        supporting_snippet="A quotation long enough to serve as supporting evidence here.",
        source_url=source,
    )


def state_with(*sources: str) -> SharedState:
    state = SharedState(original_query="four day work week productivity")
    state.add_verified_findings([finding(url, f"Claim from {url}") for url in sources])
    return state


class TestSufficientEvidence:
    def test_stops_once_the_threshold_is_met(self):
        state = state_with("https://a.org/1", "https://a.org/1", "https://b.org/2", "https://c.org/3")
        decision = StoppingRule().evaluate(state, rounds_used=1)

        assert decision.should_stop is True
        assert decision.cause == "sufficient_evidence"
        assert decision.findings_count == 4
        assert decision.distinct_sources == 3

    def test_enough_findings_from_one_source_is_not_enough(self):
        # Four findings, but all from the same paper: no corroboration.
        state = state_with(*["https://a.org/1"] * 4)
        decision = StoppingRule().evaluate(state, rounds_used=1)

        assert decision.should_stop is False
        assert decision.cause == "continue"
        assert "independent sources" in decision.reason

    def test_enough_sources_but_too_few_findings_is_not_enough(self):
        state = state_with("https://a.org/1", "https://b.org/2")
        decision = StoppingRule().evaluate(state, rounds_used=1)

        assert decision.should_stop is False
        assert "2 of 4 findings" in decision.reason


class TestRoundLimit:
    def test_stops_at_the_limit_and_says_the_evidence_is_thin(self):
        state = state_with("https://a.org/1", "https://b.org/2")
        decision = StoppingRule(max_rounds=3).evaluate(state, rounds_used=3)

        assert decision.should_stop is True
        assert decision.cause == "round_limit"
        assert "should say so" in decision.reason
        assert decision.has_usable_evidence is True

    def test_a_run_that_found_nothing_is_reported_differently(self):
        empty = SharedState(original_query="q")
        decision = StoppingRule(max_rounds=2).evaluate(empty, rounds_used=2)

        assert decision.should_stop is True
        assert decision.cause == "no_evidence"
        assert decision.has_usable_evidence is False

    def test_sufficient_evidence_wins_over_the_round_limit(self):
        state = state_with("https://a.org/1", "https://a.org/1", "https://b.org/2", "https://c.org/3")
        decision = StoppingRule(max_rounds=3).evaluate(state, rounds_used=3)
        assert decision.cause == "sufficient_evidence"


class TestStagnation:
    def test_stops_when_a_round_finds_nothing_new(self):
        state = state_with("https://a.org/1")
        decision = StoppingRule().evaluate(state, rounds_used=1, new_sources_last_round=0)

        assert decision.should_stop is True
        assert decision.cause == "no_new_sources"
        assert "repeat work" in decision.reason

    def test_stagnation_with_nothing_verified_reports_the_more_important_fact(self):
        # Both "no new sources" and "nothing verified" are true here. The
        # caller needs the second one, so that is the cause reported.
        empty = SharedState(original_query="q")
        decision = StoppingRule().evaluate(empty, rounds_used=2, new_sources_last_round=0)

        assert decision.should_stop is True
        assert decision.cause == "no_evidence"
        assert decision.has_usable_evidence is False
        # The stagnation is still mentioned, just not as the headline.
        assert "already been seen" in decision.reason

    def test_new_sources_mean_the_run_continues(self):
        state = state_with("https://a.org/1")
        decision = StoppingRule().evaluate(state, rounds_used=1, new_sources_last_round=3)

        assert decision.should_stop is False

    def test_stagnation_never_triggers_before_the_first_round_completes(self):
        empty = SharedState(original_query="q")
        decision = StoppingRule().evaluate(empty, rounds_used=0, new_sources_last_round=0)
        assert decision.should_stop is False


class TestTuning:
    @pytest.mark.parametrize(
        "min_findings, min_sources, expect_stop",
        [(4, 2, False), (2, 2, True), (2, 1, True), (10, 5, False)],
    )
    def test_thresholds_are_the_only_thing_that_changes_the_verdict(
        self, min_findings, min_sources, expect_stop
    ):
        state = state_with("https://a.org/1", "https://b.org/2", "https://b.org/2")
        rule = StoppingRule(min_findings=min_findings, min_distinct_sources=min_sources)
        assert rule.evaluate(state, rounds_used=1).should_stop is expect_stop

    def test_every_decision_explains_itself(self):
        # Each reason must name the actual numbers, not just be non-empty.
        # An earlier version asserted only `.strip()`, which is true of any
        # string at all.
        state = state_with("https://a.org/1", "https://b.org/2")
        for rounds in (0, 1, 3):
            decision = StoppingRule().evaluate(state, rounds_used=rounds)
            assert str(decision.findings_count) in decision.reason
            assert len(decision.reason) > 30

    def test_the_shipped_defaults_are_pinned(self):
        # A mutation test found that every threshold test passed explicit
        # values, so the defaults a user actually gets were unprotected:
        # max_rounds could be changed to 99 with the suite still green.
        rule = StoppingRule()
        assert rule.min_findings == 4
        assert rule.min_distinct_sources == 2
        assert rule.max_rounds == 3

    def test_the_default_round_limit_actually_governs_a_run(self):
        # Pinning the value is not enough; it has to be the value used.
        state = state_with("https://a.org/1")
        assert StoppingRule().evaluate(state, rounds_used=2).should_stop is False
        assert StoppingRule().evaluate(state, rounds_used=3).should_stop is True
