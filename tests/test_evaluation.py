"""Tests for the evaluation harness (Phase 6.1).

The harness is evaluated against the offline demo pipeline, which is the
only way to test it deterministically. What is being checked is not "does
the system answer well" -- that needs a real corpus and a human -- but
whether the harness measures honestly: does it count a correct refusal as
correct, does it survive a query that raises, and does it produce a
worksheet a person can actually use.
"""
import pytest

from research_assistant.demo import build_demo_pipeline
from research_assistant.evaluation import (
    TEST_QUERIES,
    EvalQuery,
    evaluate,
    query_by_id,
    render_evaluation_markdown,
)
from research_assistant.evaluation.metrics import measure_run, summarise


class TestEvalQuerySet:
    def test_includes_queries_that_should_find_nothing(self):
        negatives = [q for q in TEST_QUERIES if not q.expect_findings]
        assert len(negatives) >= 2, (
            "an evaluation with only answerable questions cannot detect fabrication"
        )

    def test_every_query_explains_what_it_tests(self):
        for query in TEST_QUERIES:
            assert query.tests.strip()
            assert query.question.strip()

    def test_query_ids_are_unique(self):
        ids = [q.id for q in TEST_QUERIES]
        assert len(ids) == len(set(ids))

    def test_lookup_by_id(self):
        assert query_by_id("four_day_week").difficulty == "straightforward"
        with pytest.raises(KeyError):
            query_by_id("no_such_query")


class TestMeasuring:
    async def test_an_answerable_query_is_measured_as_a_success(self):
        query = query_by_id("four_day_week")
        run = await build_demo_pipeline().run(query.question)
        metrics = measure_run(run, query)

        assert metrics.produced_report is True
        assert metrics.behaved_as_expected is True
        assert metrics.claims_verified > 0
        assert metrics.verification_pass_rate == 1.0
        assert metrics.citation_density and metrics.citation_density > 0.5
        assert metrics.report_validation_passed is True

    async def test_a_correct_refusal_counts_as_expected_behaviour(self):
        # The point of this test: finding nothing on an unstudied subject is
        # the right answer, and the harness must not score it as a failure.
        query = query_by_id("nonexistent_framework")
        run = await build_demo_pipeline().run(query.question)
        metrics = measure_run(run, query)

        assert metrics.produced_report is False
        assert metrics.expected_findings is False
        assert metrics.behaved_as_expected is True
        assert metrics.stop_cause == "no_evidence"

    async def test_a_fabricated_report_on_an_unstudied_subject_would_be_a_failure(self):
        query = EvalQuery(
            id="mislabelled",
            question="What is the effect of a four-day work week on productivity?",
            difficulty="should_find_little",
            tests="Deliberately mislabelled to prove the harness would catch fabrication.",
            expect_findings=False,
        )
        run = await build_demo_pipeline().run(query.question)
        metrics = measure_run(run, query)

        assert metrics.produced_report is True
        assert metrics.behaved_as_expected is False


class TestHarness:
    async def test_runs_every_query_and_summarises(self):
        result, runs = await evaluate(build_demo_pipeline(), spot_checks_per_query=2)

        assert result.summary.queries_run == len(TEST_QUERIES)
        assert len(runs) == len(TEST_QUERIES)
        assert result.failures == []
        assert result.summary.total_claims_verified > 0

    async def test_both_deliberately_unanswerable_queries_behave_correctly(self):
        result, _ = await evaluate(build_demo_pipeline())
        by_id = {m.query_id: m for m in result.summary.per_query}

        for query_id in ("microdosing_productivity", "nonexistent_framework"):
            assert by_id[query_id].produced_report is False
            assert by_id[query_id].behaved_as_expected is True

    async def test_a_query_that_raises_is_recorded_not_fatal(self):
        class ExplodingPipeline:
            def __init__(self):
                self.calls = 0

            async def run(self, question):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("simulated failure")
                return await build_demo_pipeline().run(question)

        queries = [query_by_id("four_day_week"), query_by_id("remote_work_wages")]
        result, runs = await evaluate(ExplodingPipeline(), queries)

        assert len(result.failures) == 1
        assert "simulated failure" in result.failures[0]
        assert len(runs) == 1  # the second query still ran

    async def test_spot_checks_are_sampled_from_real_findings(self):
        result, runs = await evaluate(build_demo_pipeline(), spot_checks_per_query=2)
        real_claims = {f.claim for run in runs for f in run.findings}

        assert result.spot_checks
        for item in result.spot_checks:
            assert item.claim in real_claims
            assert item.quoted_evidence.strip()
            assert item.source_url.startswith("http")

    async def test_sampling_is_reproducible(self):
        first, _ = await evaluate(build_demo_pipeline(), seed=7)
        second, _ = await evaluate(build_demo_pipeline(), seed=7)
        assert [c.claim for c in first.spot_checks] == [c.claim for c in second.spot_checks]


class TestReportRendering:
    async def test_the_written_report_contains_what_a_reader_needs(self):
        result, _ = await evaluate(build_demo_pipeline(), spot_checks_per_query=1)
        markdown = render_evaluation_markdown(result)

        assert "# Evaluation results" in markdown
        assert "## Spot-check worksheet" in markdown
        assert "| Query | Difficulty |" in markdown
        # Every query appears in the table.
        for query in TEST_QUERIES:
            assert query.id in markdown
        # The worksheet is checkable.
        assert "- [ ] The quotation appears in the source" in markdown
        # The limits of what was measured are stated, not implied.
        assert "cannot be measured by the system that produced the citations" in markdown
        assert "Comparison against doing it by hand" in markdown

    async def test_an_optional_note_is_surfaced_at_the_top(self):
        result, _ = await evaluate(build_demo_pipeline())
        markdown = render_evaluation_markdown(result, note="Run against the offline pipeline.")
        assert "> **Run against the offline pipeline.**" in markdown

    def test_an_empty_evaluation_still_renders(self):
        from research_assistant.evaluation.harness import EvaluationResult

        markdown = render_evaluation_markdown(
            EvaluationResult(summary=summarise([]), spot_checks=[], failures=[])
        )
        assert "nothing to spot-check" in markdown
