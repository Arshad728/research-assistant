"""Measuring a run (Phase 6.1).

Chapter 7.1 names three measures that matter: citation accuracy, finding
relevance, and time compared to doing the research by hand. Only one of
those can be computed by a program.

**Time** is trivially measurable and is measured here.

**Citation accuracy** -- whether each claim is actually supported by the
source it cites -- cannot be honestly self-assessed. The system already
checked its own citations during extraction; running the same check again
and reporting a high score would be measuring nothing. Chapter 7.1 says as
much: this is "checked by a person spot-reading a sample of the report's
claims against the linked sources." So the harness produces a spot-check
worksheet instead of a number, and the number comes back from a human.

**Finding relevance** is the same: whether the evidence actually addresses
the question asked is a judgement, not a calculation.

What is computed here are the mechanical properties that stand in for
neither: how often the model's proposed claims survived verification, how
much of the report is cited, how many independent sources it rests on, and
whether the run behaved as the query expected. These are worth tracking
because a change in them points at something, even if none of them is
"accuracy."
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from ..orchestration.pipeline import ResearchRun
from ..report.models import CITATION_MARKER
from .queries import EvalQuery


class RunMetrics(BaseModel):
    """Everything measurable about one run, without human judgement."""

    query_id: str
    question: str
    difficulty: str

    # Did it work at all?
    produced_report: bool = False
    expected_findings: bool = True
    behaved_as_expected: bool = False

    # Effort and cost proxies.
    elapsed_seconds: float = 0.0
    rounds_run: int = 0
    sources_found: int = 0
    sources_read: int = 0
    sources_unreadable: int = 0

    # Verification.
    claims_proposed: int = 0
    claims_verified: int = 0
    claims_rejected: int = 0

    # Report quality proxies.
    distinct_sources_cited: int = 0
    body_paragraphs: int = 0
    cited_paragraphs: int = 0
    report_validation_passed: bool = False
    writer_revisions: int = 0
    stop_cause: str = ""

    @property
    def verification_pass_rate(self) -> Optional[float]:
        """Share of proposed claims that survived verification.

        A very high rate may mean the extractor is careful, or that the
        checks are too lenient. A very low one may mean the extractor is
        careless, or that the checks are too strict. It is a number to
        investigate, not a score to maximise.
        """
        if not self.claims_proposed:
            return None
        return self.claims_verified / self.claims_proposed

    @property
    def citation_density(self) -> Optional[float]:
        """Share of body paragraphs carrying at least one citation."""
        if not self.body_paragraphs:
            return None
        return self.cited_paragraphs / self.body_paragraphs


def measure_run(run: ResearchRun, query: EvalQuery) -> RunMetrics:
    """Compute every mechanical measure for one completed run."""
    report = run.report
    paragraphs = report.body_paragraphs() if report else []
    cited = [p for p in paragraphs if CITATION_MARKER.search(p)]

    produced = bool(run.findings) and bool(report and report.sections)

    metrics = RunMetrics(
        query_id=query.id,
        question=query.question,
        difficulty=query.difficulty,
        produced_report=produced,
        expected_findings=query.expect_findings,
        # A query expected to find little is handled correctly when it finds
        # little. Treating "no report" as a failure everywhere would reward a
        # system that fabricates rather than one that declines.
        behaved_as_expected=(produced == query.expect_findings),
        elapsed_seconds=round(run.elapsed_seconds, 2),
        rounds_run=len(run.rounds),
        sources_found=len(run.sources),
        sources_read=sum(r.sources_read for r in run.rounds),
        sources_unreadable=sum(r.unreadable_sources for r in run.rounds),
        claims_proposed=sum(r.candidates_proposed for r in run.extraction_reports),
        claims_verified=len(run.findings),
        claims_rejected=sum(len(r.rejected) for r in run.extraction_reports),
        distinct_sources_cited=len({c.source_url for c in report.citations}) if report else 0,
        body_paragraphs=len(paragraphs),
        cited_paragraphs=len(cited),
        report_validation_passed=bool(run.writer and run.writer.passed_validation),
        writer_revisions=run.writer.revisions_used if run.writer else 0,
        stop_cause=run.stop_decision.cause if run.stop_decision else "",
    )
    return metrics


class SpotCheckItem(BaseModel):
    """One claim for a human to check against its source."""

    query_id: str
    claim: str
    quoted_evidence: str
    source_url: str


class EvaluationSummary(BaseModel):
    """Aggregate numbers across a whole evaluation."""

    queries_run: int = 0
    behaved_as_expected: int = 0
    reports_produced: int = 0
    validation_passed: int = 0
    total_claims_proposed: int = 0
    total_claims_verified: int = 0
    total_seconds: float = 0.0
    per_query: List[RunMetrics] = Field(default_factory=list)

    @property
    def expected_behaviour_rate(self) -> Optional[float]:
        if not self.queries_run:
            return None
        return self.behaved_as_expected / self.queries_run

    @property
    def overall_verification_rate(self) -> Optional[float]:
        if not self.total_claims_proposed:
            return None
        return self.total_claims_verified / self.total_claims_proposed

    @property
    def mean_seconds(self) -> Optional[float]:
        if not self.queries_run:
            return None
        return self.total_seconds / self.queries_run


def summarise(all_metrics: List[RunMetrics]) -> EvaluationSummary:
    return EvaluationSummary(
        queries_run=len(all_metrics),
        behaved_as_expected=sum(1 for m in all_metrics if m.behaved_as_expected),
        reports_produced=sum(1 for m in all_metrics if m.produced_report),
        validation_passed=sum(1 for m in all_metrics if m.report_validation_passed),
        total_claims_proposed=sum(m.claims_proposed for m in all_metrics),
        total_claims_verified=sum(m.claims_verified for m in all_metrics),
        total_seconds=round(sum(m.elapsed_seconds for m in all_metrics), 2),
        per_query=list(all_metrics),
    )
