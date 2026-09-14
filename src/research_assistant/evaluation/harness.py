"""Running the evaluation and writing it up (Phase 6.1).

Chapter 7.1 describes the method: "pick a handful of test queries spanning
different topics and difficulty levels, run each one through the system, and
separately do a quick manual version of the same research. Comparing the two
side by side, rather than judging the system's output in isolation, makes
weaknesses far easier to spot than they are in the abstract."

The harness automates the first half. The second half is a person's job, and
the point of the spot-check worksheet is to make that job small enough that
it actually gets done: a sampled claim, the exact words the system says
support it, and a link, with somewhere to write down whether it holds up.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ..orchestration.pipeline import ResearchPipeline, ResearchRun
from .metrics import (
    EvaluationSummary,
    RunMetrics,
    SpotCheckItem,
    measure_run,
    summarise,
)
from .queries import TEST_QUERIES, EvalQuery


class EvaluationResult(BaseModel):
    """Everything one evaluation pass produced."""

    summary: EvaluationSummary
    spot_checks: List[SpotCheckItem] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)


async def evaluate(
    pipeline: ResearchPipeline,
    queries: Optional[Sequence[EvalQuery]] = None,
    *,
    spot_checks_per_query: int = 3,
    seed: int = 0,
) -> Tuple[EvaluationResult, List[ResearchRun]]:
    """Run every test query through the pipeline and measure what happened.

    A query that raises is recorded as a failure rather than ending the
    evaluation. An evaluation that stops at the first exception tells you
    about one query; one that finishes tells you about all of them.
    """
    queries = list(queries or TEST_QUERIES)
    rng = random.Random(seed)

    all_metrics: List[RunMetrics] = []
    spot_checks: List[SpotCheckItem] = []
    failures: List[str] = []
    runs: List[ResearchRun] = []

    for query in queries:
        try:
            run = await pipeline.run(query.question)
        except Exception as exc:  # noqa: BLE001 - one bad query must not end the evaluation
            failures.append(f"{query.id}: {type(exc).__name__}: {exc}")
            continue

        runs.append(run)
        all_metrics.append(measure_run(run, query))

        if run.findings:
            sample = rng.sample(
                list(run.findings), min(spot_checks_per_query, len(run.findings))
            )
            spot_checks.extend(
                SpotCheckItem(
                    query_id=query.id,
                    claim=finding.claim,
                    quoted_evidence=finding.supporting_snippet,
                    source_url=finding.source_url,
                )
                for finding in sample
            )

    return (
        EvaluationResult(
            summary=summarise(all_metrics), spot_checks=spot_checks, failures=failures
        ),
        runs,
    )


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render_evaluation_markdown(
    result: EvaluationResult,
    queries: Optional[Sequence[EvalQuery]] = None,
    *,
    note: Optional[str] = None,
) -> str:
    """Write the evaluation up as a document a person can read and act on."""
    queries = list(queries or TEST_QUERIES)
    by_id = {q.id: q for q in queries}
    summary = result.summary

    lines = [
        "# Evaluation results",
        "",
        "Produced by `research-assistant evaluate`. Every number below is mechanical. "
        "The two measures that matter most, citation accuracy and finding relevance, are "
        "judgements and appear at the end as a worksheet rather than as a score.",
        "",
    ]
    if note:
        lines.extend([f"> **{note}**", ""])
    lines.extend([
        "## Headline",
        "",
        f"- Queries run: **{summary.queries_run}**",
        f"- Behaved as expected: **{summary.behaved_as_expected} of {summary.queries_run}** "
        f"({_percent(summary.expected_behaviour_rate)})",
        f"- Reports produced: **{summary.reports_produced}**",
        f"- Reports passing validation: **{summary.validation_passed}**",
        f"- Claims proposed by extraction: **{summary.total_claims_proposed}**; "
        f"verified: **{summary.total_claims_verified}** "
        f"({_percent(summary.overall_verification_rate)})",
    ])
    if summary.mean_seconds is not None:
        lines.append(f"- Mean time per query: **{summary.mean_seconds:.1f}s**")
    lines.append("")

    lines.extend([
        "\"Behaved as expected\" is not the same as \"produced a report.\" Two of the test "
        "queries are deliberately about things that are barely studied or do not exist. For "
        "those, producing no report is the correct outcome, and producing a confident one "
        "would be the worst possible result.",
        "",
        "## Per query",
        "",
        "| Query | Difficulty | As expected | Findings | Verified / proposed | Cited paras | "
        "Validation | Time | Stopped because |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for metrics in summary.per_query:
        verified = (
            f"{metrics.claims_verified}/{metrics.claims_proposed}"
            if metrics.claims_proposed
            else "-"
        )
        cited = (
            f"{metrics.cited_paragraphs}/{metrics.body_paragraphs}"
            if metrics.body_paragraphs
            else "-"
        )
        lines.append(
            f"| {metrics.query_id} | {metrics.difficulty} | "
            f"{'yes' if metrics.behaved_as_expected else '**NO**'} | "
            f"{metrics.claims_verified} | {verified} | {cited} | "
            f"{'pass' if metrics.report_validation_passed else 'fail'} | "
            f"{metrics.elapsed_seconds:.1f}s | {metrics.stop_cause} |"
        )

    lines.extend(["", "### What each query is testing", ""])
    for metrics in summary.per_query:
        query = by_id.get(metrics.query_id)
        if query:
            lines.append(f"- **{query.id}** — {query.tests}")
    lines.append("")

    if result.failures:
        lines.extend(["## Queries that raised an error", ""])
        lines.extend(f"- {failure}" for failure in result.failures)
        lines.append("")

    lines.extend([
        "## Spot-check worksheet",
        "",
        "Citation accuracy cannot be measured by the system that produced the citations. "
        "Open each source below, find the quoted text, and decide whether the claim is a fair "
        "reading of it. Mark each one and count the results: that count is the citation "
        "accuracy figure.",
        "",
        "Judge three things separately: does the quotation appear in the source; does it say "
        "what the claim says it says; and does the claim omit context that changes its "
        "meaning. The third is the one the system cannot check for itself.",
        "",
    ])

    if not result.spot_checks:
        lines.append("_No findings were produced, so there is nothing to spot-check._")
    for index, item in enumerate(result.spot_checks, start=1):
        lines.extend([
            f"### {index}. `{item.query_id}`",
            "",
            f"**Claim.** {item.claim}",
            "",
            f"**Quoted as evidence.** \"{item.quoted_evidence}\"",
            "",
            f"**Source.** <{item.source_url}>",
            "",
            "- [ ] The quotation appears in the source",
            "- [ ] The quotation supports the claim",
            "- [ ] The claim omits no context that changes its meaning",
            "",
        ])

    lines.extend([
        "## Comparison against doing it by hand",
        "",
        "Chapter 7.1 asks for the system's time to be compared against a careful person "
        "answering the same question, holding depth roughly constant. That number is not here "
        "because it cannot be automated: it requires someone to actually do the research. The "
        "per-query times above are one half of that comparison; the other half has to be "
        "measured by hand, once, and written down.",
        "",
    ])

    return "\n".join(lines).rstrip() + "\n"
