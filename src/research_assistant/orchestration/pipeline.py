"""The Orchestrator: one question in, one cited report out (Phases 5.1 and 5.3).

Chapter 4.2 gives the Orchestrator three jobs -- planning, sequencing and
stopping -- and Chapter 4.6 adds a fourth that is easy to overlook: every
hand-off passes through it, "which keeps the Orchestrator's shared state,
the running record of what has been searched, extracted, and written so far,
always up to date."

That shared state is the ``SharedState`` object designed back in Phase 1.3,
used here for the first time and unchanged since. Its one piece of real
logic, counting how many distinct sources the verified findings draw from,
is what the stopping rule now runs on.

Sequencing and stopping live in this file as ordinary control flow rather
than as an agent's decisions. Decision Record 0002 explains that choice.
The short version: a research system's value is that its output can be
trusted, and a pipeline whose control flow is itself a model's improvisation
cannot be reproduced, tested, or explained when it goes wrong.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..agents.extraction_agent import ExtractionAgent
from ..agents.search_agent import SearchAgent
from ..agents.writer_agent import WriterAgent, WriterOutcome
from ..extraction.extract import ExtractionReport
from ..report.models import ResearchReport
from ..schemas import CandidateSource, SharedState, VerifiedFinding
from ..search.planner import normalize_url
from .planning import (
    PLANNING_SYSTEM_PROMPT,
    build_planning_prompt,
    fallback_angles,
    next_unused_angle,
    parse_planning_reply,
)
from .stopping import StopDecision, StoppingRule

CompletionFn = Callable[[str, str], Awaitable[str]]


class RoundRecord(BaseModel):
    """What one search-extract cycle did."""

    number: int
    angle: str
    sources_found: int = 0
    sources_new: int = 0
    sources_read: int = 0
    findings_added: int = 0
    claims_rejected: int = 0
    unreadable_sources: int = 0
    stop_decision: Optional[StopDecision] = None


class ResearchRun(BaseModel):
    """Everything one end-to-end run produced, and how it got there."""

    question: str
    angles_planned: List[str] = Field(default_factory=list)
    used_fallback_plan: bool = False
    rounds: List[RoundRecord] = Field(default_factory=list)
    findings: List[VerifiedFinding] = Field(default_factory=list)
    sources: List[CandidateSource] = Field(default_factory=list)
    extraction_reports: List[ExtractionReport] = Field(default_factory=list)
    stop_decision: Optional[StopDecision] = None
    report: Optional[ResearchReport] = None
    writer: Optional[WriterOutcome] = None
    elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        """A run succeeded if it produced a report with evidence behind it."""
        return bool(self.findings) and self.report is not None and bool(self.report.sections)

    def summary(self) -> str:
        lines = [
            f'Question: "{self.question}"',
            f"Plan: {len(self.angles_planned)} search angle(s)"
            + (" (fallback plan)" if self.used_fallback_plan else ""),
            f"Rounds run: {len(self.rounds)}",
            f"Sources found: {len(self.sources)}; "
            f"sources read: {sum(r.sources_read for r in self.rounds)}",
            f"Verified findings: {len(self.findings)} from "
            f"{len({f.source_url for f in self.findings})} source(s)",
            f"Claims rejected in verification: {sum(r.claims_rejected for r in self.rounds)}",
        ]
        if self.stop_decision:
            lines.append(f"Stopped because: {self.stop_decision.reason}")
        if self.writer:
            lines.append(
                "Report validation: "
                + ("passed" if self.writer.passed_validation else "FAILED")
                + f" ({self.writer.revisions_used} revision(s))"
            )
        lines.append(f"Elapsed: {self.elapsed_seconds:.1f}s")
        return "\n".join(lines)


class ResearchPipeline:
    """Runs the whole system: plan, search, extract, verify, write.

    Every agent is injected, so the entire control flow -- including the
    stopping rule and the hand-offs between agents -- can be exercised with
    stand-ins that need no API key and no network.
    """

    def __init__(
        self,
        *,
        search_agent: Optional[SearchAgent] = None,
        extraction_agent: Optional[ExtractionAgent] = None,
        writer_agent: Optional[WriterAgent] = None,
        stopping_rule: Optional[StoppingRule] = None,
        complete: Optional[CompletionFn] = None,
        model: Optional[str] = None,
        provider: str = "claude",
        max_sources_per_round: int = 5,
    ) -> None:
        # Planning (below), Search, Extraction and Writer all accept the same
        # ``provider`` and default their own model calls from it, so one
        # switch moves every model call in the pipeline together.
        self.search_agent = search_agent or SearchAgent(provider=provider)
        self.extraction_agent = extraction_agent or ExtractionAgent(model=model, provider=provider)
        self.writer_agent = writer_agent or WriterAgent(model=model, provider=provider)
        self.stopping_rule = stopping_rule or StoppingRule()
        self.provider = provider
        self.max_sources_per_round = max_sources_per_round

        if complete is not None:
            self._complete = complete
        else:
            from ..agents.extraction_agent import get_completion_fn

            self._complete = get_completion_fn(provider, model)

    async def plan(self, question: str) -> tuple[List[str], bool]:
        """Decide which searches to run. Returns the angles and whether they are a fallback."""
        try:
            reply = await self._complete(PLANNING_SYSTEM_PROMPT, build_planning_prompt(question))
            angles = parse_planning_reply(reply)
        except Exception:  # noqa: BLE001 - planning must never sink the run
            angles = None

        if angles:
            return angles, False
        return fallback_angles(question), True

    async def run(self, question: str) -> ResearchRun:
        """Answer one research question end to end."""
        started = time.monotonic()
        angles, used_fallback = await self.plan(question)

        run = ResearchRun(
            question=question, angles_planned=list(angles), used_fallback_plan=used_fallback
        )
        state = SharedState(original_query=question)
        seen_urls: set = set()
        tried_angles: List[str] = []

        for round_number in range(1, self.stopping_rule.max_rounds + 1):
            angle = next_unused_angle(angles, tried_angles)
            if angle is None:
                # The plan is exhausted; fall back to a broader version rather
                # than re-running something already tried.
                remaining = [a for a in fallback_angles(question, count=4) if a not in tried_angles]
                angle = next_unused_angle(remaining, tried_angles)
            if angle is None:
                break
            tried_angles.append(angle)

            record = RoundRecord(number=round_number, angle=angle)

            outcome = await self.search_agent.find_sources(angle)
            record.sources_found = len(outcome.sources)

            fresh = [s for s in outcome.sources if normalize_url(s.source_url) not in seen_urls]
            for source in fresh:
                seen_urls.add(normalize_url(source.source_url))
            record.sources_new = len(fresh)

            to_read = fresh[: self.max_sources_per_round]
            record.sources_read = len(to_read)

            findings, reports = await self.extraction_agent.extract_from_sources(to_read, question)
            record.findings_added = len(findings)
            record.claims_rejected = sum(len(r.rejected) for r in reports)
            record.unreadable_sources = sum(
                1 for r in reports if r.status in ("fetch_failed", "unreadable")
            )

            state.add_search_round(fresh)
            state.add_verified_findings(findings)
            run.extraction_reports.extend(reports)

            decision = self.stopping_rule.evaluate(
                state, round_number, new_sources_last_round=record.sources_new
            )
            record.stop_decision = decision
            run.rounds.append(record)
            run.stop_decision = decision

            if decision.should_stop:
                break

        run.findings = list(state.verified_findings)
        run.sources = list(state.candidate_sources)

        writer_outcome = await self.writer_agent.write_report(question, run.findings, run.sources)
        run.writer = writer_outcome
        run.report = writer_outcome.report

        # The reader should be told when a report rests on thin evidence,
        # not left to infer it from a short reference list.
        if run.stop_decision and run.stop_decision.cause in ("round_limit", "no_new_sources"):
            run.report.caveats.append(
                f"evidence_gathering: {run.stop_decision.reason}"
            )

        run.elapsed_seconds = time.monotonic() - started
        return run
