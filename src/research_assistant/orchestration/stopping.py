"""When to stop searching (Phase 5.2).

Chapter 4.2 calls this "one of the harder problems in the whole system.
Search indefinitely and the system never finishes; stop too early and the
report is thin or one-sided." Chapter 6.6 proposes a starting rule:
"stopping once a minimum number of verified findings from at least two
independent sources has been reached," and says to treat it as something to
tune rather than get perfect first time.

That rule is implemented here, with two additions that only became obvious
once the pipeline existed to run it.

The first is a stagnation check. The book's rule says when there is *enough*
evidence. It does not say what to do when a round adds nothing at all --
and a search that keeps returning sources already seen will otherwise burn
every remaining round discovering the same papers. Stopping when a round
adds no new source is not giving up early; it is noticing that the query
has been exhausted and that further rounds would cost money to learn
nothing.

The second is that a run which reaches its round limit with nothing at all
is reported differently from one that reaches it with partial evidence.
Both stop. Only one of them should produce a report.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from ..schemas import SharedState

StopCause = Literal[
    "sufficient_evidence",
    "round_limit",
    "no_new_sources",
    "no_evidence",
    "continue",
]


class StopDecision(BaseModel):
    """Whether to stop, why, and what the evidence looks like right now."""

    should_stop: bool
    cause: StopCause
    reason: str
    findings_count: int
    distinct_sources: int
    rounds_used: int

    @property
    def has_usable_evidence(self) -> bool:
        return self.findings_count > 0


class StoppingRule(BaseModel):
    """The Orchestrator's stopping policy, in one place where it can be tuned.

    Every threshold here is a judgement call rather than a discovered truth.
    They are grouped into one object precisely so that changing the
    system's appetite for evidence is a matter of changing a number, not of
    hunting through the control flow for the condition that governs it.
    """

    min_findings: int = 4
    min_distinct_sources: int = 2
    max_rounds: int = 3

    def evaluate(
        self,
        state: SharedState,
        rounds_used: int,
        *,
        new_sources_last_round: Optional[int] = None,
    ) -> StopDecision:
        findings = len(state.verified_findings)
        sources = state.distinct_sources_backing_findings()

        def decide(should_stop: bool, cause: StopCause, reason: str) -> StopDecision:
            return StopDecision(
                should_stop=should_stop,
                cause=cause,
                reason=reason,
                findings_count=findings,
                distinct_sources=sources,
                rounds_used=rounds_used,
            )

        if findings >= self.min_findings and sources >= self.min_distinct_sources:
            return decide(
                True,
                "sufficient_evidence",
                f"{findings} verified findings from {sources} independent sources meets the "
                f"threshold of {self.min_findings} findings from {self.min_distinct_sources} "
                "sources.",
            )

        hit_round_limit = rounds_used >= self.max_rounds
        stagnated = new_sources_last_round == 0 and rounds_used > 0

        # When the run is ending with nothing verified, that is the fact the
        # caller most needs, whichever condition happened to end it. Reporting
        # "the last round found no new sources" would be true and would bury
        # the more important "and nothing at all was verifiable."
        if (hit_round_limit or stagnated) and findings == 0:
            trailing = (
                "The last round also found no sources that had not already been seen."
                if stagnated
                else ""
            )
            return decide(
                True,
                "no_evidence",
                f"Stopped after {rounds_used} round(s) with no verified findings at all. "
                "Either the sources found did not address the question, or nothing they "
                f"claimed could be verified against them. {trailing}".strip(),
            )

        if hit_round_limit:
            return decide(
                True,
                "round_limit",
                f"Stopped at the {self.max_rounds}-round limit with {findings} finding(s) from "
                f"{sources} source(s), short of the {self.min_findings}-from-"
                f"{self.min_distinct_sources} target. The report will be written from thinner "
                "evidence than intended and should say so.",
            )

        if stagnated:
            return decide(
                True,
                "no_new_sources",
                f"The last round found no sources that had not already been seen, so further "
                f"rounds would repeat work. Stopping with {findings} finding(s) from "
                f"{sources} source(s).",
            )

        shortfall = []
        if findings < self.min_findings:
            shortfall.append(f"{findings} of {self.min_findings} findings")
        if sources < self.min_distinct_sources:
            shortfall.append(f"{sources} of {self.min_distinct_sources} independent sources")
        return decide(
            False,
            "continue",
            "Not enough evidence yet: " + ", ".join(shortfall) + ". Searching again.",
        )
