"""The Orchestrator: planning, sequencing, stopping (Phase 5)."""

from .pipeline import ResearchPipeline, ResearchRun, RoundRecord
from .planning import (
    PLANNING_SYSTEM_PROMPT,
    build_planning_prompt,
    fallback_angles,
    next_unused_angle,
    parse_planning_reply,
)
from .stopping import StopDecision, StoppingRule

__all__ = [
    "PLANNING_SYSTEM_PROMPT",
    "ResearchPipeline",
    "ResearchRun",
    "RoundRecord",
    "StopDecision",
    "StoppingRule",
    "build_planning_prompt",
    "fallback_angles",
    "next_unused_angle",
    "parse_planning_reply",
]
