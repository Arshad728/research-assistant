"""Evaluating the system against test queries (Phase 6.1)."""

from .harness import EvaluationResult, evaluate, render_evaluation_markdown
from .metrics import (
    EvaluationSummary,
    RunMetrics,
    SpotCheckItem,
    measure_run,
    summarise,
)
from .queries import TEST_QUERIES, EvalQuery, query_by_id

__all__ = [
    "EvaluationResult",
    "EvaluationSummary",
    "RunMetrics",
    "SpotCheckItem",
    "TEST_QUERIES",
    "EvalQuery",
    "evaluate",
    "measure_run",
    "query_by_id",
    "render_evaluation_markdown",
    "summarise",
]
