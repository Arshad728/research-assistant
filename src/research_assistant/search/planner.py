"""Query planning: the logic that makes the Search Agent iterate.

Chapter 4.3 sets the requirement plainly: if a first search comes back thin,
the Search Agent "is expected to reformulate rather than repeat: trying a
different phrasing, a different combination of keywords, or a different one
of its three retrieval tools, rather than resubmitting the same query and
expecting a different result."

That requirement splits into two very different kinds of work. Judging which
*rephrasing* is likely to work better is language work, and belongs to the
language model. Remembering what has already been tried, noticing that a
result set is thin, and refusing to run the same query twice are bookkeeping,
and belong in ordinary code where they behave identically every single run.
This module is the second half. Decision Record 0001 made the same split at
the framework level: determinism comes from the architecture, not from hoping
the model is consistent.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Literal, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from ..schemas import CandidateSource

Verdict = Literal["adequate", "thin", "off_topic"]

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "there", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "about", "into", "over", "under", "effect", "effects",
    "research", "study", "studies", "evidence", "current", "recent", "say", "says",
}

# Modern arXiv identifiers (2401.01234) and the pre-2007 scheme (cs/0301001),
# both optionally carrying a version suffix.
_ARXIV_ID_PATTERN = re.compile(
    r"/(?:abs|pdf)/((?:[0-9]{4}\.[0-9]{4,5})|(?:[a-z-]+(?:\.[a-z]{2})?/[0-9]{7}))(v[0-9]+)?",
    re.IGNORECASE,
)


def keywords(text: str) -> set:
    """The meaningful words in a piece of text, lowercased.

    Deliberately crude: this exists to catch gross mismatches between a query
    and what came back, not to understand meaning. Chapter 5.5's embeddings
    would do that better, and are noted there as an optional enhancement
    rather than something the first working version needs.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}


def normalize_url(url: str) -> str:
    """Reduce a URL to a comparable form, so the same source is not counted twice.

    Two things matter in practice. First, the same paper reaches this system
    by several routes -- an arXiv abstract page, an arXiv PDF link, and a
    Semantic Scholar record can all point at one piece of work -- so arXiv
    identifiers are canonicalised, version suffix included. Second, trivial
    differences (scheme, "www.", a trailing slash, a fragment) should never
    make one source look like two.
    """
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()

    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")

    if host.endswith("arxiv.org"):
        match = _ARXIV_ID_PATTERN.search(path)
        if match:
            return f"arxiv:{match.group(1)}"

    return urlunsplit(("https", host, path, parts.query, ""))


def deduplicate(sources: Iterable[CandidateSource]) -> List[CandidateSource]:
    """Keep the first occurrence of each distinct source, preserving order."""
    seen = set()
    unique: List[CandidateSource] = []
    for source in sources:
        key = normalize_url(source.source_url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def distinct_domains(sources: Sequence[CandidateSource]) -> int:
    """How many different sites/repositories the results come from."""
    domains = set()
    for source in sources:
        key = normalize_url(source.source_url)
        if key.startswith("arxiv:"):
            domains.add("arxiv.org")
            continue
        host = urlsplit(key).hostname
        if host:
            domains.add(host)
    return len(domains)


class ResultAssessment(BaseModel):
    """The planner's read on one round of results, and why."""

    verdict: Verdict
    reason: str
    result_count: int
    distinct_domains: int
    keyword_overlap: float


class QueryAttempt(BaseModel):
    """One query that was actually run, and what it produced."""

    query: str
    tool: str
    result_count: int
    assessment: Optional[ResultAssessment] = None


class SearchPlanner(BaseModel):
    """Tracks what has been tried and decides whether to keep going.

    The planner never calls an API itself. It is handed the results of each
    round and answers three questions: was that good enough, has this query
    already been tried, and is there any round left to spend.
    """

    original_query: str
    max_rounds: int = 3
    min_results: int = 4
    min_distinct_domains: int = 2
    min_keyword_overlap: float = 0.15

    attempts: List[QueryAttempt] = Field(default_factory=list)
    collected: List[CandidateSource] = Field(default_factory=list)

    # -- memory ---------------------------------------------------------

    @staticmethod
    def _fingerprint(query: str) -> frozenset:
        """A query's identity for repeat-detection: its meaningful words, unordered.

        This deliberately treats "productivity four-day week" and "four day
        week productivity" as the same query, because they are: running the
        second after the first is the exact behaviour Chapter 4.3 rules out.
        """
        return frozenset(keywords(query))

    def is_repeat(self, query: str) -> bool:
        fingerprint = self._fingerprint(query)
        if not fingerprint:
            return True  # an empty or all-stopword query is never worth running
        return any(self._fingerprint(a.query) == fingerprint for a in self.attempts)

    def tried_queries(self) -> List[str]:
        return [attempt.query for attempt in self.attempts]

    # -- judgement ------------------------------------------------------

    def assess(self, results: Sequence[CandidateSource]) -> ResultAssessment:
        count = len(results)
        domains = distinct_domains(results)
        overlap = self._overlap_with_query(results)

        if count == 0:
            verdict: Verdict = "thin"
            reason = "No results at all. The query may be too narrow or too specific."
        elif count < self.min_results:
            verdict = "thin"
            reason = (
                f"Only {count} result(s); at least {self.min_results} are wanted before "
                "drawing any conclusion. Try broadening the query."
            )
        elif domains < self.min_distinct_domains:
            verdict = "thin"
            reason = (
                f"{count} results but all from {domains} source(s). Evidence from a single "
                "place is not corroboration; try a different tool or phrasing."
            )
        elif overlap < self.min_keyword_overlap:
            verdict = "off_topic"
            reason = (
                f"{count} results, but they share little vocabulary with the original "
                f"question (overlap {overlap:.0%}). They may be about something else."
            )
        else:
            verdict = "adequate"
            reason = (
                f"{count} results from {domains} distinct sources, on topic "
                f"(overlap {overlap:.0%})."
            )

        return ResultAssessment(
            verdict=verdict,
            reason=reason,
            result_count=count,
            distinct_domains=domains,
            keyword_overlap=round(overlap, 3),
        )

    def _overlap_with_query(self, results: Sequence[CandidateSource]) -> float:
        query_words = keywords(self.original_query)
        if not query_words or not results:
            return 0.0
        scores = []
        for source in results:
            text_words = keywords(f"{source.title} {source.relevance_note}")
            scores.append(len(query_words & text_words) / len(query_words))
        return sum(scores) / len(scores)

    # -- progress -------------------------------------------------------

    def register_attempt(
        self, query: str, tool: str, results: Sequence[CandidateSource]
    ) -> ResultAssessment:
        """Record one round of searching and return the verdict on it."""
        assessment = self.assess(results)
        self.attempts.append(
            QueryAttempt(
                query=query,
                tool=tool,
                result_count=len(results),
                assessment=assessment,
            )
        )
        self.collected = deduplicate([*self.collected, *results])
        return assessment

    @property
    def rounds_used(self) -> int:
        return len(self.attempts)

    def should_continue(self) -> bool:
        """Is another round both permitted and worth running?"""
        if self.rounds_used >= self.max_rounds:
            return False
        if not self.attempts:
            return True
        return self.assess(self.collected).verdict != "adequate"

    def stop_reason(self) -> str:
        if self.rounds_used >= self.max_rounds:
            return f"Reached the {self.max_rounds}-round limit."
        overall = self.assess(self.collected)
        if overall.verdict == "adequate":
            return f"Enough material gathered: {overall.reason}"
        return "Still searching."

    # -- deterministic fallback reformulation ---------------------------

    def broaden(self, query: str, keep: int = 3) -> str:
        """Reduce a query to fewer, more distinctive terms.

        Fewer required terms means more matches, which is what broadening
        means here. Which terms to keep matters though: an early version of
        this kept the *shortest* words, and turned "effect of a four day work
        week on productivity" into "day four work" -- broader, but no longer
        about productivity at all. Keeping the longest words instead retains
        the distinctive ones, so the query gets wider without drifting off
        the subject.

        This is a fallback, not the main path: the language model is expected
        to produce better reformulations than this. It exists so the loop
        still makes progress if the model returns something unusable, and so
        the broadening behaviour can be tested without a model.
        """
        stripped = re.sub(r'"[^"]*"', " ", query)              # drop quoted phrases
        stripped = re.sub(r"\b(19|20)\d{2}\b", " ", stripped)  # drop year filters
        words = [w for w in re.split(r"\s+", stripped.strip()) if w]
        meaningful = [w for w in words if w.lower().strip(".,;:?!") not in _STOPWORDS]
        if len(meaningful) > keep:
            most_distinctive = set(sorted(meaningful, key=len, reverse=True)[:keep])
            meaningful = [w for w in meaningful if w in most_distinctive]
        return " ".join(meaningful) if meaningful else self.original_query

    def suggest_next_query(self) -> Optional[str]:
        """A fallback query to try next, or None if nothing new is left."""
        last = self.attempts[-1].query if self.attempts else self.original_query
        for candidate in (self.broaden(last), self.broaden(self.original_query)):
            if candidate and not self.is_repeat(candidate):
                return candidate
        return None
