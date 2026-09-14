"""Cross-checking the Search Agent's shortlist against what was really retrieved.

The Search Agent does two separable things: it decides which retrieved
sources are worth passing on (judgement, which needs a language model), and
it reports those sources (data, which does not). This module keeps the
second from depending on the first.

Every source in the final shortlist is rebuilt from the record of what the
tools actually returned. The model's contribution is which URLs to keep and
why; its words are never trusted for a title or a link. If the model names a
URL that no tool ever returned -- the search-layer version of the fabricated
citation problem in Chapter 1.5 -- that entry is dropped and counted, rather
than passed downstream where a later agent would have no way to know it was
never real.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..schemas import CandidateSource
from .planner import normalize_url

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


class Shortlist(BaseModel):
    """The reconciled result of one Search Agent run."""

    sources: List[CandidateSource] = Field(default_factory=list)
    dropped_unknown_urls: List[str] = Field(default_factory=list)
    parse_error: Optional[str] = None

    @property
    def used_fallback(self) -> bool:
        """True when the model's output could not be used and everything was kept."""
        return self.parse_error is not None


def extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON value out of a model's reply.

    Models wrap JSON in prose or fences often enough that being strict here
    would fail for cosmetic reasons. This tries the whole string, then any
    fenced block, then the widest bracketed span it can find.
    """
    if not text:
        return None

    candidates: List[str] = [text.strip()]
    candidates.extend(match.strip() for match in _JSON_FENCE.findall(text))

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _as_entry_list(parsed: Any) -> Optional[List[Dict[str, Any]]]:
    """Accept either a bare list or a dict wrapping one under a familiar key."""
    if isinstance(parsed, list):
        entries = parsed
    elif isinstance(parsed, dict):
        for key in ("sources", "selected", "shortlist", "results"):
            value = parsed.get(key)
            if isinstance(value, list):
                entries = value
                break
        else:
            return None
    else:
        return None
    return [entry for entry in entries if isinstance(entry, dict)]


def reconcile_shortlist(
    model_text: str,
    retrieved: Sequence[CandidateSource],
    *,
    max_items: Optional[int] = None,
) -> Shortlist:
    """Build the final shortlist from the model's selection and the real results.

    If the model's output cannot be parsed at all, everything retrieved is
    returned instead, with ``parse_error`` set. Degrading to "unfiltered but
    real" is the right failure mode here: the next agent can still work, and
    the caller can see that no filtering happened.
    """
    by_url = {normalize_url(source.source_url): source for source in retrieved}

    parsed = extract_json(model_text)
    entries = _as_entry_list(parsed) if parsed is not None else None

    if entries is None:
        return Shortlist(
            sources=list(retrieved)[:max_items] if max_items else list(retrieved),
            parse_error=(
                "Could not read a JSON shortlist from the agent's reply; returning every "
                "retrieved source unfiltered."
            ),
        )

    sources: List[CandidateSource] = []
    dropped: List[str] = []
    already_taken = set()

    for entry in entries:
        raw_url = entry.get("source_url") or entry.get("url") or ""
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        key = normalize_url(raw_url)

        match = by_url.get(key)
        if match is None:
            dropped.append(raw_url)
            continue
        if key in already_taken:
            continue
        already_taken.add(key)

        note = entry.get("relevance_note") or entry.get("note") or ""
        sources.append(
            CandidateSource(
                # Title, link and type always come from the retrieved record,
                # never from the model's reply.
                title=match.title,
                source_url=match.source_url,
                source_type=match.source_type,
                relevance_note=note.strip() if isinstance(note, str) and note.strip() else match.relevance_note,
            )
        )
        if max_items is not None and len(sources) >= max_items:
            break

    return Shortlist(sources=sources, dropped_unknown_urls=dropped)
