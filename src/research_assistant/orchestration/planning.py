"""Breaking a question into search angles (Phase 5.1).

Chapter 4.2 gives planning as the Orchestrator's first job: "when a query
arrives, the Orchestrator breaks it into the sub-tasks the other agents
will need to execute. A broad query like 'summarize current research on
four-day work weeks' might be split into two or three narrower search
angles, since a single search is unlikely to surface the full picture."

Planning is judgement, so a model does it. But the pipeline must still run
when the model returns something unusable, so a deterministic fallback
exists: use the question itself, then progressively broader versions of it
produced by the Phase 2.2 planner. A research run that degrades to "search
the question as asked" is worse than one with well-chosen angles, and far
better than one that fails.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..search.planner import SearchPlanner
from ..search.reconcile import extract_json

PLANNING_SYSTEM_PROMPT = """\
You are the Orchestrator of a multi-agent research system. Your job right now
is only to plan: given a research question, decide what separate searches
would be needed to answer it properly.

Produce between 2 and 4 search angles. A good set of angles:
- Covers genuinely different aspects of the question, not rephrasings of one
  aspect. "four day week productivity trials" and "productivity four day
  week studies" are the same angle twice.
- Includes the counter-case where one exists. If the question asks whether
  something works, one angle should look for evidence that it does not.
- Uses the vocabulary the literature would use, not the vocabulary of the
  question. A researcher searching for this would not type a full sentence.

Reply with ONLY a JSON array of strings:

["first search angle", "second search angle", "third search angle"]
"""


def build_planning_prompt(question: str) -> str:
    return (
        f"RESEARCH QUESTION: {question}\n\n"
        "Plan the searches needed to answer this properly. Reply with the JSON array only."
    )


def parse_planning_reply(reply: str, *, max_angles: int = 4) -> Optional[List[str]]:
    """Read the planner's reply into a list of search angles."""
    parsed = extract_json(reply or "")
    if isinstance(parsed, dict):
        for key in ("angles", "searches", "queries", "plan"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break
    if not isinstance(parsed, list):
        return None

    angles = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
    if not angles:
        return None

    # Drop near-duplicates using the same unordered-keyword comparison the
    # Search Agent uses to refuse a repeated query.
    unique: List[str] = []
    seen = set()
    for angle in angles:
        fingerprint = SearchPlanner._fingerprint(angle)
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(angle)

    return unique[:max_angles] or None


def fallback_angles(question: str, count: int = 3) -> List[str]:
    """Angles to use when the model gives nothing usable.

    The question itself, then progressively broader versions of it. Not
    clever, but it keeps a run moving and never repeats a query.
    """
    planner = SearchPlanner(original_query=question)
    angles = [question]
    broadened = planner.broaden(question, keep=4)
    if broadened and not planner._fingerprint(broadened) == planner._fingerprint(question):
        angles.append(broadened)
    narrower = planner.broaden(question, keep=2)
    if narrower and planner._fingerprint(narrower) not in {
        planner._fingerprint(a) for a in angles
    }:
        angles.append(narrower)
    return angles[:count]


def next_unused_angle(angles: Sequence[str], already_tried: Sequence[str]) -> Optional[str]:
    """The first planned angle that has not effectively been searched already."""
    tried = {SearchPlanner._fingerprint(a) for a in already_tried}
    for angle in angles:
        fingerprint = SearchPlanner._fingerprint(angle)
        if fingerprint and fingerprint not in tried:
            return angle
    return None
