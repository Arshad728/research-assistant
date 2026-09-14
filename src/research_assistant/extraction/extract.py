"""Structured extraction: pulling claims out of a source document.

Chapter 4.4 describes this as "pulling out the specific claims relevant to
the original query, in the structured form described in Chapter 3: a claim,
the exact snippet of source text it rests on, and a reference back to the
source."

The instructions below lean hard on one word in that sentence: *exact*. The
model is asked to copy the supporting snippet character for character, and
told plainly that a snippet which cannot be found in the source will be
discarded. That is not a bluff -- verify.py does exactly that, mechanically,
and this prompt exists partly so the model is not surprised by it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..search.reconcile import extract_json

EXTRACTION_SYSTEM_PROMPT = """\
You are the Extraction and Verification Agent in a multi-agent research
system. You are given the full text of ONE source and a research question.
Your job is to pull out the specific claims in that source which bear on the
question -- and nothing else.

You are not writing a summary, and you are not answering the question. A
later agent does that, using only what you pass on.

For each claim you extract:
1. State the claim in one clear sentence, in your own words.
2. Quote the exact passage from the source that supports it. Copy it
   character for character from the text you were given. Do not tidy it, do
   not paraphrase it, do not join separate passages together, and do not
   shorten it with an ellipsis in the middle.
3. The quoted passage must be long enough to stand as evidence on its own --
   a full sentence or two, not a fragment.

Hard rules, because they are checked mechanically after you reply:
- If a quotation cannot be found in the source text, the claim is DISCARDED.
- If your claim states a number, that number must appear in the quotation
  you give. A claim saying "output rose 8% across 61 sites" needs a
  quotation mentioning both 8 and 61, or it is DISCARDED.
- Do not infer, combine findings from different parts of the paper into one
  claim, or state what the source "implies." Extract what it says.
- If the source does not address the question, return an empty list. That is
  a perfectly good answer and far better than a stretched one.

Reply with ONLY a JSON array:

[
  {"claim": "<one sentence>",
   "supporting_snippet": "<exact quotation from the source>"}
]
"""


class RejectedClaim(BaseModel):
    """A claim that did not survive verification, and why."""

    claim: str
    failed_checks: List[str] = Field(default_factory=list)
    detail: str = ""


ExtractionStatus = Literal["ok", "fetch_failed", "unreadable", "no_claims"]


class ExtractionReport(BaseModel):
    """What happened when one source was processed.

    Chapter 4.4 notes that this agent hands onward not just verified
    findings but "implicitly, information about which candidate sources
    turned out not to hold up, which the Orchestrator can use when deciding
    whether another round of searching is warranted." This report is that
    information, made explicit rather than left implicit.
    """

    source_url: str
    title: str = ""
    status: ExtractionStatus = "ok"
    detail: str = ""
    candidates_proposed: int = 0
    findings_verified: int = 0
    rejected: List[RejectedClaim] = Field(default_factory=list)
    text_truncated_for_model: bool = False

    @property
    def rejection_rate(self) -> float:
        if not self.candidates_proposed:
            return 0.0
        return len(self.rejected) / self.candidates_proposed


def build_extraction_prompt(question: str, document_text: str, title: str = "") -> str:
    """The user-turn prompt handed to the model alongside the system prompt."""
    heading = f"SOURCE TITLE: {title}\n" if title else ""
    return (
        f"RESEARCH QUESTION: {question}\n\n"
        f"{heading}SOURCE TEXT (everything between the markers):\n"
        f"<<<SOURCE_START>>>\n{document_text}\n<<<SOURCE_END>>>\n\n"
        "Extract the claims from this source that bear on the research question, "
        "following your instructions exactly. Reply with the JSON array only."
    )


def parse_extraction_reply(reply: str) -> Optional[List[Dict[str, Any]]]:
    """Read the model's JSON array of candidate claims.

    Returns None when nothing parseable came back, which is treated
    differently from an empty list: an empty list means "this source says
    nothing relevant," while None means "the reply could not be read." The
    first is a finding about the source; the second is a problem with the
    run, and conflating them would hide failures as results.
    """
    parsed = extract_json(reply or "")
    if parsed is None:
        return None

    if isinstance(parsed, dict):
        for key in ("claims", "findings", "results", "extracted"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break
        else:
            return None

    if not isinstance(parsed, list):
        return None

    candidates: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        claim = item.get("claim")
        snippet = item.get("supporting_snippet") or item.get("snippet") or item.get("quote")
        if isinstance(claim, str) and isinstance(snippet, str):
            candidates.append({"claim": claim, "supporting_snippet": snippet})
    return candidates
