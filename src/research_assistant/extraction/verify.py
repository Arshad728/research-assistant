"""Verification: checking a claim against the source it supposedly came from.

This is the single most important module in the project. Chapter 4.4 calls
it "the single most important defense this whole architecture has against
the hallucination problem," and Chapter 6.4 gives explicit advice about how
to build it: "build 3.3 to be strict rather than lenient in its first
version. It is much easier to loosen an overly cautious verification check
later, once you can see it discarding things that were actually fine, than
to discover after the fact that a lax check let fabricated claims through
into a report that looked entirely trustworthy."

That advice is followed here. Every check below is required, and a claim
that fails any of them does not become a VerifiedFinding at all.

What makes this layer valuable is that none of it asks a language model
whether the model was telling the truth. These are mechanical checks on
text. A model that invents a quotation cannot talk its way past a substring
search, and a model that attaches a real quotation to a claim containing a
statistic the quotation does not mention cannot talk its way past a number
comparison.

WHAT THESE CHECKS DO NOT DO, stated plainly because an adversarial review
of an earlier version found the surrounding documentation overclaiming:

Every check here operates on strings. None of them understands meaning, so
none of them can tell that a claim REVERSES what its quotation says. A claim
of "remote work does not increase productivity", quoted against a source
sentence reporting that it increased output by 8%, passes all five checks:
the quotation is real, the numbers line up, the vocabulary overlaps. So does
a claim that attributes to the authors a view the source is describing in
order to reject it, and a claim that drops a "we caution that..." qualifier.

Numbers written as words ("forty-five percent") contain no digits, so the
numeric check has nothing to compare and passes vacuously. Units and
magnitudes are not understood either: a claim of "8x" against a source
saying "8%" passes.

These are not bugs to be fixed by tightening a threshold; they are the
boundary of what string comparison can establish. What this layer
guarantees is GROUNDING -- that the quotation is really in the source, and
that the figures in the claim are really in the quotation. It does not
guarantee FAIR READING. That is why the evaluation in Phase 6.1 ends with a
worksheet for a human rather than a score, and why Chapter 7.3's warning
about confident prose still applies to this system's own output.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Literal, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from ..schemas import VerifiedFinding
from ..search.planner import keywords
from .documents import SourceDocument, normalize_text

# A snippet shorter than this matches too much to be evidence of anything.
MIN_SNIPPET_CHARS = 40
# How close a fuzzy match has to be before it counts as "the source says this".
DEFAULT_MIN_SIMILARITY = 0.82
# Fuzzy matching compares windows around candidate anchors, not the whole
# document; this caps how many it will look at on a very long paper.
MAX_ANCHORS = 40
# Comparing a very long "quotation" against many windows is quadratic, and a
# quotation longer than this is not a quotation. Capping it stops a careless
# or hostile source from stalling a run for minutes.
MAX_SNIPPET_CHARS = 2_000

# A number, but not one glued to a word by a hyphen: "COVID-19" and "GPT-4"
# must not donate their digits to a claim's evidence pool.
_NUMBER = re.compile(r"(?<![\w.\-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)")


class CheckOutcome(BaseModel):
    """One named check and how it went."""

    name: str
    passed: bool
    detail: str


class SnippetMatch(BaseModel):
    """Whether the quoted snippet was actually found in the source text."""

    found: bool
    method: Literal["exact", "fuzzy", "none"] = "none"
    similarity: float = 0.0
    matched_text: str = ""
    position: int = -1


class VerificationResult(BaseModel):
    """The full verdict on one candidate claim."""

    passed: bool
    checks: List[CheckOutcome] = Field(default_factory=list)
    snippet_match: SnippetMatch = Field(default_factory=lambda: SnippetMatch(found=False))

    @property
    def failed_checks(self) -> List[CheckOutcome]:
        return [check for check in self.checks if not check.passed]

    def summary(self) -> str:
        if self.passed:
            return "verified: " + "; ".join(c.detail for c in self.checks if c.name == "snippet_grounded")
        return "rejected: " + "; ".join(f"{c.name} -- {c.detail}" for c in self.failed_checks)


# --- numbers -----------------------------------------------------------


def numbers_in(text: str) -> Set[float]:
    """Every number mentioned in a piece of text, as comparable values.

    "1,200" and "1200" are the same number; "8" and "8.0" are the same
    number. Treating them as different would reject honest claims over
    formatting, which is the same mistake normalising quotes avoids.
    """
    values: Set[float] = set()
    for raw in _NUMBER.findall(text or ""):
        try:
            values.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


# --- locating the snippet ---------------------------------------------


def _shingles(words: Sequence[str], size: int) -> List[str]:
    return [" ".join(words[i : i + size]) for i in range(0, max(0, len(words) - size + 1))]


def _snap_to_words(text: str, start: int, end: int) -> Tuple[int, int]:
    """Widen a span so it begins and ends on whole words.

    Aligning a fuzzy window can land a character or two inside a word, and a
    citation that begins "ourly productivity rose..." looks like a bug even
    when the match behind it is sound. Since this snippet is what ends up
    quoted in the final report, it is worth getting right.
    """
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "-'"):
        start -= 1
    while end < len(text) and (text[end].isalnum() or text[end] in "-'"):
        end += 1
    return start, end


def locate_snippet(
    snippet: str,
    document_text: str,
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> SnippetMatch:
    """Find a quoted snippet inside a source document.

    Exact matching is tried first and is what most honest quotations hit.
    When it fails -- because the extractor tidied a line break, dropped a
    citation marker, or joined two sentences -- a fuzzy search runs, but
    only around places where a distinctive phrase from the snippet actually
    occurs. That keeps it fast on a long paper and, more importantly, keeps
    it honest: a snippet with no recognisable phrase anywhere in the
    document does not get a second chance to score well by coincidence.
    """
    needle = normalize_text(snippet)
    haystack = normalize_text(document_text)
    if not needle or not haystack or len(needle) > MAX_SNIPPET_CHARS:
        return SnippetMatch(found=False)

    # Case-insensitive search done with a regex on the ORIGINAL string, rather
    # than by lowercasing it first. Lowercasing is not length-preserving for
    # every character -- Turkish dotted capital I becomes two characters -- so
    # offsets taken from a lowercased copy can be wrong by a few positions,
    # and the stored citation then starts in the middle of a different
    # sentence. A page could exploit that deliberately.
    exact = re.search(re.escape(needle), haystack, re.IGNORECASE)
    if exact is not None:
        return SnippetMatch(
            found=True,
            method="exact",
            similarity=1.0,
            matched_text=haystack[exact.start() : exact.end()],
            position=exact.start(),
        )

    # Same hazard as above, for the fuzzy path: offsets are taken from the
    # lowercased copy but slice the original, so if lowercasing changed the
    # length, matching is done case-sensitively instead. Slightly less
    # forgiving, and correct.
    lowered_needle = needle.lower()
    lowered_haystack = haystack.lower()
    if len(lowered_haystack) != len(haystack) or len(lowered_needle) != len(needle):
        lowered_needle, lowered_haystack = needle, haystack
    words = lowered_needle.split()

    # Each anchor records where a distinctive phrase from the quotation sits
    # in the document AND where it sits inside the quotation. Both are needed:
    # subtracting the second from the first aligns the comparison window with
    # the quotation instead of centring it on the phrase, which otherwise
    # drags in unrelated trailing text and makes an honest quotation score
    # like a bad one.
    anchors: List[Tuple[int, int]] = []
    for size in (6, 4, 3):
        if len(words) < size:
            continue
        for shingle in _shingles(words, size):
            offset_in_needle = lowered_needle.find(shingle)
            if offset_in_needle == -1:
                continue
            found_at = lowered_haystack.find(shingle)
            while found_at != -1 and len(anchors) < MAX_ANCHORS:
                anchors.append((found_at, offset_in_needle))
                found_at = lowered_haystack.find(shingle, found_at + 1)
            if len(anchors) >= MAX_ANCHORS:
                break
        if anchors:
            break

    if not anchors:
        return SnippetMatch(found=False)

    best = SnippetMatch(found=False)
    length = len(needle)
    for found_at, offset_in_needle in anchors:
        start = max(0, found_at - offset_in_needle)
        for width in (length, int(length * 1.15) + 8):
            snapped_start, snapped_end = _snap_to_words(haystack, start, start + width)
            candidate = haystack[snapped_start:snapped_end]
            if not candidate:
                continue
            ratio = SequenceMatcher(None, lowered_needle, candidate.lower()).ratio()
            if ratio > best.similarity:
                best = SnippetMatch(
                    found=ratio >= min_similarity,
                    method="fuzzy",
                    similarity=round(ratio, 3),
                    matched_text=candidate,
                    position=snapped_start,
                )
    return best


# --- the checks --------------------------------------------------------


def verify_claim(
    claim: str,
    snippet: str,
    document_text: str,
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> VerificationResult:
    """Run every check on one claim-and-snippet pair."""
    checks: List[CheckOutcome] = []

    claim_text = (claim or "").strip()
    snippet_text = (snippet or "").strip()

    checks.append(
        CheckOutcome(
            name="claim_present",
            passed=bool(claim_text),
            detail="A claim was provided." if claim_text else "The claim is empty.",
        )
    )

    normalized_snippet = normalize_text(snippet_text)
    long_enough = len(normalized_snippet) >= MIN_SNIPPET_CHARS
    checks.append(
        CheckOutcome(
            name="snippet_length",
            passed=long_enough,
            detail=(
                f"Snippet is {len(normalized_snippet)} characters."
                if long_enough
                else f"Snippet is only {len(normalized_snippet)} characters; at least "
                f"{MIN_SNIPPET_CHARS} are needed for a quotation to be evidence of anything."
            ),
        )
    )

    match = locate_snippet(snippet_text, document_text, min_similarity=min_similarity)
    checks.append(
        CheckOutcome(
            name="snippet_grounded",
            passed=match.found,
            detail=(
                f"Found in the source ({match.method} match, similarity {match.similarity:.2f})."
                if match.found
                else (
                    "This snippet does not appear in the source text"
                    + (
                        f" (closest match scored {match.similarity:.2f}, below the "
                        f"{min_similarity:.2f} threshold)."
                        if match.similarity
                        else " at all."
                    )
                )
            ),
        )
    )

    # The remaining checks run against the text as it appears IN THE SOURCE,
    # not against the text the model submitted.
    #
    # This distinction is the whole ballgame, and getting it wrong was a real
    # defect in an earlier version. Fuzzy matching tolerates a quotation that
    # differs from the source by a few percent, which is necessary for honest
    # quotations. But it means a model can submit a quotation with one digit
    # altered -- "output rose 80%" against a source saying "output rose 8%" --
    # and still match. If the numeric check then ran on the submitted text,
    # the 80 would look supported, while the citation actually stored and
    # shown to the reader would be the source's "8%". A claim of 80% beside a
    # quotation of 8%, marked verified.
    #
    # Checking the matched source text instead closes that hole: the evidence
    # a check passes against is the same evidence the reader is shown.
    evidence_text = match.matched_text if match.found else snippet_text

    claim_numbers = numbers_in(claim_text)
    evidence_numbers_found = numbers_in(evidence_text)
    unsupported = claim_numbers - evidence_numbers_found
    checks.append(
        CheckOutcome(
            name="numeric_support",
            passed=not unsupported,
            detail=(
                "Every number in the claim appears in the source text quoted as evidence."
                if not unsupported
                else "The claim states "
                + ", ".join(_format_number(n) for n in sorted(unsupported))
                + ", which the source text quoted as evidence does not mention."
            ),
        )
    )

    claim_words = keywords(claim_text)
    snippet_words = keywords(evidence_text)
    shared = claim_words & snippet_words
    checks.append(
        CheckOutcome(
            name="shared_vocabulary",
            passed=bool(shared),
            detail=(
                f"Claim and evidence share: {', '.join(sorted(shared)[:5])}."
                if shared
                else "The claim and the quoted evidence have no significant words in common, "
                "so the quotation appears to be about something else."
            ),
        )
    )

    return VerificationResult(
        passed=all(check.passed for check in checks),
        checks=checks,
        snippet_match=match,
    )


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def verify_candidate(
    candidate: dict,
    document: SourceDocument,
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> Tuple[Optional[VerifiedFinding], VerificationResult]:
    """Check one extracted candidate and build a VerifiedFinding only if it passes.

    Returning None for a failure is deliberate rather than returning a
    finding with ``verified=False``: per the Phase 1.3 schema, a
    VerifiedFinding that exists at all is one that passed. A rejected claim
    leaves behind a reason, not a record that could later be mistaken for
    evidence.
    """
    claim = str(candidate.get("claim") or "")
    snippet = str(candidate.get("supporting_snippet") or candidate.get("snippet") or "")

    result = verify_claim(claim, snippet, document.text, min_similarity=min_similarity)
    if not result.passed:
        return None, result

    finding = VerifiedFinding(
        claim=claim.strip(),
        # The snippet stored is the text as it appears in the source, not as
        # the model retyped it, so the citation quotes the document itself.
        supporting_snippet=(result.snippet_match.matched_text or snippet).strip(),
        source_url=document.source_url,
        verified=True,
    )
    return finding, result
