"""The Extraction and Verification Agent (Chapter 4.4).

The two jobs in this agent's name are deliberately kept together, and
deliberately kept apart from writing. Chapter 4.4 explains why: "an agent
whose only job is to write a fluent report has every incentive, in a sense,
to smooth over a shaky claim, while an agent whose only job is to check
facts has no such pull."

Inside the agent the same separation repeats at a smaller scale. The model
proposes claims; it does not decide whether they hold up. That decision is
made by verify.py, mechanically, against the source text. A claim the model
is confident about and the source does not support is discarded exactly as
readily as an obvious fabrication, because the check never asks the model
what it thinks.

The model call is injected rather than hard-wired, which lets the whole
extract-and-verify path -- including how rejections are counted and
reported -- be tested without an API key or a network connection.
"""
from __future__ import annotations

from typing import Awaitable, Callable, List, Optional, Sequence, Tuple

from ..extraction.documents import DEFAULT_MODEL_TEXT_LIMIT, SourceDocument
from ..extraction.extract import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionReport,
    RejectedClaim,
    build_extraction_prompt,
    parse_extraction_reply,
)
from ..extraction.fetch import FetchError, fetch_source
from ..extraction.parsers import ParseError
from ..extraction.verify import DEFAULT_MIN_SIMILARITY, verify_candidate
from ..schemas import CandidateSource, VerifiedFinding

# (system_prompt, user_prompt) -> the model's reply text
CompletionFn = Callable[[str, str], Awaitable[str]]


async def complete_with_sdk(system_prompt: str, user_prompt: str, *, model: Optional[str] = None) -> str:
    """Default one-shot model call, used when no other completion is injected."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=[],
        max_turns=1,
        model=model,
    )

    chunks: List[str] = []
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content or []:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "\n".join(chunks)


# The default model id is an alias, not a dated snapshot: Google documents it
# as always pointing at the current Flash release, so this stays correct
# without needing to be updated by hand as new Gemini versions ship.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


async def complete_with_gemini(system_prompt: str, user_prompt: str, *, model: Optional[str] = None) -> str:
    """One-shot model call for the 'gemini' provider, an alternative to Claude.

    Same (system_prompt, user_prompt) -> reply contract as ``complete_with_sdk``,
    so it can be used anywhere a ``CompletionFn`` is expected -- see
    ``get_completion_fn`` below. The google-genai client's call is synchronous,
    so it is run in a worker thread rather than blocking the event loop the
    rest of the pipeline runs on.
    """
    import asyncio

    from google import genai
    from google.genai import types

    from ..config import get_settings

    api_key = get_settings().require(
        "gemini_api_key", "GEMINI_API_KEY", "https://aistudio.google.com/apikey"
    )

    def _call() -> str:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model or DEFAULT_GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text or ""

    return await asyncio.to_thread(_call)


def get_completion_fn(provider: str = "claude", model: Optional[str] = None) -> CompletionFn:
    """Pick the model-calling function for a provider name.

    The Planning, Extraction and Writer steps all accept a ``provider``
    string and use this to build their default completion function, so the
    choice is made the same way everywhere and a bad provider name fails
    here -- with a clear message -- instead of deep inside a prompt.
    """
    if provider == "claude":
        return lambda system, user: complete_with_sdk(system, user, model=model)
    if provider == "gemini":
        return lambda system, user: complete_with_gemini(system, user, model=model)
    raise ValueError(f"Unknown provider {provider!r}. Choose 'claude' or 'gemini'.")


class ExtractionAgent:
    """Turns candidate sources into verified findings, and says what it rejected."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        provider: str = "claude",
        complete: Optional[CompletionFn] = None,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        text_limit: int = DEFAULT_MODEL_TEXT_LIMIT,
    ) -> None:
        self.model = model
        self.provider = provider
        self.min_similarity = min_similarity
        self.text_limit = text_limit
        self._complete = complete or get_completion_fn(provider, model)

    async def extract_from_document(
        self, document: SourceDocument, question: str
    ) -> Tuple[List[VerifiedFinding], ExtractionReport]:
        """Extract and verify claims from a document that has already been fetched."""
        report = ExtractionReport(
            source_url=document.source_url,
            title=document.title,
            text_truncated_for_model=document.was_truncated_for_model(self.text_limit),
        )

        if not document.is_usable:
            report.status = "unreadable"
            report.detail = (
                f"Only {document.char_count} characters of text could be extracted. The "
                "source may be a scanned image, or behind a paywall that served a stub page."
            )
            return [], report

        reply = await self._complete(
            EXTRACTION_SYSTEM_PROMPT,
            build_extraction_prompt(
                question, document.text_for_model(self.text_limit), document.title
            ),
        )

        candidates = parse_extraction_reply(reply)
        if candidates is None:
            report.status = "no_claims"
            report.detail = "The model's reply could not be read as a list of claims."
            return [], report

        report.candidates_proposed = len(candidates)
        if not candidates:
            report.status = "no_claims"
            report.detail = "The source does not address the question."
            return [], report

        findings: List[VerifiedFinding] = []
        for candidate in candidates:
            finding, result = verify_candidate(
                candidate, document, min_similarity=self.min_similarity
            )
            if finding is not None:
                findings.append(finding)
                continue
            report.rejected.append(
                RejectedClaim(
                    claim=str(candidate.get("claim", ""))[:300],
                    failed_checks=[check.name for check in result.failed_checks],
                    detail="; ".join(check.detail for check in result.failed_checks),
                )
            )

        report.findings_verified = len(findings)
        if not findings:
            report.status = "no_claims"
            report.detail = (
                f"All {report.candidates_proposed} proposed claim(s) failed verification."
            )
        return findings, report

    async def extract_from_source(
        self, source: CandidateSource, question: str
    ) -> Tuple[List[VerifiedFinding], ExtractionReport]:
        """Fetch one candidate source, then extract and verify claims from it."""
        try:
            document = await fetch_source(source)
        except FetchError as exc:
            return [], ExtractionReport(
                source_url=source.source_url,
                title=source.title,
                status="fetch_failed",
                detail=str(exc),
            )
        except ParseError as exc:
            return [], ExtractionReport(
                source_url=source.source_url,
                title=source.title,
                status="unreadable",
                detail=str(exc),
            )

        return await self.extract_from_document(document, question)

    async def extract_from_sources(
        self, sources: Sequence[CandidateSource], question: str
    ) -> Tuple[List[VerifiedFinding], List[ExtractionReport]]:
        """Process a shortlist, one source at a time.

        Sources are handled sequentially rather than in parallel on purpose:
        several may live on the same host, and a burst of simultaneous
        downloads is the fastest way to get an IP blocked by a publisher who
        would have served the requests happily one at a time.
        """
        all_findings: List[VerifiedFinding] = []
        reports: List[ExtractionReport] = []
        for source in sources:
            findings, report = await self.extract_from_source(source, question)
            all_findings.extend(findings)
            reports.append(report)
        return all_findings, reports


def summarise_reports(reports: Sequence[ExtractionReport]) -> str:
    """A short account of how a batch went, for the Orchestrator and for logs."""
    if not reports:
        return "No sources were processed."

    verified = sum(r.findings_verified for r in reports)
    proposed = sum(r.candidates_proposed for r in reports)
    rejected = sum(len(r.rejected) for r in reports)
    unusable = [r for r in reports if r.status in ("fetch_failed", "unreadable")]
    productive = [r for r in reports if r.findings_verified > 0]

    parts = [
        f"{len(reports)} source(s) processed; {len(productive)} produced usable findings.",
        f"{verified} of {proposed} proposed claim(s) verified, {rejected} rejected.",
    ]
    if unusable:
        parts.append(f"{len(unusable)} source(s) could not be read at all.")
    return " ".join(parts)
