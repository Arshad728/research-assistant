"""Reading sources and checking what they actually say (Phase 3)."""

from .documents import SourceDocument, dehyphenate, normalize_text
from .extract import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionReport,
    RejectedClaim,
    build_extraction_prompt,
    parse_extraction_reply,
)
from .fetch import FetchError, fetch_source, resolve_fetch_url
from .parsers import ParseError, extract_html_text, extract_pdf_text
from .verify import (
    CheckOutcome,
    SnippetMatch,
    VerificationResult,
    locate_snippet,
    numbers_in,
    verify_candidate,
    verify_claim,
)

__all__ = [
    "CheckOutcome",
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractionReport",
    "FetchError",
    "ParseError",
    "RejectedClaim",
    "SnippetMatch",
    "SourceDocument",
    "VerificationResult",
    "build_extraction_prompt",
    "dehyphenate",
    "extract_html_text",
    "extract_pdf_text",
    "fetch_source",
    "locate_snippet",
    "normalize_text",
    "numbers_in",
    "parse_extraction_reply",
    "resolve_fetch_url",
    "verify_candidate",
    "verify_claim",
]
