"""What a fetched source looks like once it has been turned into text.

Chapter 4.4 describes extraction as "opening each source, whether that
requires parsing a PDF or reading a web page's text." This module defines
what comes out of that step, and it makes one decision worth stating
plainly.

A long paper will not fit in a language model's context, so the text given
to the model for extraction has to be capped. But the *verification* step
later has to search the source for a snippet, and a snippet that exists on
page 30 of a truncated document would look invented. So the document keeps
its full text, and truncation happens only at the point of handing text to
the model. Extraction sees a prefix; verification sees everything.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal, Optional

from pydantic import BaseModel, Field

ContentKind = Literal["pdf", "html", "text"]

# Roughly 30k tokens of English, which leaves room for instructions and a
# reply in a long-context model without being wasteful.
DEFAULT_MODEL_TEXT_LIMIT = 120_000

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}

_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
}

_DASH_MAP = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"}


def normalize_text(text: str) -> str:
    """Put text into a comparable form without changing what it says.

    PDFs in particular are full of characters that look identical on screen
    and differ in bytes: typographic quotes, four kinds of dash, the "fi"
    ligature, non-breaking spaces. Left alone, these make a snippet that a
    human would call an exact quote fail an exact-match check. Normalising
    both sides before comparing is what stops the verification step from
    rejecting honest evidence over typography.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for source, replacement in {**_LIGATURES, **_QUOTE_MAP, **_DASH_MAP}.items():
        text = text.replace(source, replacement)
    text = text.replace("­", "")  # soft hyphen
    text = text.replace(" ", " ")  # non-breaking space
    return re.sub(r"\s+", " ", text).strip()


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break by PDF layout.

    A two-column paper is full of "produc-\\ntivity". Without this, a snippet
    quoting that sentence can never be found in the extracted text, and a
    perfectly real quotation gets discarded as unverifiable.
    """
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


class SourceDocument(BaseModel):
    """One retrieved source, converted to text."""

    source_url: str
    title: str = ""
    text: str = Field(description="The full extracted text. Never truncated.")
    content_kind: ContentKind = "text"
    page_count: Optional[int] = None

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_usable(self) -> bool:
        """Enough text to be worth extracting from at all.

        A PDF of page images returns almost nothing from a text extractor.
        Recognising that as "no text" rather than passing a few stray
        characters to the model saves a pointless call and, more
        importantly, stops the system reporting that it read something it
        did not.
        """
        return len(normalize_text(self.text)) >= 200

    def text_for_model(self, limit: int = DEFAULT_MODEL_TEXT_LIMIT) -> str:
        """The prefix of the text that is small enough to send to a model."""
        if len(self.text) <= limit:
            return self.text
        return self.text[:limit]

    def was_truncated_for_model(self, limit: int = DEFAULT_MODEL_TEXT_LIMIT) -> bool:
        return len(self.text) > limit
