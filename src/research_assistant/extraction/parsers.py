"""Turning raw bytes into readable text: PDFs and web pages.

Two formats, two libraries, one shared concern. Chapter 5.4 makes the point
that neither tool does anything intelligent with the content -- "they
mechanically pull text out of a file format that was designed for visual
layout, not for being read by a program." What matters is that the text
they produce is faithful enough that a quotation from it can later be
found again, which is what the whole verification step depends on.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional, Tuple

from .documents import dehyphenate


class ParseError(RuntimeError):
    """A source could not be turned into text."""


# --- PDF ---------------------------------------------------------------


def extract_pdf_text(data: bytes) -> Tuple[str, int]:
    """Extract text from PDF bytes. Returns the text and the page count."""
    import pymupdf

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - any failure to open is the same story
        raise ParseError(f"Could not open this PDF: {exc}") from exc

    try:
        if document.needs_pass:
            raise ParseError("This PDF is password protected, so its text cannot be read.")
        pages = []
        for page in document:
            try:
                pages.append(page.get_text("text"))
            except Exception:  # noqa: BLE001 - one bad page should not lose the rest
                continue
        page_count = document.page_count
    finally:
        document.close()

    return dehyphenate("\n\n".join(pages)), page_count


# --- HTML --------------------------------------------------------------


class _TagStripper(HTMLParser):
    """A crude last-resort HTML-to-text fallback using only the standard library."""

    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self._chunks.append(data)

    @property
    def text(self) -> str:
        joined = "".join(self._chunks)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def extract_html_text(html: str, url: Optional[str] = None) -> Tuple[str, str]:
    """Extract the main article text from an HTML page.

    Uses trafilatura, which exists specifically to find the substantive part
    of a page and leave out navigation, cookie banners and related-article
    rails. That matters more here than it looks: boilerplate is not just
    noise in the model's context, it is text a later claim could be
    "verified" against, which would mean verifying a claim against a cookie
    notice. If trafilatura finds nothing at all, a plain tag-stripper runs
    instead, so a page never fails silently.
    """
    if not html or not html.strip():
        raise ParseError("The page was empty.")

    import trafilatura

    title = ""
    try:
        metadata = trafilatura.extract_metadata(html)
        if metadata is not None and getattr(metadata, "title", None):
            title = metadata.title or ""
    except Exception:  # noqa: BLE001 - metadata is a nicety, not a requirement
        title = ""

    text = None
    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except Exception:  # noqa: BLE001 - fall through to the crude extractor
        text = None

    if not text or len(text.strip()) < 200:
        stripper = _TagStripper()
        try:
            stripper.feed(html)
        except Exception as exc:  # noqa: BLE001
            raise ParseError(f"Could not read this page: {exc}") from exc
        fallback = stripper.text
        if len(fallback) > len(text or ""):
            text = fallback

    if not text or not text.strip():
        raise ParseError("No readable text could be extracted from this page.")

    if not title:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = re.sub(r"\s+", " ", match.group(1)).strip()

    return text.strip(), title
