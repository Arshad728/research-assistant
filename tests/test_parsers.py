"""Tests for source parsing (Phase 3.1).

The PDF tests build real PDFs in memory and read them back, so they exercise
the actual library rather than a stand-in. The HTML tests use page markup
shaped like a real article page: navigation, a cookie banner, a script, and
a related-articles rail around the part that actually matters.
"""
import pytest

from research_assistant.extraction.documents import SourceDocument
from research_assistant.extraction.parsers import (
    ParseError,
    extract_html_text,
    extract_pdf_text,
)

ARTICLE_HTML = """<!DOCTYPE html>
<html><head>
  <title>Four-Day Week Trial Results | Example Journal</title>
  <script>var tracking = {id: 42};</script>
  <style>.nav { color: red; }</style>
</head>
<body>
  <nav><a href="/">Home</a> <a href="/archive">Archive</a> <a href="/about">About</a></nav>
  <div class="cookie-banner">We use cookies to improve your experience. Accept all cookies?</div>
  <article>
    <h1>Four-Day Week Trial Results</h1>
    <p>Across the trial period, hourly productivity increased by 8%, while total weekly
    output remained statistically unchanged across the participating organisations.</p>
    <p>The authors caution that the absence of a control group limits how strongly these
    results can be interpreted, a limitation common to workplace trials of this kind.</p>
    <p>Participating organisations numbered 61 in total, spread across four countries and
    covering a range of sectors from software to manufacturing and professional services.</p>
  </article>
  <aside class="related"><h2>Related</h2><a href="/x">Another article entirely</a></aside>
  <footer>Copyright Example Journal 2025. All rights reserved.</footer>
</body></html>
"""


def build_pdf(pages: list) -> bytes:
    """Create a small PDF in memory with the given page texts."""
    import pymupdf

    document = pymupdf.open()
    for body in pages:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), body, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


class TestPdfParsing:
    def test_reads_text_and_counts_pages(self):
        data = build_pdf(["First page about productivity.", "Second page about scheduling."])
        text, page_count = extract_pdf_text(data)

        assert page_count == 2
        assert "First page about productivity." in text
        assert "Second page about scheduling." in text

    def test_rejoins_words_split_across_a_line_break(self):
        # A narrow box forces the text to wrap with a hyphen, as a real
        # two-column paper does.
        import pymupdf

        document = pymupdf.open()
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 150, 400),
            "Hourly productivity increased substantially during the measured trial period.",
            fontsize=11,
        )
        data = document.tobytes()
        document.close()

        text, _ = extract_pdf_text(data)
        assert "-\n" not in text

    def test_a_non_pdf_raises_a_clear_error(self):
        with pytest.raises(ParseError, match="Could not open this PDF"):
            extract_pdf_text(b"this is definitely not a pdf")

    def test_a_password_protected_pdf_says_so(self):
        import pymupdf

        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "secret")
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
        )
        document.close()

        with pytest.raises(ParseError, match="password protected"):
            extract_pdf_text(data)


class TestHtmlParsing:
    def test_extracts_the_article_body(self):
        text, _ = extract_html_text(ARTICLE_HTML, url="https://example.org/trial")
        assert "hourly productivity increased by 8%" in text.lower()
        assert "61 in total" in text

    def test_leaves_out_navigation_scripts_and_boilerplate(self):
        text, _ = extract_html_text(ARTICLE_HTML, url="https://example.org/trial")
        lowered = text.lower()
        assert "var tracking" not in lowered
        assert "cookies" not in lowered
        assert "all rights reserved" not in lowered

    def test_extracts_the_title(self):
        _, title = extract_html_text(ARTICLE_HTML, url="https://example.org/trial")
        assert "Four-Day Week Trial Results" in title

    def test_falls_back_when_the_page_has_no_recognisable_article(self):
        # No <article>, no structure trafilatura expects -- but there is text,
        # and losing it entirely would be worse than a rough extraction.
        markup = "<html><body>" + "".join(
            f"<span>Sentence number {i} about measured productivity outcomes. </span>"
            for i in range(40)
        ) + "</body></html>"
        text, _ = extract_html_text(markup)
        assert "measured productivity outcomes" in text

    def test_an_empty_page_raises(self):
        with pytest.raises(ParseError, match="empty"):
            extract_html_text("   ")


class TestSourceDocument:
    def test_reports_usability_by_how_much_text_there_is(self):
        tiny = SourceDocument(source_url="https://example.org/x", text="a few words")
        real = SourceDocument(source_url="https://example.org/y", text="word " * 200)
        assert tiny.is_usable is False
        assert real.is_usable is True

    def test_truncates_only_the_copy_given_to_the_model(self):
        document = SourceDocument(source_url="https://example.org/x", text="x" * 500)

        assert document.char_count == 500
        assert len(document.text_for_model(limit=100)) == 100
        assert document.was_truncated_for_model(limit=100) is True
        # The document itself is untouched, so verification still sees everything.
        assert document.char_count == 500

    def test_short_documents_are_not_truncated(self):
        document = SourceDocument(source_url="https://example.org/x", text="short")
        assert document.was_truncated_for_model() is False
        assert document.text_for_model() == "short"
