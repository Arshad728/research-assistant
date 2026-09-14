"""Tests for the verification layer (Phase 3.3).

These are the most important tests in the project. Everything else can be
slightly wrong and produce a worse report; this being wrong produces a
report that looks trustworthy and is not.

The source text below is deliberately awkward in the ways a real PDF is:
typographic quotes, an en-dash, a word split across a line break, and a
sentence with several numbers in it.
"""
import pytest

from research_assistant.extraction.documents import SourceDocument, dehyphenate, normalize_text
from research_assistant.extraction.verify import (
    DEFAULT_MIN_SIMILARITY,
    MAX_SNIPPET_CHARS,
    MIN_SNIPPET_CHARS,
    locate_snippet,
    numbers_in,
    verify_candidate,
    verify_claim,
)

SOURCE_TEXT = dehyphenate(
    "Introduction. Several organisations have trialled reduced working weeks in recent years.\n\n"
    "Results. Across the trial period, hourly produc-\n"
    "tivity increased by 8%, while total weekly output remained statistically unchanged. "
    "Participating organisations numbered 61 in total, spread across four countries.\n\n"
    "Discussion. The authors note that these findings “should be read with caution” given "
    "the absence of a control group – a limitation common to trials of this kind.\n"
)

DOCUMENT = SourceDocument(
    source_url="https://example.org/four-day-week-trial-2025",
    title="The Four-Day Week: Assessing Global Trials",
    text=SOURCE_TEXT,
    content_kind="pdf",
    page_count=12,
)

GOOD_SNIPPET = (
    "hourly productivity increased by 8%, while total weekly output remained "
    "statistically unchanged"
)


class TestNormalisation:
    def test_dehyphenation_rejoins_words_split_across_lines(self):
        assert "productivity" in SOURCE_TEXT
        assert "produc-" not in SOURCE_TEXT

    def test_typographic_characters_are_normalised(self):
        normalised = normalize_text("“should be read with caution” – a limitation")
        assert '"should be read with caution"' in normalised
        assert "-" in normalised
        assert "–" not in normalised

    def test_whitespace_is_collapsed(self):
        assert normalize_text("a   b\n\nc") == "a b c"


class TestNumbersIn:
    def test_finds_plain_and_formatted_numbers(self):
        assert numbers_in("8% of 1,200 sites over 3.5 years") == {8.0, 1200.0, 3.5}

    def test_treats_equivalent_formats_as_the_same_number(self):
        assert numbers_in("8 percent") == numbers_in("8.0%")

    def test_no_numbers_is_an_empty_set(self):
        assert numbers_in("no digits here") == set()


class TestLocateSnippet:
    def test_exact_quotation_is_found(self):
        match = locate_snippet(GOOD_SNIPPET, SOURCE_TEXT)
        assert match.found is True
        assert match.method == "exact"
        assert match.similarity == 1.0

    def test_quotation_differing_only_in_typography_is_found(self):
        match = locate_snippet(
            'The authors note that these findings "should be read with caution" given the '
            "absence of a control group - a limitation",
            SOURCE_TEXT,
        )
        assert match.found is True

    def test_lightly_reworded_quotation_still_matches_fuzzily(self):
        match = locate_snippet(
            "hourly productivity increased by 8% while total weekly output remained "
            "statistically unchanged.",
            SOURCE_TEXT,
        )
        assert match.found is True
        assert match.similarity >= 0.82

    def test_invented_quotation_is_not_found(self):
        match = locate_snippet(
            "Researchers concluded that the four-day week doubled employee output across "
            "every participating organisation without exception.",
            SOURCE_TEXT,
        )
        assert match.found is False

    def test_returns_the_text_as_it_appears_in_the_source(self):
        match = locate_snippet(GOOD_SNIPPET, SOURCE_TEXT)
        assert "productivity increased by 8%" in match.matched_text

    def test_the_matched_text_never_starts_or_ends_mid_word(self):
        # Regression: window alignment used to land a character inside a word,
        # producing citations that began "ourly productivity rose...".
        match = locate_snippet(
            "Hourly productivity increased by 8% while total weekly output remained "
            "statistically unchanged",
            SOURCE_TEXT,
        )
        assert match.found is True
        assert match.matched_text[0].isalnum()
        assert match.matched_text.lower().startswith("hourly")

    def test_empty_inputs_do_not_match(self):
        assert locate_snippet("", SOURCE_TEXT).found is False
        assert locate_snippet(GOOD_SNIPPET, "").found is False


class TestVerifyClaim:
    def test_a_well_supported_claim_passes_every_check(self):
        result = verify_claim(
            "Output per hour rose by 8% while total weekly output was unchanged.",
            GOOD_SNIPPET,
            SOURCE_TEXT,
        )
        assert result.passed is True
        assert result.failed_checks == []

    def test_a_fabricated_quotation_is_rejected(self):
        result = verify_claim(
            "The four-day week doubled output.",
            "Researchers concluded that the four-day week doubled employee output across "
            "every participating organisation without exception.",
            SOURCE_TEXT,
        )
        assert result.passed is False
        assert "snippet_grounded" in {c.name for c in result.failed_checks}

    def test_a_claim_with_an_unsupported_number_is_rejected(self):
        # The quotation is real. The claim adds a figure it does not contain.
        result = verify_claim(
            "Output per hour rose by 8% across 61 organisations.",
            GOOD_SNIPPET,
            SOURCE_TEXT,
        )
        assert result.passed is False
        failed = {c.name: c.detail for c in result.failed_checks}
        assert "numeric_support" in failed
        assert "61" in failed["numeric_support"]

    def test_a_too_short_snippet_is_rejected(self):
        result = verify_claim("Productivity rose.", "productivity increased", SOURCE_TEXT)
        assert result.passed is False
        assert "snippet_length" in {c.name for c in result.failed_checks}
        assert str(MIN_SNIPPET_CHARS) in next(
            c.detail for c in result.failed_checks if c.name == "snippet_length"
        )

    def test_a_real_quotation_attached_to_an_unrelated_claim_is_rejected(self):
        result = verify_claim(
            "Alpine lichen fixes carbon more slowly at altitude.",
            GOOD_SNIPPET,
            SOURCE_TEXT,
        )
        assert result.passed is False
        assert "shared_vocabulary" in {c.name for c in result.failed_checks}

    def test_an_empty_claim_is_rejected(self):
        result = verify_claim("", GOOD_SNIPPET, SOURCE_TEXT)
        assert result.passed is False
        assert "claim_present" in {c.name for c in result.failed_checks}

    def test_every_check_reports_a_human_readable_reason(self):
        result = verify_claim("The week doubled output.", "far too short", SOURCE_TEXT)
        assert result.passed is False
        for check in result.checks:
            assert check.detail.strip()
        assert result.summary().startswith("rejected:")


class TestVerifyCandidate:
    def test_a_passing_candidate_becomes_a_verified_finding(self):
        finding, result = verify_candidate(
            {
                "claim": "Output per hour rose by 8% while total weekly output was unchanged.",
                "supporting_snippet": GOOD_SNIPPET,
            },
            DOCUMENT,
        )
        assert result.passed is True
        assert finding is not None
        assert finding.verified is True
        assert finding.source_url == DOCUMENT.source_url

    def test_the_stored_snippet_is_the_sources_wording_not_the_models(self):
        finding, _ = verify_candidate(
            {
                "claim": "Output per hour rose by 8%.",
                # Retyped with different punctuation and casing.
                "supporting_snippet": "Hourly productivity increased by 8% while total weekly "
                "output remained statistically unchanged",
            },
            DOCUMENT,
        )
        assert finding is not None
        # The source writes "8%," with a comma; the model's version dropped it.
        # Asserting on that difference is what makes this test discriminating:
        # an earlier version asserted on text common to both, so it passed
        # whichever string was stored and tested nothing at all.
        assert "8%," in finding.supporting_snippet
        assert (
            "Hourly productivity increased by 8% while" not in finding.supporting_snippet
        ), "the model's retyped wording was stored instead of the source's"

    def test_a_failing_candidate_produces_no_finding_at_all(self):
        finding, result = verify_candidate(
            {
                "claim": "The trial covered 61 organisations and doubled output.",
                "supporting_snippet": "Researchers concluded that output doubled at every site "
                "in the programme without exception.",
            },
            DOCUMENT,
        )
        assert finding is None
        assert result.passed is False

    def test_a_candidate_missing_its_snippet_is_rejected(self):
        finding, result = verify_candidate({"claim": "Something happened."}, DOCUMENT)
        assert finding is None
        assert result.passed is False


class TestDoctoredQuotations:
    """The hole an adversarial review found, and the checks that now close it.

    Fuzzy matching has to tolerate a quotation that differs slightly from its
    source, or honest quotations get rejected. That tolerance is exactly what
    a model would exploit to submit a quotation with one digit changed: it
    still matches, and if the numeric check ran on the submitted text rather
    than on the source text, the altered figure would look supported while
    the citation shown to the reader said something else.
    """

    def test_a_quotation_with_an_altered_figure_is_rejected(self):
        result = verify_claim(
            "Output per hour rose by 80% after the switch.",
            # The source says 8%. One digit added; still a near-perfect match.
            "hourly productivity increased by 80%, while total weekly output remained "
            "statistically unchanged",
            SOURCE_TEXT,
        )

        assert result.snippet_match.found is True, "the doctored quote should still match"
        assert result.passed is False, "but the claim must not pass verification"
        assert "numeric_support" in {c.name for c in result.failed_checks}

    def test_no_finding_is_built_from_a_doctored_quotation(self):
        finding, result = verify_candidate(
            {
                "claim": "Output per hour rose by 80% after the switch.",
                "supporting_snippet": "hourly productivity increased by 80%, while total "
                "weekly output remained statistically unchanged",
            },
            DOCUMENT,
        )
        assert finding is None
        assert result.passed is False

    def test_padding_a_quotation_with_real_text_cannot_smuggle_a_figure_in(self):
        # A long quotation of genuine surrounding text, with an invented
        # sentence appended, matches well on similarity alone.
        padded = (
            "Introduction. Several organisations have trialled reduced working weeks in "
            "recent years. Results. Across the trial period, hourly productivity increased "
            "by 8%, while total weekly output remained statistically unchanged. "
            "Mortality fell by 40% in the treatment arm."
        )
        result = verify_claim("Mortality fell by 40%.", padded, SOURCE_TEXT)

        assert result.passed is False
        assert "numeric_support" in {c.name for c in result.failed_checks}

    def test_the_checks_run_against_what_the_reader_will_be_shown(self):
        # Whatever text ends up stored is the text the numeric check used.
        finding, result = verify_candidate(
            {
                "claim": "Output per hour rose by 8%.",
                "supporting_snippet": "hourly productivity increased by 8% while total weekly "
                "output remained statistically unchanged",
            },
            DOCUMENT,
        )
        assert finding is not None
        assert numbers_in("Output per hour rose by 8%.") <= numbers_in(
            finding.supporting_snippet
        )


class TestNumberExtraction:
    def test_digits_glued_to_a_word_do_not_count_as_evidence(self):
        # "COVID-19" must not license a claim about 19 trials. An earlier
        # version's regex allowed exactly that.
        assert numbers_in("a COVID-19 era survey") == set()
        assert numbers_in("GPT-4 and GPT-5") == set()

    def test_real_numbers_are_still_found_next_to_punctuation(self):
        assert numbers_in("rose by 8%, from 1,200 to 3.5 million") == {8.0, 1200.0, 3.5}

    def test_a_ranged_number_is_still_read(self):
        assert 2019.0 in numbers_in("published in 2019")


class TestPerformanceGuards:
    def test_an_absurdly_long_quotation_is_refused_rather_than_matched_slowly(self):
        # Comparing a huge "quotation" against many windows is quadratic. A
        # 49KB snippet previously took minutes; it is now refused outright.
        import time

        document = (SOURCE_TEXT + " ") * 400
        enormous = ("hourly productivity increased by 8% " * 200)[: MAX_SNIPPET_CHARS + 500]

        started = time.monotonic()
        match = locate_snippet(enormous, document)
        elapsed = time.monotonic() - started

        assert match.found is False
        assert elapsed < 1.0, f"took {elapsed:.1f}s; the length cap is not working"


class TestDefaults:
    """The shipped defaults are themselves a decision, so they are pinned.

    A mutation test found that every threshold test passed an explicit value,
    so the defaults could be changed to anything without a test failing.
    """

    def test_the_similarity_threshold_is_what_the_documentation_says(self):
        assert DEFAULT_MIN_SIMILARITY == 0.82

    def test_the_minimum_snippet_length_is_pinned(self):
        assert MIN_SNIPPET_CHARS == 40

    def test_the_snippet_length_cap_is_pinned(self):
        assert MAX_SNIPPET_CHARS == 2_000


class TestFuzzyThreshold:
    """The rejecting direction of fuzzy matching, which had no test at all.

    A mutation test showed the threshold could be lowered to 0.30, or removed
    entirely, with the whole suite still green: the one test for an invented
    quotation was passing at the anchor stage, never reaching the threshold.
    """

    def test_a_snippet_that_anchors_but_diverges_is_rejected(self):
        # Opens with real text, so it finds an anchor and is scored -- then
        # continues into invention and must fail on similarity.
        match = locate_snippet(
            "Across the trial period, hourly productivity increased by a factor of three "
            "and every participating organisation reported that staff turnover had fallen "
            "to zero within a fortnight of the change taking effect.",
            SOURCE_TEXT,
        )

        assert match.method == "fuzzy", "this snippet should reach the scoring stage"
        assert 0.0 < match.similarity < DEFAULT_MIN_SIMILARITY
        assert match.found is False

    def test_lowering_the_threshold_would_accept_it(self):
        # Proves the previous test is actually governed by the threshold,
        # rather than passing for some unrelated reason. Derived from the
        # observed score rather than hard-coded, so it cannot drift.
        snippet = (
            "Across the trial period, hourly productivity increased by a factor of three "
            "and every participating organisation reported that staff turnover had fallen "
            "to zero within a fortnight of the change taking effect."
        )
        observed = locate_snippet(snippet, SOURCE_TEXT).similarity
        assert observed > 0

        forgiving = locate_snippet(snippet, SOURCE_TEXT, min_similarity=observed - 0.01)
        assert forgiving.found is True, "the threshold is not what decides this"


class TestStrictness:
    """Chapter 6.4 asks for this layer to start strict. These pin that down."""

    @pytest.mark.parametrize(
        "claim, snippet",
        [
            # Number the evidence never mentions.
            ("Output rose 12%.", GOOD_SNIPPET),
            # Real-sounding but absent quotation.
            ("Output rose.", "Weekly output rose sharply at every participating organisation."),
            # Evidence too short to mean anything.
            ("Output rose 8%.", "rose by 8%"),
        ],
    )
    def test_borderline_claims_are_rejected_rather_than_allowed(self, claim, snippet):
        assert verify_claim(claim, snippet, SOURCE_TEXT).passed is False
