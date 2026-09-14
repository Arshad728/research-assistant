"""Tests for the Streamlit interface (Phase 6.2).

Streamlit ships a headless test harness, which means the web interface can
be tested properly rather than merely imported and hoped for. These run the
page, set the widgets, click the button, and check what the page then shows
-- including that a run which finds nothing says so rather than rendering an
empty report as though it were a result.
"""
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit is an optional extra: pip install -e '.[ui]'"
)

APP = str(Path(__file__).resolve().parents[1] / "src" / "research_assistant" / "ui" / "app.py")


def fresh_app():
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=120)
    app.run()
    return app


class TestInitialPage:
    def test_the_page_loads_without_error(self):
        app = fresh_app()
        assert not app.exception
        assert app.title[0].value == "Multi-Agent Research Assistant"

    def test_demo_mode_is_on_by_default_when_there_is_no_api_key(self, no_api_keys):
        app = fresh_app()
        assert app.checkbox[0].label.startswith("Demo mode")
        assert app.checkbox[0].value is True

    def test_nothing_is_rendered_before_a_run(self):
        app = fresh_app()
        assert not app.tabs
        assert not app.metric


class TestRunning:
    def test_a_demo_run_produces_a_report_and_its_working(self):
        app = fresh_app()
        app.text_input[0].set_value(
            "What is the effect of a four-day work week on productivity?"
        )
        app.button[0].click().run()

        assert not app.exception
        metrics = {m.label: m.value for m in app.metric}
        assert int(metrics["Findings verified"]) > 0
        assert int(metrics["Sources found"]) > 0
        assert app.success  # a run that found evidence reports success
        # The report, the working and the rejections are all on the page.
        assert len(app.tabs) == 3

    def test_a_question_with_no_evidence_says_so_instead_of_inventing(self):
        app = fresh_app()
        app.text_input[0].set_value(
            "What are the performance characteristics of the Zephyrine consensus protocol?"
        )
        app.button[0].click().run()

        assert not app.exception
        assert app.warning, "a run with no evidence should warn rather than look successful"
        assert "No report could be written" in app.warning[0].value
        assert int({m.label: m.value for m in app.metric}["Findings verified"]) == 0

    def test_an_empty_question_is_refused(self):
        app = fresh_app()
        app.text_input[0].set_value("   ")
        app.button[0].click().run()

        assert app.error
        assert "Enter a research question" in app.error[0].value

    def test_turning_off_demo_mode_without_a_key_is_refused(self, no_api_keys):
        app = fresh_app()
        app.checkbox[0].set_value(False)
        app.text_input[0].set_value("Anything at all")
        app.button[0].click().run()

        assert app.error
        assert "ANTHROPIC_API_KEY" in app.error[0].value
