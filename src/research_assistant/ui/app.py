"""A lightweight web interface (Phase 6.2, optional half).

Chapter 5.6 calls this "a reasonable next step once the underlying pipeline
is solid, since it lets someone submit a query and read the resulting report
in a browser rather than a terminal." That is what this is, and no more.

The page shows the same thing the command line does, including the parts a
polished product would hide: which angles were searched, what each round
found, why the run stopped, and every claim that failed verification. For
this system those are not diagnostics, they are the reason to trust the
report, so they belong in front of the person reading it.

Run it with:

    streamlit run src/research_assistant/ui/app.py
"""
from __future__ import annotations

import asyncio

import streamlit as st

from research_assistant.config import get_settings

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="*", layout="wide")


def run_pipeline(question: str, demo: bool, max_rounds: int):
    """Run one research question and return the completed run."""
    if demo:
        from research_assistant.demo import build_demo_pipeline

        pipeline = build_demo_pipeline(max_rounds=max_rounds)
    else:
        from research_assistant.orchestration import ResearchPipeline, StoppingRule

        pipeline = ResearchPipeline(stopping_rule=StoppingRule(max_rounds=max_rounds))
    return asyncio.run(pipeline.run(question))


def render_run(run) -> None:
    if run.succeeded:
        st.success(
            f"{len(run.findings)} verified findings from "
            f"{len({f.source_url for f in run.findings})} sources."
        )
    else:
        st.warning(
            "No report could be written from verified evidence. That is a result rather than "
            "an error: nothing found could be checked against its source."
        )

    if run.stop_decision:
        st.caption(f"Stopped because: {run.stop_decision.reason}")

    columns = st.columns(4)
    columns[0].metric("Rounds", len(run.rounds))
    columns[1].metric("Sources found", len(run.sources))
    columns[2].metric("Findings verified", len(run.findings))
    columns[3].metric(
        "Claims rejected", sum(len(r.rejected) for r in run.extraction_reports)
    )

    report_tab, working_tab, rejected_tab = st.tabs(
        ["Report", "How it got there", "Rejected claims"]
    )

    with report_tab:
        if run.writer and not run.writer.passed_validation:
            st.error(
                "This report did not pass every check. The unresolved problems are printed "
                "at the end of it."
            )
        if run.report:
            st.markdown(run.report.to_markdown())
            st.download_button(
                "Download as Markdown",
                run.report.to_markdown(),
                file_name="report.md",
                mime="text/markdown",
            )

    with working_tab:
        st.subheader("Search plan")
        if run.used_fallback_plan:
            st.caption("The model's plan was unusable; the fallback plan was used.")
        for index, angle in enumerate(run.angles_planned, start=1):
            st.write(f"{index}. {angle}")

        st.subheader("Rounds")
        for record in run.rounds:
            with st.expander(
                f"Round {record.number}: {record.findings_added} finding(s) "
                f"from {record.sources_read} source(s) read"
            ):
                st.write(f"**Angle searched.** {record.angle}")
                st.write(
                    f"Found {record.sources_found}, of which {record.sources_new} were new; "
                    f"read {record.sources_read}; {record.unreadable_sources} could not be read."
                )
                st.write(f"Claims rejected in verification: {record.claims_rejected}")
                if record.stop_decision:
                    st.write(f"**Decision.** {record.stop_decision.reason}")

    with rejected_tab:
        rejections = [(r, item) for r in run.extraction_reports for item in r.rejected]
        if not rejections:
            st.write("No claims were rejected in this run.")
        st.caption(
            "Every claim below was proposed by the extraction step and then discarded, "
            "because it could not be checked against the source it came from."
        )
        for report, rejection in rejections:
            with st.expander(f"{', '.join(rejection.failed_checks)} — {report.source_url}"):
                st.write(f"**Claim.** {rejection.claim}")
                st.write(f"**Why it was rejected.** {rejection.detail}")


def main() -> None:
    st.title("Multi-Agent Research Assistant")
    st.caption(
        "Four agents: one searches, one reads and verifies, one writes, one decides when "
        "there is enough. Every claim in the report is checked against the source it cites."
    )

    settings = get_settings()
    has_key = bool(settings.anthropic_api_key)

    with st.sidebar:
        st.header("Settings")
        demo = st.checkbox(
            "Demo mode (offline)",
            value=not has_key,
            help="Runs the real pipeline against three built-in documents. No API key, no "
            "network, no cost.",
        )
        max_rounds = st.slider("Maximum search rounds", 1, 5, 3)
        if not has_key:
            st.info("No ANTHROPIC_API_KEY found, so demo mode is on.")
        st.caption(f"Web search provider: {settings.search_provider}")

    question = st.text_input(
        "Research question",
        value="What is the effect of a four-day work week on productivity?",
        placeholder="What does the evidence say about...",
    )

    if st.button("Research", type="primary"):
        if not question.strip():
            st.error("Enter a research question first.")
            return
        if not demo and not has_key:
            st.error("Set ANTHROPIC_API_KEY in your .env file, or turn on demo mode.")
            return

        with st.spinner("Searching, reading, verifying, writing..."):
            run = run_pipeline(question.strip(), demo, max_rounds)
        st.session_state["run"] = run

    if "run" in st.session_state:
        render_run(st.session_state["run"])


main()
