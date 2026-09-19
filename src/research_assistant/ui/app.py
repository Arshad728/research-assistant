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
import os

import streamlit as st

from research_assistant.config import get_settings

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="*", layout="wide")

_SECRET_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "SERPER_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
)


def _get_secret(key: str) -> str | None:
    """Read one key from ``st.secrets`` without ever raising.

    A plain dict's ``.get()`` returns ``None`` for a missing key. Streamlit's
    does not: when *no* secrets have been configured at all -- true both
    running locally with no ``secrets.toml`` and on a freshly deployed
    Streamlit Community Cloud app before anything has been pasted into its
    Secrets box -- ``st.secrets.get(...)`` raises ``StreamlitSecretNotFoundError``
    instead of returning ``None``, which took the whole page down before it
    could render anything. This normalises that to "no secret set", the same
    outcome a real dict would give, so the app is never one crash away from
    a config step nobody has done yet.
    """
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def _load_secrets_into_env() -> None:
    """Make Streamlit Community Cloud's secrets visible the same way a local ``.env`` is.

    Deployed on Streamlit Community Cloud, API keys live in the dashboard's
    "Secrets" box (``st.secrets``), not in a ``.env`` file -- ``.env`` is
    gitignored and never leaves this machine. ``get_settings()`` reads
    ``os.environ`` either way, exactly like the CLI does, so this copies any
    of the keys above from ``st.secrets`` into the environment before
    settings are read. Running locally with no ``secrets.toml``, ``st.secrets``
    is empty and this does nothing -- the ``.env`` file keeps working as before.
    """
    for key in _SECRET_ENV_KEYS:
        value = _get_secret(key)
        if value and not os.environ.get(key):
            os.environ[key] = value


def _check_password() -> bool:
    """Gate the page behind a password, but only when one is actually configured.

    ``APP_PASSWORD`` is meant to be set once, in the Streamlit Community
    Cloud secrets box, for a link that anyone on the internet could open --
    it stops a stranger from burning through your API quota, nothing more.
    Locally, with no ``secrets.toml``, nothing is configured and this
    returns ``True`` immediately, so local use is unaffected.
    """
    configured = _get_secret("APP_PASSWORD")
    if not configured:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("Multi-Agent Research Assistant")
    st.caption("This deployment is password-protected.")
    entered = st.text_input("Password", type="password", key="password_entry")
    if st.button("Enter"):
        if entered == configured:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def run_pipeline(question: str, demo: bool, max_rounds: int, provider: str = "claude"):
    """Run one research question and return the completed run."""
    if demo:
        from research_assistant.demo import build_demo_pipeline

        pipeline = build_demo_pipeline(max_rounds=max_rounds)
    else:
        from research_assistant.orchestration import ResearchPipeline, StoppingRule

        pipeline = ResearchPipeline(
            stopping_rule=StoppingRule(max_rounds=max_rounds), provider=provider
        )
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
    _load_secrets_into_env()
    if not _check_password():
        return

    st.title("Multi-Agent Research Assistant")
    st.caption(
        "Four agents: one searches, one reads and verifies, one writes, one decides when "
        "there is enough. Every claim in the report is checked against the source it cites."
    )

    # get_settings() is @lru_cache'd -- deliberately, so the CLI (a fresh
    # process per run) doesn't re-read the environment on every call. But
    # Streamlit reruns this whole script many times in one long-lived
    # process, and _load_secrets_into_env() only just updated os.environ
    # above -- without clearing the cache first, a Settings computed on an
    # earlier rerun (e.g. before secrets were saved) would stick around for
    # the rest of the process's life. Clearing it here is cheap enough to do
    # on every rerun.
    get_settings.cache_clear()
    settings = get_settings()
    has_key = bool(settings.anthropic_api_key)
    has_gemini_key = bool(settings.gemini_api_key)
    has_groq_key = bool(settings.groq_api_key)
    has_any_key = has_key or has_gemini_key or has_groq_key

    with st.sidebar:
        st.header("Settings")
        demo = st.checkbox(
            "Demo mode (offline)",
            value=not has_any_key,
            help="Runs the real pipeline against three built-in documents. No API key, no "
            "network, no cost.",
        )
        provider_options = ["claude", "gemini", "groq", "ollama"]
        provider = st.selectbox(
            "Model (search, planning, extraction and writing all use this one)",
            options=provider_options,
            index=provider_options.index(settings.default_provider),
            help="Defaults to gemini when GEMINI_API_KEY is set (it has a free tier), groq "
            "next (also free, more generous daily limit), claude otherwise -- pick any of "
            "the four explicitly here. ollama needs no key but needs `ollama serve` running "
            "on this same machine.",
        )
        max_rounds = st.slider("Maximum search rounds", 1, 5, 3)
        if not has_any_key:
            st.info(
                "No ANTHROPIC_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY found, so demo mode "
                "is on."
            )
        if provider == "claude" and not has_key:
            st.warning("No ANTHROPIC_API_KEY found. Add one to your .env file to use Claude.")
        if provider == "gemini" and not has_gemini_key:
            st.warning("No GEMINI_API_KEY found. Add one to your .env file to use Gemini.")
        if provider == "groq" and not has_groq_key:
            st.warning("No GROQ_API_KEY found. Add one to your .env file to use Groq.")
        if provider == "ollama":
            st.info(
                "No key needed, but this only works when `ollama serve` is running on the "
                "same machine as this app. That's true running locally -- it is NOT true for "
                "an app deployed on Streamlit Community Cloud or similar, since 'localhost' "
                "there is the hosting server, not your computer."
            )
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
        if not demo and provider == "claude" and not has_key:
            st.error("Set ANTHROPIC_API_KEY in your .env file, switch the model to gemini or "
                      "groq, or turn on demo mode.")
            return
        if not demo and provider == "gemini" and not has_gemini_key:
            st.error("Set GEMINI_API_KEY in your .env file, switch the model to claude or "
                      "groq, or turn on demo mode.")
            return
        if not demo and provider == "groq" and not has_groq_key:
            st.error("Set GROQ_API_KEY in your .env file, switch the model to claude or "
                      "gemini, or turn on demo mode.")
            return

        with st.spinner("Searching, reading, verifying, writing..."):
            run = run_pipeline(question.strip(), demo, max_rounds, provider)
        st.session_state["run"] = run

    if "run" in st.session_state:
        render_run(st.session_state["run"])


main()
