"""Shared test fixtures.

Two things need neutralising for tests to be fast and independent: the
per-API rate limiters (which would otherwise add real seconds of sleeping
to a test run) and the cached settings object (which would otherwise carry
one test's fake API key into the next test).
"""
import pytest

from research_assistant import config
from research_assistant.retrieval import arxiv as arxiv_module
from research_assistant.retrieval import semantic_scholar as s2_module


@pytest.fixture(autouse=True)
def no_rate_limiting():
    """Run the real rate-limiter code path, but with a zero-second interval."""
    originals = {}
    for module in (arxiv_module, s2_module):
        limiter = module._RATE_LIMITER
        originals[limiter] = limiter._min_interval
        limiter._min_interval = 0.0
        limiter._last_call = None
    yield
    for limiter, interval in originals.items():
        limiter._min_interval = interval


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Make sure each test reads the environment fresh."""
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def no_api_keys(monkeypatch):
    for name in (
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "TAVILY_API_KEY",
        "SERPER_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    config.get_settings.cache_clear()
