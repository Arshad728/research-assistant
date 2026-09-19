"""Configuration loading for the research assistant.

All configuration comes from environment variables, loaded from a local .env
file if one exists (see .env.example at the project root for the full list
and where to get each key). Nothing here requires every key to be present at
import time -- individual settings are read lazily, so pieces of the system
can be built and tested before every API key has been obtained.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()  # no-op if there is no .env file yet


class MissingAPIKeyError(RuntimeError):
    """Raised when code tries to use a key that has not been configured yet."""


class Settings(BaseModel):
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    ollama_host: Optional[str] = None
    tavily_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None

    def require(self, field_name: str, human_name: str, where: str) -> str:
        """Return a setting's value, or raise a clear, actionable error.

        Used by later agents/tools instead of touching the field directly, so
        a missing key fails with instructions instead of a confusing crash.
        """
        value = getattr(self, field_name)
        if not value:
            raise MissingAPIKeyError(
                f"{human_name} is not set. Add it to your .env file -- "
                f"get one at {where} -- then try again."
            )
        return value

    @property
    def search_provider(self) -> str:
        """Which web search provider is configured: 'tavily', 'serper', or 'none'.

        Chapter 5.3 of the book treats Tavily and Serper as interchangeable;
        this project only ever needs one of the two configured at a time.
        """
        if self.tavily_api_key:
            return "tavily"
        if self.serper_api_key:
            return "serper"
        return "none"

    @property
    def default_provider(self) -> str:
        """Which provider the whole pipeline uses when none is chosen explicitly.

        Search, planning, extraction and writing all follow this one choice.
        Prefers Gemini whenever a key for it is configured -- Gemini has a
        free tier and Claude does not, so that is the more useful default.
        Falls back to Groq next (also free, and far more generous per-day
        than Gemini's free tier, so it is a reasonable second choice rather
        than an immediate drop to the paid option), and only falls back to
        Claude when neither free key is configured. This only decides the
        *default*; --provider (CLI) and the model dropdown (UI) can still
        always override it explicitly, in any direction.

        Ollama is deliberately never chosen here, even when ``ollama_host``
        is set: unlike an API key, a configured host is not the same as a
        reachable server with the right model pulled, and this property is
        a plain field lookup with no network I/O to confirm one actually
        is. Ollama only runs when asked for explicitly (``--provider
        ollama`` / the UI's model dropdown).
        """
        if self.gemini_api_key:
            return "gemini"
        if self.groq_api_key:
            return "groq"
        return "claude"


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process from the current environment."""
    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        ollama_host=os.environ.get("OLLAMA_HOST"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        serper_api_key=os.environ.get("SERPER_API_KEY"),
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
    )
