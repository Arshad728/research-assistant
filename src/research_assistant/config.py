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


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process from the current environment."""
    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        serper_api_key=os.environ.get("SERPER_API_KEY"),
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
    )
