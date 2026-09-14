#!/usr/bin/env python3
"""Thin wrapper kept for convenience. The packaged command is:

    research-assistant run "<question>"
    research-assistant demo
"""
import sys

from research_assistant.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    command = "demo" if "--demo" in argv else "run"
    raise SystemExit(main([command] + [a for a in argv if a != "--demo"]))
