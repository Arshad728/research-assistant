#!/usr/bin/env python3
"""Thin wrapper. The packaged command is: research-assistant read <url>"""
import sys

from research_assistant.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["read"] + sys.argv[1:]))
