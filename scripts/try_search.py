#!/usr/bin/env python3
"""Thin wrapper. The packaged command is: research-assistant search "<query>" """
import sys

from research_assistant.cli import main

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("--tools", "--agent")]
    if not args:
        args = ["effect of a four day work week on productivity"]
    raise SystemExit(main(["search"] + args))
