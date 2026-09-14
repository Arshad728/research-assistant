"""Checks that the package can actually be installed and used (Phase 6).

An adversarial review found that `reportlab` was imported by the PDF
renderer but never declared in `pyproject.toml`. Every test passed, because
the development environment happened to have it installed. The README's
headline two-command quickstart crashed on a clean machine.

That is the worst kind of defect a test suite can miss: invisible to
everyone who already has the project working, and fatal to everyone who
does not. These tests read the declared dependencies and compare them with
what the code actually imports.
"""
import ast
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "research_assistant"

# Import name -> distribution name, where they differ.
DISTRIBUTION_NAMES = {
    "dotenv": "python-dotenv",
    "fitz": "pymupdf",
    "claude_agent_sdk": "claude-agent-sdk",
}

# Declared as optional extras rather than core requirements.
OPTIONAL_EXTRAS = {"streamlit", "pytest", "respx"}


def declared_dependencies() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)
    project = config["project"]

    def names(requirements):
        found = set()
        for requirement in requirements:
            name = requirement.split("==")[0].split(">=")[0].split("[")[0].strip()
            found.add(name.lower())
        return found

    core = names(project.get("dependencies", []))
    extras = set()
    for group in project.get("optional-dependencies", {}).values():
        extras |= names(group)
    return {"core": core, "extras": extras}


def third_party_imports() -> dict:
    """Every top-level module imported by the package, and where from."""
    stdlib = set(sys.stdlib_module_names)
    found: dict = {}

    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import, inside the package
                    continue
                modules = [(node.module or "").split(".")[0]]
            else:
                continue

            for module in modules:
                if not module or module in stdlib or module == "research_assistant":
                    continue
                found.setdefault(module, set()).add(path.relative_to(ROOT).as_posix())
    return found


class TestDeclaredDependencies:
    def test_everything_the_package_imports_is_declared(self):
        declared = declared_dependencies()
        all_declared = declared["core"] | declared["extras"]

        undeclared = {}
        for module, files in third_party_imports().items():
            distribution = DISTRIBUTION_NAMES.get(module, module).lower()
            if distribution not in all_declared:
                undeclared[module] = sorted(files)

        assert not undeclared, (
            "these modules are imported but not declared in pyproject.toml, so a clean "
            f"install will crash: {undeclared}"
        )

    def test_the_core_path_does_not_depend_on_an_optional_extra(self):
        """`research-assistant demo` must work without the `ui` extra."""
        declared = declared_dependencies()
        optional_only = declared["extras"] - declared["core"]

        offenders = {}
        for module, files in third_party_imports().items():
            distribution = DISTRIBUTION_NAMES.get(module, module).lower()
            if distribution not in optional_only:
                continue
            core_files = [f for f in files if "/ui/" not in f]
            if core_files:
                offenders[module] = sorted(core_files)

        assert not offenders, (
            f"optional dependencies imported from the core package: {offenders}"
        )

    def test_the_console_entry_point_is_declared(self):
        with open(ROOT / "pyproject.toml", "rb") as handle:
            config = tomllib.load(handle)
        scripts = config["project"].get("scripts", {})
        assert scripts.get("research-assistant") == "research_assistant.cli:main"


class TestEntryPointWorks:
    def test_the_cli_module_exposes_every_documented_subcommand(self):
        from research_assistant.cli import build_parser

        parser = build_parser()
        subparsers = [
            action for action in parser._actions if hasattr(action, "choices") and action.choices
        ]
        assert subparsers, "the CLI has no subcommands"
        commands = set(subparsers[0].choices)

        documented = {"check", "run", "demo", "search", "read", "write", "evaluate"}
        assert documented <= commands, f"missing subcommands: {documented - commands}"

    @pytest.mark.parametrize(
        "command", ["check", "run", "demo", "search", "read", "write", "evaluate"]
    )
    def test_every_subcommand_parses_its_own_help(self, command):
        from research_assistant.cli import build_parser

        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args([command, "--help"])
        assert exit_info.value.code == 0
