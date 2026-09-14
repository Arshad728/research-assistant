# Decision Record 0001: Agent Framework and Tooling Selection

**Phase:** 1.1 — Setup and Research
**Status:** Accepted
**Date:** 2026-09-13

## Context

Chapter 4 of the book fixes the architecture this project has to run on top of: one Orchestrator
Agent and three specialist agents (Search, Extraction & Verification, Writer), coordinating through
structured hand-offs rather than free-form messages. Whatever framework is chosen has to make three
things easy: giving each agent its own role and tools, passing structured data between them at each
hand-off, and letting the Orchestrator control sequencing (including sending the Search Agent out
again). It does not need to make the language model itself pluggable — Chapter 5.1 already commits
this project to Claude as the model layer for every agent.

## Options considered

Three frameworks come up constantly in current (September 2026) discussion of multi-agent
research/reporting pipelines: the Claude Agent SDK, LangGraph, and CrewAI. A fresh comparison was run
for this decision rather than assuming the book's Chapter 5.2 summary was still current, since
framework trade-offs in this space move quickly.

| | CrewAI | LangGraph | Claude Agent SDK |
|---|---|---|---|
| Fastest to a first prototype | Yes — a "researcher" and "writer" role can be working in well under a day | No — explicit graph/state design takes longer up front | Yes — simple/single-agent start, subagents added incrementally |
| Determinism / good fit for a verification-heavy pipeline | Weakest of the three; reports of using up to ~3x the tokens of LangGraph on comparable workflows | Strongest — explicit state machine, persistent checkpointers, native `interrupt_before`/`interrupt_after` for human-in-the-loop validation | Strong, but earned differently — see Decision below |
| Model lock-in | None (multi-provider) | None (multi-provider) | Built around Claude specifically |
| Learning curve | Gentle | Steepest | Gentle to moderate |
| Typically recommended for | Solo builder / fast demo, migrate later if volume grows | Enterprise / regulated / high-volume, compliance-minded teams | Projects already committed to Claude as the model |

(Sources for this comparison are listed at the end of this record.)

Read at face value, this comparison could argue for LangGraph on the strength of its native
checkpointing, since "check the previous step before trusting it" is exactly this project's central
idea (Chapter 3.2's hand-offs, Chapter 4.4's verification step). It could also argue for CrewAI purely
on speed, since this is a solo, resume-oriented build with a seven-week target (Chapter 6), not an
enterprise deployment.

## Decision

**The Claude Agent SDK is confirmed as the framework**, consistent with Chapter 5.2's original choice,
but for a more specific reason than "it's from the same vendor as the model":

The determinism argument for LangGraph is a real advantage of that framework in general, but this
project does not actually need the framework to supply determinism. It supplies its own: the
Extraction and Verification Agent is a dedicated hand-off checkpoint by design (Chapter 4.4), whether
the underlying framework is a state machine or not. What the framework needs to do is get out of the
way of that architecture, not enforce it from below. On that narrower requirement, three things found
while installing and inspecting the current SDK (version 0.2.152, see below) matter more than they did
when Chapter 5.2 was first written:

- `AgentDefinition`, subagent lifecycle hooks (`SubagentStartHookInput`, `SubagentStopHookInput`), and
  `list_subagents` are now first-class primitives. The SDK has a native concept of "more than one named
  agent with its own role," rather than that structure being something to build manually on top of a
  single-agent loop.
- Custom tools are defined in-process with the `@tool` decorator and `create_sdk_mcp_server`, with no
  subprocess management. This maps directly onto the three tool groups Chapter 5 already calls for: web
  search, arXiv/Semantic Scholar, and PDF/HTML parsing.
- `PreToolUse` / `PostToolUse` hooks give an explicit place to enforce guardrails on what a tool call is
  allowed to do (relevant to the Search Agent's retrieval calls) — a smaller-grained version of the
  human-in-the-loop control that made LangGraph attractive above.

CrewAI is not chosen because the project's core value proposition is trustworthiness of the final
report, not speed of the first prototype; the token overhead and weaker determinism are a poor trade
for a system whose entire point is careful verification.

**Revisit this decision if:** daily query volume grows large enough that token cost becomes a binding
constraint, or the project later needs stronger built-in observability/state-machine guarantees than
SDK hooks provide. Chapter 5.2 already notes this swap would mean redoing Phase 1–2's integration work,
not rethinking the Chapter 4 architecture — that remains true and is the main reason this decision is
safe to revisit later without a rewrite.

## Verified facts (checked directly in this project's environment, not just from documentation)

- Package: `claude-agent-sdk` on PyPI. Latest version at the time of this decision: **0.2.152**.
  Requires Python 3.10+ (this environment has 3.11.15).
- **Gotcha found during setup:** an unpinned `pip install claude-agent-sdk` in this environment resolved
  to a stale cached version (0.1.73) even though 0.2.152 was available. `pip index versions
  claude-agent-sdk` confirmed the true latest. Lesson for Phase 1.2: always pin the exact version in
  `requirements.txt` and don't assume an unpinned install got the latest release — verify with `pip
  index versions` (or equivalent) if the installed version looks old.
- The package bundles its own CLI: a self-contained ~207 MB binary at
  `claude_agent_sdk/_bundled/claude`. No separate Node.js installation is required to use it, contrary
  to what older Claude Code tutorials might suggest.
- **Naming history:** this SDK was renamed in June 2026. Python: `claude_code_sdk` →
  `claude_agent_sdk`, and the options class `ClaudeCodeOptions` → `ClaudeAgentOptions`. TypeScript:
  `@anthropic-ai/claude-code` → `@anthropic-ai/claude-agent-sdk`. Anyone following an older tutorial
  that imports `claude_code_sdk` is reading pre-rename documentation. As of version 0.1.0+, the SDK also
  stopped auto-applying Claude Code's own coding-assistant system prompt by default — every agent in
  this project supplies its own role-specific prompt anyway (Chapter 4), so this default actually works
  in this project's favor rather than needing to be worked around.
- Import smoke test passed: `import claude_agent_sdk` succeeds and exposes `query`, `ClaudeSDKClient`,
  `ClaudeAgentOptions`, `AgentDefinition`, `tool`, `create_sdk_mcp_server`, and the hook/session/subagent
  types named above.
- Not tested here: an actual `query()` call, since that requires a real `ANTHROPIC_API_KEY` and spends
  the project owner's API credits. That is scoped to Phase 1.2 (Environment and API Setup), once API
  access is deliberately wired up.

## Sources

- [Claude Agent SDK for Python — PyPI](https://pypi.org/project/claude-agent-sdk/)
- [anthropics/claude-agent-sdk-python — GitHub](https://github.com/anthropics/claude-agent-sdk-python)
- [claude-code-sdk Import Errors After the June 2026 Rename: Why and the Fix](https://ofox.ai/blog/claude-agent-sdk-migration-import-errors-2026/)
- [Claude Agent SDK vs LangGraph vs CrewAI: Complete 2026 Benchmark](https://pasqualepillitteri.it/en/news/3095/claude-agent-sdk-vs-langgraph-vs-crewai-benchmark-2026-en)
