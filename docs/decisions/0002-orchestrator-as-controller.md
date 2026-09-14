# Decision Record 0002: The Orchestrator is a controller, not an improvising agent

**Phase:** 5.1 — Orchestration and Integration
**Status:** Accepted
**Date:** 2026-09-13

## Context

Chapter 4.2 of the book describes the Orchestrator as an agent with three jobs: planning the
sub-tasks, sequencing the other three agents, and deciding when enough evidence has been
gathered. Decision Record 0001 chose the Claude Agent SDK partly because it has first-class
support for named subagents, which makes a literal reading of that description easy to build:
an Orchestrator agent that is handed the other three as subagents and decides, turn by turn,
which to invoke next.

Phase 5 had to choose between that and a plainer arrangement: ordinary Python control flow
that calls each agent in sequence, with a model used only for the planning step.

## Options considered

**A. Orchestrator as an LLM agent with subagents.** Closest to the book's wording, and to how
"multi-agent system" is usually demonstrated. The Orchestrator reasons about what to do next
and delegates. Sequencing and stopping are things it decides.

**B. Orchestrator as a deterministic controller with an LLM planning step.** The model breaks
the question into search angles, because that is judgement. Everything after that — which
agent runs next, how many sources to read, when to stop — is code.

## Decision

**Option B.** The pipeline in `orchestration/pipeline.py` is ordinary control flow. The model
plans; it does not drive.

The reasoning is the same argument this project has made at every layer, applied one level
higher. This system's entire value proposition is that its output can be trusted more than a
chatbot's. Chapter 8.2 puts it as "every claim in the final report can be traced back to a
real source." A pipeline whose control flow is itself a model's improvisation undermines that
in three specific ways:

1. **It cannot be reproduced.** The same question could run three rounds today and one
   tomorrow, for reasons nobody can inspect. When a report comes out thin, "why did it stop?"
   should have an answer better than "the model decided to."

2. **It cannot be tested.** The stopping rule is the single most consequential piece of
   behaviour in the system — Chapter 4.2 calls it "one of the harder problems in the whole
   system" — and as code it can be unit tested against every combination of evidence and
   round count. As an agent's judgement it can only be sampled.

3. **It costs more to do worse.** An orchestrating agent spends tokens on every routing
   decision. A `for` loop does not, and gets the ordering right every time.

None of this argues that agentic orchestration is a bad pattern generally. It argues that
this system's specific requirement — that the process behind a claim be explainable — is
better served by a process that does not vary. The subagent primitives that made the SDK
attractive in Decision Record 0001 are still what run each individual agent; what changed is
that nothing delegates *between* them.

## Consequences

- Sequencing and stopping are covered by unit tests, and the stopping thresholds live in one
  `StoppingRule` object where they can be tuned without touching control flow.
- A run produces a `ResearchRun` record showing every round, what it found, and the stopping
  decision with its reason. That is an audit trail, not a log.
- The system cannot invent a novel strategy mid-run — for example, deciding to read one source
  very closely instead of five shallowly. That flexibility is genuinely lost. It is a
  reasonable trade for a first version whose job is to be trustworthy rather than clever.

**Revisit this if** the pipeline starts needing genuinely adaptive behaviour that cannot be
expressed as a rule — for instance, choosing different strategies for different *kinds* of
question. At that point a hybrid is likely better than either extreme: an agent that selects
among a few known-good deterministic strategies, rather than one improvising the whole flow.
