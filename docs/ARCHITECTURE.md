# Architecture

This document explains how the system is put together and, more usefully, why it is put
together this way. The book that accompanies this repository explains the concepts from first
principles; this assumes you already know roughly what an AI agent is and want to understand
this particular design.

## The problem the design is answering

A general-purpose chatbot answers a research question by predicting plausible text. Most of the
time the answer is right. There is no step in which anything is checked, so when it is wrong it
is wrong in a way that reads exactly like being right: a confident paragraph, a well-formatted
citation, and nothing behind either.

Every structural decision below follows from one goal: a claim in the finished report should be
traceable to something a person could go and read. Not "usually accurate" — traceable.

## The four agents

```
                        User's research question
                                  |
                                  v
                    +-----------------------------+
                    |     Orchestrator            |
                    |  plans, sequences, stops    |
                    +-----------------------------+
                       |            |           |
              (1) search      (2) verify    (3) write
                       |            |           |
                       v            v           v
              +------------+ +-------------+ +-----------+
              |   Search   | | Extraction  | |  Writer   |
              |   Agent    | |     and     | |   Agent   |
              |            | | Verification| |           |
              +------------+ +-------------+ +-----------+
                    |               |              |
              web / arXiv /    PDF & HTML     Markdown + PDF
              Semantic Scholar  parsing        with citations
```

Each agent has one job, and each hand-off between them is a checkpoint where the receiving side
does not have to trust what it was given.

| Agent | Job | Cannot |
|---|---|---|
| Orchestrator | Plan search angles, sequence the others, decide when there is enough | Read sources or write prose |
| Search | Turn an angle into real retrieved candidate sources | Make claims about what they say |
| Extraction and Verification | Read each source, extract claims, check each against the source text | Write the report |
| Writer | Group verified findings by theme and draft a cited report | See a URL, or any unverified claim |

## The one idea that runs through everything

**The model supplies judgement. Code supplies the invariants.**

This split appears at every layer, and it is the thing worth understanding about this codebase:

| Layer | The model decides | Code guarantees |
|---|---|---|
| Search (`search/`, `tools/`) | Which tool to use, how to phrase a query, which results are worth keeping | The same query is never run twice against the same tool in one run; no URL enters the shortlist that a tool did not return |
| Extraction (`extraction/`) | Which claims a source supports, and which passage evidences each | The quotation appears in the source; the claim's numbers appear in the quotation |
| Writing (`report/`) | Themes, structure, prose, which findings matter | Every citation resolves; every figure traces to evidence; unresolved problems are printed on the report |
| Orchestration (`orchestration/`) | How to break a question into search angles | Which agent runs when, how many sources are read, when to stop |

A language model asked to be careful is usually careful. A `for` loop is always a `for` loop.
Where the difference between "usually" and "always" is the entire product, the guarantee belongs
in code — see `docs/decisions/0002-orchestrator-as-controller.md` for the fullest version of
this argument.

A consequence worth stating: **the pipeline runs end to end with no API key and no network.** The
Extraction Agent, the Writer Agent and the Orchestrator's planning step each take a `complete`
function, so their model calls can be scripted. The Search Agent is the exception: it drives a
tool-calling loop through the SDK client, so it is substituted wholesale rather than through a
completion function (see `demo.py`). That is why failure modes like "the extractor fabricated a
quotation" or "the writer invented a statistic" are triggered deliberately in the test suite
rather than waited for, while the Search Agent's own SDK loop is covered only by the opt-in
integration tests.

## Package layout

```
src/research_assistant/
  config.py         environment and API keys, read lazily so a missing key fails with instructions
  schemas.py        CandidateSource, VerifiedFinding, SharedState — the hand-off shapes
  retrieval/        one client per source (arXiv, Semantic Scholar, web), each split into
                    an HTTP half and a pure parsing half so parsing is testable offline
  tools/            thin adapters exposing retrieval to the Claude Agent SDK as callable tools
  search/           query planning, repeat detection, deduplication, shortlist reconciliation
  extraction/       fetching, PDF/HTML parsing, and the verification checks
  report/           report structure, citation numbering, validation, Markdown and PDF rendering
  orchestration/    search-angle planning, the stopping rule, and the pipeline
  agents/           the four agents; each one is thin, because the logic lives in the modules above
  evaluation/       test queries, metrics, and the spot-check worksheet generator
  ui/               the optional Streamlit interface
  cli.py            the command line
  demo.py           an offline pipeline: real orchestration and verification, scripted models
```

The agents are deliberately the thinnest files in the project. Anything that can be tested
without a model lives below them.

## Data flow

1. **Plan.** The Orchestrator asks a model for 2–4 search angles. If the reply is unusable, a
   deterministic fallback derives angles from the question itself.
2. **Search.** For each round, the Search Agent runs its tools. Results become `CandidateSource`
   records. The model selects which to keep; the shortlist is then *rebuilt* from the tool
   records, so a URL the model named but no tool returned is dropped and counted.
3. **Deduplicate.** Sources already seen this run are skipped. arXiv identifiers are
   canonicalised, so the same paper reached via `/abs`, `/pdf`, or a Semantic Scholar record
   counts once.
4. **Read and verify.** Each new source is fetched (arXiv abstract pages are rewritten to the
   PDF), parsed to text, and passed to the extraction model. Every proposed claim must pass five
   mechanical checks before becoming a `VerifiedFinding`. Failures are recorded with reasons.
5. **Decide.** The stopping rule looks at the shared state: enough findings from enough distinct
   sources, out of rounds, or a round that found nothing new. Every decision carries a reason.
6. **Write.** The Writer Agent receives findings labelled `F1`, `F2`… and cites those. Citation
   numbers are assigned per source afterwards. The finished report is validated; failures go back
   as revision instructions; anything unresolved is printed on the report.

## The verification checks

These are in `extraction/verify.py` and are the heart of the system. All five must pass:

| Check | Catches |
|---|---|
| `claim_present` | A quotation with no claim attached |
| `snippet_length` | A fragment too short to evidence anything |
| `snippet_grounded` | A fabricated quotation (literal search of the source, with normalisation and a fuzzy fallback) |
| `numeric_support` | A real quotation paired with a claim stating a figure the quotation never mentions |
| `shared_vocabulary` | A real quotation attached to an unrelated claim |

None of them asks a model whether the model was being careful. They are string operations.

Two details in there are load-bearing, and both were wrong in an earlier version until an
adversarial review found them:

**The last two checks run against the matched source text, not the text the model submitted.**
Fuzzy matching has to tolerate a quotation that differs slightly from its source, or honest
quotations get rejected. That tolerance is exploitable: submit a quotation with one digit
changed — "output rose 80%" against a source saying 8% — and it still matches at 0.99. If the
numeric check then ran on the submitted text, the 80 would look supported while the citation
stored and shown to the reader said 8%. Checking the matched text closes it: the evidence a
check passes against is the evidence the reader sees.

**Numbers glued to words by a hyphen do not count.** The regex used to read `19` out of
`COVID-19`, so any year or model name in a quotation licensed that figure in a claim.

Strictness is deliberate, following the book's advice that loosening a cautious check later is
easier than discovering a lax one let fabrications through. Two things make that strictness
usable: text is normalised before comparison, so typographic quotes, en-dashes and ligatures do
not cause false rejections; and PDF line-break hyphenation is repaired at parse time, so a
genuine quotation spanning a column break can still be found.

## Known limits

- **Small integers escape the report-level numeric check.** Reports legitimately say "three
  themes emerged," and demanding evidence for those would produce false alarms that train a
  reader to ignore warnings. Percentages, decimals and numbers ≥ 10 are always checked. The
  finding-level check is stricter and runs first.
- **Verification proves grounding, not fair reading.** This is the deepest limit in the project
  and deserves more than a line. Every check is a string operation, so none of them understands
  meaning. All of the following pass every check, because the quotation is genuinely in the
  source and the vocabulary and figures line up:
  - a claim that **reverses** its source ("remote work does not increase productivity", quoted
    against a sentence reporting an 8% increase);
  - a claim that **attributes to the authors** a view the source describes in order to reject it;
  - a claim that **drops a qualifier** ("we caution that this should not be read as a general
    estimate");
  - a **spelled-out** number ("forty-five percent"), which contains no digits for the numeric
    check to compare;
  - a **unit or magnitude swap** ("8x" against a source saying "8%").

  These are not thresholds to tighten. They are the boundary of what string comparison can
  establish, and they are why the evaluation ends with a worksheet for a human rather than a
  score. A reader who wants a claim to be *fairly* read still has to read the source.
- **A long fabrication is easier to pass than a short honest quotation.** Similarity is
  proportional, so a 500-character quotation with an invented clause inserted can score above
  the threshold while an 80-character genuine quotation missing a citation marker scores below
  it. The numeric check catches the common case of an invented figure; it does not catch
  invented prose.
- **Paywalled sources are out of scope.** The system reads what it can reach.
- **The stopping rule is a heuristic.** "Four findings from two sources" is a starting value in
  one tunable object, not a discovered truth.

## Testing

```bash
pytest                 # offline: no network, no keys, no cost
pytest -m integration  # opt-in: hits the real APIs, needs keys
```

The offline suite covers the whole system, including the failure paths, because every model call
is injectable and every HTTP call is interceptable. The integration suite exists to catch a
different kind of failure: an external API changing its response shape or auth scheme underneath
us. It is opt-in rather than skipped-on-failure, because tests that fail for environmental
reasons teach you to ignore failures.
