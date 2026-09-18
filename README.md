# Multi-Agent Research Assistant

Four cooperating AI agents that take a research question, search academic papers and the web,
verify what they find against the original sources, and write a structured report with citations.

The point of the system is not that it writes reports. It is that every claim in a report can be
traced back to a passage in a real source, because claims that cannot be are discarded before
they reach the page.

```bash
pip install -e .
research-assistant demo          # the whole pipeline, offline, no API key
```

That command runs the real orchestration, the real extraction and the real verification against
three built-in documents. It prints the plan, each search round, why it stopped and every claim
that failed verification, then writes the finished report to `demo_report.md` and
`demo_report.pdf`.

Try `research-assistant demo --unanswerable` too. It asks about something that does not exist,
and shows the system declining to answer rather than producing a confident report — which is the
behaviour the whole design is for.

## What it does

```
question -> Orchestrator -> Search Agent -> Extraction & Verification -> Writer -> cited report
                 ^                                    |
                 +------- not enough evidence? -------+
```

- The **Search Agent** queries arXiv, Semantic Scholar and the web, and cannot report a source
  that no tool actually returned.
- The **Extraction and Verification Agent** reads each source and checks every claim against the
  passage quoted to support it. A claim whose quotation is not in the source, or which states a
  number the quotation never mentions, is discarded with a recorded reason.
- The **Writer Agent** groups verified findings by theme. It is never shown a URL, so it cannot
  invent one; the reference list is assembled from the findings.
- The **Orchestrator** plans the searches and decides when there is enough evidence. That
  decision is ordinary code, not a model's judgement, so it is reproducible and testable.

See `docs/ARCHITECTURE.md` for the design and `docs/decisions/` for why each significant choice
was made. The accompanying book explains the whole thing from first principles, and its Build Log
appendix records how it was actually built, mistakes included.

## Requirements

- Python 3.10 or newer.
- API keys, for the commands that reach the internet or call a model: `run`, `read` (without
  `--no-model`), `write` (without `--demo`) and `evaluate` (without `--demo`). Installing the
  package, running `check`, `demo`, `search`, and the `--demo`/`--no-model` variants needs no
  keys at all.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ui for the web interface: ".[dev,ui]"
cp .env.example .env               # then fill in your keys
research-assistant check
```

`check` reports your Python version, which packages are installed, which keys are present, and
whether the keyless APIs are reachable from your network. Nothing here requires having every key
on day one.

### Where to get each key

| Key | Where | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) | Set this, `GEMINI_API_KEY`, or both — whichever is active runs every agent. Pick with `--provider` / the web UI's model dropdown; with both set, Gemini is used by default. |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free tier, no card | See above. |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com/) — free tier is 1,000 searches/month, no card | Yes, unless using Serper. |
| `SERPER_API_KEY` | [serper.dev](https://serper.dev/) | Alternative to Tavily; set only one. |
| `SEMANTIC_SCHOLAR_API_KEY` | [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api#api-key) | Technically optional, practically needed: the keyless pool throttles `/paper/search` heavily. |
| arXiv | No key. The API is open. | — |

## Commands

```bash
research-assistant check                      # environment and API diagnostics
research-assistant run "<question>"           # the whole pipeline -> report.md + report.pdf
research-assistant demo                       # the whole pipeline, offline, no keys
research-assistant demo --unanswerable        # watch it correctly refuse to answer
research-assistant search "<query>"           # retrieval only: no model, no cost
research-assistant read <url> --no-model      # fetch and parse one source, no cost
research-assistant read <url>                 # read it and verify claims from it
research-assistant write findings.json        # write a report from verified findings
research-assistant write --demo               # see the report format with no keys
research-assistant evaluate --demo            # run the test-query evaluation offline
```

Every command that costs money has a mode that does not, because a tool nobody can try before
committing a credit card does not get tried.

### Web interface (optional)

```bash
pip install -e ".[ui]"
streamlit run src/research_assistant/ui/app.py
```

The page shows the report alongside the things a polished product would hide: the search plan,
each round, why the run stopped, and every rejected claim. For this system those are not
diagnostics — they are the reason to trust the report.

## Evaluation

```bash
research-assistant evaluate
```

Runs six test queries. Two of them are deliberately about subjects that are barely studied or do
not exist: a system that produces a confident report for those is fabricating, so finding nothing
is the correct outcome and is scored as such.

It writes `evaluation.md` with the mechanical measures and a **spot-check worksheet**. Citation
accuracy is not in the summary numbers, because a system cannot honestly grade its own citations
— the worksheet puts a sample of claims, their quoted evidence and their links in front of a
person, which is where that number has to come from. `examples/evaluation.md` is a sample run.

## Tests

```bash
pytest                 # offline: no network, no keys, no cost
pytest -m integration  # opt-in: hits the real APIs, needs keys
```

Every model call is injectable and every HTTP call interceptable, so the offline suite covers the
failure paths too — fabricated quotations, invented statistics, dead links, unreadable PDFs, a
writer that strays outside its evidence.

## Project layout

```
src/research_assistant/
  config.py         environment and API keys
  schemas.py        the shared data shapes every hand-off uses
  retrieval/        arXiv, Semantic Scholar and web search clients
  tools/            those clients, exposed to the Claude Agent SDK
  search/           query planning, deduplication, shortlist reconciliation
  extraction/       fetching, PDF/HTML parsing, claim verification
  report/           report structure, citations, validation, Markdown and PDF
  orchestration/    planning, the stopping rule, the pipeline
  agents/           the four agents
  evaluation/       test queries, metrics, spot-check worksheet
  ui/               optional Streamlit interface
  cli.py            the command line
  demo.py           the offline pipeline
docs/               ARCHITECTURE.md and decision records
examples/           a sample report, a recorded run, a sample evaluation
scripts/            thin wrappers around the CLI, kept for convenience
tests/              the test suite
```

## What this system does not do

Worth knowing before you trust a report it produced. Verification proves that a quotation is
really in its source and that the figures in a claim are really in the quotation. It does not
prove the claim is a *fair reading* of the source: a claim that reverses its source's finding,
or drops a qualifier, or attributes to the authors a view they were rejecting, passes every
check, because every check is a string comparison. `docs/ARCHITECTURE.md` lists these limits in
full. They are the reason the evaluation ends with a worksheet for a human instead of a score.
