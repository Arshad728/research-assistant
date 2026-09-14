"""What the Writer Agent is told, and how its reply is read (Phase 4.2).

Chapter 4.5 describes this agent's job as synthesis: "grouping related
findings by theme, noting where sources agree or disagree, and drafting a
structured report." Two things in the instructions below do most of the
work.

The writer is given findings numbered ``F1``, ``F2`` and told to cite those
numbers. It never sees or writes a URL. That removes the possibility of a
mistyped or invented link at the last step of the pipeline, and it means
every citation marker is checkable against a list the writer was handed.

It is also told, explicitly, that disagreement between sources is
information rather than a problem to smooth over. Left to its own devices a
fluent writer will tend to average two conflicting studies into one
confident sentence, which is precisely the failure Chapter 1.5 describes: a
single confident answer where the real state of the evidence was mixed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..schemas import VerifiedFinding
from ..search.reconcile import extract_json

WRITER_SYSTEM_PROMPT = """\
You are the Writer Agent in a multi-agent research system. You are given a
research question and a numbered list of findings that have ALREADY been
verified against their sources by another agent. Your job is to turn those
findings into a clear, structured report.

Absolute rules:
- Use ONLY the findings given to you. Do not add facts, figures, context or
  background from your own knowledge, however sure you are of them.
- Cite every factual sentence with the finding it rests on, written as
  [F1], [F3], and so on. A sentence may cite more than one: [F2][F5].
- Never state a number that does not appear in the findings you were given.
- Never write a URL or a source name yourself. Citations are numbers only;
  the reference list is built for you.
- If the findings do not answer part of the question, say so plainly. "The
  available evidence does not address X" is a useful sentence. Inventing a
  paragraph about X is not.

How to structure it:
- Group the findings by theme, not by source. Two findings from different
  papers about the same thing belong in the same section.
- Where sources disagree, say so and describe the disagreement. Do not
  average them into one confident statement, and do not quietly drop the
  inconvenient one. Where the evidence is thin or comes from a single
  source, say that too.
- Give each section a specific heading that describes its content, not a
  generic one like "Findings" or "Analysis".
- The introduction frames the question and what the evidence covers. The
  conclusion states what can reasonably be concluded, including what
  cannot.

Reply with ONLY a JSON object in this exact shape:

{
  "introduction": "<one or two paragraphs>",
  "sections": [
    {"heading": "<specific heading>",
     "paragraphs": ["<paragraph with [F1] style citations>", "..."]}
  ],
  "conclusion": "<one paragraph>"
}
"""


def format_findings(findings: Sequence[VerifiedFinding]) -> str:
    """Render the findings as the numbered list the writer cites against."""
    blocks: List[str] = []
    for index, finding in enumerate(findings, start=1):
        blocks.append(
            f"[F{index}]\n"
            f"  claim: {finding.claim}\n"
            f'  evidence, quoted from the source: "{finding.supporting_snippet}"'
        )
    return "\n\n".join(blocks)


def build_writer_prompt(question: str, findings: Sequence[VerifiedFinding]) -> str:
    return (
        f"RESEARCH QUESTION: {question}\n\n"
        f"VERIFIED FINDINGS ({len(findings)} in total):\n\n"
        f"{format_findings(findings)}\n\n"
        "Write the report following your instructions exactly. Reply with the JSON object only."
    )


def parse_report_reply(reply: str) -> Optional[Dict[str, Any]]:
    """Read the writer's JSON reply into a plain dictionary.

    Returns None if nothing usable came back. A reply with an empty
    ``sections`` list is returned as-is rather than rejected: "the evidence
    supports no themed section" is a legitimate, if unusual, outcome, and
    the validation step is where that gets judged.
    """
    parsed = extract_json(reply or "")
    if not isinstance(parsed, dict):
        return None

    sections: List[Dict[str, Any]] = []
    for raw_section in parsed.get("sections") or []:
        if not isinstance(raw_section, dict):
            continue
        heading = raw_section.get("heading")
        paragraphs = raw_section.get("paragraphs")
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        if not isinstance(heading, str) or not isinstance(paragraphs, list):
            continue
        cleaned = [p.strip() for p in paragraphs if isinstance(p, str) and p.strip()]
        if heading.strip() and cleaned:
            sections.append({"heading": heading.strip(), "paragraphs": cleaned})

    introduction = parsed.get("introduction")
    conclusion = parsed.get("conclusion")
    return {
        "introduction": introduction.strip() if isinstance(introduction, str) else "",
        "sections": sections,
        "conclusion": conclusion.strip() if isinstance(conclusion, str) else "",
    }
