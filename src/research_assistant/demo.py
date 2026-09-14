"""An offline demonstration of the whole system (Phase 6.2).

Everything here exists so the pipeline can be run and watched without an API
key, a search key, or a network connection. What is stood in for is narrow
and deliberate: the searching, the downloading, and the model calls. The
orchestration, the extraction, the verification, the citation numbering and
the report validation are all the real code paths.

That matters for a demonstration of this particular system, because the
interesting behaviour is not the report at the end -- it is the control flow
and the checking that produced it, and those are exactly the parts a faked
demo would normally skip.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from .agents.extraction_agent import ExtractionAgent
from .agents.writer_agent import WriterAgent
from .extraction.documents import SourceDocument
from .orchestration import ResearchPipeline, StoppingRule
from .schemas import CandidateSource

DEMO_QUESTION = "What is the effect of a four-day work week on productivity?"

DEMO_DOCUMENTS: Dict[str, str] = {
    "https://example.org/trial": (
        "Across the trial period, hourly productivity increased by 8%, while total weekly "
        "output remained statistically unchanged across the participating organisations. "
        "Participating organisations numbered 61 in total, spread across four countries and "
        "covering sectors from software to manufacturing and professional services."
    ),
    "https://example.org/survey": (
        "Respondents reported markedly higher satisfaction scores under the shorter working "
        "schedule than under their previous arrangements, with most citing reduced fatigue. "
        "The survey covered the same programme as the trial and relies entirely on "
        "self-reported measures rather than on observed output."
    ),
    "https://example.org/critique": (
        "The absence of a control group means the observed changes cannot be attributed to the "
        "schedule change with confidence, a limitation common to workplace trials of this kind. "
        "Several participating organisations had already begun other efficiency programmes "
        "before the trial began, which complicates attribution further."
    ),
}

DEMO_SOURCES = [
    CandidateSource(
        title="Multi-Site Trial",
        source_url="https://example.org/trial",
        source_type="journal article",
        relevance_note="Measures output before and after the schedule change.",
    ),
    CandidateSource(
        title="Participant Survey",
        source_url="https://example.org/survey",
        source_type="web page",
        relevance_note="Self-reported outcomes from the same programme.",
    ),
    CandidateSource(
        title="A Critical Read",
        source_url="https://example.org/critique",
        source_type="web page",
        relevance_note="Notes methodological limits of the trial.",
    ),
]

# A query the demo should visibly fail to answer, so the honest-failure path
# can be watched as easily as the success path.
DEMO_UNANSWERABLE = "What are the performance characteristics of the Zephyrine consensus protocol?"


# What the three built-in documents are actually about. A demo search that
# returned them for any question at all would make the system look like it
# fabricates answers, which is the opposite of what there is to demonstrate.
_CORPUS_TOPIC = {
    "four", "day", "week", "work", "working", "hours", "schedule", "shorter",
    "productivity", "output", "satisfaction", "employees", "workplace",
}


class DemoSearchAgent:
    """Returns the built-in sources only for questions they actually bear on.

    The interesting half of this system is what it does when there is nothing
    to find, so the demo has to be able to find nothing. Anything outside the
    three built-in documents' subject returns no sources, and the run then
    takes the honest-failure path all the way through to a report that says
    it has no evidence.
    """

    async def find_sources(self, query: str):
        from .agents.search_agent import SearchOutcome
        from .search.planner import keywords

        if keywords(query) & _CORPUS_TOPIC:
            return SearchOutcome(query=query, sources=list(DEMO_SOURCES))
        return SearchOutcome(query=query, sources=[])


class DemoExtractionAgent(ExtractionAgent):
    """Real extraction and real verification; only the download is skipped."""

    async def extract_from_source(self, source: CandidateSource, question: str):
        text = DEMO_DOCUMENTS.get(source.source_url)
        if text is None:
            return await super().extract_from_source(source, question)
        document = SourceDocument(
            source_url=source.source_url, title=source.title, text=text, content_kind="html"
        )
        return await self.extract_from_document(document, question)


def summarise_sentence(sentence: str, limit: int = 110) -> str:
    """Shorten a sentence on a word boundary, the way a careful extractor would."""
    clean = sentence.strip().rstrip(".")
    if len(clean) <= limit:
        body = clean
    else:
        body = clean[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    return f"The source states that {body[0].lower()}{body[1:]}."


def demo_completion():
    """Scripted stand-ins for the three kinds of model call the pipeline makes."""

    async def _complete(system_prompt: str, user_prompt: str) -> str:
        if "Orchestrator" in system_prompt:
            # Derive the angles from the question actually asked. A planner
            # that ignored its input would make every demo run look alike and
            # would hide the behaviour worth demonstrating.
            from .search.planner import SearchPlanner

            question = user_prompt.split("RESEARCH QUESTION:")[-1].split("\n")[0].strip()
            planner = SearchPlanner(original_query=question)
            angles = [planner.broaden(question, keep=5), planner.broaden(question, keep=3)]
            return json.dumps([a for a in dict.fromkeys(angles) if a])

        if "Extraction and Verification Agent" in system_prompt:
            body = user_prompt.split("<<<SOURCE_START>>>")[-1].split("<<<SOURCE_END>>>")[0]
            sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", body) if len(s.strip()) > 80]
            return json.dumps(
                [
                    {"claim": summarise_sentence(s), "supporting_snippet": s}
                    for s in sentences[:2]
                ]
            )

        findings = re.findall(r"\[F(\d+)\]\n  claim: (.+)", user_prompt)
        if not findings:
            return json.dumps({"introduction": "", "sections": [], "conclusion": ""})
        paragraphs: List[str] = [f"{claim.strip()} [F{n}]" for n, claim in findings]
        half = max(1, len(paragraphs) // 2)
        return json.dumps(
            {
                "introduction": "This report summarises the verified evidence gathered.",
                "sections": [
                    {"heading": "What the evidence shows", "paragraphs": paragraphs[:half]},
                    {"heading": "Caveats and limitations", "paragraphs": paragraphs[half:]},
                ],
                "conclusion": f"The evidence rests on {len(findings)} verified finding(s) "
                f"[F{findings[0][0]}].",
            }
        )

    return _complete


def build_demo_pipeline(*, max_rounds: int = 3) -> ResearchPipeline:
    """A fully wired pipeline that needs nothing from the outside world."""
    complete = demo_completion()
    return ResearchPipeline(
        search_agent=DemoSearchAgent(),
        extraction_agent=DemoExtractionAgent(complete=complete),
        writer_agent=WriterAgent(complete=complete, max_revisions=1),
        stopping_rule=StoppingRule(max_rounds=max_rounds),
        complete=complete,
    )
