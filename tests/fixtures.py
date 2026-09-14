"""Saved API response shapes used by the retrieval tests.

These are hand-written to match the documented response formats of each
API. They are deliberately messy in the ways real responses are -- extra
whitespace in arXiv titles, a result with no usable link, a paper with no
abstract -- so the parsers are tested against realistic input rather than
an idealised version of it.
"""

ARXIV_ATOM_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.01234v1</id>
    <published>2024-01-02T18:00:00Z</published>
    <updated>2024-01-05T09:30:00Z</updated>
    <title>Shorter Working Weeks and Measured Output:
      A Multi-Site Trial</title>
    <summary>  We report results from a multi-site trial of a four-day working week,
      measuring output per hour and total weekly output across eleven organisations
      over a twelve-month period. Output per hour rose while total output held level.
    </summary>
    <author><name>A. Researcher</name></author>
    <author><name>B. Collaborator</name></author>
    <arxiv:primary_category term="econ.GN" scheme="http://arxiv.org/schemas/atom"/>
    <category term="econ.GN" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2401.01234v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.01234v1" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.05678v2</id>
    <published>2024-02-11T12:00:00Z</published>
    <title>Scheduling Effects on Knowledge Work</title>
    <summary>A survey study of scheduling changes in knowledge work settings.</summary>
    <author><name>C. First</name></author>
    <author><name>D. Second</name></author>
    <author><name>E. Third</name></author>
    <author><name>F. Fourth</name></author>
    <arxiv:primary_category term="cs.CY" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>not-a-url</id>
    <title>Malformed entry that should be skipped</title>
    <summary>This entry has no usable link.</summary>
  </entry>
</feed>
"""

SEMANTIC_SCHOLAR_RESPONSE = {
    "total": 3,
    "offset": 0,
    "next": 3,
    "data": [
        {
            "paperId": "abc123",
            "title": "The Four-Day Week: Assessing Global Trials",
            "abstract": "We assess outcomes from coordinated four-day week trials across "
            "several countries, focusing on productivity and wellbeing measures.",
            "url": "https://www.semanticscholar.org/paper/abc123",
            "year": 2025,
            "venue": "Journal of Work Studies",
            "citationCount": 143,
            "publicationTypes": ["JournalArticle"],
            "externalIds": {"DOI": "10.1234/jws.2025.1"},
            "openAccessPdf": {"url": "https://example.org/jws-2025-1.pdf", "status": "GOLD"},
        },
        {
            "paperId": "def456",
            "title": "Reduced Hours in Practice",
            "abstract": None,
            "url": None,
            "year": 2023,
            "venue": "",
            "citationCount": 7,
            "publicationTypes": None,
            "externalIds": {"ArXiv": "2303.09999"},
            "openAccessPdf": None,
        },
        {
            "paperId": "ghi789",
            "title": "",
            "abstract": "An untitled record that should be skipped.",
            "url": "https://www.semanticscholar.org/paper/ghi789",
        },
    ],
}

TAVILY_RESPONSE = {
    "query": "four day work week productivity",
    "answer": "Trials generally report stable or improved output.",
    "results": [
        {
            "title": "What 61 companies learned from a four-day week",
            "url": "https://example.org/four-day-week-report",
            "content": "A report covering 61 organisations that trialled a four-day week, "
            "summarising revenue, staff turnover and self-reported productivity.",
            "score": 0.97,
        },
        {
            "title": "Opinion: the four-day week is a fad",
            "url": "https://example.com/opinion-column",
            "content": "A columnist argues the idea will not last.",
            "score": 0.51,
        },
        {"title": "Missing link result", "url": None, "content": "No link here."},
    ],
    "response_time": 1.23,
}

SERPER_RESPONSE = {
    "searchParameters": {"q": "four day work week productivity", "type": "search"},
    "organic": [
        {
            "title": "Four-day week pilot results",
            "link": "https://example.org/pilot-results",
            "snippet": "Headline findings from a national pilot programme.",
            "position": 1,
        },
        {
            "title": "No link",
            "snippet": "This result has no link and should be skipped.",
            "position": 2,
        },
    ],
}
