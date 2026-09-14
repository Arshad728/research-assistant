# Evaluation results

Produced by `research-assistant evaluate`. Every number below is mechanical. The two measures that matter most, citation accuracy and finding relevance, are judgements and appear at the end as a worksheet rather than as a score.

> **Run against the OFFLINE DEMO pipeline, whose corpus is three documents about one topic. Queries on any other subject correctly return nothing, which shows up here as "did not behave as expected" only because the test set assumes a real corpus. Treat this file as proof that the harness works, not as a measurement of the system.**

## Headline

- Queries run: **6**
- Behaved as expected: **4 of 6** (67%)
- Reports produced: **2**
- Reports passing validation: **6**
- Claims proposed by extraction: **12**; verified: **12** (100%)
- Mean time per query: **0.0s**

"Behaved as expected" is not the same as "produced a report." Two of the test queries are deliberately about things that are barely studied or do not exist. For those, producing no report is the correct outcome, and producing a confident one would be the worst possible result.

## Per query

| Query | Difficulty | As expected | Findings | Verified / proposed | Cited paras | Validation | Time | Stopped because |
|---|---|---|---|---|---|---|---|---|
| four_day_week | straightforward | yes | 6 | 6/6 | 7/8 | pass | 0.0s | sufficient_evidence |
| rag_hallucination | straightforward | **NO** | 0 | - | 0/1 | pass | 0.0s | no_evidence |
| remote_work_wages | moderate | yes | 6 | 6/6 | 7/8 | pass | 0.0s | sufficient_evidence |
| ai_regulation_transparency | moderate | **NO** | 0 | - | 0/1 | pass | 0.0s | no_evidence |
| microdosing_productivity | should_find_little | yes | 0 | - | 0/1 | pass | 0.0s | no_evidence |
| nonexistent_framework | should_find_little | yes | 0 | - | 0/1 | pass | 0.0s | no_evidence |

### What each query is testing

- **four_day_week** — The ordinary case: a well-studied question with both academic and general-web coverage. If this fails, something is broken rather than hard.
- **rag_hallucination** — Heavy arXiv coverage and almost no general-web coverage. Exercises the academic retrieval path in isolation.
- **remote_work_wages** — A question where the honest answer is mixed. Tests whether the Writer Agent reports disagreement instead of averaging it away.
- **ai_regulation_transparency** — Moves quickly and lives in policy documents rather than papers. Tests whether web retrieval carries a query that academic search cannot.
- **microdosing_productivity** — Deliberately over-specific: plausible-sounding, almost certainly unstudied. A system that produces a confident report here is fabricating. The correct outcome is few or no verified findings and a report that says so.
- **nonexistent_framework** — Names something that does not exist. Tests the strongest failure mode in the system: whether it will invent sources for an invented subject.

## Spot-check worksheet

Citation accuracy cannot be measured by the system that produced the citations. Open each source below, find the quoted text, and decide whether the claim is a fair reading of it. Mark each one and count the results: that count is the citation accuracy figure.

Judge three things separately: does the quotation appear in the source; does it say what the claim says it says; and does the claim omit context that changes its meaning. The third is the one the system cannot check for itself.

### 1. `four_day_week`

**Claim.** The source states that the survey covered the same programme as the trial and relies entirely on self-reported measures rather than.

**Quoted as evidence.** "The survey covered the same programme as the trial and relies entirely on self-reported measures rather than on observed output."

**Source.** <https://example.org/survey>

- [ ] The quotation appears in the source
- [ ] The quotation supports the claim
- [ ] The claim omits no context that changes its meaning

### 2. `four_day_week`

**Claim.** The source states that several participating organisations had already begun other efficiency programmes before the trial began.

**Quoted as evidence.** "Several participating organisations had already begun other efficiency programmes before the trial began, which complicates attribution further."

**Source.** <https://example.org/critique>

- [ ] The quotation appears in the source
- [ ] The quotation supports the claim
- [ ] The claim omits no context that changes its meaning

### 3. `four_day_week`

**Claim.** The source states that across the trial period, hourly productivity increased by 8%, while total weekly output remained.

**Quoted as evidence.** "Across the trial period, hourly productivity increased by 8%, while total weekly output remained statistically unchanged across the participating organisations."

**Source.** <https://example.org/trial>

- [ ] The quotation appears in the source
- [ ] The quotation supports the claim
- [ ] The claim omits no context that changes its meaning

### 4. `remote_work_wages`

**Claim.** The source states that respondents reported markedly higher satisfaction scores under the shorter working schedule than under their.

**Quoted as evidence.** "Respondents reported markedly higher satisfaction scores under the shorter working schedule than under their previous arrangements, with most citing reduced fatigue."

**Source.** <https://example.org/survey>

- [ ] The quotation appears in the source
- [ ] The quotation supports the claim
- [ ] The claim omits no context that changes its meaning

### 5. `remote_work_wages`

**Claim.** The source states that the absence of a control group means the observed changes cannot be attributed to the schedule change with.

**Quoted as evidence.** "The absence of a control group means the observed changes cannot be attributed to the schedule change with confidence, a limitation common to workplace trials of this kind."

**Source.** <https://example.org/critique>

- [ ] The quotation appears in the source
- [ ] The quotation supports the claim
- [ ] The claim omits no context that changes its meaning

### 6. `remote_work_wages`

**Claim.** The source states that the survey covered the same programme as the trial and relies entirely on self-reported measures rather than.

**Quoted as evidence.** "The survey covered the same programme as the trial and relies entirely on self-reported measures rather than on observed output."

**Source.** <https://example.org/survey>

- [ ] The quotation appears in the source
- [ ] The quotation supports the claim
- [ ] The claim omits no context that changes its meaning

## Comparison against doing it by hand

Chapter 7.1 asks for the system's time to be compared against a careful person answering the same question, holding depth roughly constant. That number is not here because it cannot be automated: it requires someone to actually do the research. The per-query times above are one half of that comparison; the other half has to be measured by hand, once, and written down.
