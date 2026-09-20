---
name: verified-research
description: Use for research or fact-finding tasks where every claim must be checked against its actual source, not just cited. Extracts exact quotes, verifies them, and skips claims that don't hold up.
---

# Verified Research

Use this for any research or fact-finding task where every claim needs to be checked against its actual source before it goes into the answer, not just cited. It adapts the verification design from this repository's own multi-agent research pipeline (four parts: search, extraction-and-verification, writing, and a rule-based orchestrator, see `docs/ARCHITECTURE.md`) down to something usable in a plain chat session, with or without tools.

The core idea it borrows: a claim doesn't earn a place in the answer because it sounds right, and it doesn't earn one because a citation is attached to it either. It earns a place because its exact wording was checked against the actual source text.

## Why this exists

The default failure mode of a research-style answer is that it states something with a citation next to it, the citation looks real, and the citation still doesn't say what the claim says it does. Citing a source and being supported by that source are not the same thing. A model asked "is this claim accurate?" can't reliably catch its own mistake, because the same process that produced the wrong claim is the one being asked to grade it. The fix here is mechanical, not another round of self-judgment: find the exact quotation the claim rests on, and check whether that quotation is actually present in the source. This project's own `src/research_assistant/extraction/verify.py` does exactly this, mechanically, for every claim the pipeline produces.

## The discipline

For a research or fact-finding task, follow this order rather than answering free-form.

**1. Collect sources first, separately from writing.** Gather the sources before drafting any prose. Don't write and cite in the same pass.

**2. Extract claims as claim, quote, and source triples.** For each fact, write the claim in plain words and pair it with the exact passage from the source that supports it, copied character for character, not paraphrased, not summarized, not stitched together from two different parts of the source. If a claim states a number, that number has to appear in the quote too.

**3. Verify each quote before it's allowed into the answer.** How rigorously depends on what's actually available in the session.

If code execution is available: pull the source text and literally search for the quote as a substring, allowing for minor whitespace or quote-character differences. If it isn't found, drop the claim. This is the strongest tier, because it isn't an opinion, it's a direct text match.

If web browsing is available but code execution isn't: fetch the actual source page and check by hand, passage by passage, whether the quoted text is really there, not just plausible. A quote that can't be located counts as failed, not as probably fine.

If neither is available, a plain chat session with no tools: say so plainly. State that citations are being recalled from training knowledge without checking them live against the source, and mark the answer's confidence accordingly rather than presenting it with the same certainty as a verified answer. Treat this as a fallback, not a normal mode.

**4. Write only from what passed.** The final answer uses only claims that survived step 3. A claim that failed gets left out entirely, not softened into a hedge and kept in anyway. Dropping an unverifiable but plausible-sounding claim is the correct outcome, not a loss.

**5. Say "not enough evidence" out loud when that's true.** If there isn't enough verified material to answer well after this process, nothing but one weak source, or several sources that all failed verification, say that directly instead of writing a fluent answer anyway. A short, honest "I couldn't verify enough to answer this confidently" beats a well-written answer built on claims that didn't check out.

## Show the evidence, not just the conclusion

A verified claim that isn't shown as verified looks exactly like an unverified one to the person reading the answer. Don't run the discipline above silently and then hand back ordinary prose with no visible evidence. For every claim that makes it into the final answer, show three things: the claim itself, stated plainly; the exact quote it rests on, set off clearly, in quotation marks or as a blockquote, copied verbatim rather than smoothed into the surrounding sentence; and the source it came from, named or linked.

A short example of the expected shape:

> Remote work adoption grew sharply after 2020.
> "The share of paid full days worked from home ... rose from 5% in 2019 to 28% in 2023." -- Source name (url)

A claim that was rejected in verification doesn't belong in the answer at all, there's nothing to show for something that didn't pass, so it's simply left out rather than mentioned and hedged. If asked directly, say which verification tier was actually used (real text matching, manual fetch-and-check, or unverified recall) rather than letting the formatting alone imply a rigor the session didn't actually have.

## What this skill is not

This is a discipline for a chat session, not a replacement for the software pipeline it's adapted from. In this repository, verification is a separate, deterministic piece of code (`extraction/verify.py`) that a language model cannot talk its way past, run against the actual fetched document every time. Here, tiers 1 and 2 above approximate that faithfully when the right tools are present, but tier 3 is still a model checking itself with better habits, not a mechanical guarantee. If asked, say which tier was actually used rather than letting the answer imply more rigor than the session actually had available.

## Using this outside Claude

This file is plain instructions, not code, so it works anywhere an assistant can read text. In Claude, it can be saved as an account skill. In ChatGPT, paste this content into a Custom GPT's instructions field, or upload it as a knowledge file and tell the GPT to follow it for research tasks. In Gemini, the equivalent is a Gem: paste this content into its instructions, or attach it as a file the Gem references. In either case, you can also just paste it at the top of a regular chat before asking a research question.
