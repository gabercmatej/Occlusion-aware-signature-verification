# Literature notes - work with the assistant

## What is already prepared

Each of the nine local papers has a linked note with correct PDF path, DOI, reading rank and a few page-checked starter observations. Note filenames use the same 01–09 reading sequence as the PDFs; P1–P8 remain paper IDs, not dates. F0 and P5 both fall on October 18; P7 and P8 both fall on October 23. Reading status changes only from explicit personal progress. An AI summary is not proof of understanding or a full paper audit.

## A practical 90-minute session

- 10 minutes: you skim abstract, figures and conclusion; list what is confusing.
- 25 minutes: ask the assistant to explain the method, using the local PDF rather than memory.
- 25 minutes: inspect data, writer split, reference count, forgeries and threshold selection together.
- 20 minutes: save 3-5 useful paraphrases, one small excerpt or figure pointer, one caveat and one experiment idea.
- 10 minutes: explain it back in your own words, update the matrix, then log the saved output in `../../PLAN.md`.

## Copy this request into a research chat

> Open the local PDF linked in this paper's note. Help me understand its task and method first. Ask me to explain the idea back before marking it read. Then update this existing note with 3-5 source-checked paraphrases, a short exact excerpt if useful, PDF page/section/table locations, the evaluation protocol, limitations and one experiment idea. Separate what the authors showed from our interpretation. Do not invent missing details or copy long passages. Update the literature matrix, but leave my reading status unchanged until I confirm my understanding. End with two proposal-ready sentences and one unresolved question.

## Keep snippets useful later

For each snippet keep: claim, paper ID/DOI, PDF page (1-based viewer page), section/figure/table, exact quote versus paraphrase, surrounding context, why it matters, caveat and verification status. A figure pointer plus a description is usually enough; only save a small local figure crop when it helps us reason. Keep subscription figures private and check reuse permissions before publication.

Prefer 3-5 paraphrases and at most one short quotation over a pasted abstract. PDF viewer pages and printed journal pages can differ: label which you mean. Every numerical result also needs dataset/version, writer split, enrollment count, forgery type, metric and threshold protocol. If unknown, record UNKNOWN. Keep interpretations and experiment ideas explicitly labeled as ours.

## Status vocabulary

- Your reading: not started -> discussing -> understood -> synthesized.
- Evidence: pending -> source checked -> rechecked for proposal.
- Numerical result: unselected until its full protocol is recorded.

## Note index, in reading order

- 01: [P01 - real-world-verification](01-P1-real-world-verification.md)
- 02: [F0 - foundation-feature-learning](02-F0-foundation-feature-learning.md)
- 03: [P05 - spatial-transformer](03-P5-spatial-transformer.md)
- 04: [P02 - HTCSigNet](04-P2-HTCSigNet.md)
- 05: [P03 - real-document-stamp-removal](05-P3-real-document-stamp-removal.md)
- 06: [P04 - SERGAN](06-P4-SERGAN.md)
- 07: [P06 - region-based-metric-learning](07-P6-region-based-metric-learning.md)
- 08: [P07 - riemannian-meta-learning](08-P7-riemannian-meta-learning.md)
- 09: [P08 - CCASigNet](09-P8-CCASigNet.md)
