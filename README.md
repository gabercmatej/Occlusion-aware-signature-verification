# Occlusion-Aware Signature Verification

Bachelor thesis research on verifying partially stamped offline signatures against genuine references.

**Status — 30 October 2026:** illustrated proposal drafts are available in English and Slovenian. The author's full content review is planned for October 31; the proposal has not been recorded as sent or approved. No trained verifier, completed experiments or performance results are claimed.

## Read the proposal

| Language | Current PDF | Editable source |
|---|---|---|
| English | [Read the English proposal](proposal/proposal_v1.2_EN.pdf) | [Word source](proposal/proposal_v1.2_EN.docx) |
| Slovenščina | [Preberi slovenski predlog](proposal/proposal_v1.2_SL.pdf) | [Izvorni dokument Word](proposal/proposal_v1.2_SL.docx) |

Both versions use the same nine-page structure, illustrations and styling. Labels inside the shared illustrations remain in English. Figures illustrate proposed methods; they are not experimental results. [Proposal files and draft history](proposal/README.md).

## Research direction

The working question is whether visibility-aware comparison can improve verification under stamp occlusion without reconstructing potentially unreliable hidden strokes. Candidate approaches include direct verification, regional reliability weighting and a bounded restoration comparison. Architecture and scope remain provisional until reviewed with the supervisor.

Evaluation should control writer splits, reference counts and stamp conditions, select operating thresholds on validation data, and report false acceptance and rejection alongside EER and runtime. Visually successful restoration alone does not establish preserved identity evidence.

## Reading guide

- [Literature overview and paper links](literature/START_HERE.md)
- [Paper index and access information](literature/papers/README.md)
- [Reading notes](literature/notes/README.md)
- [Glossary](literature/notes/GLOSSARY.md)
- [Literature matrix](literature/literature_matrix.csv)

All nine papers have study notes. Notes distinguish reported findings from proposed extensions and retain protocol caveats. Some study used guided explanations; note existence alone is not proof of full-paper reading. Publisher PDFs are kept locally because redistribution permission is not established for every paper. The index provides DOI links instead.

## Proposal and progress

- [Live plan and progress](PLAN.md)
- [Current proposal PDFs and sources](proposal/README.md)
- [Scope and experiment planning sheet](proposal/PLANNING_SHEET.md)
- [Initial concept brief](proposal/CONCEPT_BRIEF.md)
- [Proposal outline and supervisor questions](proposal/OUTLINE.md)
- [Research direction](RESEARCH.md), [question backlog](QUESTIONS.md), [proposed experiments](EXPERIMENTS.md)

The current illustrated proposal covers the practical problem, research questions, related work, stamp handling, architecture candidates, data fallback, evaluation and staged work plan. The student reviews and sends it personally. This repository provides supporting preparation, not an approved proposal or finished system.

## Folder structure

```text
literature/      Paper index, reading sequence, notes, glossary and evidence matrix
proposal/        Current bilingual proposal, editable sources, figures and draft history
data/           Data guidance; company material remains excluded
notebooks/      Exploratory notebook scaffolds
src/            Future datasets, preprocessing, segmentation and verification code
experiments/    Configuration, figures and result scaffolding
models/         Local model storage guidance
tests/          Future validation scaffolding
```

Folder scaffolding does not imply implemented methods. Public/synthetic data form the proposed feasible core; company-data access and use remain conditional. Confidential data, embeddings and identifying model artifacts must remain restricted. No company data are included in this repository.
