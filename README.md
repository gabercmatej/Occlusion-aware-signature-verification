# Occlusion-Aware Signature Verification

Bachelor thesis research on verifying partially stamped offline signatures against genuine references.

**Status:** literature study and first professor proposal preparation. Supervisor approval is pending; no trained verifier, completed experiments or performance results are claimed. The proposal package is targeted for October 29, 2026, before Friday departure. Implementation follows supervisor approval.

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
- [Initial concept brief](proposal/CONCEPT_BRIEF.md)
- [Proposal outline and supervisor questions](proposal/OUTLINE.md)
- [Research direction](RESEARCH.md), [question backlog](QUESTIONS.md), [proposed experiments](EXPERIMENTS.md)

The intended first package is a 2–4-page proposal plus references, a simple method diagram, feasible data fallback, evaluation plan and focused supervisor questions. The student reviews and sends it personally. This repository is supporting preparation, not an approved proposal or finished system.

## Folder structure

```text
literature/      Paper index, reading sequence, notes, glossary and evidence matrix
proposal/        Concept brief and proposal outline; future drafts
data/           Data guidance; company material remains excluded
notebooks/      Exploratory notebook scaffolds
src/            Future datasets, preprocessing, segmentation and verification code
experiments/    Configuration, figures and result scaffolding
models/         Local model storage guidance
tests/          Future validation scaffolding
```

Folder scaffolding does not imply implemented methods. Public/synthetic data form the proposed feasible core; company-data access and use remain conditional. Confidential data, embeddings and identifying model artifacts must remain restricted. No company data are included in this repository.
