# Occlusion-Aware Signature Verification

Research workspace for a bachelor thesis on offline handwritten signature verification when rubber stamps partially occlude signatures on financial and contractual documents.

This is a research project, not an application. The initial goal is to understand the literature, define defensible experiments, and establish reliable evaluation before building a production interface.

## Start here

1. Read [RESEARCH.md](RESEARCH.md) for the current research direction.
2. Read [literature/START_HERE.md](literature/START_HERE.md) and the downloaded starter papers.
3. Record every paper in [literature/literature_matrix.csv](literature/literature_matrix.csv). A formatted Excel view is available at `literature/literature_matrix.xlsx`.
4. Put open questions in [QUESTIONS.md](QUESTIONS.md), without forcing early answers.
5. Register experiments before running them in [EXPERIMENTS.md](EXPERIMENTS.md).

## Research principles

- Keep train, validation, and test writers disjoint for writer-independent evaluation.
- Report security-relevant metrics such as FAR, FRR, EER, ROC/AUC, and TAR at a fixed low FAR.
- Treat stamp removal as a hypothesis, not as the assumed solution.
- Compare restoration, mask-based verification, and direct occlusion-aware verification.
- Measure both residual stamp content and lost signature evidence.
- Prefer abstention and human review when the visible evidence is insufficient.
- Never invent or reconstruct hidden strokes without explicitly measuring the security risk this creates.

## Project map

- `literature/`: papers, structured reading notes, and the literature matrix.
- `data/`: public, confidential company, and synthetic datasets. See `data/README.md` before adding anything.
- `notebooks/`: exploration and baseline experiments only.
- `src/`: reusable research code, separated by responsibility.
- `experiments/`: configurations, results, and figures.
- `models/`: local model checkpoints.
- `tests/`: tests for reusable preprocessing, data, and evaluation code.

## Data safety

Company documents, contracts, signatures, and derived identifying material must not be committed to a public repository. The entire `data/company/` directory is ignored by Git except for its safety note. Keep a written data-governance record covering authorization, anonymization, retention, and deletion before using company data.

## First-week outcome

At the end of the first week, you should be able to explain the verification task, writer-dependent versus writer-independent evaluation, skilled forgeries, metric learning, stamp separation versus restoration, forensic evidence preservation, and FAR/FRR/EER without relying on an implementation.
