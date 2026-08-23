# Research Direction

## Status

Version 0.1. This is a working research direction, not a final thesis proposal.

## Problem

Offline handwritten signature verification in financial contracts is complicated by rubber stamps that partially overlap signatures. Traditional color- and threshold-based separation can fail when stamp and handwriting have similar visual characteristics, while aggressive removal can destroy forensically useful signature information.

## Security objective

Minimize the false acceptance of forged signatures while automatically processing high-confidence genuine signatures and referring uncertain cases for manual review.

The intended decision space is:

- high-confidence genuine;
- high-confidence mismatch;
- insufficient visible evidence, requiring human review.

## Core problems

1. Stamp/signature separation under same-color overlap.
2. Preservation of forensic handwriting characteristics.
3. Writer-independent signature verification.
4. Natural intra-writer variability.
5. Skilled-forgery detection.
6. Verification with incomplete or occluded evidence.
7. Confidence estimation, calibration, and safe abstention.

## Candidate research question

Can a partially occluded offline signature be verified directly from reliable visible evidence, without reconstructing information that is no longer observable?

## Competing paradigms

### A. Restore

Remove the stamp, reconstruct a clean-looking signature, then run a conventional verifier.

Main risk: the restoration model may hallucinate or erase discriminative strokes and make the verifier confidently wrong.

### B. Mask

Estimate stamp, signature, and ambiguous regions, then verify only features supported by reliable visible pixels.

Main advantage: hidden strokes are not invented. Main challenge: the verifier must remain useful as visible evidence decreases.

### C. Direct occlusion-aware verification

Give the questioned stamped signature, clean references, and optionally an occlusion mask to a verifier that learns which evidence remains trustworthy.

Main advantage: the system can optimize the actual verification objective instead of image appearance. Main challenge: careful controls are needed to show that the model is not exploiting stamp or document artifacts.

## Primary comparison

Compare restoration, mask-based verification, and direct occlusion-aware verification under controlled occlusion severity. Measure verification performance, calibration, abstention behavior, residual stamp content, and loss of genuine signature evidence.

## Initial hypotheses

- Restoration will improve visual quality but may not reliably improve verification.
- Mask-based or direct verification will be safer at moderate occlusion because they do not invent hidden strokes.
- All methods should abstain more often as reliable visible signature area decreases.
- Models trained only on digitally overlaid stamps will generalize poorly to physically stamped documents.

## Evaluation requirements

- Person-disjoint train, validation, and test splits.
- Genuine, random-forgery, and skilled-forgery evaluation where datasets permit.
- FAR, FRR, EER, ROC/AUC, and TAR at operationally low FAR.
- Risk-coverage curves for selective prediction.
- Performance versus occlusion percentage and visible-signature percentage.
- Separate measurement of residual stamp pixels and accidentally removed signature pixels.
- Cross-dataset or cross-script evaluation where practical.

## Scope boundary

The first phase does not build an end-user app, automate decisions on real contracts, or train on confidential company documents. It establishes baselines, data governance, and an evidence-backed experimental design using public and synthetic data.
