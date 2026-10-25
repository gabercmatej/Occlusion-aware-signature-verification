# Concept brief - discussion draft, 16 October 2026

## Working title

Occlusion-Aware Offline Handwritten Signature Verification in Financial Documents

## Problem

Handwritten signatures on scanned contracts can overlap stamps, including cases where stamp and pen colors are similar. Cleaning may remove discriminative handwriting or generate plausible-looking content. Genuine signatures also vary, and skilled imitations can resemble the reference. The task is therefore to compare a questioned image with genuine reference signatures while accounting for incomplete visible evidence, not merely to produce a cleaner image.

## Established starting point

Engin et al. (P1, PDF pp. 1 and 5) already study real occluded banking signatures with CycleGAN cleaning and CNN representations, including multiple clean/stamped controls. Lee et al. (P3, PDF pp. 1 and 6) examine physical stamping, hybrid training and missing-versus-extra restoration errors. Liu et al. (P6, PDF p. 6) provide a local-region similarity/fusion approach. These sources motivate a controlled comparison; they do not by themselves prove that a new mask-aware system will be safer.

## Proposed main question

Under the same writer-disjoint split and enrollment conditions, does visibility-aware signature comparison reduce verification errors relative to direct scoring as stamp overlap increases?

## Feasible study

After supervisor approval, establish a reproducible verifier on a licensed public signature benchmark. Generate synthetic stamp overlap with exact source/mask provenance and held-out stamp families. Compare clean controls, direct scoring and mask-aware region weighting. Add a feasible restoration method if scope and compute permit. Study both verification errors and lost signature evidence. Use validation for model/threshold selection and keep the test cohort untouched. Analyze results by overlap severity, forgery type and reference count, with writer-aware uncertainty.

## Data and scope limits

Company samples were promised but receipt, authorization, writer counts, clean references and reliable forgery labels are not yet established. The core study must remain viable on public/synthetic data. A small authorized real pilot can explore transfer, not certify production security. No offline image directly reveals true pen velocity, pressure or stroke order. A restored hidden stroke is not observed evidence.

## Proposed contribution

A reproducible evaluation of how evidence-preserving and restoration-based approaches affect verification under stamp occlusion, including failure analysis and clear limits. Novelty must be established through further reading and supervisor discussion; performance improvements are hypotheses.

## What this draft is for

Use this as a starting paragraph set, then refine it in the student's own words after reading. By October 31, assemble the 2-4 page proposal, annotated bibliography, method diagram and mentor questions. Implementation follows the professor's green light.

References and current reading status: `../literature/literature_matrix.csv`; source-grounded evidence: `../literature/notes/`.
