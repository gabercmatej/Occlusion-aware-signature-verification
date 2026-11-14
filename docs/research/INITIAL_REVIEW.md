# Initial Strategy Review (M0)

**Date:** 11 November 2026. **Scope:** one bounded session: six web searches, five primary pages, local notes P1, P6 and the literature matrix. Nothing was installed, downloaded, trained or run. All web pages below were consulted on 11 November 2026. This is not an exhaustive survey.

## Question

Which pipeline gives the partner a useful local prototype soonest, given mandatory stamp removal, reliability-weighted comparison, an Apple M5 with 16 GB, and no confirmed data or labels?

## Sources checked

| Source | What it establishes | Limits |
|---|---|---|
| [luizgh/sigver](https://github.com/luizgh/sigver) (Hafemann et al., F0) | BSD 3-clause code; SigNet and SigNet-F weights on Google Drive; 150×220 input after background removal, centering and resize | README: "the trained models used the GPDS dataset for training, which is restricted for non-commercial use" |
| [tallesbrito/contrastive_sigver](https://github.com/tallesbrito/contrastive_sigver) | BSD 3-clause code; three weight files stored in the repository, including "SigNet (Synthetic)" | Same GPDS non-commercial note |
| [GPDSsynthetic license agreement](https://www.gpds.ulpgc.es/downloadnew/agreements/LICENSEAGREEMENTSyntheticOffSignature.pdf) | Non-commercial research only; clause 3.3 excludes commercial applications "in any form (e.g., original files, encrypted files, files containing extracted features, etc)"; no redistribution | Silent on trained weights; who may sign is not stated |
| [kdMoura/proto_hsv](https://github.com/kdmoura/proto_hsv) (2026, [arXiv 2606.18528](https://arxiv.org/abs/2606.18528)) | A current writer-independent method builds a dissimilarity classifier (SVM/SGD) on frozen SigNet-S features, evaluated on GPDS-S, CEDAR, MCYT | README only; no license or weights seen; paper not read; no stamps |
| [GuohaoCui/DiffusionModel](https://github.com/GuohaoCui/DiffusionModel) (optional matrix entry) | Diffusion stamp-removal code, weights and stamp-area dataset on Google Drive | No license; CUDA environment; trained on document stamps, not signatures; needs paired data to retrain |
| Local notes P1, P6 | P1: cleaning reduced stamped-query EER from 33% to 23% for one model, still short of unstamped 18%. P6: regional gains come from a network trained on regions; fusion is equal averaging | Neither tests reliability weighting or skilled forgeries under stamps |

## Paths compared

| | A — classical cleaner + frozen SigNet + weighted windows | B — same, generic encoder | C — learned cleaner + trained regional network |
|---|---|---|---|
| Signature-domain fit | Signature-trained features | Not signature-trained | Best in principle, if trained well |
| Hallucination / identity damage | Cleaner cannot draw strokes; deletions recorded | Same | Generative cleaner can invent or erase strokes |
| Permissions | Weights non-commercial; customer use unclear | Likely permissive; verify | Stamp weights unlicensed; training data need agreements |
| Local compute | CPU-sized | CPU/MPS-sized | Long training; CUDA-oriented code |
| Time to first prototype | Shortest | Short | Longest |

## Decisions

- **Keep:** mandatory cleaning with preserved original, stamp, intervention and change maps; regional comparison with reliability weights and equal-weight fusion as control; writer/source splits before augmentation; review instead of verdicts until calibrated.
- **Change:** classical color separation without inpainting instead of an early GAN/diffusion cleaner; frozen SigNet-family weights instead of training ResNet-18; no training for the MVP, later a small dissimilarity classifier on frozen features; cleaning/provenance command first (M1a), encoder second (M1b).
- **Defer:** learned stamp segmentation or restoration; encoder fine-tuning; hybrid CNN–transformer; Riemannian and CCA methods; automatic signature detection in full documents; human-comparison study.
- **Proposed to owner, not adopted:** score the uncleaned query as an evaluation-only control, so the effect of mandatory cleaning can be measured.

## Uncertainties

- Whether GPDS-trained weights may be used in the partner's setting: the license does not address trained models. Treat SigNet output as diagnostic until cleared.
- Whether a whole-signature encoder gives meaningful scores on windows. Untested; the plan gates on a signal check.
- Whether color separation works on the customer's stamps. Blue pen under a blue stamp, or black stamps, will defeat it.
- CEDAR and BHSig260 terms, the DINOv2 and torchvision weight licenses, and PyTorch loading of the 2019-era weights were not checked this session.
- The search for newer work on stamped-signature verification and for generic-feature baselines was inconclusive, not negative evidence.
- The Lee et al. (P3) code and dataset were not located; the dataset is described as available on request.
