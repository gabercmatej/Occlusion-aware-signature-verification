# 05 / P3 — Stamp removal from real documents: A diffusion-based approach with EA-RMSE evaluation

**Lee et al. (2026)** · DOI: 10.1016/j.array.2026.100855  
[Local PDF](../papers/05-P3-real-document-stamp-removal.pdf). Pages below are 1-based PDF viewer pages.  
**Personal progress:** reader explicitly confirmed reading and supplied an accurate synthesis on 24 October 2026. Notes combine that understanding with source-checked discussion.

## Main contribution

Improve **data realism and restoration evaluation**, using an existing conditional diffusion architecture rather than claiming a wholly new denoiser. Digitally overlaid stamps miss physical ink, pressure, degradation, paper and scan effects; synthetic-only training can fail on physical documents. (pp. 1–4)

**Pipeline:** stamped input conditions a U-Net diffusion denoiser → repeated denoising → restored document. Training adds artificial random noise to clean targets; this diffusion noise is different from stamp interference. A standalone U-Net predicts restoration in one pass; the diffusion model repeatedly uses a U-Net denoiser. (pp. 7–9)

## Data and training

- **12,000 physical stamped scans:** 120 combinations of shape, colour, resolution, placement and damage, with one stamp per image. Backgrounds mainly use public-domain Korean novel pages, not representative bank transactions. Clean PDF renderings provide paired targets; registration aligns scans and targets. (pp. 4, 17)
- **Document split:** 9,600 training / 1,200 validation / 1,200 testing images. Equal training budgets compare real-only, synthetic-only and hybrid data; default hybrid uses 4,800 of each. (pp. 5–7)
- **Fine-tuning:** adapting synthetic-pretrained models with real-only data can hurt synthetic performance, consistent with catastrophic forgetting. Hybrid exposure better retains both domains in their experiments. (pp. 8–9)
- **Auxiliary virtual documents:** generated layouts/text are printed, physically stamped and scanned. They are distinct from fully digital stamp overlays and excluded from the primary hybrid experiment. Broader layouts help some results, but logo erasure persists. (pp. 5, 13–15)

## EA-RMSE: measure two failures

Compare the predicted image with an aligned clean ground truth; threshold meaningful pixel differences and compute area-weighted error magnitudes separately. In the dark-ink/light-background interpretation:

| Component | Interpretation | Desired direction |
|---|---|---|
| **Extra** | Unexpected darker content, interpreted as residual stamp interference | Lower |
| **Missing** | Unexpected lighter content, interpreted as erased or altered document content | Lower |

The paper uses CIELab conversion and a fixed threshold τ=5; sensitivity is examined separately. These are difference-based masks, not semantic proof of stamp ownership or lost writer identity. Colour changes, alignment errors and generated content can affect scores. Both extent and magnitude matter. Global PSNR/SSIM/RMSE may hide small but consequential failures. (pp. 5–6; Appendix B)

## Selected evidence and limitations

- **Real-world test, from-scratch training, Table 2 (p. 7):** synthetic-only Missing/Extra = 5.25/3.06; real-only = 2.19/1.77; hybrid = 1.90/2.06. Hybrid reduces Missing but does not beat real-only Extra. Report the tradeoff, not universal dominance.
- **Latency, Table 5 (p. 10):** real-world setting reports 20.80 s/image for 25-step diffusion, 8.42 s for 10 steps, versus 0.02 s for standalone U-Net. Hardware, patching and image size matter; these are not predictions for signature crops.
- **Human evaluation (p. 12):** 10 participants rate 100 outputs. EA-RMSE correlates better with their ratings than conventional metrics in this study, but this is perceived restoration quality, not authenticity or forensic identity preservation. Validation is limited to particular outputs and conditions.
- Main data diversity is limited; unfamiliar graphics can be erased and overlapping stamps remain difficult. A plausible reconstructed stroke need not be the stroke originally written. No signature verifier or human-verification superiority is established. (pp. 13–16)

## Thesis use — our interpretation

The reader's priority is **preserving genuine signature evidence rather than visual cleanliness**. Real plus synthetic occlusion could improve robustness, but effectiveness on signatures requires testing. EA-RMSE can complement, not replace, verification FAR/FRR and evidence-preservation checks. Its threshold and image-wide normalization may miss subtle, identity-critical strokes.

**Proposed experiment:** same held-out writers, references and validation-selected threshold; compare clean queries, stamped queries without treatment, masking and restoration. With paired clean targets, report missing/extra errors, signature-region damage, verification outcomes and runtime. Do not infer success from PSNR alone. Implementation awaits supervisor approval.

**Reading anchors:** Figure 1 p. 3; Figure 4 and metric p. 6; Tables 2–5 pp. 7–10.  
**Open question:** which restoration changes affect identity decisions despite small perceptual error?
