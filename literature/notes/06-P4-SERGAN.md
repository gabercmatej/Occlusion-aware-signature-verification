# 06 / P4 — Generative adversarial networks for rubber stamp extraction and removal (SERGAN)

**Cui, Yang & Guo (2024)** · DOI: 10.1016/j.dsp.2023.104358  
[Local PDF](../papers/06-P4-SERGAN.pdf). Pages below are 1-based PDF viewer pages.  
**Personal progress:** reader confirmed reading and supplied an accurate synthesis on 24 October 2026. Notes combine that understanding with source-checked discussion.

## Main idea and cycles

SERGAN jointly **extracts a stamp** and **produces a cleaned document**, instead of only translating stamped to clean-looking images. Explicit stamp information makes recombination less ambiguous than asking a clean document alone to determine which stamp to regenerate. (pp. 1–5)

```text
Cycle A: stamped image → separate → stamp + cleaned document
                                      → recombine → original-like stamped image
Cycle B: stamp + clean document → combine → stamped image
                                      → separate → recover both starting pieces
```

**Critical caveat:** reconstruction can succeed even if handwriting leaks into the extracted stamp component. Cycle consistency encourages recoverability; it does not prove correct separation or authentic reconstruction of occluded strokes.

## Architecture and learning

- Attention U-Net generators use an encoder/decoder structure and skip connections for spatial detail; shared processing feeds extraction/removal outputs. Learned attention emphasizes features but is not a guarantee of stamp/handwriting separation. Different from P5's STN and P2's ViT. (p. 4)
- Three discriminator roles judge stamped-document, clean-document and isolated-stamp domains. **PatchGAN** examines local patches with a stated 70×70 receptive field, encouraging local realism. Realistic detail is not necessarily faithful identity evidence. (pp. 3–5)
- Adversarial losses encourage plausible outputs; cycle losses encourage reconstruction of images/components. The reported cycle-loss weight λ is 1.0. Equations and notation contain ambiguities; inspect implementation before reproduction. (pp. 3–5)

## Data and protocol

**LRSD:** 7,950 images based on real scanned company-document backgrounds, with digitally generated stamp overlays. This is not P3's physical stamp-and-scan construction. Circular, oval and rectangular stamps include red/blue variants, augmentations and one to three overlays; some deliberately overlap text. Stamp-region crops and mask annotations are part of the setup. Data availability is on request. No company data were accessed in our review. (pp. 5–7, 11)

The paper describes separate training/test images using a two-thirds/one-third LRSD split. It does not provide sufficient detail to assume original-background or stamp-template disjointness. Targets include clean backgrounds and extracted stamps; do not assume identical supervision requirements to P1's unpaired setting. Mask/localization requirements at inference need clarification. (pp. 6–7)

**CLWD:** separate training/testing on a watermark dataset demonstrates architectural applicability to another removal task. It is not zero-shot transfer of an unchanged LRSD-trained model. (pp. 7–8)

## Findings and limits

| LRSD method, Table 2 (p. 8) | RMSE ↓ | SSIM ↑ | PSNR ↑ |
|---|---:|---:|---:|
| CycleGAN | 7.61 | 0.98 | 31.93 |
| WDNet | 3.18 | 0.99 | 41.53 |
| SERGAN | 4.16 | 0.98 | 41.95 |

SERGAN has the highest listed PSNR, but WDNet has better RMSE/SSIM. Qualitative examples support inspecting removal/preservation tradeoffs, not a claim of perfect preservation. Metric aggregation details matter when interpreting PSNR/RMSE rankings.

Ablations support coupled extraction/removal and expanded model variants; architecture and parameter capacity also change, so not every gain is attributable solely to cycle consistency. No comparable latency benchmark establishes a speed advantage over P3. No signature-authenticity evaluation is reported. Binary-image removal is identified as future work. (pp. 9–11)

## P3/P4 comparison and thesis use — our interpretation

| Aspect | P3 | P4 |
|---|---|---|
| Core focus | Physical-data realism, hybrid training, error evaluation | Explicit decomposition/recombination architecture |
| Restoration | Conditional iterative diffusion | GAN-based extraction and cleaning |
| Main data | Physical scans plus synthetic overlays | Synthetic stamp overlays on scanned backgrounds |
| Measures | Global metrics plus Missing/Extra | Global metrics and visual examples |
| Identity preservation established? | No | No |

**Reader's proposal:** connect stamp extraction with verification and measure whether cleaning helps without damaging identity-related features. Candidate diagnostic: inspect whether original signature strokes appear in the extracted stamp output. A predicted stamp map could support masking or visibility weighting; this is a hypothesis, not a chosen solution or demonstrated novelty.

**Controlled experiment:** direct verification versus masking versus SERGAN-style restoration; fixed writers, references and threshold policy; assess residual stamp, signature-stroke damage, FAR/FRR and runtime. A mask cannot recover genuinely hidden evidence, and generative output may invent it. Implementation awaits supervisor approval.

**Reading anchors:** Figure 2 p. 3; Figure 7 p. 6; Table 2 p. 8.  
**Open question:** can separation preserve distinctive handwriting when stamp ink overlaps it or has the same colour?
