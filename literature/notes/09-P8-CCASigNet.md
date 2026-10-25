# 09 / P8 — Learning discriminative representations by a Canonical Correlation Analysis-based Siamese Network for offline signature verification

**Zheng et al. (2025)** · DOI: 10.1016/j.engappai.2024.109640  
[Local PDF](../papers/09-P8-CCASigNet.pdf). Pages below are 1-based PDF viewer pages.  
**Reader contribution:** supplied a synthesis and occlusion-pair training proposal on 25 October 2026. Notes consolidate that synthesis and checked source passages; full-paper reading and plan completion are not inferred.

## Main idea and pipeline

Add correlation-based pair objectives to F0-style writer/forgery classification learning. The shared SigNet backbone learns features, then a separate RBF-SVM is fitted for each new writer. Thus, **feature learning is transferable; the complete reported decision system is writer dependent**. CCA is a training objective, not the final customer-verification rule. (pp. 4–6)

```text
Training tuple: G1, G2, S1, S2, R
→ same SigNet weights → features
→ CCA pair objectives + writer and forgery classification objectives

Enrollment: genuine customer references + other writers' genuine negatives
→ shared learned features → customer's RBF-SVM
Verification: query → shared features → customer's SVM score → threshold
```

## Pair objectives and loss

Training uses two distinct genuine signatures and two distinct skilled forgeries targeting the same writer, plus a random forgery (another writer's genuine signature). Five input instances share one feature extractor; “Siamese” here is not limited to exactly two training inputs. (p. 4)

| Pair | Intended correlation behaviour |
|---|---|
| G–G | Increase correlation between genuine examples of the same writer |
| S–S | Increase correlation between selected skilled forgeries targeting that writer |
| G–S | Reduce correlation between genuine examples and targeted imitations |
| G–R | Reduce correlation with another writer's genuine signature |

**CCA:** Canonical Correlation Analysis finds linear projections of two groups of variables with high correlation. The paper differentiates correlation-based objectives through learned features; this is not simply pixel similarity or a guarantee that matching vectors are identical. S–S encouragement does not mean two forgeries are accepted as genuine at inference. (pp. 4–5)

Writer classification and genuine/forged classification remain complementary objectives. The initial combined loss explicitly restricts writer classification to genuine examples. Their rewritten objective uses α to balance grouped identity/genuine-related terms against skilled-forgery-related terms; α is **not merely a standalone CCA-loss weight**, nor P5's focal-loss α. Among tested settings, α=0.05 performs best on the reported GPDS comparison. More emphasis on forgery objectives is not automatically better. (pp. 6–7)

## Data and experimental conditions

SigNet uses five convolution layers and 2,048-dimensional feature layers. Preprocessing uses Otsu foreground/background separation, inversion, resizing to 170×242 and random 150×220 training crops. None of this establishes separation of stamp ink from handwriting. (pp. 4, 6–7)

Feature learning uses GPDSsynthetic writers 301–1000; evaluation uses 1–300. Validation is described as writers 1001–1050, beyond the earlier “first 1000” description, so preserve this discrepancy for reproduction. Target-dataset counts, especially UTSig, also require version reconciliation rather than assuming agreement with other papers. Shared features transfer to other datasets; separately labelled fine-tuning variants use five target writers. (pp. 6–8)

Personal SVMs use genuine references as positives and other writers' genuine samples as negatives, with class weighting. Skilled forgeries are needed for shared development training, not each customer's SVM enrollment. Experiments repeat SVM fitting with different selections ten times. EER is computed using optimal **per-writer threshold sweeps**; it is not evidence of enrollment-only threshold calibration. (pp. 7–8)

## Selected results — un-fine-tuned feature transfer

| Dataset | EER | Genuine references used to fit personal SVM |
|---|---:|---:|
| GPDSsynthetic | 2.54% | 12 |
| CEDAR | 4.15% | 12 |
| UTSig | 11.26% | 12 |
| Bengali | 4.94% | 10 |
| Hindi | 3.05% | 10 |

**Sources:** Tables 7/9, p. 8; Tables 10/11, p. 10. These are skilled-forgery user-threshold EERs, not general accuracy or low-FAR guarantees. The GPDS baseline changes from 2.68% to 2.54%: 0.14 percentage points, with reported variation of similar scale. At 10 references it ties the baseline at 2.80%. On UTSig with 12 references, 11.26% is slightly worse than the listed baseline 11.23%. Do not claim consistent superiority or statistical significance.

**Ablation:** Table 12 (p. 10) reports a reduced loss combination reaching 2.50% versus 2.54% for all terms; the authors argue the full loss transfers better in other settings. More loss terms need not improve every result. Table 8's five-reference entry also conflicts with Table 7/text. Source-check exact conditions before proposal reuse.

## Thesis relevance — reader's proposed extension

**Question:** can clean–occluded pair training suppress stamp interference without suppressing differences that expose skilled forgery?

- Pair clean genuine examples with stamped genuine examples to encourage appropriate consistency.
- Include genuine/skilled-forgery pairs with stamp conditions varied independently of labels, preventing stamp colour, position or template from becoming a shortcut.
- Preserve identity discrimination while encouraging robustness: indiscriminate correlation can collapse useful distinctions or align stamp features instead of handwriting.
- Compare classification-only features with ordinary stamp augmentation, then CCA-augmented training under the same data, backbone and compute budget. Test on held-out writers and stamp templates with fixed references and validation-selected thresholds; report FAR/FRR, EER and runtime.
- Combining CCA-trained features with P6-style regional reliability is a separate candidate experiment, not demonstrated by either paper. Implementation awaits supervisor approval.

**Scope:** retain F0-style features as a baseline, regional visibility weighting as a directly relevant candidate, and CCA as an optional training extension. This is a provisional prioritization, not a final design or novelty claim. No stamp-occlusion or human-superiority evaluation is reported in P8.

**Reading anchors:** training architecture Figure 2 p. 5; objective and personal SVM p. 6; baseline Table 7 p. 8; transfer/ablation p. 10.  
**Open question:** does CCA add robustness beyond matched stamp augmentation while retaining skilled-forgery rejection?
