# 04 / P2 — HTCSigNet: A Hybrid Transformer and Convolution Signature Network for offline signature verification

**Zheng et al. (2025)** · DOI: 10.1016/j.patcog.2024.111146  
**Paper:** [local PDF](../papers/04-P2-HTCSigNet.pdf). All pages below refer to 1-based PDF viewer pages.  
**Personal progress, 19 October 2026:** reader studied the guided explanation and explained the main architecture and training objectives back. Correctly identified personal-SVM verification as writer dependent; explanation clarified in this discussion. This does not claim full-paper reading or completed abstract/figure inspection.

## Main idea and architecture sketch

Combine global relationships and local stroke information in one feature extractor. CNNs emphasize local patterns; vision-transformer attention connects regions across the signature. These are useful architectural tendencies, not exclusive abilities. (pp. 1–4)

```text
ONE signature → preprocessing / initial convolution processing
                       ├→ ViT branch → global features ──────┐
                       └→ CNN branch → local features ───────┤
                              SPD-Conv → SCConv → convolutions
                                                           ↓
                                              concatenate feature vectors
                                                           ↓
                                              combined representation
                                                           ↓
                                   personal RBF-SVM OR Euclidean comparison
```

**Reading anchors:** Figure 2, **p. 3** (architecture); Section 3.4–3.5, **p. 5** (training and decisions); Table 2, **p. 7** (same-setting baseline comparison).

## Components worth remembering

- **Vision transformer (ViT):** represents image patches with positional information and uses self-attention to exchange information between regions. Captures broader layout and relationships. Initial convolutions provide local processing; a residual path helps retain input information. This is not P5's spatial transformer, which learns geometric resampling/alignment. (p. 4)
- **SPD-Conv:** space-to-depth rearranges nearby spatial values into channels, shrinking spatial dimensions without discarding values in the rearrangement itself. A following stride-one convolution learns combinations and may compress information. Intended to retain fine stroke detail better than immediate lossy downsampling; it does not specifically distinguish signature ink from stamp ink. (p. 4)
- **SCConv:** spatial and channel reconstruction reorganizes feature maps to reduce redundancy and encourage useful features. Spatial reconstruction separates/recombines information; channel reconstruction splits, transforms and fuses it. This is learned feature processing, not stamp segmentation. (p. 4)
- **Final convolutions:** produce a 512-dimensional local vector. Concatenation joins local and global vectors; it does not automatically judge which branch is trustworthy under occlusion. (p. 5)

## Feature learning versus final decisions

**Training:** individual genuine and skilled-forged development images teach writer classification and genuine/forged classification. Cross-entropy losses are combined with λ=0.95 weighting the forgery loss. This resembles F0's multitask idea, but the displayed objective does not explicitly show F0's genuine-only mask on writer loss. It is not focal loss or direct Siamese pair training. (pp. 5–6)

**WD option:** shared HTCSigNet features feed a separate RBF-SVM for every enrolled writer. Their genuine references are positives; other writers' genuine signatures provide negatives. SVM means **support vector machine**. The system is writer dependent because its learned decision boundary changes with the claimed customer, even though the feature extractor is shared. (p. 5)

**WI option:** compare reference/query feature vectors by Euclidean distance and threshold the result. Smaller distance means a closer match. This avoids fitting a separate personal SVM. Shared features alone do not determine whether the final system is WD or WI. Multiple-reference score aggregation and a validation-only threshold policy need clarification before implementation. (pp. 5, 8–9)

## Data, findings and evidence limits

- Main training uses **GPDSsynthetic**, with development writers 301–1000 and separate verification writers 1–300. The transformer starts with ImageNet21k-pretrained weights. Background removal, inversion, resizing and random crops are not demonstrated stamp separation. Synthetic signatures are different from adding synthetic stamps to real signatures. (p. 6)
- **Same-setting WD comparison, Table 2 (p. 7):** with 12 genuine references per writer on GPDSsynthetic, EER is 3.33% for SigNet-F, 1.41% for ViT and 1.16% for HTCSigNet. The latter improves on ViT by 0.25 percentage points; not every listed metric improves. EER is not universal accuracy, and reported FAR/FRR need not use the EER threshold.
- **Transfer, Tables 4–6 (p. 8):** GPDS-trained features feed personal classifiers on other datasets; separately labelled versions fine-tune using five target-dataset writers. CEDAR with 12 references reports 3.99% EER before fine-tuning and 2.36% after. These are different protocols, not interchangeable results.
- **WI is a different experiment:** feature extractors are trained within each dataset using held-out writer partitions. Reference counts 1–12 are tried and best results presented. CEDAR's reported 0% EER uses five test writers. Threshold sweeping and best-configuration selection are not evidence of a locked deployment policy. Some Table 7 accuracy/FAR/FRR values do not reconcile under standard same-decision binary metrics. (pp. 8–9)
- **Ablations and visualizations:** component removal supports contributions under studied WD settings; SCConv does not improve every dataset. Attention maps illustrate responses but do not prove reliance on identity evidence. Cross-dataset gains do not establish absence of future overfitting. (pp. 9–10, Table 8; Figures 5–8)

## Thesis relevance — our hypotheses, not findings from this paper

Global/local fusion might preserve useful evidence when stamps or restoration damage one type of cue. It could also preserve stamp detail, attend to stamp layout or amplify restoration artifacts. SPD-Conv and SCConv are candidate components, not established necessities for our task. No controlled stamp-occlusion or human-superiority claim is established here.

**Reader's hybrid-Siamese idea:** use the same hybrid extractor on the reference and query, then train a pair comparator. This is structurally compatible: global/local branches operate *inside* each shared extractor; Siamese sharing operates *between* the two images. It is a candidate experiment, not a demonstrated contribution or novelty claim. Distinguish true pair-based training from simply running a shared extractor twice and applying a fixed distance.

**Controlled experiment:** compare CNN-only, ViT-only and hybrid features using fixed writers, references, data conditions and validation-selected thresholds. Test clean and stamped queries; add masking/restoration or pairwise training as separate factors. Measure error tradeoffs and runtime rather than assuming extra modules improve both. Implementation waits for supervisor approval.

**Open question:** can local/global features remain complementary when stamp interference or restoration artifacts affect both?
