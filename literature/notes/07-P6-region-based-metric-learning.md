# 07 / P6 — Offline signature verification using a region based deep metric learning network

**Liu, Huang, Yin & Chen (2021)** · DOI: 10.1016/j.patcog.2021.108009  
[Local PDF](../papers/07-P6-region-based-metric-learning.pdf). Page numbers below are 1-based PDF viewer pages.  
**Reader contribution:** supplied a synthesis on 25 October 2026, including regional comparison, shared MSDN, threshold distinctions and a visibility-weighting hypothesis. This note consolidates that synthesis and source-checked details; it does not infer a full-paper reading claim or change plan completion.

## Main idea and pipeline

Compare corresponding local regions, then combine their similarity scores. Fine differences in loops, connections and endings may distinguish genuine signatures from skilled imitations better than whole-image processing. Feature extraction and similarity prediction are trained jointly. (pp. 3–6)

```text
Reference + query → preprocessing → matching overlapping regions
→ shared Mutual Signature DenseNet (MSDN), with pair interactions
→ absolute feature-difference vector → learned sigmoid similarity
→ average region scores → average over references where applicable
→ global or writer-specific threshold → decision
```

## Method details worth retaining

- **Preprocessing:** grayscale, tilt correction, Otsu background separation, foreground grayscale normalization, moment-based size/position normalization. Normalized canvas is 512×224 pixels. These operations are not stamp segmentation and may change useful image evidence. (pp. 3–4)
- **Nine regions:** 224×224 windows slide left to right with a 36-pixel step and 188-pixel overlap, indexed 0–8. One stamp can therefore affect several region scores; the windows are not independent evidence. (pp. 4, 7)
- **MSDN:** two shared-parameter DenseNet branches include Squeeze-and-Excitation (SE) blocks using differences between the paired inputs to weight channels. This mutual interaction means it is not simply two independent, fixed embeddings; caching features without accounting for pair dependence requires investigation. One shared network processes all region positions. (pp. 4–6, Figure 6)
- **Multi-level features and SPP:** combine outputs at different depths; spatial pyramid pooling uses 1×1, 2×2 and 4×4 bins to create a fixed-size 10,752-value representation for whole-image or regional input. Absolute feature differences feed a learned fully connected sigmoid comparator; binary cross-entropy trains both extractor and comparator. Sigmoid outputs are model scores, not established calibrated probabilities. (pp. 5–6)
- **Fusion:** equal averaging of corresponding regional scores. The selected three-window set {1,4,7} gives a useful speed/accuracy compromise, not the lowest EER on every dataset. All nine windows perform better on GPDS and ChnSig; {1,4,7} is best on CEDAR among the listed regional combinations. (pp. 7–9)

## Data, results and protocol caveats

CEDAR uses 50 training / 5 test writers; GPDS Synthetic uses 2,000 / 2,000; ChnSig uses 1,000 / 243. CEDAR and ChnSig models are fine-tuned from GPDS pretraining, not zero-shot transfers. Pair counts reuse underlying images and should not be interpreted as independent signature counts. Skilled-forgery evaluation is the main setting. (p. 7)

| Dataset | Whole-image WI EER | {1,4,7} WI EER | WD EER, 12 references |
|---|---:|---:|---:|
| CEDAR | 8.26% | 6.74% | 1.67% |
| GPDS Synthetic | 8.99% | 8.24% | 1.65% |

**Sources:** Table 5, p. 8; Table 7, p. 9; WD explanation p. 10. WI uses a shared threshold; WD tunes a threshold for each writer, while retaining the shared neural model—no personal SVM is trained. WD also averages query scores across genuine references and repeats random reference selection ten times. The comparison changes both threshold policy and reference count; it does not isolate one effect.

Writer-specific thresholds are adjusted to obtain EER, not demonstrated as an enrollment-only threshold-selection method. These values are evaluation operating points, not a proven deployment guarantee. CEDAR's small test-writer set limits confidence. Table 5 reports 88 ms/pair for {1,4,7} versus 86 ms whole-image and 142 ms all-nine; another section reports 258 ms overall, so timing scope needs clarification before reuse. (pp. 7–10)

**Complexity caveat:** Table 6 (p. 9) gives CEDAR EER 4.55% for plain DenseNet versus 6.74% for DenseNet+SE+multi-level features. The fuller network helps GPDS/ChnSig in that table but not CEDAR. More components are not automatically better.

## Thesis direction — reader's proposal, not a result of P6

**Question:** can visible regional evidence verify partially stamped signatures more reliably than reconstructing hidden strokes?

1. Extract corresponding regions from reference and query.
2. Estimate stamp contamination or usable signature evidence per region.
3. Predict regional similarity with genuine references.
4. Weight scores by estimated reliability rather than equal averaging.
5. Evaluate against whole-image verification, unweighted region fusion and restoration under matched conditions.

**Controls:** fix writers, reference sets, backbone/training budget and validation-selected threshold policy. Compare oracle synthetic masks with predicted masks to separate mask-estimation errors from fusion behaviour. Inspect overlapping-window redundancy and alignment sensitivity; visible blank area is not the same as visible handwriting evidence.

**Risks:** downweighting may hide the very stroke differences exposing a skilled forgery. Measure FAR as well as FRR across occlusion severity; consider abstention when too little usable handwriting remains. A stamp mask is not automatically a reliability score, and none of the paper's experiments establish stamp robustness. Visibility weighting is a candidate contribution; novelty requires further literature checking.

**Reading anchors:** framework Figure 2 p. 4; MSDN Figure 6 and fusion pp. 6; region results Table 5 p. 8; WD procedure p. 10.  
**Open question:** can reliability weighting suppress interference without selectively discarding evidence of forgery?
