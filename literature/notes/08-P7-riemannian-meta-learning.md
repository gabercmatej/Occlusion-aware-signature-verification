# 08 / P7 — Metric meta-learning and intrinsic Riemannian embedding for writer independent offline signature verification

**Giazitzis & Zois (2025)** · DOI: 10.1016/j.eswa.2024.125470  
[Local PDF](../papers/08-P7-riemannian-meta-learning.pdf). Pages below are 1-based PDF viewer pages.  
**Reader contribution:** supplied a synthesis and thesis experiment ideas on 25 October 2026. This note consolidates that synthesis and checked source passages; no full-paper reading or day completion is inferred.

## Main idea and pipeline

Describe a signature through statistical relationships between predefined image features rather than a conventional CNN embedding. Compare covariance matrices using suitable geometry, embed pairwise comparisons in a compact vector, then learn a shared comparison metric. (pp. 4–7)

```text
Signature → thresholding/thinning → ten per-pixel feature planes
→ 10×10 signature covariance matrix (foreground trace pixels)
Reference/query matrices → distances between matching feature submatrices
→ ERP pair-comparison vector → SigmML learned metric → dissimilarity
→ minimum score across ten references → threshold evaluation
```

## Representations and learning

- **Ten features:** preprocessed intensity, first derivatives Ix/Iy, second derivatives Ixx/Ixy/Iyy, gradient magnitude/direction and normalized x/y coordinates. These capture intensity variation and spatial relationships; they are not direct measurements of writing dynamics or a dedicated curvature detector. Background pixels are excluded. A small diagonal regularizer (10⁻⁴ I) makes matrices strictly positive definite when needed. (p. 4)
- **Covariance / SPD:** covariance summarizes feature co-variation; diagonal entries reflect variances and off-diagonal entries reflect joint variation. Symmetric positive-definite matrices have constrained structure, motivating geometry-aware comparisons. Statistical summaries do not retain every stroke or its exact arrangement.
- **AIM:** Affine-Invariant Metric compares SPD matrices under an appropriate geometry; AIM and Log-Euclidean Metric (LEM) perform strongly in the preliminary comparisons, and AIM is selected. Affine invariance of covariance-feature transformations is not a guarantee of immunity to arbitrary image rotations, stamp addition or missing strokes. (pp. 5, 8–10)
- **ERP — Embedded Riemannian Pyramid:** distances between corresponding overlapping principal submatrices form a pairwise descriptor. Full construction has 45 components; selection yields lower-dimensional versions, including a shared five-component configuration using submatrix sizes 7 and 10. These are **feature-coordinate submatrices**, not P6-style physical image regions. ERP is a description of a pair, not five raw features describing one signature. (pp. 5–6, 10–12)
- **SigmML:** a Riemannian meta-learner uses modified LSTMs to learn optimization direction and step size, maintaining an SPD Mahalanobis parameter. A contrastive objective encourages smaller dissimilarity for genuine pairs and larger values for forgery pairs. This learns a shared verifier, not a separate personal SVM. A five-dimensional symmetric metric parameter has 15 unique entries; that does not mean the entire optimizer and preprocessing pipeline have only 15 parameters or negligible cost. (pp. 6–7, 12)
- **Configuration selection:** Cluster Separability Measure (CSM) ranks genuine-pair and forgery-pair descriptor separation using learning data, reducing the expense of training every possible meta-learner. (pp. 8–10)

## Evaluation and reported results

**Protocol (p. 8):** intra-dataset testing uses repeated 5-by-2 writer-disjoint splits. Learning-pair data are divided 70/30 for training/validation. Cross-lingual experiments train on one dataset and test on another. Ten genuine references are randomly selected per test writer; each query gets ten scores and the **minimum dissimilarity** is retained. Reference selection/testing is repeated ten times. Training negatives vary between random-only, skilled-only and mixed forgeries. These are distinct conditions.

| Dataset | Reported summary EER, Table 10 (p. 12) |
|---|---:|
| CEDAR | 0.04% |
| Bengali | 0.03% |
| Hindi | 0.19% |
| MCYT-75 | 0.17% |
| UTSig | 0.75% |

These reproduce the paper's summary row; they must not be treated as one matched experimental configuration. Table 9 (p. 11) varies training dataset and random-forgery ratio; the summary values require tracing to their individual conditions before proposal citation. Main tables describe average EER across writers; pooled global ROC illustrations are separate. Very low EER does not establish a validation-locked global operating threshold, real-world accuracy, one-reference performance or stamp robustness. Dataset-version/count descriptions also need reconciliation before reproduction. (pp. 8, 10–12)

Minimum-over-references means the best-matching reference determines the score. It may tolerate genuine variation but can also allow an impostor to benefit from one unusually close match; evaluate aggregation alternatives under fixed conditions.

## Limitations and thesis role — our interpretation

No stamp-occlusion evaluation is reported. A stamp changes gradients, foreground selection, coordinate normalization and covariance; interference can therefore affect the entire descriptor. Thresholding/thinning and masking may remove useful stroke evidence. Using only visible pixels also changes the sample population and can make covariance unstable or poorly comparable when little evidence remains.

**Candidate experiment:** establish a covariance+AIM baseline; measure degradation under controlled stamp severity; compare unmasked, oracle-masked and predicted-mask covariance against CNN/hybrid and regional baselines. Fix writer splits, reference count, aggregation and validation-selected threshold policy; report FAR/FRR and EER separately. Include a minimum-evidence/abstention rule and check covariance conditioning. Full SigmML is an optional extension, not required for the first baseline.

P6 is more directly suited to spatial visibility weighting. P7 offers an alternative statistical representation and a reminder that feature geometry and reference aggregation matter. Neither its novelty for occlusion nor superiority under stamps is established.

**Cost/access:** the paper notes expensive ERP construction and reports multi-hour learning per fold. It announces [new code](https://bitbucket.org/agiaz/ESWA_2024) and links [earlier SigmML code](https://bitbucket.org/agiaz/sigmml) (p. 3); current availability has not been checked. These are recorded references, not accessed external resources.

**Reading anchors:** covariance features p. 4; ERP Figure 1 p. 5/Table 1 p. 6; protocol p. 8; five-dimensional results Table 9 p. 11 and summary p. 12.  
**Open question:** can covariance from reliably visible handwriting remain sufficiently discriminative and stable as occlusion increases?
