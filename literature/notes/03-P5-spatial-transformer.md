# 03 / P5 — Learning features for offline handwritten signature verification using spatial transformer network

**Xiao & Wu (2025)** · DOI: 10.1038/s41598-025-92704-3  
**Paper:** [local PDF](../papers/03-P5-spatial-transformer.pdf). Pages below are 1-based PDF viewer pages.  
**Personal progress, 19 October:** reader reports reading the guided summary and explains shared pairwise extraction, spatial transformation and difficult-example emphasis. This is summary-based study, not a claim of full-paper reading. Focal-loss speed misconception corrected in the accompanying explanation; confirmation of that correction remains pending.

## Main idea

Learn a shared signature-pair comparator, improve its view of handwriting through a spatial transformer (STN), and use focal loss to emphasize difficult training examples. The paper describes this as a writer-independent approach; it does not train F0-style personal SVMs. (pp. 1–4)

```text
Reference + query → fixed preprocessing
→ original views and learned spatially transformed views
→ shared CNN feature extraction → pair comparison
Training: original-pair focal loss + λ × transformed-pair focal loss
```

## Three useful concepts

- **Siamese network:** the same feature extractor, with shared weights, processes both reference and query. Their vectors can then be compared using a consistent representation. Shared weights alone do not guarantee better accuracy; the training objective and evaluation determine this. The paper's exact final comparison function is insufficiently clear to assume cosine similarity. (pp. 3–4, 6–8)
- **Spatial transformer:** learns affine adjustments such as translation, rotation, scaling and shear. Its localization network predicts parameters, a grid generator maps sample locations, and a sampler uses bilinear interpolation. Differentiability lets verification errors train the transformation. It changes the view of existing information; it does not recover strokes hidden by stamps. (pp. 5–6)
- **Focal loss:** downweights confidently correct examples so hard or misclassified pairs have more influence during training. **It does not skip easy pairs, automatically speed up training, or make inference faster.** α weights classes; γ controls easy-example downweighting (reported α=0.25, γ=2). Easy pairs are not automatically negative and hard pairs are not automatically forgeries. Hard examples may also be mislabeled or corrupted. (pp. 3, 7–8)

## Method and experimental context

- Preprocessing uses background separation, grayscale and moment-based position/size normalization. This is not demonstrated stamp removal. Four CNN blocks feed a 2,048-dimensional representation. (pp. 4–7, Table 1)
- Both original and transformed pair comparisons contribute to training; transformations are assumed to preserve pair labels. λ balances the objectives; the reported selected value is 3. (pp. 7–8)
- CEDAR, Bengali, Hindi and a Chinese forensic-case dataset provide different scripts/settings. Positive pairs contain genuine signatures of one writer; benchmark negative pairs combine genuine signatures with targeted forgeries. Many pairs reuse the same images, so pair count is not independent image count. The paper describes writer partitioning but leaves important split details unclear. Separate experiments across scripts do not establish cross-script transfer. (pp. 8–9)

## Reported findings — contextual examples, not deployment guarantees

Tables 5–7 (pp. 10–11) report CEDAR accuracy 97.02% (FRR 6.81%, FAR 1.15%), Bengali 94.42%, Hindi 91.76% and Chinese 75.44%. These accuracy values cannot be ranked directly against F0's EER or P1's different data/protocol. Exact threshold selection and reproducible splits require clarification before quantitative reuse. Figures illustrate processing, not proof that the model uses only reliable identity cues.

**Reporting caveats:** input sizes conflict between text and Table 1; λ search descriptions differ; class-imbalance direction reverses in the text; baseline labels/references are inconsistent. Presented experiments do not clearly isolate STN-only and focal-loss-only gains. Treat claimed superiority cautiously. (pp. 4, 7–11)

## Relevance to our thesis — hypotheses to test

A Siamese comparator is a candidate alternative to personal SVMs. STN alignment could be tested before or after stamp treatment, but no ordering or improvement is established here. It could focus on stamp structure, crop identity-bearing strokes or normalize away useful differences. Focal loss could improve learning from difficult signature pairs; runtime efficiency needs separate measurement and design.

**Possible controlled experiment:** same data, writer split, references and validation-selected threshold policy; compare a Siamese baseline, +STN, +focal loss and +both. Evaluate clean and controlled stamped queries, including false acceptance/rejection and stroke preservation where measurable. Restoration is another separate factor, not an automatic prerequisite.

**Decision:** retain these as candidate components; choose after reviewing the other papers and data feasibility. Aim for the strongest feasible, evidence-backed verifier rather than assuming more components improve it. Implementation awaits supervisor approval.

**Open question:** does learned alignment emphasize visible signature evidence or stamp-related shortcuts?
