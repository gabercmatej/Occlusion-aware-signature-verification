# 02 / F0 — Learning features for offline handwritten signature verification using deep convolutional neural networks

**Hafemann, Sabourin & Oliveira (2017)** · DOI: 10.1016/j.patcog.2017.05.012  
**Paper:** [local PDF](../papers/02-F0-foundation-feature-learning.pdf). Page numbers below are 1-based PDF viewer pages.  
**Personal progress:** reader explicitly confirmed reading and understanding on 19 October 2026. Notes synthesize the local paper and discussion. P5 and the daily learning block are not marked complete.

## Main idea and pipeline

Learn one **shared CNN feature extractor** from many development writers, then train a **writer-dependent support vector machine (SVM)** for each new enrolled writer. SVM means support vector machine, not single vector machine. Few references cannot support training a large CNN separately for every person. (PDF pp. 3–6)

```text
Development writers → train CNN → shared feature extractor

Enrollment:
New writer's genuine references (+) + other writers' genuine signatures (−)
→ shared CNN → feature vectors → train that writer's SVM

Verification:
Query image → preprocessing → same CNN → 2,048-number vector
→ claimed writer's SVM → score → threshold → accept/reject
```

The CNN transfers to unseen writers; the final decision system is writer dependent. Enrollment trains the small SVM, not the whole CNN. Other writers' genuine signatures act as random-forgery negatives. Skilled forgeries of the enrolled writer are held back for testing, not required to train their SVM. Linear and more flexible RBF SVMs are evaluated; class weights compensate for many more negatives than positives. (pp. 6–7)

## How they learn the features

- **SigNet — genuine-only learning:** train the CNN to identify development writers from genuine signatures. Then use the representation before its final classification output as features for new writers. The original writer labels are a training exercise, not the final list of customers. (p. 4)
- **Forgeries as separate classes:** give each writer a genuine class and a forgery class. This performed worse than genuine-only learning on their validation set. More classes and strong resemblance between imitations and originals may explain the difficulty. (pp. 4–5, 9)
- **SigNet-F — multitask learning:** shared layers learn through two outputs: writer identity and genuine/forged status. Their selected formulation applies the identity loss only to genuine signatures, while the forgery loss uses both genuine signatures and forgeries. A weight λ balances the losses. This combined learning improved GPDS verification; combining tasks was not itself the unsuccessful idea. (pp. 5, 8–10)
- **Auxiliary forgery output:** useful for shaping features, but less effective as a standalone detector for unseen writers. In their specific test, its error rose from 2.21% on held-out images of CNN-training writers to 14.37% on separate validation writers. The final verifier still uses references through a personal SVM. (p. 8)

## Evaluation and findings worth retaining

- **Writer separation:** CNN learning uses 531 writers; verification-method validation uses 50 different writers; final GPDS evaluations use separate groups of 160 or 300 writers. Internal CNN validation shares training identities but uses separate samples; it is different from verification validation on unseen writers. Method choices use development validation, not final test results. (pp. 6–7, Figure 3)
- **Threshold qualification:** on GPDS-160, with 12 genuine references per writer and RBF SVMs, SigNet-F reports **1.72% EER with optimal user-specific thresholds**, versus **3.61% global-threshold EER**. Genuine-only SigNet reports 2.63% user-threshold EER in the corresponding comparison. These EERs concern skilled forgeries; they are not general accuracy claims. Practical selection of personal thresholds is left unresolved. GPDS FAR/FRR are separately reported with a global threshold selected on validation writers. (p. 10, Tables 6–7)
- **Transfer:** GPDS-trained features are reused without CNN fine-tuning on MCYT, CEDAR and Brazilian PUC-PR, with new personal SVMs. Transfer works well, but SigNet-F does not consistently beat SigNet: no significant difference on MCYT/CEDAR, and worse performance on Brazilian PUC-PR. The authors suggest learning dataset-specific forgery traits. (pp. 10–11)
- **References:** more genuine enrollment examples generally help, with diminishing returns. Natural within-writer variation matters alongside sample count. (pp. 2–3, 11–12, Figure 6)

## Relevance to our thesis — interpretation, not a tested result here

**Domain shift:** feature behaviour can change with scanning, resolution, background interference, stamps and restoration artifacts. F0 demonstrates transfer across signature datasets, but does not establish stamp robustness or test restoration. A cleaner-looking restored image may still lose or invent identity-bearing strokes.

**Possible baseline:** use a shared signature feature extractor with a simple decision method; compare unchanged, stamped and cleaned/masked queries under the same writer split, reference count and validation-selected threshold policy. Whether to use personal SVMs or a shared pairwise comparator is a design choice, not settled by this paper. No implementation before supervisor approval.

**Preprocessing matters:** signatures are already extracted from documents. They centre, remove background, invert and resize/crop images; they do not solve signature detection or stamp separation. Their signature-trained CNN is distinct from generic ImageNet models, which serve as baselines. (pp. 5–7)

**Takeaway:** good shared features can support verification with few genuine references, but training objective, data conditions, reference availability and threshold policy determine what the result means.

**Open research question:** do identity-discriminative features remain reliable after stamp occlusion and restoration?  
**Next:** P5 — shared Siamese comparison, learned spatial alignment and focal loss.
