# Second Research and Architecture Review

**Date:** 11 November 2026. **Scope:** research and planning only. Nothing was installed, downloaded, trained, run on data, committed or sent to anyone. No agreement was accepted and no dataset or weight file was fetched. The [initial review](INITIAL_REVIEW.md) is kept unchanged as history; corrections to it are listed in section 2. The resulting roadmap is [BUILD_PLAN.md](../../BUILD_PLAN.md).

## 1. Problem statement

A reviewer must compare a questioned handwritten signature with a few genuine references of a claimed writer when a document stamp overlaps the questioned signature. The system has to (a) remove visible stamp interference while recording exactly what it changed, (b) compare what remains with the references, (c) say how much usable evidence the comparison rests on, and (d) send weak or ambiguous cases to a human. It must not rebuild hidden strokes from the references, and it must not present an uncalibrated score as a verdict.

Two things are separate throughout this review: **removing visible stamp ink** (possible to varying degrees) and **recovering handwriting the stamp made unreadable** (not possible from the image alone; any method that appears to do it is guessing).

## 2. What changed from the first review, and why

| First review | Second review | Reason |
|---|---|---|
| Colour-separation cleaner "cannot hallucinate", treated as low identity risk | Deletion is still identity damage. The cleaner must report ink it removed or could not separate, and be measured on verification, not on appearance | A deleted or thinned stroke changes the signature as surely as an invented one (P1, P3 notes) |
| Same-hue and black stamps "marked ambiguous" by the cleaner | A colour method **cannot detect** a black or grey stamp, and with same-hue pen and stamp it can delete the signature itself. Self-detection is limited to what colour statistics show | Colour carries no signal in those cases. The first plan's exit check ("status `review`" for black stamps) assumed an ability the method does not have |
| No-op on clean input satisfies the requirement | The no-op status is renamed **"no stamp detected"**, never "clean", and an optional reviewer-indicated stamp region turns an undetected stamp into an explicit review case | A no-op can satisfy the rule mechanically while a same-colour stamp sits untouched on the signature |
| Frozen SigNet as the MVP encoder, "diagnostic only" | SigNet is a **research benchmark reference**, used only if the owner accepts the non-commercial statement for research. The customer build defaults to components with explicit permissive terms | "Diagnostic only" is not a licence exemption. Commercial status is unknown |
| Cosine similarity on SigNet features | The published use of these features is a per-writer SVM or a dissimilarity classifier, not raw cosine. A small comparator is planned on top of any frozen encoder | F0 note; [proto_hsv paper](https://arxiv.org/html/2606.18528) |
| Three overlapping windows through a whole-image encoder | Dropped. Regional comparison moves to **patch-token local matching**, where features are local by construction | P6's windows work with a network trained on 224×224 regions of a 512×224 normalised canvas; nothing supports feeding windows to a whole-signature CNN |
| Reliability = ink outside the stamp | Three separate quantities: similarity, evidence coverage (query-side and reference-side) and a review rule. Weighting may remove evidence, never add confidence | Query-side ink cannot show what the stamp hid; the references show where ink was expected |
| Synthetic fixtures only | Add a small **physically stamped, paired** set made by the owner with fictional signatures | P3: synthetic-only training gave Missing/Extra 5.25/3.06 on real scans against 2.19/1.77 for real-only |
| No training for the MVP; learning "later" | Still no training before first delivery, but a concrete, locally feasible training route now exists (see 4) | A CC BY 4.0 genuine-signature set of 420 writers was found |

Confirmed from the first review: proto_hsv does use SigNet-S features (trained on GPDS Synthetic users 5001–7000) with a linear or RBF SVM on dissimilarity vectors. The first review stated this from the README alone.

## 3. Findings by research question

**A. Cleaner.** No source examined offers a usable, licensed, signature-safe learned stamp remover. The newest end-to-end system found, [SignForensics (2025)](https://ieeexplore.ieee.org/document/11192260), again uses CycleGAN cleaning and reports appearance metrics (PSNR 29.5, SSIM 0.965), with no released code found and no stamped-versus-clean verification result at abstract level. The only public stamp segmenter found has a stamp IoU of 0.38 from 60 training images and licence "other"; rejected. Forensic practice separates signature and stamp inks by colour-channel mixing ([Chaikovsky et al. 2003](https://store.astm.org/jfs2002425.html); title and keywords verified, abstract not retrieved). For overlap that colour cannot resolve, [Jo et al. (2019)](https://arxiv.org/abs/1906.05229) trained a CNN on synthetic overlaps only and report real-document OCR improving from 71.13% to 92.50% after removing handwriting from print. That is evidence that synthetic-overlap training can transfer, for a different pair of classes.

Position: a classical colour cleaner is credible for **different-colour** stamp and pen, plausible but untested for similar hues (separation in optical-density space is our engineering hypothesis, not a sourced result), and **not credible** for black stamps, grayscale scans or identical inks. Those cases need structure (stamp geometry, a recurring stamp template), a reviewer hint, or a learned segmenter trained on synthetic overlaps.

**B. Comparator.** Four candidates were compared.

| Candidate | What its objective teaches | Use with few references | Local features | Permissions |
|---|---|---|---|---|
| SigNet / SigNet-S | Writer classification on genuine signatures (F0) | Published use: per-writer SVM, or dissimilarity classifier trained on other writers | No; fully connected head on a 150×220 input | Code BSD-3; author states models used GPDS, "restricted for non-commercial use" |
| [DetailSemNet](https://arxiv.org/html/2511.16364v1) (ECCV 2024) | Pairwise genuine-versus-forged, global L2 plus patch matching by Earth Mover's Distance | Pairwise comparator, no enrolment training | Yes; token-to-token flow, per-token weights | Repo MIT; weights for CEDAR, BHSig-B, BHSig-H; dataset terms not stated; code pinned to torch 1.12 + CUDA 10.2 |
| DINOv2 ViT-S/14 | Generic self-supervised image similarity | Needs a comparator on top | Yes; patch tokens | Code and weights Apache-2.0 |
| Own SigNet-style encoder | Writer classification on permitted data | As SigNet | Optional (conv maps) | Depends on training data; CC BY 4.0 option exists |

DetailSemNet's in-dataset EER is 0.58–2.11%, but trained on one dataset and tested on another it is 7.32–16.35% (its Table 3), and its CEDAR test split has five writers. In-dataset numbers therefore do not predict behaviour on the customer's signatures. No evidence was found that generic features separate skilled forgeries; DINOv2 is a baseline and a source of patch tokens, not an assumed solution. The self-supervised option examined ([SWIS](https://arxiv.org/html/2202.13078v2)) reports 72% accuracy on BHSig where supervised methods report about 85%; deferred.

**C. Unreliable regions.** The closest transferable idea is from partial person re-identification: [VPM](https://arxiv.org/abs/1904.00537) compares two images on regions visible in both, and later work supervises part visibility with simulated occlusion (search excerpts only; not read in full). DetailSemNet's EMD matching already accepts per-token weights, which gives a direct place to insert provenance-based reliability. Nothing examined tests any of this on stamped signatures. Each approach has a distinct failure: global comparison is disturbed by residue; fixed windows misalign; flexible local matching can match a forgery's strokes too easily; learned attention can learn to ignore exactly the region that exposes a forgery.

**D. Learning directions.** Shortlist and ranking are in the build plan (section 11). The strongest new fact: the [Mendeley "Offline Handwriting Signature" set](https://data.mendeley.com/datasets/ghvhwcpkbg/1) (420 writers × 30 genuine, CC BY 4.0) has the same order of writers as the 531 development writers F0 used to train SigNet. A SigNet-style encoder on clearly licensed data is therefore a small local training job, not a cloud project. Its quality is unknown until inspected.

**E. Permissions.** See the register below and the decision table in the plan. Unresolved: CEDAR terms (official page returned 403), BHSig260 terms (no official terms page found; the IAPR TC4 list links a third-party mirror), whether GPDS restrictions bind trained weights (the licence is silent; the model author asserts the restriction), and whether the partner's use is commercial.

**F. Evaluation.** Staged protocol in the plan (section 10). The points that most change the design: stamp templates must be split like writers; the same signature must not cross splits under different corruptions; the fixture generator must not use the cleaner's own mixing model alone, or the test is circular; error must be reported against retained coverage, not only overall.

## 4. End-to-end paths compared

| | A. First-review path | B. Evidence-first, permissive | C. Existing local matcher | D. Fully learned |
|---|---|---|---|---|
| Pipeline | Colour cleaner → frozen SigNet → window scores → weighted fusion | Colour cleaner with honest failure states and optional stamp-region hint → permissive encoder (DINOv2 tokens, later own encoder) → local matching with reliability → coverage and reference-relative score → review | Colour cleaner → DetailSemNet weights, reliability as EMD token weights | Learned segmenter or restorer → trained occlusion-aware regional network |
| Fit to partner's task | Medium; blocked if use is commercial | High; no permission dependency for delivery | Medium; unknown transfer to new writers | High in principle, late |
| Evidence strength | Encoder strong on clean benchmarks; windows unsupported | Cleaner and provenance sound; comparator unproven for skilled forgeries | Strong in-dataset, weak cross-dataset | P1 only (33% → 23% EER, still above 18% clean) |
| Main failure modes | Licence; window misalignment; same-colour stamps | Generic features may not see forgeries | Old CUDA-pinned code may not run on the M5; five-writer test sets | Needs data and long training; generative edits |
| Data and permissions | GPDS-derived weights | Apache-2.0 weights; CC BY data for own training | MIT code; weight training data terms unstated | Agreements, paired data |
| Local feasibility | High | High | Unknown until tried | Low without extra compute |
| Time to first useful result | Short | Short | Medium | Long |
| Research growth | Limited by frozen features | Grows into reliability-aware matching and own training | Good as a comparison system | Final vision if experiments justify it |

## 5. Recommendations

1. **Earliest credible prototype: path B.** Provenance-preserving cleaner, reviewer-visible evidence, a comparison whose score is shown only if a benchmark gate passes, coverage and review flags. It is the only path whose delivery does not depend on an unresolved licence.
2. **Fallback.** If generic features fail the gate, train the small clean-licence encoder (modest local training, owner approval needed) and meanwhile deliver the workbench with aligned side-by-side evidence, coverage and flags but no identity score. If colour cleaning fails on the partner's stamps, promote learned stamp/ink segmentation from stretch to research.
3. **Research path.** H1: reliability-aware local comparison is safer than global comparison of the cleaned image at matched evidence. H2: cleaning provenance predicts where the comparator fails, so a provenance-based review rule beats a score-only one. Both are adaptations of existing ideas to an untested setting; novelty is not claimed.
4. **Final vision, conditional.** A trained occlusion-aware local verifier with a learned stamp/ink segmenter, only if H1 holds on frozen features and the segmenter beats the classical cleaner on physical samples.

**Is this the best supported practical choice among those examined?** Yes for delivery order and permissions; not yet for comparison quality, which no examined source settles for our setting. It would be overturned by: the partner's use being non-commercial with GPDS-derived weights acceptable and SigNet clearly ahead on the benchmark (use path A's encoder inside B's structure); DetailSemNet running locally and transferring well to unseen writers (adopt C's matcher); or customer stamps being mostly black or same-ink (the cleaner choice changes before anything else).

## 6. Evidence register

All pages consulted on 11 November 2026 through a summarising fetch tool; wording should be rechecked at the source before formal citation.

| Source (date) | Establishes | Limits |
|---|---|---|
| [luizgh/sigver](https://github.com/luizgh/sigver) (repo, 2019-era) | BSD-3 code; SigNet, SigNet-F weights; statement that models used GPDS, non-commercial; centre on canvas, resize to 170×242, crop 150×220; WD classifier workflow | Feature use as raw cosine not documented; PyTorch version unstated |
| [proto_hsv paper](https://arxiv.org/html/2606.18528) (June 2026) | SigNet-S (GPDS Synthetic 5001–7000), 2048-d; dissimilarity vectors; linear SVM trained without skilled forgeries; 1–12 references | No cross-dataset test; some backbones got worse; reference fusion rule not stated |
| [DetailSemNet repo](https://github.com/nycu-acm/DetailSemNet_OSV) and [paper](https://arxiv.org/html/2511.16364v1) (2024/2025) | MIT repo; release weights; EMD patch matching with per-token weights; cross-dataset EER 7.32–16.35% | torch 1.12 + cu102; dataset terms absent; forgeries used in training; input size and parameter count not found |
| [DINOv2 repo](https://github.com/facebookresearch/dinov2) | "DINOv2 code and model weights are released under the Apache License 2.0"; ViT-S/14 has 21 M parameters | No signature evidence |
| [Mendeley signature set](https://data.mendeley.com/datasets/ghvhwcpkbg/1) (Aug 2023) | CC BY 4.0; 420 writers × 30; blue or black ink; resized to 500×600 | No forgeries; consent statement and scan resolution not on the page; one collection condition |
| [ChiSig](https://github.com/dskezju/ChiSig) | Skilled forgeries; "non-commercial purposes" only | Chinese names; Google Drive |
| [IAPR TC4 list](https://iapr-tc4.org/signature-datasets/) | Index of signature sets and links | States no terms for any set; CEDAR and GPDS not listed |
| [StaVer on Kaggle](https://www.kaggle.com/datasets/rtatman/stamp-verification-staver-dataset) | 400 stamped invoice scans, pixel masks of stamp strokes, black stamps included, some overlap with signatures | Licence shown as "Unknown"; original DFKI terms not found |
| [DDI-100](https://github.com/machine-intelligence-laboratory/DDI-100) | MIT repo; document images with stamp masks | How stamps were applied and their colours not verified |
| [SignForensics](https://ieeexplore.ieee.org/document/11192260) (2025) | YOLOv10 → CycleGAN cleaning → modified SigNet and CapsNet; reports 100% accuracy on CEDAR | Abstract-level only; no code found; protocol unread, so the accuracy figure is not relied on |
| [SWIS](https://arxiv.org/html/2202.13078v2) (2022) | Label-free pretraining on signature patches, ResNet-18 | Weak results, single runs, no code |
| [Jo et al.](https://arxiv.org/abs/1906.05229) (2019) | Synthetic-only training for overlapping handwriting/print segmentation transferred to real documents | Abstract only; different classes; no code link |
| [arXiv 2510.17724](https://arxiv.org/abs/2510.17724) (2025) | Cross-dataset degradation is a recognised open issue | Undergraduate thesis; abstract only, no numbers |
| Local notes P1, P3, P4, P6, F0 | As cited above | See each note |

## 7. Remaining uncertainties and what resolves them

| Uncertainty | Resolved by |
|---|---|
| Commercial status of the partner's use | Owner/partner answer |
| Stamp and pen colours; whether a few stamps recur; colour or grayscale scans | Partner answer or three sample scans viewed locally with authorization |
| Whether GPDS terms bind published weights | GPDS group or model authors; until then treated as non-commercial research only |
| CEDAR, BHSig260, MCYT, StaVer terms | Owner reads the official pages or the professor advises; not inferable from mirrors |
| Whether any permissive encoder separates skilled forgeries | Benchmark gate (plan M2b) |
| Whether the Mendeley set is clean enough to train on | Inspection after approved download |
| Whether DetailSemNet runs on arm64/CPU | A bounded load test, if the owner approves the attempt |
| PyTorch support for the installed Python 3.14 and loading of 2019 checkpoints | Official install notes at M2a |
| Novelty of H1/H2 | A dedicated related-work search before any thesis claim |

## 8. Rejected or deferred

- **Generative cleaning (CycleGAN, diffusion) for the product:** can add or remove strokes without record; no licensed signature-safe weights; P3 reports 8–21 s per image for diffusion. Kept only as a possible research comparison.
- **Public stamp segmenter (SegFormer-b0):** IoU 0.38, 60 images, licence "other".
- **Window scoring on SigNet:** unsupported by its architecture and training.
- **Self-supervised pretraining now:** weak published results; revisit only if labelled-writer training fails.
- **Riemannian, CCA and hybrid transformer verifiers (P7, P8, P2):** no released weights located; no occlusion evidence; high cost.
- **Automatic signature detection in full documents:** manual crop first; revisit after partner feedback.
- **Joint restoration and verification:** deferred until a segmenter is shown to help.

## 9. Counts and limitations

- **Searches:** 14 completed (three further attempts were rate-limited and returned nothing). **Pages examined meaningfully:** 16. **Paper reads:** three at full-text level (DetailSemNet, proto_hsv, SWIS), three at abstract level (SignForensics, Jo et al., arXiv 2510.17724), plus five local notes. Budgets of 20 / 18 / 6 were not exceeded.
- **Failed retrievals:** SignForensics full PDF, CEDAR data page (403), GPDS site (frames), the 2003 forensic abstract, the StaVer source page. Their rows above are correspondingly weak.
- Pages were read through an automatic summariser, so a misread detail is possible. VPM and its successors were seen only as search excerpts.
- No code was run, so every statement about feasibility on the M5 is an expectation.
- This is not a systematic survey; "not found" is not "does not exist", especially for stamped-signature verification work.
