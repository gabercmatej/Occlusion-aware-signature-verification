# Proposal planning sheet — 27 October 2026

**Status:** working proposal for reader/supervisor review; no implementation or empirical results. Local paper notes support the method; official web sources below were inspected October 27 for preliminary data feasibility. No dataset archive was downloaded or tested. No company data accessed. No files pushed or messages sent.

## Problem and practical setting

Offline signature queries on real documents commonly overlap stamps, while genuine references are expected to be clean. Stamps may disrupt similarity measurements; generative restoration can remove or invent identity-bearing strokes. The proposed study tests whether scoring visible regional evidence improves verification under controlled stamp interference while retaining skilled-forgery rejection. Public data with synthetic overlays provide an independent core study; authorized company documents provide a separate transfer pilot, conditional on available references and reliable labels.

**Confirmed by reader:** writer-independent training using public data; separate cousin/company real-world testing; stamped queries are common; clean references are a working assumption to confirm. Synthetic data may help training, not only evaluation; digital-versus-physical differences should be examined separately.

## Recommended questions for the draft — supervisor review pending

- **Primary RQ:** Under controlled stamp overlap, can visibility-aware regional comparison reduce genuine-signature rejection at a validation-selected false-acceptance target compared with equal regional averaging? Evaluate achieved test FAR as well as FRR; do not assume the target transfers to test data.
- **Secondary RQ1 — architecture:** Does one selected CNN–transformer hybrid improve robustness or the error/runtime tradeoff over a shared CNN comparator under matched evaluation conditions?
- **Secondary RQ2 — training/transfer:** Does synthetic-stamp augmentation improve performance on held-out stamp styles and, if available, a separately collected physical-data pilot?

Restoration is an optional additional control, not another required main research question. If data/compute are limited, protect the CNN/regional/visibility comparison and drop the architecture or restoration extensions. These are recommended draft choices made at the reader's request, not claims of final supervisor approval.

**Contribution hypothesis:** controlled evidence about regional reliability, augmentation and failure modes. No first-ever claim, promised improvement or human-superiority result.

## Preliminary dataset shortlist

| Candidate | Intended role | Verified source evidence and remaining checks |
|---|---|---|
| CEDAR | Small real-handwriting feasibility benchmark and possible core pilot | [University source](https://cedar.buffalo.edu/NIJ/publications.html) links its signature archive. Local reviewed papers report 55 writers, 24 genuine and 24 skilled forgeries per writer. Archive/download integrity and applicable usage/derivative/redistribution terms remain unverified. Too small alone for strong low-FAR claims or large-model-from-scratch training. |
| GPDS Synthetic | Larger synthetic-development candidate | [Official GPDS page](https://gpds.ulpgc.es/downloadnew/download.htm) describes 4,000- and 10,000-identity variants with non-commercial research agreements. Access agreement/delivery and exact version remain unresolved; do not conflate versions or generated forgeries with human skilled imitations. |
| BHSig260 | Additional real-handwriting training/transfer candidate | Original authors describe research availability and human imitations in their [dataset paper](https://core.ac.uk/download/pdf/143896733.pdf). Original distribution, current terms and archive still need checking; no mirror license presumed authoritative. |
| UTSig | Optional cross-script stress test | [Author page](https://sites.google.com/view/asoleimanib/publications) provides dataset links and citation instructions. Precise use terms and forgery taxonomy need checking; descriptions differ across papers and author page. |
| SSDV | Newer document-noise alternative to investigate | [Official researcher repository](https://github.com/SU-Signature/SSDV) describes genuine/forged signatures on noisy documents and displays a CC0 repository license. Dataset licensing scope, contents, labels, writer split and downloadable payload are not yet verified; README comparison counts are inconsistent, so not selected on headline counts. |

**Fallback decision rule:** use the first suitable no-fee research dataset whose original access conditions and labels can be verified; select by writer count, genuine/reference availability, human skilled-forgery provenance, image quality and reproducibility, not recency. CEDAR is provisional for a small feasibility study; larger development data are preferred if promptly accessible. If no acceptable access is secured, keep the proposal conditional and ask the supervisor rather than claim guaranteed feasibility. Data permission does not imply permission for commercial use or public GitHub redistribution.

## Data construction and controls — proposed defaults

- Clean genuine references; synthetic stamps applied to genuine and skilled-forged **queries**, with matched stamp distributions across labels. Retain unstamped controls. Generated stamp graphics contain fictional text and no actual company seals.
- Primary reference count **3**, reflecting the working practical scenario; sensitivity **1 and 5** if data support it. Distinct genuine query images never double as their own references. These counts are proposed, not confirmed company facts.
- Split public writers provisionally **60/20/20** train/validation/test, with integer counts and seed set after dataset selection. Repeat writer partitions if feasible and report writer-level uncertainty; never split augmented copies or pairs independently. No writer overlap with external company evaluation.
- Hold out stamp templates/families for testing; separate train/validation template sets. Use identical query instances across methods. Vary shape, colour, opacity, blur and placement, including ink colours similar to handwriting.
- Define overlap as fraction of original foreground-signature pixels intersecting the stamp-support mask; this is contamination, not guaranteed information destruction. Initial bands: clean, >0–10%, >10–25%, >25–50%, >50%. Preserve source image, alpha/compositing settings, masks and random seed. Document translucent versus opaque conditions separately.
- Compare matched clean-only versus stamp-augmented training for core methods. All tuning uses public development/validation data. A physical pilot uses the frozen model/threshold; adaptation would require a separately reserved company development group, not test-data tuning.

## Experiment table

| ID | Method/comparison | Purpose and control |
|---|---|---|
| A | Whole-image shared CNN comparator | Simple reference point; F0-inspired signature features, but shared pair decision for WI deployment, not a personal SVM claim. Exact backbone provisional. |
| B | Regional shared comparator with equal averaging | P6-inspired baseline; use the same regional model and inputs as C. |
| C | Same regional model + visibility-weighted score fusion | Isolate weighting benefit. Compare oracle synthetic masks versus predicted masks; oracle masks provide an idealized localization diagnostic, not deployment performance or a guaranteed upper bound on verification accuracy. |
| H | One selected hybrid CNN–transformer comparator | Bounded secondary architecture comparison. Use the same writer/reference/stamp protocol and comparable tuning budget; report pretraining, capacity and runtime differences. Add visibility weighting to the hybrid only if needed to answer a measured interaction. |
| D | Optional restoration before a fixed comparator | Secondary extension under the same data/reference protocol; evaluate content alteration and additional runtime. |
| Training factor | Clean-only versus matched synthetic-stamp augmentation for A–C | Separate augmentation gains from weighting gains. Keep source writers, training budget and test queries matched. |
| External pilot | Frozen selected pipeline on authorized real queries | Transfer evidence; do not claim final performance before data/labels exist. |

**Primary operating measure:** FRR at a threshold chosen on validation skilled-forgery comparisons to target FAR ≤1%, subject to sufficient independent writers/forgeries. Lock one threshold per system/training condition and reference policy; do not retune by test writer or severity. The 1% target is provisional, not a banking safety standard. Report achieved test FAR with uncertainty alongside FRR; if sample support is inadequate, do not claim reliable 1% or rarer-risk performance. Supplement with skilled/random-forgery EER separately, ROC/AUC, severity breakdown and runtime. Regional abstention reports coverage and errors including a fixed coverage comparison; abstaining is not a correct classification. Validate any abstention rule before final testing.

## Real-world pilot and forgery limitation

Company data are optional until access, permission, writer IDs, clean references, query provenance and authenticity labels are confirmed. If the real set contains only known genuine queries, report genuine acceptance/rejection and operational observations; it cannot estimate skilled-forgery FAR or EER.

Reader proposes creating forgeries: distinguish consented human imitations, ordinary other-writer impostors and digitally generated/manipulated samples. Preserve collection protocol and provenance; never label all of them “skilled forgery.” Real documents with unknown authenticity are not trusted negatives. Physical stamping of public/consented samples could independently test digital-to-physical transfer without company-data dependence.

## Decisions still needed

1. Supervisor/reader review of recommended primary question, architecture secondary question and provisional reference/data defaults; exact dataset selection remains open.
2. Actual dataset access and use terms; choose a main dataset and verify files/labels before implementation.
3. Supervisor agreement on WI comparator versus personal-SVM alternative, operational metric and minimum evaluation size.
4. Company permission, clean-reference availability/counts, genuine-label confidence, and how consented imitation samples would be collected.
5. Compute budget, exact feature extractor, mask estimator, reliability rule and restoration extension. More architectures are not a prerequisite for this proposal.

## Draft bridge — revised October 27; draft due October 28

This sheet supplies scope, data and experiment bullets for the full proposal. Next, write 2–4 pages plus references, reconcile the literature matrix and add a roughly 300-word related-work synthesis. Keep unresolved decisions explicit. Target complete draft and first review October 28 and final review/ready-to-send package October 29; reader sends personally. Implementation starts after supervisor approval.

## Scope decision for tomorrow — October 27

**Reader's priority:** a useful, reliable solution to the cousin's stamped-signature workflow is more important than novelty for its own sake. The reader delegates a recommendation and wants a few alternative main questions available for the professor. Architecture development remains an interest, not an obligation to invent a new backbone.

**Recommendation:** make occlusion-aware regional verification the main investigation, supported by one CNN baseline and at most one selected hybrid architecture comparison. P6 directly motivates spatial score fusion; P2 motivates a bounded hybrid alternative but does not establish stamp robustness. Which model is operationally best remains an empirical question. Do not commit to a full Cartesian product of architectures, losses, cleaners, mask methods and training recipes.

**Alternatives to present briefly to the professor:**

| Direction | Alternative primary question | Why/when to choose it |
|---|---|---|
| Architecture-centred | Which of a selected CNN and CNN–transformer hybrid gives the better verification/error-runtime tradeoff under controlled stamp occlusion? | If architectural comparison is preferred and data/compute support credible matched training. No universal “best architecture” claim. |
| Training-centred | How much does clean–stamped pair augmentation improve transfer to unseen stamp styles and physically stamped documents? | If data realism is the main bottleneck and a reliable physical test can be collected. |
| Restoration-centred | When does stamp removal improve verification, and when does lost or generated handwriting increase errors? | If masks/restoration resources are available; downstream verification remains the outcome. |

**Practical product target after approval:** reference/query input with a match score and human-review recommendation when evidence is insufficient. Initially assume cropped signatures; document extraction/localization, a polished application and automated financial decisions are outside the core experiment. Assess feasibility against actual document layout once authorized data arrive. This is a proposed prototype goal, not a shipped capability.

**How we choose the eventual implementation:** consider achieved FAR/FRR with uncertainty, stamp-severity behaviour, fraction requiring human review, latency, memory/compute and failure modes. A model with lower EER but unsupported operating-point reliability is not automatically the better practical choice. Choose using development/validation evidence; reserve final test and company test data for assessment. Any company-specific adaptation needs a separate development set.

**Custom/modern architecture policy:** during the implementation literature update, consider one additional architecture only if it addresses a diagnosed failure or clear resource constraint; document what it replaces and drop another optional comparison to keep scope bounded. Current state of the art is not exhaustively assessed by the nine-paper review. A custom visibility/fusion module may be the model contribution even with an established feature extractor; novelty must be checked. Negative results remain informative and may favour a simpler practical system.

## Ready-to-draft checklist for Wednesday October 28

- [x] Glossary reviewed by reader; nine paper notes saved.
- [x] Reader's practical objective and clean-reference/stamped-query assumption recorded.
- [x] Recommended main question, secondary questions and professor alternatives prepared.
- [x] Provisional data fallback, reference/split defaults, comparison table and evaluation policy documented.
- [x] Unknown access terms, forgery provenance, company permissions, mask feasibility and compute constraints explicitly retained.
- [ ] Verify/reconcile all nine literature matrix rows against saved notes.
- [ ] Write roughly 300 words of related work using the notes: P1 problem, F0 foundation, P6 regional motivation, P2 architecture alternative, P3/P4 preservation risks; P5/P7/P8 as bounded alternatives.
- [ ] Write complete 2–4 page proposal v0.1 plus references using proposal/OUTLINE.md; include recommended scope and a short “alternatives for supervisor discussion” paragraph.
- [ ] Perform first claim/scope review; final review, diagram and export remain Thursday October 29.

No additional full-paper reading or model implementation is required tonight. Scope preparation for the October 26 task is complete on October 27 at the proposal-planning level. Access confirmation and final supervisor scope approval remain open and must not be represented as achieved.
