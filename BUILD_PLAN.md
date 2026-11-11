# Signature Verification Build Plan

**Status — 11 November 2026:** second research review complete; this roadmap replaces the M0 plan. No product code, environment, trained model or measured result exists yet. Evidence and sources: [second review](docs/research/SECOND_REVIEW.md); history: [initial review](docs/research/INITIAL_REVIEW.md). This file is the single authoritative plan. Documents under `docs/superpowers/` are background only.

Labels used below: **[C]** needed for the early customer prototype, **[R]** research development, **[S]** optional stretch work that needs evidence first.

## 1. Goals

1. **Early local prototype for the cousin's document review.** It helps a reviewer inspect a stamped signature against genuine references: what the stamp covered, what cleaning changed, how the remaining evidence compares, and when evidence is too weak to say anything. It must be useful before any research milestone finishes.
2. **A substantial research and deep-learning project** for master's and internship applications, with a credible route to a thesis. The thesis is to be discussed with the professor in January 2027; approval, scope and novelty are not established.

The two goals share one codebase. Each milestone is independently useful, and the customer prototype (M1–M3) does not depend on any research or stretch milestone.

## 2. Owner requirements

These are fixed unless the owner changes them. A reviewer may challenge one, but only as "proposed — owner approval required".

| ID | Requirement |
|---|---|
| R1 | Every product query passes through stamp removal before identity comparison. A clean input may produce an explicitly measured no-op |
| R2 | Preserve the untouched original, the stamp estimate, the intervention mask, the actual change map, and uncertainty/provenance information |
| R3 | Give greater influence to useful signature evidence less affected by stamping and removal. The mechanism is open and must be tested |
| R4 | Never reconstruct a query from the claimed writer's references |
| R5 | Customer data and identifying artifacts stay local; nothing is uploaded to external services |
| R6 | An uncalibrated similarity score is not an authenticity probability or a verdict |
| R7 | Insufficient or ambiguous evidence stays visible to the reviewer |
| R8 | Public documentation and demonstrations use permitted material and report real results |
| R9 | Split by writer and source document before augmentation; fit, select thresholds and test on separate data |

**Pending owner decision (unchanged from the first review).** *Proposed — owner approval required:* score the **uncleaned** query as an evaluation-only control, never shown as a product result and never used in an assessment. Its value: it is the only way to measure whether mandatory cleaning helps or harms the comparison; P1 measured exactly this (33% EER stamped, 23% cleaned, 18% unstamped). Without it we can measure how far cleaned queries fall short of clean ones, but not whether cleaning beat doing nothing. Evaluation rows that need it are marked *(control)* and run only if approved.

**Design consequences of the requirements (not changes to them).**

- R1's no-op is reported as **"no stamp detected"**, never "clean". A colour-based cleaner cannot see a black or same-ink stamp, so a no-op is not evidence that no stamp is present.
- R3 is bounded by a safety rule: reliability weighting may remove evidence from a comparison but may never raise confidence. Less usable evidence moves a case toward review.
- R4 allows references to be used for *measuring coverage* (where ink was expected), since nothing is drawn into the query.

## 3. Assumptions and open owner questions

Assumed until corrected: inputs are full documents and signature crops, genuine references and a claimed writer; one reviewer works locally on a laptop; the first customer sample is small (about ten people, three to five references, five to ten queries each) and serves inspection, not evaluation.

Unconfirmed, in order of effect on the plan:

1. Is the cousin's use commercial or operational? (Decides which weights and datasets may appear in the customer build.)
2. Stamp and pen colours; colour or grayscale scans; do a few stamp designs recur? (Decides whether the classical cleaner is enough.)
3. Decision on the evaluation-only uncleaned control.
4. May public datasets and weights be downloaded for local research once their terms are read? Which terms does the owner accept?
5. Will the owner make a small physically stamped test set with fictional signatures (section 10)?
6. References per person, file formats, scan resolution, and whether any known forgeries or labels exist.

None of these blocks M1a.

## 4. What is established and what is not

| Status | Statement |
|---|---|
| Supported by primary evidence | Stamps degrade verification and cleaning recovers part of the loss for one model (P1). Synthetic-only stamp data transfers poorly to physical scans (P3). Signature features trained on about 500 writers transfer to unseen writers (F0). Local patch matching helps in-dataset, and cross-dataset error is several times higher (DetailSemNet). DINOv2 code and weights are Apache-2.0. A CC BY 4.0 set of 420 writers' genuine signatures exists |
| Architectural hypotheses | Colour separation is adequate for the cousin's stamps. Provenance-based reliability improves comparison. Local matching with reliability is safer than global comparison. Generic patch features carry useful signature evidence |
| Untested claims | Everything about accuracy, speed and memory on the M5. Whether any permissive encoder separates skilled forgeries. Whether the cleaner can flag its own failures. Whether old checkpoints load in a current PyTorch |

## 5. Selected architecture

```text
document or crop ──► manual crop (+ optional reviewer-marked stamp region)
      │
      ▼
CLEANING [C]   stamp estimate → conservative removal (no inpainting)
      │        artifacts: original + hash, stamp mask, cleaned image, intervention mask,
      │        change map, ink mask, ambiguous mask, report.json (status, reasons, metrics)
      ▼
NORMALISATION [C]   ink bounding box, scale and position normalisation; masks transformed identically
      ▼
ENCODING [C]   pluggable encoder: global vector and/or patch tokens (cleaned query, references)
      ▼
COMPARISON [C]   similarity per reference → aggregate → position relative to the references' own spread
      │          reliability enters here: masked or weighted local matching (equal weighting kept as control)
      ▼
EVIDENCE + REVIEW [C]   coverage (query-side, reference-side), flags, review recommendation
      ▼
REVIEWER UI [C]   original / cleaned / masks / references side by side; notes; local case export
```

**Module boundaries.** One installable package `osv` under `src/osv/` (name is a default the owner may change before M1a), with the existing empty folders moved beneath it.

| Module | Responsibility | Must not |
|---|---|---|
| `osv.datasets` | Synthetic fixtures, dataset readers, manifests, split files | Read customer data without authorization |
| `osv.segmentation` | Stamp, ink and ambiguous-region estimates from the observed image | Receive ground-truth masks or references |
| `osv.preprocessing` | Removal, normalisation | Draw strokes; modify the original |
| `osv.provenance` | Artifact writing, report schema, hashing | Drop any artifact silently |
| `osv.verification` | Encoders, comparators, reference enrolment | Read the uncleaned query in product mode |
| `osv.uncertainty` | Coverage, review rule, later calibration | Output "genuine"/"forged" before calibration |
| `osv.evaluation` | Metrics, benchmark runners, ablation tables | Tune on final-test writers |
| `osv.cli`, later `osv.app` | Commands and the local UI | Send data off the machine |

**Data contracts.** Images are RGB `uint8` arrays; masks are boolean arrays of the image's height and width. Every stage's output carries the versions of the code and configuration that produced it. `report.json` has a `schema_version`.

## 6. Intake, cropping and reference enrolment [C]

- **Intake.** Accept common image formats and PDF pages rendered locally. The original file is copied byte-for-byte and hashed before anything else happens.
- **Cropping.** Manual rectangle first. The reviewer may also mark the stamp region; this hint is optional and is recorded as provenance. Automatic signature detection is stretch work [S].
- **Enrolment.** A writer is a pseudonymous ID with a folder of genuine reference crops, their hashes, and cached features tagged with encoder name and weight hash. References go through the same cleaner in checking mode: a reference in which a stamp is detected is flagged, because a stamped reference distorts every later comparison.
- **Reference quality shown to the reviewer.** Number of references, and how similar they are to each other (leave-one-out). Fewer than three references, or references that disagree strongly, is itself a review reason. References signed in one sitting understate natural variation (P1), so a tight spread is not proof of a strict writer.
- **Local only.** `data/company/` stays git-ignored; exports are written only where the reviewer chooses.

## 7. Cleaning and provenance [C]

**Position.** A classical cleaner is the starting point because every change it makes is a recorded deletion and it needs no training data. It is not assumed safe: deletion can erase or thin strokes, and overlap pixels that are kept still look different from clean ink.

| Case | Expected behaviour of the classical cleaner | Handling |
|---|---|---|
| Stamp and pen of different colours | Separable | Remove stamp-only pixels; keep overlap pixels and mark them ambiguous |
| Similar hues | Partly separable at best | M1b tests separation in optical-density space; low separability → review |
| Same hue (blue pen, blue stamp) | Danger: the "stamp colour" also describes the signature | Guard: if the stamp colour model explains most ink, delete nothing and return review |
| Black or grey stamp, grayscale scan | Invisible to colour | "No stamp detected" unless the reviewer marked a region; then review with the region marked unreliable |
| Faint strokes, JPEG, anti-aliasing, textured paper | Edge pixels mix colours | Measured on fixtures and on physical samples; soft masks rather than hard thresholds where it helps |
| Translucent stamp over ink | Overlap pixels darker than either ink | Kept, marked ambiguous, down-weighted |

**What the cleaner can and cannot report about itself.** It can report how separable the two ink colours were, how much estimated ink lay inside the stamp, and how many pixels it changed. It cannot certify that no same-colour stamp is present or that a kept dark pixel is handwriting. The status vocabulary reflects this: `no_stamp_detected`, `cleaned`, `review` (with reasons such as `stamp_indicated_not_separated`, `ink_and_stamp_colour_overlap`, `low_reliable_ink`, `grayscale_input`).

**Inference never receives ground truth.** Synthetic masks are used only by tests and metrics. The cleaning API accepts an image and an optional reviewer region, nothing else.

**Upgrade routes, in order of cost:** optical-density unmixing (M1b) → stamp-template registration if the cousin's stamps recur → learned stamp/ink segmentation trained on synthetic overlaps (M7). Restoration of hidden strokes is not on the route; no examined method can know what a stamp hid.

## 8. Comparison, reliability and review [C → R]

Three quantities are defined separately and never merged into one number.

- **Similarity.** Output of the comparator between the cleaned query and each reference, aggregated over references (median by default; the rule is recorded). It is displayed relative to the references' own leave-one-out similarities ("inside / at the edge of / outside the spread of the references"). It is not a probability.
- **Evidence coverage.** *Query-side:* share of estimated query ink that is reliable (outside stamp, intervention and ambiguous areas). *Reference-side:* after alignment, share of each reference's ink that falls where the query is reliable. The second number is the one that shows hidden evidence, because only the references indicate where ink was expected. Alignment error is part of its uncertainty.
- **Review rule.** Before calibration every case is a review case, and the rule only orders and annotates: insufficient evidence (coverage below a provisional bound, cleaner status `review`, fewer than three references, inconsistent references) is shown first and without a similarity headline. After calibration (M4) thresholds come from validation data at an operating point the owner chooses, and the output is still a recommendation.

**Approaches to reliability, compared.**

| Approach | When it enters | How it can fail |
|---|---|---|
| Global features + coverage-based review | M2a baseline | Stamp residue still perturbs the whole vector; no localisation |
| Fixed windows with weights (first-review idea) | Not built | Windows do not hold the same handwriting after crop variation; unsupported for whole-image encoders |
| Patch-token local matching with reliability masks or masses and a spatial-consistency term | M2a simple version, M5 full | Flexible matching can pair a forgery's strokes with genuine ones; masking can hide the region that exposes a forgery |
| Learned occlusion-aware metric (visibility-aware parts, simulated occlusion) | M6/M8 | Needs training data; attention can learn shortcuts; hardest to audit |

Controls kept in every experiment: equal weighting; masking the same regions on both query and reference (compare like with like); oracle synthetic masks against predicted masks, to separate mask error from fusion behaviour.

## 9. Permissions and decision branches

Code, checkpoint, training data and evaluation data are tracked separately. "Diagnostic only" is not a licence category. Nothing below has been downloaded or accepted.

| Artifact | Terms found (11 Nov 2026) | Local research | Public demo | Cousin's operational use | Open point |
|---|---|---|---|---|---|
| DINOv2 ViT-S/14 code and weights | Apache-2.0 | Yes | Yes | Yes, under Apache terms | No signature-specific evidence |
| sigver code | BSD-3 | Yes | Yes | Code only | — |
| SigNet, SigNet-F, SigNet-S weights | Author: trained on GPDS, "restricted for non-commercial use" | If the owner accepts that statement for non-commercial research | Not until clarified | No, unless use is confirmed non-commercial and terms are clarified | Whether dataset terms bind weights is unresolved |
| DetailSemNet code / weights | Repo MIT; weights trained on CEDAR, BHSig260 | Likely; dataset terms unstated | Unclear | Unclear | Runs on arm64? Old CUDA-pinned stack |
| Mendeley Offline Handwriting Signature (420 × 30 genuine) | CC BY 4.0 | Yes | Yes with attribution | Yes with attribution | No forgeries; consent wording not on page; quality uninspected |
| CEDAR (55 writers, skilled forgeries) | Not verified; official page inaccessible | Unresolved | Unresolved | Unresolved | Owner or professor to read official terms |
| BHSig260 | No official terms found | Unresolved | Unresolved | Unresolved | Same |
| ChiSig | "non-commercial purposes" only | Yes (non-commercial) | Figures only if terms allow | No | Chinese names; domain gap |
| GPDS Synthetic | Signed agreement; non-commercial research; no redistribution | Only with a signed agreement | No | No | Possibly via the university |
| StaVer stamps (400 scans, stroke masks, black stamps) | Kaggle mirror: licence "Unknown" | Unresolved | No | No | Find the original DFKI terms |
| DDI-100 | Repo MIT | Yes | Likely | Likely | Stamp realism unverified |

**Branches.**

- *Commercial status unknown (today):* the customer build uses only rows marked "Yes" in the last column. Research benchmarks may use more, each recorded with its terms.
- *Cousin's use confirmed non-commercial and terms clarified:* a GPDS-derived encoder may be considered for the customer build if it wins the benchmark.
- *Cousin's use commercial:* permissive weights and own-trained models on CC BY data only.
- *No skilled-forgery dataset with acceptable terms:* skilled-forgery claims are not made; results are limited to random impostors and stated as such.

## 10. Evaluation protocol

**Stages.**

| Stage | Data | Establishes | Cannot establish |
|---|---|---|---|
| E0 | Synthetic fixtures with exact masks | Code correctness, invariants | Any real-world quality |
| E1 | Permitted real signatures, clean | Identity comparison quality per encoder and comparator | Stamp behaviour |
| E2 | E1 data + synthetic stamp corruption | Effect of stamps, cleaning, reliability handling, by occlusion level | Physical ink and scanner effects |
| E3 | Owner-made physical set: fictional signatures scanned before and after real stamping | Cleaner behaviour on real ink; paired damage measurement | Writer-level statistics (too few writers) |
| E4 | Customer samples, local, when authorized | Whether the tool behaves sensibly on the real workflow | Error rates, unless labels and enough writers exist |

**Split and leakage rules.** Writers are split into development, validation and final test before anything else. Stamp templates, colours and placement seeds are split the same way. Every derived image inherits the split of its source signature, so a signature never appears on two sides as clean, stamped or cleaned versions. References and queries are different images. Synthetic stamps are composited in a physically plausible way (darkening mixture, blur, scanner noise, JPEG) and with at least one generator that does **not** share the cleaner's mixing model, so the cleaner is not tested on its own assumption. The final test split is evaluated once per frozen configuration.

**Roles kept separate.** Reference enrolment (a few genuine images of test writers) · model or comparator fitting (development writers) · threshold and review-bound selection (validation writers) · final test.

**Comparisons.** Genuine, random impostor and skilled forgery are reported separately. The ablation grid isolates one factor at a time:

- cleaning: clean query (ceiling) · stamped and cleaned · stamped uncleaned *(control)* · cleaner variants;
- encoder: classical descriptor floor · generic · signature-trained;
- comparison: global · local equal · local reliability-aware · both-sides masking;
- masks: oracle · predicted.

**Metrics.** Verification: FAR and FRR at a validation-chosen threshold, EER and AUC as descriptive summaries, with bootstrap intervals resampled by writer. Review: error on retained cases against review rate, and FAR against coverage. Cleaning: ink lost and stamp left (the Missing/Extra idea from P3 applied to ink masks), measured separately inside and outside the overlap; shift of the comparison score between the clean and cleaned versions of the same signature, relative to genuine variation. Failure detection: how well cleaner status predicts actual cleaning failure on E2/E3. Cost: seconds per query and peak memory on the M5.

**Limits to state with every result.** Without forgery labels only random-impostor behaviour is known. With few writers intervals are wide; the customer sample is an inspection set. An identical-image check is a software sanity test, not verification evidence.

**Provisional acceptance criteria.** These are proposals with reasons, not established thresholds.

| Gate | Provisional criterion | Rationale |
|---|---|---|
| Cleaner invariants (M1a) | Exact: no change on no-stamp input, change map inside intervention mask, original bytes intact | Provenance must be exact to be trusted |
| Cleaner usefulness (M1b) | On different-colour cases, most stamp-only pixels removed with ink loss outside the overlap near zero; the numbers are recorded first and regression floors set from them | A preset figure would be a guess |
| Score display (M2b) | An encoder's score is shown in the product only if it beats the classical floor on skilled forgeries with intervals that do not overlap, on writers unseen in fitting | A score that cannot beat edge histograms would mislead |
| Reliability safety (M4) | Reliability-aware comparison must not raise FAR at matched FRR against equal weighting | R3 must not weaken forgery rejection |
| Cleaning benefit (M4) | Cleaned queries recover a clear share of the gap between stamped *(control)* and clean queries; P1's recovery (10 of 15 EER points) is the reference level | Only published comparison available |
| Interactive use (M3) | A query against five references completes in a few seconds on the M5 | Reviewer workflow; to be confirmed with the cousin |

## 11. Research hypotheses and learning directions [R]

**H1.** At matched usable evidence, reliability-aware local comparison of stamped queries yields lower false acceptance at a fixed false-rejection rate than (a) global comparison of the cleaned image and (b) equal-weight local comparison.

**H2.** Cleaning provenance (separability, coverage, change extent) predicts where the comparator errs, so a provenance-based review rule gives lower error on retained cases at the same review rate than a rule based on the score alone.

Classification: building the pipeline is engineering integration; running SigNet, DINOv2 and DetailSemNet under one protocol is replication; H1 and H2 adapt ideas from regional signature matching and occluded re-identification to stamped signatures. Whether either is novel is **not established**; the review found no prior test but was not a systematic search.

| Direction | Failure it addresses | Evidence and transfer limit | Data / labels | Local cost | Smallest justifying experiment | Stop if |
|---|---|---|---|---|---|---|
| D1 Small comparator on frozen features | Raw cosine is not how signature features are meant to be used | proto_hsv: linear model on dissimilarity vectors, trained without skilled forgeries; tested only in-dataset | Genuine signatures of development writers | Seconds to minutes, CPU | Linear comparator against cosine on validation writers | No gain outside interval |
| D2 Clean-licence signature encoder | Permission block on GPDS-derived weights; generic features may miss forgeries | F0: writer classification on ~500 writers transfers; our data differ in collection | CC BY set, writer IDs only | Small CNN; expected under a few hours, to be timed first | Train on 320 writers, test genuine against random impostor on 100 held-out writers, then skilled forgeries where permitted | Below the generic baseline on held-out writers |
| D3 Stamp-aware training (synthetic stamps, cleaned inputs) | Features shift under stamp residue and cleaning | P1 fine-tuned on cleaned images; P3 hybrid data; re-identification occlusion simulation | D2 data + stamp generator | As D2 | Fine-tune with and without stamp augmentation; test on E2 and E3 | No gain on physical set E3 |
| D4 Learned stamp/ink segmentation | Same-colour, black and grayscale stamps | Jo et al.: synthetic-only overlap training transferred for a different class pair | Synthetic overlaps; E3 and possibly StaVer for testing | Small U-Net on crops; hours | Four-class segmenter against the classical cleaner on same-colour fixtures and E3 | No better ink recovery under the stamp than keeping all dark pixels |
| D5 Learned reliability-aware local matching | Hand-set reliability may be suboptimal | DetailSemNet per-token weights; VPM shared-region comparison; neither on stamps | D2/D3 data | Hours to days; may exceed local budget | Only after H1 holds with frozen tokens | H1 fails |
| D6 Joint restoration and verification | Residual stamp interference | P1 only; generative risk | Paired data | Beyond local budget | Not before D4 succeeds | — |
| D7 Self-supervised signature pretraining | Too few labelled writers | SWIS: weak results | Unlabelled signatures | Hours | Only if D2 fails for lack of writers | — |

Order: D1 → D2 → D3, with D4 promoted if the cousin's stamps defeat colour separation. D5 is the ambitious step and depends on H1.

## 12. Compute and resources

- **Machine:** Apple M5, arm64, 16 GB. Python 3.14.0 and `uv` are installed; no project environment exists. Whether current NumPy, OpenCV and PyTorch wheels support Python 3.14 on macOS arm64 must be checked against official install notes before use; a `uv`-managed 3.12 environment is the fallback (owner approval needed for either).
- **Practical locally:** all cleaning; inference for every listed encoder; linear comparators; training a small CNN or small U-Net on tens of thousands of crops. Each training plan starts with a timed one-epoch dry run; nothing is assumed about speed.
- **Needs approval each time:** package installs, dataset or weight downloads, any training run expected to exceed about 30 minutes.
- **Outside current resources:** diffusion or GAN cleaners, training DetailSemNet-scale models from scratch, large hyperparameter sweeps. These stay optional and are not on the critical path.

## 13. Milestones

Effort is in coding sessions (one bounded objective, roughly one to two hours of owner attention each). Ranges assume approvals arrive, wheels install and no gate fails; they are not dates.

| ID | Label | Deliverable | Depends on | Exit check | Effort |
|---|---|---|---|---|---|
| M1a | C | Fixture generator, provenance contract, baseline colour cleaner, `clean` command | Approval to create an environment | Section 14 tests pass; README shows the command and one fixture figure | 1–2 |
| M1b | C | Optical-density cleaner variant, stamp-region hint in the CLI, cleaning metrics, comparison table; protocol sheet for the physical set E3 | M1a | Both cleaners measured on all fixture kinds and, when available, E3; better one chosen per case; failure cases documented | 2 |
| M2a | C | Normalisation, encoder interface, classical floor + DINOv2 (global and patch tokens), reference enrolment, similarity relative to reference spread, coverage, review flags, `compare` command | M1a; approval for PyTorch and weight download | Runs end to end on fixtures; identical image ranks first (sanity only); masks transform with the image; latency and memory recorded | 2–3 |
| M2b | C (gate) | Clean benchmark E1 on permitted data: manifest, writer splits, metrics with intervals; encoder comparison | M2a; dataset decision | Report produced on the final split once; score-display gate decided and recorded | 2–3 |
| M3 | C | Local reviewer UI and case export; feedback form; **prototype v0.1 delivered** | M1b, M2a (M2b decides whether a score is shown) | Cousin completes a small batch locally; every case shows original, cleaned, masks, coverage and flags; nothing leaves the machine | 2–3 |
| M4 | R | Stamped benchmark E2/E3; full ablation grid; calibration of review bounds; H2 test | M2b, M1b | Ablation table with intervals; coverage–error curves; reliability safety gate evaluated | 3–4 |
| M5 | R | Reliability-aware local matching on frozen tokens; H1 test | M4 | H1 supported, rejected or inconclusive, written up with failure cases | 2–3 |
| M6 | R | D1 comparator, then D2 encoder, then D3 stamp-aware training | M2b; training approval | Each step compared with its predecessor on validation, then once on test | 4–6 |
| M7 | S (R if colour cleaning fails on customer stamps) | D4 learned stamp/ink segmenter | M1b, E3 | Beats classical cleaner on same-colour cases and E3 without more ink loss | 3–5 |
| M8 | S | D5 learned occlusion-aware verifier; automatic signature detection; DetailSemNet comparison | H1 supported; extra compute likely | Decided after M5 | open |
| M9 | R | Portfolio release: reproducible demo on permitted data, model and data cards, results, short video | M3 and at least M4 | A fresh clone reproduces the demo and the fixture-level numbers; no private data in the repository | 2 |
| M10 | R | Thesis discussion pack for January 2027 | M4, ideally M5 | Section 18 contents ready | 1 |

First useful delivery (M1a → M3) is about 9–13 sessions under the stated assumptions. M9 can ship a first public version after M3 and be refreshed after each research milestone.

**Stop / change criteria.**

| Observation | Change |
|---|---|
| Customer stamps are mostly black, grey or the pen's colour | Promote M7; until then such cases go to review with the region marked unreliable |
| No permissive encoder passes the score-display gate | Deliver v0.1 without an identity score; pull D2 forward |
| Reliability-aware comparison raises FAR | Keep coverage-based review only; report the negative result |
| Cleaning makes results worse than the uncleaned control *(if approved)* | Report to the owner as evidence bearing on R1; requirement stands unless the owner changes it |
| Mendeley data unusable (quality, consent doubts) | D2 waits for a dataset agreement, possibly via the university |
| Wheels unavailable for Python 3.14 | Use a `uv`-managed 3.12 environment |
| A milestone exceeds twice its upper effort bound | Stop, record why, re-plan with the owner |

## 14. Next coding session — M1a (detailed)

**Objective.** A tested command that takes one image and writes the complete provenance artifact set, using a baseline colour cleaner that is honest about what it cannot see. No encoder, no PyTorch, no datasets, no customer files, no UI, no optical-density unmixing (that is M1b).

**Before writing code.** Ask the owner to approve creating a local environment and installing NumPy, an OpenCV build, Pillow and pytest. Check that wheels exist for the installed Python; if not, propose the 3.12 fallback and wait.

**Allowed files.**

- new: `pyproject.toml`; `src/osv/__init__.py`; `src/osv/datasets/synthetic_fixtures.py`; `src/osv/segmentation/stamp_estimate.py`; `src/osv/preprocessing/stamp_removal.py`; `src/osv/provenance.py`; `src/osv/cli.py`; `tests/test_fixtures.py`, `tests/test_cleaning.py`, `tests/test_provenance.py`; one fixture figure under `docs/assets/`;
- moved: the empty placeholder folders under `src/` into `src/osv/`;
- updated: `README.md` (progress row, exact command), this plan's current state, `docs/SESSION_LOG.md`.

**Interfaces.**

```python
make_fixture(kind: str, seed: int, size=(240, 480)) -> Fixture
#   Fixture: image, pen_mask, stamp_mask, overlap_mask, meta
#   kinds: no_stamp, red_stamp_black_pen, blue_stamp_black_pen, blue_stamp_blue_pen,
#          black_stamp_black_pen, gray_scan, faint_stamp_jpeg

estimate_stamp(image_rgb, stamp_region=None) -> StampEstimate
#   StampEstimate: stamp_mask, ink_mask, ambiguous_mask, separability, reasons

remove_stamp(image_rgb, estimate) -> CleaningResult
#   CleaningResult: cleaned, intervention_mask, change_map, status, reasons, metrics

write_artifacts(original_path, estimate, result, out_dir) -> dict   # the report
```

Command: `python -m osv.cli clean <image> --out <dir> [--stamp-region x,y,w,h]`.

Artifacts in `<dir>`: byte-identical copy of the original, `stamp_mask.png`, `cleaned.png`, `intervention_mask.png`, `change_map.png`, `ink_mask.png`, `ambiguous_mask.png`, `report.json` (`schema_version`, input SHA-256, image size, cleaner name/version/parameters, `status`, `reasons`, separability, fractions of estimated ink that are reliable, ambiguous and changed, optional reviewer region).

**Implementation steps.**

1. Fixture generator: fictional pen strokes (random smooth curves, variable width), paper tone, stamp shapes with text-like texture; composite by darkening mixture, not opaque overlay; optional blur, noise and JPEG; deterministic by seed; exact masks returned.
2. Stamp estimate: chroma-based separation of coloured stamp ink from dark low-chroma ink and paper; an ink mask; the guard that returns `review` and an empty stamp mask when the stamp colour model also explains most of the ink.
3. Removal: stamp-only pixels become the local paper tone; pixels that are both ink-dark and inside the stamp are kept and marked ambiguous; nothing else changes.
4. Provenance writer and CLI.
5. README command and one figure built from a fixture.

**Tests that must exist and pass.**

- `no_stamp`: zero changed pixels, cleaned equals input, status `no_stamp_detected`.
- Red and blue stamp over black pen: measure and record stamp-only removal and pen loss outside the overlap against the fixture masks; pen pixels outside the stamp mask are unchanged exactly; after the first measured run, set regression floors from the recorded values and label them as floors, not quality claims.
- `blue_stamp_blue_pen`: the signature is not deleted (pen loss stays under the recorded floor) and status is `review` with `ink_and_stamp_colour_overlap`.
- `black_stamp_black_pen` and `gray_scan` without a hint: status `no_stamp_detected`, zero change, in a test named as a **known limitation**. With `--stamp-region`: status `review`, reason `stamp_indicated_not_separated`, zero deleted pixels, region marked ambiguous.
- Invariants on every fixture: original bytes and hash intact; all masks share the image size; change map non-zero only inside the intervention mask; intervention mask inside the stamp mask; cleaned differs from the input exactly where the change map is non-zero; same input gives byte-identical output.
- The cleaning functions accept no ground-truth argument (checked by signature).

**Exit.** All tests pass; the measured numbers are written into the session log; the README command is the one actually run.

## 15. Reproducibility and records

- Every experiment is registered in `EXPERIMENTS.md` before it runs, with a config in `experiments/configs/` and a result folder holding the resolved config, git commit, environment listing, dataset manifest and split hashes, seeds, encoder name and weight SHA-256, metrics and per-sample outputs. Negative results are kept.
- Each dataset gets a provenance note (source, terms as read, date, checksum) under `data/public/` before use.
- Artifact formats are versioned (`schema_version`, cleaner and comparator versions) so old case exports stay readable.
- Results quoted in the README link to the result folder and state data, split, references per writer, forgery type and threshold rule.

## 16. Reviewer workflow and customer feedback [C]

Workflow for v0.1: choose or create a writer → add references (quality summary shown) → open a document or crop → draw the crop and, optionally, the stamp region → view original, cleaned, masks and references side by side with coverage and flags → write a note and a reviewer decision → export the case locally.

The screen never shows "genuine" or "forged". Low-evidence cases show the reason before any score.

Feedback worth collecting from the cousin, on fictional or authorized samples: which stamp and pen colours occur; whether the cleaned image is easier or harder to judge than the original; where the tool flagged review and the reviewer disagreed, and the reverse; whether drawing the crop and stamp region is acceptable effort; how many references are typically available; what would make the export useful.

## 17. Portfolio evidence

Per milestone: the problem solved, the decision and its alternatives, a reproducible command, measured results with conditions, and at least one failure case. Planned visuals: pipeline diagram; provenance panel (original, stamp estimate, cleaned, change map); cleaner failure gallery (same-colour, black stamp); ablation table; coverage–error curve; a short demo recording on fictional signatures. The README keeps planned, implemented and validated work in separate columns and makes no claim without a linked result.

## 18. Thesis preparation (for discussion, not assumed)

- **Candidate question.** How should stamp-affected evidence be treated in offline signature verification: does provenance-aware, reliability-weighted local comparison verify partially stamped signatures more safely than cleaning followed by global comparison?
- **Possible scope.** H1 and H2 on public data with synthetic and physical stamps; D2/D3 as the learning component; D4 or D5 as an extension if results justify it.
- **Evidence to bring in January 2027.** Working pipeline and demo; clean benchmark (M2b); stamped ablations (M4) and, if ready, the H1 result (M5); the permission table; failure analysis; this review trail.
- **Questions for the professor.** Is the question thesis-sized and sufficiently new, and what related work should be checked first? Can the university sign dataset agreements (GPDS, MCYT) and clarify CEDAR/BHSig260 terms? Is compute available for D5? What evaluation standard and baselines are expected? May any customer-derived material appear in a thesis (default assumption: no)?

## 19. Session rules and current state

Each coding prompt names one milestone or sub-task, its allowed files, an observable result and its verification. Each session ends with: what changed; commands actually run and their outcomes; what remains unverified; the next smallest action; an entry in `docs/SESSION_LOG.md`; this plan's state updated; README progress updated when a milestone moves. Commits and pushes only when the owner authorizes them in that session.

- **Done:** nine-paper study; first review (M0); second review and this roadmap (11 November 2026).
- **Next:** owner answers the questions in section 3 that are answerable now; then run M1a.
- **Not started:** environment, code, datasets, weights, any measurement.

## References

- [Second review with evidence register](docs/research/SECOND_REVIEW.md) · [initial review](docs/research/INITIAL_REVIEW.md) · [session log](docs/SESSION_LOG.md)
- [Literature matrix](literature/literature_matrix.csv) · [note index](literature/notes/README.md) · [experiment registry](EXPERIMENTS.md)
- [Starting instructions](docs/GETTING_STARTED.md) · [coordination workflow](docs/CLAUDE_WORKFLOW.md)
- Background only: [November 10 design](docs/superpowers/specs/2026-11-10-customer-verification-design.md), [earlier detailed plan](docs/superpowers/plans/2026-11-10-customer-verification.md)
