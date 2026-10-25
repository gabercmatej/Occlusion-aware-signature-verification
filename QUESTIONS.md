# Research Questions

> Current phase (16 October 2026): literature understanding and first professor proposal. Follow [PLAN.md](PLAN.md). All core PDFs are local; implementation and the experiments below remain proposed until supervisor approval. The primary comparison is direct versus visibility-aware verification; restoration and other advanced methods are conditional extensions. No study results are claimed.

Add questions freely. Do not force early answers; link answers to evidence, experiments, or literature notes later.

## Stamp and signature separation

- Can stamp and handwriting be separated when both have the same color?
- Is stamp removal the correct formulation, or should the system estimate signature, stamp, and ambiguous masks?
- How much genuine signature information is destroyed by each removal method?
- Can segmentation preserve forensic stroke characteristics better than generative restoration?
- Can synthetic stamp overlap generalize to real physically stamped documents?
- How much real stamped data is necessary?
- Can uncertainty identify pixels whose ownership cannot be determined reliably?

## Signature verification

- How many reference signatures are necessary per signer?
- How should multiple references be aggregated?
- How robust are writer-independent models to large natural intra-writer variation?
- Which local stroke features are most useful for detecting skilled forgery?
- Does binarization destroy useful forensic evidence?
- Are RGB, grayscale, or raw-resolution inputs better than traditional preprocessing?
- Do hybrid CNN/transformer representations outperform pure CNN metric learning on the target domain?

## Combined problem

- How does stamp occlusion percentage affect verification performance?
- Does removing a stamp improve verification, or do restoration artifacts make it worse?
- Is direct occlusion-aware verification better than reconstructing the hidden signature?
- Can a verifier ignore stamp-contaminated regions without explicit removal?
- Does joint segmentation and verification outperform a two-stage pipeline?
- How does performance change as the genuinely visible signature area decreases?
- Can a partially occluded signature be verified from visible evidence alone?

## Reliability and deployment

- How should confidence be calibrated across writers and datasets?
- At what evidence threshold should the model abstain?
- How should TAR at a very low FAR be reported for the intended financial setting?
- What out-of-distribution signals identify unsupported stamp types, scanners, scripts, or document backgrounds?
- Which failure cases require mandatory human review regardless of model confidence?
