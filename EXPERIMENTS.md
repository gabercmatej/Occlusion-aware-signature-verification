# Experiment Registry

> Current phase (16 October 2026): literature understanding and first professor proposal. Follow [PLAN.md](PLAN.md). All core PDFs are local; implementation and the experiments below remain proposed until supervisor approval. The primary comparison is direct versus visibility-aware verification; restoration and other advanced methods are conditional extensions. No study results are claimed.

Register an experiment before running it. Give each experiment a stable ID and keep the configuration, outputs, and interpretation linked.

## Template

### E000 — Short title

- **Question:** What precise research question does this answer?
- **Hypothesis:** What result is expected, and why?
- **Data:** Dataset, writers, sample counts, and provenance.
- **Split:** Person-disjoint train/validation/test definition.
- **Inputs:** RGB, grayscale, binary, masks, references, and resolution.
- **Method:** Baseline/model and version.
- **Controls:** What isolates the variable being tested?
- **Metrics:** FAR, FRR, EER, ROC/AUC, TAR at fixed FAR, calibration, or restoration metrics.
- **Config:** `experiments/configs/E000-...`
- **Results:** `experiments/results/E000-...`
- **Figures:** `experiments/figures/E000-...`
- **Decision rule:** What outcome would support or reject the hypothesis?
- **Status:** proposed / running / complete / invalidated.
- **Notes:** Confounders, failures, and follow-up work.

## Proposed experiments

### E001 — Synthetic occlusion generator sanity check

- **Question:** Can synthetic stamps create controlled, measurable signature occlusions without changing pixels outside the mask?
- **Hypothesis:** A simple compositing pipeline can generate clean input, stamp layer, occluded image, stamp mask, signature mask, and visible-signature mask with exact pixel-level ground truth.
- **Data:** A small public signature subset plus procedurally generated circular, elliptical, and rectangular stamps.
- **Controls:** Preserve the unoccluded signature and exact alpha mask; vary color, opacity, angle, blur, print degradation, and overlap percentage one factor at a time.
- **Metrics:** Mask consistency, achieved occlusion percentage, and visual inspection.
- **Status:** proposed.

### E002 — Verification degradation versus occlusion

- **Question:** How quickly does a clean-signature baseline degrade as the visible signature area decreases?
- **Hypothesis:** EER and FRR will increase nonlinearly with occlusion; confidence will become miscalibrated before accuracy collapses.
- **Data:** Public signature dataset with synthetic stamps at controlled severity bands.
- **Controls:** Person-disjoint split; identical verifier and references across severity bands.
- **Metrics:** FAR, FRR, EER, AUC, TAR at fixed FAR, calibration error, and risk-coverage.
- **Status:** proposed.

### E003 — Restore versus mask versus direct verification

- **Question:** Which paradigm provides the safest verification under stamp occlusion?
- **Hypothesis:** Restoration may improve appearance while mask-based or direct verification better preserves security-relevant evidence.
- **Data:** Public signatures with synthetic occlusion first; physically stamped data only after governance and annotation are ready.
- **Controls:** Shared verifier capacity, reference sets, split, occlusion bands, and compute budget.
- **Metrics:** Verification metrics plus residual-stamp error, lost-signature error, calibration, and abstention coverage.
- **Status:** proposed.
