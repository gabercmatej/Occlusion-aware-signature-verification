# Research Code

Move reusable logic out of notebooks into focused modules:

- `preprocessing/`: normalization and classical image baselines.
- `segmentation/`: stamp, signature, and ambiguous-region masks.
- `verification/`: writer-dependent and writer-independent verifiers.
- `metric_learning/`: pair and reference-set objectives and samplers.
- `uncertainty/`: calibration, abstention, and out-of-distribution methods.
- `datasets/`: dataset readers, manifests, and person-disjoint splits.
- `evaluation/`: verification, restoration, calibration, and risk-coverage metrics.

Notebooks should call this code rather than becoming the only implementation.
