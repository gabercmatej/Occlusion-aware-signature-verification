# Starter Reading Set

The `papers/` folder intentionally contains a small set, not a PDF dump. Read in this order and update the literature matrix and note file after each paper.

## Downloaded starter papers

1. **Hafemann, Sabourin, and Oliveira (2017), _Learning features for offline handwritten signature verification using deep convolutional neural networks_.** Read this for the foundational writer-independent feature-learning formulation, person-disjoint validation, skilled forgeries, and benchmark protocols.
2. **Xiao and Wu (2025), _Learning features for offline handwritten signature verification using spatial transformer network_.** Read this for a recent two-stage Siamese-style system, attention to discriminative regions, and difficult skilled-forgery cases.

## Very relevant, linked but not downloaded automatically

These papers are already recorded in the literature matrix. Use the open publisher page, your university library, or an author manuscript as indicated.

- **Lee et al. (2026), _Stamp removal from real documents: A diffusion-based approach with EA-RMSE evaluation_:** read this first for the real-versus-synthetic stamp gap and for an evaluation metric that separates residual stamp content from destroyed background content. The article is open access, but the publisher required an interactive download challenge during setup. DOI: <https://doi.org/10.1016/j.array.2026.100855>
- **HTCSigNet (2025):** essential follow-up for local CNN features plus global transformer features. DOI: <https://doi.org/10.1016/j.patcog.2024.111146>
- **SERGAN (2024):** important GAN baseline for simultaneous stamp extraction and removal. DOI: <https://doi.org/10.1016/j.dsp.2023.104358>
- **Enhanced diffusion-based stamp removal (2024):** useful predecessor to the 2026 real-document paper. DOI: <https://doi.org/10.1016/j.compeleceng.2024.109738>
- **Metric meta-learning with Riemannian embeddings (2025):** advanced follow-up after the Siamese and foundational papers. DOI: <https://doi.org/10.1016/j.eswa.2024.125470>

## Reading discipline

For each paper, record:

- the exact task and evaluation protocol;
- whether the split is writer-disjoint;
- whether skilled forgeries and stamp occlusion are present;
- what input information preprocessing removes;
- the most important weakness for this thesis;
- one experiment the paper enables.

Do not compare headline EER values until you have checked that datasets, numbers of reference signatures, forgery types, and split protocols are comparable.
