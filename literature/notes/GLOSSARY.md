# Signature verification — working glossary

Compiled from P1, F0 and P5 discussions; consult each paper note for source pages.

| Term | Meaning |
|---|---|
| Offline verification | Check a claimed identity using a static signature image; pen dynamics are not directly available. |
| Reference / query | Previously collected genuine example / signature currently being checked. |
| Genuine variation | Natural differences between signatures by the same writer. |
| Random / skilled forgery | Another person's ordinary signature used as an impostor / deliberate imitation of the claimed writer. |
| Feature vector / embedding | Learned numerical representation of an image, not automatically a probability. |
| Writer-independent features | Shared features evaluated on writers absent from feature training. |
| Writer-dependent classifier | Decision component trained for a particular enrolled writer, such as F0's SVM. |
| Siamese network | Shared feature extractor processes both images; training teaches useful pair comparison. |
| Spatial transformer (STN) | Learns geometric resampling/alignment; does not reconstruct hidden strokes. |
| Focal loss | Downweights easy training examples; α weights classes and γ controls focusing. Not inference acceleration. |
| False acceptance / rejection | Accept an impostor comparison / reject a genuine one. |
| Threshold / EER | Score cutoff / operating point where false acceptance and rejection rates are equal. |
| Domain shift | Data conditions change, potentially changing model performance: scans, scripts, stamps or restoration artifacts. |
| Ablation | Controlled comparison that adds/removes a component to measure its contribution. |

**Pipeline:** reference and query → preprocessing → shared features → pairwise comparison (or personal classifier) → score → threshold → decision. Cleaning and alignment are optional experimental components whose effect must be measured.

## Additional terms from P1, F0 and P2

| Term | Meaning |
|---|---|
| CNN / convolution | Learned filters detect patterns across an image; deeper layers combine them into more complex features. |
| Pooling / downsampling | Reduce spatial size to make processing manageable; some operations can discard fine details. |
| Feature channel | One learned feature map; different channels can respond to different patterns. |
| SPD-Conv | Space-to-depth rearranges neighbouring spatial values into channels, then a stride-one convolution combines them. The rearrangement preserves values; later processing can lose information. Intended to retain fine details during downsampling. |
| SCConv | Spatial and Channel Reconstruction Convolution: reorganizes feature information to reduce redundancy. Not a stamp detector. |
| SRU / CRU | SCConv's Spatial Reconstruction Unit separates/recombines spatial information; its Channel Reconstruction Unit splits, transforms and fuses channel information. |
| Global / local features | Overall signature layout and relationships / fine stroke shapes, connections and small differences. These roles are complementary, not exclusive to one architecture. |
| Concatenation | Join feature vectors into a longer vector, as in P2's global/local fusion. |
| Self-attention / ViT | Self-attention lets image-region representations use information from other regions; a Vision Transformer uses patch representations and position information. Different from an STN. |
| Affine transform / interpolation | Geometric shifting, rotation, scaling or shear / estimating values between pixels when resampling. An STN learns transformation parameters. |
| SVM / RBF kernel | Support vector machine: learns a decision boundary. An RBF kernel permits a more flexible boundary than a linear SVM. |
| Cosine similarity / Euclidean distance | Compare vector directions / measure distance between vector points. Scores depend on representation and normalization; neither is automatically a probability. |
| CycleGAN / unpaired translation | Learn mappings between two image collections, such as stamped and clean signatures, without exact matching before/after examples. |
| Adversarial loss / discriminator | Training feedback that encourages generated images to resemble a target collection / the network judging that resemblance. Visual realism does not guarantee identity preservation. |
| Cycle consistency | Encourage a round trip, F(G(x))≈x and G(F(y))≈y. It does not prove faithful recovery of hidden strokes. |
| Architecture / weights | Network structure / learned numerical parameters. VGG-16, ResNet-50 and HTCSigNet are architectures; pretrained weights are prior learning. |
| Pretraining / fine-tuning | Learn on an initial dataset / adapt those learned parameters on another dataset. |
| Residual connection | An additional path carrying information around processing layers. |
| Multitask learning / loss | Learn shared features through several objectives / a penalty measuring training mistakes. F0 and P2 combine writer and forgery objectives. |
| α / γ / λ | Context-dependent settings: in focal loss α weights classes and γ controls focusing; λ balances loss terms. They are not universal constants or interchangeable across papers. |
| Enrollment | Collect genuine references and, for WD methods, fit a personal classifier. Distinct from learning the shared feature extractor. |
| Writer-disjoint split | Different writers in feature-learning and final evaluation groups; protects assessment on unseen identities. |
| Validation / test set | Data used to choose settings / held-out data used to assess the chosen system. Selecting thresholds on final test outcomes gives an evaluation result, not a locked deployment policy. |
| Global / personal threshold | One score cutoff across writers / a separate cutoff for each writer. Optimal personal EER thresholds do not establish how to choose them in practice. |
| ROC / AUC / accuracy | Error tradeoff across thresholds / area under the ROC curve / fraction correct at a particular decision rule and class mixture. Distinct from EER. |
| Augmentation / regularization / overfitting | Vary training inputs / discourage overly narrow fitting / learn patterns that fail to generalize beyond training conditions. |
| Batch normalization / ReLU | Normalize internal activations to support training / nonlinear operation that replaces negative inputs with zero. |
| Otsu preprocessing | Choose an image-intensity threshold to separate foreground and background; does not inherently separate overlapping signature and stamp ink. |
| Occlusion / restoration artifact | Signature evidence obscured by interference / changes introduced when cleaning or reconstructing an image. Cleaner appearance does not imply more reliable verification. |

**Coverage:** P1, F0, P5 and P2. Reviewed by the reader on 19 October; SPD-Conv/SCConv and supporting terms added at their explicit request. This is a compact reference, not a requirement to memorize every implementation detail.

## Stamp restoration and evaluation — P3/P4

| Term | Meaning |
|---|---|
| Conditional diffusion / DDPM | Learn denoising guided by an input, here a stamped document. Artificial diffusion noise differs from stamp interference. |
| DDIM sampling / denoising steps | A sampling procedure that produces a restoration through repeated updates; fewer steps can change quality and runtime. |
| U-Net / attention U-Net | Encoder/decoder with skip connections to retain detail / a variant with learned feature weighting. A U-Net can be standalone or the denoiser inside diffusion. |
| Generator / discriminator | Network producing an output / network judging generated versus real-domain examples during adversarial training. |
| PatchGAN | Discriminator judging local image patches; local realism does not establish correct identity evidence. |
| Decomposition / recombination | Separate stamp and document components / combine them again to test reconstruction. Correct reconstruction does not prove correct component assignment. |
| Extraction / removal / restoration | Isolate stamp content / suppress it / reconstruct affected document content. These are different objectives. |
| Stamp mask / visibility map | Estimate stamp-contaminated or usable regions. Prediction errors can suppress genuine handwriting; masks do not recover hidden strokes. |
| Paired ground truth | A corresponding known clean target for each stamped input. Enables reference-based restoration evaluation. |
| Registration | Align corresponding images before pixel comparisons; scan misalignment can resemble content damage. |
| Physical versus overlay data | Ink actually applied to printed paper and scanned / digital stamps composited onto backgrounds, even if backgrounds are real scans. |
| Hybrid training | Mix real and synthetic examples to expose the model to both distributions; ratios and total training budget matter. |
| Catastrophic forgetting | Adapting to new data damages previously learned capabilities; mixed-domain exposure may mitigate it. |
| EA-RMSE | Error-Area Root Mean Square Error: thresholded, area-weighted image differences separated into Missing and Extra. Requires aligned clean targets. |
| Missing / Extra | Pixel-difference components interpreted as erased/altered content / unwanted remaining content. Not direct semantic measures of lost identity or stamp ownership. Lower is better. |
| τ (tau) | Threshold deciding which pixel differences count as errors; P3's default is 5 in its stated colour-processing setup. Different from a verifier's decision threshold. |
| PSNR / SSIM / RMSE | Pixel-error-derived quality / structural similarity / root mean square pixel error. Higher PSNR/SSIM and lower RMSE are preferred; none proves identity preservation. |
| Hallucination / content leakage | Generate plausible unsupported strokes / place genuine document content into the extracted stamp component. Both can undermine verification. |
| Perceptual rating / MOS | Human quality judgment / mean opinion score across raters. Restoration appearance is distinct from authenticity. |
| Spearman correlation | Agreement between rankings; correlation with human ratings does not mean perfect agreement or forensic validity. |
| Latency / quality tradeoff | Time per output versus restoration quality; measure under matched hardware, image size and processing conditions. |

**P3/P4 principle:** evaluate remaining interference, preserved content and downstream signature decisions separately. Neither a realistic image nor successful cycle reconstruction proves faithful signature recovery.

## Regional verification — P6

| Term | Meaning |
|---|---|
| Deep metric learning | Jointly learn features and a comparison rule so genuine pairs receive higher similarity than impostor/forgery pairs. |
| Region-based verification | Compare corresponding local windows rather than relying only on a whole-image representation. |
| Sliding window / overlap / stride | Move a crop across the image / shared pixels between crops / displacement between crop positions. Overlapping scores are correlated. |
| Score-level fusion | Combine comparison scores, for example by averaging. Different from concatenating feature vectors before comparison. |
| Multi-reference aggregation | Combine a query's scores against several genuine references; distinct from combining regions within one pair. |
| DenseNet / DenseBlock | Architecture reusing earlier feature maps through dense connections, promoting feature reuse. |
| MSDN | Mutual Signature DenseNet: P6's shared pair-processing network with interactions between the two signatures. |
| Squeeze-and-Excitation (SE) | Channel weighting based on summarized feature information; P6 uses paired-input differences to inform weights. Not an explicit occlusion mask. |
| Multi-level features | Combine information from different network depths; extra detail can also increase overfitting. |
| Global average pooling | Summarize each channel by its average over spatial positions. |
| Spatial pyramid pooling (SPP) | Pool into several grid resolutions to produce a fixed-size representation from varying spatial input sizes. |
| Absolute feature difference | Elementwise magnitude of differences between paired feature vectors, used by P6's learned comparator. |
| Sigmoid / binary cross-entropy | Map a value into 0–1 / penalize incorrect binary predictions during training. A sigmoid score is not necessarily calibrated confidence. |
| Correspondence / alignment | Ensure compared regions refer to sufficiently matching parts of the signatures; normalization does not guarantee perfect correspondence. |
| Visibility weighting | Proposed fusion that reduces influence of regions with unreliable evidence; should depend on usable handwriting, not only unoccluded area. |
| Oracle versus predicted mask | Known synthetic ground-truth occlusion map / model-estimated map. Comparing them reveals the cost of mask errors. |
| Abstention / coverage | Defer uncertain cases to human review / proportion of cases on which the system makes a decision. Evaluate error rates alongside coverage. |

**P6 distinction:** its WD result uses personal thresholds with a shared network, unlike F0/P2's personal SVM classifiers. Neither optimal per-writer EER nor extra references automatically establishes a practical deployment policy.

## Covariance geometry and meta-learning — P7

| Term | Meaning |
|---|---|
| Covariance / variance | Joint variation of two features / variation of one feature. A covariance matrix summarizes these relationships across samples such as foreground pixels. |
| SPD matrix | Symmetric positive-definite matrix: a constrained matrix with positive quadratic form for every nonzero vector. Different meaning of SPD from P2's space-to-depth. |
| Derivative / gradient magnitude and direction | Measures of image-intensity change / its strength and orientation. These are static image cues, not pen motion measurements. |
| Thinning | Reduce foreground strokes toward a skeleton; may discard thickness or other evidence. |
| Matrix regularization / conditioning | Add stabilizing terms or constrain a matrix / assess sensitivity of computations. Few visible pixels can make covariance estimation unreliable. |
| Riemannian manifold / geodesic | A structured space with a geometry for distances and optimization / its geometry-respecting path. |
| AIM / LEM | Affine-Invariant Metric / Log-Euclidean Metric: different ways to compare SPD matrices. Their invariance properties do not automatically confer image-occlusion robustness. |
| Principal submatrix | Select matching rows and columns of a matrix. In P7, these describe feature groups, not physical signature windows. |
| ERP | Embedded Riemannian Pyramid: vector of distances between corresponding covariance submatrices of a signature pair. |
| Mahalanobis metric | Learned quadratic-form comparison that weights feature directions and interactions; P7 learns an SPD parameter for ERP dissimilarity. |
| Meta-learning / learned optimizer | Learn aspects of how learning proceeds / predict optimization updates such as direction and step size. Does not inherently mean personal retraining at enrollment. |
| Contrastive objective | Encourage genuine-pair and forgery-pair scores to lie on appropriate sides of training margins. Different from focal-loss reweighting. |
| CSM | Cluster Separability Measure: ranks how separated genuine/forgery descriptor groups are; P7 uses it to select configurations. |
| Minimum dissimilarity fusion | Use the closest reference match as a query's score; distinct from averaging references or averaging spatial regions. |
| 5-by-2 cross-validation | Five repetitions of two-way splits, swapping learning/test portions; writer partitions matter more than the fold count alone. |
| Cross-lingual evaluation | Train using one dataset/script and test on another; distinguish from separately training a model in every script. |

**P7 principle:** small final descriptor dimension does not imply a cheap full pipeline, and exceptional EER values remain tied to reference aggregation, threshold calculation and individual experimental conditions.

## Correlation-based learning — P8

| Term | Meaning |
|---|---|
| CCA / canonical correlation | Find linear projections of two variable groups whose projected values are correlated; P8 uses correlation objectives to shape neural representations. |
| Correlation versus similarity | Statistical co-variation is not identical to equal vectors, pixel resemblance or a calibrated match probability. |
| Canonical direction / learned subspace | Projection used to expose correlated structure / representation coordinates in which comparisons are made. |
| SVD | Singular Value Decomposition: matrix factorization used in solving the paper's correlation objective. |
| G–G / G–S / G–R / S–S | Genuine–genuine / genuine–skilled / genuine–random / skilled–skilled training pairs. In P8, G–G and S–S examples target the same writer; encouraging S–S correlation does not authorize accepting forgeries. |
| Training tuple | Several related examples processed with shared weights to construct multiple objectives; P8 uses five input signatures. |
| Auxiliary correlation objective | Extra training feedback shaping features; distinct from the final SVM decision rule. |
| Clean–occluded consistency | Proposed training pressure for clean and obscured handwriting to remain appropriately comparable; must retain genuine/forgery separation. |
| Representation collapse | Different inputs become insufficiently distinguishable; matching nuisance variation must not erase identity differences. |
| Shortcut learning | Exploit accidental cues such as stamp templates correlated with labels rather than handwriting identity. |
| Matched augmentation control | Give compared methods the same stamped examples so an apparent gain can be attributed to the objective rather than extra data exposure. |

**P8 clarification:** α=0.05 balances the paper's grouped objectives; it is not a universal CCA weight or the same parameter as focal-loss α. Pair-based feature training can still lead to writer-dependent personal-SVM verification.
