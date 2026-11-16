"""Encoder contract for the benchmark, and the classical descriptor as its first implementation (M2b).

An *encoder* here is whatever turns two normalised signature images into one distance. The
contract is deliberately pairwise and small:

* :class:`EncoderSpec` — what the encoder **declares**: name, version, configuration, the input
  it reads, the canvas it needs, how the descriptor is obtained, what the distance means, when a
  comparison is undefined, and what excluding a region does and does not guarantee.
* :meth:`Encoder.compare` — one query against one reference, with an optional exclusion mask on
  the query's canvas. It returns a :class:`PairResult`: a distance (lower is more similar) or an
  explicit undefined status, never a silent zero.
* :meth:`Encoder.invariance_check` — the hook that measures the declared masking claim instead of
  trusting it.

Why ``compare`` and not ``embed`` + ``distance``: the classical descriptor's support depends on
*both* images (padding of either is excluded), so a per-image embedding does not exist for it.
An encoder that does have per-image embeddings (a frozen transformer, for example) caches them
inside ``compare``.

Masking claims are a closed vocabulary so that a report cannot overstate them:

* ``independent_of_excluded_canvas_pixels`` — changing canvas pixels inside the exclusion mask
  cannot change a defined distance beyond ``tolerance``. The classical descriptor earns this
  through its finite, explicit footprint. A transformer earns it only if excluded patches never
  enter the token sequence (or are replaced before the first layer) *and* the invariance check
  passes; it is not required to have a finite receptive field, it is required to pass the test.
* ``context_contaminated`` — excluded content is down-weighted or dropped after encoding, so
  retained features can still depend on it. The check then *measures* the shift; it cannot fail.
* ``no_masking`` — the exclusion mask is ignored.

The scope string says where the claim stops (for every encoder here: at the canvas; the cleaner,
the paper-luma estimate and resampling happen before it).
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from osv.preprocessing.normalise import NORMALISER_NAME, NORMALISER_VERSION, NormalisedImage, NormaliseParams
from osv.verification.descriptor import DescriptorParams, GridOrientationDescriptor
from osv.verification.pairs import MIN_RETAINED_EVIDENCE_SHARE, PAIR_SUPPORT_RULE, compare_pair

CONTRACT_VERSION = "1.0"
MASKING_CLAIMS = ("independent_of_excluded_canvas_pixels", "context_contaminated", "no_masking")
DISTANCE_DIRECTION = "lower_is_more_similar"


@dataclass(frozen=True)
class MaskingSpec:
    claim: str  # one of MASKING_CLAIMS
    support_semantics: str  # what "excluded" does to the computation
    scope: str  # where the claim stops
    tolerance: float | None = None  # largest allowed change of a defined distance under the claim

    def __post_init__(self):
        if self.claim not in MASKING_CLAIMS:
            raise ValueError(f"masking claim must be one of {MASKING_CLAIMS}")
        if self.claim == "independent_of_excluded_canvas_pixels" and self.tolerance is None:
            raise ValueError("an independence claim needs a stated tolerance")


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    version: str
    config: dict
    input_representation: str
    canvas: dict
    descriptor_extraction: str
    distance: dict  # name, range, direction, symmetric
    undefined_behaviour: str
    masking: MaskingSpec
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        if not self.name or not self.version:
            raise ValueError("an encoder declares a name and a version")
        if self.distance.get("direction") != DISTANCE_DIRECTION:
            raise ValueError(f"the benchmark scores distances: direction must be {DISTANCE_DIRECTION!r}")

    def record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PairResult:
    status: str  # "defined" or an explicit reason
    distance: float | None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if (self.status == "defined") != (self.distance is not None):
            raise ValueError("a distance exists exactly when the status is 'defined'")


@dataclass(frozen=True)
class InvarianceReport:
    claim: str
    tolerance: float | None
    trials: int
    compared_trials: int  # trials in which both the baseline and the altered comparison were defined
    max_abs_distance_change: float | None
    status_changes: int  # trials whose defined/undefined status differs from the baseline
    holds: bool | None  # None when the claim makes no promise to test (measurement only)

    def record(self) -> dict:
        return asdict(self)


class Encoder(abc.ABC):
    """Base class: implement :attr:`spec`, :attr:`normalise_params` and :meth:`compare`."""

    spec: EncoderSpec
    normalise_params: NormaliseParams

    @abc.abstractmethod
    def compare(self, query: NormalisedImage, reference: NormalisedImage,
                query_exclusion: np.ndarray | None = None) -> PairResult:
        """Distance between a query and a reference; ``query_exclusion`` is a bool canvas mask."""

    def invariance_check(self, query: NormalisedImage, reference: NormalisedImage, exclusion: np.ndarray,
                         rng: np.random.Generator, trials: int = 5) -> InvarianceReport:
        """Replace the query's canvas content inside ``exclusion`` and measure the effect.

        Each trial overwrites darkness and RGB inside the mask with random values, then compares
        again under the same mask. For an independence claim the result ``holds`` when no defined
        distance moved by more than the declared tolerance; a status change is reported
        separately because an encoder may refuse more often when excluded content grows (it must
        say so in its scope). For other claims the report is a measurement and ``holds`` is None.
        """
        if exclusion.dtype != bool or exclusion.shape != query.darkness.shape:
            raise ValueError("exclusion must be a boolean mask of the query canvas")
        base = self.compare(query, reference, exclusion)
        changes, status_changes = [], 0
        for _ in range(trials):
            darkness = query.darkness.copy()
            darkness[exclusion] = rng.random(int(exclusion.sum()), dtype=np.float32)
            rgb = query.rgb.copy()
            rgb[exclusion] = rng.integers(0, 256, (int(exclusion.sum()), 3), dtype=np.uint8)
            altered = self.compare(replace(query, darkness=darkness, rgb=rgb), reference, exclusion)
            if (altered.distance is None) != (base.distance is None):
                status_changes += 1
            elif altered.distance is not None:
                changes.append(abs(altered.distance - base.distance))
        claim, tol = self.spec.masking.claim, self.spec.masking.tolerance
        worst = max(changes) if changes else None
        holds = None
        if claim == "independent_of_excluded_canvas_pixels":
            holds = bool(changes) and worst <= tol
        return InvarianceReport(claim=claim, tolerance=tol, trials=trials, compared_trials=len(changes),
                                max_abs_distance_change=worst, status_changes=status_changes, holds=holds)


class ClassicalGridEncoder(Encoder):
    """The M2a grid orientation histogram behind the contract; behaviour is that of ``compare_pair``."""

    def __init__(self, descriptor_params: DescriptorParams = DescriptorParams(),
                 normalise_params: NormaliseParams = NormaliseParams()):
        self.descriptor = GridOrientationDescriptor(descriptor_params)
        self.normalise_params = normalise_params
        rec = self.descriptor.record()
        self.spec = EncoderSpec(
            name=self.descriptor.name,
            version=self.descriptor.version,
            config={"descriptor": rec["parameters"], "min_retained_evidence_share": MIN_RETAINED_EVIDENCE_SHARE,
                    "thresholds_status": "provisional; chosen on fictional synthetic images, not tuned on this benchmark"},
            input_representation="ink darkness clip(1 - Y / Y_paper, 0, 1) of the normalised canvas (float32, colour agnostic)",
            canvas={"normaliser": NORMALISER_NAME, "normaliser_version": NORMALISER_VERSION,
                    "parameters": normalise_params.as_dict(),
                    "crop_convention": "content box of dark pixels of the image itself, uniform scale, real paper kept, "
                                       "window parts outside the source are padding and excluded"},
            descriptor_extraction="grid of unsigned gradient-orientation histograms (Gaussian pre-blur, 3x3 Sobel), "
                                  "magnitude votes, normalised by the total; computed per pair on the common support",
            distance={"name": "total variation, 0.5 * L1", "range": [0.0, 1.0], "direction": DISTANCE_DIRECTION,
                      "symmetric": "yes for a fixed support; the exclusion mask belongs to the query"},
            undefined_behaviour=("no distance when either side has too little edge evidence on the pair support, or "
                                 f"keeps less than {MIN_RETAINED_EVIDENCE_SHARE} of its own edge evidence; the status "
                                 "names the side and the reason"),
            masking=MaskingSpec(
                claim="independent_of_excluded_canvas_pixels",
                support_semantics=PAIR_SUPPORT_RULE + "; " + rec["exclusion"]["rule"],
                scope=("canvas level, for the value of a defined distance. Whether a distance is defined can depend "
                       "on excluded content (the retained-evidence share reads the whole image, so excluded residue "
                       "can only add refusals). Not covered: cleaner decisions, whole-image paper luma, resampling."),
                tolerance=1e-9),
        )

    def compare(self, query: NormalisedImage, reference: NormalisedImage,
                query_exclusion: np.ndarray | None = None) -> PairResult:
        row = compare_pair(self.descriptor, query, reference, query_exclusion)
        details = {k: v for k, v in row.items() if k not in ("status", "distance")}
        return PairResult(status=row["status"], distance=row["distance"], details=details)
