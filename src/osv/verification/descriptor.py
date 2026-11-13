"""Classical descriptor and distance for the M2a engineering baseline.

Descriptor: a grid of gradient-orientation histograms (HOG-like) on the normalised ink-darkness
canvas. Each pixel votes its gradient magnitude into one of ``bins`` unsigned orientations
(linear interpolation between neighbouring bins) inside a fixed grid cell; the whole vector is
then divided by its total, so it describes *where* stroke edges are and *which directions* they
run, independent of ink colour and total ink amount.

Why this one: it needs only NumPy/OpenCV, every number can be traced to pixels, the dependency
footprint of every vote is finite and explicit, and gradient histograms are a standard classical
baseline family for offline signatures. It does not model
stroke order, pressure, local deformation or the natural variation of a writer, and the fixed
grid makes it sensitive to layout differences that normalisation does not remove.

Distance: total-variation distance ``0.5 * sum |a - b|`` in [0, 1]; 0 for identical vectors.

Exclusion semantics (0.2.0). A vote at canvas pixel ``q`` is computed from a Gaussian pre-blur
(explicit kernel radius ``ceil(4 * sigma)``) followed by a 3x3 Sobel, both with reflect-101
borders, so it reads darkness only within Chebyshev distance ``footprint_radius`` of ``q`` (with
reflection at the canvas border). With an exclusion mask, every vote whose footprint contains an
excluded pixel is dropped; the remaining positions are the *descriptor support*. Excluded
content is never replaced by a constant, so no artificial boundary is ever counted. Guarantee:
two canvases that agree outside the mask give the same masked descriptor. In 0.1.0 only votes
inside the mask were dropped and blurred/gradient contributions leaked outward (up to 5 px).
The guarantee is about the canvas only; see :mod:`osv.verification.compare` for what happens
before the canvas exists. Unmasked output is identical to 0.1.0.

Edge evidence (0.3.0, M2a.2). A vote is *strong* when its magnitude reaches
``max(strong_gradient_floor, noise_factor * paper noise)``, where paper noise is the median
magnitude over support positions whose darkness is below ``paper_darkness`` (paper, not ink; the
first draft used the whole support, which put the threshold above stroke edges on dense ink).
Only support positions are read, so excluded pixels never influence it. ``edge_evidence`` is the summed
magnitude of strong retained votes. The descriptor is undefined (``None``, status
``insufficient_edge_evidence``) when ``edge_evidence < min_edge_evidence``, which is what one
short stroke (20 canvas px, 2 px wide, darkness 0.8) produces; blank paper noise and paper
shading stay below the floor. The vector itself is unchanged: all retained votes, normalised.
Canvas padding is not handled here: callers pass it as excluded (see
:mod:`osv.verification.pairs`). :meth:`distance` refuses undefined inputs rather than returning a
number. All three evidence constants are provisional, chosen from fictional synthetic images.

The descriptor is deliberately separate from reference aggregation
(:mod:`osv.verification.compare`), so a learned encoder can later replace it behind the same
``extract`` / ``distance`` interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

DESCRIPTOR_NAME = "grid_orientation_histogram"
DESCRIPTOR_VERSION = "0.3.0"
SOBEL_RADIUS = 1  # 3x3 kernel
BLUR_TRUNCATE = 4.0  # kernel radius = ceil(BLUR_TRUNCATE * sigma); OpenCV's own float32 default
BORDER = cv2.BORDER_REFLECT_101
EXCLUSION_RULE = ("drop every gradient vote whose Gaussian+Sobel footprint (Chebyshev radius "
                  "footprint_radius_px, reflect-101 borders) contains an excluded canvas pixel; "
                  "excluded content is not filled or counted")


class InsufficientEvidenceError(ValueError):
    """A distance was requested for an undefined descriptor (no usable edge evidence)."""


@dataclass(frozen=True)
class DescriptorParams:
    cell_size: int = 32  # pixels on the normalised canvas
    bins: int = 8  # unsigned orientations over [0, pi)
    pre_blur_sigma: float = 1.0
    strong_gradient_floor: float = 0.10  # provisional; blank-paper noise peaks near 0.011
    noise_factor: float = 10.0  # provisional; strong = above 10x the paper's median magnitude
    paper_darkness: float = 0.10  # provisional; positions this light count as paper for the noise estimate
    min_edge_evidence: float = 175.0  # provisional; one 20 px stroke (see module docstring)


@dataclass(frozen=True)
class DescriptorResult:
    vector: np.ndarray | None
    status: str  # "defined" | "insufficient_edge_evidence"
    support: np.ndarray  # bool canvas: positions whose votes were used
    support_pixels: int
    retained_edge_magnitude: float
    total_edge_magnitude: float  # unmasked, same image
    edge_evidence: float = 0.0  # summed magnitude of strong retained votes
    strong_threshold: float = 0.0

    @property
    def retained_edge_share(self) -> float | None:
        return None if self.total_edge_magnitude <= 0 else self.retained_edge_magnitude / self.total_edge_magnitude


class GridOrientationDescriptor:
    name = DESCRIPTOR_NAME
    version = DESCRIPTOR_VERSION

    def __init__(self, params: DescriptorParams = DescriptorParams()):
        self.params = params
        sigma = params.pre_blur_sigma
        self.blur_radius = int(np.ceil(BLUR_TRUNCATE * sigma)) if sigma > 0 else 0
        self.footprint_radius = self.blur_radius + SOBEL_RADIUS

    def record(self) -> dict:
        return {"name": self.name, "version": self.version, "parameters": asdict(self.params),
                "distance": "total variation 0.5 * L1 between magnitude-normalised descriptors, in [0, 1]",
                "exclusion": {"rule": EXCLUSION_RULE, "footprint_radius_px": self.footprint_radius,
                              "blur_kernel_radius_px": self.blur_radius, "sobel_radius_px": SOBEL_RADIUS,
                              "undefined_when": (f"edge evidence (summed magnitude of retained votes >= max("
                                                 f"{self.params.strong_gradient_floor}, {self.params.noise_factor} x "
                                                 f"median magnitude on paper-like support positions, darkness < {self.params.paper_darkness})) "
                                                 f"< {self.params.min_edge_evidence}"),
                              "thresholds_status": "provisional; chosen on fictional synthetic images"}}

    def magnitude_and_orientation(self, darkness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = darkness.astype(np.float32)
        if self.blur_radius > 0:
            k = 2 * self.blur_radius + 1
            d = cv2.GaussianBlur(d, (k, k), self.params.pre_blur_sigma, borderType=BORDER)
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3, borderType=BORDER)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3, borderType=BORDER)
        mag = np.hypot(gx, gy)
        ang = np.mod(np.arctan2(gy, gx), np.pi)
        return mag, ang

    def support(self, exclude: np.ndarray | None, shape: tuple[int, int] | None = None) -> np.ndarray:
        """Canvas positions whose votes do not depend on any excluded pixel.

        Mirrors the computation: dilate by the blur kernel, then by the Sobel kernel, each with the
        same reflect-101 border rule as the filter it stands for.
        """
        if exclude is None:
            return np.ones(shape, bool)
        if exclude.dtype != bool or exclude.ndim != 2:
            raise ValueError("exclude must be a 2-D boolean mask")
        tainted = exclude.astype(np.uint8)
        if tainted.any():
            for r in (self.blur_radius, SOBEL_RADIUS):
                if r > 0:
                    tainted = cv2.dilate(tainted, np.ones((2 * r + 1, 2 * r + 1), np.uint8), borderType=BORDER)
        return tainted == 0

    def describe(self, darkness: np.ndarray, exclude: np.ndarray | None = None) -> DescriptorResult:
        if darkness.ndim != 2:
            raise ValueError("darkness must be a 2-D array")
        p = self.params
        h, w = darkness.shape
        if h % p.cell_size or w % p.cell_size:
            raise ValueError(f"canvas {w}x{h} is not a multiple of cell size {p.cell_size}")
        if exclude is not None and exclude.shape != darkness.shape:
            raise ValueError("exclude must be a boolean mask of the canvas size")
        mag, ang = self.magnitude_and_orientation(darkness)
        total_mag = float(mag.sum(dtype=np.float64))
        support = self.support(exclude, (h, w))
        if exclude is not None:
            mag = np.where(support, mag, 0.0).astype(np.float32)
        pos = ang / np.pi * p.bins - 0.5
        lo = np.floor(pos).astype(np.int64)
        frac = (pos - lo).astype(np.float32)
        lo_bin, hi_bin = lo % p.bins, (lo + 1) % p.bins
        rows, cols = h // p.cell_size, w // p.cell_size
        cell = (np.arange(h)[:, None] // p.cell_size) * cols + (np.arange(w)[None, :] // p.cell_size)
        hist = np.zeros(rows * cols * p.bins, np.float64)
        np.add.at(hist, (cell * p.bins + lo_bin).ravel(), (mag * (1 - frac)).ravel())
        np.add.at(hist, (cell * p.bins + hi_bin).ravel(), (mag * frac).ravel())
        total = float(hist.sum())
        paper = support & (darkness < p.paper_darkness)
        noise = float(np.median(mag[paper])) if paper.any() else 0.0
        strong = max(p.strong_gradient_floor, p.noise_factor * noise)
        evidence = float(mag[support & (mag >= strong)].sum(dtype=np.float64))
        defined = evidence >= p.min_edge_evidence and total > 0
        return DescriptorResult(vector=hist / total if defined else None,
                                status="defined" if defined else "insufficient_edge_evidence",
                                support=support, support_pixels=int(support.sum()),
                                retained_edge_magnitude=float(mag.sum(dtype=np.float64)),
                                total_edge_magnitude=total_mag, edge_evidence=evidence,
                                strong_threshold=strong)

    def extract(self, darkness: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray | None:
        """Descriptor vector, or ``None`` when no edge evidence remains after exclusion."""
        return self.describe(darkness, exclude).vector

    @staticmethod
    def distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            raise InsufficientEvidenceError("distance is undefined: a descriptor has insufficient edge evidence")
        if a.shape != b.shape:
            raise ValueError("descriptor shapes differ")
        return float(0.5 * np.abs(a - b).sum())
