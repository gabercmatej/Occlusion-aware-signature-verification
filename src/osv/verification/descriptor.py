"""Classical descriptor and distance for the M2a engineering baseline.

Descriptor: a grid of gradient-orientation histograms (HOG-like) on the normalised ink-darkness
canvas. Each pixel votes its gradient magnitude into one of ``bins`` unsigned orientations
(linear interpolation between neighbouring bins) inside a fixed grid cell; the whole vector is
then divided by its total, so it describes *where* stroke edges are and *which directions* they
run, independent of ink colour and total ink amount.

Why this one: it needs only NumPy/OpenCV, every number can be traced to pixels, pixels can be
excluded exactly (no receptive field leaks a masked region into the vector), and gradient
histograms are a standard classical baseline family for offline signatures. It does not model
stroke order, pressure, local deformation or the natural variation of a writer, and the fixed
grid makes it sensitive to layout differences that normalisation does not remove.

Distance: total-variation distance ``0.5 * sum |a - b|`` in [0, 1]; 0 for identical vectors.

The descriptor is deliberately separate from reference aggregation
(:mod:`osv.verification.compare`), so a learned encoder can later replace it behind the same
``extract`` / ``distance`` interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

DESCRIPTOR_NAME = "grid_orientation_histogram"
DESCRIPTOR_VERSION = "0.1.0"


@dataclass(frozen=True)
class DescriptorParams:
    cell_size: int = 32  # pixels on the normalised canvas
    bins: int = 8  # unsigned orientations over [0, pi)
    pre_blur_sigma: float = 1.0
    min_total_magnitude: float = 1.0  # below this the image has no usable edge evidence


class GridOrientationDescriptor:
    name = DESCRIPTOR_NAME
    version = DESCRIPTOR_VERSION

    def __init__(self, params: DescriptorParams = DescriptorParams()):
        self.params = params

    def record(self) -> dict:
        return {"name": self.name, "version": self.version, "parameters": asdict(self.params),
                "distance": "total variation 0.5 * L1 between magnitude-normalised descriptors, in [0, 1]"}

    def magnitude_and_orientation(self, darkness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = darkness.astype(np.float32)
        if self.params.pre_blur_sigma > 0:
            d = cv2.GaussianBlur(d, (0, 0), self.params.pre_blur_sigma)
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = np.mod(np.arctan2(gy, gx), np.pi)
        return mag, ang

    def extract(self, darkness: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray | None:
        """Descriptor vector, or ``None`` when no edge evidence remains after exclusion."""
        if darkness.ndim != 2:
            raise ValueError("darkness must be a 2-D array")
        p = self.params
        h, w = darkness.shape
        if h % p.cell_size or w % p.cell_size:
            raise ValueError(f"canvas {w}x{h} is not a multiple of cell size {p.cell_size}")
        mag, ang = self.magnitude_and_orientation(darkness)
        if exclude is not None:
            if exclude.shape != darkness.shape or exclude.dtype != bool:
                raise ValueError("exclude must be a boolean mask of the canvas size")
            mag = np.where(exclude, 0.0, mag).astype(np.float32)
        pos = ang / np.pi * p.bins - 0.5
        lo = np.floor(pos).astype(np.int64)
        frac = (pos - lo).astype(np.float32)
        lo_bin, hi_bin = lo % p.bins, (lo + 1) % p.bins
        rows, cols = h // p.cell_size, w // p.cell_size
        cell = (np.arange(h)[:, None] // p.cell_size) * cols + (np.arange(w)[None, :] // p.cell_size)
        hist = np.zeros(rows * cols * p.bins, np.float64)
        np.add.at(hist, (cell * p.bins + lo_bin).ravel(), (mag * (1 - frac)).ravel())
        np.add.at(hist, (cell * p.bins + hi_bin).ravel(), (mag * frac).ravel())
        total = hist.sum()
        if total < p.min_total_magnitude:
            return None
        return hist / total

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            raise ValueError("descriptor shapes differ")
        return float(0.5 * np.abs(a - b).sum())
