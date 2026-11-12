"""Geometry and intensity normalisation for references and cleaned queries (M2a).

One transform per image: a uniform scale that fits the whole image into a fixed canvas,
followed by centring. Aspect ratio is preserved; the image is never cropped, so residual
stamp ink and ambiguous handwriting stay inside the canvas exactly as they were. Every
associated mask goes through the *same* resampling (``cv2.resize`` with ``INTER_AREA`` into the
same target size, then the same offset); masks are resampled as coverage fractions and then
binarised with a recorded rule.

Coordinates follow :data:`osv.preprocessing.image_io.COORDINATE_CONVENTION` (corner origin).
In that convention the mapping is exactly ``x' = scale_x * x + offset_x`` and
``y' = scale_y * y + offset_y``; ``scale_x`` and ``scale_y`` differ only by the rounding of the
target size and both are recorded.

Intensity: the image is converted to *ink darkness* ``clip(1 - Y / Y_paper, 0, 1)`` where ``Y`` is
Rec.601 luma and ``Y_paper`` is the 95th percentile of luma in that image. This is colour
agnostic: a blue and a black signature of the same shape look alike to the descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from osv.preprocessing.image_io import COORDINATE_CONVENTION

NORMALISER_NAME = "fit_canvas_uniform_scale"
NORMALISER_VERSION = "0.1.0"

# Mask binarisation after resampling: a canvas pixel is flagged if any source pixel under it
# was flagged (conservative for evidence-limiting masks); ink needs half coverage.
MASK_RULES = {
    "stamp_mask": "any",
    "intervention_mask": "any",
    "ambiguous_mask": "any",
    "reviewer_region_mask": "any",
    "ink_mask": "half",
}
_ANY_EPS = 1e-6


@dataclass(frozen=True)
class NormaliseParams:
    canvas_height: int = 128
    canvas_width: int = 320
    paper_percentile: float = 95.0

    def as_dict(self) -> dict:
        return {"canvas_height": self.canvas_height, "canvas_width": self.canvas_width,
                "paper_percentile": self.paper_percentile}


@dataclass(frozen=True)
class Transform:
    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    target_width: int
    target_height: int
    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int

    def matrix(self) -> list[list[float]]:
        return [[self.scale_x, 0.0, float(self.offset_x)], [0.0, self.scale_y, float(self.offset_y)]]

    def map_points(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, np.float64)
        return np.stack([xy[..., 0] * self.scale_x + self.offset_x, xy[..., 1] * self.scale_y + self.offset_y], -1)

    def unmap_points(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, np.float64)
        return np.stack([(xy[..., 0] - self.offset_x) / self.scale_x, (xy[..., 1] - self.offset_y) / self.scale_y], -1)

    def record(self) -> dict:
        return {
            "source_size": [self.source_width, self.source_height],
            "canvas_size": [self.canvas_width, self.canvas_height],
            "target_size": [self.target_width, self.target_height],
            "scale_x": round(self.scale_x, 9),
            "scale_y": round(self.scale_y, 9),
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "matrix_2x3": [[round(v, 9) for v in row] for row in self.matrix()],
            "convention": COORDINATE_CONVENTION + "; canvas uses the same convention",
            "maps": "x' = scale_x * x + offset_x, y' = scale_y * y + offset_y (no crop, no rotation)",
            "interpolation": "cv2.resize INTER_AREA for image, darkness and mask coverage",
        }


@dataclass
class NormalisedImage:
    darkness: np.ndarray  # float32 (H, W) in [0, 1], 0 = paper
    rgb: np.ndarray  # uint8 (H, W, 3), padding filled with the estimated paper colour
    valid_mask: np.ndarray  # bool: canvas pixels covered by the source image
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # bool, canvas size
    transform: Transform | None = None
    paper_luma: float = 0.0
    params: NormaliseParams = field(default_factory=NormaliseParams)

    def record(self) -> dict:
        return {
            "normaliser": {"name": NORMALISER_NAME, "version": NORMALISER_VERSION, "parameters": self.params.as_dict()},
            "transform": self.transform.record(),
            "paper_luma": round(self.paper_luma, 4),
            "intensity": "ink darkness clip(1 - Y / Y_paper, 0, 1); Y = Rec.601 luma; Y_paper = percentile of luma",
            "mask_rules": {k: MASK_RULES[k] for k in sorted(self.masks)},
        }


def compute_transform(width: int, height: int, params: NormaliseParams = NormaliseParams()) -> Transform:
    if width < 1 or height < 1:
        raise ValueError("image has no pixels")
    s = min(params.canvas_width / width, params.canvas_height / height)
    tw = int(min(params.canvas_width, max(1, round(width * s))))
    th = int(min(params.canvas_height, max(1, round(height * s))))
    return Transform(
        source_width=int(width), source_height=int(height),
        canvas_width=params.canvas_width, canvas_height=params.canvas_height,
        target_width=tw, target_height=th,
        scale_x=tw / width, scale_y=th / height,
        offset_x=(params.canvas_width - tw) // 2, offset_y=(params.canvas_height - th) // 2,
    )


def _place(t: Transform, small: np.ndarray, fill) -> np.ndarray:
    shape = (t.canvas_height, t.canvas_width) + small.shape[2:]
    canvas = np.empty(shape, small.dtype)
    canvas[...] = fill
    canvas[t.offset_y:t.offset_y + t.target_height, t.offset_x:t.offset_x + t.target_width] = small
    return canvas


def _resample(t: Transform, array: np.ndarray, fill) -> np.ndarray:
    small = cv2.resize(array, (t.target_width, t.target_height), interpolation=cv2.INTER_AREA)
    return _place(t, small, fill)


def ink_darkness(rgb: np.ndarray, percentile: float = 95.0) -> tuple[np.ndarray, float]:
    y = 0.299 * rgb[..., 0].astype(np.float32) + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    paper = float(np.percentile(y, percentile))
    paper = max(paper, 1.0)
    return np.clip(1.0 - y / paper, 0.0, 1.0).astype(np.float32), paper


def normalise(rgb: np.ndarray, masks: dict[str, np.ndarray] | None = None,
              params: NormaliseParams = NormaliseParams()) -> NormalisedImage:
    """Normalise one display-oriented RGB image and its masks with one shared transform."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an (H, W, 3) uint8 array")
    h, w = rgb.shape[:2]
    masks = masks or {}
    for name, m in masks.items():
        if name not in MASK_RULES:
            raise ValueError(f"unknown mask {name!r}; expected one of {tuple(MASK_RULES)}")
        if m.shape != (h, w) or m.dtype != bool:
            raise ValueError(f"mask {name} must be boolean with shape {(h, w)}")
    t = compute_transform(w, h, params)
    darkness, paper = ink_darkness(rgb, params.paper_percentile)
    paper_rgb = np.percentile(rgb.reshape(-1, 3), params.paper_percentile, axis=0).round().astype(np.uint8)

    out_masks = {}
    for name, m in masks.items():
        frac = _resample(t, m.astype(np.float32), 0.0)
        out_masks[name] = frac > _ANY_EPS if MASK_RULES[name] == "any" else frac >= 0.5
    return NormalisedImage(
        darkness=np.clip(_resample(t, darkness, 0.0), 0.0, 1.0).astype(np.float32),
        rgb=_resample(t, rgb, paper_rgb),
        valid_mask=_resample(t, np.ones((h, w), np.float32), 0.0) > 0.5,
        masks=out_masks,
        transform=t,
        paper_luma=paper,
        params=params,
    )
