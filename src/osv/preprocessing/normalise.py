"""Geometry and intensity normalisation for references and cleaned queries (M2a).

Normaliser 0.2.0 (M2a.2) is *content based*: the canvas is placed around estimated dark content
of the image itself, so the result no longer depends on how much paper the crop happened to
include (checkpoint review A1; 0.1.0 fitted the whole crop rectangle).

1. **Ink darkness** ``clip(1 - Y / Y_paper, 0, 1)`` with Rec.601 luma ``Y`` and ``Y_paper`` the
   95th luma percentile of the whole image (colour agnostic).
2. **Content box**: the bounding box of pixels with darkness >= ``ink_threshold`` after removing
   8-connected pieces smaller than ``min_ink_component_px``. It is estimated dark content, not
   necessarily handwriting: stamp residue, printed text or ruling lines also count. It is
   computed from this image only, never from references. Without such pixels the whole image is
   used and the flag ``no_content_detected`` is set.
3. **Window**: one uniform scale fits the content box inside the canvas inset by
   ``canvas_margin_px`` on every side (a margin relative to the box size, since it is fixed in
   canvas pixels); the window of the canvas's aspect ratio is centred on the box and taken from
   the source image. Real paper around the box is therefore kept. Window parts outside the source
   image are *padding*: their canvas pixels are not valid (``valid_mask`` False) and the
   descriptor excludes them together with every vote that depends on them.
4. Image, darkness, validity and every mask go through the same integer window and the same
   ``cv2.resize(..., INTER_AREA)``; masks are resampled as coverage fractions and binarised with a
   recorded rule; a canvas pixel is valid only if it is fully covered by source pixels.

Coordinates follow :data:`osv.preprocessing.image_io.COORDINATE_CONVENTION` (corner origin).
The complete source-to-canvas map is ``x' = scale_x * (x - window_x)``,
``y' = scale_y * (y - window_y)``; ``scale_x`` and ``scale_y`` differ only by the rounding of the
integer window and both are recorded with the 2x3 matrix.

**Uncertain box.** When an evidence-limiting mask (stamp, intervention, proposed intervention,
ambiguous, reviewer region) lies within one margin width of a box edge, that edge is reported as
``content_box_uncertain:<side>``: a stamp over the signature's extremity may have hidden or
added the content that defines it. ``content_box_touches_image_border:<side>`` reports content
cut by the crop. Both are descriptive flags, not corrections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from osv.preprocessing.image_io import COORDINATE_CONVENTION

NORMALISER_NAME = "content_box_uniform_scale"
NORMALISER_VERSION = "0.2.0"

# Mask binarisation after resampling: a canvas pixel is flagged if any source pixel under it
# was flagged (conservative for evidence-limiting masks); ink needs half coverage.
MASK_RULES = {
    "stamp_mask": "any",
    "intervention_mask": "any",
    "proposed_intervention_mask": "any",
    "ambiguous_mask": "any",
    "reviewer_region_mask": "any",
    "ink_mask": "half",
}
EVIDENCE_LIMITING = tuple(k for k, v in MASK_RULES.items() if v == "any")
_ANY_EPS = 1e-6
SIDES = ("left", "top", "right", "bottom")
CONTENT_RULE = ("bounding box of pixels with ink darkness >= ink_threshold after dropping 8-connected "
                "pieces smaller than min_ink_component_px; estimated dark content, not necessarily "
                "handwriting (stamp residue, print and lines count too); computed from this image only")


@dataclass(frozen=True)
class NormaliseParams:
    canvas_height: int = 128
    canvas_width: int = 320
    paper_percentile: float = 95.0
    ink_threshold: float = 0.30  # provisional; blank-paper noise and 20 % shading stay below it
    min_ink_component_px: int = 4  # provisional; source pixels, resolution dependent
    canvas_margin_px: int = 8  # content box inset; exceeds the descriptor footprint (5 px)

    def as_dict(self) -> dict:
        return {"canvas_height": self.canvas_height, "canvas_width": self.canvas_width,
                "paper_percentile": self.paper_percentile, "ink_threshold": self.ink_threshold,
                "min_ink_component_px": self.min_ink_component_px, "canvas_margin_px": self.canvas_margin_px}


@dataclass(frozen=True)
class Transform:
    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    window_x: int
    window_y: int
    window_width: int
    window_height: int
    scale_x: float
    scale_y: float

    def matrix(self) -> list[list[float]]:
        return [[self.scale_x, 0.0, -self.scale_x * self.window_x],
                [0.0, self.scale_y, -self.scale_y * self.window_y]]

    def map_points(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, np.float64)
        return np.stack([(xy[..., 0] - self.window_x) * self.scale_x,
                         (xy[..., 1] - self.window_y) * self.scale_y], -1)

    def unmap_points(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, np.float64)
        return np.stack([xy[..., 0] / self.scale_x + self.window_x, xy[..., 1] / self.scale_y + self.window_y], -1)

    @property
    def target_width(self) -> float:
        """Width of the whole source image on the canvas (may exceed the canvas)."""
        return self.source_width * self.scale_x

    @property
    def target_height(self) -> float:
        return self.source_height * self.scale_y

    def record(self) -> dict:
        return {
            "source_size": [self.source_width, self.source_height],
            "canvas_size": [self.canvas_width, self.canvas_height],
            "source_window": {"x": self.window_x, "y": self.window_y,
                              "w": self.window_width, "h": self.window_height},
            "scale_x": round(self.scale_x, 9),
            "scale_y": round(self.scale_y, 9),
            "matrix_2x3": [[round(v, 9) for v in row] for row in self.matrix()],
            "convention": COORDINATE_CONVENTION + "; canvas uses the same convention",
            "maps": "x' = scale_x * (x - window.x), y' = scale_y * (y - window.y); no rotation",
            "outside_window": "source pixels outside the window are not on the canvas",
            "padding": "window parts outside the source image are padding: not valid, excluded by the descriptor",
            "interpolation": "cv2.resize INTER_AREA of the integer window for image, darkness, validity and mask coverage",
        }


@dataclass
class NormalisedImage:
    darkness: np.ndarray  # float32 (H, W) in [0, 1], 0 = paper (and padding)
    rgb: np.ndarray  # uint8 (H, W, 3), padding filled with the estimated paper colour
    valid_mask: np.ndarray  # bool: canvas pixels fully covered by source pixels
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # bool, canvas size
    transform: Transform | None = None
    paper_luma: float = 0.0
    params: NormaliseParams = field(default_factory=NormaliseParams)
    content_box: tuple[int, int, int, int] | None = None  # x, y, w, h in source pixels
    content_flags: tuple[str, ...] = ()

    @property
    def padding_mask(self) -> np.ndarray:
        return ~self.valid_mask

    def record(self) -> dict:
        box = self.content_box
        return {
            "normaliser": {"name": NORMALISER_NAME, "version": NORMALISER_VERSION, "parameters": self.params.as_dict()},
            "transform": self.transform.record(),
            "content_box": {**(dict(zip(("x", "y", "w", "h"), box)) if box else dict.fromkeys(("x", "y", "w", "h"))),
                            "rule": CONTENT_RULE,
                            "fallback": "whole image when no content is detected (flag no_content_detected)"},
            "content_flags": list(self.content_flags),
            "padding_share_of_canvas": round(float(self.padding_mask.mean()), 6),
            "paper_luma": round(self.paper_luma, 4),
            "intensity": "ink darkness clip(1 - Y / Y_paper, 0, 1); Y = Rec.601 luma; Y_paper = percentile of luma",
            "mask_rules": {k: MASK_RULES[k] for k in sorted(self.masks)},
        }


def ink_darkness(rgb: np.ndarray, percentile: float = 95.0) -> tuple[np.ndarray, float]:
    y = 0.299 * rgb[..., 0].astype(np.float32) + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    paper = float(np.percentile(y, percentile))
    paper = max(paper, 1.0)
    return np.clip(1.0 - y / paper, 0.0, 1.0).astype(np.float32), paper


def content_box(darkness: np.ndarray, params: NormaliseParams = NormaliseParams()) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) of estimated dark content, or ``None`` when there is none."""
    ink = (darkness >= params.ink_threshold).astype(np.uint8)
    if not ink.any():
        return None
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= params.min_ink_component_px]
    if not keep:
        return None
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    y0 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    y1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def compute_transform(width: int, height: int, params: NormaliseParams = NormaliseParams(),
                      box: tuple[int, int, int, int] | None = None) -> Transform:
    """Transform placing ``box`` (default: the whole image) inside the inset canvas."""
    if width < 1 or height < 1:
        raise ValueError("image has no pixels")
    bx, by, bw, bh = box if box is not None else (0, 0, width, height)
    cw, ch, m = params.canvas_width, params.canvas_height, params.canvas_margin_px
    if cw <= 2 * m or ch <= 2 * m:
        raise ValueError("canvas margin leaves no room for content")
    s = min((cw - 2 * m) / max(bw, 1), (ch - 2 * m) / max(bh, 1))
    ww, wh = max(1, int(round(cw / s))), max(1, int(round(ch / s)))
    wx = int(np.floor(bx + bw / 2.0 - ww / 2.0 + 0.5))
    wy = int(np.floor(by + bh / 2.0 - wh / 2.0 + 0.5))
    return Transform(source_width=int(width), source_height=int(height), canvas_width=cw, canvas_height=ch,
                     window_x=wx, window_y=wy, window_width=ww, window_height=wh,
                     scale_x=cw / ww, scale_y=ch / wh)


def _window(t: Transform, array: np.ndarray, fill) -> np.ndarray:
    shape = (t.window_height, t.window_width) + array.shape[2:]
    out = np.empty(shape, array.dtype)
    out[...] = fill
    h, w = array.shape[:2]
    sx0, sy0 = max(0, t.window_x), max(0, t.window_y)
    sx1, sy1 = min(w, t.window_x + t.window_width), min(h, t.window_y + t.window_height)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - t.window_y:sy1 - t.window_y, sx0 - t.window_x:sx1 - t.window_x] = array[sy0:sy1, sx0:sx1]
    return out


def _resample(t: Transform, array: np.ndarray, fill) -> np.ndarray:
    return cv2.resize(_window(t, array, fill), (t.canvas_width, t.canvas_height), interpolation=cv2.INTER_AREA)


def _box_flags(box, t: Transform, flagged: np.ndarray | None, params: NormaliseParams) -> list[str]:
    bx, by, bw, bh = box
    h, w = t.source_height, t.source_width
    flags = []
    ends = {"left": bx, "top": by, "right": w - (bx + bw), "bottom": h - (by + bh)}
    flags += [f"content_box_touches_image_border:{s}" for s in SIDES if ends[s] <= 1]
    if flagged is not None and flagged.any():
        band = max(1, int(np.ceil(params.canvas_margin_px / min(t.scale_x, t.scale_y))))
        ylo, yhi = max(0, by - band), min(h, by + bh + band)
        xlo, xhi = max(0, bx - band), min(w, bx + bw + band)
        strips = {
            "left": flagged[ylo:yhi, max(0, bx - band):min(w, bx + band)],
            "right": flagged[ylo:yhi, max(0, bx + bw - band):min(w, bx + bw + band)],
            "top": flagged[max(0, by - band):min(h, by + band), xlo:xhi],
            "bottom": flagged[max(0, by + bh - band):min(h, by + bh + band), xlo:xhi],
        }
        flags += [f"content_box_uncertain:{s}" for s in SIDES if strips[s].any()]
    return flags


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
    darkness, paper = ink_darkness(rgb, params.paper_percentile)
    box = content_box(darkness, params)
    t = compute_transform(w, h, params, box)
    flagged = None
    for name in EVIDENCE_LIMITING:
        if name in masks:
            flagged = masks[name].copy() if flagged is None else flagged | masks[name]
    flags = ["no_content_detected"] if box is None else _box_flags(box, t, flagged, params)
    paper_rgb = np.percentile(rgb.reshape(-1, 3), params.paper_percentile, axis=0).round().astype(np.uint8)

    out_masks = {}
    for name, m in masks.items():
        frac = _resample(t, m.astype(np.float32), 0.0)
        out_masks[name] = frac > _ANY_EPS if MASK_RULES[name] == "any" else frac >= 0.5
    valid = _resample(t, np.ones((h, w), np.float32), 0.0) >= 1.0 - _ANY_EPS
    return NormalisedImage(
        darkness=np.clip(_resample(t, darkness, 0.0), 0.0, 1.0).astype(np.float32),
        rgb=_resample(t, rgb, paper_rgb),
        valid_mask=valid,
        masks=out_masks,
        transform=t,
        paper_luma=paper,
        params=params,
        content_box=box,
        content_flags=tuple(flags),
    )
