"""Baseline colour (chroma) stamp estimate from the observed image alone.

Model and its limits, stated plainly:

* Assumes the stamp is a *coloured* ink and the handwriting is a dark, low-chroma ink.
  It finds the dominant coloured-ink hue and calls those pixels stamp candidates.
* It cannot see a black, grey or pen-coloured stamp, nor any stamp in a grayscale scan.
  In that case the result is ``no_stamp_detected``, which does not mean "clean".
* Guard: if the coloured cluster leaves almost no neutral (pen-like) ink, the stamp colour
  could also be the pen colour. Nothing is marked for deletion and the case goes to review.
  Only neutral ink *connected* to the stamp-hue ink counts as evidence of a separate pen
  (a pen stroke crossing the stamp is; an isolated black mark elsewhere on the crop is not).
  This catches the synthetic blue-on-blue case, also with unrelated black marks added. It is
  not proof that every same-colour overlap is caught: a neutral mark that touches same-colour
  handwriting still looks like a pen crossing a stamp and can authorize deletion.

Inputs are the RGB image and an optional reviewer-drawn rectangle. No ground truth,
filename, fixture kind or other metadata is accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

ESTIMATOR_NAME = "chroma_cluster_baseline"
ESTIMATOR_VERSION = "0.2.0"

Region = tuple[int, int, int, int]  # x, y, w, h in pixels


@dataclass(frozen=True)
class EstimatorParams:
    ink_darkness: float = 0.12  # 1 - Y/Y_paper above which a pixel counts as ink
    colour_chroma: float = 18.0  # Lab chroma (relative to paper) for coloured ink
    neutral_chroma: float = 10.0  # Lab chroma below which ink counts as neutral (pen-like)
    min_colour_darkness: float = 0.04  # coloured pixels must be at least this dark
    hue_halfwidth_deg: float = 25.0  # half-width of the stamp hue cluster
    min_stamp_pixels: int = 150  # absolute floor for a stamp detection
    min_stamp_fraction: float = 0.002  # ... and relative to image area
    min_neutral_share: float = 0.10  # guard: connected neutral ink / (that + stamp-hue) below this -> review
    neutral_contact_radius: int = 3  # neutral ink components within this of stamp-hue ink count for the guard
    other_colour_share: float = 0.20  # a second coloured ink this large -> review
    overlap_darkness_margin: float = 0.15  # stamp-hue pixels darker than median + margin -> overlap
    protect_radius: int = 1  # stamp pixels this close to pen-like ink are kept
    paper_window: int = 41  # window (px) for the local paper estimate
    grayscale_tolerance: int = 2  # max channel spread for a pixel to count as gray


@dataclass
class StampEstimate:
    """Output of :func:`estimate_stamp`.

    ``action`` is ``"none"`` (no stamp evidence), ``"delete"`` (``stamp_mask & ~ambiguous_mask``
    may be deleted) or ``"review"`` (nothing may be deleted).
    """

    stamp_mask: np.ndarray
    ink_mask: np.ndarray
    ambiguous_mask: np.ndarray
    separability: float | None
    reasons: tuple[str, ...]
    action: str
    paper_rgb: np.ndarray  # float32 (H, W, 3) local paper tone estimate
    stamp_region: Region | None = None
    stats: dict = field(default_factory=dict)
    params: EstimatorParams = field(default_factory=EstimatorParams)
    # Which estimator produced this (recorded in provenance; M1b adds a second method).
    method: str = "chroma"
    estimator_name: str = ESTIMATOR_NAME
    estimator_version: str = ESTIMATOR_VERSION

    def params_dict(self) -> dict:
        return asdict(self.params)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), _disk(radius)).astype(bool)


def validate_region(region: Region | None, height: int, width: int) -> Region | None:
    if region is None:
        return None
    if len(region) != 4:
        raise ValueError("stamp region must be x,y,w,h")
    x, y, w, h = (int(v) for v in region)
    if w <= 0 or h <= 0:
        raise ValueError("stamp region width and height must be positive")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"stamp region {region} lies outside the {width}x{height} image")
    return (x, y, w, h)


def _region_mask(region: Region | None, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, bool)
    if region is None:
        m[:] = True
    else:
        x, y, w, h = region
        m[y:y + h, x:x + w] = True
    return m


def _paper_estimate(img: np.ndarray, p: EstimatorParams) -> np.ndarray:
    """Local paper tone: mean of nearby non-ink pixels (a background colour, not a reconstruction)."""
    y = _luma(img)
    k = p.paper_window
    bright = cv2.blur(cv2.dilate(y, np.ones((k, k), np.uint8)), (k, k))
    candidate_ink = y < bright * (1.0 - p.ink_darkness)
    paper = (~_dilate(candidate_ink, 1)).astype(np.float32)
    if paper.sum() == 0:
        return np.repeat(bright[..., None], 3, axis=2).astype(np.float32)
    num = np.stack([cv2.boxFilter(img[..., c] * paper, -1, (k, k), normalize=False) for c in range(3)], axis=2)
    den = cv2.boxFilter(paper, -1, (k, k), normalize=False)[..., None]
    global_paper = np.median(img[paper.astype(bool)], axis=0).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        local = np.where(den > 0.5, num / np.maximum(den, 1e-6), global_paper[None, None, :])
    return local.astype(np.float32)


def _components_touching(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Pixels of ``mask`` whose 8-connected component overlaps ``seed``."""
    if not mask.any() or not (mask & seed).any():
        return np.zeros_like(mask)
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    hit = np.unique(labels[mask & seed])
    return np.isin(labels, hit[hit > 0]) & mask


def _circular_distance(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def estimate_stamp(image_rgb: np.ndarray, stamp_region: Region | None = None) -> StampEstimate:
    """Estimate stamp, ink and ambiguous pixels from an RGB uint8 image.

    ``stamp_region`` is an optional reviewer-drawn rectangle ``(x, y, w, h)``; with it, the
    stamp is searched for (and later deleted) only inside the rectangle, and a rectangle in
    which no separable coloured stamp is found becomes an ambiguous region.
    """
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an (H, W, 3) array")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8")
    p = EstimatorParams()
    hgt, wid = image_rgb.shape[:2]
    region = validate_region(stamp_region, hgt, wid)
    in_region = _region_mask(region, (hgt, wid))

    img = image_rgb.astype(np.float32)
    paper_rgb = _paper_estimate(img, p)
    darkness = np.clip(1.0 - _luma(img) / np.maximum(_luma(paper_rgb), 1.0), 0.0, 1.0)
    ink = darkness >= p.ink_darkness

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    paper_lab = cv2.cvtColor(np.clip(np.round(paper_rgb), 0, 255).astype(np.uint8), cv2.COLOR_RGB2Lab).astype(np.float32)
    da, db = lab[..., 1] - paper_lab[..., 1], lab[..., 2] - paper_lab[..., 2]
    chroma = np.hypot(da, db)
    hue = np.degrees(np.arctan2(db, da)) % 360.0

    spread = image_rgb.max(axis=2).astype(np.int16) - image_rgb.min(axis=2).astype(np.int16)
    grayscale = bool(np.mean(spread <= p.grayscale_tolerance) >= 0.999)

    coloured = (chroma >= p.colour_chroma) & (darkness >= p.min_colour_darkness)
    neutral_ink = ink & (chroma < p.neutral_chroma)
    coloured_r = coloured & in_region
    min_pixels = max(p.min_stamp_pixels, int(round(p.min_stamp_fraction * hgt * wid)))

    empty = np.zeros((hgt, wid), bool)
    stats: dict = {
        "ink_pixels": int(ink.sum()),
        "neutral_ink_pixels": int(neutral_ink.sum()),
        "coloured_pixels_searched": int(coloured_r.sum()),
        "min_stamp_pixels": int(min_pixels),
        "grayscale_input": grayscale,
    }
    base_reasons = ("grayscale_input",) if grayscale else ()

    def result(action, reasons, stamp=empty, ambiguous=empty, separability=None):
        return StampEstimate(
            stamp_mask=stamp, ink_mask=ink, ambiguous_mask=ambiguous, separability=separability,
            reasons=tuple(dict.fromkeys(base_reasons + tuple(reasons))), action=action,
            paper_rgb=paper_rgb, stamp_region=region, stats=stats, params=p,
        )

    if coloured_r.sum() < min_pixels:
        if region is not None:
            # The reviewer says a stamp is here but colour cannot separate it.
            return result("review", ("stamp_indicated_not_separated",), ambiguous=in_region.copy())
        return result("none", ("no_coloured_stamp_ink_found",))

    # Dominant hue among coloured pixels (10-degree bins, circular smoothing).
    hist = np.bincount((hue[coloured_r] // 10).astype(int) % 36, minlength=36).astype(float)
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    centre = float(np.argmax(smooth) * 10 + 5)
    near = coloured_r & (_circular_distance(hue, centre) <= p.hue_halfwidth_deg)
    ang = np.radians(hue[near])
    peak = float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0)
    cluster = coloured_r & (_circular_distance(hue, peak) <= p.hue_halfwidth_deg)

    cluster_n = int(cluster.sum())
    other_n = int((coloured_r & ~cluster).sum())
    # Only neutral ink that reaches the stamp-hue ink is evidence of a pen distinct from the
    # stamp colour. Counting neutral ink anywhere on the crop let an unrelated black mark
    # authorize deletion of same-colour handwriting (M1a.1 regression).
    connected_neutral = _components_touching(neutral_ink, _dilate(cluster, p.neutral_contact_radius))
    neutral_n = int(connected_neutral.sum())
    neutral_share = neutral_n / max(1, neutral_n + cluster_n)
    global_n = int(neutral_ink.sum())
    global_share = global_n / max(1, global_n + cluster_n)

    # Separability: chroma distance between stamp-hue ink and the remaining ink, relative to
    # the stamp's chroma (0 = no distinct second ink population, 1 = fully distinct).
    rest = ink & ~cluster & ~coloured
    separability = 0.0
    if rest.sum() >= min_pixels:
        dist = np.hypot(da[cluster].mean() - da[rest].mean(), db[cluster].mean() - db[rest].mean())
        separability = float(np.clip(dist / max(chroma[cluster].mean(), 1e-6), 0.0, 1.0))

    stats.update({
        "stamp_hue_deg": round(peak, 2),
        "stamp_hue_pixels": cluster_n,
        "other_coloured_pixels": other_n,
        "neutral_ink_connected_pixels": neutral_n,
        "neutral_ink_share": round(neutral_share, 4),
        "neutral_ink_share_whole_crop": round(global_share, 4),
    })

    if other_n >= min_pixels and other_n >= p.other_colour_share * cluster_n:
        return result("review", ("multiple_coloured_inks",), ambiguous=coloured_r.copy(), separability=separability)

    if neutral_share < p.min_neutral_share:
        # Stamp colour model explains (almost) all ink: it may be the pen. Delete nothing.
        reasons = ("ink_and_stamp_colour_overlap",)
        if global_share >= p.min_neutral_share:
            # Neutral ink exists, but none of enough size touches the stamp colour.
            reasons += ("neutral_ink_not_connected_to_stamp_colour",)
        if region is not None:
            reasons += ("stamp_indicated_not_separated",)
        ambiguous = cluster | (ink & in_region & _dilate(cluster, 1))
        return result("review", reasons, ambiguous=ambiguous, separability=separability)

    median_dark = float(np.median(darkness[cluster]))
    dark_overlap = cluster & (darkness > median_dark + p.overlap_darkness_margin)
    protect = _dilate(neutral_ink | dark_overlap, p.protect_radius)
    deletable = cluster & ~dark_overlap & ~protect
    ambiguous = (cluster & ~deletable) | (ink & ~cluster & _dilate(cluster, 1))
    stats["stamp_median_darkness"] = round(median_dark, 4)

    reasons: tuple[str, ...] = ()
    if region is not None:
        reasons += ("reviewer_region_used",)
    if ambiguous.any():
        reasons += ("overlap_pixels_kept_ambiguous",)
    return result("delete", reasons, stamp=cluster, ambiguous=ambiguous, separability=separability)
