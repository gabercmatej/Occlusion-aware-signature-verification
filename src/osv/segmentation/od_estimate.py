"""Experimental optical-density (OD) two-ink stamp estimate (M1b).

Mathematical assumptions, stated plainly:

* Beer-Lambert: each ink absorbs independently and the sensor is linear, so the optical density
  ``OD_c = -log10(I_c / paper_c)`` of a pixel is a non-negative sum of ink contributions,
  ``OD = c_s * v_s + c_p * v_p`` with unit ink directions ``v_s`` (stamp), ``v_p`` (pen) in RGB
  OD space and concentrations ``c_s, c_p >= 0``. A pixel's OD *direction* does not depend on how
  much ink it holds, only on which inks it holds.
* Exactly two ink populations matter: a chromatic stamp ink and a near-neutral pen ink (the
  same role assumption as the baseline). When both inks are chromatic (e.g. two blues), OD
  directions may still differ, but colour cannot say which ink is the stamp; a dev-only
  "more chromatic = stamp" rule deleted most of a saturated blue pen under a navy stamp, so
  such cases are refused (``ink_roles_ambiguous``).
* Opaque inks, scanner tone curves, JPEG and colour misregistration violate the linear model.

Estimation from the observed image and the optional reviewer rectangle only:

1. Paper tone: the same local estimate as the baseline (mean of nearby non-ink pixels).
2. Ink directions: 2-means on unit OD directions of clearly inked pixels (deterministic init:
   most and least chromatic 5 %), centroids refined on the closer 70 % of each cluster. With a
   reviewer rectangle, only pixels in the rectangle (plus 10 px) are used, and only the
   rectangle may be deleted.
3. Separability diagnostics: angle between ``v_s`` and ``v_p``, a Fisher-like ratio
   (angle / pooled within-cluster angular spread), condition number of ``[v_s v_p]``.
4. Unmixing: least squares ``c = pinv([v_s v_p]) OD`` per pixel, with relative residual.
5. Decision: delete only pixels with stamp concentration and *no* pen concentration and a small
   residual, away from pen pixels. Pixels holding both inks are kept and marked ambiguous;
   their stamp component is *not* subtracted (this cleaner is deletion-only, like the baseline).

Refusals (``review``, nothing deleted): the directions are too close or not separable
relative to their spread (``optical_density_ill_conditioned``); no second ink population connected
to the stamp-direction ink (same guard semantics as the baseline estimator 0.2.0); stamp and pen
directions about equally chromatic (``ink_roles_ambiguous``); much ink unexplained by two inks.

Colour (OD direction) alone cannot establish the origin of indistinguishable ink. A neutral
mark touching same-colour handwriting still looks like "pen crossing stamp" here as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from osv.segmentation.stamp_estimate import (
    StampEstimate,
    _components_touching,
    _dilate,
    _disk,
    _luma,
    _paper_estimate,
    _region_mask,
    validate_region,
)

ESTIMATOR_NAME = "optical_density_two_ink"
ESTIMATOR_VERSION = "0.1.0"
METHOD = "optical-density"

_GRAY = np.full(3, 1.0 / np.sqrt(3.0), np.float32)


@dataclass(frozen=True)
class ODParams:
    ink_darkness: float = 0.12  # same ink definition as the baseline (1 - Y/Y_paper)
    paper_window: int = 41  # same paper estimate as the baseline
    od_floor: float = 0.04  # OD norm below which a pixel is treated as paper
    od_estimate_min: float = 0.10  # OD norm of pixels used to estimate ink directions (dev: 0.20 missed faint stamps)
    chromatic_min: float = 0.22  # sin(angle to the neutral axis) for chromatic ink
    min_stamp_pixels: int = 150
    min_stamp_fraction: float = 0.002
    min_angle_deg: float = 8.0  # stamp/pen OD directions closer than this -> ill-conditioned
    min_fisher: float = 2.0  # angle / pooled within-cluster spread below this -> not separable
    role_margin: float = 0.08  # stamp must be more chromatic than pen by this (sin units)
    pen_chromaticity_max: float = 0.20  # a chromatic 'pen' direction -> roles unknown -> review
    min_pen_share: float = 0.10  # connected pen-direction ink / (that + stamp) below this -> review
    contact_radius: int = 4  # pen ink within this of the *eroded* stamp ink counts for the guard
    seed_erosion: int = 1  # erode the stamp mask before the guard: thin colour fringes on neutral edges are not stamp evidence
    stamp_conc_min: float = 0.06  # unmixed stamp OD for a stamp pixel
    pen_conc_max: float = 0.06  # unmixed pen OD below which a stamp pixel holds no pen
    residual_max: float = 0.35  # relative unmixing residual above which a pixel is unexplained
    unexplained_share: float = 0.25  # share of strong ink unexplained by two inks -> review
    protect_radius: int = 1
    region_margin: int = 10
    grayscale_tolerance: int = 2


def _directions(od: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(od, axis=-1)
    dirs = od / np.maximum(norm, 1e-6)[..., None]
    return norm, dirs


def _chromaticity(d: np.ndarray) -> np.ndarray:
    """sin(angle between a unit OD direction and the neutral axis)."""
    c = np.clip(d @ _GRAY, -1.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - c * c))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _two_means(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic spherical 2-means on unit vectors ``x`` (N, 3). Returns centroids a, b and labels."""
    chrom = _chromaticity(x)
    order = np.argsort(chrom, kind="stable")
    k = max(1, len(x) // 20)
    a, b = _unit(x[order[-k:]].mean(0)), _unit(x[order[:k]].mean(0))
    labels = np.zeros(len(x), bool)
    for _ in range(30):
        new = (x @ a) >= (x @ b)
        if new.all() or (~new).all():
            labels = new
            break
        a, b = _unit(x[new].mean(0)), _unit(x[~new].mean(0))
        if np.array_equal(new, labels):
            break
        labels = new
    # Refine on the closer 70 % of each cluster (reduces pull from mixed pixels).
    out = []
    for sel, c in ((labels, a), (~labels, b)):
        pts = x[sel]
        if len(pts) == 0:
            out.append(c)
            continue
        cos = pts @ c
        keep = cos >= np.quantile(cos, 0.3)
        out.append(_unit(pts[keep].mean(0)))
    return out[0], out[1], labels


def _angular_spread(x: np.ndarray, c: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    ang = np.degrees(np.arccos(np.clip(x @ c, -1.0, 1.0)))
    return float(np.sqrt(np.mean(ang ** 2)))


def estimate_stamp_od(image_rgb: np.ndarray, stamp_region=None) -> StampEstimate:
    """Estimate stamp, ink and ambiguous pixels by two-ink optical-density unmixing.

    Same interface and output type as :func:`osv.segmentation.stamp_estimate.estimate_stamp`.
    """
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an (H, W, 3) array")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8")
    p = ODParams()
    hgt, wid = image_rgb.shape[:2]
    region = validate_region(stamp_region, hgt, wid)
    in_region = _region_mask(region, (hgt, wid))
    fit_zone = in_region if region is None else _dilate(in_region, p.region_margin)

    img = image_rgb.astype(np.float32)
    paper_rgb = _paper_estimate(img, p)
    darkness = np.clip(1.0 - _luma(img) / np.maximum(_luma(paper_rgb), 1.0), 0.0, 1.0)
    ink_dark = darkness >= p.ink_darkness
    od = np.maximum(0.0, -np.log10(np.maximum(img, 1.0) / np.maximum(paper_rgb, 1.0))).astype(np.float32)
    norm, dirs = _directions(od)
    chrom = _chromaticity(dirs)
    present = norm >= p.od_floor

    spread = image_rgb.max(axis=2).astype(np.int16) - image_rgb.min(axis=2).astype(np.int16)
    grayscale = bool(np.mean(spread <= p.grayscale_tolerance) >= 0.999)
    min_pixels = max(p.min_stamp_pixels, int(round(p.min_stamp_fraction * hgt * wid)))

    strong = present & (norm >= p.od_estimate_min) & fit_zone
    chromatic = strong & (chrom >= p.chromatic_min) & in_region

    empty = np.zeros((hgt, wid), bool)
    stats: dict = {
        "ink_pixels": int(ink_dark.sum()),
        "strong_ink_pixels_used_for_fit": int(strong.sum()),
        "chromatic_pixels_searched": int(chromatic.sum()),
        "min_stamp_pixels": int(min_pixels),
        "grayscale_input": grayscale,
    }
    base_reasons = ("grayscale_input",) if grayscale else ()

    def result(action, reasons, stamp=empty, ambiguous=empty, separability=None, ink=ink_dark):
        return StampEstimate(
            stamp_mask=stamp, ink_mask=ink, ambiguous_mask=ambiguous, separability=separability,
            reasons=tuple(dict.fromkeys(base_reasons + tuple(reasons))), action=action,
            paper_rgb=paper_rgb, stamp_region=region, stats=stats, params=p,  # type: ignore[arg-type]
            method=METHOD, estimator_name=ESTIMATOR_NAME, estimator_version=ESTIMATOR_VERSION,
        )

    if chromatic.sum() < min_pixels:
        if region is not None:
            return result("review", ("stamp_indicated_not_separated",), ambiguous=in_region.copy())
        return result("none", ("no_coloured_stamp_ink_found",))

    x = dirs[strong]
    a, b, labels = _two_means(x)
    ca, cb = float(_chromaticity(a[None])[0]), float(_chromaticity(b[None])[0])
    v_s, v_p = (a, b) if ca >= cb else (b, a)
    stamp_lab = labels if ca >= cb else ~labels
    chrom_s, chrom_p = max(ca, cb), min(ca, cb)
    angle = float(np.degrees(np.arccos(np.clip(float(v_s @ v_p), -1.0, 1.0))))
    pooled = float(np.sqrt((_angular_spread(x[stamp_lab], v_s) ** 2 + _angular_spread(x[~stamp_lab], v_p) ** 2) / 2))
    fisher = angle / max(pooled, 1e-6)
    m = np.stack([v_s, v_p], axis=1)  # (3, 2)
    sv = np.linalg.svd(m, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-9))
    separability = float(np.sin(np.radians(angle)))
    stats.update({
        "stamp_od_direction": [round(float(v), 4) for v in v_s],
        "pen_od_direction": [round(float(v), 4) for v in v_p],
        "stamp_chromaticity": round(chrom_s, 4),
        "pen_chromaticity": round(chrom_p, 4),
        "direction_angle_deg": round(angle, 3),
        "pooled_within_spread_deg": round(pooled, 3),
        "fisher_ratio": round(fisher, 3),
        "condition_number": round(cond, 3),
        "fit_cluster_sizes": [int(stamp_lab.sum()), int((~stamp_lab).sum())],
    })

    def refuse(reasons):
        ambiguous = in_region & present & (ink_dark | (chrom >= p.chromatic_min))
        if region is not None:
            reasons = tuple(reasons) + ("stamp_indicated_not_separated",)
        return result("review", reasons, ambiguous=ambiguous, separability=separability)

    if angle < p.min_angle_deg or fisher < p.min_fisher:
        return refuse(("ink_and_stamp_colour_overlap", "optical_density_ill_conditioned"))
    if chrom_s - chrom_p < p.role_margin or chrom_p >= p.pen_chromaticity_max:
        # Two inks may be separable in OD, but colour cannot say which one is the stamp. The
        # dev-only exploratory rule "more chromatic = stamp" deleted most of a saturated blue
        # pen under a navy stamp, so two chromatic inks are refused.
        return refuse(("ink_roles_ambiguous",))

    # Unmix every pixel.
    pinv = np.linalg.pinv(m)  # (2, 3)
    conc = od @ pinv.T  # (H, W, 2)
    c_s, c_p = conc[..., 0], conc[..., 1]
    recon = conc @ m.T
    resid = np.linalg.norm(od - recon, axis=-1) / np.maximum(norm, 1e-6)
    unexplained = present & (resid > p.residual_max)

    stamp = present & in_region & (c_s >= p.stamp_conc_min) & (c_s > c_p)
    pen_present = present & (c_p >= p.pen_conc_max)
    strong_all = present & (norm >= p.od_estimate_min)
    unexplained_share = float((unexplained & strong_all).sum()) / max(1, int(strong_all.sum()))

    # Guard evidence: only ink that itself looks like the pen counts: pen-dominant after unmixing
    # *and* near-neutral in direction. Under "over" compositing the same blue ink at higher
    # opacity has a less chromatic OD direction and unmixes as partly "pen" (dev finding).
    pen_dominant = pen_present & ink_dark & (c_p >= c_s) & (chrom < p.pen_chromaticity_max)
    stamp_core = cv2.erode(stamp.astype(np.uint8), _disk(p.seed_erosion)).astype(bool) if p.seed_erosion > 0 else stamp
    connected_pen = _components_touching(pen_dominant, _dilate(stamp_core, p.contact_radius))
    pen_n, stamp_n = int(connected_pen.sum()), int(stamp.sum())
    pen_share = pen_n / max(1, pen_n + stamp_n)
    global_pen = int(pen_dominant.sum())
    global_share = global_pen / max(1, global_pen + stamp_n)
    stats.update({
        "stamp_pixels_unmixed": stamp_n,
        "pen_ink_connected_pixels": pen_n,
        "pen_ink_share": round(pen_share, 4),
        "pen_ink_share_whole_crop": round(global_share, 4),
        "unexplained_strong_ink_share": round(unexplained_share, 4),
    })
    if unexplained_share >= p.unexplained_share:
        return refuse(("ink_unexplained_by_two_inks",))
    if pen_share < p.min_pen_share:
        reasons = ("ink_and_stamp_colour_overlap",)
        if global_share >= p.min_pen_share:
            reasons += ("neutral_ink_not_connected_to_stamp_colour",)
        return refuse(reasons)

    protect = _dilate(pen_present, p.protect_radius)
    deletable = stamp & ~pen_present & ~unexplained & ~protect
    ink = ink_dark | stamp
    ambiguous = (stamp & ~deletable) | (ink & ~stamp & _dilate(stamp, 1))
    reasons: tuple[str, ...] = ()
    if region is not None:
        reasons += ("reviewer_region_used",)
    if ambiguous.any():
        reasons += ("overlap_pixels_kept_ambiguous",)
    return result("delete", reasons, stamp=stamp, ambiguous=ambiguous, separability=separability, ink=ink)
