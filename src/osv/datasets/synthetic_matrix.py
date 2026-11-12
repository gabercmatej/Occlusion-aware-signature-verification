"""M1b evaluation matrix: scenarios x mixture models x stamp geometries x fixed seed groups.

This module generates evaluation images with exact ground truth. The original M1a generator
(:mod:`osv.datasets.synthetic_fixtures`) is untouched and stays reproducible.

Two compositing models, chosen to be genuinely different:

``multiplicative``
    The M1a model: each ink multiplies transmittance, ``I = paper * prod_k T_k ** alpha_k``,
    computed in linear 0-255 values, then blur (sigma 0.6) and noise (sd 1.5). This is exactly
    Beer-Lambert absorption, i.e. optical densities *add*. An optical-density cleaner shares
    this assumption, so this model alone would test that cleaner on its own assumption.

``alpha_over_scan``
    Opaque-ish layers composited with the "over" operator in gamma-encoded sRGB:
    ``out = under * (1 - a*o) + colour * a*o``. A later layer *hides* what is under it
    instead of adding density, and mixing happens in encoded (not linear) values. The layer
    order is random per case (``stamp_over_pen`` or ``pen_over_stamp``). Scanner effects:
    paper fibre texture, a tone curve (gamma 0.9-1.1, gain 0.97-1.02), blur sigma 0.9,
    colour-channel misregistration (red and blue shifted 0.15-0.4 px in opposite directions
    along the slow-scan axis), sensor noise sd 2.0. Under this model optical densities do not
    add, so neither cleaner is tested on its own mixing assumption.

Stamp geometries: ``round`` (the M1a ring stamp) and ``rect`` (rotated rectangular box stamp
with a double border and text-like rows). Unstamped scenarios use geometry ``none``.

Seed groups are fixed here *before* any tuning: development seeds may be inspected and used to
set the experimental cleaner's parameters; held-out seeds are run once on frozen configurations.

Ground truth (masks, scenario, parameters) is for evaluation only. Cleaners never receive it.
"""

from __future__ import annotations

import cv2
import numpy as np

from osv.datasets.synthetic_fixtures import (
    _MASK_THRESHOLD,
    _SS,
    Fixture,
    _downsample,
    _jitter_colour,
    _pen_alpha,
    _smooth_noise,
    _stamp_alpha,
)

MATRIX_VERSION = "0.1.0"
MIXTURES = ("multiplicative", "alpha_over_scan")
GEOMETRIES = ("round", "rect")
DEV_SEEDS = tuple(range(0, 10))
HELDOUT_SEEDS = tuple(range(100, 110))

_BLACK = (32, 32, 38)
_BLUE = (52, 72, 192)  # one nominal blue for both pen and stamp in the same-colour cases
_RED = (200, 42, 52)
_NAVY = (34, 40, 118)  # darker, less saturated blue
_CYAN_BLUE = (38, 112, 206)  # lighter, more saturated blue
_VIOLET = (128, 60, 176)
_MARK = (24, 24, 28)

# group = difficulty family reported in the comparison tables.
SCENARIOS: dict[str, dict] = {
    "red_stamp_black_pen": dict(pen=_BLACK, stamp=_RED, group="different_colour"),
    "blue_stamp_black_pen": dict(pen=_BLACK, stamp=_BLUE, group="different_colour"),
    # Similar but distinct blues, in both role assignments (which ink is more saturated).
    "cyan_stamp_navy_pen": dict(pen=_NAVY, stamp=_CYAN_BLUE, group="similar_colour"),
    "navy_stamp_cyan_pen": dict(pen=_CYAN_BLUE, stamp=_NAVY, group="similar_colour"),
    # Same nominal colour (independent +-6 jitter per ink).
    "blue_stamp_blue_pen": dict(pen=_BLUE, stamp=_BLUE, group="same_colour"),
    "black_stamp_black_pen": dict(pen=_BLACK, stamp=_BLACK, group="same_colour"),
    "blue_same_unrelated_mark": dict(pen=_BLUE, stamp=_BLUE, mark="unrelated", group="same_colour_with_mark"),
    "blue_same_touching_mark": dict(pen=_BLUE, stamp=_BLUE, mark="touching", group="same_colour_with_mark"),
    # Added before the held-out run, after the dev check of the M1a.1 expected-failure case: a
    # large neutral blob touching handwriting (like that case). Not used to tune any cleaner.
    "blue_same_large_touching_mark": dict(pen=_BLUE, stamp=_BLUE, mark="touching_large",
                                          group="same_colour_with_mark"),
    "faint_violet_jpeg": dict(pen=_BLACK, stamp=_VIOLET, strength=0.4, jpeg_quality=60, group="faint_compressed"),
    "black_pen_no_stamp": dict(pen=_BLACK, stamp=None, group="unstamped"),
    "blue_pen_no_stamp": dict(pen=_BLUE, stamp=None, group="unstamped"),
}


def matrix_cases(seeds) -> list[tuple[str, str, str, int]]:
    """All (scenario, mixture, geometry, seed) tuples for a seed group."""
    out = []
    for scen, spec in SCENARIOS.items():
        geoms = GEOMETRIES if spec["stamp"] is not None else ("none",)
        for mix in MIXTURES:
            for geo in geoms:
                for s in seeds:
                    out.append((scen, mix, geo, int(s)))
    return out


def _rect_stamp_alpha(rng: np.random.Generator, h: int, w: int, pen_alpha: np.ndarray) -> np.ndarray:
    """Rotated rectangular box stamp: double border, a bold title row and text-like dash rows."""
    pen_pixels = np.flatnonzero(pen_alpha >= _MASK_THRESHOLD)
    anchor = int(rng.choice(pen_pixels)) if pen_pixels.size else (h // 2) * w + w // 2
    ay, ax = divmod(anchor, w)
    bw, bh = rng.uniform(0.32, 0.40) * w, rng.uniform(0.40, 0.50) * h
    cx = ax + rng.uniform(-0.3, 0.3) * bw
    cy = ay + rng.uniform(-0.3, 0.3) * bh
    angle = rng.uniform(-10.0, 10.0)

    canvas = np.zeros((h * _SS, w * _SS), np.uint8)

    def rect(x0, y0, x1, y1, thick):
        p1 = (int(round(x0 * _SS)), int(round(y0 * _SS)))
        p2 = (int(round(x1 * _SS)), int(round(y1 * _SS)))
        cv2.rectangle(canvas, p1, p2, 255, -1 if thick < 0 else max(1, int(round(thick * _SS))), cv2.LINE_AA)

    x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    rect(x0, y0, x1, y1, rng.uniform(2.5, 3.5))
    inset = rng.uniform(4.0, 5.5)
    rect(x0 + inset, y0 + inset, x1 - inset, y1 - inset, 1.4)
    # Title row (bold blocks) and 2-3 rows of thinner dashes.
    yy = y0 + inset + 8
    x = x0 + inset + 6
    while x < x1 - inset - 10:
        d = rng.uniform(4, 9)
        rect(x, yy - 3, min(x + d, x1 - inset - 6), yy + 3, -1)
        x += d + rng.uniform(2.0, 3.5)
    n_rows = int(rng.integers(2, 4))
    for r in range(n_rows):
        yy = y0 + inset + 20 + r * (bh - 2 * inset - 24) / max(1, n_rows)
        x = x0 + inset + 6
        while x < x1 - inset - 10:
            d = rng.uniform(3, 7)
            rect(x, yy - 1.6, min(x + d, x1 - inset - 6), yy + 1.6, -1)
            x += d + rng.uniform(1.5, 3.0)

    m = cv2.getRotationMatrix2D((cx * _SS, cy * _SS), angle, 1.0)
    canvas = cv2.warpAffine(canvas, m, (w * _SS, h * _SS), flags=cv2.INTER_LINEAR, borderValue=0)
    alpha = _downsample(canvas, h, w)
    texture = 0.65 + 0.35 * _smooth_noise(rng, h, w, 6)
    dropout = (_smooth_noise(rng, h, w, 3) > 0.12).astype(np.float32)
    return np.clip(alpha * texture * dropout, 0.0, 1.0)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _mark_alpha(rng: np.random.Generator, kind: str, h: int, w: int,
                pen_alpha: np.ndarray, stamp_alpha: np.ndarray) -> tuple[np.ndarray, dict]:
    """A neutral (black) mark: ``unrelated`` touches no ink; ``touching`` crosses handwriting
    away from the stamp. Placement uses the drawn alphas (generation, not inference)."""
    canvas = np.zeros((h * _SS, w * _SS), np.uint8)
    if kind == "unrelated":
        busy = _dilate((pen_alpha > 0) | (stamp_alpha > 0), 6)
        for _ in range(4):
            mh, mw = int(rng.integers(24, 37)), int(rng.integers(50, 81))
            corners = [(3, 3), (3, w - mw - 3), (h - mh - 3, 3), (h - mh - 3, w - mw - 3)]
            for i in rng.permutation(4):
                y, x = corners[i]
                if not busy[y:y + mh, x:x + mw].any():
                    canvas[y * _SS:(y + mh) * _SS, x * _SS:(x + mw) * _SS] = 255
                    return _downsample(canvas, h, w), {"mark": "unrelated", "box_yxhw": [y, x, mh, mw]}
        raise RuntimeError("no free corner for an unrelated mark")
    if kind in ("touching", "touching_large"):
        large = kind == "touching_large"
        stamp_zone = _dilate(stamp_alpha > 0, 40 if large else 14)
        cand = np.flatnonzero((pen_alpha >= _MASK_THRESHOLD) & ~stamp_zone)
        for _ in range(50):
            if cand.size == 0:
                break
            py, px = divmod(int(rng.choice(cand)), w)
            if large:
                length, thick, ang = rng.uniform(60, 90), rng.uniform(26, 38), rng.uniform(0, 180)
            else:
                length, thick, ang = rng.uniform(26, 44), rng.uniform(7, 11), rng.uniform(0, 180)
            c = np.zeros_like(canvas)
            cv2.ellipse(c, (int(px * _SS), int(py * _SS)), (int(length / 2 * _SS), int(thick / 2 * _SS)),
                        ang, 0, 360, 255, -1, cv2.LINE_AA)
            a = _downsample(c, h, w)
            if not ((a > 0) & _dilate(stamp_alpha > 0, 2)).any():
                return a, {"mark": kind, "centre_yx": [py, px], "length": round(length, 2),
                           "thickness": round(thick, 2), "angle_deg": round(ang, 2)}
        raise RuntimeError(f"could not place a {kind} mark away from the stamp")
    raise ValueError(kind)


def _shift(channel: np.ndarray, dy: float) -> np.ndarray:
    m = np.float32([[1, 0, 0], [0, 1, dy]])
    return cv2.warpAffine(channel, m, (channel.shape[1], channel.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def make_case(scenario: str, mixture: str, geometry: str, seed: int, size: tuple[int, int] = (240, 480)) -> Fixture:
    """Generate one matrix case deterministically. Same seed and geometry give the same pen,
    stamp shape and mark across scenarios and mixtures (paired comparisons)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    if mixture not in MIXTURES:
        raise ValueError(f"unknown mixture {mixture!r}")
    spec = SCENARIOS[scenario]
    stamped = spec["stamp"] is not None
    if stamped and geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}")
    if not stamped and geometry != "none":
        raise ValueError("unstamped scenarios use geometry 'none'")
    h, w = size
    geo_id = {"none": 0, "round": 1, "rect": 2}[geometry]
    shape_rng = np.random.default_rng([int(seed), geo_id, 104729])
    look_rng = np.random.default_rng([int(seed), list(SCENARIOS).index(scenario), MIXTURES.index(mixture), 15485863])

    pen_alpha = _pen_alpha(shape_rng, h, w)
    stamp_alpha = np.zeros((h, w), np.float32)
    if stamped:
        stamp_alpha = (_stamp_alpha if geometry == "round" else _rect_stamp_alpha)(shape_rng, h, w, pen_alpha)
    mark_alpha = np.zeros((h, w), np.float32)
    mark_meta = None
    if spec.get("mark"):
        mark_alpha, mark_meta = _mark_alpha(shape_rng, spec["mark"], h, w, pen_alpha, stamp_alpha)

    pen_rgb = _jitter_colour(look_rng, spec["pen"])
    stamp_rgb = _jitter_colour(look_rng, spec["stamp"]) if stamped else None
    mark_rgb = _jitter_colour(look_rng, _MARK, 3)
    strength = float(spec.get("strength", 1.0))
    paper_rgb = np.array(_jitter_colour(look_rng, (243, 240, 232), 4), np.float32)
    illum = 1.0 + 0.025 * (_smooth_noise(look_rng, h, w, 60) - 0.5)
    meta: dict = {
        "matrix_version": MATRIX_VERSION, "scenario": scenario, "group": spec["group"], "mixture": mixture,
        "geometry": geometry, "seed": int(seed), "size": [h, w], "pen_rgb": list(pen_rgb),
        "stamp_rgb": list(stamp_rgb) if stamp_rgb else None, "stamp_strength": strength,
        "paper_rgb": [float(v) for v in paper_rgb], "mark": mark_meta,
        "jpeg_quality": spec.get("jpeg_quality"),
    }

    if mixture == "multiplicative":
        img = paper_rgb[None, None, :] * illum[..., None]

        def transmit(rgb):
            return np.clip(np.array(rgb, np.float32) / 255.0, 0.02, 1.0)

        img = img * transmit(pen_rgb)[None, None, :] ** pen_alpha[..., None]
        img = img * transmit(mark_rgb)[None, None, :] ** mark_alpha[..., None]
        if stamped:
            img = img * transmit(stamp_rgb)[None, None, :] ** (stamp_alpha * strength)[..., None]
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        img = img + look_rng.normal(0.0, 1.5, img.shape).astype(np.float32)
        meta.update(blur_sigma=0.6, noise_sd=1.5,
                    model="I = paper * prod T_k ** alpha_k (linear values; optical densities add)")
    else:
        fibre = 3.0 * (_smooth_noise(look_rng, h, w, 2) - 0.5)
        base = (paper_rgb[None, None, :] * illum[..., None] + fibre[..., None]) / 255.0
        order = "stamp_over_pen" if look_rng.random() < 0.5 else "pen_over_stamp"
        pen_op = float(look_rng.uniform(0.92, 0.98))
        stamp_op = float(look_rng.uniform(0.70, 0.90))
        mark_op = 0.95

        def over(under, rgb, a):
            col = np.array(rgb, np.float32)[None, None, :] / 255.0
            return under * (1.0 - a[..., None]) + col * a[..., None]

        layers = [("pen", pen_rgb, pen_alpha * pen_op), ("mark", mark_rgb, mark_alpha * mark_op)]
        if stamped:
            st = ("stamp", stamp_rgb, stamp_alpha * stamp_op * strength)
            layers = layers + [st] if order == "stamp_over_pen" else [st] + layers
        for _, rgb, a in layers:
            base = over(base, rgb, a)
        gamma = float(look_rng.uniform(0.9, 1.1))
        gain = float(look_rng.uniform(0.97, 1.02))
        img = 255.0 * gain * np.clip(base, 0.0, 1.0) ** gamma
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.9)
        shift = float(look_rng.uniform(0.15, 0.4))
        img[..., 0] = _shift(img[..., 0], shift)
        img[..., 2] = _shift(img[..., 2], -shift)
        img = img + look_rng.normal(0.0, 2.0, img.shape).astype(np.float32)
        meta.update(layer_order=order if stamped else "pen_only", pen_opacity=round(pen_op, 4),
                    stamp_opacity=round(stamp_op, 4) if stamped else None, mark_opacity=mark_op,
                    tone_gamma=round(gamma, 4), tone_gain=round(gain, 4), blur_sigma=0.9,
                    channel_misregistration_px=round(shift, 4), noise_sd=2.0, paper_fibre_amplitude=3.0,
                    model="over operator in gamma-encoded sRGB; later layer hides earlier ones")

    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    q = spec.get("jpeg_quality")
    if q is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    pen_mask = pen_alpha >= _MASK_THRESHOLD
    stamp_mask = stamp_alpha >= _MASK_THRESHOLD
    return Fixture(
        image=np.ascontiguousarray(img), pen_mask=pen_mask, stamp_mask=stamp_mask,
        overlap_mask=pen_mask & stamp_mask, meta=meta, pen_alpha=pen_alpha.astype(np.float32),
        stamp_alpha=stamp_alpha.astype(np.float32), mark_mask=mark_alpha >= _MASK_THRESHOLD,
    )
