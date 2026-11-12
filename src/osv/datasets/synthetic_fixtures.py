"""Deterministic synthetic fixtures for testing the stamp cleaner (evaluation stage E0).

Every fixture is generated from a kind name and a seed. Pen strokes are random smooth
curves (fictional, not anyone's signature); stamps are rings with glyph-like texture.
Inks are composited by a multiplicative (darkening) mixture, never as an opaque overlay.

Ground-truth masks are returned for tests and metrics only. They must never be passed
to the cleaner; the cleaning API has no parameter that could accept them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

GENERATOR_VERSION = "0.1.0"

KINDS = (
    "no_stamp",
    "red_stamp_black_pen",
    "blue_stamp_black_pen",
    "blue_stamp_blue_pen",
    "black_stamp_black_pen",
    "gray_scan",
    "faint_stamp_jpeg",
)

# Ink colours as RGB transmittance targets (what the ink looks like at full coverage on white).
_BLACK_PEN = (32, 32, 38)
_BLUE_PEN = (30, 48, 150)
_RED_STAMP = (200, 42, 52)
_BLUE_STAMP = (52, 72, 192)
_BLACK_STAMP = (48, 48, 52)
_VIOLET_STAMP = (128, 60, 176)

_SPECS: dict[str, dict] = {
    "no_stamp": dict(pen=_BLACK_PEN, stamp=None),
    "red_stamp_black_pen": dict(pen=_BLACK_PEN, stamp=_RED_STAMP),
    "blue_stamp_black_pen": dict(pen=_BLACK_PEN, stamp=_BLUE_STAMP),
    "blue_stamp_blue_pen": dict(pen=_BLUE_PEN, stamp=_BLUE_STAMP),
    "black_stamp_black_pen": dict(pen=_BLACK_PEN, stamp=_BLACK_STAMP),
    # A red stamp scanned in grayscale: the stamp is present but carries no colour.
    "gray_scan": dict(pen=_BLACK_PEN, stamp=_RED_STAMP, grayscale=True),
    "faint_stamp_jpeg": dict(pen=_BLACK_PEN, stamp=_VIOLET_STAMP, strength=0.4, jpeg_quality=60),
    # Extra kinds: generated on request, not part of the seven measured KINDS.
    # A blue signature with no stamp at all (colour alone cannot tell it from a blue stamp).
    "blue_pen_no_stamp": dict(pen=_BLUE_PEN, stamp=None),
}
EXTRA_KINDS = ("blue_pen_no_stamp",)

_SS = 4  # supersampling factor for anti-aliased drawing
_MASK_THRESHOLD = 0.5  # coverage at which a pixel counts as pen/stamp in the ground truth


@dataclass(frozen=True)
class Fixture:
    """A synthetic stamped-signature image with exact ground truth.

    ``image`` is RGB uint8 (H, W, 3). Masks are boolean (H, W). ``pen_alpha`` and
    ``stamp_alpha`` are the drawn coverages in [0, 1] before blur, noise and JPEG;
    the masks threshold them at 0.5. ``stamp_alpha`` is coverage before the faint-ink
    strength factor, so a faint stamp still has its full shape in the mask.
    """

    image: np.ndarray
    pen_mask: np.ndarray
    stamp_mask: np.ndarray
    overlap_mask: np.ndarray
    meta: dict = field(default_factory=dict)
    pen_alpha: np.ndarray | None = None
    stamp_alpha: np.ndarray | None = None


def _jitter_colour(rng: np.random.Generator, rgb: tuple[int, int, int], amount: int = 6) -> tuple[int, ...]:
    return tuple(int(np.clip(c + rng.integers(-amount, amount + 1), 1, 254)) for c in rgb)


def _smooth_noise(rng: np.random.Generator, h: int, w: int, cell: int) -> np.ndarray:
    small = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _downsample(canvas: np.ndarray, h: int, w: int) -> np.ndarray:
    alpha = cv2.resize(canvas.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(alpha, 0.0, 1.0)


def _draw_curve(canvas: np.ndarray, xs: np.ndarray, ys: np.ndarray, radii: np.ndarray) -> None:
    for x, y, r in zip(xs, ys, radii):
        cv2.circle(canvas, (int(round(x * _SS)), int(round(y * _SS))), max(1, int(round(r * _SS))), 255, -1)


def _pen_alpha(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Fictional cursive-like strokes: a looping main stroke, a flourish and an accent."""
    canvas = np.zeros((h * _SS, w * _SS), np.uint8)

    x0, x1 = rng.uniform(0.08, 0.18) * w, rng.uniform(0.78, 0.92) * w
    base = rng.uniform(0.45, 0.58) * h
    t = np.linspace(0.0, 1.0, 4000)
    freq = rng.uniform(5.0, 8.0)
    loop_amp = rng.uniform(0.045, 0.07) * w
    xs = x0 + (x1 - x0) * t + loop_amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    ys = (
        base
        + rng.uniform(0.12, 0.2) * h * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        + rng.uniform(-0.08, 0.08) * h * np.sin(np.pi * t)
    )
    radii = rng.uniform(1.1, 1.8) + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 3) * t + rng.uniform(0, 6)))
    _draw_curve(canvas, xs, ys, radii)

    # Underline flourish.
    t2 = np.linspace(0.0, 1.0, 1500)
    fx0, fx1 = rng.uniform(0.15, 0.35) * w, rng.uniform(0.7, 0.95) * w
    fy = base + rng.uniform(0.2, 0.28) * h
    xs2 = fx0 + (fx1 - fx0) * t2
    ys2 = fy + rng.uniform(-0.04, 0.04) * h * np.sin(np.pi * t2 * rng.uniform(1, 2)) - 0.05 * h * t2
    radii2 = 0.9 + 0.8 * np.sin(np.pi * t2)
    _draw_curve(canvas, xs2, ys2, radii2)

    # Short accent stroke above the line.
    t3 = np.linspace(0.0, 1.0, 300)
    ax, ay = rng.uniform(0.3, 0.7) * w, base - rng.uniform(0.25, 0.32) * h
    xs3 = ax + rng.uniform(10, 18) * t3
    ys3 = ay - rng.uniform(4, 9) * t3
    _draw_curve(canvas, xs3, ys3, np.full_like(t3, 1.3))

    return _downsample(canvas, h, w)


def _stamp_alpha(rng: np.random.Generator, h: int, w: int, pen_alpha: np.ndarray) -> np.ndarray:
    """Round stamp with two rings, glyph-like marks and uneven ink, overlapping the pen."""
    pen_pixels = np.flatnonzero(pen_alpha >= _MASK_THRESHOLD)
    anchor = int(rng.choice(pen_pixels)) if pen_pixels.size else (h // 2) * w + w // 2
    ay, ax = divmod(anchor, w)
    radius = rng.uniform(0.27, 0.34) * h
    ang = rng.uniform(0, 2 * np.pi)
    off = rng.uniform(0.2, 0.6) * radius
    cx, cy = ax + off * np.cos(ang), ay + off * np.sin(ang)

    canvas = np.zeros((h * _SS, w * _SS), np.uint8)
    c = (int(round(cx * _SS)), int(round(cy * _SS)))
    outer_t = rng.uniform(3.0, 4.5)
    cv2.circle(canvas, c, int(round((radius - outer_t / 2) * _SS)), 255, int(round(outer_t * _SS)), cv2.LINE_AA)
    inner_r = radius - rng.uniform(13, 16)
    cv2.circle(canvas, c, int(round(inner_r * _SS)), 255, int(round(1.8 * _SS)), cv2.LINE_AA)

    # Glyph-like marks between the rings (text-like texture, not real letters).
    band = (inner_r + radius - outer_t) / 2
    n_glyphs = int(rng.integers(20, 30))
    for k in range(n_glyphs):
        theta = 2 * np.pi * k / n_glyphs + rng.uniform(-0.03, 0.03)
        tx, ty = -np.sin(theta), np.cos(theta)  # tangent
        nx, ny = np.cos(theta), np.sin(theta)  # normal (outward)
        px, py = cx + band * nx, cy + band * ny
        pts = []
        for _ in range(int(rng.integers(2, 4))):
            u, v = rng.uniform(-2.5, 2.5), rng.uniform(-3.5, 3.5)
            pts.append((px + u * tx + v * nx, py + u * ty + v * ny))
        arr = np.round(np.array(pts) * _SS).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr], False, 255, int(round(1.4 * _SS)), cv2.LINE_AA)

    # Centre: a few rows of dashes, like printed text.
    rows = int(rng.integers(2, 4))
    for r in range(rows):
        yy = cy + (r - (rows - 1) / 2) * 10
        half = inner_r * 0.7 * np.sqrt(max(0.0, 1 - ((yy - cy) / inner_r) ** 2))
        x = cx - half
        while x < cx + half - 3:
            dash = rng.uniform(3, 8)
            p1 = (int(round(x * _SS)), int(round((yy - 2.5) * _SS)))
            p2 = (int(round(min(x + dash, cx + half) * _SS)), int(round((yy + 2.5) * _SS)))
            cv2.rectangle(canvas, p1, p2, 255, -1)
            x += dash + rng.uniform(1.5, 3.5)

    alpha = _downsample(canvas, h, w)
    # Uneven inking: smooth variation plus small dropouts.
    texture = 0.65 + 0.35 * _smooth_noise(rng, h, w, 6)
    dropout = (_smooth_noise(rng, h, w, 3) > 0.12).astype(np.float32)
    return np.clip(alpha * texture * dropout, 0.0, 1.0)


def make_fixture(kind: str, seed: int, size: tuple[int, int] = (240, 480)) -> Fixture:
    """Generate fixture ``kind`` deterministically from ``seed``. ``size`` is (height, width)."""
    if kind not in _SPECS:
        raise ValueError(f"unknown fixture kind {kind!r}; expected one of {KINDS + EXTRA_KINDS}")
    spec = _SPECS[kind]
    h, w = size
    # One stream per (kind, seed) so kinds with the same seed share pen geometry only by design.
    rng = np.random.default_rng([seed, 7919])

    pen_alpha = _pen_alpha(rng, h, w)
    pen_rgb = _jitter_colour(rng, spec["pen"])
    stamp_rgb = None
    stamp_alpha = np.zeros((h, w), np.float32)
    if spec["stamp"] is not None:
        stamp_alpha = _stamp_alpha(rng, h, w, pen_alpha)
        stamp_rgb = _jitter_colour(rng, spec["stamp"])
    strength = float(spec.get("strength", 1.0))

    paper_rgb = np.array(_jitter_colour(rng, (243, 240, 232), 4), np.float32)
    illum = 1.0 + 0.025 * (_smooth_noise(rng, h, w, 60) - 0.5)
    img = paper_rgb[None, None, :] * illum[..., None]

    def transmit(rgb: tuple[int, ...]) -> np.ndarray:
        return np.clip(np.array(rgb, np.float32) / 255.0, 0.02, 1.0)

    img = img * transmit(pen_rgb)[None, None, :] ** pen_alpha[..., None]
    if stamp_rgb is not None:
        img = img * transmit(stamp_rgb)[None, None, :] ** (stamp_alpha * strength)[..., None]

    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    img = img + rng.normal(0.0, 1.5, img.shape).astype(np.float32)
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    grayscale = bool(spec.get("grayscale", False))
    if grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.repeat(gray[..., None], 3, axis=2)

    jpeg_quality = spec.get("jpeg_quality")
    if jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    pen_mask = pen_alpha >= _MASK_THRESHOLD
    stamp_mask = stamp_alpha >= _MASK_THRESHOLD
    meta = {
        "kind": kind,
        "seed": int(seed),
        "size": [int(h), int(w)],
        "generator_version": GENERATOR_VERSION,
        "pen_rgb": list(pen_rgb),
        "stamp_rgb": list(stamp_rgb) if stamp_rgb else None,
        "stamp_strength": strength,
        "grayscale": grayscale,
        "jpeg_quality": jpeg_quality,
        "mixture": "multiplicative transmittance (darkening)",
        "blur_sigma": 0.6,
        "noise_sd": 1.5,
    }
    return Fixture(
        image=np.ascontiguousarray(img),
        pen_mask=pen_mask,
        stamp_mask=stamp_mask,
        overlap_mask=pen_mask & stamp_mask,
        meta=meta,
        pen_alpha=pen_alpha.astype(np.float32),
        stamp_alpha=stamp_alpha.astype(np.float32),
    )


def stamp_support_with_margin(fixture: Fixture, margin: int = 2) -> np.ndarray:
    """Pixels touched by any stamp ink (coverage > 0), dilated by ``margin`` for blur spread."""
    support = (fixture.stamp_alpha > 0).astype(np.uint8)
    if margin > 0:
        support = cv2.dilate(support, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return support.astype(bool)


def cleaning_metrics(fixture: Fixture, intervention_mask: np.ndarray, change_map: np.ndarray,
                     ambiguous_mask: np.ndarray) -> dict:
    """Evaluation-only: compare a cleaner's output with the fixture's ground truth.

    The cleaner never sees these masks; this function runs after cleaning.
    """
    stamp_only = fixture.stamp_mask & ~fixture.pen_mask
    pen_only = fixture.pen_mask & ~fixture.stamp_mask
    overlap = fixture.overlap_mask
    deleted = intervention_mask.astype(bool)
    changed = change_map > 0
    away = fixture.pen_mask & ~stamp_support_with_margin(fixture)

    def frac(num: np.ndarray, den: np.ndarray) -> float | None:
        d = int(den.sum())
        return None if d == 0 else float((num & den).sum()) / d

    return {
        "stamp_only_pixels": int(stamp_only.sum()),
        "pen_pixels": int(fixture.pen_mask.sum()),
        "overlap_pixels": int(overlap.sum()),
        "stamp_only_removed": frac(deleted, stamp_only),
        # Stamp-only pixels neither deleted nor flagged ambiguous: residue the reviewer is not warned about.
        "stamp_only_unflagged": frac(~deleted & ~ambiguous_mask.astype(bool), stamp_only),
        "pen_loss_outside_overlap": frac(deleted, pen_only),
        "pen_loss_in_overlap": frac(deleted, overlap),
        "pen_loss_total": frac(deleted, fixture.pen_mask),
        "overlap_marked_ambiguous": frac(ambiguous_mask.astype(bool), overlap),
        "pen_changed_away_from_stamp": int((changed & away).sum()),
        "deleted_pixels": int(deleted.sum()),
        "deleted_outside_gt_ink": int((deleted & ~fixture.stamp_mask & ~fixture.pen_mask).sum()),
    }
