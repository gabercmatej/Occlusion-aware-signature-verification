"""Cleaner behaviour on synthetic fixtures (E0: code correctness, not real-world quality).

Ground-truth masks are used here, after cleaning, to measure. They never enter the cleaner.
Floors below are regression floors set from the first measured run (see docs/SESSION_LOG.md),
not quality claims.
"""

import inspect

import numpy as np
import pytest

from osv.datasets.synthetic_fixtures import KINDS, cleaning_metrics, make_fixture, stamp_support_with_margin
from osv.preprocessing.stamp_removal import remove_stamp
from osv.segmentation.stamp_estimate import estimate_stamp

SEEDS = range(10)

# Regression floors set from the first measured run (2026-11-11, seeds 0-9, cleaner 0.1.0):
# measured stamp-only removal min/mean: red 0.826/0.899, blue 0.745/0.863, faint 0.223/0.510;
# measured pen loss outside the overlap: 0 except blue_stamp_black_pen seed 7, 1 pixel (0.000186),
# which lies within the stamp's blur margin.
# Floors sit just below the measured minimum (ceiling just above the measured maximum). They
# guard against regressions and are not quality claims about real scans.
FLOOR_STAMP_ONLY_REMOVED = {"red_stamp_black_pen": 0.80, "blue_stamp_black_pen": 0.72, "faint_stamp_jpeg": 0.20}
CEIL_PEN_LOSS_OUTSIDE_OVERLAP = 0.001


def run(image, region=None):
    est = estimate_stamp(image, stamp_region=region)
    return est, remove_stamp(image, est)


def gt_bbox(fixture):
    """Simulates a reviewer drawing a rectangle around the visible stamp (a human input)."""
    ys, xs = np.nonzero(fixture.stamp_mask)
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


# ---------------------------------------------------------------- invariants on every fixture

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_invariants(kind, seed):
    fx = make_fixture(kind, seed)
    before = fx.image.copy()
    est, res = run(fx.image)
    h, w = fx.image.shape[:2]
    assert np.array_equal(fx.image, before), "input array was modified"
    for m in (est.stamp_mask, est.ink_mask, est.ambiguous_mask, res.intervention_mask):
        assert m.shape == (h, w) and m.dtype == bool
    assert res.cleaned.shape == fx.image.shape and res.cleaned.dtype == np.uint8
    assert res.change_map.shape == (h, w)
    changed = res.change_map > 0
    assert not (changed & ~res.intervention_mask).any(), "change outside intervention mask"
    assert not (res.intervention_mask & ~est.stamp_mask).any(), "intervention outside stamp mask"
    assert not (res.intervention_mask & est.ambiguous_mask).any(), "deleted an ambiguous pixel"
    differs = (res.cleaned != fx.image).any(axis=2)
    assert np.array_equal(differs, changed), "change map does not match actual differences"
    assert res.status in {"no_stamp_detected", "cleaned", "review"}


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic(kind):
    fx = make_fixture(kind, 3)
    _, a = run(fx.image)
    _, b = run(fx.image.copy())
    assert np.array_equal(a.cleaned, b.cleaned)
    assert np.array_equal(a.intervention_mask, b.intervention_mask)
    assert a.status == b.status and a.reasons == b.reasons and a.metrics == b.metrics


def test_cleaner_accepts_no_ground_truth():
    assert list(inspect.signature(estimate_stamp).parameters) == ["image_rgb", "stamp_region"]
    assert list(inspect.signature(remove_stamp).parameters) == ["image_rgb", "estimate"]


# ---------------------------------------------------------------- specific behaviours

@pytest.mark.parametrize("seed", SEEDS)
def test_no_stamp_is_exact_noop(seed):
    fx = make_fixture("no_stamp", seed)
    est, res = run(fx.image)
    assert res.status == "no_stamp_detected"
    assert int((res.change_map > 0).sum()) == 0
    assert np.array_equal(res.cleaned, fx.image)
    assert not res.intervention_mask.any()


@pytest.mark.parametrize("kind", ["red_stamp_black_pen", "blue_stamp_black_pen", "faint_stamp_jpeg"])
@pytest.mark.parametrize("seed", SEEDS)
def test_coloured_stamp_over_black_pen(kind, seed):
    fx = make_fixture(kind, seed)
    est, res = run(fx.image)
    m = cleaning_metrics(fx, res.intervention_mask, res.change_map, est.ambiguous_mask)
    # Exact: pen pixels untouched by any stamp ink (2 px blur margin) are never changed.
    assert m["pen_changed_away_from_stamp"] == 0
    assert m["pen_loss_outside_overlap"] <= CEIL_PEN_LOSS_OUTSIDE_OVERLAP
    assert m["stamp_only_removed"] >= FLOOR_STAMP_ONLY_REMOVED[kind]
    # Exact: every deleted pixel lies within 2 px (blur margin) of drawn stamp ink.
    assert not (res.intervention_mask & ~stamp_support_with_margin(fx, 2)).any()
    assert res.status in {"cleaned", "review"}
    assert res.intervention_mask.any()


@pytest.mark.parametrize("seed", SEEDS)
def test_blue_stamp_blue_pen_guard_preserves_handwriting(seed):
    """The stamp colour also describes the signature: delete nothing, ask for review.

    This shows the guard fires on this synthetic same-hue case; it does not show that
    every real same-colour overlap is detected.
    """
    fx = make_fixture("blue_stamp_blue_pen", seed)
    est, res = run(fx.image)
    m = cleaning_metrics(fx, res.intervention_mask, res.change_map, est.ambiguous_mask)
    assert res.status == "review"
    assert "ink_and_stamp_colour_overlap" in res.reasons
    assert m["pen_loss_total"] == 0.0
    assert int((res.change_map > 0).sum()) == 0
    assert not est.stamp_mask.any()


@pytest.mark.parametrize("kind", ["black_stamp_black_pen", "gray_scan"])
@pytest.mark.parametrize("seed", SEEDS)
def test_known_limitation_neutral_stamp_invisible_without_region(kind, seed):
    """KNOWN LIMITATION: a colour cleaner cannot see a black stamp or any stamp in a
    grayscale scan. The stamp is present (ground truth) yet the status is
    ``no_stamp_detected`` and nothing changes. This status is not a claim of cleanliness."""
    fx = make_fixture(kind, seed)
    assert fx.stamp_mask.sum() > 0
    est, res = run(fx.image)
    assert res.status == "no_stamp_detected"
    assert int((res.change_map > 0).sum()) == 0
    if kind == "gray_scan":
        assert "grayscale_input" in res.reasons


@pytest.mark.parametrize("kind", ["black_stamp_black_pen", "gray_scan"])
@pytest.mark.parametrize("seed", SEEDS)
def test_neutral_stamp_with_reviewer_region_goes_to_review(kind, seed):
    fx = make_fixture(kind, seed)
    region = gt_bbox(fx)
    est, res = run(fx.image, region)
    x, y, w, h = region
    assert res.status == "review"
    assert "stamp_indicated_not_separated" in res.reasons
    assert int((res.change_map > 0).sum()) == 0
    assert est.ambiguous_mask[y:y + h, x:x + w].all()
    assert not est.ambiguous_mask.sum() > w * h


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_region_hint_limits_deletion_to_region(seed):
    fx = make_fixture("red_stamp_black_pen", seed)
    x, y, w, h = gt_bbox(fx)
    region = (x, y, w // 2, h)  # deliberately only half of the stamp
    est, res = run(fx.image, region)
    inside = np.zeros(fx.stamp_mask.shape, bool)
    inside[y:y + h, x:x + w // 2] = True
    assert res.intervention_mask.any()
    assert not (res.intervention_mask & ~inside).any()


@pytest.mark.parametrize("region", [(0, 0, 0, 10), (-5, 0, 10, 10), (500, 0, 10, 10)])
def test_invalid_region_rejected(region):
    fx = make_fixture("no_stamp", 0)
    with pytest.raises(ValueError):
        estimate_stamp(fx.image, stamp_region=region)


def test_rejects_non_rgb_input():
    with pytest.raises(ValueError):
        estimate_stamp(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        estimate_stamp(np.zeros((10, 10, 3), np.float32))


# ---------------------------------------------------------------- M1a.1: unrelated neutral marks
#
# Regression for the coordination-review counterexample: a black mark far from the blue stamp
# and blue signature raised the *image-wide* neutral-ink share above the same-colour guard, so
# blue handwriting in the stamp's hue cluster was deleted. Marks are pasted onto the image
# before cleaning (an ordinary input change); ground truth is read only after processing, to
# check the mark does not touch drawn ink and to measure damage.

# (y0, y1, x0, x1, grey value): the reported case, moved, resized, and a lighter grey.
UNRELATED_MARKS = {
    "reported_top_left": (8, 48, 8, 88, 20),
    "small_top_right": (8, 20, 440, 470, 20),
    "bottom_left": (204, 236, 4, 64, 20),
    "large_top_left": (2, 50, 2, 110, 20),
    "tall_left_edge": (150, 236, 2, 40, 20),
    "grey_bottom_left": (204, 236, 4, 64, 90),
}


def with_mark(image, mark):
    y0, y1, x0, x1, v = mark
    out = image.copy()
    out[y0:y1, x0:x1] = v
    return out


def mark_touches_drawn_ink(fx, mark, margin=4):
    y0, y1, x0, x1, _ = mark
    region = np.zeros(fx.pen_mask.shape, bool)
    region[max(0, y0 - margin):y1 + margin, max(0, x0 - margin):x1 + margin] = True
    return bool((region & ((fx.pen_alpha > 0) | (fx.stamp_alpha > 0))).any())


def test_reported_counterexample_unrelated_black_mark():
    fx = make_fixture("blue_stamp_blue_pen", 0)
    image = with_mark(fx.image, UNRELATED_MARKS["reported_top_left"])
    est, res = run(image)
    assert int(((res.change_map > 0) & fx.pen_mask).sum()) == 0
    assert int((res.change_map > 0).sum()) == 0
    assert np.array_equal(res.cleaned, image)
    assert res.status == "review"
    assert "ink_and_stamp_colour_overlap" in res.reasons
    assert est.action == "review", "the estimator must refuse before deletion, not after"


@pytest.mark.parametrize("kind", ["blue_stamp_blue_pen", "blue_pen_no_stamp"])
@pytest.mark.parametrize("mark_name", list(UNRELATED_MARKS))
@pytest.mark.parametrize("seed", SEEDS)
def test_unrelated_neutral_mark_does_not_authorize_deletion(kind, mark_name, seed):
    fx = make_fixture(kind, seed)
    mark = UNRELATED_MARKS[mark_name]
    est, res = run(with_mark(fx.image, mark))
    # Post-processing precondition: this test is only about marks that touch no drawn ink.
    assert not mark_touches_drawn_ink(fx, mark)
    assert est.action == "review"
    assert res.status == "review"
    assert int((res.change_map > 0).sum()) == 0
    assert not res.intervention_mask.any()


@pytest.mark.parametrize("seed", SEEDS)
def test_blue_signature_without_stamp_is_preserved_for_review(seed):
    """Formalizes the earlier ad hoc check. A blue signature with no stamp is one coloured
    ink with no separate pen-like ink: colour cannot tell it from a stamp, so the cleaner
    deletes nothing and asks for review (a conservative false alarm, not a stamp claim)."""
    fx = make_fixture("blue_pen_no_stamp", seed)
    assert not fx.stamp_mask.any()
    est, res = run(fx.image)
    assert res.status == "review"
    assert "ink_and_stamp_colour_overlap" in res.reasons
    assert int((res.change_map > 0).sum()) == 0
    assert np.array_equal(res.cleaned, fx.image)


@pytest.mark.parametrize("kind", ["red_stamp_black_pen", "blue_stamp_black_pen"])
@pytest.mark.parametrize("seed", [0, 3, 7])
def test_unrelated_mark_does_not_break_coloured_stamp_over_black_pen(kind, seed):
    """The fix must keep the useful case: a black pen crossing a coloured stamp still cleans,
    with or without an unrelated mark elsewhere."""
    fx = make_fixture(kind, seed)
    mark = UNRELATED_MARKS["reported_top_left"]
    est, res = run(with_mark(fx.image, mark))
    assert not mark_touches_drawn_ink(fx, mark)
    m = cleaning_metrics(fx, res.intervention_mask, res.change_map, est.ambiguous_mask)
    assert est.action == "delete"
    assert m["pen_changed_away_from_stamp"] == 0
    assert m["stamp_only_removed"] >= FLOOR_STAMP_ONLY_REMOVED[kind]


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN LIMITATION: a neutral mark that touches same-colour handwriting looks, by colour "
    "and connectivity, like a black pen crossing a coloured stamp. The guard cannot tell "
    "them apart, so pen-coloured strokes can still be deleted. Recorded, not fixed."))
def test_known_limitation_neutral_mark_touching_same_colour_ink():
    fx = make_fixture("blue_stamp_blue_pen", 1)
    mark = (190, 232, 380, 472, 20)
    est, res = run(with_mark(fx.image, mark))
    assert mark_touches_drawn_ink(fx, mark)
    assert int(((res.change_map > 0) & fx.pen_mask).sum()) == 0
