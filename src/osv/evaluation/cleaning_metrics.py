"""Cleaning metrics against synthetic ground truth (evaluation stage E0).

Evaluation-only: these functions read ground-truth masks *after* a cleaner has run. The
cleaners never receive masks, fixture kinds, scenario names, filenames or generator
parameters. ``cleaning_metrics`` keeps the exact M1a output (moved here from
``osv.datasets.synthetic_fixtures`` in M1b; the old import path still works).
"""

from __future__ import annotations

import cv2
import numpy as np


def stamp_support_with_margin(fixture, margin: int = 2) -> np.ndarray:
    """Pixels touched by any stamp ink (coverage > 0), dilated by ``margin`` for blur spread."""
    support = (fixture.stamp_alpha > 0).astype(np.uint8)
    if margin > 0:
        support = cv2.dilate(support, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return support.astype(bool)


def cleaning_metrics(fixture, intervention_mask: np.ndarray, change_map: np.ndarray,
                     ambiguous_mask: np.ndarray) -> dict:
    """Compare a cleaner's output with the fixture's ground truth (M1a metric set)."""
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


def case_metrics(fixture, intervention_mask: np.ndarray, change_map: np.ndarray,
                 ambiguous_mask: np.ndarray) -> dict:
    """M1a metrics plus pixel counts used by the M1b comparison.

    Pen changes are counted from the actual change map (not only the intervention mask), so
    any change to handwriting is visible even if the two masks were ever to disagree.
    """
    m = cleaning_metrics(fixture, intervention_mask, change_map, ambiguous_mask)
    changed = change_map > 0
    pen = fixture.pen_mask
    overlap = fixture.overlap_mask
    mark = getattr(fixture, "mark_mask", None)
    m.update({
        "changed_pixels": int(changed.sum()),
        "pen_changed_pixels": int((changed & pen).sum()),
        "pen_changed_outside_overlap_pixels": int((changed & pen & ~fixture.stamp_mask).sum()),
        "pen_changed_in_overlap_pixels": int((changed & overlap).sum()),
        "mark_pixels": 0 if mark is None else int(mark.sum()),
        "mark_changed_pixels": 0 if mark is None else int((changed & mark).sum()),
    })
    return m
