"""Method selection for stamp cleaning.

The method is chosen explicitly by the caller (CLI ``--method``). It is never chosen per image
from ground truth, fixture categories or filenames. ``chroma`` (the M1a/M1a.1 baseline) stays
the default; ``optical-density`` is the M1b experiment.
"""

from __future__ import annotations

import numpy as np

from osv.preprocessing.stamp_removal import CleaningResult, remove_stamp
from osv.segmentation.od_estimate import estimate_stamp_od
from osv.segmentation.stamp_estimate import StampEstimate, estimate_stamp

METHODS = {
    "chroma": estimate_stamp,
    "optical-density": estimate_stamp_od,
}
DEFAULT_METHOD = "chroma"


def clean_image(image_rgb: np.ndarray, method: str = DEFAULT_METHOD,
                stamp_region=None) -> tuple[StampEstimate, CleaningResult]:
    if method not in METHODS:
        raise ValueError(f"unknown cleaning method {method!r}; expected one of {tuple(METHODS)}")
    estimate = METHODS[method](image_rgb, stamp_region=stamp_region)
    return estimate, remove_stamp(image_rgb, estimate)
