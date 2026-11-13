"""One support rule for every pairwise descriptor comparison (M2a.2).

Every distance in a case report, query-to-reference or reference-to-reference, is computed by
:func:`compare_pair` under the same rule:

    excluded = padding of either image  |  the query's exclusion mask (masked rows only)

Both images are described on the same descriptor support (the footprint rule of
:mod:`osv.verification.descriptor`), so a distance and the reference spread it is shown against
use the same rule. With no query exclusion the masked rule reduces exactly to the all-pixel rule.

Minimum retained evidence. A side whose edge evidence under the pair's support is less than
``MIN_RETAINED_EVIDENCE_SHARE`` of its own evidence (that image alone, padding excluded) gives no
distance: the status says which side and why. The share's denominator reads the whole image,
including pixels the query's mask excludes, so excluded residue can only make the rule stricter.
The value is provisional, chosen before calibration data exist; it is not an authenticity
threshold.
"""

from __future__ import annotations

import numpy as np

from osv.preprocessing.normalise import NormalisedImage
from osv.verification.descriptor import DescriptorResult, GridOrientationDescriptor

MIN_RETAINED_EVIDENCE_SHARE = 0.25
RETAINED_EVIDENCE_NOTE = ("provisional: a side keeping less than this share of its own edge evidence gives no "
                          "distance; chosen before validation data exist, not a calibrated threshold")
PAIR_SUPPORT_RULE = ("both images described on one support: canvas padding of either image plus (masked rows) "
                     "the query's exclusion mask, then the descriptor footprint rule")


def pair_exclusion(a: NormalisedImage, b: NormalisedImage, extra: np.ndarray | None = None) -> np.ndarray:
    if a.darkness.shape != b.darkness.shape:
        raise ValueError("canvases differ in size")
    excl = a.padding_mask | b.padding_mask
    if extra is not None:
        if extra.shape != excl.shape or extra.dtype != bool:
            raise ValueError("extra exclusion must be a boolean canvas mask")
        excl = excl | extra
    return excl


def describe_normalised(desc: GridOrientationDescriptor, n: NormalisedImage,
                        extra: np.ndarray | None = None) -> DescriptorResult:
    """Describe one normalised image with its own padding (and ``extra``) excluded."""
    excl = n.padding_mask if extra is None else (n.padding_mask | extra)
    return desc.describe(n.darkness, excl)


def _share(retained: float, own: float) -> float | None:
    return None if own <= 0 else retained / own


def compare_pair(desc: GridOrientationDescriptor, a: NormalisedImage, b: NormalisedImage,
                 extra: np.ndarray | None = None, roles: tuple[str, str] = ("query", "reference")) -> dict:
    """Distance between ``a`` and ``b`` on their common support, or an explicit undefined status."""
    excl = pair_exclusion(a, b, extra)
    ra, rb = desc.describe(a.darkness, excl), desc.describe(b.darkness, excl)
    sa = _share(ra.edge_evidence, describe_normalised(desc, a).edge_evidence)
    sb = _share(rb.edge_evidence, describe_normalised(desc, b).edge_evidence)
    if ra.vector is None:
        status = f"insufficient_{roles[0]}_edge_evidence"
    elif rb.vector is None:
        status = f"insufficient_{roles[1]}_edge_evidence"
    elif sa is None or sa < MIN_RETAINED_EVIDENCE_SHARE:
        status = f"insufficient_{roles[0]}_retained_evidence"
    elif sb is None or sb < MIN_RETAINED_EVIDENCE_SHARE:
        status = f"insufficient_{roles[1]}_retained_evidence"
    else:
        status = "defined"
    return {
        "status": status,
        "distance": desc.distance(ra.vector, rb.vector) if status == "defined" else None,
        "support_pixels": ra.support_pixels,
        f"{roles[0]}_retained_evidence_share": None if sa is None else round(sa, 6),
        f"{roles[1]}_retained_evidence_share": None if sb is None else round(sb, 6),
        f"{roles[1]}_edge_share_outside_support": (None if rb.total_edge_magnitude <= 0
                                                   else round(1.0 - rb.retained_edge_magnitude / rb.total_edge_magnitude, 6)),
    }


def summarise(rows: list[dict]) -> dict:
    vals = [r["distance"] for r in rows if r["distance"] is not None]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    if not vals:
        return {"n_pairs": len(rows), "n_defined": 0, "min": None, "median": None, "max": None, "statuses": statuses}
    return {"n_pairs": len(rows), "n_defined": len(vals), "min": round(min(vals), 6),
            "median": round(float(np.median(vals)), 6), "max": round(max(vals), 6), "statuses": statuses}
