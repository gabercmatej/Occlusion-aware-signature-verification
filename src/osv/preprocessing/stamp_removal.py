"""Conservative, deletion-only stamp removal.

Deletable stamp pixels (``stamp_mask & ~ambiguous_mask``) are set to the local paper tone.
Nothing else is changed: no inpainting, no stroke drawing, no use of reference signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from osv.segmentation.stamp_estimate import StampEstimate

REMOVER_NAME = "deletion_to_paper_tone"
REMOVER_VERSION = "0.1.0"
LOW_RELIABLE_SHARE = 0.5  # provisional: kept ink less than half reliable -> review


@dataclass
class CleaningResult:
    cleaned: np.ndarray
    intervention_mask: np.ndarray
    change_map: np.ndarray  # uint8 (H, W): max absolute channel difference, 0 where unchanged
    status: str  # no_stamp_detected | cleaned | review
    reasons: tuple[str, ...]
    metrics: dict = field(default_factory=dict)


def remove_stamp(image_rgb: np.ndarray, estimate: StampEstimate) -> CleaningResult:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be an (H, W, 3) uint8 array")
    if estimate.stamp_mask.shape != image_rgb.shape[:2]:
        raise ValueError("estimate does not match the image size")

    if estimate.action == "delete":
        intervention = estimate.stamp_mask & ~estimate.ambiguous_mask
    else:
        intervention = np.zeros(image_rgb.shape[:2], bool)

    cleaned = image_rgb.copy()
    if intervention.any():
        paper = np.clip(np.round(estimate.paper_rgb), 0, 255).astype(np.uint8)
        cleaned[intervention] = paper[intervention]

    diff = np.abs(cleaned.astype(np.int16) - image_rgb.astype(np.int16)).max(axis=2).astype(np.uint8)
    changed = diff > 0

    ink = estimate.ink_mask
    ink_n = int(ink.sum())
    changed_ink = ink & changed
    ambiguous_ink = ink & estimate.ambiguous_mask & ~changed
    reliable_ink = ink & ~estimate.ambiguous_mask & ~changed

    def share(m: np.ndarray) -> float | None:
        return None if ink_n == 0 else round(float(m.sum()) / ink_n, 6)

    kept = int(reliable_ink.sum() + ambiguous_ink.sum())
    reliable_of_kept = None if kept == 0 else float(reliable_ink.sum()) / kept

    reasons = list(estimate.reasons)
    if estimate.action == "none":
        status = "no_stamp_detected"
    elif estimate.action == "review":
        status = "review"
    else:
        status = "cleaned"
        if not changed.any():
            status = "review"
            reasons.append("stamp_detected_nothing_removable")
        if reliable_of_kept is not None and reliable_of_kept < LOW_RELIABLE_SHARE:
            status = "review"
            reasons.append("low_reliable_ink")

    metrics = {
        "image_pixels": int(changed.size),
        "changed_pixels": int(changed.sum()),
        "intervention_pixels": int(intervention.sum()),
        "stamp_pixels_estimated": int(estimate.stamp_mask.sum()),
        "ambiguous_pixels": int(estimate.ambiguous_mask.sum()),
        "ink_pixels_estimated": ink_n,
        "ink_fraction_reliable": share(reliable_ink),
        "ink_fraction_ambiguous": share(ambiguous_ink),
        "ink_fraction_changed": share(changed_ink),
        "reliable_share_of_kept_ink": None if reliable_of_kept is None else round(reliable_of_kept, 6),
    }
    return CleaningResult(
        cleaned=cleaned,
        intervention_mask=intervention,
        change_map=diff,
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
        metrics=metrics,
    )
