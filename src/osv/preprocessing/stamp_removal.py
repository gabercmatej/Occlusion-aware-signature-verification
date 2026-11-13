"""Conservative, deletion-only stamp removal.

Deletable stamp pixels (``stamp_mask & ~ambiguous_mask``) are set to the local paper tone.
Nothing else is changed: no inpainting, no stroke drawing, no use of reference signatures.

Remover 0.2.0 (M2a.2): ``review`` means untouched. When the remover's own post-deletion checks
fail (nothing removable, or too little reliable ink left), the proposal is rejected and the
original pixels are returned; the rejected deletion is kept separately as
``proposed_intervention_mask`` (information, *not* applied), with reason
``proposed_deletion_reverted``. ``intervention_mask`` and ``change_map`` always describe what was
actually changed. This does not detect damage in results reported as ``cleaned``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from osv.segmentation.stamp_estimate import StampEstimate

REMOVER_NAME = "deletion_to_paper_tone"
REMOVER_VERSION = "0.2.0"
LOW_RELIABLE_SHARE = 0.5  # provisional: kept ink less than half reliable -> review


@dataclass
class CleaningResult:
    cleaned: np.ndarray
    intervention_mask: np.ndarray  # pixels actually changed by deletion
    change_map: np.ndarray  # uint8 (H, W): max absolute channel difference, 0 where unchanged
    status: str  # no_stamp_detected | cleaned | review
    reasons: tuple[str, ...]
    metrics: dict = field(default_factory=dict)
    proposed_intervention_mask: np.ndarray | None = None  # rejected deletion; never applied

    def __post_init__(self):
        if self.proposed_intervention_mask is None:
            self.proposed_intervention_mask = np.zeros(self.cleaned.shape[:2], bool)


def _apply(image_rgb: np.ndarray, estimate: StampEstimate, intervention: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cleaned = image_rgb.copy()
    if intervention.any():
        paper = np.clip(np.round(estimate.paper_rgb), 0, 255).astype(np.uint8)
        cleaned[intervention] = paper[intervention]
    diff = np.abs(cleaned.astype(np.int16) - image_rgb.astype(np.int16)).max(axis=2).astype(np.uint8)
    return cleaned, diff


def _ink_accounting(estimate: StampEstimate, changed: np.ndarray) -> dict:
    ink = estimate.ink_mask
    ink_n = int(ink.sum())
    changed_ink = ink & changed
    ambiguous_ink = ink & estimate.ambiguous_mask & ~changed
    reliable_ink = ink & ~estimate.ambiguous_mask & ~changed

    def share(m: np.ndarray) -> float | None:
        return None if ink_n == 0 else round(float(m.sum()) / ink_n, 6)

    kept = int(reliable_ink.sum() + ambiguous_ink.sum())
    return {"ink_n": ink_n, "reliable": share(reliable_ink), "ambiguous": share(ambiguous_ink),
            "changed": share(changed_ink),
            "reliable_of_kept": None if kept == 0 else float(reliable_ink.sum()) / kept}


def remove_stamp(image_rgb: np.ndarray, estimate: StampEstimate) -> CleaningResult:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be an (H, W, 3) uint8 array")
    if estimate.stamp_mask.shape != image_rgb.shape[:2]:
        raise ValueError("estimate does not match the image size")

    if estimate.action == "delete":
        proposal = estimate.stamp_mask & ~estimate.ambiguous_mask
    else:
        proposal = np.zeros(image_rgb.shape[:2], bool)
    proposed_img, proposed_diff = _apply(image_rgb, estimate, proposal)
    proposed_changed = proposed_diff > 0
    acc = _ink_accounting(estimate, proposed_changed)

    reasons = list(estimate.reasons)
    if estimate.action == "none":
        status = "no_stamp_detected"
    elif estimate.action == "review":
        status = "review"
    else:
        status = "cleaned"
        if not proposed_changed.any():
            status = "review"
            reasons.append("stamp_detected_nothing_removable")
        if acc["reliable_of_kept"] is not None and acc["reliable_of_kept"] < LOW_RELIABLE_SHARE:
            status = "review"
            reasons.append("low_reliable_ink")

    rejected = np.zeros(image_rgb.shape[:2], bool)
    if status == "cleaned":
        cleaned, diff, intervention = proposed_img, proposed_diff, proposal
    else:
        if proposal.any():
            rejected = proposal
            reasons.append("proposed_deletion_reverted")
        cleaned, diff, intervention = image_rgb.copy(), np.zeros(image_rgb.shape[:2], np.uint8), np.zeros_like(proposal)
    changed = diff > 0
    actual = acc if status == "cleaned" else _ink_accounting(estimate, changed)

    metrics = {
        "image_pixels": int(changed.size),
        "changed_pixels": int(changed.sum()),
        "intervention_pixels": int(intervention.sum()),
        "proposed_intervention_pixels": int(rejected.sum()),
        "proposed_changed_ink_fraction": acc["changed"] if rejected.any() else None,
        "stamp_pixels_estimated": int(estimate.stamp_mask.sum()),
        "ambiguous_pixels": int(estimate.ambiguous_mask.sum()),
        "ink_pixels_estimated": actual["ink_n"],
        "ink_fraction_reliable": actual["reliable"],
        "ink_fraction_ambiguous": actual["ambiguous"],
        "ink_fraction_changed": actual["changed"],
        "reliable_share_of_kept_ink": None if actual["reliable_of_kept"] is None else round(actual["reliable_of_kept"], 6),
    }
    return CleaningResult(
        cleaned=cleaned,
        intervention_mask=intervention,
        change_map=diff,
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
        metrics=metrics,
        proposed_intervention_mask=rejected,
    )
