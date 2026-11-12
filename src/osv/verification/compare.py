"""Diagnostic reference/query comparison with the classical baseline (M2a, unvalidated).

Flow for one case::

    manifest ──► enrol references (check-mode cleaner, normalise, describe)
             └─► query: load ──► selected cleaner (always) ──► normalise cleaned image + masks
                                                         └─► describe ──► per-reference distances
                                                                      └─► median aggregation

There is no path from a raw query to a distance: :func:`compare_query` accepts only a
:class:`CleanedQuery`, which only :func:`prepare_query` creates, and that always runs the
cleaner. References are never used to reconstruct, inpaint or align the query.

Two distances are reported per reference:

* ``all_pixels`` — the cleaned query as it is (equal weighting; the control).
* ``excluding_query_flagged`` — query and reference both described on the same *descriptor
  support*. The exclusion mask is the query's estimated stamp, cleaner intervention, ambiguous
  pixels and any reviewer region after normalisation, dilated by ``EXCLUSION_DILATION_PX`` (a
  margin for mask-boundary uncertainty, not for filter leakage). The descriptor then drops every
  gradient vote whose finite footprint touches the mask (see
  :mod:`osv.verification.descriptor`). The support depends only on the mask, so query and
  references are restricted identically. Excluding a region can also hide exactly the strokes
  that would expose a forgery; both numbers are shown for that reason. A side without enough
  retained edge evidence gives no distance and an explicit status, never 0.

``reference_edge_share_outside_descriptor_support`` is the share of a reference's edge magnitude
at canvas positions the masked comparison did not use. No registration is performed, so this is
an approximate indication of hidden evidence, not a measurement of it.

Scope of the exclusion guarantee. Exact at the canvas level only: two canvases that agree
outside the exclusion mask give the same masked descriptor. Before the canvas exists:

* resampling (``INTER_AREA``) mixes neighbouring source pixels; the flagged masks go through the
  same resampling with the ``any`` rule, so an unflagged canvas pixel can depend on flagged
  source pixels only through coverage weights at or below 1e-6;
* ink darkness divides by a paper luma percentile of the *whole* cleaned image, so flagged or
  cleaned pixels can rescale darkness everywhere (relevant when they cover a large share);
* the cleaner's decisions (which pixels to change, the status) are global and are not covered.

Nothing here outputs a genuine/forged verdict, a probability, or a "safe" result.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from osv.cleaning import clean_image
from osv.preprocessing.image_io import COORDINATE_CONVENTION, LoadedImage, load_image
from osv.preprocessing.normalise import NormalisedImage, NormaliseParams, normalise
from osv.preprocessing.stamp_removal import CleaningResult
from osv.provenance import STATUS_MEANING, sha256_file, write_artifacts
from osv.segmentation.stamp_estimate import StampEstimate, validate_region
from osv.verification.descriptor import GridOrientationDescriptor
from osv.verification.enrolment import Enrolment, descriptor_sha256, enrol
from osv.verification.manifest import Entry, Manifest, load_manifest

REPORT_SCHEMA_VERSION = "m2a-case-0.2"
COMPARISON_VERSION = "0.2.0"
EXCLUSION_DILATION_PX = 2
LOW_SUPPORTED_INK_SHARE = 0.5  # provisional, descriptive
AGGREGATION_RULE = ("median over references with a defined distance; min and max listed; "
                    "no reference weighting")
EXCLUSION_MASK_RULE = (f"stamp | intervention | ambiguous | reviewer region, after normalisation "
                       f"('any' coverage rule), dilated by {EXCLUSION_DILATION_PX} px")
EXCLUSION_GUARANTEE = (
    "Canvas level only: canvases that agree outside the exclusion mask give the same masked descriptor. "
    "Not covered: the cleaner's global decisions, the whole-image paper-luma percentile used for darkness, "
    "and resampling contributions with coverage weight <= 1e-6.")
RESULT_LABEL = "UNVALIDATED CLASSICAL BASELINE (diagnostic only; not an authenticity result)"

GENERAL_LIMITATIONS = (
    "Unvalidated classical baseline: distances are not probabilities and establish neither identity, authenticity nor forgery.",
    "Pixels not flagged by the cleaner are not proven reliable: the cleaners can miss stamps (black, grey, pen-coloured, faint) and can delete or keep handwriting incorrectly even when the status is 'cleaned'.",
    "No registration: query exclusions are applied to references at the same normalised canvas positions only.",
    "Exclusion is exact at the descriptor/canvas level only; the cleaner and the whole-image paper-luma normalisation can still be influenced by flagged pixels.",
    "References are separate signing instances; they were not used to reconstruct or modify the query.",
    "Synthetic or few references do not represent a writer's natural variation.",
)
STATUS_LIMITATIONS = {
    "no_stamp_detected": "Cleaner status no_stamp_detected: nothing was changed; this does not mean the query is free of stamp ink.",
    "cleaned": "Cleaner status cleaned: pixels were deleted; deleted areas may have hidden handwriting, and handwriting damage is not detectable by the cleaner itself.",
    "review": "Cleaner status review: stamp and handwriting were not separated reliably; the cleaned image may contain stamp ink or lack handwriting. Distances are diagnostic only.",
}

_PREPARED = object()


@dataclass(frozen=True)
class CleanedQuery:
    entry: Entry
    loaded: LoadedImage
    method: str
    estimate: StampEstimate
    result: CleaningResult
    normalised: NormalisedImage
    exclusion: np.ndarray
    _origin: object = field(default=None, repr=False)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def prepare_query(entry: Entry, method: str, params: NormaliseParams = NormaliseParams()) -> CleanedQuery:
    """Load the query and run the selected cleaner. The only way to obtain a comparable query."""
    loaded = load_image(entry.resolved)
    if loaded.sha256 != entry.sha256:
        raise RuntimeError(f"query {entry.id} changed after the manifest was checked")
    h, w = loaded.rgb.shape[:2]
    region = validate_region(entry.stamp_region, h, w)
    estimate, result = clean_image(loaded.rgb, method=method, stamp_region=region)
    region_mask = np.zeros((h, w), bool)
    if region is not None:
        x, y, rw, rh = region
        region_mask[y:y + rh, x:x + rw] = True
    masks = {
        "stamp_mask": estimate.stamp_mask,
        "intervention_mask": result.intervention_mask,
        "ambiguous_mask": estimate.ambiguous_mask,
        "reviewer_region_mask": region_mask,
        "ink_mask": estimate.ink_mask,
    }
    norm = normalise(result.cleaned, masks, params)
    flagged = norm.masks["stamp_mask"] | norm.masks["intervention_mask"] | norm.masks["ambiguous_mask"] | norm.masks["reviewer_region_mask"]
    return CleanedQuery(entry, loaded, method, estimate, result, norm, _dilate(flagged, EXCLUSION_DILATION_PX), _PREPARED)


def _r(v: float | None) -> float | None:
    return None if v is None else round(float(v), 6)


def compare_query(query: CleanedQuery, enrolment: Enrolment) -> dict:
    """Per-reference distances and their aggregation. Accepts only a prepared (cleaned) query."""
    if not isinstance(query, CleanedQuery) or query._origin is not _PREPARED:
        raise TypeError("only a query produced by prepare_query (which always cleans) can be compared")
    if query.method != enrolment.cleaner_method:
        raise ValueError("query and references were checked with different cleaner methods")
    desc = enrolment.descriptor
    excl = query.exclusion
    q_all = desc.extract(query.normalised.darkness)
    q_res = desc.describe(query.normalised.darkness, exclude=excl)
    q_masked = q_res.vector
    rows = []
    for ref in enrolment.references:
        d_all = None if (q_all is None or ref.descriptor is None) else desc.distance(q_all, ref.descriptor)
        r_res = desc.describe(ref.normalised.darkness, exclude=excl)
        if not np.array_equal(r_res.support, q_res.support):  # same mask, same rule; defensive
            raise RuntimeError("query and reference descriptor supports differ")
        if q_masked is None:
            status, d_masked = "insufficient_query_edge_evidence", None
        elif r_res.vector is None:
            status, d_masked = "insufficient_reference_edge_evidence", None
        else:
            status, d_masked = "defined", desc.distance(q_masked, r_res.vector)
        share = r_res.retained_edge_share
        rows.append({"reference_id": ref.entry.id, "distance_all_pixels": _r(d_all),
                     "distance_excluding_query_flagged": _r(d_masked),
                     "excluding_query_flagged_status": status,
                     "reference_edge_share_outside_descriptor_support": None if share is None else _r(1.0 - share)})

    def agg(key: str) -> dict:
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return {"n": 0, "median": None, "min": None, "max": None}
        return {"n": len(vals), "median": _r(np.median(vals)), "min": _r(min(vals)), "max": _r(max(vals))}

    return {
        "per_reference": rows,
        "aggregation_rule": AGGREGATION_RULE,
        "aggregate": {"distance_all_pixels": agg("distance_all_pixels"),
                      "distance_excluding_query_flagged": agg("distance_excluding_query_flagged")},
        "query_descriptor_sha256": {"all_pixels": descriptor_sha256(q_all), "excluding_query_flagged": descriptor_sha256(q_masked)},
        "descriptor_support": {"rule": desc.record()["exclusion"]["rule"],
                               "footprint_radius_px": desc.footprint_radius,
                               "support_pixels": q_res.support_pixels,
                               "canvas_pixels": int(q_res.support.size),
                               "applied_to": "query and every reference (same canvas positions)",
                               "query_status": q_res.status,
                               "guarantee": EXCLUSION_GUARANTEE},
    }


def query_evidence(query: CleanedQuery, descriptor: GridOrientationDescriptor | None = None) -> dict:
    """Evidence accounting on the area the masked comparison actually used (descriptor support)."""
    desc = descriptor or GridOrientationDescriptor()
    n = query.normalised
    res = desc.describe(n.darkness, exclude=query.exclusion)
    ink = n.masks["ink_mask"] & n.valid_mask
    ink_n = int(ink.sum())
    area = max(1, int(n.valid_mask.sum()))

    def share(mask):
        return None if ink_n == 0 else float((ink & mask).sum()) / ink_n

    return {
        "ink_pixels_on_canvas": ink_n,
        "ink_share_in_descriptor_support": _r(share(res.support)),
        "ink_share_in_descriptor_support_meaning": ("estimated ink at canvas positions used by the masked comparison; "
                                                    "NOT proven reliable"),
        "ink_share_outside_exclusion_mask": _r(share(~query.exclusion)),
        "query_edge_share_in_descriptor_support": _r(res.retained_edge_share),
        "descriptor_support_share_of_image_area": _r(float((res.support & n.valid_mask).sum()) / area),
        "exclusion_mask_share_of_image_area": _r(float((query.exclusion & n.valid_mask).sum()) / area),
        "exclusion_mask_rule": EXCLUSION_MASK_RULE,
        "descriptor_support_rule": desc.record()["exclusion"]["rule"],
        "descriptor_footprint_radius_px": desc.footprint_radius,
    }


def review_flags(manifest: Manifest, enrolment: Enrolment, query: CleanedQuery, evidence: dict,
                 comparison: dict) -> list[str]:
    flags = list(enrolment.flags)
    if "query_shares_source_document_with_reference" in manifest.warnings:
        flags.append("query_shares_source_document_with_reference")
    status = query.result.status
    flags.append(f"cleaner_status:{status}")
    if status == "review":
        flags.extend(f"cleaner_reason:{r}" for r in query.result.reasons)
    if query.result.metrics.get("changed_pixels", 0) > 0:
        flags.append("query_pixels_deleted_by_cleaner")
    share = evidence["ink_share_in_descriptor_support"]
    if share is None or share < LOW_SUPPORTED_INK_SHARE:
        flags.append("low_query_ink_in_descriptor_support")
    if any(r["excluding_query_flagged_status"] != "defined" for r in comparison["per_reference"]):
        flags.append("masked_distance_undefined")
    return flags


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path, format="PNG", optimize=False)


def _mask_png(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 255, 0).astype(np.uint8)


def _darkness_png(d: np.ndarray) -> np.ndarray:
    return np.clip(np.round(255 * (1.0 - d)), 0, 255).astype(np.uint8)


def run_case(manifest_path, out_dir, params: NormaliseParams = NormaliseParams()) -> dict:
    """Validate, enrol, clean, compare, and write all artifacts and ``case_report.json``."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory {out_dir} is not empty; refusing to overwrite")
    manifest = load_manifest(manifest_path)
    descriptor = GridOrientationDescriptor()
    enrolment = enrol(manifest, descriptor, params)
    query = prepare_query(manifest.query, manifest.config["cleaner_method"], params)
    comparison = compare_query(query, enrolment)
    evidence = query_evidence(query, descriptor)

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaning_report = write_artifacts(manifest.query.resolved, query.estimate, query.result, out_dir / "query_cleaning")
    norm_dir = out_dir / "normalised"
    norm_dir.mkdir()
    n = query.normalised
    _save_png(norm_dir / "query_cleaned_rgb.png", n.rgb)
    _save_png(norm_dir / "query_cleaned_darkness.png", _darkness_png(n.darkness))
    _save_png(norm_dir / "query_exclusion_mask.png", _mask_png(query.exclusion))
    for name, m in sorted(n.masks.items()):
        _save_png(norm_dir / f"query_{name}.png", _mask_png(m))
    ref_dir = out_dir / "references"
    ref_dir.mkdir()
    for ref in enrolment.references:
        copy = ref_dir / f"{ref.entry.id}_original{ref.entry.resolved.suffix.lower()}"
        shutil.copyfile(ref.entry.resolved, copy)
        if sha256_file(copy) != ref.entry.sha256:
            raise RuntimeError(f"copy of reference {ref.entry.id} is not byte-identical")
        _save_png(norm_dir / f"ref_{ref.entry.id}_rgb.png", ref.normalised.rgb)
        _save_png(norm_dir / f"ref_{ref.entry.id}_darkness.png", _darkness_png(ref.normalised.darkness))

    enrol_record = enrolment.record()
    (out_dir / "enrolment.json").write_text(json.dumps(enrol_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    status = query.result.status
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "result_type": RESULT_LABEL,
        "case_id": manifest.case_id,
        "claimed_writer_id": manifest.claimed_writer_id,
        "manifest_sha256": manifest.manifest_sha256,
        "config": {**manifest.config, "comparison_version": COMPARISON_VERSION,
                   "descriptor": descriptor.record(), "normalisation": params.as_dict(),
                   "exclusion_dilation_px": EXCLUSION_DILATION_PX,
                   "exclusion_mask_rule": EXCLUSION_MASK_RULE,
                   "descriptor_support_rule": descriptor.record()["exclusion"]["rule"]},
        "coordinate_convention": COORDINATE_CONVENTION,
        "query": {
            **manifest.query.record(),
            "image": query.loaded.record(),
            "normalisation": n.record(),
            "cleaner": {"method": query.method, "status": status, "status_meaning": STATUS_MEANING[status],
                        "reasons": list(query.result.reasons), "metrics": query.result.metrics,
                        "artifacts_dir": "query_cleaning", "report_sha256": sha256_file(out_dir / "query_cleaning" / "report.json")},
            "evidence": evidence,
        },
        "reference_count": len(enrolment.references),
        "reference_count_distinct_files": enrolment.unique_count,
        "reference_pair_distance_summary": enrolment.spread(),
        "comparison": comparison,
        "review_flags": review_flags(manifest, enrolment, query, evidence, comparison),
        "evidence_limitations": list(GENERAL_LIMITATIONS) + [STATUS_LIMITATIONS[status]],
        "labels_note": "manifest labels are not read by this comparison",
        "enrolment_sha256": sha256_file(out_dir / "enrolment.json"),
    }
    if cleaning_report["input"]["sha256"] != manifest.query.sha256:
        raise RuntimeError("query changed during the run")
    (out_dir / "case_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
