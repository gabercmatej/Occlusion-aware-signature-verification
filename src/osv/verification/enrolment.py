"""Explicit reference enrolment for one claimed writer (M2a).

Each reference listed in the manifest is loaded (hash checked), run through the selected
cleaner in *check mode* only, normalised and described. Check mode never changes the
reference: the descriptor is computed from the reference's own pixels, and a detected stamp
is reported as a flag, because a stamped reference distorts every later comparison.

Flags are descriptive and provisional, not validated quality thresholds:

* ``fewer_than_three_references`` — fewer than three distinct reference files.
* ``reference_stamp_indicated:<id>`` — the check-mode cleaner did not return
  ``no_stamp_detected`` (it may also be triggered by a coloured pen; see the cleaner docs).
* ``reference_without_edge_evidence:<id>`` — nothing left to describe.
* ``reference_spread_outlier:<id>`` — with at least three usable references, the reference's
  median distance to the others exceeds ``OUTLIER_FACTOR`` x the median distance among the
  *other* references (pairs not involving it).
* ``reference_content_box_touches_image_border:<id>`` — the reference's estimated content reaches
  the crop edge, so part of it may be cut off.
* Manifest warnings (duplicates, shared documents or signing instances, unconfirmed labels).

Pair distances (0.2.0) use :func:`osv.verification.pairs.compare_pair`: both references on their
common support (padding of either excluded), with the same minimum-retained-evidence rule as the
query rows; undefined pairs are counted, never given a number.

References signed in one sitting understate natural variation, so a tight spread is not
evidence of a consistent writer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from osv.cleaning import clean_image
from osv.preprocessing.image_io import LoadedImage, load_image
from osv.preprocessing.normalise import NormalisedImage, NormaliseParams, normalise
from osv.verification.descriptor import GridOrientationDescriptor
from osv.verification.manifest import Entry, Manifest
from osv.verification.pairs import PAIR_SUPPORT_RULE, compare_pair, describe_normalised, summarise

ENROLMENT_VERSION = "0.2.0"
MIN_REFERENCES = 3
OUTLIER_FACTOR = 1.5


def descriptor_sha256(vec: np.ndarray | None) -> str | None:
    return None if vec is None else hashlib.sha256(np.ascontiguousarray(vec, np.float64).tobytes()).hexdigest()


@dataclass
class EnrolledReference:
    entry: Entry
    loaded: LoadedImage
    normalised: NormalisedImage
    descriptor: np.ndarray | None
    check_status: str
    check_reasons: tuple[str, ...]

    def record(self) -> dict:
        return {
            **self.entry.record(),
            "image": self.loaded.record(),
            "normalisation": self.normalised.record(),
            "descriptor_sha256": descriptor_sha256(self.descriptor),
            "cleaner_check": {"status": self.check_status, "reasons": list(self.check_reasons),
                              "mode": "check only; reference pixels are not modified"},
        }


@dataclass
class Enrolment:
    claimed_writer_id: str
    references: list[EnrolledReference]
    pair_distances: dict[str, float]  # defined pairs only
    flags: list[str]
    descriptor: GridOrientationDescriptor
    cleaner_method: str
    config: dict = field(default_factory=dict)
    pair_rows: list[dict] = field(default_factory=list)

    @property
    def unique_count(self) -> int:
        return len({r.entry.sha256 for r in self.references})

    def usable(self) -> list[EnrolledReference]:
        return [r for r in self.references if r.descriptor is not None]

    def spread(self) -> dict:
        """All-pixel reference spread: each pair on its common support (padding only)."""
        return summarise(self.pair_rows)

    def spread_under(self, extra: np.ndarray | None) -> tuple[dict, list[dict]]:
        """Reference spread with ``extra`` (e.g. the query's exclusion mask) added to every pair."""
        rows = []
        refs = self.usable()
        for i, a in enumerate(refs):
            for b in refs[i + 1:]:
                rows.append({"pair": f"{a.entry.id}|{b.entry.id}",
                             **compare_pair(self.descriptor, a.normalised, b.normalised, extra, roles=("a", "b"))})
        return summarise(rows), rows

    def record(self) -> dict:
        return {
            "enrolment_version": ENROLMENT_VERSION,
            "claimed_writer_id": self.claimed_writer_id,
            "reference_count": len(self.references),
            "reference_count_distinct_files": self.unique_count,
            "references": [r.record() for r in self.references],
            "pair_distances": {k: round(v, 6) for k, v in sorted(self.pair_distances.items())},
            "pair_distance_summary": self.spread(),
            "pair_support_rule": PAIR_SUPPORT_RULE,
            "flags": list(self.flags),
            "flag_note": "descriptive, provisional rules; not validated quality thresholds",
            "outlier_rule": f"median distance to the other references > {OUTLIER_FACTOR} x median distance among the other references (>= {MIN_REFERENCES} usable references)",
            "descriptor": self.descriptor.record(),
            "cleaner_method_for_check": self.cleaner_method,
            "config": self.config,
        }


def enrol(manifest: Manifest, descriptor: GridOrientationDescriptor | None = None,
          params: NormaliseParams = NormaliseParams()) -> Enrolment:
    descriptor = descriptor or GridOrientationDescriptor()
    method = manifest.config["cleaner_method"]
    refs: list[EnrolledReference] = []
    flags: list[str] = list(w for w in manifest.warnings if w != "query_shares_source_document_with_reference")
    for entry in manifest.references:
        loaded = load_image(entry.resolved)
        if loaded.sha256 != entry.sha256:
            raise RuntimeError(f"reference {entry.id} changed after the manifest was checked")
        _est, res = clean_image(loaded.rgb, method=method)
        norm = normalise(loaded.rgb, params=params)
        vec = describe_normalised(descriptor, norm).vector
        refs.append(EnrolledReference(entry, loaded, norm, vec, res.status, res.reasons))
        if res.status != "no_stamp_detected":
            flags.append(f"reference_stamp_indicated:{entry.id}")
        if vec is None:
            flags.append(f"reference_without_edge_evidence:{entry.id}")
        if any(f.startswith("content_box_touches_image_border") for f in norm.content_flags):
            flags.append(f"reference_content_box_touches_image_border:{entry.id}")

    if len({r.entry.sha256 for r in refs}) < MIN_REFERENCES:
        flags.append("fewer_than_three_references")

    enrolment = Enrolment(claimed_writer_id=manifest.claimed_writer_id, references=refs, pair_distances={},
                          flags=flags, descriptor=descriptor, cleaner_method=method, config=dict(manifest.config))
    _summary, rows = enrolment.spread_under(None)
    enrolment.pair_rows = rows
    pairs = {r["pair"]: r["distance"] for r in rows if r["distance"] is not None}
    enrolment.pair_distances = pairs
    usable = enrolment.usable()
    if len(usable) >= MIN_REFERENCES:
        for r in usable:
            own = [v for k, v in pairs.items() if r.entry.id in k.split("|")]
            rest = [v for k, v in pairs.items() if r.entry.id not in k.split("|")]
            if not own or not rest:
                continue
            rest_median = float(np.median(rest))
            if rest_median > 0 and float(np.median(own)) > OUTLIER_FACTOR * rest_median:
                flags.append(f"reference_spread_outlier:{r.entry.id}")
    return enrolment
