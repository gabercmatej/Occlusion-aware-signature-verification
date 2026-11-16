"""Reader, local manifest and writer splits for a folder-per-writer signature dataset (M2b).

Layout read: ``<root>/<writer folder>/<image file>``. The folder name is the dataset's *claimed*
writer grouping; it is recorded as such and checked for one kind of grouping error (identical
images under different writers), not trusted as proof of identity.

Everything here is independent of comparison scores. Three steps, each producing a JSON document
with a canonical SHA-256 (sorted keys, compact separators, UTF-8):

1. :func:`scan` — one record per file: byte hash, decoded-pixel hash, format, mode, size, simple
   paper/ink statistics, and whether the project's own loader accepts it.
2. :func:`build_manifest` — applies the exclusion rules below and lists eligible images.
3. :func:`make_splits` — writer-disjoint development / validation / test partitions.

Exclusion rules (``EXCLUSION_RULES``; fixed before any score exists):

* ``undecodable`` / ``unsupported_by_loader`` — the file cannot be read by ``load_image``.
* ``blank_image`` — the decoded image is (near) uniform: luma range below ``BLANK_LUMA_RANGE``.
  Faint or unusual signatures are *not* excluded here; if preprocessing cannot use them they
  show up as undefined comparisons in the benchmark.
* ``duplicate_within_writer`` — same decoded pixels as an earlier file (path order) of the same
  writer: not a separate signing instance. The first file is kept.
* ``duplicate_across_writers`` — same decoded pixels under two or more writers: the label is
  ambiguous, every copy is excluded, and the writers involved are tied into one group so that
  they can never be split across partitions.

Splits: writer groups are ordered by ``sha256("<seed>:<smallest writer id in the group>")`` and
assigned in that order to development, then validation, then test, by cumulative writer count.
No random state, so the result does not depend on the Python or NumPy version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from osv.preprocessing.image_io import UnsupportedImageError, load_image

MANIFEST_SCHEMA = "osv-dataset-manifest-1.0"
SPLIT_SCHEMA = "osv-writer-splits-1.0"
PARTITIONS = ("development", "validation", "test")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
BLANK_LUMA_RANGE = 8  # 8-bit luma levels between the 0.1st and 99.9th percentile
INK_DARKNESS = 0.30  # same value as the normaliser's provisional ink threshold; descriptive here
BLUE_MARGIN = 25  # median (B - R) of dark pixels above this is recorded as "blue-ish"
EXCLUSION_RULES = {
    "undecodable": "file is not a decodable image",
    "unsupported_by_loader": "osv.preprocessing.image_io.load_image refuses the file (format, mode, frames, transparency)",
    "blank_image": f"luma range between the 0.1st and 99.9th percentile is below {BLANK_LUMA_RANGE} levels",
    "duplicate_within_writer": "decoded pixels identical to an earlier file (path order) of the same writer; first kept",
    "duplicate_across_writers": "decoded pixels identical under two or more writers; every copy excluded, writers grouped",
}


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def write_json(path, obj) -> str:
    """Write ``obj`` readable (indented) and return its canonical hash."""
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return canonical_sha256(obj)


def read_json(path, expected_sha256: str | None = None):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_sha256 is not None and canonical_sha256(obj) != expected_sha256:
        raise ValueError(f"{Path(path).name}: content hash differs from the frozen value {expected_sha256[:12]}…")
    return obj


def _stats(rgb: np.ndarray) -> dict:
    luma = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = np.percentile(luma, [0.1, 99.9])
    paper = float(np.percentile(luma, 95.0))
    darkness = np.clip(1.0 - luma / max(paper, 1.0), 0.0, 1.0)
    dark = darkness >= INK_DARKNESS
    n_dark = int(dark.sum())
    blue = None
    if n_dark:
        blue = float(np.median(rgb[..., 2][dark].astype(np.int16) - rgb[..., 0][dark].astype(np.int16)))
    return {"luma_range": round(float(hi - lo), 2), "paper_luma_p95": round(paper, 2),
            "dark_pixel_share": round(n_dark / dark.size, 6),
            "dark_blue_minus_red_median": blue,
            "ink_tone": None if blue is None else ("blue-ish" if blue > BLUE_MARGIN else "neutral")}


def scan_file(root: Path, path: Path) -> dict:
    rel = path.relative_to(root).as_posix()
    rec = {"path": rel, "writer": path.parent.relative_to(root).as_posix(), "problem": None}
    try:
        loaded = load_image(path)
    except UnsupportedImageError as exc:
        data = path.read_bytes()
        rec.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        try:
            with Image.open(path) as im:
                im.load()
            rec["problem"] = "unsupported_by_loader"
        except Exception:
            rec["problem"] = "undecodable"
        rec["problem_detail"] = str(exc).split(": ", 1)[-1]
        return rec
    except Exception as exc:  # truncated or corrupt data inside a recognised container
        data = path.read_bytes()
        rec.update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), problem="undecodable",
                   problem_detail=type(exc).__name__)
        return rec
    h, w = loaded.rgb.shape[:2]
    pixel_hash = hashlib.sha256(f"{h}x{w}:".encode() + loaded.rgb.tobytes()).hexdigest()
    rec.update(sha256=loaded.sha256, size_bytes=loaded.size_bytes, format=loaded.format, mode=loaded.mode,
               width=int(w), height=int(h), exif_orientation=loaded.exif_orientation, pixel_sha256=pixel_hash,
               **_stats(loaded.rgb))
    if rec["luma_range"] < BLANK_LUMA_RANGE:
        rec["problem"] = "blank_image"
    return rec


def scan(root) -> list[dict]:
    """One record per image file under ``root`` (sorted by path). Other files are listed by the caller."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                   and not p.name.startswith("."))
    return [scan_file(root, p) for p in files]


def build_manifest(records: list[dict], dataset: dict) -> dict:
    """Apply the exclusion rules; return the manifest (records, exclusions, writer groups, counts)."""
    records = sorted((dict(r) for r in records), key=lambda r: r["path"])
    by_pixels: dict[str, list[dict]] = {}
    for r in records:
        r["excluded"] = r.pop("problem", None)
        if r["excluded"] is None:
            by_pixels.setdefault(r["pixel_sha256"], []).append(r)
    parent: dict[str, str] = {r["writer"]: r["writer"] for r in records}

    def find(w: str) -> str:
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w

    duplicate_sets = []
    for pix, group in by_pixels.items():
        if len(group) < 2:
            continue
        writers = sorted({r["writer"] for r in group})
        duplicate_sets.append({"pixel_sha256": pix, "paths": [r["path"] for r in group], "writers": writers})
        if len(writers) > 1:
            for r in group:
                r["excluded"] = "duplicate_across_writers"
            for w in writers[1:]:
                parent[find(w)] = find(writers[0])
        else:
            for r in group[1:]:
                r["excluded"] = "duplicate_within_writer"
    groups: dict[str, list[str]] = {}
    for w in sorted(parent):
        groups.setdefault(find(w), []).append(w)
    exclusion_counts: dict[str, int] = {}
    for r in records:
        if r["excluded"]:
            exclusion_counts[r["excluded"]] = exclusion_counts.get(r["excluded"], 0) + 1
    eligible: dict[str, int] = {}
    for r in records:
        eligible.setdefault(r["writer"], 0)
        if not r["excluded"]:
            eligible[r["writer"]] += 1
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset": dataset,
        "label_note": "writer = folder name as distributed (claimed grouping); not verified identity",
        "exclusion_rules": EXCLUSION_RULES,
        "records": records,
        "duplicate_sets": sorted(duplicate_sets, key=lambda d: d["paths"]),
        "writer_groups": sorted(sorted(g) for g in groups.values()),
        "counts": {"files": len(records), "writers": len(parent), "eligible_files": sum(eligible.values()),
                   "excluded": exclusion_counts, "eligible_per_writer_min": min(eligible.values()) if eligible else 0,
                   "eligible_per_writer_max": max(eligible.values()) if eligible else 0},
    }


def eligible_by_writer(manifest: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in manifest["records"]:
        if not r["excluded"]:
            out.setdefault(r["writer"], []).append(r)
    return out


def make_splits(manifest: dict, seed: int, fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
                min_images: int = 1) -> dict:
    """Deterministic writer-disjoint partitions. Writers with fewer than ``min_images`` eligible images are left out."""
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError("fractions must be positive and sum to 1")
    eligible = eligible_by_writer(manifest)
    left_out = sorted(w for g in manifest["writer_groups"] for w in g if len(eligible.get(w, [])) < min_images)
    groups = [[w for w in g if w not in left_out] for g in manifest["writer_groups"]]
    groups = [g for g in groups if g]
    groups.sort(key=lambda g: hashlib.sha256(f"{seed}:{min(g)}".encode()).hexdigest())
    total = sum(len(g) for g in groups)
    bounds = (fractions[0] * total, (fractions[0] + fractions[1]) * total)
    parts: dict[str, list[str]] = {p: [] for p in PARTITIONS}
    placed = 0
    for g in groups:
        k = 0 if placed < bounds[0] else (1 if placed < bounds[1] else 2)
        parts[PARTITIONS[k]].extend(g)
        placed += len(g)
    return {
        "schema": SPLIT_SCHEMA,
        "rule": "writer groups ordered by sha256('<seed>:<smallest writer id>'), assigned to development, validation, "
                "test by cumulative writer count; groups (writers tied by cross-writer duplicates) are never divided",
        "seed": seed,
        "fractions": list(fractions),
        "min_eligible_images_per_writer": min_images,
        "writers_left_out": left_out,
        "partitions": {p: sorted(ws) for p, ws in parts.items()},
        "counts": {p: len(ws) for p, ws in parts.items()},
    }


def check_splits(splits: dict, manifest: dict | None = None) -> None:
    """Raise if a writer is in two partitions or (with a manifest) a writer group is divided."""
    where: dict[str, str] = {}
    for p in PARTITIONS:
        for w in splits["partitions"][p]:
            if w in where:
                raise ValueError(f"writer {w} is in both {where[w]} and {p}")
            where[w] = p
    if manifest is not None:
        for g in manifest["writer_groups"]:
            if len({where[w] for w in g if w in where}) > 1:
                raise ValueError(f"writer group {g} is divided across partitions")
