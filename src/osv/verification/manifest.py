"""Local case manifest: one claimed writer, explicit genuine references, one query (M2a).

Format (JSON, manifest_version "1.0")::

    {
      "manifest_version": "1.0",
      "case_id": "demo-001",
      "claimed_writer_id": "W-DEMO-A",            # pseudonymous; never a real name
      "config": {"config_version": "m2a-classical-0.1", "cleaner_method": "chroma"},
      "references": [
        {"id": "r1", "path": "references/r1.png", "sha256": "<64 hex>",
         "source_document_id": "doc-01" | null, "signing_instance_id": "sig-01" | null,
         "label": {"status": "known", "value": "genuine", "source": "<who/what asserted it>"}}
      ],
      "query": {"id": "q1", "path": "query/q1.png", "sha256": "<64 hex>",
                "source_document_id": null, "signing_instance_id": null,
                "stamp_region": null | [x, y, w, h],
                "label": {"status": "unknown", "value": null, "source": null}}
    }

Rules:

* Paths are relative to the manifest file; absolute paths are refused so that manifests and
  reports never embed local machine paths. Hashes are required and checked against the files.
* Labels are explicit fields. Nothing is inferred from folder or file names. Labels are recorded
  for later evaluation and are never read by the comparison.
* A reference labelled anything other than genuine is refused; an unconfirmed reference label is
  a warning. A query that is byte-identical to a reference, or shares its signing instance, is
  refused (reference/query separation). Other overlaps are warnings.
* Customer manifests stay local: keep them under ``data/company/`` or name them ``*.local.json``
  (both git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

MANIFEST_VERSION = "1.0"
CONFIG_VERSION = "m2a-classical-0.1"
CLEANER_METHODS = ("chroma", "optical-density")
LABEL_VALUES = ("genuine", "forged")

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_TOP_KEYS = {"manifest_version", "case_id", "claimed_writer_id", "config", "references", "query"}
_ENTRY_KEYS = {"id", "path", "sha256", "source_document_id", "signing_instance_id", "label"}
_QUERY_KEYS = _ENTRY_KEYS | {"stamp_region"}
_CONFIG_KEYS = {"config_version", "cleaner_method"}


class ManifestError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("invalid manifest: " + "; ".join(self.problems))


@dataclass(frozen=True)
class Entry:
    id: str
    path: str  # as written in the manifest (relative, POSIX)
    sha256: str
    source_document_id: str | None
    signing_instance_id: str | None
    label: dict
    resolved: Path = field(repr=False, compare=False, default=Path())
    stamp_region: tuple[int, int, int, int] | None = None

    def record(self) -> dict:
        out = {"id": self.id, "path": self.path, "sha256": self.sha256,
               "source_document_id": self.source_document_id, "signing_instance_id": self.signing_instance_id}
        if self.stamp_region is not None:
            out["stamp_region"] = list(self.stamp_region)
        return out


@dataclass(frozen=True)
class Manifest:
    case_id: str
    claimed_writer_id: str
    config: dict
    references: tuple[Entry, ...]
    query: Entry
    warnings: tuple[str, ...]
    manifest_sha256: str

    def labels_record(self) -> dict:
        return {"note": "recorded for later evaluation only; not read by the comparison",
                "query": self.query.label, "references": {r.id: r.label for r in self.references}}


def _check_label(where: str, label, problems: list[str]) -> None:
    if not isinstance(label, dict) or set(label) != {"status", "value", "source"}:
        problems.append(f"{where}.label must be an object with exactly status, value, source")
        return
    if label["status"] == "unknown":
        if label["value"] is not None:
            problems.append(f"{where}.label: status unknown requires value null")
    elif label["status"] == "known":
        if label["value"] not in LABEL_VALUES:
            problems.append(f"{where}.label.value must be one of {LABEL_VALUES} when status is known")
        if not isinstance(label["source"], str) or not label["source"].strip():
            problems.append(f"{where}.label.source must name who or what asserted the label")
    else:
        problems.append(f"{where}.label.status must be 'known' or 'unknown'")


def _check_path(where: str, text, problems: list[str]) -> str | None:
    if not isinstance(text, str) or not text.strip():
        problems.append(f"{where}.path must be a non-empty string")
        return None
    if (text.startswith(("/", "\\", "~")) or PureWindowsPath(text).drive
            or PurePosixPath(text).is_absolute()):
        problems.append(f"{where}.path must be relative to the manifest (absolute paths are not allowed)")
        return None
    return text


def _optional_id(where: str, value, problems: list[str]) -> None:
    if value is not None and (not isinstance(value, str) or not _ID.match(value)):
        problems.append(f"{where} must be null or an identifier matching {_ID.pattern}")


def _parse_entry(where: str, raw, allowed: set[str], base: Path, problems: list[str]) -> Entry | None:
    if not isinstance(raw, dict):
        problems.append(f"{where} must be an object")
        return None
    unknown = set(raw) - allowed
    missing = (allowed - {"stamp_region"}) - set(raw)
    if unknown:
        problems.append(f"{where} has unknown keys {sorted(unknown)}")
    if missing:
        problems.append(f"{where} is missing keys {sorted(missing)}")
        return None
    n = len(problems)
    if not isinstance(raw["id"], str) or not _ID.match(raw["id"]):
        problems.append(f"{where}.id must match {_ID.pattern}")
    path = _check_path(where, raw["path"], problems)
    if not isinstance(raw["sha256"], str) or not _SHA.match(raw["sha256"]):
        problems.append(f"{where}.sha256 must be 64 lowercase hex characters")
    _optional_id(f"{where}.source_document_id", raw["source_document_id"], problems)
    _optional_id(f"{where}.signing_instance_id", raw["signing_instance_id"], problems)
    _check_label(where, raw["label"], problems)
    region = raw.get("stamp_region")
    if region is not None:
        if (not isinstance(region, list) or len(region) != 4 or not all(isinstance(v, int) for v in region)
                or region[2] <= 0 or region[3] <= 0 or region[0] < 0 or region[1] < 0):
            problems.append(f"{where}.stamp_region must be null or [x, y, w, h] with non-negative integers and w, h > 0")
            region = None
    if len(problems) > n or path is None:
        return None
    resolved = (base / path)
    if not resolved.is_file():
        problems.append(f"{where}: file not found: {path}")
        return None
    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual != raw["sha256"]:
        problems.append(f"{where}: sha256 mismatch for {path}")
        return None
    return Entry(id=raw["id"], path=path, sha256=raw["sha256"], source_document_id=raw["source_document_id"],
                 signing_instance_id=raw["signing_instance_id"], label=dict(raw["label"]), resolved=resolved,
                 stamp_region=None if region is None else tuple(region))


def load_manifest(path) -> Manifest:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path.name}")
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError([f"not valid UTF-8 JSON ({exc.__class__.__name__})"]) from exc
    if not isinstance(raw, dict):
        raise ManifestError(["top level must be an object"])
    problems: list[str] = []
    unknown, missing = set(raw) - _TOP_KEYS, _TOP_KEYS - set(raw)
    if unknown:
        problems.append(f"unknown top-level keys {sorted(unknown)}")
    if missing:
        raise ManifestError(problems + [f"missing top-level keys {sorted(missing)}"])
    if raw["manifest_version"] != MANIFEST_VERSION:
        problems.append(f"manifest_version must be {MANIFEST_VERSION!r}")
    for key in ("case_id", "claimed_writer_id"):
        if not isinstance(raw[key], str) or not _ID.match(raw[key]):
            problems.append(f"{key} must match {_ID.pattern}")

    cfg = raw["config"]
    config = {}
    if not isinstance(cfg, dict) or "config_version" not in cfg or set(cfg) - _CONFIG_KEYS:
        problems.append(f"config must be an object with config_version and optional keys {sorted(_CONFIG_KEYS)}")
    else:
        if cfg["config_version"] != CONFIG_VERSION:
            problems.append(f"config.config_version {cfg['config_version']!r} is not supported; expected {CONFIG_VERSION!r}")
        method = cfg.get("cleaner_method", "chroma")
        if method not in CLEANER_METHODS:
            problems.append(f"config.cleaner_method must be one of {CLEANER_METHODS}")
        config = {"config_version": cfg["config_version"], "cleaner_method": method}

    base = path.parent
    refs_raw = raw["references"]
    refs: list[Entry] = []
    if not isinstance(refs_raw, list) or not refs_raw:
        problems.append("references must be a non-empty list")
    else:
        for i, r in enumerate(refs_raw):
            e = _parse_entry(f"references[{i}]", r, _ENTRY_KEYS, base, problems)
            if e is not None:
                refs.append(e)
    query = _parse_entry("query", raw["query"], _QUERY_KEYS, base, problems)

    warnings: list[str] = []
    if refs and query is not None and len(refs) == len(refs_raw):
        ids = [r.id for r in refs]
        if len(set(ids)) != len(ids) or query.id in ids:
            problems.append("reference and query ids must be unique")
        for r in refs:
            if r.label["status"] == "known" and r.label["value"] != "genuine":
                problems.append(f"reference {r.id} is labelled {r.label['value']!r}; only genuine references can be enrolled")
            elif r.label["status"] == "unknown":
                warnings.append(f"reference_label_not_confirmed:{r.id}")
        if query.sha256 in {r.sha256 for r in refs}:
            problems.append("query file is byte-identical to a reference (reference/query separation)")
        if Path(query.path).as_posix() in {Path(r.path).as_posix() for r in refs}:
            problems.append("query path is also listed as a reference")
        ref_instances = [r.signing_instance_id for r in refs if r.signing_instance_id]
        if query.signing_instance_id and query.signing_instance_id in ref_instances:
            problems.append("query shares a signing_instance_id with a reference (reference/query separation)")
        ref_docs = [r.source_document_id for r in refs if r.source_document_id]
        if query.source_document_id and query.source_document_id in ref_docs:
            warnings.append("query_shares_source_document_with_reference")
        shas = [r.sha256 for r in refs]
        if len(set(shas)) != len(shas):
            warnings.append("duplicate_reference_files")
        if len(set(ref_instances)) != len(ref_instances):
            warnings.append("references_share_signing_instance")
        if len(set(ref_docs)) != len(ref_docs):
            warnings.append("references_share_source_document")
    if problems:
        raise ManifestError(problems)
    return Manifest(case_id=raw["case_id"], claimed_writer_id=raw["claimed_writer_id"], config=config,
                    references=tuple(refs), query=query, warnings=tuple(warnings),
                    manifest_sha256=hashlib.sha256(raw_bytes).hexdigest())
