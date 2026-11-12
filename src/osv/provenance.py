"""Provenance artifacts for one cleaning run.

Writes a byte-identical copy of the original, every mask, the cleaned image, the actual
change map and ``report.json``. The report has no timestamps or absolute paths, so the
same input produces byte-identical output.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image

import osv
from osv.preprocessing.stamp_removal import REMOVER_NAME, REMOVER_VERSION, CleaningResult
from osv.segmentation.stamp_estimate import ESTIMATOR_NAME, ESTIMATOR_VERSION, StampEstimate

SCHEMA_VERSION = "1.0"

STATUS_MEANING = {
    "no_stamp_detected": (
        "No coloured stamp ink was found and nothing was changed. This does not establish that the "
        "image is clean: black, grey or pen-coloured stamps and stamps in grayscale scans are "
        "invisible to this cleaner."
    ),
    "cleaned": (
        "Coloured stamp pixels were deleted (set to the local paper tone). Overlap pixels were kept "
        "and marked ambiguous. Deleted pixels may have hidden handwriting; nothing was restored."
    ),
    "review": "The cleaner could not separate stamp from handwriting reliably; see reasons.",
}

MASK_FILES = {
    "stamp_mask": "stamp_mask.png",
    "intervention_mask": "intervention_mask.png",
    "ink_mask": "ink_mask.png",
    "ambiguous_mask": "ambiguous_mask.png",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_rgb(path: Path) -> tuple[np.ndarray, str]:
    """Decode an image file to RGB uint8 without changing the file. EXIF orientation is not applied."""
    with Image.open(path) as im:
        mode = im.mode
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    return rgb, mode


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path, format="PNG", optimize=False)


def _mask_png(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 255, 0).astype(np.uint8)


def write_artifacts(original_path, estimate: StampEstimate, result: CleaningResult, out_dir) -> dict:
    original_path, out_dir = Path(original_path), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory {out_dir} is not empty; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    original_bytes = original_path.read_bytes()
    input_sha = sha256_bytes(original_bytes)
    copy_name = "original" + original_path.suffix.lower()
    shutil.copyfile(original_path, out_dir / copy_name)
    if sha256_file(out_dir / copy_name) != input_sha:
        raise RuntimeError("copy of the original is not byte-identical")

    h, w = result.cleaned.shape[:2]
    _save_png(out_dir / "cleaned.png", result.cleaned)
    _save_png(out_dir / "change_map.png", result.change_map)
    masks = {
        "stamp_mask": estimate.stamp_mask,
        "intervention_mask": result.intervention_mask,
        "ink_mask": estimate.ink_mask,
        "ambiguous_mask": estimate.ambiguous_mask,
    }
    for key, name in MASK_FILES.items():
        if masks[key].shape != (h, w):
            raise ValueError(f"{key} has shape {masks[key].shape}, expected {(h, w)}")
        _save_png(out_dir / name, _mask_png(masks[key]))

    artifacts = {"original_copy": copy_name, "cleaned": "cleaned.png", "change_map": "change_map.png", **MASK_FILES}
    region = estimate.stamp_region
    report = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "filename": original_path.name,
            "sha256": input_sha,
            "size_bytes": len(original_bytes),
            "width": int(w),
            "height": int(h),
        },
        "reviewer_stamp_region": None if region is None else dict(zip(("x", "y", "w", "h"), region)),
        "cleaner": {
            "estimator": {"name": ESTIMATOR_NAME, "version": ESTIMATOR_VERSION, "parameters": estimate.params_dict()},
            "remover": {"name": REMOVER_NAME, "version": REMOVER_VERSION, "method": "deletion to local paper tone; no inpainting"},
            "uses_only": "input image pixels and the optional reviewer stamp region",
        },
        "status": result.status,
        "status_meaning": STATUS_MEANING[result.status],
        "reasons": list(result.reasons),
        "separability": None if estimate.separability is None else round(float(estimate.separability), 6),
        "metrics": result.metrics,
        "estimate_stats": estimate.stats,
        "definitions": {
            "ink_fraction_reliable": "estimated ink neither changed nor marked ambiguous; with no_stamp_detected this does not exclude an undetected stamp",
            "ink_fraction_ambiguous": "estimated ink kept but marked ambiguous (possible stamp/handwriting overlap or reviewer-marked region)",
            "ink_fraction_changed": "estimated ink that the cleaner deleted",
            "change_map": "max absolute RGB difference between cleaned and original, per pixel",
            "separability": "chroma distance between stamp-hue ink and remaining ink relative to stamp chroma; null if no coloured stamp ink was found",
        },
        "artifacts": {k: {"file": v, "sha256": sha256_file(out_dir / v)} for k, v in artifacts.items()},
        "environment": {
            "osv": osv.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
        },
    }
    if sha256_bytes(original_path.read_bytes()) != input_sha:
        raise RuntimeError("original file changed during the run")
    report["input"]["unchanged_after_run"] = True
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
