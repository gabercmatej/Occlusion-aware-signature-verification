"""Explicit image loading for the comparison pipeline (M2a).

Supported inputs (decided by the decoded format, not the file extension):

* PNG, JPEG, BMP, and single-frame TIFF.
* Pixel modes ``1``, ``L``, ``P``, ``RGB``; ``LA``, ``PA`` and ``RGBA`` only when fully opaque.

Refused with an error, never silently reduced:

* Multi-frame files (multipage TIFF, animated images): split pages explicitly first.
* PDF: page rendering needs a renderer dependency that is not installed; deferred.
* 16-bit and floating-point modes, CMYK, and transparent pixels (no paper colour can be assumed).

Coordinate convention (all M2a records): ``(x, y)`` in pixels of the *display-oriented* image,
i.e. after the EXIF orientation tag has been applied. The origin is the top-left corner of the
top-left pixel, x grows to the right, y grows down, and pixel ``[row i, column j]`` covers
``[j, j+1) x [i, i+1)`` with its centre at ``(j + 0.5, i + 0.5)``. Reviewer rectangles
(``x, y, w, h``) use the same grid.

The file on disk is only read; its bytes are hashed before and after decoding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_FORMATS = ("PNG", "JPEG", "BMP", "TIFF")
SUPPORTED_MODES = ("1", "L", "P", "RGB", "LA", "PA", "RGBA")
COORDINATE_CONVENTION = (
    "x right, y down, origin at the top-left corner of the top-left pixel of the display-oriented "
    "image (EXIF orientation applied); pixel [i, j] covers [j, j+1) x [i, i+1)"
)
EXIF_ORIENTATION_TAG = 0x0112


class UnsupportedImageError(ValueError):
    """The file is readable but outside the explicitly supported inputs."""


@dataclass(frozen=True)
class LoadedImage:
    rgb: np.ndarray  # (H, W, 3) uint8, display-oriented
    sha256: str
    size_bytes: int
    format: str
    mode: str
    stored_size: tuple[int, int]  # (width, height) as stored, before orientation
    exif_orientation: int  # 1 when absent
    orientation_applied: bool

    def record(self) -> dict:
        h, w = self.rgb.shape[:2]
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "mode": self.mode,
            "stored_width": self.stored_size[0],
            "stored_height": self.stored_size[1],
            "exif_orientation": self.exif_orientation,
            "orientation_applied": self.orientation_applied,
            "width": int(w),
            "height": int(h),
        }


def load_image(path) -> LoadedImage:
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if data[:5] == b"%PDF-":
        raise UnsupportedImageError(f"{path.name}: PDF rendering is deferred (needs a renderer dependency)")
    try:
        opened = Image.open(path)
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError(f"{path.name}: not a decodable image") from exc
    with opened as im:
        fmt = im.format or "unknown"
        if fmt == "PDF":
            raise UnsupportedImageError(f"{path.name}: PDF rendering is deferred (needs a renderer dependency)")
        if fmt not in SUPPORTED_FORMATS:
            raise UnsupportedImageError(f"{path.name}: format {fmt} is not supported; expected one of {SUPPORTED_FORMATS}")
        frames = int(getattr(im, "n_frames", 1))
        if frames > 1:
            raise UnsupportedImageError(
                f"{path.name}: {frames} frames/pages; multipage files are not processed, split pages explicitly"
            )
        mode = im.mode
        if mode not in SUPPORTED_MODES:
            raise UnsupportedImageError(f"{path.name}: pixel mode {mode} is not supported; expected one of {SUPPORTED_MODES}")
        orientation = int(im.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
        stored_size = (int(im.width), int(im.height))
        oriented = ImageOps.exif_transpose(im)
        if mode in ("LA", "PA", "RGBA") or (mode == "P" and "transparency" in im.info):
            alpha = np.array(oriented.convert("RGBA"))[..., 3]
            if (alpha < 255).any():
                raise UnsupportedImageError(f"{path.name}: transparent pixels; no paper colour can be assumed")
        rgb = np.ascontiguousarray(np.array(oriented.convert("RGB"), dtype=np.uint8))
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise RuntimeError(f"{path.name} changed while it was being read")
    return LoadedImage(
        rgb=rgb,
        sha256=sha,
        size_bytes=len(data),
        format=fmt,
        mode=mode,
        stored_size=stored_size,
        exif_orientation=orientation,
        orientation_applied=orientation != 1,
    )
