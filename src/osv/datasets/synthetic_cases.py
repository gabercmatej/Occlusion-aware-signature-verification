"""Fictional reference/query cases for exercising the M2a pipeline (software tests only).

A fictional "writer" is a seed that fixes the shape parameters of smooth cursive-like strokes;
each "signing instance" perturbs them slightly (phase, amplitude, scale, position, width).
These are not anyone's signature and do not model real writer variation or forgery, so results
on them establish software behaviour only, never writer discrimination or forgery rejection.

Drawing helpers and the stamp model are reused from :mod:`osv.datasets.synthetic_fixtures`
(unchanged); composition is the same multiplicative mixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from osv.datasets.synthetic_fixtures import _SS, _downsample, _draw_curve, _smooth_noise, _stamp_alpha
from osv.verification.manifest import CONFIG_VERSION, MANIFEST_VERSION

CASES_VERSION = "0.1.0"
_SIZE = (240, 480)
_PEN = (32, 32, 38)
_RED_STAMP = (200, 42, 52)


def _writer_params(writer_seed: int) -> dict:
    rng = np.random.default_rng([writer_seed, 104729])
    return {
        "x0": rng.uniform(0.08, 0.18), "x1": rng.uniform(0.78, 0.92), "base": rng.uniform(0.45, 0.58),
        "freq": rng.uniform(5.0, 8.0), "loop_amp": rng.uniform(0.045, 0.07), "phase1": rng.uniform(0, 2 * np.pi),
        "vamp": rng.uniform(0.12, 0.2), "phase2": rng.uniform(0, 2 * np.pi), "slope": rng.uniform(-0.08, 0.08),
        "width": rng.uniform(1.1, 1.6), "fx0": rng.uniform(0.15, 0.35), "fx1": rng.uniform(0.7, 0.95),
        "fy": rng.uniform(0.2, 0.28), "ax": rng.uniform(0.3, 0.7),
    }


def signature_alpha(writer_seed: int, instance_seed: int, size: tuple[int, int] = _SIZE) -> np.ndarray:
    h, w = size
    p = _writer_params(writer_seed)
    rng = np.random.default_rng([writer_seed, instance_seed, 15485863])
    j = lambda sd: float(rng.normal(0.0, sd))  # noqa: E731
    scale = 1.0 + j(0.03)
    dx, dy = j(3.0), j(3.0)
    canvas = np.zeros((h * _SS, w * _SS), np.uint8)

    def place(xs, ys):
        return (xs - w / 2) * scale + w / 2 + dx, (ys - h / 2) * scale + h / 2 + dy

    t = np.linspace(0.0, 1.0, 4000)
    freq = p["freq"] * (1 + j(0.01))
    x0, x1, base = p["x0"] * w, p["x1"] * w, p["base"] * h
    xs = x0 + (x1 - x0) * t + p["loop_amp"] * w * (1 + j(0.04)) * np.sin(2 * np.pi * freq * t + p["phase1"] + j(0.08))
    ys = base + p["vamp"] * h * (1 + j(0.04)) * np.sin(2 * np.pi * freq * t + p["phase2"] + j(0.08)) + p["slope"] * h * np.sin(np.pi * t)
    width = p["width"] * (1 + j(0.05))
    _draw_curve(canvas, *place(xs, ys), width + 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)))

    t2 = np.linspace(0.0, 1.0, 1500)
    xs2 = p["fx0"] * w + (p["fx1"] - p["fx0"]) * w * t2
    ys2 = base + p["fy"] * h - 0.05 * h * t2 + j(1.5)
    _draw_curve(canvas, *place(xs2, ys2), 0.9 + 0.8 * np.sin(np.pi * t2))

    t3 = np.linspace(0.0, 1.0, 300)
    ax, ay = p["ax"] * w + j(2.0), base - 0.28 * h + j(2.0)
    _draw_curve(canvas, *place(ax + 14 * t3, ay - 6 * t3), np.full_like(t3, 1.3))
    return _downsample(canvas, h, w)


def render_instance(writer_seed: int, instance_seed: int, stamped: bool = False) -> np.ndarray:
    h, w = _SIZE
    pen = signature_alpha(writer_seed, instance_seed)
    rng = np.random.default_rng([writer_seed, instance_seed, 7, int(stamped)])
    img = np.array([243, 240, 232], np.float32)[None, None, :] * (1.0 + 0.025 * (_smooth_noise(rng, h, w, 60) - 0.5))[..., None]
    img = img * (np.array(_PEN, np.float32) / 255.0)[None, None, :] ** pen[..., None]
    if stamped:
        stamp = _stamp_alpha(rng, h, w, pen)
        img = img * (np.array(_RED_STAMP, np.float32) / 255.0)[None, None, :] ** stamp[..., None]
    img = cv2.GaussianBlur(img, (0, 0), 0.6) + rng.normal(0.0, 1.5, img.shape).astype(np.float32)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(rel: str, base: Path, label: dict, doc: str | None, inst: str | None, eid: str) -> dict:
    return {"id": eid, "path": rel, "sha256": _sha(base / rel), "source_document_id": doc,
            "signing_instance_id": inst, "label": label}


def write_demo_case(out_dir, n_references: int = 4) -> list[Path]:
    """Write fictional images and two manifests into ``out_dir``; returns the manifest paths.

    References: fictional writer A, instances 0..n-1, unstamped. Queries: writer A instance 10
    and fictional writer B instance 0, both with a red stamp. Both query labels are ``unknown``.
    """
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory {out} is not empty; refusing to overwrite")
    (out / "references").mkdir(parents=True)
    (out / "queries").mkdir()
    ref_label = {"status": "known", "value": "genuine", "source": "fictional synthetic instance (software test only)"}
    unknown = {"status": "unknown", "value": None, "source": None}
    refs = []
    for i in range(n_references):
        rel = f"references/A_{i}.png"
        Image.fromarray(render_instance(1, i)).save(out / rel, format="PNG")
        refs.append(_entry(rel, out, ref_label, f"doc-A-{i}", f"sig-A-{i}", f"A{i}"))
    manifests = []
    for name, (writer, inst) in {"query_1": (1, 10), "query_2": (2, 0)}.items():
        rel = f"queries/{name}.png"
        Image.fromarray(render_instance(writer, inst, stamped=True)).save(out / rel, format="PNG")
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "case_id": f"demo-{name}",
            "claimed_writer_id": "W-DEMO-A",
            "config": {"config_version": CONFIG_VERSION, "cleaner_method": "chroma"},
            "references": refs,
            "query": {**_entry(rel, out, unknown, None, None, name), "stamp_region": None},
        }
        path = out / f"case_{name}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests.append(path)
    (out / "README.txt").write_text(
        "Fictional synthetic case generated by osv.datasets.synthetic_cases "
        f"{CASES_VERSION}. query_1 is drawn from the same fictional writer seed as the references, "
        "query_2 from a different seed. Software demonstration only; not evidence of writer "
        "discrimination or forgery rejection.\n", encoding="utf-8")
    return manifests
