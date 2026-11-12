"""Command line entry point.

    python -m osv.cli clean <image> --out <dir> [--stamp-region x,y,w,h]
    python -m osv.cli make-fixture <kind> --seed N --out <file.png>
    python -m osv.cli measure-fixtures [--seeds 10] [--json <file>]
    python -m osv.cli fixture-figure --out <file.png>

``clean`` reads only the image and the optional region. ``make-fixture`` writes only the
image; ground-truth masks stay inside the measurement code and are never written next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from osv.datasets.synthetic_fixtures import KINDS, cleaning_metrics, make_fixture
from osv.preprocessing.stamp_removal import remove_stamp
from osv.provenance import load_rgb, sha256_file, write_artifacts
from osv.segmentation.stamp_estimate import estimate_stamp


def parse_region(text: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x,y,w,h")
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("x,y,w,h must be integers") from exc


def cmd_clean(args: argparse.Namespace) -> int:
    path = Path(args.image)
    sha_before = sha256_file(path)
    rgb, _mode = load_rgb(path)
    estimate = estimate_stamp(rgb, stamp_region=args.stamp_region)
    result = remove_stamp(rgb, estimate)
    report = write_artifacts(path, estimate, result, args.out)
    if report["input"]["sha256"] != sha_before:
        raise RuntimeError("input changed between reading and writing artifacts")
    m = result.metrics
    print(f"status: {result.status}")
    print(f"meaning: {report['status_meaning']}")
    print(f"reasons: {', '.join(result.reasons) or '-'}")
    print(f"changed pixels: {m['changed_pixels']} of {m['image_pixels']}")
    print(f"ink reliable/ambiguous/changed: {m['ink_fraction_reliable']} / {m['ink_fraction_ambiguous']} / {m['ink_fraction_changed']}")
    print(f"artifacts: {args.out}")
    return 0


def cmd_make_fixture(args: argparse.Namespace) -> int:
    fx = make_fixture(args.kind, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fx.image).save(out, format="PNG")
    print(f"wrote {out} ({args.kind}, seed {args.seed})")
    return 0


def measure(seeds: int) -> dict:
    rows: dict[str, list[dict]] = {}
    for kind in KINDS:
        for seed in range(seeds):
            fx = make_fixture(kind, seed)
            est = estimate_stamp(fx.image)
            res = remove_stamp(fx.image, est)
            m = cleaning_metrics(fx, res.intervention_mask, res.change_map, est.ambiguous_mask)
            m.update(status=res.status, reasons=list(res.reasons), seed=seed)
            rows.setdefault(kind, []).append(m)
    return rows


def _summ(values: list) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    return f"{np.mean(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]"


def cmd_measure(args: argparse.Namespace) -> int:
    rows = measure(args.seeds)
    print(f"Synthetic fixtures, seeds 0-{args.seeds - 1}, no reviewer region. Mean [min, max].")
    print("| kind | statuses | stamp-only removed | stamp-only unflagged residue | pen loss outside overlap | pen loss in overlap | overlap marked ambiguous | pen changed away from stamp (px, max) |")
    print("|---|---|---|---|---|---|---|---|")
    for kind, rs in rows.items():
        statuses = {}
        for r in rs:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        st = ", ".join(f"{k} x{v}" for k, v in sorted(statuses.items()))
        print(
            f"| {kind} | {st} | {_summ([r['stamp_only_removed'] for r in rs])} | {_summ([r['stamp_only_unflagged'] for r in rs])} | "
            f"{_summ([r['pen_loss_outside_overlap'] for r in rs])} | {_summ([r['pen_loss_in_overlap'] for r in rs])} | "
            f"{_summ([r['overlap_marked_ambiguous'] for r in rs])} | {max(r['pen_changed_away_from_stamp'] for r in rs)} |"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _tile(img: np.ndarray, label: str) -> np.ndarray:
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    bar = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _overlay(base: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    out = (base.astype(np.float32) * 0.35 + 165).astype(np.uint8)
    out[mask] = colour
    return out


def cmd_figure(args: argparse.Namespace) -> int:
    rows = []
    for kind, seed in (("red_stamp_black_pen", 0), ("blue_stamp_blue_pen", 0), ("black_stamp_black_pen", 0)):
        fx = make_fixture(kind, seed)
        est = estimate_stamp(fx.image)
        res = remove_stamp(fx.image, est)
        change = np.where(res.change_map > 0, 0, 255).astype(np.uint8)
        tiles = [
            _tile(fx.image, f"original: {kind}"),
            _tile(_overlay(fx.image, est.stamp_mask, (220, 0, 120)), "stamp estimate"),
            _tile(_overlay(fx.image, est.ambiguous_mask, (240, 150, 0)), "ambiguous (kept)"),
            _tile(change, f"change map ({int((res.change_map > 0).sum())} px)"),
            _tile(res.cleaned, f"output, status: {res.status}"),
        ]
        sep = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
        row = tiles[0]
        for t in tiles[1:]:
            row = np.hstack([row, sep, t])
        rows.append(row)
    gap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    sheet = rows[0]
    for r in rows[1:]:
        sheet = np.vstack([sheet, gap, r])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out, format="PNG")
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="osv", description="Occlusion-aware signature verification tools (M1a).")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("clean", help="estimate and conservatively remove a coloured stamp; write provenance")
    c.add_argument("image")
    c.add_argument("--out", required=True, help="output directory (must be new or empty)")
    c.add_argument("--stamp-region", type=parse_region, default=None, metavar="x,y,w,h",
                   help="optional reviewer-marked stamp rectangle in pixels")
    c.set_defaults(func=cmd_clean)

    f = sub.add_parser("make-fixture", help="write one synthetic fixture image (image only)")
    f.add_argument("kind", choices=KINDS)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_make_fixture)

    m = sub.add_parser("measure-fixtures", help="measure the cleaner against fixture ground truth")
    m.add_argument("--seeds", type=int, default=10)
    m.add_argument("--json", default=None)
    m.set_defaults(func=cmd_measure)

    g = sub.add_parser("fixture-figure", help="write the README comparison figure")
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_figure)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
