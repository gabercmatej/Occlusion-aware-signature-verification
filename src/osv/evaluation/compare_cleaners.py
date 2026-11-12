"""Repeatable comparison of cleaning methods on the M1b synthetic matrix (evaluation only).

Each cleaner receives only the case image. Scenario, mixture, geometry, seed and ground truth
are attached to the result row *after* cleaning, for grouping and metrics.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

import osv
from osv.cleaning import METHODS, clean_image
from osv.datasets.synthetic_matrix import (
    DEV_SEEDS, GEOMETRIES, HELDOUT_SEEDS, MATRIX_VERSION, MIXTURES, SCENARIOS, make_case, matrix_cases,
)
from osv.evaluation.cleaning_metrics import case_metrics

SPLITS = {"dev": DEV_SEEDS, "heldout": HELDOUT_SEEDS}
DAMAGE_PX = 1  # any changed handwriting pixel counts as damage in the case counts


def run_case(scenario: str, mixture: str, geometry: str, seed: int, method: str) -> dict:
    fx = make_case(scenario, mixture, geometry, seed)
    image = fx.image.copy()
    t0 = time.perf_counter()
    est, res = clean_image(image, method=method)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    m = case_metrics(fx, res.intervention_mask, res.change_map, est.ambiguous_mask)
    m.update({
        "scenario": scenario, "group": SCENARIOS[scenario]["group"], "mixture": mixture,
        "geometry": geometry, "seed": seed, "method": method, "status": res.status,
        "action": est.action, "reasons": list(res.reasons),
        "separability": None if est.separability is None else round(float(est.separability), 6),
        "diagnostics": {k: v for k, v in est.stats.items() if not isinstance(v, (list, dict))},
        "layer_order": fx.meta.get("layer_order"),
        "runtime_ms": round(runtime_ms, 3),
    })
    return m


def run_matrix(split: str, methods=tuple(METHODS)) -> list[dict]:
    rows = []
    for case in matrix_cases(SPLITS[split]):
        for method in methods:
            rows.append(run_case(*case, method))
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return None if not vals else float(np.mean(vals))


def _max(vals):
    vals = [v for v in vals if v is not None]
    return None if not vals else float(np.max(vals))


def summarize_rows(rows: list[dict]) -> dict:
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    stamped = [r for r in rows if r["stamp_only_pixels"] > 0]
    return {
        "n": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "stamp_only_removed_mean": _mean([r["stamp_only_removed"] for r in stamped]),
        "stamp_only_unflagged_mean": _mean([r["stamp_only_unflagged"] for r in stamped]),
        "pen_loss_outside_overlap_mean": _mean([r["pen_loss_outside_overlap"] for r in rows]),
        "pen_loss_outside_overlap_max": _max([r["pen_loss_outside_overlap"] for r in rows]),
        "pen_loss_in_overlap_mean": _mean([r["pen_loss_in_overlap"] for r in rows]),
        "pen_changed_pixels_total": int(sum(r["pen_changed_pixels"] for r in rows)),
        "cases_with_pen_damage": int(sum(r["pen_changed_pixels"] >= DAMAGE_PX for r in rows)),
        "cases_with_pen_damage_outside_overlap": int(sum(r["pen_changed_outside_overlap_pixels"] >= DAMAGE_PX for r in rows)),
        "cases_with_any_change": int(sum(r["changed_pixels"] > 0 for r in rows)),
        "mark_changed_pixels_total": int(sum(r["mark_changed_pixels"] for r in rows)),
        "runtime_ms_median": float(np.median([r["runtime_ms"] for r in rows])) if rows else None,
    }


def summarize(rows: list[dict]) -> dict:
    def group_by(keys):
        out: dict[str, dict] = {}
        buckets: dict[tuple, list] = {}
        for r in rows:
            buckets.setdefault(tuple(r[k] for k in keys), []).append(r)
        for k, rs in sorted(buckets.items()):
            out["|".join(map(str, k))] = summarize_rows(rs)
        return out

    return {
        "by_method": group_by(["method"]),
        "by_group_mixture_method": group_by(["group", "mixture", "method"]),
        "by_scenario_mixture_method": group_by(["scenario", "mixture", "method"]),
        "by_geometry_method": group_by(["geometry", "method"]),
    }


def _fmt(v, digits=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _status(c: dict) -> str:
    short = {"cleaned": "C", "review": "R", "no_stamp_detected": "N"}
    return " ".join(f"{short.get(k, k)}{v}" for k, v in c.items())


def markdown(summary: dict, split: str) -> str:
    head = ("| {key} | method | n | status C/R/N | stamp-only removed | unflagged residue | "
            "pen loss outside overlap (mean / max) | pen loss in overlap | cases with pen damage (any / outside overlap) | "
            "cases changed | runtime ms (median) |")
    lines = [f"# M1b cleaner comparison — split `{split}`", "",
             "Synthetic only (stage E0). C = cleaned, R = review, N = no_stamp_detected. Fractions are of "
             "ground-truth pixels; removal and residue are averaged over stamped cases, counting review/no-op "
             "cases as zero removal. Pen damage counts cases with at least one changed handwriting pixel.", ""]

    def table(title, section, key_name, key_cols):
        lines.extend([f"## {title}", "", head.format(key=key_name), "|" + "---|" * 11])
        for k, s in summary[section].items():
            parts = k.split("|")
            key = " / ".join(parts[:key_cols]) if key_cols else "all"
            method = parts[-1]
            lines.append(
                f"| {key} | {method} | {s['n']} | {_status(s['status_counts'])} | {_fmt(s['stamp_only_removed_mean'])} | "
                f"{_fmt(s['stamp_only_unflagged_mean'])} | {_fmt(s['pen_loss_outside_overlap_mean'], 4)} / "
                f"{_fmt(s['pen_loss_outside_overlap_max'], 4)} | {_fmt(s['pen_loss_in_overlap_mean'])} | "
                f"{s['cases_with_pen_damage']} / {s['cases_with_pen_damage_outside_overlap']} | {s['cases_with_any_change']} | "
                f"{_fmt(s['runtime_ms_median'], 1)} |")
        lines.append("")

    table("By difficulty group and mixture model", "by_group_mixture_method", "group / mixture", 2)
    table("By scenario and mixture model", "by_scenario_mixture_method", "scenario / mixture", 2)
    table("By stamp geometry", "by_geometry_method", "geometry", 1)
    table("Overall", "by_method", "all", 0)
    return "\n".join(lines) + "\n"


def config_record(split: str, methods) -> dict:
    from osv.segmentation import od_estimate, stamp_estimate
    from osv.preprocessing import stamp_removal

    return {
        "split": split,
        "seeds": list(SPLITS[split]),
        "seed_groups": {"dev": list(DEV_SEEDS), "heldout": list(HELDOUT_SEEDS)},
        "matrix_version": MATRIX_VERSION,
        "mixtures": list(MIXTURES),
        "geometries": list(GEOMETRIES),
        "scenarios": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in SCENARIOS.items()},
        "methods": {
            "chroma": {"estimator": stamp_estimate.ESTIMATOR_NAME, "version": stamp_estimate.ESTIMATOR_VERSION,
                       "parameters": asdict(stamp_estimate.EstimatorParams())},
            "optical-density": {"estimator": od_estimate.ESTIMATOR_NAME, "version": od_estimate.ESTIMATOR_VERSION,
                                "parameters": asdict(od_estimate.ODParams())},
        },
        "methods_run": list(methods),
        "remover": {"name": stamp_removal.REMOVER_NAME, "version": stamp_removal.REMOVER_VERSION},
        "reviewer_region": None,
        "inputs_to_cleaner": "case image only",
        "environment": {
            "osv": osv.__version__, "python": platform.python_version(), "numpy": np.__version__,
            "opencv": cv2.__version__, "machine": platform.machine(), "system": platform.system(),
            "opencv_threads": cv2.getNumThreads(),
        },
    }


def write_comparison(split: str, out_dir, methods=tuple(METHODS)) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = run_matrix(split, methods)
    summary = summarize(rows)
    (out / "config.json").write_text(json.dumps(config_record(split, methods), indent=2, sort_keys=True) + "\n")
    (out / "cases.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "summary.md").write_text(markdown(summary, split))
    return summary
