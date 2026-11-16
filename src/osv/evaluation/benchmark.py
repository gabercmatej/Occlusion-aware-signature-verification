"""Clean-signature benchmark runner (M2b): genuine versus random impostor on unseen writers.

One run scores the partitions named in the configuration against frozen, hashed inputs (dataset
manifest, writer splits, sampling protocol). The run refuses to start when a hash differs, and it
refuses the ``test`` partition outright: opening the final test split is a separate, owner-approved
step that this module does not implement.

Per comparison (one query against one claimed writer's ``k`` references):

* the query **always** goes through the selected cleaner (``prepare_query``), also when the
  result is a no-op (``no_stamp_detected``); there is no uncleaned path;
* references are checked by the cleaner but never modified (as in enrolment);
* the encoder gives one distance per reference under two support rules — ``all_pixels`` and
  ``excluding_query_flagged`` (the query's stamp / intervention / proposed / ambiguous pixels
  excluded); with an empty exclusion the two are the same computation and it is done once;
* the comparison score is the **median over references with a defined distance**; with no defined
  reference the comparison is *undefined* and stays in every count.

Review-status queries are scored like any other and reported separately as well; nothing is
dropped to improve a number.

Thresholds. ``development`` yields a descriptive EER and the threshold at that point. That
threshold, and only that one, is applied to ``validation`` (``development_threshold_on_validation``).
The EER computed on validation itself is labelled descriptive. Bootstrap intervals for the
applied threshold treat it as fixed (its own selection uncertainty is not propagated).
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import PIL

from osv.cleaning import METHODS, clean_image
from osv.datasets.signature_folder import canonical_sha256, check_splits, read_json
from osv.evaluation import metrics as M
from osv.evaluation.bootstrap import DESCRIPTIVE_METRICS, fixed_threshold_metrics, writer_bootstrap
from osv.evaluation.protocol import check_protocol
from osv.preprocessing.image_io import load_image
from osv.preprocessing.normalise import NORMALISER_VERSION, normalise
from osv.preprocessing.stamp_removal import REMOVER_VERSION
from osv.verification.compare import COMPARISON_VERSION, EXCLUSION_MASK_RULE, prepare_query
from osv.verification.encoder import ClassicalGridEncoder, Encoder
from osv.verification.manifest import Entry

RESULT_SCHEMA = "osv-clean-benchmark-1.0"
CLAIM_LIMIT = ("Genuine signatures and random impostors only; no skilled-forgery or stamp-robustness claim. "
               "Final test not scored.")
SCORABLE_PARTITIONS = ("development", "validation")
SCORE_RULES = ("excluding_query_flagged", "all_pixels")
AGGREGATION = "median over references with a defined distance; undefined when no reference gives a distance"
ENCODERS = {"classical_grid": ClassicalGridEncoder}
KNOWN_LIMITATIONS = (
    "Normaliser 0.2.0 accepted exception: one synthetic rescaling case gives self-distance 0.0319 against the "
    "unchanged 0.03 tolerance (strict expected failure in the suite); not re-tuned for this benchmark.",
    "Crop convention: images are used as distributed (the dataset's own crops, resized by its authors to a common "
    "size); the canvas is placed on each image's own dark-content box.",
    "All descriptor and evidence thresholds are provisional values chosen on fictional synthetic images.",
    "One dataset, one collection condition; the folder name is taken as the writer label.",
)


class FinalTestClosedError(ValueError):
    """The final test split was requested; this runner never scores it."""


def aggregate(distances: list[float | None]) -> float | None:
    vals = [d for d in distances if d is not None]
    return None if not vals else float(np.median(vals))


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def score_unit(unit: dict, root: str, encoder_name: str, method: str) -> dict:
    """Score one claimed writer: every genuine and impostor query against its references."""
    encoder: Encoder = ENCODERS[encoder_name]()
    params = encoder.normalise_params
    base = Path(root)
    references, reference_checks = [], []
    for r in unit["references"]:
        loaded = load_image(base / r["path"])
        if loaded.sha256 != r["sha256"]:
            raise RuntimeError(f"{r['path']} changed after the protocol was frozen")
        _est, res = clean_image(loaded.rgb, method=method)
        references.append(normalise(loaded.rgb, params=params))
        reference_checks.append(res.status)
    rows = []
    queries = [("genuine", unit["claimed_writer"], q) for q in unit["genuine_queries"]]
    queries += [("impostor", q["writer"], q) for q in unit["impostor_queries"]]
    for kind, query_writer, q in queries:
        t0 = time.perf_counter()
        entry = Entry(id="q", path=q["path"], sha256=q["sha256"], source_document_id=None, signing_instance_id=None,
                      label={}, resolved=base / q["path"])
        prepared = prepare_query(entry, method, params)
        t1 = time.perf_counter()
        excl = prepared.exclusion
        has_exclusion = bool(excl.any())
        all_rows, masked_rows = [], []
        for ref in references:
            a = encoder.compare(prepared.normalised, ref)
            all_rows.append(a)
            masked_rows.append(encoder.compare(prepared.normalised, ref, excl) if has_exclusion else a)
        t2 = time.perf_counter()
        metrics = prepared.result.metrics
        rows.append({
            "claimed_writer": unit["claimed_writer"], "kind": kind, "query_writer": query_writer, "query_path": q["path"],
            "cleaner_status": prepared.result.status, "cleaner_reasons": list(prepared.result.reasons),
            "cleaner_changed_pixels": int(metrics.get("changed_pixels", 0)),
            "exclusion_share_of_canvas": round(float(excl.mean()), 6),
            "content_flags": list(prepared.normalised.content_flags),
            "score": {"excluding_query_flagged": aggregate([r.distance for r in masked_rows]),
                      "all_pixels": aggregate([r.distance for r in all_rows])},
            "reference_statuses": {"excluding_query_flagged": [r.status for r in masked_rows],
                                   "all_pixels": [r.status for r in all_rows]},
            "prepare_ms": round(1000 * (t1 - t0), 3), "compare_ms": round(1000 * (t2 - t1), 3),
        })
    return {"claimed_writer": unit["claimed_writer"], "reference_check_statuses": reference_checks, "rows": rows,
            "peak_rss_mb": round(_peak_rss_mb(), 1)}


def writer_scores(rows: list[dict], rule: str, keep=None) -> dict:
    out: dict[str, dict[str, list]] = {}
    for r in rows:
        slot = out.setdefault(r["claimed_writer"], {"genuine": [], "impostor": []})
        if keep is None or keep(r):
            slot[r["kind"]].append(r["score"][rule])
    return out


def _pooled(scores: dict) -> tuple[list, list]:
    return ([s for w in sorted(scores) for s in scores[w]["genuine"]],
            [s for w in sorted(scores) for s in scores[w]["impostor"]])


def _count(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _undefined_reasons(rows: list[dict], rule: str) -> dict:
    return _count("+".join(sorted(set(r["reference_statuses"][rule]))) for r in rows if r["score"][rule] is None)


def summarise_partition(rows: list[dict], boot: dict) -> dict:
    """Descriptive summary of one partition for both score rules, with writer-bootstrap intervals."""
    out = {"comparisons": {k: sum(r["kind"] == k for r in rows) for k in ("genuine", "impostor")},
           "claimed_writers": len({r["claimed_writer"] for r in rows}),
           "query_cleaner_status": {k: _count(r["cleaner_status"] for r in rows if r["kind"] == k)
                                    for k in ("genuine", "impostor")},
           "query_cleaner_reasons": _count(reason for r in rows for reason in r["cleaner_reasons"]),
           "queries_with_pixels_deleted": sum(r["cleaner_changed_pixels"] > 0 for r in rows),
           "queries_with_nonempty_exclusion": sum(r["exclusion_share_of_canvas"] > 0 for r in rows),
           "query_content_flags": _count(f.split(":")[0] for r in rows for f in r["content_flags"]),
           "by_score_rule": {}}
    for rule in SCORE_RULES:
        scores = writer_scores(rows, rule)
        g, i = _pooled(scores)
        block = M.summarise(g, i)
        block["undefined_reasons"] = _undefined_reasons(rows, rule)
        block["bootstrap"] = writer_bootstrap(scores, DESCRIPTIVE_METRICS, **boot)
        strata = {}
        for status in sorted({r["cleaner_status"] for r in rows}):
            sg, si = _pooled(writer_scores(rows, rule, keep=lambda r, s=status: r["cleaner_status"] == s))
            s = M.summarise(sg, si)
            strata[status] = {"genuine": {k: s["genuine"][k] for k in ("attempted", "usable", "undefined", "coverage")},
                              "impostor": {k: s["impostor"][k] for k in ("attempted", "usable", "undefined", "coverage")},
                              "genuine_median": (s["genuine"]["distribution"] or {}).get("median"),
                              "impostor_median": (s["impostor"]["distribution"] or {}).get("median"),
                              "eer": None if s["eer"] is None else s["eer"]["eer"], "auc": s["auc"]}
        block["by_query_cleaner_status"] = {"note": "descriptive strata of the same comparisons; not a control experiment",
                                            "strata": strata}
        out["by_score_rule"][rule] = block
    return out


def apply_threshold(rows: list[dict], rule: str, threshold: float, boot: dict) -> dict:
    scores = writer_scores(rows, rule)
    g, i = _pooled(scores)
    return {**M.operating_point(g, i, threshold),
            "bootstrap": writer_bootstrap(scores, fixed_threshold_metrics(threshold), **boot),
            "interval_note": "threshold treated as fixed; its selection uncertainty is not propagated"}


def _timing(rows: list[dict], units: list[dict], wall: float, workers: int) -> dict:
    def stat(key):
        v = np.array([r[key] for r in rows])
        return {"median_ms": round(float(np.median(v)), 2), "p95_ms": round(float(np.percentile(v, 95)), 2),
                "max_ms": round(float(v.max()), 2)}
    return {"prepare_query": stat("prepare_ms"), "compare_query_to_references": stat("compare_ms"),
            "wall_seconds_partition": round(wall, 1), "workers": workers,
            "peak_rss_mb_max_over_workers": max(u["peak_rss_mb"] for u in units),
            "scope": "prepare_query = load + selected cleaner + normalise of one query image; "
                     "compare = that query against k references under both support rules (once when the exclusion is "
                     "empty); measured per query inside worker processes running concurrently, so values include "
                     "contention. Peak RSS = largest ru_maxrss of any worker process (interpreter and libraries "
                     "included). Reference preparation and metric computation are outside both timers."}


def load_frozen(config: dict, base: Path) -> tuple[dict, dict, dict]:
    """Load manifest, splits and protocol, verifying the frozen hashes and their mutual consistency."""
    manifest = read_json(base / config["paths"]["manifest"], config["sha256"]["manifest"])
    splits = read_json(base / config["paths"]["splits"], config["sha256"]["splits"])
    protocol = read_json(base / config["paths"]["protocol"], config["sha256"]["protocol"])
    check_splits(splits, manifest)
    check_protocol(protocol, splits, manifest)
    return manifest, splits, protocol


def check_partitions(partitions) -> None:
    for p in partitions:
        if p == "test":
            raise FinalTestClosedError("the final test partition is closed: this runner does not score it")
        if p not in SCORABLE_PARTITIONS:
            raise ValueError(f"unknown partition {p!r}; scorable partitions are {SCORABLE_PARTITIONS}")


def run_benchmark(config_path, out_dir, workers: int = 1, base_dir=".", limit_writers: int | None = None) -> dict:
    """Score the configured partitions and write results; ``limit_writers`` is for smoke tests only."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    check_partitions(config["partitions"])
    if config["encoder"] not in ENCODERS:
        raise ValueError(f"unknown encoder {config['encoder']!r}; available: {tuple(ENCODERS)}")
    if config["cleaner_method"] not in METHODS:
        raise ValueError(f"unknown cleaner method {config['cleaner_method']!r}")
    if config["primary_score"] not in SCORE_RULES:
        raise ValueError(f"primary_score must be one of {SCORE_RULES}")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory {out_dir} is not empty; refusing to overwrite")
    base = Path(base_dir)
    manifest, splits, protocol = load_frozen(config, base)
    root = str(base / config["paths"]["dataset_root"])
    encoder = ENCODERS[config["encoder"]]()
    boot = {"n_resamples": config["bootstrap"]["n_resamples"], "seed": config["bootstrap"]["seed"],
            "level": config["bootstrap"]["level"]}
    out_dir.mkdir(parents=True, exist_ok=True)

    partitions, all_rows = {}, {}
    for part in config["partitions"]:
        units = protocol["partitions"][part]
        if limit_writers is not None:
            units = units[:limit_writers]
        t0 = time.perf_counter()
        args = (units, [root] * len(units), [config["encoder"]] * len(units), [config["cleaner_method"]] * len(units))
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                scored = list(pool.map(score_unit, *args, chunksize=1))
        else:
            scored = [score_unit(*a) for a in zip(*args)]
        wall = time.perf_counter() - t0
        rows = [r for u in scored for r in u["rows"]]
        all_rows[part] = rows
        summary = summarise_partition(rows, boot)
        summary["reference_cleaner_check_status"] = _count(s for u in scored for s in u["reference_check_statuses"])
        summary["timing"] = _timing(rows, scored, wall, workers)
        partitions[part] = summary
        with (out_dir / f"comparisons_{part}.local.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")

    thresholds = {}
    if "development" in partitions and "validation" in partitions:
        for rule in SCORE_RULES:
            dev_eer = partitions["development"]["by_score_rule"][rule]["eer"]
            if dev_eer is None:
                thresholds[rule] = None
                continue
            thresholds[rule] = {
                "selection_rule": "development EER threshold: " + M.THRESHOLD_RULE_EER,
                "selected_on": "development", "applied_to": "validation",
                "development_point": dev_eer,
                "validation": apply_threshold(all_rows["validation"], rule, dev_eer["threshold"], boot),
            }

    probe_rng = np.random.default_rng(0)
    probe_canvas = (encoder.normalise_params.canvas_height, encoder.normalise_params.canvas_width)
    probe_mask = np.zeros(probe_canvas, bool)
    probe_mask[probe_canvas[0] // 4: probe_canvas[0] // 2, probe_canvas[1] // 3: probe_canvas[1] // 2] = True
    first = protocol["partitions"][config["partitions"][0]][0]
    pq = normalise(load_image(Path(root) / first["genuine_queries"][0]["path"]).rgb, params=encoder.normalise_params)
    pr = normalise(load_image(Path(root) / first["references"][0]["path"]).rgb, params=encoder.normalise_params)
    invariance = encoder.invariance_check(pq, pr, probe_mask, probe_rng).record()

    result = {
        "schema": RESULT_SCHEMA,
        "claim_limit": CLAIM_LIMIT,
        "experiment_id": config.get("experiment_id"),
        "smoke_run_limited_to_writers": limit_writers,
        "final_test": "not scored; not loaded by this runner",
        "known_limitations": list(KNOWN_LIMITATIONS),
        "config": config,
        "resolved": {
            "encoder": encoder.spec.record(),
            "cleaner": {"method": config["cleaner_method"], "remover_version": REMOVER_VERSION,
                        "query_path": "every query through prepare_query (cleaner always runs); references checked, never modified",
                        "exclusion_mask_rule": EXCLUSION_MASK_RULE},
            "comparison_version": COMPARISON_VERSION, "normaliser_version": NORMALISER_VERSION,
            "aggregation": AGGREGATION, "score_direction": M.SCORE_DIRECTION, "primary_score": config["primary_score"],
            "protocol": {k: protocol[k] for k in ("rule", "seed", "k_references", "n_impostors_per_writer", "impostor_type")},
            "splits": {"rule": splits["rule"], "seed": splits["seed"], "counts": splits["counts"]},
            "dataset": manifest["dataset"], "dataset_counts": manifest["counts"],
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                            "pillow": PIL.__version__, "machine": platform.machine(), "system": platform.system()},
        },
        "artifact_sha256": {**config["sha256"], "config": canonical_sha256(config)},
        "encoder_invariance_check": {**invariance, "probe": "one development query/reference pair, rectangular exclusion"},
        "partitions": partitions,
        "development_threshold_on_validation": thresholds,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "validation_table.md").write_text(render_table(result), encoding="utf-8")
    return result


def _iv(b: dict, digits: int = 4) -> str:
    if b["estimate"] is None:
        return "undefined"
    if b["low"] is None:
        return f"{b['estimate']:.{digits}f}"
    return f"{b['estimate']:.{digits}f} [{b['low']:.{digits}f}, {b['high']:.{digits}f}]"


def render_table(result: dict) -> str:
    """Markdown summary; the claim limit is its first line."""
    cfg, res = result["config"], result["resolved"]
    lines = [f"**{result['claim_limit']}**", ""]
    if result.get("smoke_run_limited_to_writers"):
        lines += [f"SMOKE RUN limited to {result['smoke_run_limited_to_writers']} writers per partition: not a result.", ""]
    lines += [f"Experiment {result['experiment_id']} — encoder `{res['encoder']['name']}` {res['encoder']['version']}, "
              f"cleaner `{cfg['cleaner_method']}` (remover {res['cleaner']['remover_version']}), "
              f"normaliser {res['normaliser_version']}; k = {res['protocol']['k_references']} references; "
              f"score = {res['aggregation']}; {res['score_direction']}.",
              f"Intervals: {int(100 * cfg['bootstrap']['level'])} % percentile, {cfg['bootstrap']['n_resamples']} "
              f"resamples of claimed writers (seed {cfg['bootstrap']['seed']}).", ""]
    for rule in SCORE_RULES:
        tag = " (primary)" if rule == cfg["primary_score"] else ""
        lines += [f"### Score rule `{rule}`{tag}", "",
                  "| Partition | Writers | Genuine attempted / usable | Impostor attempted / usable | Genuine median | "
                  "Impostor median | EER (descriptive) | AUC (descriptive) |", "|---|---|---|---|---|---|---|---|"]
        for part, p in result["partitions"].items():
            b, bs = p["by_score_rule"][rule], p["by_score_rule"][rule]["bootstrap"]["metrics"]
            gm = (b["genuine"]["distribution"] or {}).get("median")
            im = (b["impostor"]["distribution"] or {}).get("median")
            lines.append(f"| {part} | {p['claimed_writers']} | {b['genuine']['attempted']} / {b['genuine']['usable']} | "
                         f"{b['impostor']['attempted']} / {b['impostor']['usable']} | "
                         f"{'n/a' if gm is None else f'{gm:.4f}'} | {'n/a' if im is None else f'{im:.4f}'} | "
                         f"{_iv(bs['eer'])} | {_iv(bs['auc'])} |")
        lines.append("")
        t = result["development_threshold_on_validation"].get(rule)
        if t:
            v, vb = t["validation"], t["validation"]["bootstrap"]["metrics"]
            lines += [f"Development-selected threshold {t['development_point']['threshold']:.6f} "
                      f"(development EER point) applied to validation:", "",
                      "| FAR (usable) | FRR (usable) | FAR (attempted) | Genuine not accepted (attempted) | "
                      "Undefined genuine | Undefined impostor |", "|---|---|---|---|---|---|",
                      f"| {_iv(vb['far_usable'])} | {_iv(vb['frr_usable'])} | {_iv(vb['far_attempted'])} | "
                      f"{_iv(vb['genuine_not_accepted_attempted'])} | {v['genuine']['undefined']} | "
                      f"{v['impostor']['undefined']} |", "",
                      "The EER and AUC in the validation row above are derived from validation itself (descriptive). "
                      "Undefined comparisons are never accepted and are excluded from the usable rates.", ""]
    lines += ["### Query cleaner status (all queries passed through the cleaner)", "",
              "| Partition | Kind | " + " | ".join(("no_stamp_detected", "cleaned", "review")) + " | Pixels deleted (queries) |",
              "|---|---|---|---|---|---|"]
    for part, p in result["partitions"].items():
        for kind in ("genuine", "impostor"):
            c = p["query_cleaner_status"][kind]
            lines.append(f"| {part} | {kind} | " + " | ".join(str(c.get(s, 0)) for s in ("no_stamp_detected", "cleaned", "review"))
                         + (f" | {p['queries_with_pixels_deleted']} (both kinds) |" if kind == "genuine" else " | |"))
    lines += ["", "Limits: " + " ".join(result["known_limitations"]), ""]
    return "\n".join(lines)
