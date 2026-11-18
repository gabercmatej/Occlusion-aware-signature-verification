"""M2b.1 evaluation-only control: how cleaning and predicted exclusions change the E004 comparisons.

EVALUATION ONLY. Arm A below scores the *original* query without the cleaner. That path exists
only inside this module, for the experiment the owner approved on 16 November 2026 (Mendeley
dataset, frozen E004 comparisons). It is not a cleaner, it is not registered in
``osv.cleaning.METHODS``, and no product command (``clean``, ``compare-case``) can reach it:
product queries always pass through ``prepare_query``.

Three arms on identical comparisons (same frozen manifest, splits, protocol, references, k,
aggregation):

* **A — evaluation-only original-image control**: original query → normaliser → classical
  encoder; padding exclusion only. No cleaner and no cleaner-derived exclusion; arm A has no
  cleaner status of its own.
* **B — cleaned, all pixels**: ``prepare_query`` (selected cleaner) → the same encoder with
  padding exclusion only. Same computation as E004 ``all_pixels``.
* **C — cleaned with predicted exclusions**: the same prepared query with its exclusion mask.
  Same computation as E004 ``excluding_query_flagged``.

Normalisation runs on each arm's own input. A versus B is therefore the *total pipeline effect* of
cleaning (deleted pixels plus any change of content box and transform that the deletion causes),
not the effect of deletion alone. B versus C is the effect of predicted exclusions given cleaning.

Arms are joined by a stable comparison identity (partition, claimed writer, role, query writer,
query path and file hash), never by row position or count. B and C are recomputed and checked
against an archived E004 run, comparison by comparison.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from osv.cleaning import METHODS, clean_image
from osv.datasets.signature_folder import canonical_sha256
from osv.evaluation import metrics as M
from osv.evaluation.benchmark import (
    AGGREGATION, ENCODERS, KNOWN_LIMITATIONS, _count, _peak_rss_mb, aggregate, check_partitions, load_frozen,
)
from osv.evaluation.bootstrap import (
    DESCRIPTIVE_METRICS, fixed_threshold_metrics, paired_writer_bootstrap, writer_bootstrap,
)
from osv.preprocessing.image_io import load_image
from osv.preprocessing.normalise import NORMALISER_VERSION, normalise
from osv.preprocessing.stamp_removal import REMOVER_VERSION
from osv.verification.compare import COMPARISON_VERSION, EXCLUSION_MASK_RULE, prepare_query
from osv.verification.manifest import Entry

RESULT_SCHEMA = "osv-evaluation-control-1.0"
CONFIG_KIND = "evaluation_only_original_image_control"
CONTROL_LABEL = "evaluation-only original-image control"
BANNER = ("EVALUATION-ONLY CONTROL (M2b.1) — NOT A PRODUCT RESULT. Arm A scores original images without the cleaner "
          "for measurement only; product queries always pass through stamp removal.")
CLAIM_LIMIT = ("Genuine signatures and random impostors only; no skilled-forgery or stamp-robustness claim; no "
               "authenticity accuracy. Diagnostic on development/validation data already examined in E004; final "
               "test not scored.")
ARMS = {
    "A": {"name": "original_control", "label": CONTROL_LABEL,
          "input": "original query image (no cleaner)", "exclusion": "padding only",
          "cleaner": "not applied (" + CONTROL_LABEL + "); no cleaner status exists for this arm"},
    "B": {"name": "cleaned_all_pixels", "label": "cleaned, all pixels",
          "input": "query after the selected cleaner (prepare_query)", "exclusion": "padding only",
          "e004_rule": "all_pixels"},
    "C": {"name": "cleaned_predicted_exclusions", "label": "cleaned with predicted exclusions",
          "input": "query after the selected cleaner (prepare_query)",
          "exclusion": "padding and the cleaner's flagged pixels: " + EXCLUSION_MASK_RULE,
          "e004_rule": "excluding_query_flagged"},
}
PAIRS = (("A", "B"), ("B", "C"), ("A", "C"))
PAIR_MEANING = {
    ("A", "B"): "total pipeline effect of cleaning (deletion plus any normalisation change it causes), all pixels",
    ("B", "C"): "effect of the cleaner's predicted exclusions, given cleaning",
    ("A", "C"): "combined effect of cleaning and predicted exclusions",
}
KEY_FIELDS = ("partition", "claimed_writer", "kind", "query_writer", "query_path", "query_sha256")
TIMING_FIELDS = ("prepare_ms", "control_ms", "compare_ms")
COHORT_NOTE = ("common-defined cohort: comparisons for which every listed arm gives a defined score. It is selected by "
               "the arms' own refusals (and so by cleaner decisions), not at random; its metrics describe that subset "
               "only and must be read with its size and with the full-attempt outcomes.")
INK_TONE_NOTE = ("query ink tone from the dataset manifest: median (B - R) of dark pixels > 25 is 'blue-ish'. A "
                 "descriptive estimate, not a verified pen or stamp label.")
SOURCE_FILES = ("evaluation/control.py", "evaluation/benchmark.py", "evaluation/metrics.py", "evaluation/bootstrap.py",
                "verification/compare.py", "verification/encoder.py", "verification/pairs.py",
                "verification/descriptor.py", "preprocessing/normalise.py", "preprocessing/stamp_removal.py",
                "preprocessing/image_io.py", "segmentation/stamp_estimate.py", "segmentation/od_estimate.py",
                "cleaning.py")


class ControlConfigError(ValueError):
    """The configuration is not an evaluation-only control on the frozen E004 inputs."""


def comparison_key(row: dict) -> str:
    return "|".join(str(row[k]) for k in KEY_FIELDS)


# ------------------------------------------------------------------------------------------ scoring


def score_unit_control(unit: dict, root: str, encoder_name: str, method: str, partition: str,
                       ink_tone: dict) -> dict:
    """One claimed writer: every genuine and impostor query under arms A, B and C."""
    encoder = ENCODERS[encoder_name]()
    params = encoder.normalise_params
    base = Path(root)
    references, reference_checks, reference_sha = [], [], []
    for r in unit["references"]:  # identical to the E004 runner: checked by the cleaner, never modified
        loaded = load_image(base / r["path"])
        if loaded.sha256 != r["sha256"]:
            raise RuntimeError(f"{r['path']} changed after the protocol was frozen")
        _est, res = clean_image(loaded.rgb, method=method)
        references.append(normalise(loaded.rgb, params=params))
        reference_checks.append(res.status)
        reference_sha.append(r["sha256"])
    rows = []
    queries = [("genuine", unit["claimed_writer"], q) for q in unit["genuine_queries"]]
    queries += [("impostor", q["writer"], q) for q in unit["impostor_queries"]]
    for kind, query_writer, q in queries:
        t0 = time.perf_counter()
        entry = Entry(id="q", path=q["path"], sha256=q["sha256"], source_document_id=None, signing_instance_id=None,
                      label={}, resolved=base / q["path"])
        prepared = prepare_query(entry, method, params)  # verifies the file hash
        t1 = time.perf_counter()
        original = normalise(prepared.loaded.rgb, params=params)  # arm A: same decoded bytes, no cleaner
        t2 = time.perf_counter()
        excl = prepared.exclusion
        has_exclusion = bool(excl.any())
        a_res, b_res, c_res = [], [], []
        for ref in references:
            a_res.append(encoder.compare(original, ref))
            b = encoder.compare(prepared.normalised, ref)
            b_res.append(b)
            c_res.append(encoder.compare(prepared.normalised, ref, excl) if has_exclusion else b)
        t3 = time.perf_counter()
        nb = prepared.normalised
        row = {
            "partition": partition, "claimed_writer": unit["claimed_writer"], "kind": kind,
            "query_writer": query_writer, "query_path": q["path"], "query_sha256": q["sha256"],
            "reference_sha256": reference_sha, "query_ink_tone": ink_tone.get(q["path"]),
            "cleaner_status": prepared.result.status, "cleaner_reasons": list(prepared.result.reasons),
            "cleaner_changed_pixels": int(prepared.result.metrics.get("changed_pixels", 0)),
            "exclusion_share_of_canvas": round(float(excl.mean()), 6),
            "normalisation": {
                "A": {"content_box": None if original.content_box is None else list(original.content_box),
                      "content_flags": list(original.content_flags)},
                "B_C": {"content_box": None if nb.content_box is None else list(nb.content_box),
                        "content_flags": list(nb.content_flags)},
                "content_box_changed_by_cleaning": original.content_box != nb.content_box,
            },
            "arms": {
                "A": {"score": aggregate([r.distance for r in a_res]), "reference_statuses": [r.status for r in a_res]},
                "B": {"score": aggregate([r.distance for r in b_res]), "reference_statuses": [r.status for r in b_res]},
                "C": {"score": aggregate([r.distance for r in c_res]), "reference_statuses": [r.status for r in c_res]},
            },
            "prepare_ms": round(1000 * (t1 - t0), 3), "control_ms": round(1000 * (t2 - t1), 3),
            "compare_ms": round(1000 * (t3 - t2), 3),
        }
        row["comparison_key"] = comparison_key(row)
        rows.append(row)
    return {"claimed_writer": unit["claimed_writer"], "reference_check_statuses": reference_checks, "rows": rows,
            "peak_rss_mb": round(_peak_rss_mb(), 1)}


# ------------------------------------------------------------------------------------------ alignment


def arm_view(rows: list[dict], arm: str) -> list[dict]:
    """One arm's comparisons as flat records keyed by comparison identity."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    out = []
    for r in rows:
        rec = {k: r[k] for k in KEY_FIELDS}
        rec.update(comparison_key=comparison_key(r), score=r["arms"][arm]["score"],
                   reference_statuses=r["arms"][arm]["reference_statuses"], query_ink_tone=r.get("query_ink_tone"),
                   case_cleaner_status=r["cleaner_status"])
        rec["cleaner_status"] = None if arm == "A" else r["cleaner_status"]
        out.append(rec)
    return out


def _index(view: list[dict], name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rec in view:
        key = comparison_key(rec)
        if rec.get("comparison_key", key) != key:
            raise ValueError(f"{name}: stored comparison key does not match its fields: {key}")
        if key in out:
            raise ValueError(f"{name}: comparison {key} occurs twice")
        out[key] = rec
    return out


def align(*views: list[dict]) -> list[tuple[dict, ...]]:
    """Join arm views by comparison identity; refuse duplicates and any case missing from an arm.

    Matching row counts are not enough: the key sets must be identical. Output is ordered by key.
    """
    if len(views) < 2:
        raise ValueError("align needs at least two arms")
    indexed = [_index(v, f"arm {n}") for n, v in enumerate(views)]
    keys = set(indexed[0])
    for n, ix in enumerate(indexed[1:], start=1):
        if set(ix) != keys:
            missing, extra = sorted(keys - set(ix))[:3], sorted(set(ix) - keys)[:3]
            raise ValueError(f"arms 0 and {n} hold different comparisons (missing {missing}, extra {extra})")
    return [tuple(ix[k] for ix in indexed) for k in sorted(keys)]


def _writer_scores(records: list[dict], writers: list[str]) -> dict:
    out = {w: {"genuine": [], "impostor": []} for w in writers}
    for r in records:
        out[r["claimed_writer"]][r["kind"]].append(r["score"])
    return out


# ------------------------------------------------------------------------------------------ analysis


def _cov_block(records: list[dict]) -> dict:
    g = [r["score"] for r in records if r["kind"] == "genuine"]
    i = [r["score"] for r in records if r["kind"] == "impostor"]
    s = M.summarise(g, i)
    return {"genuine": {k: s["genuine"][k] for k in ("attempted", "usable", "undefined", "coverage")},
            "impostor": {k: s["impostor"][k] for k in ("attempted", "usable", "undefined", "coverage")},
            "genuine_median": (s["genuine"]["distribution"] or {}).get("median"),
            "impostor_median": (s["impostor"]["distribution"] or {}).get("median"),
            "eer": None if s["eer"] is None else s["eer"]["eer"], "auc": s["auc"]}


def arm_summary(records: list[dict], arm: str, boot: dict) -> dict:
    writers = sorted({r["claimed_writer"] for r in records})
    scores = _writer_scores(records, writers)
    g = [r["score"] for r in records if r["kind"] == "genuine"]
    i = [r["score"] for r in records if r["kind"] == "impostor"]
    block = M.summarise(g, i)
    block["arm"] = {"id": arm, **ARMS[arm]}
    block["claimed_writers"] = len(writers)
    block["undefined_reasons"] = _count("+".join(sorted(set(r["reference_statuses"]))) for r in records
                                        if r["score"] is None)
    block["bootstrap"] = writer_bootstrap(scores, DESCRIPTIVE_METRICS, **boot)
    statuses = sorted({r["case_cleaner_status"] for r in records})
    block["by_case_cleaner_status"] = {
        "note": ("cases grouped by the status the cleaner gives the same query in arms B/C; arm A itself is uncleaned"
                 if arm == "A" else "this arm's own cleaner status; review cases stay in every count"),
        "strata": {s: _cov_block([r for r in records if r["case_cleaner_status"] == s]) for s in statuses}}
    tones = sorted({str(r["query_ink_tone"]) for r in records})
    block["by_query_ink_tone"] = {"note": INK_TONE_NOTE,
                                  "strata": {t: _cov_block([r for r in records if str(r["query_ink_tone"]) == t])
                                             for t in tones}}
    return block


def threshold_transfer(dev: list[dict], val: list[dict], boot: dict) -> dict | None:
    """The arm's own development EER threshold (same rule for every arm), applied unchanged to validation."""
    dg = [r["score"] for r in dev if r["kind"] == "genuine"]
    di = [r["score"] for r in dev if r["kind"] == "impostor"]
    point = M.eer(*(M.split_defined(x)[0] for x in (dg, di)))
    if point is None:
        return None
    t = point["threshold"]
    writers = sorted({r["claimed_writer"] for r in val})
    vg = [r["score"] for r in val if r["kind"] == "genuine"]
    vi = [r["score"] for r in val if r["kind"] == "impostor"]
    return {"selection_rule": "development EER threshold: " + M.THRESHOLD_RULE_EER, "selected_on": "development",
            "applied_to": "validation", "development_point": point,
            "validation": {**M.operating_point(vg, vi, t),
                           "bootstrap": writer_bootstrap(_writer_scores(val, writers), fixed_threshold_metrics(t), **boot),
                           "interval_note": "threshold treated as fixed; its selection uncertainty is not propagated"}}


def _shift_metrics() -> dict:
    def med(arr):
        return None if arr.size == 0 else float(np.median(arr))

    def mean(arr):
        return None if arr.size == 0 else float(arr.mean())

    return {"genuine_median_shift": lambda g, i, ug, ui: med(g), "impostor_median_shift": lambda g, i, ug, ui: med(i),
            "genuine_mean_shift": lambda g, i, ug, ui: mean(g), "impostor_mean_shift": lambda g, i, ug, ui: mean(i)}


def _cohort_metrics() -> dict:
    return {"eer": DESCRIPTIVE_METRICS["eer"], "auc": DESCRIPTIVE_METRICS["auc"]}


def _attempt_metrics(threshold: float) -> dict:
    m = fixed_threshold_metrics(threshold)
    return {"far_attempted": m["far_attempted"], "genuine_not_accepted_attempted": m["genuine_not_accepted_attempted"]}


def _cohort(aligned: list[tuple[dict, ...]], writers: list[str], boot: dict, arm_ids: tuple[str, ...]) -> dict:
    keep = [t for t in aligned if all(r["score"] is not None for r in t)]
    per_arm = {a: [t[n] for t in keep] for n, a in enumerate(arm_ids)}
    out = {"arms": list(arm_ids), "note": COHORT_NOTE,
           "size": {k: sum(t[0]["kind"] == k for t in keep) for k in ("genuine", "impostor")},
           "attempted": {k: sum(t[0]["kind"] == k for t in aligned) for k in ("genuine", "impostor")},
           "claimed_writers_with_cohort_comparisons": len({t[0]["claimed_writer"] for t in keep}),
           "per_arm": {a: _cov_block(v) for a, v in per_arm.items()}}
    if len(arm_ids) == 2 and keep:
        out["paired_bootstrap_b_minus_a"] = paired_writer_bootstrap(
            _writer_scores(per_arm[arm_ids[0]], writers), _writer_scores(per_arm[arm_ids[1]], writers),
            _cohort_metrics(), **boot)
    return out


def paired_analysis(view_a: list[dict], view_b: list[dict], ids: tuple[str, str], boot: dict,
                    thr_a: float | None = None, thr_b: float | None = None, by_tone: bool = True) -> dict:
    """Paired effect of arm ``ids[1]`` against arm ``ids[0]`` on identical comparisons (differences are B - A)."""
    aligned = align(view_a, view_b)
    writers = sorted({t[0]["claimed_writer"] for t in aligned})
    a_recs, b_recs = [t[0] for t in aligned], [t[1] for t in aligned]
    sa, sb = _writer_scores(a_recs, writers), _writer_scores(b_recs, writers)
    out = {"arms": list(ids), "meaning": PAIR_MEANING.get(ids, ""), "difference": f"{ids[1]} - {ids[0]}",
           "comparisons": len(aligned)}
    cov = {k: v for k, v in DESCRIPTIVE_METRICS.items() if k.startswith("coverage")}
    out["coverage"] = paired_writer_bootstrap(sa, sb, cov, **boot)
    trans = {}
    for kind in ("genuine", "impostor"):
        c = {"both_defined": 0, "only_" + ids[0]: 0, "only_" + ids[1]: 0, "neither": 0}
        for a, b in aligned:
            if a["kind"] != kind:
                continue
            da, db = a["score"] is not None, b["score"] is not None
            c["both_defined" if da and db else ("only_" + ids[0]) if da else ("only_" + ids[1]) if db else "neither"] += 1
        trans[kind] = c
    out["definedness_transitions"] = trans
    both = [(a, b) for a, b in aligned if a["score"] is not None and b["score"] is not None]
    shifts = {"note": f"per-comparison score shift {ids[1]} - {ids[0]} where both arms give a score; positive = "
                      f"larger distance under {ids[1]}"}
    for kind in ("genuine", "impostor"):
        d = np.array([b["score"] - a["score"] for a, b in both if a["kind"] == kind], np.float64)
        shifts[kind] = {"n": int(d.size), "distribution": M.describe(d),
                        "unchanged_share": None if d.size == 0 else float((np.abs(d) <= 1e-12).mean())}
    delta_scores = {w: {"genuine": [], "impostor": []} for w in writers}
    for a, b in both:
        delta_scores[a["claimed_writer"]][a["kind"]].append(b["score"] - a["score"])
    shifts["bootstrap"] = writer_bootstrap(delta_scores, _shift_metrics(), **boot)
    out["score_shift_both_defined"] = shifts
    out["common_defined_cohort"] = _cohort(aligned, writers, boot, ids)
    if thr_a is not None and thr_b is not None:
        out["full_attempt_outcomes"] = {
            "note": ("each arm at its own development-selected threshold; every attempted comparison counts and an "
                     "undefined one is never accepted, so refusing hard cases cannot appear as an improvement"),
            "thresholds": {ids[0]: thr_a, ids[1]: thr_b},
            "paired_bootstrap_b_minus_a": paired_writer_bootstrap(sa, sb, _attempt_metrics(thr_a), **boot,
                                                                  metrics_b=_attempt_metrics(thr_b))}
    if by_tone:
        strata = {}
        for tone in sorted({str(t[0]["query_ink_tone"]) for t in aligned}):
            sub = [t for t in aligned if str(t[0]["query_ink_tone"]) == tone]
            sub_both = [(a, b) for a, b in sub if a["score"] is not None and b["score"] is not None]
            st = {"comparisons": len(sub), ids[0]: _cov_block([t[0] for t in sub]), ids[1]: _cov_block([t[1] for t in sub]),
                  "cohort": {"size": len(sub_both), ids[0]: _cov_block([a for a, _ in sub_both]),
                             ids[1]: _cov_block([b for _, b in sub_both])}}
            for kind in ("genuine", "impostor"):
                d = np.array([b["score"] - a["score"] for a, b in sub_both if a["kind"] == kind])
                st[f"{kind}_median_shift"] = None if d.size == 0 else float(np.median(d))
            strata[tone] = st
        out["by_query_ink_tone"] = {"note": INK_TONE_NOTE + " Descriptive strata, no intervals.", "strata": strata}
    return out


def normalisation_change(rows: list[dict]) -> dict:
    out = {}
    for s in sorted({r["cleaner_status"] for r in rows}):
        sub = [r for r in rows if r["cleaner_status"] == s]
        out[s] = {"queries": len(sub),
                  "content_box_changed_by_cleaning": sum(r["normalisation"]["content_box_changed_by_cleaning"] for r in sub),
                  "pixels_deleted": sum(r["cleaner_changed_pixels"] > 0 for r in sub)}
    return {"note": "A and B normalise their own inputs; a changed content box changes the canvas transform, so A vs B "
                    "is not deletion alone", "by_cleaner_status": out}


# ------------------------------------------------------------------------------------------ archive check


def check_against_archive(rows: list[dict], archive_rows: list[dict]) -> dict:
    """Compare recomputed B/C with archived E004 rows, joined by identity (E004 rows carry no file hash)."""
    def k(r):
        return "|".join(str(r[f]) for f in ("claimed_writer", "kind", "query_writer", "query_path"))
    ours, theirs = {}, {}
    for src, dst in ((rows, ours), (archive_rows, theirs)):
        for r in src:
            if k(r) in dst:
                raise ValueError(f"comparison {k(r)} occurs twice")
            dst[k(r)] = r
    out = {"compared": 0, "only_here": len(set(ours) - set(theirs)), "only_in_archive": len(set(theirs) - set(ours)),
           "mismatch": {"B_score": 0, "C_score": 0, "cleaner_status": 0, "reference_statuses": 0}}
    for key in sorted(set(ours) & set(theirs)):
        a, b = ours[key], theirs[key]
        out["compared"] += 1
        out["mismatch"]["B_score"] += a["arms"]["B"]["score"] != b["score"]["all_pixels"]
        out["mismatch"]["C_score"] += a["arms"]["C"]["score"] != b["score"]["excluding_query_flagged"]
        out["mismatch"]["cleaner_status"] += a["cleaner_status"] != b["cleaner_status"]
        out["mismatch"]["reference_statuses"] += (a["arms"]["B"]["reference_statuses"] != b["reference_statuses"]["all_pixels"]
                                                  or a["arms"]["C"]["reference_statuses"]
                                                  != b["reference_statuses"]["excluding_query_flagged"])
    out["identical"] = (out["only_here"] == 0 and out["only_in_archive"] == 0
                        and not any(out["mismatch"].values()))
    return out


# ------------------------------------------------------------------------------------------ provenance


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_record(base: Path) -> dict:
    src = Path(__file__).resolve().parents[1]
    files = {f: _file_sha(src / f) for f in SOURCE_FILES if (src / f).exists()}
    rec = {"source_sha256": files, "source_set_sha256": canonical_sha256(files),
           "comparison_version": COMPARISON_VERSION, "normaliser_version": NORMALISER_VERSION,
           "remover_version": REMOVER_VERSION}
    try:
        rec["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True,
                                         check=True).stdout.strip()
        rec["git_src_modified"] = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=src,
                                                 capture_output=True, text=True, check=True).stdout.split("\n")
        rec["git_src_modified"] = [x for x in rec["git_src_modified"] if x]
    except (OSError, subprocess.CalledProcessError):
        rec["git_head"] = None
    return rec


def check_config(config: dict, base: Path) -> dict:
    """Refuse anything that is not an evaluation-only control on exactly the frozen E004 inputs."""
    if config.get("kind") != CONFIG_KIND or config.get("evaluation_only") is not True:
        raise ControlConfigError(f"not an evaluation-only control configuration (kind must be {CONFIG_KIND!r} and "
                                 "evaluation_only true)")
    check_partitions(config["partitions"])
    if config["encoder"] not in ENCODERS:
        raise ControlConfigError(f"unknown encoder {config['encoder']!r}")
    if config["cleaner_method"] not in METHODS:
        raise ControlConfigError(f"arms B/C need a production cleaner; {config['cleaner_method']!r} is not one")
    ref = config["reuses_frozen_inputs_of"]
    frozen = json.loads((base / ref["config"]).read_text(encoding="utf-8"))
    if frozen.get("experiment_id") != ref["experiment_id"]:
        raise ControlConfigError("the referenced frozen configuration has a different experiment id")
    for field in ("paths", "sha256", "encoder", "cleaner_method", "frozen", "bootstrap", "dataset"):
        if config.get(field) != frozen.get(field):
            raise ControlConfigError(f"{field!r} differs from {ref['experiment_id']}; the control must reuse its frozen "
                                     "inputs unchanged")
    return frozen


# ------------------------------------------------------------------------------------------ runner


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_control(config_path, out_dir, workers: int = 1, base_dir=".", limit_writers: int | None = None) -> dict:
    """Score arms A/B/C on the frozen development/validation comparisons and write the paired analysis."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    base = Path(base_dir)
    frozen_config = check_config(config, base)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory {out_dir} is not empty; refusing to overwrite")
    manifest, splits, protocol = load_frozen(config, base)
    ink_tone = {r["path"]: r.get("ink_tone") for r in manifest["records"]}
    root = str(base / config["paths"]["dataset_root"])
    boot = {"n_resamples": config["bootstrap"]["n_resamples"], "seed": config["bootstrap"]["seed"],
            "level": config["bootstrap"]["level"]}
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_part, timing, ref_checks = {}, {}, {}
    for part in config["partitions"]:
        units = protocol["partitions"][part]
        if limit_writers is not None:
            units = units[:limit_writers]
        n = len(units)
        args = (units, [root] * n, [config["encoder"]] * n, [config["cleaner_method"]] * n, [part] * n, [ink_tone] * n)
        t0 = time.perf_counter()
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                scored = list(pool.map(score_unit_control, *args, chunksize=1))
        else:
            scored = [score_unit_control(*a) for a in zip(*args)]
        wall = time.perf_counter() - t0
        rows = [r for u in scored for r in u["rows"]]
        _index(rows, part)  # every comparison identity unique
        expected = sum(len(u["genuine_queries"]) + len(u["impostor_queries"]) for u in units)
        if len(rows) != expected:
            raise RuntimeError(f"{part}: {len(rows)} comparisons scored, protocol has {expected}")
        rows_by_part[part] = rows
        ref_checks[part] = _count(s for u in scored for s in u["reference_check_statuses"])
        timing[part] = {"wall_seconds": round(wall, 1), "workers": workers,
                        "peak_rss_mb_max_over_workers": max(u["peak_rss_mb"] for u in scored),
                        **{f: {"median_ms": round(float(np.median([r[f] for r in rows])), 2),
                               "p95_ms": round(float(np.percentile([r[f] for r in rows], 95)), 2)}
                           for f in TIMING_FIELDS}}
        with (out_dir / f"control_comparisons_{part}.local.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")

    result = analyse(rows_by_part, boot)
    archive = {}
    arch_dir = config.get("archived_reference_run")
    for part, rows in rows_by_part.items():
        path = base / arch_dir / f"comparisons_{part}.local.jsonl" if arch_dir else None
        if path is None or not path.exists():
            archive[part] = {"status": "archive not available; B/C recomputed only"}
        elif limit_writers is not None:
            archive[part] = {"status": "skipped in a smoke run"}
        else:
            archive[part] = check_against_archive(rows, _read_jsonl(path))
    result.update({
        "schema": RESULT_SCHEMA, "banner": BANNER, "evaluation_only": True, "product_result": False,
        "claim_limit": CLAIM_LIMIT, "experiment_id": config["experiment_id"],
        "smoke_run_limited_to_writers": limit_writers, "final_test": "not scored; not loaded by this runner",
        "known_limitations": list(KNOWN_LIMITATIONS) + [
            "Validation results of arms B/C were examined in E004 before this diagnostic was designed.",
            "Bootstrap resamples claimed writers; impostor images shared across claimed writers are not modelled.",
            "A vs B includes normalisation changes caused by cleaning, not deletion alone.",
            INK_TONE_NOTE],
        "config": config,
        "artifact_sha256": {**config["sha256"], "config": canonical_sha256(config),
                            "frozen_config": canonical_sha256(frozen_config)},
        "resolved": {"arms": ARMS, "aggregation": AGGREGATION, "score_direction": M.SCORE_DIRECTION,
                     "encoder": ENCODERS[config["encoder"]]().spec.record(),
                     "cleaner": {"method": config["cleaner_method"], "remover_version": REMOVER_VERSION},
                     "comparison_identity": list(KEY_FIELDS),
                     "protocol": {k: protocol[k] for k in ("rule", "seed", "k_references", "n_impostors_per_writer",
                                                           "impostor_type")},
                     "splits": {"rule": splits["rule"], "seed": splits["seed"], "counts": splits["counts"]}},
        "implementation": implementation_record(base),
        "archive_check_against": arch_dir, "archive_check": archive,
        "reference_cleaner_check_status": ref_checks,
        "timing": timing,
    })
    (out_dir / "results.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "control_table.md").write_text(render_table(result), encoding="utf-8")
    return result


def analyse(rows_by_part: dict[str, list[dict]], boot: dict) -> dict:
    """Arm summaries, threshold transfer and paired analyses; deterministic for fixed rows."""
    views = {p: {a: arm_view(rows, a) for a in ARMS} for p, rows in rows_by_part.items()}
    for p in views:  # every arm holds exactly the same comparisons
        align(*(views[p][a] for a in ARMS))
    partitions = {}
    for p, rows in rows_by_part.items():
        partitions[p] = {
            "arms": {a: arm_summary(views[p][a], a, boot) for a in ARMS},
            "cleaner_status": {k: _count(r["cleaner_status"] for r in rows if r["kind"] == k)
                               for k in ("genuine", "impostor")},
            "queries_with_pixels_deleted": sum(r["cleaner_changed_pixels"] > 0 for r in rows),
            "queries_with_nonempty_exclusion": sum(r["exclusion_share_of_canvas"] > 0 for r in rows),
            "normalisation_change": normalisation_change(rows),
        }
    thresholds = {}
    if "development" in views and "validation" in views:
        thresholds = {a: threshold_transfer(views["development"][a], views["validation"][a], boot) for a in ARMS}
    paired = {}
    for p in views:
        writers = sorted({r["claimed_writer"] for r in rows_by_part[p]})
        block = {}
        for a, b in PAIRS:
            ta = (thresholds.get(a) or {}).get("development_point", {}).get("threshold") if p == "validation" else None
            tb = (thresholds.get(b) or {}).get("development_point", {}).get("threshold") if p == "validation" else None
            block[f"{a}_vs_{b}"] = paired_analysis(views[p][a], views[p][b], (a, b), boot, ta, tb)
        block["common_defined_cohort_ABC"] = _cohort(align(*(views[p][a] for a in ARMS)), writers, boot, tuple(ARMS))
        paired[p] = block
    return {"partitions": partitions, "development_threshold_on_validation": thresholds, "paired": paired}


# ------------------------------------------------------------------------------------------ table


def _f(v, d=4) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def _iv(b: dict, d: int = 4) -> str:
    if b["estimate"] is None:
        return "undefined"
    return _f(b["estimate"], d) if b["low"] is None else f"{b['estimate']:.{d}f} [{b['low']:.{d}f}, {b['high']:.{d}f}]"


def _div(m: dict, d: int = 4) -> str:
    if m["difference_b_minus_a"] is None:
        return "undefined"
    s = f"{m['difference_b_minus_a']:+.{d}f}"
    return s if m["low"] is None else f"{s} [{m['low']:+.{d}f}, {m['high']:+.{d}f}]"


def render_table(result: dict) -> str:
    """Markdown report; the evaluation-only banner is its first line. No timing fields."""
    L = [f"**{result['banner']}**", "", f"_{result['claim_limit']}_", ""]
    if result.get("smoke_run_limited_to_writers"):
        L += [f"SMOKE RUN limited to {result['smoke_run_limited_to_writers']} writers per partition: not a result.", ""]
    L += [f"Experiment {result['experiment_id']}; frozen inputs of {result['config']['reuses_frozen_inputs_of']['experiment_id']} "
          f"(manifest {result['artifact_sha256']['manifest'][:12]}…, splits {result['artifact_sha256']['splits'][:12]}…, "
          f"protocol {result['artifact_sha256']['protocol'][:12]}…). Score = {result['resolved']['aggregation']}; "
          f"{result['resolved']['score_direction']}. Intervals: 95 % percentile over resampled claimed writers.", "",
          "Arms: " + "; ".join(f"**{a}** {v['label']} ({v['input']}; exclusion: {v['exclusion'].split(':')[0]})"
                               for a, v in result["resolved"]["arms"].items()) + ".",
          "A vs B is the total pipeline effect of cleaning, including normalisation changes, not deletion alone.", ""]
    for part, p in result["partitions"].items():
        L += [f"### {part}: per arm (all attempted comparisons)", "",
              "| Arm | Genuine attempted / defined / undefined | Impostor attempted / defined / undefined | "
              "Coverage g / i | Genuine median | Impostor median | EER (descriptive, defined only) | AUC (descriptive) |",
              "|---|---|---|---|---|---|---|---|"]
        for a, b in p["arms"].items():
            g, i, bs = b["genuine"], b["impostor"], b["bootstrap"]["metrics"]
            L.append(f"| {a} | {g['attempted']} / {g['usable']} / {g['undefined']} | {i['attempted']} / {i['usable']} / "
                     f"{i['undefined']} | {_f(g['coverage'], 3)} / {_f(i['coverage'], 3)} | "
                     f"{_f((g['distribution'] or {}).get('median'))} | {_f((i['distribution'] or {}).get('median'))} | "
                     f"{_iv(bs['eer'])} | {_iv(bs['auc'])} |")
        L += ["", f"Cleaner status of queries (B/C; review cases are in every count): genuine "
                  f"{p['cleaner_status']['genuine']}, impostor {p['cleaner_status']['impostor']}; queries with pixels "
                  f"deleted {p['queries_with_pixels_deleted']}; content box changed by cleaning: "
              + ", ".join(f"{s} {v['content_box_changed_by_cleaning']}/{v['queries']}"
                          for s, v in p["normalisation_change"]["by_cleaner_status"].items()) + ".", ""]
        L += ["Coverage and descriptive EER by the cleaner status of the same query (A is uncleaned; grouping only):", "",
              "| Status | Arm | Genuine defined / attempted | Impostor defined / attempted | EER | AUC |", "|---|---|---|---|---|---|"]
        for s in p["arms"]["A"]["by_case_cleaner_status"]["strata"]:
            for a in result["resolved"]["arms"]:
                st = p["arms"][a]["by_case_cleaner_status"]["strata"][s]
                L.append(f"| {s} | {a} | {st['genuine']['usable']} / {st['genuine']['attempted']} | "
                         f"{st['impostor']['usable']} / {st['impostor']['attempted']} | {_f(st['eer'])} | {_f(st['auc'])} |")
        L.append("")
    thr = result["development_threshold_on_validation"]
    if thr:
        L += ["### Each arm's development EER threshold applied unchanged to validation", "",
              "| Arm | Threshold | FAR (defined) | FRR (defined) | FAR (all attempted) | Genuine not accepted (all attempted) |",
              "|---|---|---|---|---|---|"]
        for a, t in thr.items():
            if t is None:
                L.append(f"| {a} | undefined | | | | |")
                continue
            vb = t["validation"]["bootstrap"]["metrics"]
            L.append(f"| {a} | {t['development_point']['threshold']:.6f} | {_iv(vb['far_usable'])} | {_iv(vb['frr_usable'])} | "
                     f"{_iv(vb['far_attempted'])} | {_iv(vb['genuine_not_accepted_attempted'])} |")
        L += ["", "Same selection rule for every arm; thresholds chosen on development only.", ""]
    for part, blocks in result["paired"].items():
        L += [f"### {part}: paired effects (difference = second arm − first arm, same comparisons)", "",
              "| Pair | Meaning | Δ coverage genuine | Δ coverage impostor | Both defined g / i | Median score shift g | "
              "Median score shift i |", "|---|---|---|---|---|---|---|"]
        for name, b in blocks.items():
            if name.startswith("common"):
                continue
            c, sh = b["coverage"]["metrics"], b["score_shift_both_defined"]["bootstrap"]["metrics"]
            L.append(f"| {name.replace('_vs_', ' → ')} | {b['meaning']} | {_div(c['coverage_genuine'], 3)} | "
                     f"{_div(c['coverage_impostor'], 3)} | {b['definedness_transitions']['genuine']['both_defined']} / "
                     f"{b['definedness_transitions']['impostor']['both_defined']} | {_iv(sh['genuine_median_shift'])} | "
                     f"{_iv(sh['impostor_median_shift'])} |")
        L += ["", "Common-defined cohorts (selected by the arms' own refusals; not a random sample):", "",
              "| Cohort | Size g / i (of attempted) | Arm | EER | AUC | Paired Δ EER | Paired Δ AUC |", "|---|---|---|---|---|---|---|"]
        for name, b in blocks.items():
            co = b if name.startswith("common") else b["common_defined_cohort"]
            label = "A∩B∩C" if name.startswith("common") else name.replace("_vs_", "∩")
            pb = co.get("paired_bootstrap_b_minus_a", {}).get("metrics")
            for n, a in enumerate(co["arms"]):
                st = co["per_arm"][a]
                d_eer = _div(pb["eer"]) if pb and n == 1 else ""
                d_auc = _div(pb["auc"]) if pb and n == 1 else ""
                size = (f"{co['size']['genuine']} / {co['size']['impostor']} (of {co['attempted']['genuine']} / "
                        f"{co['attempted']['impostor']})") if n == 0 else ""
                L.append(f"| {label if n == 0 else ''} | {size} | {a} | {_f(st['eer'])} | {_f(st['auc'])} | {d_eer} | {d_auc} |")
        L.append("")
        fa = [(n, b["full_attempt_outcomes"]) for n, b in blocks.items() if "full_attempt_outcomes" in b]
        if fa:
            L += ["Full-attempt outcomes at each arm's own development threshold (undefined = not accepted):", "",
                  "| Pair | Δ genuine not accepted (attempted) | Δ FAR (attempted) |", "|---|---|---|"]
            for n, f in fa:
                m = f["paired_bootstrap_b_minus_a"]["metrics"]
                L.append(f"| {n.replace('_vs_', ' → ')} | {_div(m['genuine_not_accepted_attempted'])} | "
                         f"{_div(m['far_attempted'])} |")
            L.append("")
        L += ["By query ink tone (descriptive estimate from the manifest, not a pen/stamp label; no intervals):", "",
              "| Pair | Tone | Comparisons | Coverage g first → second | Cohort size | Cohort EER first / second | "
              "Median shift g / i |", "|---|---|---|---|---|---|---|"]
        for name, b in blocks.items():
            if name.startswith("common"):
                continue
            x, y = b["arms"]
            for tone, st in b["by_query_ink_tone"]["strata"].items():
                L.append(f"| {name.replace('_vs_', ' → ')} | {tone} | {st['comparisons']} | "
                         f"{_f(st[x]['genuine']['coverage'], 3)} → {_f(st[y]['genuine']['coverage'], 3)} | "
                         f"{st['cohort']['size']} | {_f(st['cohort'][x]['eer'])} / {_f(st['cohort'][y]['eer'])} | "
                         f"{_f(st['genuine_median_shift'])} / {_f(st['impostor_median_shift'])} |")
        L.append("")
    L += ["### Reproduction check of B/C against the archived E004 run", ""]
    for part, a in result["archive_check"].items():
        L.append(f"- {part}: " + (a["status"] if "status" in a else
                                   f"{a['compared']} comparisons joined by identity; identical: {a['identical']}; "
                                   f"mismatches {a['mismatch']}; only here {a['only_here']}, only in archive "
                                   f"{a['only_in_archive']}"))
    L += ["", "Limits: " + " ".join(result["known_limitations"]), ""]
    return "\n".join(L)


def strip_timing(obj):
    """Copy of a result or row without timing fields, for reproducibility comparisons."""
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if k not in ("timing",) + TIMING_FIELDS
                and k not in ("peak_rss_mb",)}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj
