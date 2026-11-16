"""M2b: dataset manifest, writer splits, sampling protocol, metrics, bootstrap, encoder contract, runner.

All images here are fictional synthetic signatures; these tests establish software behaviour only.
Hand-computed expectations are written out next to each toy example.
"""

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from osv.cli import main
from osv.datasets.signature_folder import (
    PARTITIONS, build_manifest, canonical_sha256, check_splits, make_splits, read_json, scan, write_json,
)
from osv.datasets.synthetic_cases import render_instance
from osv.evaluation import metrics as M
from osv.evaluation.benchmark import (
    CLAIM_LIMIT, FinalTestClosedError, aggregate, render_table, run_benchmark, summarise_partition,
)
from osv.evaluation.bootstrap import (
    DESCRIPTIVE_METRICS, fixed_threshold_metrics, paired_writer_bootstrap, writer_bootstrap,
)
from osv.evaluation.protocol import build_protocol, check_protocol
from osv.preprocessing.normalise import normalise
from osv.verification.descriptor import GridOrientationDescriptor
from osv.verification.encoder import (
    DISTANCE_DIRECTION, ClassicalGridEncoder, Encoder, EncoderSpec, MaskingSpec, PairResult,
)
from osv.verification.pairs import compare_pair

G = np.array([0.1, 0.2, 0.3, 0.6])
I = np.array([0.25, 0.5, 0.7, 0.9])


# ---------------------------------------------------------------- metrics

def test_far_frr_direction_accepts_at_or_below_threshold():
    # t = 0.25: impostors <= t: {0.25} -> FAR 1/4; genuine > t: {0.3, 0.6} -> FRR 2/4
    assert M.far_frr(G, I, 0.25) == (0.25, 0.5)
    # equality counts as accepted on both sides
    assert M.far_frr(np.array([0.3]), np.array([0.3]), 0.3) == (1.0, 0.0)


def test_eer_on_hand_computed_example():
    # thresholds:   <0.1   0.1   0.2   0.25  0.3   0.5 ...
    # FAR:           0     0     0     1/4   1/4   2/4
    # FRR:           1     3/4   2/4   2/4   1/4   1/4
    # first threshold with the smallest |FAR - FRR| is 0.3, where both are 1/4
    r = M.eer(G, I)
    assert r == {"eer": 0.25, "threshold": 0.3, "far": 0.25, "frr": 0.25}


def test_auc_on_hand_computed_example_and_direction():
    # genuine below each impostor: 0.25 -> 2, 0.5 -> 3, 0.7 -> 4, 0.9 -> 4; 13 of 16 pairs
    assert M.auc(G, I) == 13 / 16
    # scores the wrong way round (similarities passed as distances) fall below chance
    assert M.auc(I, G) == 3 / 16
    assert M.auc(np.array([0.5]), np.array([0.5])) == 0.5  # a tie counts one half


def test_perfect_and_empty_cases():
    g, i = np.array([0.1, 0.2]), np.array([0.8, 0.9])
    assert M.eer(g, i)["eer"] == 0.0 and M.auc(g, i) == 1.0
    assert M.eer(np.empty(0), i) is None and M.auc(g, np.empty(0)) is None


def test_undefined_scores_are_never_acceptances():
    genuine = [0.1, None, 0.6]  # one accepted, one undefined, one rejected at t = 0.3
    impostor = [0.2, None, 0.9, float("nan")]  # one accepted, two undefined, one rejected
    op = M.operating_point(genuine, impostor, 0.3)
    assert op["far_usable"] == 1 / 2 and op["frr_usable"] == 1 / 2
    assert op["far_attempted"] == 1 / 4  # undefined impostors are not accepted
    assert op["genuine_not_accepted_attempted"] == 2 / 3  # the undefined genuine counts against the system
    assert op["genuine"] == {"attempted": 3, "usable": 2, "undefined": 1, "accepted": 1}
    assert op["impostor"] == {"attempted": 4, "usable": 2, "undefined": 2, "accepted": 1}


def test_summary_reports_coverage_beside_descriptive_metrics():
    s = M.summarise([0.1, None, 0.3], [0.5, 0.9, None, None])
    assert s["genuine"]["attempted"] == 3 and s["genuine"]["usable"] == 2 and s["genuine"]["coverage"] == 2 / 3
    assert s["impostor"]["undefined"] == 2 and s["impostor"]["coverage"] == 0.5
    assert s["eer"]["eer"] == 0.0 and s["auc"] == 1.0  # on defined scores only; coverage says what is missing
    all_undefined = M.summarise([None], [None])
    assert all_undefined["eer"] is None and all_undefined["auc"] is None and all_undefined["genuine"]["coverage"] == 0.0


# ---------------------------------------------------------------- bootstrap

def _toy_scores(seed=0, n=12, shift=0.3, missing=False):
    rng = np.random.default_rng(seed)
    out = {}
    for w in range(n):
        g = list(rng.normal(0.3, 0.08, 6))
        i = list(rng.normal(0.3 + shift, 0.08, 6))
        if missing and w % 4 == 0:
            g[0] = None
        out[f"w{w:02d}"] = {"genuine": g, "impostor": i}
    return out


def test_writer_bootstrap_is_reproducible_and_seed_dependent():
    scores = _toy_scores(shift=0.1, missing=True)
    a = writer_bootstrap(scores, DESCRIPTIVE_METRICS, n_resamples=200, seed=5)
    b = writer_bootstrap(scores, DESCRIPTIVE_METRICS, n_resamples=200, seed=5)
    c = writer_bootstrap(scores, DESCRIPTIVE_METRICS, n_resamples=200, seed=6)
    assert a == b
    assert a["metrics"]["eer"]["low"] != c["metrics"]["eer"]["low"] or a["metrics"]["auc"]["high"] != c["metrics"]["auc"]["high"]
    for m in a["metrics"].values():
        assert m["low"] <= m["estimate"] <= m["high"]
    assert a["metrics"]["coverage_genuine"]["estimate"] == (72 - 3) / 72
    assert a["n_writers"] == 12 and "writer" in a["resampling_unit"]


def test_bootstrap_resamples_writers_not_comparisons():
    # every writer is internally constant, so a writer-level resample can only produce pooled
    # values that are mixtures of whole writers: with one writer the interval collapses.
    one = {"w": {"genuine": [0.1, 0.2, 0.3], "impostor": [0.7, 0.8, 0.9]}}
    r = writer_bootstrap(one, fixed_threshold_metrics(0.25), n_resamples=50, seed=1)
    assert r["metrics"]["frr_usable"] == {"estimate": 1 / 3, "low": 1 / 3, "high": 1 / 3, "undefined_replicates": 0}


def test_bootstrap_counts_replicates_where_a_metric_is_undefined():
    scores = {"a": {"genuine": [0.1], "impostor": [None]}, "b": {"genuine": [None], "impostor": [0.9]}}
    r = writer_bootstrap(scores, DESCRIPTIVE_METRICS, n_resamples=100, seed=3)
    assert r["metrics"]["eer"]["estimate"] == 0.0
    assert r["metrics"]["eer"]["undefined_replicates"] > 0  # resamples of only "a" or only "b"


def test_paired_bootstrap_identical_systems_give_zero_difference():
    scores = _toy_scores()
    r = paired_writer_bootstrap(scores, scores, DESCRIPTIVE_METRICS, n_resamples=100, seed=2)
    e = r["metrics"]["eer"]
    assert e["difference_b_minus_a"] == 0.0 and e["low"] == 0.0 and e["high"] == 0.0
    assert e["interval_excludes_zero"] is False


def test_paired_bootstrap_detects_a_clearly_better_system_and_is_reproducible():
    worse, better = _toy_scores(shift=0.05), _toy_scores(shift=0.6)
    r1 = paired_writer_bootstrap(worse, better, DESCRIPTIVE_METRICS, n_resamples=300, seed=4)
    r2 = paired_writer_bootstrap(worse, better, DESCRIPTIVE_METRICS, n_resamples=300, seed=4)
    assert r1 == r2
    assert r1["metrics"]["eer"]["high"] < 0 and r1["metrics"]["eer"]["interval_excludes_zero"] is True
    assert r1["metrics"]["auc"]["low"] > 0


def test_paired_bootstrap_refuses_unpaired_inputs():
    a = _toy_scores()
    b = dict(a)
    b.pop("w00")
    with pytest.raises(ValueError, match="same claimed writers"):
        paired_writer_bootstrap(a, b, DESCRIPTIVE_METRICS, n_resamples=10)
    c = {w: {"genuine": v["genuine"][:-1], "impostor": v["impostor"]} for w, v in a.items()}
    with pytest.raises(ValueError, match="different numbers"):
        paired_writer_bootstrap(a, c, DESCRIPTIVE_METRICS, n_resamples=10)


# ---------------------------------------------------------------- manifest, duplicates, splits

def _records(n_writers=20, per_writer=6):
    recs = []
    for w in range(n_writers):
        for k in range(per_writer):
            tag = f"w{w:02d}-{k}"
            recs.append({"path": f"w{w:02d}/{k}.png", "writer": f"w{w:02d}", "problem": None,
                         "sha256": canonical_sha256(tag), "pixel_sha256": canonical_sha256("px" + tag)})
    return recs


def test_duplicates_within_a_writer_keep_the_first_file_only():
    recs = _records(3)
    recs[2]["pixel_sha256"] = recs[0]["pixel_sha256"]  # w00/2 repeats w00/0
    m = build_manifest(recs, {"name": "toy"})
    status = {r["path"]: r["excluded"] for r in m["records"]}
    assert status["w00/0.png"] is None and status["w00/2.png"] == "duplicate_within_writer"
    assert m["counts"]["excluded"] == {"duplicate_within_writer": 1} and m["counts"]["eligible_files"] == 17
    assert all(len(g) == 1 for g in m["writer_groups"])


def test_duplicates_across_writers_are_excluded_and_tie_the_writers_together():
    recs = _records(20)
    by_path = {r["path"]: r for r in recs}
    by_path["w07/1.png"]["pixel_sha256"] = by_path["w03/4.png"]["pixel_sha256"]
    by_path["w11/0.png"]["pixel_sha256"] = by_path["w07/5.png"]["pixel_sha256"]
    m = build_manifest(recs, {"name": "toy"})
    status = {r["path"]: r["excluded"] for r in m["records"]}
    for p in ("w03/4.png", "w07/1.png", "w07/5.png", "w11/0.png"):
        assert status[p] == "duplicate_across_writers"  # no copy survives: the label is ambiguous
    assert ["w03", "w07", "w11"] in m["writer_groups"]
    for seed in range(25):  # the tied writers always land in one partition
        splits = make_splits(m, seed)
        check_splits(splits, m)
        homes = {p for p in PARTITIONS for w in ("w03", "w07", "w11") if w in splits["partitions"][p]}
        assert len(homes) == 1


def test_splits_are_writer_disjoint_complete_and_deterministic():
    m = build_manifest(_records(40), {"name": "toy"})
    a, b = make_splits(m, 7), make_splits(m, 7)
    assert a == b and canonical_sha256(a) == canonical_sha256(b)
    parts = [set(a["partitions"][p]) for p in PARTITIONS]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert parts[0] | parts[1] | parts[2] == {f"w{w:02d}" for w in range(40)}
    assert a["counts"] == {"development": 20, "validation": 10, "test": 10}
    assert make_splits(m, 8)["partitions"] != a["partitions"]
    # record order in the input does not matter
    assert make_splits(build_manifest(list(reversed(_records(40))), {"name": "toy"}), 7) == a


def test_check_splits_rejects_a_writer_in_two_partitions():
    m = build_manifest(_records(8), {"name": "toy"})
    s = make_splits(m, 1)
    s["partitions"]["test"].append(s["partitions"]["development"][0])
    with pytest.raises(ValueError, match="is in both"):
        check_splits(s, m)


def test_writers_with_too_few_images_are_left_out_of_every_partition():
    recs = [r for r in _records(10) if not (r["writer"] == "w04" and r["path"] > "w04/2.png")]
    m = build_manifest(recs, {"name": "toy"})
    s = make_splits(m, 3, min_images=4)
    assert s["writers_left_out"] == ["w04"]
    assert all("w04" not in s["partitions"][p] for p in PARTITIONS)


def test_frozen_json_is_refused_when_its_content_changes(tmp_path):
    m = build_manifest(_records(4), {"name": "toy"})
    sha = write_json(tmp_path / "m.json", m)
    assert read_json(tmp_path / "m.json", sha) == m
    m["records"][0]["writer"] = "someone-else"
    (tmp_path / "m.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="differs from the frozen value"):
        read_json(tmp_path / "m.json", sha)


# ---------------------------------------------------------------- protocol

def _frozen(n_writers=24, per_writer=6, seed=11, k=3, n_imp=4):
    m = build_manifest(_records(n_writers, per_writer), {"name": "toy"})
    s = make_splits(m, seed, min_images=k + 1)
    return m, s, build_protocol(m, s, k, n_imp, seed)


def test_references_and_queries_never_share_an_image():
    m, s, p = _frozen()
    check_protocol(p, s, m)
    for part in PARTITIONS:
        all_refs = {r["path"] for u in p["partitions"][part] for r in u["references"]}
        for u in p["partitions"][part]:
            refs = {r["path"] for r in u["references"]}
            genuine = {q["path"] for q in u["genuine_queries"]}
            assert len(refs) == 3 and len(genuine) == 3 and not (refs & genuine)
            assert all(q["path"].startswith(u["claimed_writer"] + "/") for q in u["genuine_queries"])
            assert not ({q["path"] for q in u["impostor_queries"]} & all_refs)


def test_impostors_come_from_distinct_other_writers_of_the_same_partition():
    m, s, p = _frozen()
    for part in PARTITIONS:
        members = set(s["partitions"][part])
        for u in p["partitions"][part]:
            writers = [q["writer"] for q in u["impostor_queries"]]
            assert len(writers) == len(set(writers)) == 4
            assert u["claimed_writer"] not in writers and set(writers) <= members


def test_protocol_is_deterministic_and_seed_dependent():
    _, _, a = _frozen(seed=11)
    _, _, b = _frozen(seed=11)
    _, _, c = _frozen(seed=12)
    assert canonical_sha256(a) == canonical_sha256(b) != canonical_sha256(c)


def test_check_protocol_rejects_leaks():
    m, s, p = _frozen()
    unit = p["partitions"]["development"][0]
    leaked = json.loads(json.dumps(p))
    leaked["partitions"]["development"][0]["genuine_queries"][0] = dict(unit["references"][0])
    with pytest.raises(ValueError, match="is also a reference"):
        check_protocol(leaked, s, m)
    foreign = json.loads(json.dumps(p))
    other = p["partitions"]["validation"][0]["genuine_queries"][0]
    foreign["partitions"]["development"][0]["impostor_queries"][0] = {
        "writer": p["partitions"]["validation"][0]["claimed_writer"], **other}
    with pytest.raises(ValueError, match="not from another writer of development"):
        check_protocol(foreign, s, m)


def test_protocol_refuses_a_writer_without_query_images():
    m = build_manifest(_records(8, per_writer=3), {"name": "toy"})
    with pytest.raises(ValueError, match="needs more than k=3"):
        build_protocol(m, make_splits(m, 1), 3, 2, 1)


# ---------------------------------------------------------------- scanning real files

def _write_dataset(root, n_writers=12, per_writer=6):
    for w in range(n_writers):
        d = root / f"writer_{w:02d}"
        d.mkdir(parents=True)
        for k in range(per_writer):
            Image.fromarray(render_instance(100 + w, k)).save(d / f"{k}.png")


def test_scan_flags_blank_truncated_and_reencoded_duplicates(tmp_path):
    root = tmp_path / "data"
    _write_dataset(root, n_writers=2, per_writer=3)
    w0 = root / "writer_00"
    Image.fromarray(np.full((80, 120, 3), 240, np.uint8)).save(w0 / "blank.png")
    data = (w0 / "0.png").read_bytes()
    (w0 / "truncated.png").write_bytes(data[: len(data) // 2])
    Image.open(w0 / "1.png").save(w0 / "z_copy.bmp")  # same pixels, different bytes
    (w0 / "notes.txt").write_text("not an image")
    m = build_manifest(scan(root), {"name": "toy"})
    status = {r["path"]: r["excluded"] for r in m["records"]}
    assert status["writer_00/blank.png"] == "blank_image"
    assert status["writer_00/truncated.png"] == "undecodable"
    assert status["writer_00/z_copy.bmp"] == "duplicate_within_writer" and status["writer_00/1.png"] is None
    assert "writer_00/notes.txt" not in status
    assert m["counts"]["files"] == 9 and m["counts"]["eligible_files"] == 6
    ok = next(r for r in m["records"] if r["path"] == "writer_01/0.png")
    assert ok["format"] == "PNG" and ok["width"] > 0 and ok["ink_tone"] in ("neutral", "blue-ish")


# ---------------------------------------------------------------- encoder contract

def _pair():
    q = normalise(render_instance(3, 1))
    r = normalise(render_instance(3, 2))
    mask = np.zeros(q.darkness.shape, bool)
    mask[30:70, 100:180] = True
    return q, r, mask


def test_classical_encoder_declares_the_contract_fields():
    spec = ClassicalGridEncoder().spec
    rec = spec.record()
    for key in ("name", "version", "config", "input_representation", "canvas", "descriptor_extraction", "distance",
                "undefined_behaviour", "masking", "contract_version"):
        assert rec[key], key
    assert rec["distance"]["direction"] == DISTANCE_DIRECTION and rec["distance"]["range"] == [0.0, 1.0]
    assert rec["masking"]["claim"] == "independent_of_excluded_canvas_pixels"
    assert "canvas level" in rec["masking"]["scope"] and rec["masking"]["tolerance"] is not None
    assert rec["canvas"]["parameters"]["canvas_width"] == 320
    json.dumps(rec)  # the declaration is serialisable as written


def test_classical_encoder_reproduces_compare_pair_exactly():
    q, r, mask = _pair()
    enc = ClassicalGridEncoder()
    for excl in (None, mask):
        row = compare_pair(GridOrientationDescriptor(), q, r, excl)
        res = enc.compare(q, r, excl)
        assert res.status == row["status"] == "defined" and res.distance == row["distance"]
    blank = normalise(np.full((120, 300, 3), 240, np.uint8))
    res = enc.compare(blank, r)
    assert res.distance is None and res.status == "insufficient_query_edge_evidence"


def test_pair_result_cannot_carry_a_number_for_an_undefined_comparison():
    with pytest.raises(ValueError):
        PairResult("insufficient_query_edge_evidence", 0.0)
    with pytest.raises(ValueError):
        PairResult("defined", None)


def test_spec_refuses_unknown_claims_untested_tolerances_and_similarity_scores():
    with pytest.raises(ValueError, match="masking claim"):
        MaskingSpec(claim="robust", support_semantics="-", scope="-")
    with pytest.raises(ValueError, match="tolerance"):
        MaskingSpec(claim="independent_of_excluded_canvas_pixels", support_semantics="-", scope="-")
    good = ClassicalGridEncoder().spec
    with pytest.raises(ValueError, match="direction"):
        replace(good, distance={**good.distance, "direction": "higher_is_more_similar"})


def test_invariance_hook_passes_for_the_classical_encoder():
    q, r, mask = _pair()
    rep = ClassicalGridEncoder().invariance_check(q, r, mask, np.random.default_rng(0), trials=4)
    assert rep.holds is True and rep.compared_trials + rep.status_changes == 4
    assert rep.max_abs_distance_change <= 1e-9


class _GlobalEncoder(Encoder):
    """A stand-in for an encoder with global context: every canvas pixel reaches the distance."""

    def __init__(self, claim, tolerance=None):
        base = ClassicalGridEncoder()
        self.normalise_params = base.normalise_params
        self.spec = replace(base.spec, name="global_mean_stub", version="0",
                            masking=MaskingSpec(claim=claim, support_semantics="mask ignored by the features",
                                                scope="none", tolerance=tolerance))

    def compare(self, query, reference, query_exclusion=None):
        return PairResult("defined", float(abs(query.darkness.mean() - reference.darkness.mean())))


def test_invariance_hook_fails_an_encoder_that_overclaims_independence():
    q, r, mask = _pair()
    rep = _GlobalEncoder("independent_of_excluded_canvas_pixels", tolerance=1e-6).invariance_check(
        q, r, mask, np.random.default_rng(0))
    assert rep.holds is False and rep.max_abs_distance_change > 1e-3


def test_invariance_hook_only_measures_when_no_independence_is_claimed():
    q, r, mask = _pair()
    rep = _GlobalEncoder("context_contaminated").invariance_check(q, r, mask, np.random.default_rng(0))
    assert rep.holds is None and rep.max_abs_distance_change > 1e-3  # a measurement, not a pass


# ---------------------------------------------------------------- runner

def test_aggregate_is_the_median_of_defined_distances_or_undefined():
    assert aggregate([0.2, None, 0.6]) == pytest.approx(0.4)
    assert aggregate([0.1, 0.5, 0.3]) == 0.3
    assert aggregate([None, None]) is None and aggregate([]) is None


def _row(writer, kind, score, status="no_stamp_detected"):
    ref_status = ["defined"] * 3 if score is not None else ["insufficient_query_edge_evidence"] * 3
    return {"claimed_writer": writer, "kind": kind, "query_writer": writer, "query_path": "x", "cleaner_status": status,
            "cleaner_reasons": [], "cleaner_changed_pixels": 0, "exclusion_share_of_canvas": 0.0, "content_flags": [],
            "score": {"excluding_query_flagged": score, "all_pixels": score},
            "reference_statuses": {"excluding_query_flagged": ref_status, "all_pixels": ref_status},
            "prepare_ms": 1.0, "compare_ms": 1.0}


def test_partition_summary_keeps_undefined_and_review_comparisons_in_the_counts():
    rows = [_row("a", "genuine", 0.1), _row("a", "genuine", None), _row("a", "impostor", 0.8),
            _row("b", "genuine", 0.2, "review"), _row("b", "impostor", None, "review"), _row("b", "impostor", 0.9)]
    s = summarise_partition(rows, {"n_resamples": 20, "seed": 0, "level": 0.95})
    block = s["by_score_rule"]["excluding_query_flagged"]
    assert s["comparisons"] == {"genuine": 3, "impostor": 3}
    assert block["genuine"]["attempted"] == 3 and block["genuine"]["usable"] == 2 and block["genuine"]["undefined"] == 1
    assert block["impostor"]["undefined"] == 1 and block["impostor"]["coverage"] == 2 / 3
    assert block["undefined_reasons"] == {"insufficient_query_edge_evidence": 2}
    assert s["query_cleaner_status"]["genuine"] == {"no_stamp_detected": 2, "review": 1}
    strata = block["by_query_cleaner_status"]["strata"]
    assert strata["review"]["genuine"]["attempted"] == 1 and strata["review"]["impostor"]["undefined"] == 1


@pytest.fixture(scope="module")
def frozen_benchmark(tmp_path_factory):
    base = tmp_path_factory.mktemp("bench")
    _write_dataset(base / "images", n_writers=12, per_writer=6)
    info = base / "info.json"
    info.write_text(json.dumps({"name": "fictional synthetic signatures", "licence": "generated"}))
    assert main(["dataset-manifest", str(base / "images"), "--dataset-info", str(info),
                 "--out", str(base / "manifest.local.json")]) == 0
    manifest = read_json(base / "manifest.local.json")
    assert main(["benchmark-freeze", "--manifest", str(base / "manifest.local.json"), "--manifest-sha256",
                 canonical_sha256(manifest), "--seed", "5", "--k", "3", "--impostors", "2", "--fractions", "0.5,0.25,0.25",
                 "--out-dir", str(base)]) == 0
    config = {
        "experiment_id": "TEST", "encoder": "classical_grid", "cleaner_method": "chroma",
        "partitions": ["development", "validation"], "primary_score": "excluding_query_flagged",
        "paths": {"dataset_root": "images", "manifest": "manifest.local.json", "splits": "splits.local.json",
                  "protocol": "protocol.local.json"},
        "sha256": {"manifest": canonical_sha256(manifest),
                   "splits": canonical_sha256(read_json(base / "splits.local.json")),
                   "protocol": canonical_sha256(read_json(base / "protocol.local.json"))},
        "bootstrap": {"n_resamples": 50, "seed": 1, "level": 0.95},
    }
    (base / "config.json").write_text(json.dumps(config))
    return base, config


def test_benchmark_runs_end_to_end_and_never_touches_test_writers(frozen_benchmark):
    base, config = frozen_benchmark
    result = run_benchmark(base / "config.json", base / "run", base_dir=base)
    splits = read_json(base / "splits.local.json")
    assert splits["counts"] == {"development": 6, "validation": 3, "test": 3}
    assert result["claim_limit"] == CLAIM_LIMIT and set(result["partitions"]) == {"development", "validation"}
    test_writers = set(splits["partitions"]["test"])
    for part in ("development", "validation"):
        rows = [json.loads(line) for line in (base / "run" / f"comparisons_{part}.local.jsonl").read_text().splitlines()]
        n = splits["counts"][part]
        assert len(rows) == n * 3 + n * 2  # 3 genuine queries and 2 impostors per claimed writer
        assert not ({r["claimed_writer"] for r in rows} | {r["query_writer"] for r in rows}) & test_writers
        assert all(r["cleaner_status"] in ("no_stamp_detected", "cleaned", "review") for r in rows)  # the cleaner ran
        p = result["partitions"][part]
        block = p["by_score_rule"]["excluding_query_flagged"]
        assert block["genuine"]["attempted"] == n * 3 and block["impostor"]["attempted"] == n * 2
        assert sum(p["query_cleaner_status"]["genuine"].values()) == n * 3
    table = (base / "run" / "validation_table.md").read_text()
    assert table.splitlines()[0] == f"**{CLAIM_LIMIT}**" and table == render_table(result)
    thr = result["development_threshold_on_validation"]["excluding_query_flagged"]
    assert thr["selected_on"] == "development" and thr["applied_to"] == "validation"
    assert thr["validation"]["threshold"] == thr["development_point"]["threshold"]
    assert result["encoder_invariance_check"]["holds"] is True
    # same frozen inputs, same numbers
    again = run_benchmark(base / "config.json", base / "run2", base_dir=base)
    for part in ("development", "validation"):
        for rule in ("excluding_query_flagged", "all_pixels"):
            a, b = (r["partitions"][part]["by_score_rule"][rule] for r in (result, again))
            assert a["eer"] == b["eer"] and a["auc"] == b["auc"] and a["bootstrap"] == b["bootstrap"]
    with pytest.raises(FileExistsError):
        run_benchmark(base / "config.json", base / "run", base_dir=base)


def test_benchmark_refuses_the_test_partition(frozen_benchmark, tmp_path):
    base, config = frozen_benchmark
    for parts in (["test"], ["development", "validation", "test"]):
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({**config, "partitions": parts}))
        with pytest.raises(FinalTestClosedError):
            run_benchmark(cfg, tmp_path / "out", base_dir=base)
    assert not (tmp_path / "out").exists()


def test_benchmark_refuses_inputs_that_differ_from_the_frozen_hashes(frozen_benchmark, tmp_path):
    base, config = frozen_benchmark
    bad = {**config, "sha256": {**config["sha256"], "splits": "0" * 64}}
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="differs from the frozen value"):
        run_benchmark(cfg, tmp_path / "out", base_dir=base)
