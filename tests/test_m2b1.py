"""M2b.1: evaluation-only original-image control versus cleaned arms on frozen benchmark comparisons."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from osv.cleaning import METHODS
from osv.cli import build_parser, main
from osv.datasets.signature_folder import canonical_sha256, read_json
from osv.datasets.synthetic_cases import render_instance
from osv.evaluation.benchmark import FinalTestClosedError, run_benchmark
from osv.evaluation.bootstrap import fixed_threshold_metrics, paired_writer_bootstrap
from osv.evaluation.control import (
    BANNER, CONTROL_LABEL, ControlConfigError, align, analyse, arm_summary, arm_view, comparison_key,
    paired_analysis, render_table, run_control, strip_timing,
)
from osv.verification.compare import compare_query

BOOT = {"n_resamples": 50, "seed": 3, "level": 0.95}


# ---------------------------------------------------------------- hand-made rows


def _row(writer, kind, path, a, b, c, status="no_stamp_detected", tone="neutral"):
    r = {"partition": "validation", "claimed_writer": writer, "kind": kind, "query_writer": writer if kind == "genuine" else "x",
         "query_path": path, "query_sha256": "h" + path, "query_ink_tone": tone, "cleaner_status": status,
         "cleaner_changed_pixels": 0, "exclusion_share_of_canvas": 0.0,
         "normalisation": {"content_box_changed_by_cleaning": False},
         "arms": {arm: {"score": s, "reference_statuses": ["defined" if s is not None else "query_insufficient"] * 3}
                  for arm, s in zip("ABC", (a, b, c))}}
    r["comparison_key"] = comparison_key(r)
    return r


def _toy_rows():
    return [
        _row("w1", "genuine", "w1/1", 0.10, 0.12, None, "review", "blue-ish"),
        _row("w1", "genuine", "w1/2", 0.20, 0.20, 0.20),
        _row("w1", "impostor", "v/1", 0.60, None, None, "review", "blue-ish"),
        _row("w1", "impostor", "v/2", 0.70, 0.65, 0.66, "cleaned", "blue-ish"),
        _row("w2", "genuine", "w2/1", None, 0.30, 0.30),
        _row("w2", "impostor", "v/3", 0.80, 0.90, 0.90),
    ]


def test_align_joins_by_identity_not_by_position():
    rows = _toy_rows()
    a, b = arm_view(rows, "A"), arm_view(list(reversed(rows)), "B")
    pairs = align(a, b)
    assert len(pairs) == 6 and all(x["comparison_key"] == y["comparison_key"] for x, y in pairs)
    assert [x["comparison_key"] for x, _ in pairs] == sorted(r["comparison_key"] for r in rows)


def test_align_refuses_same_count_but_different_cases_and_duplicates():
    rows = _toy_rows()
    other = [dict(r) for r in rows]
    other[0] = _row("w1", "genuine", "w1/OTHER", 0.1, 0.1, 0.1)
    with pytest.raises(ValueError, match="different comparisons"):
        align(arm_view(rows, "A"), arm_view(other, "B"))
    with pytest.raises(ValueError, match="occurs twice"):
        align(arm_view(rows + rows[:1], "A"), arm_view(rows, "B"))
    swapped = [dict(rows[0], query_sha256="tampered")] + rows[1:]  # stored key no longer matches its fields
    with pytest.raises(ValueError, match="does not match"):
        align(arm_view(rows, "A"), [dict(arm_view(swapped, "B")[0], comparison_key=rows[0]["comparison_key"])]
              + arm_view(swapped, "B")[1:])


def test_arm_a_has_no_cleaner_status_and_review_cases_stay_counted():
    rows = _toy_rows()
    assert all(r["cleaner_status"] is None for r in arm_view(rows, "A"))
    assert [r["cleaner_status"] for r in arm_view(rows, "C")][0] == "review"
    s = arm_summary(arm_view(rows, "C"), "C", BOOT)
    assert s["genuine"] == {**s["genuine"], "attempted": 3, "usable": 2, "undefined": 1}
    assert s["impostor"]["attempted"] == 3 and s["impostor"]["undefined"] == 1
    review = s["by_case_cleaner_status"]["strata"]["review"]
    assert review["genuine"]["attempted"] == 1 and review["genuine"]["undefined"] == 1
    sa = arm_summary(arm_view(rows, "A"), "A", BOOT)
    assert sa["arm"]["label"] == CONTROL_LABEL and "uncleaned" in sa["by_case_cleaner_status"]["note"]
    assert sa["genuine"]["undefined"] == 1 and sa["impostor"]["undefined"] == 0


def test_paired_analysis_on_a_known_example():
    rows = _toy_rows()
    out = paired_analysis(arm_view(rows, "A"), arm_view(rows, "B"), ("A", "B"), BOOT, 0.5, 0.5)
    assert out["definedness_transitions"]["genuine"] == {"both_defined": 2, "only_A": 0, "only_B": 1, "neither": 0}
    assert out["definedness_transitions"]["impostor"] == {"both_defined": 2, "only_A": 1, "only_B": 0, "neither": 0}
    sh = out["score_shift_both_defined"]
    assert sh["genuine"]["n"] == 2 and sh["genuine"]["distribution"]["median"] == pytest.approx(0.01)
    assert sh["impostor"]["distribution"]["median"] == pytest.approx(0.025)  # shifts -0.05 and +0.10
    assert sh["genuine"]["unchanged_share"] == 0.5
    cov = out["coverage"]["metrics"]
    assert cov["coverage_genuine"]["a"] == pytest.approx(2 / 3) and cov["coverage_genuine"]["b"] == 1.0
    assert cov["coverage_impostor"]["difference_b_minus_a"] == pytest.approx(-1 / 3)
    cohort = out["common_defined_cohort"]
    assert cohort["size"] == {"genuine": 2, "impostor": 2} and cohort["attempted"] == {"genuine": 3, "impostor": 3}
    fa = out["full_attempt_outcomes"]["paired_bootstrap_b_minus_a"]["metrics"]
    # A at 0.5: genuine accepted 2/3; B at 0.5: genuine accepted 3/3
    assert fa["genuine_not_accepted_attempted"]["difference_b_minus_a"] == pytest.approx(-1 / 3)
    assert set(out["by_query_ink_tone"]["strata"]) == {"blue-ish", "neutral"}
    again = paired_analysis(arm_view(rows, "A"), arm_view(rows, "B"), ("A", "B"), BOOT, 0.5, 0.5)
    assert json.dumps(out, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_refusing_hard_cases_cannot_look_like_an_improvement_in_full_attempt_outcomes():
    # C refuses the one genuine that A rejects: C looks perfect on its defined subset but not on attempts.
    rows = [_row("w", "genuine", f"g{k}", 0.1, 0.1, 0.1) for k in range(3)] + [_row("w", "genuine", "hard", 0.9, 0.9, None)]
    rows += [_row("w", "impostor", f"i{k}", 0.8, 0.8, 0.8) for k in range(4)]
    out = paired_analysis(arm_view(rows, "A"), arm_view(rows, "C"), ("A", "C"), BOOT, 0.5, 0.5)
    fa = out["full_attempt_outcomes"]["paired_bootstrap_b_minus_a"]["metrics"]
    assert fa["genuine_not_accepted_attempted"]["a"] == 0.25 and fa["genuine_not_accepted_attempted"]["b"] == 0.25
    assert fa["genuine_not_accepted_attempted"]["difference_b_minus_a"] == 0.0


def test_paired_bootstrap_accepts_per_system_metrics_with_the_same_names():
    scores = {"w": {"genuine": [0.2, 0.4], "impostor": [0.5, 0.7]}}
    out = paired_writer_bootstrap(scores, scores, fixed_threshold_metrics(0.3), n_resamples=20, seed=0,
                                  metrics_b=fixed_threshold_metrics(0.6))
    m = out["metrics"]
    assert m["frr_usable"]["a"] == 0.5 and m["frr_usable"]["b"] == 0.0 and m["far_usable"]["b"] == 0.5
    with pytest.raises(ValueError, match="same metrics"):
        paired_writer_bootstrap(scores, scores, fixed_threshold_metrics(0.3), metrics_b={"far_usable": lambda *a: 0})


def test_analyse_is_deterministic_and_refuses_arms_with_different_cases():
    rows = _toy_rows()
    a, b = analyse({"validation": rows}, BOOT), analyse({"validation": rows}, BOOT)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    bad = [dict(r) for r in rows]
    bad.append(dict(bad[0]))
    with pytest.raises(ValueError, match="occurs twice"):
        analyse({"validation": bad}, BOOT)


# ---------------------------------------------------------------- end to end on fictional images


@pytest.fixture(scope="module")
def control_setup(tmp_path_factory):
    base = tmp_path_factory.mktemp("control")
    for w in range(12):
        d = base / "images" / f"writer_{w:02d}"
        d.mkdir(parents=True)
        for k in range(6):
            Image.fromarray(render_instance(100 + w, k)).save(d / f"{k}.png")
    (base / "info.json").write_text(json.dumps({"name": "fictional synthetic signatures", "licence": "generated"}))
    assert main(["dataset-manifest", str(base / "images"), "--dataset-info", str(base / "info.json"),
                 "--out", str(base / "manifest.local.json")]) == 0
    manifest = read_json(base / "manifest.local.json")
    assert main(["benchmark-freeze", "--manifest", str(base / "manifest.local.json"), "--manifest-sha256",
                 canonical_sha256(manifest), "--seed", "5", "--k", "3", "--impostors", "2", "--out-dir", str(base)]) == 0
    frozen = {
        "experiment_id": "TESTF", "encoder": "classical_grid", "cleaner_method": "chroma",
        "partitions": ["development", "validation"], "primary_score": "excluding_query_flagged",
        "dataset": {"name": "fictional"}, "frozen": {"seed": 5},
        "paths": {"dataset_root": "images", "manifest": "manifest.local.json", "splits": "splits.local.json",
                  "protocol": "protocol.local.json"},
        "sha256": {"manifest": canonical_sha256(manifest),
                   "splits": canonical_sha256(read_json(base / "splits.local.json")),
                   "protocol": canonical_sha256(read_json(base / "protocol.local.json"))},
        "bootstrap": {"n_resamples": 50, "seed": 1, "level": 0.95},
    }
    (base / "frozen.json").write_text(json.dumps(frozen))
    run_benchmark(base / "frozen.json", base / "archive", base_dir=base)
    control = {k: frozen[k] for k in ("encoder", "cleaner_method", "partitions", "dataset", "frozen", "paths", "sha256",
                                      "bootstrap")}
    control.update(experiment_id="TESTC", kind="evaluation_only_original_image_control", evaluation_only=True,
                   reuses_frozen_inputs_of={"experiment_id": "TESTF", "config": "frozen.json"},
                   archived_reference_run="archive")
    (base / "control.json").write_text(json.dumps(control))
    result = run_control(base / "control.json", base / "run", base_dir=base)
    return base, control, result


def _rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_control_scores_exactly_the_frozen_comparisons_in_every_arm(control_setup):
    base, _cfg, result = control_setup
    protocol = read_json(base / "protocol.local.json")
    for part in ("development", "validation"):
        rows = _rows(base / "run" / f"control_comparisons_{part}.local.jsonl")
        expected = set()
        for u in protocol["partitions"][part]:
            refs = [r["sha256"] for r in u["references"]]
            for q in u["genuine_queries"]:
                expected.add((u["claimed_writer"], "genuine", u["claimed_writer"], q["path"], q["sha256"]))
            for q in u["impostor_queries"]:
                expected.add((u["claimed_writer"], "impostor", q["writer"], q["path"], q["sha256"]))
            assert all(r["reference_sha256"] == refs for r in rows if r["claimed_writer"] == u["claimed_writer"])
        got = [(r["claimed_writer"], r["kind"], r["query_writer"], r["query_path"], r["query_sha256"]) for r in rows]
        assert len(got) == len(set(got)) and set(got) == expected
        keys = [{x["comparison_key"] for x in arm_view(rows, a)} for a in "ABC"]
        assert keys[0] == keys[1] == keys[2]
        for a in "ABC":  # undefined accounting: every attempt is either defined or undefined, nothing dropped
            blk = result["partitions"][part]["arms"][a]
            for kind in ("genuine", "impostor"):
                n = sum(r["kind"] == kind for r in rows)
                assert blk[kind]["attempted"] == n == blk[kind]["usable"] + blk[kind]["undefined"]
                assert blk[kind]["undefined"] == sum(r["kind"] == kind and r["arms"][a]["score"] is None for r in rows)


def test_cleaned_arms_reproduce_the_archived_benchmark_rows_exactly(control_setup):
    _base, _cfg, result = control_setup
    for part in ("development", "validation"):
        chk = result["archive_check"][part]
        assert chk["identical"] is True and chk["compared"] > 0


def test_control_outputs_cannot_be_mistaken_for_product_results(control_setup):
    base, _cfg, result = control_setup
    assert result["evaluation_only"] is True and result["product_result"] is False
    assert result["schema"].startswith("osv-evaluation-control")
    table = (base / "run" / "control_table.md").read_text()
    assert table.splitlines()[0] == f"**{BANNER}**" and "NOT A PRODUCT RESULT" in table
    assert not (base / "run" / "validation_table.md").exists()
    assert result["resolved"]["arms"]["A"]["label"] == CONTROL_LABEL
    assert "no cleaner status" in result["resolved"]["arms"]["A"]["cleaner"]
    assert set(result["implementation"]["source_sha256"]) >= {"evaluation/control.py", "verification/compare.py"}


def test_repeat_run_reproduces_scores_and_tables(control_setup):
    base, _cfg, result = control_setup
    again = run_control(base / "control.json", base / "run2", base_dir=base)
    assert (base / "run" / "control_table.md").read_text() == (base / "run2" / "control_table.md").read_text()
    for k in ("partitions", "paired", "development_threshold_on_validation", "archive_check"):
        assert strip_timing(result[k]) == strip_timing(again[k])
    for part in ("development", "validation"):
        a = strip_timing(_rows(base / "run" / f"control_comparisons_{part}.local.jsonl"))
        b = strip_timing(_rows(base / "run2" / f"control_comparisons_{part}.local.jsonl"))
        assert a == b
    assert render_table(again) == (base / "run2" / "control_table.md").read_text()


def test_control_refuses_the_final_test_changed_inputs_and_non_control_configs(control_setup, tmp_path):
    base, cfg, _ = control_setup

    def attempt(c):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(c))
        return run_control(p, tmp_path / "out", base_dir=base)

    for parts in (["test"], ["development", "validation", "test"]):
        with pytest.raises(FinalTestClosedError):
            attempt({**cfg, "partitions": parts})
    with pytest.raises(ControlConfigError, match="differs from TESTF"):
        attempt({**cfg, "sha256": {**cfg["sha256"], "splits": "0" * 64}})
    with pytest.raises(ControlConfigError, match="evaluation-only"):
        attempt({k: v for k, v in cfg.items() if k != "evaluation_only"})
    with pytest.raises(ControlConfigError, match="production cleaner"):
        attempt({**cfg, "cleaner_method": "none"})
    assert not (tmp_path / "out").exists()
    # a frozen input file whose content changed is refused by hash
    splits = base / "splits.local.json"
    original = splits.read_text()
    try:
        data = json.loads(original)
        data["seed"] = data["seed"] + 1
        splits.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="differs from the frozen value"):
            attempt(cfg)
    finally:
        splits.write_text(original)


# ---------------------------------------------------------------- product commands stay cleaned


def test_product_cleaner_registry_and_commands_offer_no_raw_path():
    assert set(METHODS) == {"chroma", "optical-density"}
    parser = build_parser()
    for argv in (["clean", "x.png", "--out", "o", "--method", "none"],
                 ["compare-case", "m.json", "--out", "o", "--no-clean"],
                 ["compare-case", "m.json", "--out", "o", "--method", "none"],
                 ["benchmark", "--config", "c.json", "--out", "o", "--original"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    with pytest.raises(TypeError, match="prepare_query"):
        compare_query(object(), None)


def test_product_benchmark_refuses_an_uncleaned_method(control_setup, tmp_path):
    base, _cfg, _ = control_setup
    frozen = json.loads((base / "frozen.json").read_text())
    for method in ("none", "original", "raw"):
        p = tmp_path / "f.json"
        p.write_text(json.dumps({**frozen, "cleaner_method": method}))
        with pytest.raises(ValueError, match="unknown cleaner method"):
            run_benchmark(p, tmp_path / "out", base_dir=base)


def test_control_runs_only_through_its_own_command(control_setup, tmp_path, capsys):
    base, cfg, _ = control_setup
    p = tmp_path / "c.json"
    p.write_text(json.dumps({**cfg, "primary_score": "all_pixels"}))
    with pytest.raises(ValueError, match="evaluation-only control"):
        run_benchmark(p, tmp_path / "b", base_dir=base)
    assert main(["benchmark", "--config", str(p), "--out", str(tmp_path / "b")]) == 2
    assert main(["evaluation-control", "--config", str(base / "frozen.json"), "--out", str(tmp_path / "e")]) == 2
    assert "evaluation-only" in capsys.readouterr().err
    assert not (tmp_path / "b").exists() and not (tmp_path / "e").exists()
