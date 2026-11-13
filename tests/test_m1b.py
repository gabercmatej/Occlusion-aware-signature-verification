"""M1b: second mixture model, experimental optical-density cleaner, comparison and provenance.

Synthetic only (E0). Ground truth is read after cleaning, for measurement. Behaviour tests use
development seeds only; held-out seeds are run once by ``compare-cleaners --split heldout``.
Floors are regression floors set from the frozen development run (2026-11-12), not quality claims.
"""

import hashlib
import inspect
import json

import numpy as np
import pytest
from PIL import Image

import osv.datasets.synthetic_fixtures as sf
from osv.cleaning import DEFAULT_METHOD, METHODS, clean_image
from osv.cli import main
from osv.datasets.synthetic_fixtures import EXTRA_KINDS, KINDS, make_fixture
from osv.datasets.synthetic_matrix import (
    DEV_SEEDS, HELDOUT_SEEDS, MIXTURES, SCENARIOS, make_case, matrix_cases,
)
from osv.evaluation import cleaning_metrics as em
from osv.evaluation.compare_cleaners import markdown, run_case, summarize
from osv.segmentation.od_estimate import ESTIMATOR_NAME as OD_NAME
from osv.segmentation.od_estimate import estimate_stamp_od

DEV = DEV_SEEDS[:5]
STAMPED = [s for s, spec in SCENARIOS.items() if spec["stamp"] is not None]
# Frozen dev run, OD 0.1.0, cleaned cases: min stamp-only removal red 0.813, blue 0.827 (both
# mixtures); faint multiplicative 0.451.
OD_FLOOR = {"red_stamp_black_pen": 0.78, "blue_stamp_black_pen": 0.78}
OD_FLOOR_FAINT_MULT = 0.40


def geoms(scenario):
    return ("round", "rect") if SCENARIOS[scenario]["stamp"] is not None else ("none",)


# ---------------------------------------------------------------- baseline preserved

def test_original_generator_outputs_unchanged():
    """The M1a generator is byte-identical to before M1b (hash recorded before any M1b change)."""
    h = hashlib.sha256()
    for k in KINDS + EXTRA_KINDS:
        for s in range(10):
            f = make_fixture(k, s)
            h.update(f.image.tobytes()); h.update(f.pen_mask.tobytes()); h.update(f.stamp_mask.tobytes())
    assert h.hexdigest() == "a3f133534b65b8170bdd063c5f784c1d20f55eb2eadf91f7e321d74e1bdc4c88"


def test_metrics_moved_with_compatible_import():
    assert sf.cleaning_metrics is em.cleaning_metrics
    assert sf.stamp_support_with_margin is em.stamp_support_with_margin


def test_default_method_is_baseline():
    assert DEFAULT_METHOD == "chroma"
    assert set(METHODS) == {"chroma", "optical-density"}
    with pytest.raises(ValueError):
        clean_image(make_fixture("no_stamp", 0).image, method="auto")


def test_cleaners_accept_no_ground_truth():
    assert list(inspect.signature(estimate_stamp_od).parameters) == ["image_rgb", "stamp_region"]
    assert list(inspect.signature(clean_image).parameters) == ["image_rgb", "method", "stamp_region"]


# ---------------------------------------------------------------- matrix generator

def test_seed_groups_fixed_and_disjoint():
    assert DEV_SEEDS == tuple(range(10)) and HELDOUT_SEEDS == tuple(range(100, 110))
    assert not set(DEV_SEEDS) & set(HELDOUT_SEEDS)
    assert len(matrix_cases(DEV_SEEDS)) == 10 * (len(STAMPED) * 2 * 2 + (len(SCENARIOS) - len(STAMPED)) * 2)


@pytest.mark.parametrize("mixture", MIXTURES)
def test_matrix_deterministic_and_parameters_recorded(mixture):
    a = make_case("red_stamp_black_pen", mixture, "rect", 3)
    b = make_case("red_stamp_black_pen", mixture, "rect", 3)
    assert np.array_equal(a.image, b.image) and a.meta == b.meta
    for key in ("matrix_version", "scenario", "mixture", "geometry", "seed", "pen_rgb", "stamp_rgb", "model"):
        assert key in a.meta
    if mixture == "alpha_over_scan":
        for key in ("layer_order", "stamp_opacity", "tone_gamma", "channel_misregistration_px"):
            assert key in a.meta


def test_mixtures_are_different_models_on_same_geometry():
    m = make_case("red_stamp_black_pen", "multiplicative", "round", 0)
    a = make_case("red_stamp_black_pen", "alpha_over_scan", "round", 0)
    assert np.array_equal(m.pen_mask, a.pen_mask) and np.array_equal(m.stamp_mask, a.stamp_mask)
    assert not np.array_equal(m.image, a.image)


def _overlap_colour(scenario, order, n=40):
    """Mean overlap colour minus pen colour, for the first alpha-over case with the given layer order."""
    for s in range(n):
        fx = make_case(scenario, "alpha_over_scan", "round", s)
        core = fx.overlap_mask & (fx.pen_alpha > 0.95) & (fx.stamp_alpha > 0.9)
        if fx.meta["layer_order"] == order and core.sum() > 20:
            return fx.image[core].mean(0), np.array(fx.meta["pen_rgb"]), np.array(fx.meta["stamp_rgb"])
    raise AssertionError("no case found")


def test_alpha_over_layer_order_hides_lower_ink():
    """Under 'over' compositing the top layer dominates overlap pixels; under multiplication both
    inks always darken. Pen over a red stamp keeps overlap close to the black pen colour."""
    under, pen, stamp = _overlap_colour("red_stamp_black_pen", "pen_over_stamp")
    over, _, _ = _overlap_colour("red_stamp_black_pen", "stamp_over_pen")
    # Pen on top: overlap red channel stays far closer to the black pen than to the red stamp.
    assert abs(under[0] - pen[0]) < 0.3 * abs(stamp[0] - pen[0])
    # Stamp on top: the red stamp lifts the red channel well above the pen-on-top case.
    assert over[0] > under[0] + 60


def test_marks_follow_their_definition():
    for s in DEV:
        fx = make_case("blue_same_unrelated_mark", "multiplicative", "round", s)
        ink = (fx.pen_alpha > 0) | (fx.stamp_alpha > 0)
        assert fx.mark_mask.any() and not (fx.mark_mask & ink).any()
        for kind in ("blue_same_touching_mark", "blue_same_large_touching_mark"):
            fx = make_case(kind, "multiplicative", "round", s)
            assert (fx.mark_mask & fx.pen_mask).any()
            assert not (fx.mark_mask & (fx.stamp_alpha > 0)).any()


def test_invalid_case_arguments():
    with pytest.raises(ValueError):
        make_case("no_such", "multiplicative", "round", 0)
    with pytest.raises(ValueError):
        make_case("black_pen_no_stamp", "multiplicative", "round", 0)
    with pytest.raises(ValueError):
        make_case("red_stamp_black_pen", "screen", "round", 0)


# ---------------------------------------------------------------- invariants, both methods

@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("mixture", MIXTURES)
def test_invariants_on_matrix(method, scenario, mixture):
    for geo in geoms(scenario):
        fx = make_case(scenario, mixture, geo, 0)
        before = fx.image.copy()
        est, res = clean_image(fx.image, method=method)
        assert np.array_equal(fx.image, before)
        changed = res.change_map > 0
        assert not (changed & ~res.intervention_mask).any()
        assert not (res.intervention_mask & ~est.stamp_mask).any()
        assert not (res.intervention_mask & est.ambiguous_mask).any()
        assert np.array_equal((res.cleaned != fx.image).any(axis=2), changed)
        if est.action != "delete":
            assert not changed.any(), "refusal must happen before deletion"
        est2, res2 = clean_image(fx.image.copy(), method=method)
        assert np.array_equal(res.cleaned, res2.cleaned) and res.reasons == res2.reasons


@pytest.mark.parametrize("kind", KINDS)
def test_od_invariants_on_original_fixtures(kind):
    for s in range(3):
        fx = make_fixture(kind, s)
        est, res = clean_image(fx.image, method="optical-density")
        changed = res.change_map > 0
        assert not (changed & ~res.intervention_mask).any()
        assert not (res.intervention_mask & est.ambiguous_mask).any()
        m = em.cleaning_metrics(fx, res.intervention_mask, res.change_map, est.ambiguous_mask)
        assert m["pen_changed_away_from_stamp"] == 0


# ---------------------------------------------------------------- OD behaviour (dev seeds)

@pytest.mark.parametrize("mixture", MIXTURES)
@pytest.mark.parametrize("scenario", ["red_stamp_black_pen", "blue_stamp_black_pen"])
def test_od_different_colour_removes_without_pen_damage_outside_overlap(scenario, mixture):
    for geo in ("round", "rect"):
        for s in DEV:
            fx = make_case(scenario, mixture, geo, s)
            est, res = clean_image(fx.image, method="optical-density")
            m = em.case_metrics(fx, res.intervention_mask, res.change_map, est.ambiguous_mask)
            assert m["pen_changed_outside_overlap_pixels"] == 0
            assert m["pen_changed_away_from_stamp"] == 0
            if res.status == "cleaned":
                assert m["stamp_only_removed"] >= OD_FLOOR[scenario]


def test_od_faint_multiplicative_floor():
    for geo in ("round", "rect"):
        for s in DEV:
            fx = make_case("faint_violet_jpeg", "multiplicative", geo, s)
            est, res = clean_image(fx.image, method="optical-density")
            m = em.case_metrics(fx, res.intervention_mask, res.change_map, est.ambiguous_mask)
            assert res.status == "cleaned" and m["stamp_only_removed"] >= OD_FLOOR_FAINT_MULT
            assert m["pen_changed_pixels"] == 0


@pytest.mark.parametrize("mixture", MIXTURES)
@pytest.mark.parametrize("scenario", [
    "blue_stamp_blue_pen", "black_stamp_black_pen", "cyan_stamp_navy_pen", "navy_stamp_cyan_pen",
    "blue_same_unrelated_mark", "blue_same_touching_mark", "black_pen_no_stamp", "blue_pen_no_stamp",
])
def test_od_refuses_same_similar_and_unstamped_without_change(scenario, mixture):
    """Same- or similar-colour inks and unstamped handwriting: the OD cleaner changes nothing
    (review or no_stamp_detected) on these dev seeds. This does not show every such real case
    is caught; see the large touching-mark results."""
    for geo in geoms(scenario):
        for s in DEV:
            fx = make_case(scenario, mixture, geo, s)
            est, res = clean_image(fx.image, method="optical-density")
            assert res.status in {"review", "no_stamp_detected"}
            assert int((res.change_map > 0).sum()) == 0


def test_od_two_chromatic_inks_refused_as_roles_ambiguous():
    fx = make_case("navy_stamp_cyan_pen", "multiplicative", "round", 0)
    est, res = clean_image(fx.image, method="optical-density")
    assert "ink_roles_ambiguous" in res.reasons
    assert est.stats["direction_angle_deg"] > 8.0, "directions differ; colour cannot assign roles"


@pytest.mark.parametrize("seed", range(10))
def test_od_no_stamp_original_fixture_exact_noop(seed):
    fx = make_fixture("no_stamp", seed)
    est, res = clean_image(fx.image, method="optical-density")
    assert res.status == "no_stamp_detected" and np.array_equal(res.cleaned, fx.image)


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("seed", range(10))
def test_reported_unrelated_mark_case_both_methods(method, seed):
    fx = make_fixture("blue_stamp_blue_pen", seed)
    image = fx.image.copy()
    image[8:48, 8:88] = 20
    est, res = clean_image(image, method=method)
    assert est.action == "review" and int((res.change_map > 0).sum()) == 0


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN LIMITATION (both methods): a neutral mark touching same-colour handwriting looks like "
    "a neutral pen crossing a coloured stamp. Optical density deletes 5,693 pen pixels and reports "
    "cleaned. (Chroma's deletion was reported review; since remover 0.2.0 it is reverted.) Recorded, not fixed."))
@pytest.mark.parametrize("method", ["optical-density"])
def test_known_limitation_touching_mark_optical_density(method):
    fx = make_fixture("blue_stamp_blue_pen", 1)
    image = fx.image.copy()
    image[190:232, 380:472] = 20
    est, res = clean_image(image, method=method)
    assert int(((res.change_map > 0) & fx.pen_mask).sum()) == 0


# ---------------------------------------------------------------- CLI and provenance

def _write(tmp_path, image, name="q.png"):
    p = tmp_path / name
    Image.fromarray(image).save(p)
    return p


def test_cli_method_recorded_in_provenance(tmp_path):
    src = _write(tmp_path, make_fixture("red_stamp_black_pen", 0).image)
    assert main(["clean", str(src), "--out", str(tmp_path / "a")]) == 0
    assert main(["clean", str(src), "--out", str(tmp_path / "b"), "--method", "optical-density"]) == 0
    ra = json.loads((tmp_path / "a" / "report.json").read_text())
    rb = json.loads((tmp_path / "b" / "report.json").read_text())
    assert ra["cleaner"]["method"] == "chroma"
    assert ra["cleaner"]["estimator"]["name"] == "chroma_cluster_baseline"
    assert rb["cleaner"]["method"] == "optical-density"
    assert rb["cleaner"]["estimator"]["name"] == OD_NAME
    assert "pen_chromaticity_max" in rb["cleaner"]["estimator"]["parameters"]
    for k in ("direction_angle_deg", "fisher_ratio", "condition_number"):
        assert k in rb["estimate_stats"]
    assert "optical-density" in rb["definitions"]["separability"]
    assert (tmp_path / "b" / "original.png").read_bytes() == src.read_bytes()
    assert rb["schema_version"] == "1.2"


def test_cli_od_byte_identical_reruns(tmp_path):
    src = _write(tmp_path, make_case("blue_stamp_black_pen", "alpha_over_scan", "rect", 1).image)
    for d in ("a", "b"):
        assert main(["clean", str(src), "--out", str(tmp_path / d), "--method", "optical-density"]) == 0
    for p in (tmp_path / "a").iterdir():
        assert p.read_bytes() == (tmp_path / "b" / p.name).read_bytes(), p.name


def test_cli_rejects_unknown_method(tmp_path):
    src = _write(tmp_path, make_fixture("no_stamp", 0).image)
    with pytest.raises(SystemExit):
        main(["clean", str(src), "--out", str(tmp_path / "o"), "--method", "auto"])


def test_od_region_without_separable_stamp_goes_to_review(tmp_path):
    fx = make_fixture("black_stamp_black_pen", 0)
    ys, xs = np.nonzero(fx.stamp_mask)
    region = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    est, res = clean_image(fx.image, method="optical-density", stamp_region=region)
    assert res.status == "review" and "stamp_indicated_not_separated" in res.reasons
    assert int((res.change_map > 0).sum()) == 0


# ---------------------------------------------------------------- comparison command

def test_comparison_rows_and_summary():
    rows = [run_case("red_stamp_black_pen", "multiplicative", "round", 0, m) for m in METHODS]
    rows += [run_case("black_pen_no_stamp", "alpha_over_scan", "none", 0, m) for m in METHODS]
    for r in rows:
        for k in ("status", "reasons", "stamp_only_removed", "pen_changed_outside_overlap_pixels",
                  "stamp_only_unflagged", "runtime_ms", "mixture", "group", "method"):
            assert k in r
    s = summarize(rows)
    assert set(s["by_method"]) == set(METHODS)
    md = markdown(s, "test")
    assert "status C/R/N" in md and "alpha_over_scan" in md


def test_compare_cli_refuses_non_empty_out(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "x").write_text("x")
    assert main(["compare-cleaners", "--split", "dev", "--out", str(out)]) == 2
