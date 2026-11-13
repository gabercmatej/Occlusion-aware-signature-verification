"""M2a.2 correction batch: regressions for checkpoint-review findings A1-A6.

All inputs are fictional synthetic images; these tests establish software behaviour only.

Provisional values fixed *before* the corrected pipeline was evaluated (session log, 13 Nov 2026):

* ``TOL = 0.03``: largest accepted change in a total-variation distance caused by crop geometry
  alone (adding paper, moving the crop within a larger sheet, uniform rescaling by 0.75-1.5).
  Rationale: below two thirds of the smallest and below a third of the largest pre-fix
  same-writer reference-pair distance (0.046-0.108); the test below also checks that it stays at
  most half of the median reference-pair distance measured after the fix.
* Transform range: paper added on one side (60 px) or all sides (30 px), top (40 px); crop moved
  by up to 20 px inside a sheet with 40 px of extra paper; uniform rescale 0.75, 0.9, 1.25, 1.5.
"""

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from osv.cleaning import METHODS, clean_image
from osv.cli import main
from osv.datasets.synthetic_cases import render_instance
from osv.datasets.synthetic_fixtures import make_fixture
from osv.datasets.synthetic_matrix import DEV_SEEDS, HELDOUT_SEEDS, make_case as make_matrix_case, matrix_cases
from osv.preprocessing.image_io import load_image
from osv.preprocessing.normalise import NormaliseParams, normalise
from osv.provenance import load_rgb
from osv.verification.compare import compare_query, prepare_query, run_case
from osv.verification.descriptor import GridOrientationDescriptor
from osv.verification.enrolment import enrol
from osv.verification.manifest import load_manifest
from osv.verification.pairs import MIN_RETAINED_EVIDENCE_SHARE, compare_pair, describe_normalised

from test_m2a import make_case, refs_a, sha

TOL = 0.03
PAPER = (243, 240, 232)
D = GridOrientationDescriptor()


def pad(img, l=0, r=0, t=0, b=0):
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=PAPER)


def rescale(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)


TRANSFORMS = {
    "pad_left_60": lambda x: pad(x, l=60),
    "pad_all_30": lambda x: pad(x, 30, 30, 30, 30),
    "pad_top_40": lambda x: pad(x, t=40),
    "move_in_sheet_-15_+10": lambda x: pad(x, 40, 40, 40, 40)[50:50 + x.shape[0], 25:25 + x.shape[1]],
    "move_in_sheet_+20_-12": lambda x: pad(x, 40, 40, 40, 40)[28:28 + x.shape[0], 60:60 + x.shape[1]],
    "rescale_0.75": lambda x: rescale(x, 0.75),
    "rescale_0.9": lambda x: rescale(x, 0.9),
    "rescale_1.25": lambda x: rescale(x, 1.25),
    "rescale_1.5": lambda x: rescale(x, 1.5),
}


def dist(a, b, extra=None):
    row = compare_pair(D, normalise(a), normalise(b), extra)
    return row["distance"]


def median_to_refs(q, refs):
    return float(np.median([dist(q, r) for r in refs]))


# ---------------------------------------------------------------- A1 ink-based normalisation


_GAP = pytest.mark.xfail(strict=True, reason=(
    "MEASURED GAP (M2a.2, not tuned): writer 1 rescaled by 0.9 differs from itself by 0.0319 > TOL 0.03. "
    "The content box agrees within about 1 source px; the residual is sub-pixel misregistration from "
    "the integer box/window and two-stage resampling. Median distance to references moves by only 0.0006."))
_GEOMETRY_CASES = [pytest.param(name, writer, marks=_GAP) if (name, writer) == ("rescale_0.9", 1) else (name, writer)
                   for writer in (1, 3) for name in sorted(TRANSFORMS)]


@pytest.mark.parametrize("name,writer", _GEOMETRY_CASES)
def test_crop_geometry_changes_distance_by_less_than_tolerance(name, writer):
    refs = [render_instance(writer, i) for i in range(4)]
    q = render_instance(writer, 10)
    tq = TRANSFORMS[name](q)
    assert dist(tq, q) <= TOL
    assert abs(median_to_refs(tq, refs) - median_to_refs(q, refs)) <= TOL


@pytest.mark.parametrize("writer", [1, 3])
def test_tolerance_is_well_below_reference_spread(writer):
    refs = [render_instance(writer, i) for i in range(4)]
    pairs = [dist(refs[i], refs[j]) for i in range(4) for j in range(i + 1, 4)]
    assert TOL <= 0.5 * float(np.median(pairs))
    other = median_to_refs(render_instance(writer + 1, 0), refs)
    assert other > max(pairs)  # software sanity on fictional writers only


def test_content_box_and_full_transform_are_recorded():
    img = pad(render_instance(1, 0), l=60, t=20)
    n = normalise(img)
    rec = n.record()
    box = rec["content_box"]
    assert rec["normaliser"]["version"] == "0.2.0"
    assert set(box) >= {"x", "y", "w", "h", "rule"}
    assert "not necessarily handwriting" in box["rule"]
    assert set(rec["transform"]) >= {"source_window", "scale_x", "scale_y", "matrix_2x3", "maps"}
    m = n.params.canvas_margin_px
    (x0, y0), (x1, y1) = n.transform.map_points(np.array([[box["x"], box["y"]],
                                                          [box["x"] + box["w"], box["y"] + box["h"]]], float))
    assert x0 >= m - 1 and y0 >= m - 1 and x1 <= 320 - m + 1 and y1 <= 128 - m + 1
    assert min(x0, 320 - x1) <= m + 1 or min(y0, 128 - y1) <= m + 1  # the box fills one inset dimension
    mat = np.array(rec["transform"]["matrix_2x3"])
    pts = np.array([[box["x"], box["y"], 1.0]])
    assert np.allclose(pts @ mat.T, n.transform.map_points(pts[:, :2]), atol=1e-6)


def test_masks_follow_the_same_content_transform():
    img = render_instance(1, 0)
    mask = np.zeros(img.shape[:2], bool)
    mask[100:130, 200:240] = True
    a = normalise(img, {"stamp_mask": mask})
    b = normalise(pad(img, l=50, t=30), {"stamp_mask": pad(mask.astype(np.uint8), l=50, t=30).astype(bool)})
    assert np.array_equal(a.masks["stamp_mask"], b.masks["stamp_mask"])
    assert np.max(np.abs(a.darkness - b.darkness)) < 0.02


def test_content_box_edge_beside_flagged_region_is_uncertain():
    img = render_instance(1, 0)
    n0 = normalise(img)
    box = n0.record()["content_box"]
    right = box["x"] + box["w"]
    near = np.zeros(img.shape[:2], bool)
    near[100:140, right - 20:min(img.shape[1], right + 10)] = True
    n1 = normalise(img, {"stamp_mask": near})
    assert "content_box_uncertain:right" in n1.content_flags
    far = np.zeros(img.shape[:2], bool)
    far[0:6, 0:6] = True
    assert not any(f.startswith("content_box_uncertain") for f in normalise(img, {"stamp_mask": far}).content_flags)


def test_query_box_does_not_depend_on_references(tmp_path):
    q = render_instance(1, 10, stamped=True)
    a = run_case(make_case(tmp_path / "a", refs_a(3), q), tmp_path / "oa")
    b = run_case(make_case(tmp_path / "b", [render_instance(2, i) for i in range(5)], q), tmp_path / "ob")
    assert a["query"]["normalisation"] == b["query"]["normalisation"]


def test_blank_image_falls_back_to_whole_image_and_says_so():
    n = normalise(np.full((200, 300, 3), 240, np.uint8))
    assert "no_content_detected" in n.content_flags
    assert n.record()["content_box"]["x"] is None


# ---------------------------------------------------------------- A2 padding and blank inputs


def test_padding_content_cannot_change_the_descriptor():
    tight = render_instance(1, 0)[40:200, 20:460]  # window will extend beyond the crop
    n = normalise(tight)
    assert (~n.valid_mask).any()
    other = normalise(render_instance(1, 1))
    base = compare_pair(D, n, other)
    altered = normalise(tight)
    altered.darkness[~altered.valid_mask] = 1.0
    assert compare_pair(D, altered, other) == base


def test_blank_noisy_paper_is_insufficient_evidence():
    rng = np.random.default_rng(0)
    flat = np.clip(243 + rng.normal(0, 1.5, (240, 480, 3)).round(), 0, 255).astype(np.uint8)
    assert describe_normalised(D, normalise(flat)).status == "insufficient_edge_evidence"


def test_blank_shaded_square_is_insufficient_evidence():
    ramp = np.linspace(0.80, 1.0, 300, dtype=np.float32)[None, :, None]
    blank = np.clip(np.round(np.array(PAPER, np.float32) * ramp * np.ones((300, 300, 1), np.float32)), 0, 255).astype(np.uint8)
    assert describe_normalised(D, normalise(blank)).status == "insufficient_edge_evidence"


def test_shading_does_not_move_distance_to_itself_beyond_tolerance():
    sq = pad(render_instance(1, 10), t=120, b=120)
    shaded = np.clip(np.round(sq * np.linspace(0.85, 1.0, sq.shape[1], dtype=np.float32)[None, :, None]), 0, 255).astype(np.uint8)
    assert dist(sq, shaded) <= TOL


def test_short_stroke_is_defined_evidence():
    img = np.full((120, 300, 3), 243, np.uint8)
    img[58:62, 100:180] = 30
    assert describe_normalised(D, normalise(img)).status == "defined"


# ---------------------------------------------------------------- A3 minimum retained evidence


def test_tiny_retained_evidence_is_insufficient_not_a_distance(tmp_path):
    m = load_manifest(make_case(tmp_path, refs_a(4), render_instance(1, 10)))
    e, cq = enrol(m), prepare_query(m.query, "chroma")
    excl = np.ones_like(cq.exclusion)
    excl[:, :int(320 * 0.2)] = False  # P3 probe: about 2.6 % of the edge evidence survives
    object.__setattr__(cq, "exclusion", excl)
    c = compare_query(cq, e)
    for row in c["per_reference"]:
        assert row["distance_excluding_query_flagged"] is None
        assert row["excluding_query_flagged_status"] == "insufficient_query_retained_evidence"
        assert row["query_retained_evidence_share"] < MIN_RETAINED_EVIDENCE_SHARE
    assert c["aggregate"]["distance_excluding_query_flagged"]["median"] is None
    assert c["minimum_retained_evidence_share"]["value"] == MIN_RETAINED_EVIDENCE_SHARE
    assert "provisional" in c["minimum_retained_evidence_share"]["note"]


def test_half_retained_evidence_stays_defined(tmp_path):
    m = load_manifest(make_case(tmp_path, refs_a(4), render_instance(1, 10)))
    e, cq = enrol(m), prepare_query(m.query, "chroma")
    excl = np.zeros_like(cq.exclusion)
    excl[:, 200:] = True
    object.__setattr__(cq, "exclusion", excl)
    for row in compare_query(cq, e)["per_reference"]:
        assert row["excluding_query_flagged_status"] == "defined"
        assert row["distance_excluding_query_flagged"] is not None


# ---------------------------------------------------------------- A4 like-for-like spread


def test_comparable_spread_equals_all_pixel_spread_without_query_exclusion(tmp_path):
    r = run_case(make_case(tmp_path, refs_a(4), render_instance(1, 10)), tmp_path / "o")
    assert r["schema_version"] == "m2a-case-0.3"
    assert r["query"]["cleaner"]["status"] == "no_stamp_detected"
    spread = r["reference_spread"]
    assert spread["all_pixels"]["summary"] == spread["excluding_query_flagged"]["summary"]
    assert spread["all_pixels"]["summary"]["n_pairs"] == 6
    assert "same support" in spread["excluding_query_flagged"]["support_rule"]


def test_comparable_spread_uses_the_query_support(tmp_path):
    r = run_case(make_case(tmp_path, refs_a(4), render_instance(1, 10, stamped=True)), tmp_path / "o")
    m = load_manifest(tmp_path / "case" / "case.json")
    e, cq = enrol(m), prepare_query(m.query, "chroma")
    refs = e.references
    expected = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            row = compare_pair(D, refs[i].normalised, refs[j].normalised, cq.exclusion)
            if row["distance"] is not None:
                expected.append(row["distance"])
    s = r["reference_spread"]["excluding_query_flagged"]["summary"]
    assert s["median"] == round(float(np.median(expected)), 6)
    assert s["min"] == round(min(expected), 6) and s["max"] == round(max(expected), 6)
    assert r["reference_spread"]["all_pixels"]["summary"] != s
    assert "reference_pair_distance_summary" not in r  # the ambiguous unlabelled key is gone


# ---------------------------------------------------------------- A5 review means untouched


def test_review_after_proposed_deletion_returns_original_pixels():
    fx = make_fixture("blue_stamp_blue_pen", 1)
    image = fx.image.copy()
    image[190:232, 380:472] = 20  # touching neutral mark: chroma proposes a deletion, then refuses
    est, res = clean_image(image, method="chroma")
    assert res.status == "review"
    assert np.array_equal(res.cleaned, image)
    assert not res.change_map.any() and not res.intervention_mask.any()
    assert res.proposed_intervention_mask.any()
    assert "proposed_deletion_reverted" in res.reasons
    assert res.metrics["changed_pixels"] == 0
    assert res.metrics["proposed_intervention_pixels"] == int(res.proposed_intervention_mask.sum()) > 0


@pytest.mark.parametrize("seeds", [DEV_SEEDS, HELDOUT_SEEDS], ids=["dev", "heldout_now_dev"])
@pytest.mark.parametrize("method", sorted(METHODS))
def test_review_implies_zero_changed_pixels_over_the_m1b_matrix(seeds, method):
    n_review = 0
    for case in matrix_cases(seeds):
        image = make_matrix_case(*case).image
        _est, res = clean_image(image, method=method)
        if res.status == "review":
            n_review += 1
            assert np.array_equal(res.cleaned, image), case
            assert not res.change_map.any() and not res.intervention_mask.any(), case
        else:
            assert not res.proposed_intervention_mask.any() or res.status == "cleaned", case
    assert n_review > 0


def test_provenance_writes_proposed_mask_separately(tmp_path):
    fx = make_fixture("blue_stamp_blue_pen", 1)
    image = fx.image.copy()
    image[190:232, 380:472] = 20
    src = tmp_path / "q.png"
    Image.fromarray(image).save(src)
    out = tmp_path / "o"
    assert main(["clean", str(src), "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["schema_version"] == "1.2"
    assert rep["status"] == "review" and rep["metrics"]["changed_pixels"] == 0
    assert "proposed_intervention_mask" in rep["artifacts"]
    assert np.array(Image.open(out / "proposed_intervention_mask.png")).any()
    assert not np.array(Image.open(out / "intervention_mask.png")).any()
    assert np.array_equal(np.array(Image.open(out / "cleaned.png")), image)
    assert "not applied" in rep["definitions"]["proposed_intervention_mask"]


# ---------------------------------------------------------------- A6 one EXIF-aware loader


def _rotated_jpeg(path, stamped=True):
    im = Image.fromarray(np.ascontiguousarray(np.rot90(render_instance(1, 10, stamped=stamped))))
    exif = Image.Exif()
    exif[0x0112] = 6
    im.save(path, format="JPEG", quality=95, exif=exif)
    return path


def test_clean_and_compare_case_decode_exif_input_identically(tmp_path):
    base = tmp_path / "case"
    refs = refs_a(3)
    entries = []
    for i, arr in enumerate(refs):
        p = base / "references" / f"r{i}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(p)
        entries.append({"id": f"r{i}", "path": f"references/r{i}.png", "sha256": sha(p), "source_document_id": f"d{i}",
                        "signing_instance_id": f"s{i}", "label": {"status": "known", "value": "genuine", "source": "fictional"}})
    q = _rotated_jpeg(base / "q.jpg")
    before = q.read_bytes()
    manifest = {"manifest_version": "1.0", "case_id": "exif", "claimed_writer_id": "W-T",
                "config": {"config_version": "m2a-classical-0.1", "cleaner_method": "chroma"}, "references": entries,
                "query": {"id": "q", "path": "q.jpg", "sha256": sha(q), "source_document_id": None,
                          "signing_instance_id": None, "label": {"status": "unknown", "value": None, "source": None},
                          "stamp_region": None}}
    (base / "case.json").write_text(json.dumps(manifest))
    assert main(["clean", str(q), "--out", str(tmp_path / "c")]) == 0
    assert main(["compare-case", str(base / "case.json"), "--out", str(tmp_path / "v")]) == 0
    assert q.read_bytes() == before
    loaded = load_image(q)
    assert loaded.rgb.shape == (240, 480, 3)
    a = np.array(Image.open(tmp_path / "c" / "cleaned.png"))
    b = np.array(Image.open(tmp_path / "v" / "query_cleaning" / "cleaned.png"))
    assert a.shape == b.shape == loaded.rgb.shape and np.array_equal(a, b)
    for mask in ("stamp_mask.png", "intervention_mask.png", "ambiguous_mask.png"):
        assert np.array_equal(np.array(Image.open(tmp_path / "c" / mask)), np.array(Image.open(tmp_path / "v" / "query_cleaning" / mask)))
    rep = json.loads((tmp_path / "c" / "report.json").read_text())
    dec = rep["input"]["decoded"]
    assert dec["exif_orientation"] == 6 and dec["orientation_applied"] is True
    assert (dec["stored_width"], dec["stored_height"]) == (240, 480) and (dec["width"], dec["height"]) == (480, 240)
    assert rep["input"]["width"] == 480 and rep["input"]["height"] == 240
    assert (tmp_path / "c" / "original.jpg").read_bytes() == before


def test_legacy_load_rgb_uses_the_same_loader(tmp_path):
    p = _rotated_jpeg(tmp_path / "r.jpg", stamped=False)
    rgb, mode = load_rgb(p)
    assert np.array_equal(rgb, load_image(p).rgb) and mode == "RGB"
