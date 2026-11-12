"""M2a slice 1: image loading, normalisation, manifest, enrolment, classical baseline comparison.

All inputs are fictional synthetic images. Passing these tests establishes software behaviour
only; synthetic identities say nothing about writer discrimination or forgery rejection.
"""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from osv.cli import main
from osv.datasets.synthetic_cases import render_instance, write_demo_case
from osv.datasets.synthetic_fixtures import make_fixture
from osv.preprocessing.image_io import UnsupportedImageError, load_image
from osv.preprocessing.normalise import NormaliseParams, compute_transform, normalise
from osv.verification.compare import (
    GENERAL_LIMITATIONS, CleanedQuery, compare_query, prepare_query, run_case,
)
from osv.verification.descriptor import GridOrientationDescriptor
from osv.verification.enrolment import enrol
from osv.verification.manifest import CONFIG_VERSION, ManifestError, load_manifest

GENUINE = {"status": "known", "value": "genuine", "source": "fictional test"}
UNKNOWN = {"status": "unknown", "value": None, "source": None}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, array, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, **kw)
    return path


def make_case(tmp_path, refs, query, *, ref_dir="references", query_dir="queries", query_label=UNKNOWN,
              ref_meta=None, query_meta=None, region=None, method="chroma", name="case.json"):
    base = tmp_path / "case"
    entries = []
    for i, arr in enumerate(refs):
        rel = f"{ref_dir}/r{i}.png"
        save(base / rel, arr, format="PNG")
        meta = (ref_meta or {}).get(i, {})
        entries.append({"id": f"r{i}", "path": rel, "sha256": sha(base / rel),
                        "source_document_id": meta.get("doc", f"doc-{i}"),
                        "signing_instance_id": meta.get("inst", f"sig-{i}"), "label": GENUINE})
    qrel = f"{query_dir}/q.png"
    save(base / qrel, query, format="PNG", compress_level=9)
    qm = query_meta or {}
    manifest = {
        "manifest_version": "1.0", "case_id": "t1", "claimed_writer_id": "W-T",
        "config": {"config_version": CONFIG_VERSION, "cleaner_method": method},
        "references": entries,
        "query": {"id": "q", "path": qrel, "sha256": sha(base / qrel), "source_document_id": qm.get("doc"),
                  "signing_instance_id": qm.get("inst"), "label": query_label, "stamp_region": region},
    }
    path = base / name
    path.write_text(json.dumps(manifest, indent=2))
    return path


def refs_a(n=4):
    return [render_instance(1, i) for i in range(n)]


# ---------------------------------------------------------------- normalisation geometry


@pytest.mark.parametrize("size", [(240, 480), (100, 600), (300, 200), (64, 64), (500, 1300)])
def test_transform_preserves_aspect_and_never_crops(size):
    h, w = size
    t = compute_transform(w, h)
    assert t.target_width <= t.canvas_width and t.target_height <= t.canvas_height
    assert abs(t.scale_x - t.scale_y) <= 1.0 / min(w, h) + 1e-9  # only rounding of the target size
    corners = t.map_points(np.array([[0, 0], [w, h]], float))
    assert corners.min() >= 0 and corners[1, 0] <= t.canvas_width and corners[1, 1] <= t.canvas_height
    pts = np.array([[3.25, 7.5], [w - 1.0, h / 2]])
    assert np.allclose(t.unmap_points(t.map_points(pts)), pts)


@pytest.mark.parametrize("size,centre", [((240, 480), (100, 60)), ((300, 200), (150, 220)), ((90, 700), (650, 45))])
def test_image_and_masks_share_one_geometry(size, centre):
    h, w = size
    rgb = np.full((h, w, 3), 240, np.uint8)
    yy, xx = np.mgrid[:h, :w]
    disc = (xx + 0.5 - centre[0]) ** 2 + (yy + 0.5 - centre[1]) ** 2 <= 20 ** 2
    rgb[disc] = 20
    n = normalise(rgb, {"ink_mask": disc, "stamp_mask": disc})
    expected = n.transform.map_points(np.array(centre, float))

    def centroid(weight):
        ys, xs = np.mgrid[:weight.shape[0], :weight.shape[1]]
        return np.array([(weight * (xs + 0.5)).sum(), (weight * (ys + 0.5)).sum()]) / weight.sum()

    assert np.allclose(centroid(n.masks["ink_mask"].astype(float)), expected, atol=0.5)
    assert np.allclose(centroid(n.masks["stamp_mask"].astype(float)), expected, atol=0.75)
    assert np.allclose(centroid(n.darkness), expected, atol=0.5)
    # Mask and image agree pixel-wise on where the disc is.
    ink = n.darkness > 0.5
    assert (ink & n.masks["ink_mask"]).sum() / (ink | n.masks["ink_mask"]).sum() > 0.9


def test_evidence_limiting_mask_survives_downscaling():
    rgb = np.full((600, 1500, 3), 240, np.uint8)
    one = np.zeros((600, 1500), bool)
    one[301, 777] = True
    n = normalise(rgb, {"intervention_mask": one, "ink_mask": one})
    assert n.masks["intervention_mask"].sum() >= 1  # "any" rule: never silently dropped
    assert n.masks["ink_mask"].sum() == 0  # ink needs half coverage
    assert n.record()["mask_rules"] == {"ink_mask": "half", "intervention_mask": "any"}


def test_normalise_rejects_mismatched_or_unknown_masks():
    rgb = np.full((50, 80, 3), 240, np.uint8)
    with pytest.raises(ValueError):
        normalise(rgb, {"ink_mask": np.zeros((50, 81), bool)})
    with pytest.raises(ValueError):
        normalise(rgb, {"mystery": np.zeros((50, 80), bool)})


def test_residual_stamp_and_edge_strokes_stay_inside_canvas():
    rgb = np.full((240, 480, 3), 240, np.uint8)
    rgb[0:4, 0:30] = (200, 40, 50)  # residual coloured ink in the corner
    rgb[-4:, -30:] = 20  # stroke at the far edge
    n = normalise(rgb)
    t = n.transform
    assert n.darkness[t.offset_y:t.offset_y + 2, t.offset_x:t.offset_x + 10].mean() > 0.2
    assert n.darkness[t.offset_y + t.target_height - 2:t.offset_y + t.target_height,
                      t.offset_x + t.target_width - 10:t.offset_x + t.target_width].mean() > 0.5


# ---------------------------------------------------------------- loading and formats


def test_exif_orientation_applied_and_recorded(tmp_path):
    raw = np.zeros((40, 70, 3), np.uint8)
    raw[:, :, 0] = np.arange(70, dtype=np.uint8)[None, :]
    raw[:, :, 1] = np.arange(40, dtype=np.uint8)[:, None]
    exif = Image.Exif()
    exif[0x0112] = 6  # display = stored rotated 90 degrees clockwise
    path = tmp_path / "rot.png"
    Image.fromarray(raw).save(path, format="PNG", exif=exif)
    before = path.read_bytes()
    img = load_image(path)
    assert path.read_bytes() == before
    assert img.exif_orientation == 6 and img.orientation_applied
    assert img.stored_size == (70, 40) and img.rgb.shape == (70, 40, 3)
    assert np.array_equal(img.rgb, np.rot90(raw, k=-1))
    rec = img.record()
    assert rec["width"] == 40 and rec["stored_width"] == 70


def test_supported_formats_load(tmp_path):
    arr = render_instance(1, 0)
    for fmt, ext in (("PNG", "png"), ("JPEG", "jpg"), ("BMP", "bmp"), ("TIFF", "tif")):
        img = load_image(save(tmp_path / f"x.{ext}", arr, format=fmt))
        assert img.format == fmt and img.rgb.shape == arr.shape and not img.orientation_applied
    gray = load_image(save(tmp_path / "g.png", arr[..., 0], format="PNG"))
    assert gray.mode == "L" and gray.rgb.shape == arr.shape


def test_multipage_pdf_transparency_and_16bit_are_refused(tmp_path):
    a = Image.fromarray(render_instance(1, 0))
    b = Image.fromarray(render_instance(1, 1))
    a.save(tmp_path / "multi.tif", format="TIFF", save_all=True, append_images=[b])
    with pytest.raises(UnsupportedImageError, match="multipage"):
        load_image(tmp_path / "multi.tif")
    a.save(tmp_path / "doc.pdf", format="PDF")
    with pytest.raises(UnsupportedImageError, match="PDF"):
        load_image(tmp_path / "doc.pdf")
    rgba = np.dstack([render_instance(1, 0), np.full((240, 480), 255, np.uint8)])
    rgba[0, 0, 3] = 0
    with pytest.raises(UnsupportedImageError, match="transparent"):
        load_image(save(tmp_path / "t.png", rgba, format="PNG"))
    Image.fromarray(np.full((20, 30), 40000, np.uint16)).save(tmp_path / "deep.png")
    with pytest.raises(UnsupportedImageError, match="mode"):
        load_image(tmp_path / "deep.png")


# ---------------------------------------------------------------- descriptor


def test_identical_image_has_zero_distance_software_sanity_only():
    d = GridOrientationDescriptor()
    n = normalise(render_instance(1, 0))
    v = d.extract(n.darkness)
    assert d.distance(v, v) == 0.0
    assert d.distance(v, d.extract(normalise(render_instance(1, 0)).darkness)) == 0.0
    assert d.extract(np.zeros_like(n.darkness)) is None  # blank: no evidence, not a distance of 0


def test_exclusion_removes_pixels_exactly():
    d = GridOrientationDescriptor()
    n = normalise(render_instance(1, 0))
    excl = np.zeros_like(n.valid_mask)
    excl[:, :160] = True
    altered = n.darkness.copy()
    altered[:, :150] = 0.0  # change only well inside the excluded area (gradients stay excluded)
    assert d.distance(d.extract(n.darkness, excl), d.extract(altered, excl)) == pytest.approx(0.0, abs=1e-12)
    assert d.distance(d.extract(n.darkness), d.extract(altered)) > 0.1


# ---------------------------------------------------------------- manifests


def _write_raw(tmp_path, data):
    p = tmp_path / "case" / "case.json"
    p.write_text(json.dumps(data))
    return p


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.pop("query"), "missing top-level"),
    (lambda m: m.update(extra=1), "unknown top-level"),
    (lambda m: m.update(manifest_version="0.9"), "manifest_version"),
    (lambda m: m["config"].update(config_version="m2a-classical-9"), "not supported"),
    (lambda m: m["config"].update(cleaner_method="magic"), "cleaner_method"),
    (lambda m: m.update(claimed_writer_id="Jane Doe"), "claimed_writer_id"),
    (lambda m: m["references"][0].update(path="/abs/r0.png"), "absolute"),
    (lambda m: m["references"][0].update(path="C:/x/r0.png"), "absolute"),
    (lambda m: m["references"][0].update(path="references/missing.png"), "not found"),
    (lambda m: m["references"][0].update(sha256="0" * 64), "sha256 mismatch"),
    (lambda m: m["references"][0].update(sha256="ABC"), "64 lowercase"),
    (lambda m: m["references"][0].pop("label"), "missing keys"),
    (lambda m: m["references"][0].update(label={"status": "known", "value": "forged", "source": "x"}), "only genuine"),
    (lambda m: m["references"][0].update(label={"status": "known", "value": "genuine", "source": ""}), "source"),
    (lambda m: m["query"].update(label={"status": "unknown", "value": "genuine", "source": None}), "value null"),
    (lambda m: m["query"].update(label={"status": "probably", "value": None, "source": None}), "status"),
    (lambda m: m["query"].update(stamp_region=[1, 2, 0, 5]), "stamp_region"),
    (lambda m: m.update(references=[]), "non-empty"),
    (lambda m: m["references"][1].update(id="r0"), "unique"),
    (lambda m: m["query"].update(signing_instance_id="sig-0"), "signing_instance"),
])
def test_invalid_manifests_are_refused(tmp_path, mutate, match):
    p = make_case(tmp_path, refs_a(3), render_instance(1, 10))
    data = json.loads(p.read_text())
    mutate(data)
    with pytest.raises(ManifestError, match=match):
        load_manifest(_write_raw(tmp_path, data))


def test_non_json_and_missing_manifest(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "bad.json")
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")


def test_query_byte_identical_to_reference_is_refused(tmp_path):
    refs = refs_a(3)
    p = make_case(tmp_path, refs, refs[0])
    data = json.loads(p.read_text())
    data["query"]["path"] = data["references"][0]["path"]
    data["query"]["sha256"] = data["references"][0]["sha256"]
    with pytest.raises(ManifestError, match="byte-identical"):
        load_manifest(_write_raw(tmp_path, data))


def test_overlap_warnings_and_reference_count(tmp_path):
    refs = refs_a(2)
    refs.append(refs[0])  # duplicate file content
    p = make_case(tmp_path, refs, render_instance(1, 10),
                  ref_meta={1: {"doc": "doc-0", "inst": "sig-1"}}, query_meta={"doc": "doc-0"})
    m = load_manifest(p)
    assert {"duplicate_reference_files", "references_share_source_document",
            "query_shares_source_document_with_reference"} <= set(m.warnings)
    e = enrol(m)
    assert e.unique_count == 2 and "fewer_than_three_references" in e.flags


def test_missing_input_at_cli_returns_error(tmp_path, capsys):
    assert main(["compare-case", str(tmp_path / "absent.json"), "--out", str(tmp_path / "o")]) == 2
    assert "manifest not found" in capsys.readouterr().err


def test_multipage_reference_is_not_silently_reduced(tmp_path, capsys):
    p = make_case(tmp_path, refs_a(3), render_instance(1, 10))
    data = json.loads(p.read_text())
    multi = tmp_path / "case" / "references" / "multi.tif"
    Image.fromarray(render_instance(1, 5)).save(multi, format="TIFF", save_all=True,
                                                append_images=[Image.fromarray(render_instance(1, 6))])
    data["references"][0].update(path="references/multi.tif", sha256=sha(multi))
    assert main(["compare-case", str(_write_raw(tmp_path, data)), "--out", str(tmp_path / "o")]) == 2
    assert "multipage" in capsys.readouterr().err


# ---------------------------------------------------------------- end to end


def test_demo_case_runs_and_preserves_originals(tmp_path, capsys):
    manifests = write_demo_case(tmp_path / "demo")
    before = {p: sha(p) for p in (tmp_path / "demo").rglob("*.png")}
    assert main(["compare-case", str(manifests[0]), "--out", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("UNVALIDATED CLASSICAL BASELINE")
    assert {p: sha(p) for p in before} == before
    out = tmp_path / "out"
    report = json.loads((out / "case_report.json").read_text())
    assert sha(out / "query_cleaning" / "original.png") == report["query"]["sha256"]
    for ref in json.loads((out / "enrolment.json").read_text())["references"]:
        assert sha(out / "references" / f"{ref['id']}_original.png") == ref["sha256"]
    text = (out / "case_report.json").read_text() + (out / "enrolment.json").read_text()
    assert str(tmp_path) not in text  # no absolute local paths
    assert report["reference_count"] == 4 and len(report["comparison"]["per_reference"]) == 4


def test_runs_are_deterministic(tmp_path):
    p = write_demo_case(tmp_path / "demo")[1]
    run_case(p, tmp_path / "o1")
    run_case(p, tmp_path / "o2")
    files = sorted(f.relative_to(tmp_path / "o1") for f in (tmp_path / "o1").rglob("*") if f.is_file())
    assert files
    for f in files:
        assert (tmp_path / "o1" / f).read_bytes() == (tmp_path / "o2" / f).read_bytes(), f


def test_refuses_non_empty_output(tmp_path):
    p = write_demo_case(tmp_path / "demo")[0]
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        run_case(p, tmp_path / "o")


def test_same_pixels_as_reference_rank_first_sanity_only(tmp_path):
    refs = refs_a(4)
    p = make_case(tmp_path, refs, refs[2].copy())  # same pixels, different PNG bytes
    r = run_case(p, tmp_path / "o")
    rows = {row["reference_id"]: row["distance_all_pixels"] for row in r["comparison"]["per_reference"]}
    assert r["query"]["cleaner"]["status"] == "no_stamp_detected"
    assert rows["r2"] == 0.0 and min(rows, key=rows.get) == "r2"


def test_folder_names_and_labels_do_not_change_comparison(tmp_path):
    q = render_instance(1, 10, stamped=True)
    a = run_case(make_case(tmp_path / "a", refs_a(3), q), tmp_path / "oa")
    b = run_case(make_case(tmp_path / "b", refs_a(3), q, ref_dir="forged", query_dir="genuine",
                           query_label={"status": "known", "value": "forged", "source": "test"}), tmp_path / "ob")
    assert a["comparison"]["per_reference"] == b["comparison"]["per_reference"]
    assert a["review_flags"] == b["review_flags"]


def test_raw_query_cannot_bypass_cleaning(tmp_path):
    m = load_manifest(make_case(tmp_path, refs_a(3), render_instance(1, 10)))
    e = enrol(m)
    with pytest.raises(TypeError):
        compare_query(render_instance(1, 10), e)
    good = prepare_query(m.query, "chroma")
    forged_obj = CleanedQuery(good.entry, good.loaded, good.method, good.estimate, good.result,
                              good.normalised, good.exclusion)
    with pytest.raises(TypeError):
        compare_query(forged_obj, e)
    assert compare_query(good, e)["per_reference"]
    with pytest.raises(ValueError, match="different cleaner"):
        compare_query(prepare_query(m.query, "optical-density"), e)


def test_query_is_compared_after_cleaning(tmp_path):
    q = render_instance(1, 10, stamped=True)
    m = load_manifest(make_case(tmp_path, refs_a(3), q))
    cq = prepare_query(m.query, "chroma")
    assert cq.result.status == "cleaned" and cq.result.metrics["changed_pixels"] > 0
    assert np.array_equal(cq.normalised.darkness, normalise(cq.result.cleaned).darkness)
    assert not np.array_equal(cq.normalised.darkness, normalise(q).darkness)


_FORBIDDEN_KEYS = ("verdict", "probability", "is_genuine", "is_forged", "authentic", "safe", "decision")
_FORBIDDEN_VALUES = {"genuine", "forged", "safe", "authentic", "match", "no_match"}


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path + "/" + k, k, v
            yield from _walk(v, path + "/" + k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


@pytest.mark.parametrize("query_kind,status", [
    ("no_stamp", "no_stamp_detected"),
    ("red_stamp_black_pen", "cleaned"),
    ("blue_stamp_blue_pen", "review"),
])
def test_status_propagation_without_safety_claims(tmp_path, query_kind, status):
    q = make_fixture(query_kind, 3).image
    r = run_case(make_case(tmp_path, [make_fixture("no_stamp", s).image for s in (0, 1, 2)], q), tmp_path / "o")
    assert r["query"]["cleaner"]["status"] == status
    assert f"cleaner_status:{status}" in r["review_flags"]
    lim = r["evidence_limitations"]
    assert all(g in lim for g in GENERAL_LIMITATIONS)
    assert any("not proven reliable" in x for x in lim)
    assert any(status in x for x in lim)
    if status == "no_stamp_detected":
        assert any("does not mean" in x for x in lim)
    if status == "review":
        assert any(f.startswith("cleaner_reason:") for f in r["review_flags"])
    assert "UNVALIDATED" in r["result_type"]
    assert "NOT proven reliable" in r["query"]["evidence"]["ink_share_in_descriptor_support_meaning"]
    enrol_json = json.loads((tmp_path / "o" / "enrolment.json").read_text())
    for doc in (r, enrol_json):
        for path, key, value in _walk(doc):
            assert not any(f in key.lower() for f in _FORBIDDEN_KEYS), path
            if isinstance(value, str):
                assert value.lower() not in _FORBIDDEN_VALUES, path


def test_reviewer_region_is_excluded(tmp_path):
    q = render_instance(1, 10)
    m = load_manifest(make_case(tmp_path, refs_a(3), q, region=[200, 60, 80, 100]))
    cq = prepare_query(m.query, "chroma")
    t = cq.normalised.transform
    (x0, y0), (x1, y1) = t.map_points(np.array([[200, 60], [280, 160]], float))
    inner = cq.exclusion[int(np.ceil(y0)):int(y1), int(np.ceil(x0)):int(x1)]
    assert inner.size and inner.all()
    assert "reviewer_region_used" in cq.result.reasons or cq.result.status == "review"


def test_enrolment_flags_stamped_reference_and_outlier(tmp_path):
    refs = refs_a(3) + [render_instance(2, 0)]  # a different fictional writer among the references
    refs[0] = render_instance(1, 0, stamped=True)
    e = enrol(load_manifest(make_case(tmp_path, refs, render_instance(1, 10))))
    assert "reference_stamp_indicated:r0" in e.flags
    assert "reference_spread_outlier:r3" in e.flags
    assert e.record()["flag_note"].startswith("descriptive")


def test_enrolment_is_deterministic_and_versioned(tmp_path):
    m = load_manifest(make_case(tmp_path, refs_a(4), render_instance(1, 10)))
    a, b = enrol(m).record(), enrol(m).record()
    assert a == b
    assert a["descriptor"]["name"] == "grid_orientation_histogram" and a["descriptor"]["version"]
    assert all(r["normalisation"]["normaliser"]["version"] for r in a["references"])
    assert all(len(r["sha256"]) == 64 and r["descriptor_sha256"] for r in a["references"])


def test_canvas_params_recorded():
    n = normalise(render_instance(1, 0), params=NormaliseParams(canvas_height=96, canvas_width=256))
    assert n.darkness.shape == (96, 256)
    assert n.record()["normaliser"]["parameters"]["canvas_width"] == 256
