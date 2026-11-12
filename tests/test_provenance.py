"""End-to-end `clean` command: artifacts, hashes, determinism, reviewer region."""

import json

import numpy as np
import pytest
from PIL import Image

from osv.cli import main
from osv.datasets.synthetic_fixtures import KINDS, make_fixture
from osv.provenance import sha256_file

EXPECTED = {
    "cleaned.png", "change_map.png", "stamp_mask.png", "intervention_mask.png",
    "ink_mask.png", "ambiguous_mask.png", "report.json",
}


def write_fixture(path, kind, seed=0, fmt="PNG"):
    Image.fromarray(make_fixture(kind, seed).image).save(path, format=fmt)
    return path


def load(path):
    return np.array(Image.open(path))


def clean(tmp_path, src, name="out", region=None):
    out = tmp_path / name
    argv = ["clean", str(src), "--out", str(out)]
    if region:
        argv += ["--stamp-region", region]
    assert main(argv) == 0
    return out, json.loads((out / "report.json").read_text())


@pytest.mark.parametrize("kind", KINDS)
def test_artifacts_complete_and_consistent(tmp_path, kind):
    src = write_fixture(tmp_path / "query.png", kind)
    raw_before = src.read_bytes()
    out, report = clean(tmp_path, src)

    assert src.read_bytes() == raw_before
    assert EXPECTED | {"original.png"} == {p.name for p in out.iterdir()}
    assert (out / "original.png").read_bytes() == raw_before
    assert report["input"]["sha256"] == sha256_file(src)
    assert report["input"]["unchanged_after_run"] is True
    assert report["schema_version"] == "1.1"  # 1.1 (M1b): additive cleaner.method field
    for key in ("status", "reasons", "separability", "metrics", "cleaner", "artifacts", "status_meaning"):
        assert key in report
    for k in ("ink_fraction_reliable", "ink_fraction_ambiguous", "ink_fraction_changed"):
        assert k in report["metrics"]
    for art in report["artifacts"].values():
        assert sha256_file(out / art["file"]) == art["sha256"]

    original = load(src)
    cleaned = load(out / "cleaned.png")
    change = load(out / "change_map.png")
    inter = load(out / "intervention_mask.png") > 0
    stamp = load(out / "stamp_mask.png") > 0
    h, w = original.shape[:2]
    for name in ("stamp_mask.png", "intervention_mask.png", "ink_mask.png", "ambiguous_mask.png", "change_map.png"):
        assert load(out / name).shape == (h, w)
    recomputed = np.abs(cleaned.astype(int) - original.astype(int)).max(axis=2)
    assert np.array_equal(recomputed, change)
    assert not ((change > 0) & ~inter).any()
    assert not (inter & ~stamp).any()


def test_no_stamp_report_wording(tmp_path):
    src = write_fixture(tmp_path / "q.png", "no_stamp")
    out, report = clean(tmp_path, src)
    assert report["status"] == "no_stamp_detected"
    assert "does not establish that the image is clean" in report["status_meaning"]
    assert report["metrics"]["changed_pixels"] == 0
    assert np.array_equal(load(out / "cleaned.png"), load(src))


def test_byte_identical_reruns(tmp_path):
    src = write_fixture(tmp_path / "q.png", "red_stamp_black_pen", 2)
    a, _ = clean(tmp_path, src, "a")
    b, _ = clean(tmp_path, src, "b")
    for name in EXPECTED | {"original.png"}:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_filename_does_not_affect_processing(tmp_path):
    img = make_fixture("blue_stamp_black_pen", 1).image
    p1, p2 = tmp_path / "no_stamp.png", tmp_path / "blue_stamp_blue_pen.png"
    Image.fromarray(img).save(p1)
    Image.fromarray(img).save(p2)
    a, ra = clean(tmp_path, p1, "a")
    b, rb = clean(tmp_path, p2, "b")
    assert ra["status"] == rb["status"] and ra["reasons"] == rb["reasons"]
    for name in EXPECTED - {"report.json"}:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_jpeg_original_copied_byte_identical(tmp_path):
    src = write_fixture(tmp_path / "q.JPG", "red_stamp_black_pen", fmt="JPEG")
    out, report = clean(tmp_path, src)
    assert (out / "original.jpg").read_bytes() == src.read_bytes()
    assert report["input"]["filename"] == "q.JPG"


def test_black_stamp_with_region_via_cli(tmp_path):
    fx = make_fixture("black_stamp_black_pen", 0)
    ys, xs = np.nonzero(fx.stamp_mask)
    region = f"{xs.min()},{ys.min()},{xs.max() - xs.min() + 1},{ys.max() - ys.min() + 1}"
    src = tmp_path / "q.png"
    Image.fromarray(fx.image).save(src)
    _, no_hint = clean(tmp_path, src, "plain")
    out, hint = clean(tmp_path, src, "hint", region)
    assert no_hint["status"] == "no_stamp_detected"
    assert hint["status"] == "review"
    assert "stamp_indicated_not_separated" in hint["reasons"]
    assert hint["metrics"]["changed_pixels"] == 0
    x, y, w, h = (int(v) for v in region.split(","))
    assert hint["reviewer_stamp_region"] == {"x": x, "y": y, "w": w, "h": h}
    assert (load(out / "ambiguous_mask.png")[y:y + h, x:x + w] > 0).all()


def test_refuses_non_empty_output(tmp_path, capsys):
    src = write_fixture(tmp_path / "q.png", "no_stamp")
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert main(["clean", str(src), "--out", str(out)]) == 2
    assert (out / "keep.txt").read_text() == "x"


def test_region_outside_image_is_error(tmp_path):
    src = write_fixture(tmp_path / "q.png", "no_stamp")
    assert main(["clean", str(src), "--out", str(tmp_path / "o"), "--stamp-region", "470,0,50,50"]) == 2
