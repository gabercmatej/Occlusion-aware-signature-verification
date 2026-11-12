"""M2a correctness: descriptor-level exclusion semantics.

Guarantee under test (descriptor 0.2.0): with an exclusion mask ``E`` on the normalised canvas,
the masked descriptor depends only on canvas darkness at pixels outside ``E``. Every gradient
vote whose computation reads an excluded pixel (Gaussian pre-blur followed by 3x3 Sobel, with
reflect-101 borders) is dropped; excluded content is never replaced by a constant.

Tolerance: two arrays identical outside ``E`` feed bit-identical inputs into every retained
vote, so the vectors are expected to be exactly equal. ``atol=1e-12`` is used only to keep the
assertion meaningful if a future OpenCV build reorders float operations per position; it is far
below any distance that matters (descriptor entries are O(1e-3)).

All inputs are synthetic; these tests establish software behaviour only, not verification quality.
"""

import cv2
import numpy as np
import pytest

from osv.datasets.synthetic_cases import render_instance
from osv.preprocessing.normalise import normalise
from osv.verification.compare import compare_query, prepare_query, query_evidence, run_case
from osv.verification.descriptor import (
    DESCRIPTOR_VERSION, DescriptorParams, GridOrientationDescriptor, InsufficientEvidenceError,
)
from osv.verification.enrolment import enrol
from osv.verification.manifest import load_manifest

from test_m2a import make_case, refs_a

H, W = 128, 320
ATOL = 1e-12


def _strokes(seed: int) -> np.ndarray:
    """Random synthetic stroke-like darkness canvas (not a signature model)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.uint8)
    for _ in range(14):
        pts = rng.integers([0, 0], [W, H], size=(4, 2)).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], False, int(rng.integers(120, 255)), int(rng.integers(1, 4)))
    return (img.astype(np.float32) / 255.0) * 0.9


def _assert_same(a, b):
    assert a is not None and b is not None
    np.testing.assert_allclose(a, b, rtol=0, atol=ATOL)


# ------------------------------------------------------------ reported regression


def _probe():
    a = np.zeros((H, W), np.float32)
    a[20:100, 35:39] = 0.8
    flagged = np.zeros(a.shape, np.uint8)
    flagged[45:80, 150:175] = 1
    exclude = cv2.dilate(flagged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    b = a.copy()
    b[flagged.astype(bool)] = 0.8
    return a, b, exclude


def test_reported_probe_changes_inside_mask_do_not_leak():
    # Before the fix this returned ~0.026138 although a and b agree outside the mask.
    a, b, exclude = _probe()
    assert np.array_equal(a[~exclude], b[~exclude])
    d = GridOrientationDescriptor()
    va, vb = d.extract(a, exclude=exclude), d.extract(b, exclude=exclude)
    _assert_same(va, vb)
    assert d.distance(va, vb) <= ATOL


# ------------------------------------------------------------ footprint


def test_footprint_is_explicit_and_recorded():
    d = GridOrientationDescriptor()
    assert d.blur_radius == 4  # ceil(4 * sigma), sigma = 1.0; equals OpenCV's own float32 default
    assert d.footprint_radius == 5
    rec = d.record()
    assert rec["version"] == DESCRIPTOR_VERSION == "0.2.0"
    assert rec["exclusion"]["footprint_radius_px"] == 5
    assert "reflect-101" in rec["exclusion"]["rule"]
    assert GridOrientationDescriptor(DescriptorParams(pre_blur_sigma=0.0)).footprint_radius == 1


@pytest.mark.parametrize("yx", [(64, 160), (0, 0), (0, 319), (127, 0), (127, 319), (0, 100),
                                (127, 200), (50, 0), (50, 319), (2, 3), (63, 31), (64, 32)])
def test_single_pixel_perturbation_stays_inside_declared_support(yx):
    """Empirical check of the footprint, including reflect-101 borders."""
    d = GridOrientationDescriptor()
    base = _strokes(5)
    mag0, ang0 = d.magnitude_and_orientation(base)
    excl = np.zeros((H, W), bool)
    excl[yx] = True
    support = d.support(excl)
    for value in (0.0, 0.5, 1.0):
        pert = base.copy()
        pert[yx] = value
        mag1, ang1 = d.magnitude_and_orientation(pert)
        changed = (mag0 != mag1) | (ang0 != ang1)
        assert not (changed & support).any()


def test_interior_support_is_tight_square():
    d = GridOrientationDescriptor()
    excl = np.zeros((H, W), bool)
    excl[64, 160] = True
    dropped = ~d.support(excl)
    r = d.footprint_radius
    assert dropped.sum() == (2 * r + 1) ** 2
    assert dropped[64 - r:64 + r + 1, 160 - r:160 + r + 1].all()


# ------------------------------------------------------------ invariance sweep


_MASKS = {
    "probe_like": (slice(45, 80), slice(150, 175)),
    "single_pixel": (slice(70, 71), slice(90, 91)),
    "small_3x3": (slice(10, 13), slice(10, 13)),
    "top_left_corner": (slice(0, 20), slice(0, 30)),
    "bottom_right_corner": (slice(110, 128), slice(290, 320)),
    "left_edge_band": (slice(0, 128), slice(0, 6)),
    "top_edge_band": (slice(0, 4), slice(40, 300)),
    "across_cell_boundary": (slice(28, 37), slice(60, 70)),
    "large_half": (slice(0, 128), slice(0, 160)),
    "thin_row": (slice(64, 65), slice(0, 320)),
}


@pytest.mark.parametrize("name", sorted(_MASKS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_arrays_identical_outside_mask_give_identical_descriptor(name, seed):
    d = GridOrientationDescriptor()
    a = _strokes(seed)
    excl = np.zeros((H, W), bool)
    excl[_MASKS[name]] = True
    rng = np.random.default_rng(100 + seed)
    for fill in (np.zeros_like(a), np.ones_like(a), rng.random(a.shape).astype(np.float32), _strokes(seed + 50)):
        b = np.where(excl, fill, a).astype(np.float32)
        assert np.array_equal(a[~excl], b[~excl])
        _assert_same(d.extract(a, excl), d.extract(b, excl))


def test_excluded_boundary_is_not_counted_as_handwriting():
    """Content only inside the mask leaves no evidence; nothing is filled in and edged."""
    d = GridOrientationDescriptor()
    a = np.zeros((H, W), np.float32)
    a[40:90, 100:200] = 0.9  # dark block entirely inside the exclusion
    excl = np.zeros((H, W), bool)
    excl[40:90, 100:200] = True
    res = d.describe(a, excl)
    assert res.vector is None and res.status == "insufficient_edge_evidence"
    assert res.retained_edge_magnitude == 0.0


# ------------------------------------------------------------ unmasked behaviour


def _old_unmasked(darkness):
    """Descriptor 0.1.0 magnitude/orientation path (cv2 default ksize for sigma=1)."""
    blurred = cv2.GaussianBlur(darkness.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy), np.mod(np.arctan2(gy, gx), np.pi)


@pytest.mark.parametrize("seed", [0, 3])
def test_unmasked_behaviour_unchanged(seed):
    d = GridOrientationDescriptor()
    img = _strokes(seed)
    mag, ang = d.magnitude_and_orientation(img)
    mag0, ang0 = _old_unmasked(img)
    assert np.array_equal(mag, mag0) and np.array_equal(ang, ang0)
    v = d.extract(img)
    assert np.array_equal(v, d.extract(img, np.zeros((H, W), bool)))
    assert d.describe(img).support_pixels == H * W


# ------------------------------------------------------------ no evidence


def test_no_remaining_evidence_is_explicit_not_zero():
    d = GridOrientationDescriptor()
    img = _strokes(1)
    everything = np.ones((H, W), bool)
    res = d.describe(img, everything)
    assert res.vector is None and res.status == "insufficient_edge_evidence"
    assert res.support_pixels == 0
    assert d.extract(img, everything) is None
    with pytest.raises(InsufficientEvidenceError):
        d.distance(None, d.extract(img))
    with pytest.raises(InsufficientEvidenceError):
        d.distance(None, None)


def test_comparison_reports_undefined_rows_when_query_evidence_is_gone(tmp_path):
    m = load_manifest(make_case(tmp_path, refs_a(3), render_instance(1, 10)))
    e = enrol(m)
    cq = prepare_query(m.query, "chroma")
    object.__setattr__(cq, "exclusion", np.ones_like(cq.exclusion))
    c = compare_query(cq, e)
    for row in c["per_reference"]:
        assert row["distance_excluding_query_flagged"] is None
        assert row["excluding_query_flagged_status"] == "insufficient_query_edge_evidence"
        assert row["distance_all_pixels"] is not None
    agg = c["aggregate"]["distance_excluding_query_flagged"]
    assert agg == {"n": 0, "median": None, "min": None, "max": None}
    assert c["descriptor_support"]["support_pixels"] == 0
    ev = query_evidence(cq)
    assert ev["query_edge_share_in_descriptor_support"] == 0.0
    assert ev["ink_share_in_descriptor_support"] == 0.0


# ------------------------------------------------------------ comparison accounting


def test_query_and_reference_use_the_same_support_and_report_it(tmp_path):
    q = render_instance(1, 10, stamped=True)
    r = run_case(make_case(tmp_path, refs_a(3), q), tmp_path / "o")
    assert r["schema_version"] == "m2a-case-0.2"
    c = r["comparison"]
    sup = c["descriptor_support"]
    assert sup["footprint_radius_px"] == 5 and "reflect-101" in sup["rule"]
    assert 0 < sup["support_pixels"] < 128 * 320
    ev = r["query"]["evidence"]
    assert "descriptor_support_rule" in ev and "exclusion_mask_rule" in ev
    # The support is strictly smaller than the complement of the dilated mask whenever something was flagged.
    assert ev["descriptor_support_share_of_image_area"] < 1.0 - ev["exclusion_mask_share_of_image_area"]
    assert ev["ink_share_in_descriptor_support"] <= ev["ink_share_outside_exclusion_mask"]
    for row in c["per_reference"]:
        assert row["excluding_query_flagged_status"] == "defined"
        assert 0.0 <= row["reference_edge_share_outside_descriptor_support"] <= 1.0
    # Reference recomputed independently with the same support gives the same distance.
    m = load_manifest(tmp_path / "case" / "case.json")
    e, cq = enrol(m), prepare_query(m.query, "chroma")
    d = e.descriptor
    qv = d.extract(cq.normalised.darkness, cq.exclusion)
    for ref, row in zip(e.references, c["per_reference"]):
        assert round(d.distance(qv, d.extract(ref.normalised.darkness, cq.exclusion)), 6) == \
            row["distance_excluding_query_flagged"]


# ------------------------------------------------------------ boundary of the guarantee (normalisation)


def _rgb_with_ink(seed):
    rgb = np.full((200, 500, 3), 245, np.uint8)
    rng = np.random.default_rng(seed)
    for _ in range(10):
        pts = rng.integers([0, 0], [500, 200], size=(4, 2)).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(rgb, [pts], False, (20, 30, 120), 3)
    return rgb


def test_normalisation_resampling_respects_any_rule_when_paper_level_is_unchanged():
    """Local part of normalisation: INTER_AREA + 'any' mask rule keeps unflagged canvas pixels
    independent of flagged source pixels (up to the 1e-6 coverage epsilon)."""
    a = _rgb_with_ink(0)
    flag = np.zeros(a.shape[:2], bool)
    flag[60:100, 200:260] = True  # small region: the 95th-percentile paper level does not move
    b = a.copy()
    b[flag] = (200, 0, 0)
    na, nb = normalise(a, {"stamp_mask": flag}), normalise(b, {"stamp_mask": flag})
    assert na.paper_luma == nb.paper_luma
    outside = ~na.masks["stamp_mask"]
    assert np.max(np.abs(na.darkness[outside] - nb.darkness[outside])) <= 1e-6


def test_normalisation_paper_level_is_global_documented_limitation():
    """Not covered by the descriptor guarantee: the paper luma percentile is computed over the
    whole image, so a large flagged region can rescale darkness everywhere."""
    a = _rgb_with_ink(0)
    flag = np.zeros(a.shape[:2], bool)
    flag[:, :300] = True  # 60 % of the image
    b = a.copy()
    b[flag] = (255, 255, 255)  # e.g. cleaned to bright paper: moves the 95th luma percentile
    na, nb = normalise(a, {"stamp_mask": flag}), normalise(b, {"stamp_mask": flag})
    assert na.paper_luma != nb.paper_luma
    outside = ~na.masks["stamp_mask"]
    assert np.max(np.abs(na.darkness[outside] - nb.darkness[outside])) > 1e-3
