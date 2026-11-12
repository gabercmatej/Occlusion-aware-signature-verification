import numpy as np
import pytest

from osv.datasets.synthetic_fixtures import KINDS, make_fixture


def test_seven_kinds():
    assert len(KINDS) == 7
    assert {"no_stamp", "blue_stamp_blue_pen", "black_stamp_black_pen", "gray_scan"} <= set(KINDS)


@pytest.mark.parametrize("kind", KINDS)
def test_shapes_and_mask_consistency(kind):
    fx = make_fixture(kind, 0, size=(240, 480))
    assert fx.image.shape == (240, 480, 3) and fx.image.dtype == np.uint8
    for m in (fx.pen_mask, fx.stamp_mask, fx.overlap_mask):
        assert m.shape == (240, 480) and m.dtype == bool
    assert np.array_equal(fx.overlap_mask, fx.pen_mask & fx.stamp_mask)
    assert fx.pen_mask.sum() > 1000
    assert fx.meta["kind"] == kind and fx.meta["seed"] == 0


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic_by_seed(kind):
    a, b, c = make_fixture(kind, 4), make_fixture(kind, 4), make_fixture(kind, 5)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.stamp_mask, b.stamp_mask)
    assert not np.array_equal(a.image, c.image)


def test_no_stamp_has_no_stamp():
    fx = make_fixture("no_stamp", 0)
    assert not fx.stamp_mask.any() and not fx.overlap_mask.any()


@pytest.mark.parametrize("kind", [k for k in KINDS if k != "no_stamp"])
@pytest.mark.parametrize("seed", range(5))
def test_stamp_overlaps_pen(kind, seed):
    fx = make_fixture(kind, seed)
    assert fx.stamp_mask.sum() > 500
    assert fx.overlap_mask.sum() > 0


def test_gray_scan_has_no_colour():
    img = make_fixture("gray_scan", 0).image
    assert np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2])


def test_darkening_mixture_not_opaque():
    """Overlap pixels are darker than stamp-only pixels: inks multiply, they do not cover."""
    fx = make_fixture("red_stamp_black_pen", 0)
    lum = fx.image.astype(float).mean(axis=2)
    stamp_only = fx.stamp_mask & ~fx.pen_mask
    assert lum[fx.overlap_mask].mean() < lum[stamp_only].mean()


def test_unknown_kind():
    with pytest.raises(ValueError):
        make_fixture("purple_stamp", 0)
