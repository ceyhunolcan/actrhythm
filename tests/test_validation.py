import numpy as np
import pytest

from actrhythm import (
    active_mask, astp, intradaily_variability, total_rest_time, rest_efficiency
)


def test_non_1d_input_raises():
    arr2d = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="activity must be 1-D"):
        astp(arr2d, 1.0)


def test_all_nan_input_raises():
    with pytest.raises(ValueError, match="activity contains only NaN"):
        intradaily_variability([np.nan, np.nan, np.nan])


def test_nonfinite_threshold_raises():
    with pytest.raises(ValueError, match="sed_threshold must be finite"):
        active_mask([1, 2, 3], sed_threshold=np.inf)


def test_valid_inputs_unchanged():
    # known values from existing tests
    series = [2, 2, 0, 0, 0, 2, 2, 2, 0, 2]
    thr = 1.0
    assert astp(series, thr) == pytest.approx(0.4)
    assert total_rest_time([0, 1, 1, 0, 1], rest_threshold=0.5) == 3
    # rest_efficiency example check
    mask = [False, True, True, True, False, True, True]
    assert rest_efficiency(mask) == pytest.approx(5 / 6)
