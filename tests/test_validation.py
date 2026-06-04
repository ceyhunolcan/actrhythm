import numpy as np
import pytest

from actrhythm import (
    active_mask, astp, intradaily_variability, total_rest_time, rest_efficiency
)


def test_non_1d_input_raises():
    arr2d = np.array([[1, 2], [3, 4]])
    with pytest.raises(Exception) as exc:
        astp(arr2d, 1.0)
    # accept either DimensionalityError or ValidationError subclasses
    assert "activity must be 1-D" in str(exc.value)


def test_all_nan_input_raises():
    with pytest.raises(Exception) as exc:
        intradaily_variability([np.nan, np.nan, np.nan])
    assert "activity contains only NaN" in str(exc.value)


def test_nonfinite_threshold_raises():
    with pytest.raises(Exception) as exc:
        active_mask([1, 2, 3], sed_threshold=np.inf)
    assert "sed_threshold must be finite" in str(exc.value)


def test_valid_inputs_unchanged():
    # known values from existing tests
    series = [2, 2, 0, 0, 0, 2, 2, 2, 0, 2]
    thr = 1.0
    assert astp(series, thr) == pytest.approx(0.4)
    assert total_rest_time([0, 1, 1, 0, 1], rest_threshold=0.5) == 3
    # rest_efficiency example check
    mask = [False, True, True, True, False, True, True]
    assert rest_efficiency(mask) == pytest.approx(5 / 6)
