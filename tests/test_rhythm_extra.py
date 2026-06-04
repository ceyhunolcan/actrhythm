import numpy as np
import pytest

from actrhythm import m10, intradaily_variability, interdaily_stability


def test_m10_raises_on_insufficient_data():
    # fewer than one full 24-hour day should raise via _hourly_profile
    with pytest.raises(ValueError):
        m10([1, 2, 3], epochs_per_hour=1)


def test_iv_accepts_list_and_numpy():
    data_list = list(np.tile([0, 1], 50))
    data_arr = np.array(data_list)
    assert intradaily_variability(data_list) == pytest.approx(intradaily_variability(data_arr))


def test_interdaily_stability_rejects_non_integer_epochs_per_hour():
    # non-integer epochs_per_hour should raise (slicing requires integer bins)
    day = np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False)) + 2
    series = np.tile(day, 3)
    with pytest.raises(Exception) as exc:
        interdaily_stability(series, epochs_per_hour=1.5)
    assert "epochs_per_hour" in str(exc.value)
