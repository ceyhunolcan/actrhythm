"""Tests for circadian rhythm metrics, pinned to analytic / known values."""
import numpy as np
import pytest

from actrhythm import (
    interdaily_stability,
    intradaily_variability,
    l5,
    m10,
    relative_amplitude,
)


def test_iv_constant_series_is_nan():
    assert np.isnan(intradaily_variability([3, 3, 3, 3]))


def test_iv_smooth_vs_fragmented_ordering():
    smooth = np.linspace(0, 1, 48)
    fragmented = np.tile([0, 1], 24)
    assert intradaily_variability(fragmented) > intradaily_variability(smooth)


def test_iv_known_value_alternating():
    x = np.tile([0, 1], 50)
    assert intradaily_variability(x) == pytest.approx(4.0)


def test_is_perfectly_repeating_days_is_one():
    one_day = np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False)) + 2
    series = np.tile(one_day, 5)
    assert interdaily_stability(series, epochs_per_hour=1) == pytest.approx(1.0)


def test_is_needs_a_full_day():
    assert np.isnan(interdaily_stability([1, 2, 3], epochs_per_hour=1))


def test_is_constant_series_is_nan():
    assert np.isnan(interdaily_stability([5] * 24, epochs_per_hour=1))


def test_l5_m10_on_known_profile():
    profile = np.ones(24)
    profile[8:18] = 10.0
    assert m10(profile, epochs_per_hour=1) == pytest.approx(10.0)
    assert l5(profile, epochs_per_hour=1) == pytest.approx(1.0)


def test_relative_amplitude_known():
    profile = np.ones(24)
    profile[8:18] = 10.0
    assert relative_amplitude(profile, epochs_per_hour=1) == pytest.approx(9 / 11)


def test_relative_amplitude_flat_is_zero():
    assert relative_amplitude(np.full(24, 4.0), epochs_per_hour=1) == pytest.approx(0.0)


def test_l5_m10_window_wraps_midnight():
    profile = np.ones(24)
    active_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
    profile[active_hours] = 10.0
    assert m10(profile, epochs_per_hour=1) == pytest.approx(10.0)


def test_epochs_per_hour_aggregation():
    profile_hourly = np.ones(24)
    profile_hourly[8:18] = 10.0
    series = np.repeat(profile_hourly, 2)
    assert m10(series, epochs_per_hour=2) == pytest.approx(10.0)
    assert l5(series, epochs_per_hour=2) == pytest.approx(1.0)


def test_invalid_epochs_per_hour_raises():
    with pytest.raises(Exception) as exc:
        interdaily_stability([1, 2], epochs_per_hour=0)
    assert "epochs_per_hour" in str(exc.value)
