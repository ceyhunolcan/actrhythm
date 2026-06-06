"""Tests for cosinor analysis, pinned to synthetic signals with known params."""
import numpy as np
import pytest

from actrhythm.cosinor import CosinorResult, cosinor


def _make_signal(mesor, amplitude, acrophase, period, n, noise=0.0, seed=0):
    """Generate y(t) = mesor + amplitude*cos(2*pi*t/period + acrophase)."""
    t = np.arange(n, dtype=float)
    y = mesor + amplitude * np.cos(2 * np.pi * t / period + acrophase)
    if noise:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0, noise, size=n)
    return t, y


def test_recovers_known_parameters_no_noise():
    M, A, phi, period = 5.0, 2.0, 0.7, 24.0
    t, y = _make_signal(M, A, phi, period, n=240)
    res = cosinor(y, period=period)
    assert isinstance(res, CosinorResult)
    assert res.mesor == pytest.approx(M, abs=1e-6)
    assert res.amplitude == pytest.approx(A, abs=1e-6)
    assert np.cos(res.acrophase) == pytest.approx(np.cos(phi), abs=1e-6)
    assert np.sin(res.acrophase) == pytest.approx(np.sin(phi), abs=1e-6)


def test_r_squared_is_one_for_perfect_cosine():
    t, y = _make_signal(3.0, 1.5, 0.0, 24.0, n=120)
    res = cosinor(y, period=24.0)
    assert res.r_squared == pytest.approx(1.0, abs=1e-9)


def test_acrophase_time_of_peak():
    t, y = _make_signal(0.0, 1.0, 0.0, 24.0, n=240)
    res = cosinor(y, period=24.0)
    assert min(res.acrophase_time, 24.0 - res.acrophase_time) == pytest.approx(0.0, abs=1e-6)


def test_acrophase_time_for_quarter_phase():
    period = 24.0
    t, y = _make_signal(0.0, 1.0, -np.pi / 2, period, n=240)
    res = cosinor(y, period=period)
    assert res.acrophase_time == pytest.approx(6.0, abs=1e-6)


def test_mesor_equals_mean_for_full_periods():
    M = 7.0
    t, y = _make_signal(M, 3.0, 1.1, 24.0, n=24 * 5)
    res = cosinor(y, period=24.0)
    assert res.mesor == pytest.approx(y.mean(), abs=1e-9)
    assert res.mesor == pytest.approx(M, abs=1e-9)


def test_custom_times_argument():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 48, size=200))
    M, A, phi, period = 2.0, 1.0, 0.3, 24.0
    y = M + A * np.cos(2 * np.pi * t / period + phi)
    res = cosinor(y, period=period, times=t)
    assert res.mesor == pytest.approx(M, abs=1e-6)
    assert res.amplitude == pytest.approx(A, abs=1e-6)


def test_recovers_params_with_noise():
    M, A, phi, period = 4.0, 2.0, 0.5, 24.0
    t, y = _make_signal(M, A, phi, period, n=24 * 20, noise=0.1, seed=3)
    res = cosinor(y, period=period)
    assert res.mesor == pytest.approx(M, abs=0.05)
    assert res.amplitude == pytest.approx(A, abs=0.05)
    assert res.r_squared > 0.98


def test_amplitude_is_nonnegative():
    t, y = _make_signal(0.0, 2.0, 2.5, 24.0, n=120)
    res = cosinor(y, period=24.0)
    assert res.amplitude >= 0.0


def test_flat_series_zero_amplitude_nan_r2():
    y = np.full(100, 5.0)
    res = cosinor(y, period=24.0)
    assert res.amplitude == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(res.r_squared)


def test_empty_raises():
    with pytest.raises(ValueError):
        cosinor([], period=24.0)


def test_bad_period_raises():
    with pytest.raises(ValueError):
        cosinor([1, 2, 3], period=0.0)


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        cosinor([1.0, 2.0], period=24.0)


def test_mismatched_times_raises():
    with pytest.raises(ValueError):
        cosinor([1, 2, 3], period=24.0, times=[0, 1])
