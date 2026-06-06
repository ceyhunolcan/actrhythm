"""Tests for fragmentation metrics, pinned to hand-computable values."""
import numpy as np
import pytest

from actrhythm import (
    active_mask,
    astp,
    bout_lengths,
    fragmentation_index,
    mean_bout_length,
    satp,
)

SERIES = [2, 2, 0, 0, 0, 2, 2, 2, 0, 2]
THR = 1.0


def test_active_mask_threshold_is_inclusive():
    assert active_mask([1.0, 0.999], 1.0).tolist() == [True, False]


def test_active_mask_nan_is_inactive():
    assert active_mask([np.nan, 5.0], 1.0).tolist() == [False, True]


def test_astp_hand_computed():
    assert astp(SERIES, THR) == pytest.approx(0.4)


def test_satp_hand_computed():
    assert satp(SERIES, THR) == pytest.approx(0.5)


def test_astp_alternating_series_is_one():
    alt = [2, 0, 2, 0, 2, 0]
    assert astp(alt, THR) == pytest.approx(1.0)


def test_bout_lengths_active():
    assert bout_lengths(SERIES, THR, active=True).tolist() == [2, 3, 1]


def test_bout_lengths_sedentary():
    assert bout_lengths(SERIES, THR, active=False).tolist() == [3, 1]


def test_mean_bout_length():
    assert mean_bout_length(SERIES, THR, active=True) == pytest.approx((2 + 3 + 1) / 3)


def test_fragmentation_index_hand_computed():
    assert fragmentation_index(SERIES, THR) == pytest.approx(4 / 9)


def test_all_active_series_has_no_transitions():
    assert fragmentation_index([5, 5, 5, 5], THR) == 0.0
    assert astp([5, 5, 5], THR) == pytest.approx(0.0)


def test_empty_series_raises():
    with pytest.raises(Exception) as exc:
        astp([], THR)
    assert "activity is empty" in str(exc.value)


def test_single_epoch_transition_is_nan():
    assert np.isnan(astp([5], THR))
