import numpy as np
import pytest

from actrhythm import (
    number_of_awakenings,
    rest_efficiency,
    total_rest_time,
    wake_after_onset,
)

# Binary rest mask: True == rest
MASK = [False, True, True, False, True, True, True, False]
# Equivalent numeric activity (lower is rest). Use threshold=2.0 so values <2 are rest
ACT = [10, 1, 1, 10, 0.5, 0.2, 0.3, 10]
THR = 2.0


def test_total_rest_time_from_mask_and_activity():
    assert total_rest_time(MASK) == 5
    assert total_rest_time(ACT, rest_threshold=THR) == 5


def test_rest_efficiency_matches_hand_calc():
    # onset at idx 1, offset at idx 6 => TIB = 6 epochs (1..6 inclusive)
    assert pytest.approx(rest_efficiency(MASK), rel=1e-12) == 5 / 6
    assert pytest.approx(rest_efficiency(ACT, rest_threshold=THR), rel=1e-12) == 5 / 6


def test_waso_and_awakenings():
    # within TIB (indices 1..6) there is one wake epoch at index 3 -> WASO=1, awakenings=1
    assert wake_after_onset(MASK) == 1
    assert wake_after_onset(ACT, rest_threshold=THR) == 1
    assert number_of_awakenings(MASK) == 1
    assert number_of_awakenings(ACT, rest_threshold=THR) == 1


def test_no_rest_returns_nan():
    assert np.isnan(rest_efficiency([0, 0, 0]))
    assert np.isnan(wake_after_onset([0, 0, 0]))
    assert np.isnan(number_of_awakenings([0, 0, 0]))
