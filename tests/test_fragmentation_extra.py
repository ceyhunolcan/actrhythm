import numpy as np

from actrhythm import (
    active_mask,
    bout_lengths,
    fragmentation_index,
    mean_bout_length,
)


def test_bout_lengths_returns_empty_for_no_active_bouts():
    # threshold 1 -> series all below threshold -> no active bouts
    assert bout_lengths([0, 0, 0], 1.0, active=True).tolist() == []


def test_mean_bout_length_nan_when_no_bouts():
    assert np.isnan(mean_bout_length([0, 0, 0], 1.0, active=True))
    assert np.isnan(mean_bout_length([5, 5, 5], 1.0, active=False))


def test_active_mask_handles_infinite_values():
    mask = active_mask([np.inf, -np.inf, 1.0], 1.0)
    assert mask.tolist() == [True, False, True]


def test_fragmentation_index_single_epoch_is_nan():
    assert np.isnan(fragmentation_index([5], 1.0))
