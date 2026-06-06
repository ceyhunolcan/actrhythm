"""Tests for gap-aware (segmented) fragmentation, pinned to hand values."""
import numpy as np
import pytest

from actrhythm import astp
from actrhythm.segmented import (
    astp_segmented,
    bout_lengths_segmented,
    satp_segmented,
    segment_indices,
    split_on_gaps,
)

THR = 1.0


def test_segment_indices_basic():
    assert segment_indices([0, 1, 2, 10, 11], step=1) == [(0, 3), (3, 5)]


def test_segment_indices_no_gap():
    assert segment_indices([0, 1, 2, 3], step=1) == [(0, 4)]


def test_segment_indices_empty_and_single():
    assert segment_indices([], step=1) == []
    assert segment_indices([5], step=1) == [(0, 1)]


def test_split_on_gaps():
    act = [2, 2, 0, 9, 9]
    idx = [0, 1, 2, 10, 11]
    segs = split_on_gaps(act, idx, step=1)
    assert len(segs) == 2
    assert segs[0].tolist() == [2.0, 2.0, 0.0]
    assert segs[1].tolist() == [9.0, 9.0]


def test_astp_segmented_pools_across_segments():
    segs = [[2, 2, 0], [2, 0, 2]]
    assert astp_segmented(segs, sed_threshold=THR) == pytest.approx(2 / 3)


def test_satp_segmented_pools_across_segments():
    segs = [[2, 2, 0], [2, 0, 2]]
    assert satp_segmented(segs, sed_threshold=THR) == pytest.approx(1.0)


def test_segmented_differs_from_naive_across_gap():
    flat = [2, 2, 0, 0]
    idx = [0, 1, 100, 101]
    assert astp(flat, THR) == pytest.approx(0.5)
    seg = astp_segmented(activity=flat, sample_index=idx, sed_threshold=THR, step=1)
    assert seg == pytest.approx(0.0)


def test_flat_input_via_sample_index_no_gap_matches_naive():
    series = [2, 2, 0, 0, 0, 2, 2, 2, 0, 2]
    idx = list(range(len(series)))
    naive = astp(series, THR)
    seg = astp_segmented(activity=series, sample_index=idx, sed_threshold=THR, step=1)
    assert seg == pytest.approx(naive)


def test_bout_lengths_segmented_no_cross_gap_bout():
    segs = [[2, 2, 0], [2, 2, 2]]
    bl = bout_lengths_segmented(segs, sed_threshold=THR, active=True)
    assert sorted(bl.tolist()) == [2, 3]


def test_step_larger_than_one():
    idx = [0, 3, 6, 100]
    act = [2, 0, 2, 9]
    segs = split_on_gaps(act, idx, step=5)
    assert len(segs) == 2
    assert segs[0].tolist() == [2.0, 0.0, 2.0]
    assert segs[1].tolist() == [9.0]


def test_error_both_input_styles():
    with pytest.raises(ValueError):
        astp_segmented([[1, 2]], sed_threshold=THR, activity=[1, 2], sample_index=[0, 1])


def test_error_no_input():
    with pytest.raises(ValueError):
        astp_segmented(sed_threshold=THR)


def test_error_mismatched_lengths():
    with pytest.raises(ValueError):
        split_on_gaps([1, 2, 3], [0, 1], step=1)


def test_all_short_segments_give_nan():
    assert np.isnan(astp_segmented([[5], [3]], sed_threshold=THR))
