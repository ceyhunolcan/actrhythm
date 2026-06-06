"""Gap-aware (segmented) fragmentation metrics.

Real accelerometer recordings are not continuous: sleep periods, non-wear, and
device gaps interrupt the wake activity stream. Computing fragmentation across
such a gap is wrong — it glues the end of one wake period to the start of the
next and invents a transition that never happened.

These functions compute fragmentation *within* contiguous segments and then
pool the counts across segments. ASTP is therefore

    ASTP = (total active->sedentary transitions across all segments)
           / (total active epochs that have a same-segment successor)

which is the pooled-count definition used in epidemiologic accelerometry
pipelines (e.g. NHANES minute-level fragmentation), not an average of
per-segment rates.

Segments can be supplied two ways:
  * `segments`     : a sequence of 1-D activity arrays (already split), or
  * `activity` + `sample_index` + `step` : a flat series plus the integer
    sample index of each epoch; a new segment begins wherever consecutive
    indices differ by more than `step` (i.e. a gap).

References
----------
Di J, et al. (2017); Wanigatunga AA, et al. (2019) for the underlying ASTP/SATP
definitions; segmentation reflects standard wake-bout handling in NHANES-style
minute-level pipelines.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .fragmentation import active_mask, bout_lengths

__all__ = [
    "segment_indices",
    "split_on_gaps",
    "astp_segmented",
    "satp_segmented",
    "bout_lengths_segmented",
]


def segment_indices(sample_index: ArrayLike, *, step: int = 1) -> list[tuple[int, int]]:
    """Return (start, stop) half-open index ranges of contiguous segments.

    A new segment begins wherever consecutive entries of ``sample_index`` differ
    by more than ``step``. ``sample_index`` must be sorted ascending.
    """
    idx = np.asarray(sample_index).ravel()
    if idx.size == 0:
        return []
    if idx.size == 1:
        return [(0, 1)]
    gaps = np.diff(idx)
    breaks = np.flatnonzero(gaps > step) + 1
    bounds = np.concatenate([[0], breaks, [idx.size]])
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.size - 1)]


def split_on_gaps(
    activity: ArrayLike, sample_index: ArrayLike, *, step: int = 1
) -> list[np.ndarray[Any, Any]]:
    """Split a flat activity series into contiguous segments at gaps."""
    a = np.asarray(activity, dtype=float).ravel()
    idx = np.asarray(sample_index).ravel()
    if a.shape != idx.shape:
        raise ValueError("activity and sample_index must have the same length")
    return [a[lo:hi] for lo, hi in segment_indices(idx, step=step)]


def _as_segments(
    segments: Sequence[ArrayLike] | None,
    activity: ArrayLike | None,
    sample_index: ArrayLike | None,
    step: int,
) -> list[np.ndarray[Any, Any]]:
    """Resolve the two input styles into a list of 1-D float arrays."""
    if segments is not None:
        if activity is not None or sample_index is not None:
            raise ValueError("pass either `segments` or `activity`+`sample_index`, not both")
        out = [np.asarray(s, dtype=float).ravel() for s in segments]
        if not out:
            raise ValueError("no segments provided")
        return out
    if activity is None or sample_index is None:
        raise ValueError("provide `segments`, or both `activity` and `sample_index`")
    return split_on_gaps(activity, sample_index, step=step)


def _pooled_transition_prob(
    segs: list[np.ndarray[Any, Any]], sed_threshold: float, *, from_state: bool
) -> float:
    """Pooled transition probability across segments. NaN if no origin epochs."""
    flips = 0
    origin = 0
    for seg in segs:
        if seg.size < 2:
            continue
        mask = active_mask(seg, sed_threshold)
        cur = mask[:-1] == from_state
        origin += int(cur.sum())
        flips += int(np.sum(cur & (mask[1:] != from_state)))
    if origin == 0:
        return float("nan")
    return flips / origin


def astp_segmented(
    segments: Sequence[ArrayLike] | None = None,
    *,
    sed_threshold: float,
    activity: ArrayLike | None = None,
    sample_index: ArrayLike | None = None,
    step: int = 1,
) -> float:
    """Gap-aware Active-to-Sedentary Transition Probability (pooled)."""
    segs = _as_segments(segments, activity, sample_index, step)
    return _pooled_transition_prob(segs, sed_threshold, from_state=True)


def satp_segmented(
    segments: Sequence[ArrayLike] | None = None,
    *,
    sed_threshold: float,
    activity: ArrayLike | None = None,
    sample_index: ArrayLike | None = None,
    step: int = 1,
) -> float:
    """Gap-aware Sedentary-to-Active Transition Probability (pooled)."""
    segs = _as_segments(segments, activity, sample_index, step)
    return _pooled_transition_prob(segs, sed_threshold, from_state=False)


def bout_lengths_segmented(
    segments: Sequence[ArrayLike] | None = None,
    *,
    sed_threshold: float,
    active: bool = True,
    activity: ArrayLike | None = None,
    sample_index: ArrayLike | None = None,
    step: int = 1,
) -> np.ndarray[Any, Any]:
    """Concatenated active (or sedentary) bout lengths computed per segment."""
    segs = _as_segments(segments, activity, sample_index, step)
    parts = [bout_lengths(seg, sed_threshold, active=active) for seg in segs if seg.size]
    parts = [p for p in parts if p.size]
    if not parts:
        return np.array([], dtype=int)
    return np.concatenate(parts).astype(int)
