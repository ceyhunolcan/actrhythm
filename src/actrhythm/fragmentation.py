"""Activity fragmentation metrics for minute-level accelerometer data.

Fragmentation quantifies how activity and rest are distributed in time — two
people with identical total activity can differ greatly in how fragmented that
activity is, which carries independent health signal (Di et al. 2017;
Wanigatunga et al. 2019).

Functions operate on a 1-D sequence of per-epoch activity values (list/ndarray)
and a sedentary threshold (float). Inputs may contain ``numpy.nan``; by
convention NaNs are treated as inactive (sedentary) for fragmentation
calculations. Functions return Python floats or numpy arrays and are pure
(predictable, no I/O).

Behavior & return values
------------------------
- Empty input raises ``ValueError``.
- When no bouts of the requested type exist, ``bout_lengths`` returns an empty
  integer array and ``mean_bout_length`` returns ``float('nan')``.
- Transition probabilities and fragmentation index return ``float('nan')``
  for series shorter than 2 epochs.

Examples
--------
>>> from actrhythm import active_mask, astp, mean_bout_length
>>> series = [0, 2, 2, 0, 3]
>>> active_mask(series, 1.0)
array([False, True, True, False, True])
>>> astp(series, 1.0)
0.5
>>> mean_bout_length(series, 1.0, active=True)
2.0

References
----------
Di J, et al. "Patterns of sedentary and active time accumulation are
  associated with mortality in US adults." (2017), bioRxiv 182337.
Wanigatunga AA, et al. "Association of total daily physical activity and
  fragmentation with mortality." J Gerontol A (2019).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "active_mask", "astp", "satp", "bout_lengths", "mean_bout_length",
    "fragmentation_index",
]


from .validation import validate_1d_array, validate_threshold


def _as_array(activity: ArrayLike) -> np.ndarray[Any, Any]:
    # validate dimensionality and emptiness; allow some NaNs but not all
    a = validate_1d_array(activity, name="activity", allow_all_nan=False).astype(float).ravel()
    return a


def active_mask(activity: ArrayLike, sed_threshold: float) -> np.ndarray[Any, Any]:
    """Boolean mask of active epochs (activity >= sed_threshold).

    NaN epochs are treated as inactive (False). Epochs at exactly the
    threshold are active, matching the common ``>=`` cut-point convention.
    """
    # validate threshold is finite
    validate_threshold(sed_threshold, name="sed_threshold")
    a = _as_array(activity)
    return np.asarray(np.nan_to_num(a, nan=-np.inf) >= sed_threshold)


def _transition_probability(mask: np.ndarray[Any, Any], from_state: bool) -> float:
    """P(state flips next epoch | currently in `from_state`)."""
    if mask.size < 2:
        return float("nan")
    cur = mask[:-1] == from_state
    n = int(cur.sum())
    if n == 0:
        return float("nan")
    flips = int(np.sum(cur & (mask[1:] != from_state)))
    return flips / n


def astp(activity: ArrayLike, sed_threshold: float) -> float:
    """Active-to-Sedentary Transition Probability.

    The probability that an active epoch is followed by a sedentary one.
    Higher ASTP = more fragmented activity (shorter active bouts). It is the
    reciprocal of mean active-bout length. See Di et al. (2017).
    """
    return _transition_probability(active_mask(activity, sed_threshold), from_state=True)


def satp(activity: ArrayLike, sed_threshold: float) -> float:
    """Sedentary-to-Active Transition Probability.

    The probability that a sedentary epoch is followed by an active one.
    Higher SATP = more fragmented rest (shorter sedentary bouts).
    """
    return _transition_probability(active_mask(activity, sed_threshold), from_state=False)


def bout_lengths(
    activity: ArrayLike, sed_threshold: float, *, active: bool = True
) -> np.ndarray[Any, Any]:
    """Lengths (in epochs) of consecutive active (or sedentary) bouts."""
    mask = active_mask(activity, sed_threshold)
    target = mask if active else ~mask
    if not target.any():
        return np.array([], dtype=int)
    change = np.diff(target.astype(int))
    starts = np.flatnonzero(np.r_[target[0], change == 1])
    ends = np.flatnonzero(np.r_[change == -1, target[-1]])
    return (ends - starts + 1).astype(int)


def mean_bout_length(activity: ArrayLike, sed_threshold: float, *, active: bool = True) -> float:
    """Mean length of active (or sedentary) bouts, in epochs. NaN if none."""
    lengths = bout_lengths(activity, sed_threshold, active=active)
    return float(lengths.mean()) if lengths.size else float("nan")


def fragmentation_index(activity: ArrayLike, sed_threshold: float) -> float:
    """Overall fragmentation: fraction of adjacent epoch-pairs that change state."""
    mask = active_mask(activity, sed_threshold)
    if mask.size < 2:
        return float("nan")
    return float(np.sum(mask[:-1] != mask[1:]) / (mask.size - 1))
