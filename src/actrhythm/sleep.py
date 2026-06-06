"""Sleep/rest-period metrics computed from a binary rest/active series or
numeric activity series.

The functions accept either
- a boolean-like sequence where True indicates a rest epoch, or
- a numeric activity sequence together with ``rest_threshold`` such that
  epochs with activity < rest_threshold are considered rest.

All functions are pure and return Python scalars (ints or floats). Missing or
degenerate inputs return ``float('nan')`` when a meaningful numeric value
cannot be produced.

References
----------
Typical definitions follow sleep-scoring conventions (e.g., Ancoli-Israel et
al., Sleep 2003) where:
- total rest time is the number of rest epochs within time-in-bed
- sleep efficiency (here rest_efficiency) = total_rest_time / time_in_bed
- WASO (wake after sleep onset) is number of wake epochs between onset and
  final awakening
- awakenings count the number of discrete wake bouts within the sleep episode
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .validation import validate_1d_array, validate_threshold

__all__ = [
    "total_rest_time",
    "rest_efficiency",
    "wake_after_onset",
    "number_of_awakenings",
]


def _to_rest_mask(data: ArrayLike, rest_threshold: float | None) -> np.ndarray:
    """Convert input to boolean rest mask (True==rest).

    If ``data`` is a boolean or integer 0/1 sequence, it is interpreted as a
    rest mask directly. If ``data`` is numeric and ``rest_threshold`` is
    provided, epochs with value < rest_threshold are considered rest. If
    numeric and ``rest_threshold`` is None, raises ValueError.
    """
    # validate dimensionality and emptiness; NaN-only arrays are invalid
    a = validate_1d_array(data, name="data", allow_all_nan=False)
    if a.dtype == bool or np.issubdtype(a.dtype, np.bool_):
        return a.astype(bool)
    # treat integer 0/1 masks as boolean masks
    if np.issubdtype(a.dtype, np.integer) and set(np.unique(a)).issubset({0, 1}):
        return a.astype(bool)
    # numeric activity series
    if np.issubdtype(a.dtype, np.number):
        if rest_threshold is None:
            raise ValueError("rest_threshold must be provided for numeric activity input")
        # validate threshold
        validate_threshold(rest_threshold, name="rest_threshold")
        # treat NaN as wake (not rest)
        return np.asarray(np.nan_to_num(a, nan=np.inf) < float(rest_threshold))
    raise TypeError("Unsupported data type for sleep metric computation")


def _onset_and_offset(mask: np.ndarray) -> tuple[int, int]:
    """Return (start_idx, end_idx) inclusive of the first and last True.

    If no True values present, raise ValueError.
    """
    true_idx = np.flatnonzero(mask)
    if true_idx.size == 0:
        raise ValueError("no rest epochs found")
    return int(true_idx[0]), int(true_idx[-1])


def total_rest_time(data: ArrayLike, *, rest_threshold: float | None = None) -> int:
    """Total rest time in epochs.

    Parameters
    ----------
    data : array-like
        Boolean rest mask (True==rest) or numeric activity series.
    rest_threshold : float, optional
        Required if passing numeric activity series; epochs with activity <
        rest_threshold are considered rest.

    Returns
    -------
    int
        Count of rest epochs (may be zero).

    Reference
    ---------
    Adapted from common sleep-scoring terminology (Ancoli-Israel et al.).
    """
    mask = _to_rest_mask(data, rest_threshold)
    return int(mask.sum())


def rest_efficiency(data: ArrayLike, *, rest_threshold: float | None = None) -> float:
    """Rest efficiency: total_rest_time / time_in_bed.

    Time-in-bed is taken as the inclusive window from the first rest epoch
    (sleep onset) to the last rest epoch (final awakening). Returns NaN if
    no rest epochs are present.
    """
    mask = _to_rest_mask(data, rest_threshold)
    try:
        start, end = _onset_and_offset(mask)
    except ValueError:
        return float("nan")
    tib = end - start + 1
    if tib <= 0:
        return float("nan")
    return float(mask[start : end + 1].sum() / tib)


def wake_after_onset(data: ArrayLike, *, rest_threshold: float | None = None) -> int | float:
    """WASO: number of wake epochs between sleep onset and final awakening.

    Returns integer count; NaN is not applicable (returns 0 when there are no
    wake epochs inside the TIB). If no rest epochs exist, returns NaN.
    """
    mask = _to_rest_mask(data, rest_threshold)
    try:
        start, end = _onset_and_offset(mask)
    except ValueError:
        return float("nan")
    window = mask[start : end + 1]
    # wake epochs are False in window
    return int((~window).sum())


def number_of_awakenings(data: ArrayLike, *, rest_threshold: float | None = None) -> int | float:
    """Number of discrete awakenings (wake bouts) between onset and offset.

    Awakening is defined as any contiguous run of wake epochs (False) within
    the time-in-bed window. Returns NaN if no rest epochs are found.
    """
    mask = _to_rest_mask(data, rest_threshold)
    try:
        start, end = _onset_and_offset(mask)
    except ValueError:
        return float("nan")
    window = mask[start : end + 1]
    if window.size == 0:
        return 0
    # count contiguous False runs
    arr = (~window).astype(int)
    if arr.sum() == 0:
        return 0
    change = np.diff(arr)
    starts = np.flatnonzero(np.r_[arr[0], change == 1])
    ends = np.flatnonzero(np.r_[change == -1, arr[-1]])
    return int((ends - starts + 1).size)
