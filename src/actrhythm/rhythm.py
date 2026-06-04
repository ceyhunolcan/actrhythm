"""Non-parametric circadian rhythm metrics for activity data.

These four metrics are the standard non-parametric description of a rest-
activity rhythm, introduced for actigraphy by Witting et al. (1990) and
formalized by Van Someren et al. (1999). They require an evenly-sampled
activity series and the number of epochs per hour.

Functions expect a 1-D sequence (list/tuple/np.ndarray) of per-epoch activity
values and an integer epochs_per_hour describing how many samples represent
one clock hour (e.g., 60 for minute-level data). Inputs may contain NaN to
represent missing epochs; NaNs are treated as "inactive" in fragmentation
utilities and propagate through mean-based rhythm metrics.

Available metrics
-----------------
  IS  Interdaily Stability   — how reproducible the 24-h pattern is across days
  IV  Intradaily Variability — fragmentation of the rhythm within a day
  RA  Relative Amplitude     — contrast between most-active 10 h and least 5 h
  L5 / M10                   — mean activity in those least/most active windows

Notes
-----
- The numerical formulas are preserved and validated against published
  reference values; do not change without re-validation.
- Many functions return ``float('nan')`` for degenerate inputs (e.g., zero
  variance or too-short series).

Examples
--------
>>> from actrhythm import m10, l5, relative_amplitude
>>> profile = [1]*24
>>> profile[8:18] = [10]*10
>>> m10(profile, epochs_per_hour=1)
10.0
>>> l5(profile, epochs_per_hour=1)
1.0
>>> relative_amplitude(profile, epochs_per_hour=1)
0.8181818181818182

References
----------
Witting W, et al. "Alterations in the circadian rest-activity rhythm in aging
  and Alzheimer's disease." Biol Psychiatry (1990).
Van Someren EJW, et al. "Bright light therapy ... by application of
  nonparametric methods." Chronobiol Int (1999).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["interdaily_stability", "intradaily_variability",
           "l5", "m10", "relative_amplitude"]


def _validate_epochs_per_hour(epochs_per_hour: int) -> int:
    """Validate epochs_per_hour is a positive integer and return int.

    Raises TypeError for non-integer values and ValueError for values < 1.
    """
    # Allow numpy integer types as well
    if not isinstance(epochs_per_hour, (int, np.integer)):
        raise TypeError("epochs_per_hour must be an integer")
    epochs_per_hour = int(epochs_per_hour)
    if epochs_per_hour < 1:
        raise ValueError("epochs_per_hour must be >= 1")
    return epochs_per_hour


def _clean(activity: ArrayLike) -> np.ndarray:
    a = np.asarray(activity, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("activity series is empty")
    return a


def intradaily_variability(activity: ArrayLike) -> float:
    """IV — ratio of mean squared first difference to overall variance.

    IV = [ n * sum((x_i - x_{i-1})^2) ] / [ (n-1) * sum((x_i - mean)^2) ]

    Larger IV means the series flips between high and low more often relative
    to its variance. Defined for n >= 2; NaN if the series has zero variance.
    """
    a = _clean(activity)
    n = a.size
    if n < 2:
        return float("nan")
    var = np.sum((a - a.mean()) ** 2)
    if var == 0:
        return float("nan")
    diff = np.sum(np.diff(a) ** 2)
    return float((n * diff) / ((n - 1) * var))


def interdaily_stability(activity: ArrayLike, epochs_per_hour: int) -> float:
    """IS — strength of coupling of the rhythm to the 24-hour day.

    The series is folded into a 24-h profile by averaging over days at each
    position; IS is the variance of that profile relative to the overall
    variance, scaled by bins. Requires at least one full day; NaN if variance
    is zero.
    """
    a = _clean(activity)
    epochs_per_hour = _validate_epochs_per_hour(epochs_per_hour)
    bins_per_day = 24 * epochs_per_hour
    usable = (a.size // bins_per_day) * bins_per_day
    if usable < bins_per_day:
        return float("nan")
    a = a[:usable]
    n = a.size
    var = np.sum((a - a.mean()) ** 2)
    if var == 0:
        return float("nan")
    profile = a.reshape(-1, bins_per_day).mean(axis=0)
    p = bins_per_day
    between = n * np.sum((profile - a.mean()) ** 2)
    return float(between / (p * var))


def _hourly_profile(activity: ArrayLike, epochs_per_hour: int) -> np.ndarray:
    """Mean activity per clock hour, folded across all days (length 24)."""
    a = _clean(activity)
    epochs_per_hour = _validate_epochs_per_hour(epochs_per_hour)
    bins_per_day = 24 * epochs_per_hour
    usable = (a.size // bins_per_day) * bins_per_day
    if usable < bins_per_day:
        raise ValueError("need at least one full 24-hour day of data")
    hourly = a[:usable].reshape(-1, epochs_per_hour).mean(axis=1)
    return hourly.reshape(-1, 24).mean(axis=0)


def _extreme_window(profile24: np.ndarray, hours: int, *, most_active: bool) -> float:
    """Mean of the most/least active `hours`-long consecutive (wrapping) window."""
    if profile24.size != 24:
        raise ValueError("expected a 24-hour profile")
    doubled = np.concatenate([profile24, profile24[: hours - 1]])
    window_means = np.convolve(doubled, np.ones(hours) / hours, mode="valid")[:24]
    return float(window_means.max() if most_active else window_means.min())


def m10(activity: ArrayLike, epochs_per_hour: int) -> float:
    """M10 — mean activity of the most active 10 consecutive hours."""
    return _extreme_window(_hourly_profile(activity, epochs_per_hour), 10, most_active=True)


def l5(activity: ArrayLike, epochs_per_hour: int) -> float:
    """L5 — mean activity of the least active 5 consecutive hours."""
    return _extreme_window(_hourly_profile(activity, epochs_per_hour), 5, most_active=False)


def relative_amplitude(activity: ArrayLike, epochs_per_hour: int) -> float:
    """RA = (M10 - L5) / (M10 + L5), in [0, 1]. NaN if M10 + L5 == 0."""
    hi = m10(activity, epochs_per_hour)
    lo = l5(activity, epochs_per_hour)
    denom = hi + lo
    if denom == 0:
        return float("nan")
    return float((hi - lo) / denom)
