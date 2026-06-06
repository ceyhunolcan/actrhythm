"""Single-component cosinor analysis for circadian activity data.

The cosinor model fits a cosine of fixed period to a time series:

    Y(t) = M + A * cos(2*pi*t / period + phi) + e(t)

where the three rhythm parameters are

  MESOR     M    — the rhythm-adjusted mean (Midline Estimating Statistic Of
                   Rhythm); the mean level about which the cosine oscillates
  amplitude A     — half the peak-to-trough difference of the fitted cosine
  acrophase phi   — the phase of the peak, in radians (and as a clock time)

Fitting is done by ordinary least squares after the linearizing substitution
    A*cos(w t + phi) = beta*cos(w t) + gamma*sin(w t),   w = 2*pi/period
which is linear in (M, beta, gamma). The rhythm parameters are then recovered as
    A   = sqrt(beta**2 + gamma**2)
    phi = atan2(-gamma, beta)

This is the classic single-component cosinor of Halberg and colleagues, widely
used to summarize the 24-hour activity rhythm.

References
----------
Cornelissen G. "Cosinor-based rhythmometry." Theor Biol Med Model (2014) 11:16
  — a modern exposition of the method introduced by Halberg et al. (1960s).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["CosinorResult", "cosinor"]

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class CosinorResult:
    """Result of a single-component cosinor fit."""

    mesor: float
    amplitude: float
    acrophase: float
    acrophase_time: float
    period: float
    r_squared: float


def cosinor(
    activity: ArrayLike,
    *,
    period: float,
    times: ArrayLike | None = None,
) -> CosinorResult:
    """Fit a single-component cosinor model to an activity series.

    Parameters
    ----------
    activity : array-like
        1-D series of activity values.
    period : float
        Period of the cosine to fit, in the same time units as `times`
        (e.g. 24 for a circadian fit when `times` are hours). Must be > 0.
    times : array-like, optional
        Sample times, same length as `activity`. If omitted, samples are
        assumed equally spaced at unit intervals (t = 0, 1, 2, ...), so
        `period` must then be expressed in number-of-samples.

    Returns
    -------
    CosinorResult

    Notes
    -----
    Ordinary least squares on the design matrix ``[1, cos(w t), sin(w t)]``
    with ``w = 2*pi/period``. At least 3 samples are required.
    """
    y = np.asarray(activity, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("activity series is empty")
    if period <= 0:
        raise ValueError("period must be > 0")
    if times is None:
        t = np.arange(y.size, dtype=float)
    else:
        t = np.asarray(times, dtype=float).ravel()
        if t.shape != y.shape:
            raise ValueError("times and activity must have the same length")
    if y.size < 3:
        raise ValueError("cosinor fit requires at least 3 samples")

    w = _TWO_PI / period
    design = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    mesor, beta, gamma = (float(coef[0]), float(coef[1]), float(coef[2]))

    amplitude = float(np.hypot(beta, gamma))
    acrophase = float(np.arctan2(-gamma, beta))
    acrophase_time = float((-acrophase / w) % period)

    fitted = design @ coef
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return CosinorResult(
        mesor=mesor,
        amplitude=amplitude,
        acrophase=acrophase,
        acrophase_time=acrophase_time,
        period=float(period),
        r_squared=r_squared,
    )
