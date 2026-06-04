"""Input validation helpers for actrhythm public APIs.

Provide consistent checks for 1-D array inputs, empty/all-NaN series, and
finite thresholds with clear error messages.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def validate_1d_array(data: ArrayLike, *, name: str = "array", allow_all_nan: bool = False) -> np.ndarray:
    """Validate and return a 1-D numpy array.

    Raises ValueError with a clear message for wrong dimensionality, empty
    arrays, or (unless allow_all_nan) arrays that are entirely NaN.
    """
    a = np.asarray(data)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not allow_all_nan and np.isnan(a).all():
        raise ValueError(f"{name} contains only NaN")
    return a


def validate_threshold(val, *, name: str = "threshold") -> float:
    """Ensure a numeric threshold is provided and finite.

    Raises ValueError for None or non-finite values.
    """
    if val is None:
        raise ValueError(f"{name} must be provided and finite")
    try:
        v = float(val)
    except Exception:
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v
