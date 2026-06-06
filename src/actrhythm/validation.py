"""Input validation helpers for actrhythm public APIs.

Provide consistent checks for 1-D array inputs, empty/all-NaN series, and
finite thresholds with clear error messages.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class ValidationError(ValueError):
    """Base class for validation-related errors."""


class DimensionalityError(ValidationError):
    pass


class EmptyError(ValidationError):
    pass


class AllNaNError(ValidationError):
    pass


class ThresholdError(ValidationError):
    pass


class EpochsPerHourError(ValidationError):
    pass


def validate_1d_array(
    data: ArrayLike, *, name: str = "array", allow_all_nan: bool = False
) -> np.ndarray[Any, Any]:
    """Validate and return a 1-D numpy array.

    Raises specialized ValidationError subclasses for clear error handling:
    - DimensionalityError: if input is not 1-D
    - EmptyError: if input has zero length
    - AllNaNError: if input contains only NaN (unless allow_all_nan=True)
    """
    a = np.asarray(data)
    if a.ndim != 1:
        raise DimensionalityError(f"{name} must be 1-D")
    if a.size == 0:
        raise EmptyError(f"{name} is empty")
    if not allow_all_nan and np.isnan(a).all():
        raise AllNaNError(f"{name} contains only NaN")
    return a


def validate_threshold(val: float, *, name: str = "threshold") -> float:
    """Ensure a numeric threshold is provided and finite.

    Accepts integer-like floats (e.g., 1.0) and numpy scalar types by coercing
    to Python float. Raises ThresholdError for None, non-numeric, or
    non-finite values.
    """
    if val is None:
        raise ThresholdError(f"{name} must be provided and finite")
    try:
        v = float(val)
    except (TypeError, ValueError) as err:
        raise ThresholdError(f"{name} must be a finite number") from err
    if not np.isfinite(v):
        raise ThresholdError(f"{name} must be finite")
    return v
