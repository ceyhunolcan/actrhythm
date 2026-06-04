"""actrhythm — rest-activity rhythm and fragmentation metrics for accelerometry.

A small, well-tested library implementing the standard non-parametric circadian
metrics (IS, IV, RA, L5/M10) and activity-fragmentation metrics (ASTP, SATP,
bout statistics) used in actigraphy and digital-phenotyping research.

Functions take a plain 1-D activity series (minute-level MIMS, ENMO, counts,
etc.) — no device- or study-specific assumptions.
"""
from .fragmentation import (
    active_mask, astp, satp, bout_lengths, mean_bout_length,
    fragmentation_index,
)
from .rhythm import (
    interdaily_stability, intradaily_variability, l5, m10,
    relative_amplitude,
)
from .sleep import (
    total_rest_time, rest_efficiency, wake_after_onset, number_of_awakenings,
)

__version__ = "0.1.0"
__all__ = [
    "astp", "satp", "bout_lengths", "mean_bout_length",
    "fragmentation_index", "active_mask",
    "interdaily_stability", "intradaily_variability",
    "l5", "m10", "relative_amplitude",
    "total_rest_time", "rest_efficiency", "wake_after_onset", "number_of_awakenings",
]
