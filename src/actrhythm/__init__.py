"""actrhythm — rest-activity rhythm and fragmentation metrics for accelerometry.

A small, well-tested library implementing the standard non-parametric circadian
metrics (IS, IV, RA, L5/M10) and activity-fragmentation metrics (ASTP, SATP,
bout statistics) used in actigraphy and digital-phenotyping research.

Functions take a plain 1-D activity series (minute-level MIMS, ENMO, counts,
etc.) — no device- or study-specific assumptions.

Module-level usage notes
------------------------
- Input: pass a 1-D sequence (list/tuple/numpy array). Many functions accept
  either a boolean rest mask (True==rest) or a numeric activity series with
  an explicit threshold where required (e.g., sleep functions).
- Missing data: NaN values are treated consistently (typically treated as
  inactive/rest for fragmentation and as numeric NaN for mean-based rhythm
  metrics); consult individual function docstrings for details.
- Interpretation: higher ASTP/SATP indicates shorter bouts and more
  fragmentation; RA (relative amplitude) near 1 indicates a strong day/night
  contrast.

Examples
--------
>>> from actrhythm import astp, m10, total_rest_time
>>> series = [0, 2, 2, 0, 1, 0, 2]
>>> astp(series, sed_threshold=1.0)
0.5
>>> m10(series, epochs_per_hour=1)
...  # see examples/ for runnable scripts

"""
from .cosinor import CosinorResult, cosinor
from .fragmentation import (
    active_mask,
    astp,
    bout_lengths,
    fragmentation_index,
    mean_bout_length,
    satp,
)
from .rhythm import (
    interdaily_stability,
    intradaily_variability,
    l5,
    m10,
    relative_amplitude,
)
from .segmented import (
    astp_segmented,
    bout_lengths_segmented,
    satp_segmented,
    segment_indices,
    split_on_gaps,
)
from .sleep import (
    number_of_awakenings,
    rest_efficiency,
    total_rest_time,
    wake_after_onset,
)

__version__ = "0.1.0"
__all__ = [
    "astp", "satp", "bout_lengths", "mean_bout_length",
    "fragmentation_index", "active_mask",
    "interdaily_stability", "intradaily_variability",
    "l5", "m10", "relative_amplitude",
    "total_rest_time", "rest_efficiency", "wake_after_onset", "number_of_awakenings",
    "cosinor", "CosinorResult",
    "astp_segmented", "satp_segmented", "bout_lengths_segmented",
    "segment_indices", "split_on_gaps",
]
