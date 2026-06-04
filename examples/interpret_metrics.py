#!/usr/bin/env python3
"""Short script showing interpretation of a few metrics.

Run: python examples/interpret_metrics.py
"""
from actrhythm import (m10, l5, relative_amplitude, rest_efficiency)

# simple illustrative profile: low values (rest) then high values (active)
profile = [1]*8 + [10]*10 + [1]*6  # length 24
print("M10 (mean of most active 10h):", m10(profile, epochs_per_hour=1))
print("L5 (mean of least active 5h):", l5(profile, epochs_per_hour=1))
print("Relative amplitude:", relative_amplitude(profile, epochs_per_hour=1))

# rest efficiency example: rest mask with small wake periods
rest_mask = [False, True, True, True, False, True, True]
print("Rest efficiency:", rest_efficiency(rest_mask))
print("Interpretation: Higher RA (~1) => strong day-night contrast; lower RA => blunt rhythm.")
