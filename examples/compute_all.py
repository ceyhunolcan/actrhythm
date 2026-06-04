#!/usr/bin/env python3
"""Compute a broad set of actrhythm metrics on a sample series.

Run: python examples/compute_all.py
"""
from actrhythm import (
    active_mask, astp, satp, bout_lengths, mean_bout_length, fragmentation_index,
    interdaily_stability, intradaily_variability, m10, l5, relative_amplitude,
    total_rest_time, rest_efficiency, wake_after_onset, number_of_awakenings,
)

SERIES = [0, 2, 2, 0, 1, 0, 2, 2, 0, 0, 1, 2] * 3  # repeat to mimic multiple days
THR = 1.0

print("ASTP", astp(SERIES, THR))
print("SATP", satp(SERIES, THR))
print("Fragmentation index", fragmentation_index(SERIES, THR))
print("Bout lengths (active)", bout_lengths(SERIES, THR, active=True).tolist())
print("Mean active bout (epochs)", mean_bout_length(SERIES, THR, active=True))
print("Intradaily variability", intradaily_variability(SERIES))
# interdaily_stability requires epochs_per_hour (use 1 for simplicity)
print("Interdaily stability (epochs_per_hour=1)", interdaily_stability(SERIES, epochs_per_hour=1))
print("M10", m10(SERIES, epochs_per_hour=1))
print("L5", l5(SERIES, epochs_per_hour=1))
print("RA", relative_amplitude(SERIES, epochs_per_hour=1))
print("Total rest time", total_rest_time(SERIES, rest_threshold=THR))
print("Rest efficiency", rest_efficiency(SERIES, rest_threshold=THR))
print("WASO", wake_after_onset(SERIES, rest_threshold=THR))
print("Awakenings", number_of_awakenings(SERIES, rest_threshold=THR))
