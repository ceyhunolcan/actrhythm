#!/usr/bin/env python3
"""Basic usage example for actrhythm.

Run: python examples/basic_usage.py
"""
from actrhythm import (
    active_mask,
    astp,
    fragmentation_index,
    rest_efficiency,
    satp,
    total_rest_time,
)

SERIES = [0, 2, 2, 0, 1, 0, 2, 2, 0, 0, 1, 2]
THR = 1.0

print("Series:", SERIES)
print("Active mask:", active_mask(SERIES, THR).tolist())
print("ASTP, SATP:", astp(SERIES, THR), satp(SERIES, THR))
print("Fragmentation index:", fragmentation_index(SERIES, THR))
print("Total rest epochs (threshold=1.0):", total_rest_time(SERIES, rest_threshold=THR))
print("Rest efficiency (TIB from first to last rest):", rest_efficiency(SERIES, rest_threshold=THR))
