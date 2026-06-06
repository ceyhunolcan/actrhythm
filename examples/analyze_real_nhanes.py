"""Run actrhythm on real NHANES hourly profiles and show results next to the
pipeline's fragmentation features.

Read-only: loads the existing CSVs, computes metrics, prints. Touches nothing.
"""
import sys
import numpy as np
import pandas as pd

import actrhythm as ar
from actrhythm.cosinor import cosinor

HOURLY = "/Users/ceyhun/Downloads/paxmin_output/hourly_profile.csv"
FRAG   = "/Users/ceyhun/Downloads/paxmin_output/fragmentation_features.csv"


def analyze(seqn: int) -> None:
    hp = pd.read_csv(HOURLY)
    sub = hp[hp["SEQN"] == seqn].sort_values("hour")
    if sub.empty:
        print(f"SEQN {seqn} not found in hourly profile")
        return
    series = sub["mean_mims"].to_numpy(dtype=float)
    print(f"\n=== SEQN {seqn} — {len(series)} hourly values ===")
    if len(series) != 24:
        print(f"  (note: {len(series)} hours, not a clean 24; metrics still computed)")

    print("\n-- actrhythm rhythm metrics (on hourly profile, epochs_per_hour=1) --")
    print(f"  IV  (intradaily variability) = {ar.intradaily_variability(series):.4f}")
    print(f"  IS  (interdaily stability)   = {ar.interdaily_stability(series, epochs_per_hour=1):.4f}")
    print(f"  L5  (least-active 5h mean)   = {ar.l5(series, epochs_per_hour=1):.4f}")
    print(f"  M10 (most-active 10h mean)   = {ar.m10(series, epochs_per_hour=1):.4f}")
    print(f"  RA  (relative amplitude)     = {ar.relative_amplitude(series, epochs_per_hour=1):.4f}")
    c = cosinor(series, period=24.0)
    print(f"  cosinor: MESOR={c.mesor:.3f}  amplitude={c.amplitude:.3f}  "
          f"acrophase={c.acrophase_time:.2f}h  R2={c.r_squared:.3f}")

    print("\n-- pipeline fragmentation features (for context; minute-level, not directly comparable) --")
    fdf = pd.read_csv(FRAG)
    frow = fdf[fdf["SEQN"] == seqn]
    if frow.empty:
        print("  (SEQN not in fragmentation_features.csv)")
    else:
        r = frow.iloc[0]
        print(f"  pipeline ASTP_minute = {r['ASTP_minute']:.4f}")
        print(f"  pipeline SATP_minute = {r['SATP_minute']:.4f}")
        print(f"  n_wake_min_total     = {int(r['n_wake_min_total'])}")
    print("\n  NOTE: pipeline ASTP/SATP are minute-level with per-block gap")
    print("  handling; actrhythm's flat astp needs the raw minute series, not")
    print("  the hourly profile. The rhythm metrics above are the valid check.")


if __name__ == "__main__":
    seqns = [int(s) for s in sys.argv[1:]] or [73557, 73558]
    for s in seqns:
        analyze(s)
