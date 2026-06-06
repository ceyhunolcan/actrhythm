"""Compute MEANINGFUL Interdaily Stability (IS) from raw PAXMIN_H.xpt.

Unlike a pre-folded hourly profile (where IS is degenerately 1.0), this builds
a multi-day series — each day's 24 hourly means laid out separately, day-major —
so IS measures genuine across-day consistency.

Streams the large .xpt in chunks, keeps only the requested SEQNs. Read-only.
"""
import sys
import numpy as np
import pandas as pd

import actrhythm as ar

PAXMIN = "/Users/ceyhun/Downloads/PAXMIN_H.xpt"
CHUNK = 500_000
NEEDED = ["SEQN", "PAXDAYM", "PAXMTSM"]


def _to_int_bytes(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: int(x.decode()) if isinstance(x, bytes) else int(x))


def collect(seqns: set[int]) -> dict[int, pd.DataFrame]:
    """Stream the xpt; return {seqn: minute-level DataFrame} for the targets."""
    buffers: dict[int, list[pd.DataFrame]] = {}
    n = 0
    with pd.read_sas(PAXMIN, format="xport", chunksize=CHUNK) as reader:
        for chunk in reader:
            n += 1
            chunk = chunk.copy()
            chunk["SEQN"] = pd.to_numeric(chunk["SEQN"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["SEQN"].isin(seqns)]
            if len(chunk):
                for seqn, g in chunk.groupby("SEQN"):
                    buffers.setdefault(int(seqn), []).append(g[NEEDED])
            if n % 20 == 0:
                print(f"  ...chunk {n}, {len(buffers)} target SEQNs seen")
    return {s: pd.concat(parts, ignore_index=True) for s, parts in buffers.items()}


def day_major_series(pdf: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Build day-major hourly series; return (series, n_complete_days)."""
    pdf = pdf.copy()
    pdf["day"] = _to_int_bytes(pdf["PAXDAYM"])
    pdf["min_in_day"] = pdf.groupby("day").cumcount()
    pdf["hour"] = (pdf["min_in_day"] // 60).clip(upper=23)
    grid = pdf.groupby(["day", "hour"])["PAXMTSM"].mean().unstack("hour")
    grid = grid.dropna(axis=0)  # complete days only
    return grid.to_numpy().ravel(), grid.shape[0]


def main() -> None:
    seqns = {int(s) for s in sys.argv[1:]} or {73557, 73558}
    print(f"Streaming {PAXMIN} for SEQNs {sorted(seqns)} (this reads a large file)...")
    data = collect(seqns)
    for seqn in sorted(seqns):
        if seqn not in data:
            print(f"\nSEQN {seqn}: not found")
            continue
        series, ndays = day_major_series(data[seqn])
        print(f"\n=== SEQN {seqn}: {ndays} complete days, {series.size} hourly values ===")
        if ndays < 3:
            print("  (<3 complete days — IS not reliable, showing anyway)")
        print(f"  IS (across-day stability) = {ar.interdaily_stability(series, epochs_per_hour=1):.4f}")
        print(f"  IV (intradaily var.)      = {ar.intradaily_variability(series):.4f}")
        print(f"  RA                        = {ar.relative_amplitude(series, epochs_per_hour=1):.4f}")
        print(f"  L5={ar.l5(series, epochs_per_hour=1):.3f}  M10={ar.m10(series, epochs_per_hour=1):.3f}")


if __name__ == "__main__":
    main()
