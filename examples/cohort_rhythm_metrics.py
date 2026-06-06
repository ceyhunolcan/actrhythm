"""Compute rhythm metrics (IS, IV, RA, L5, M10) for an entire NHANES cohort
from raw PAXMIN_H.xpt, using actrhythm.

Streams the large .xpt once. Each participant's metrics are computed and their
raw minutes freed as soon as the stream passes them, so memory stays flat
regardless of cohort size. Results are written incrementally to a CSV.

Read-only on the source data. Output is exploratory analysis, not a published
pipeline figure.
"""
import csv
import sys
import numpy as np
import pandas as pd

import actrhythm as ar

PAXMIN = "/Users/ceyhun/Downloads/PAXMIN_H.xpt"
SEQN_LIST = "/Users/ceyhun/od-activity-rhythm/data/analytic_seqn_list.csv"
OUT = "/Users/ceyhun/actrhythm/cohort_rhythm_metrics.csv"
CHUNK = 500_000
MIN_DAYS = 3
FIELDS = ["SEQN", "n_days", "IS", "IV", "RA", "L5", "M10"]


def _ib(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: int(x.decode()) if isinstance(x, bytes) else int(x))


def metrics_for(pdf: pd.DataFrame) -> dict | None:
    pdf = pdf.copy()
    pdf["day"] = _ib(pdf["PAXDAYM"])
    pdf["mid"] = pdf.groupby("day").cumcount()
    pdf["hour"] = (pdf["mid"] // 60).clip(upper=23)
    grid = pdf.groupby(["day", "hour"])["PAXMTSM"].mean().unstack("hour")
    # ensure all 24 hour-columns exist (some hours may have no recorded minutes)
    grid = grid.reindex(columns=range(24))
    # keep only days with a complete 24-hour profile
    grid = grid.dropna(axis=0)
    nd = int(grid.shape[0])
    if nd < 1:
        return None
    s = grid.to_numpy().ravel()
    if s.size % 24 != 0:
        return None
    return {
        "n_days": nd,
        "IS": round(float(ar.interdaily_stability(s, epochs_per_hour=1)), 5),
        "IV": round(float(ar.intradaily_variability(s)), 5),
        "RA": round(float(ar.relative_amplitude(s, epochs_per_hour=1)), 5),
        "L5": round(float(ar.l5(s, epochs_per_hour=1)), 4),
        "M10": round(float(ar.m10(s, epochs_per_hour=1)), 4),
    }


def main() -> None:
    targets = set(pd.read_csv(SEQN_LIST)["SEQN"].dropna().astype(int))
    print(f"Cohort: {len(targets)} SEQNs. Streaming {PAXMIN} ...")
    buffers: dict[int, list[pd.DataFrame]] = {}
    done = 0
    skipped = 0
    n_chunks = 0

    out_f = open(OUT, "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDS)
    writer.writeheader()

    def flush(seqn: int) -> None:
        nonlocal done, skipped
        pdf = pd.concat(buffers.pop(seqn), ignore_index=True)
        m = metrics_for(pdf)
        if m is None:
            skipped += 1
            return
        m["SEQN"] = seqn
        writer.writerow(m)
        out_f.flush()
        done += 1

    with pd.read_sas(PAXMIN, format="xport", chunksize=CHUNK) as reader:
        for chunk in reader:
            n_chunks += 1
            chunk = chunk.copy()
            chunk["SEQN"] = pd.to_numeric(chunk["SEQN"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["SEQN"].isin(targets)]
            present = set(chunk["SEQN"].dropna().astype(int)) if len(chunk) else set()
            if len(chunk):
                for seqn, g in chunk.groupby("SEQN"):
                    buffers.setdefault(int(seqn), []).append(g[["SEQN", "PAXDAYM", "PAXMTSM"]])
            # flush participants we've passed (file is SEQN-sorted)
            for seqn in [s for s in buffers if s not in present]:
                flush(seqn)
            if n_chunks % 20 == 0:
                print(f"  chunk {n_chunks}: {done} done, {len(buffers)} buffering, {skipped} skipped")

    for seqn in list(buffers):
        flush(seqn)
    out_f.close()

    res = pd.read_csv(OUT)
    enough = res[res["n_days"] >= MIN_DAYS]
    print(f"\nDone. {done} participants processed, {skipped} skipped (no usable days).")
    print(f"Written to {OUT}")
    print(f"\n{len(enough)} participants with >= {MIN_DAYS} complete days:")
    print(f"  IS : mean={enough['IS'].mean():.3f}  median={enough['IS'].median():.3f}  sd={enough['IS'].std():.3f}")
    print(f"  IV : mean={enough['IV'].mean():.3f}  sd={enough['IV'].std():.3f}")
    print(f"  RA : mean={enough['RA'].mean():.3f}  sd={enough['RA'].std():.3f}")


if __name__ == "__main__":
    main()
