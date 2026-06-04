# actrhythm

Rest-activity rhythm and fragmentation metrics for accelerometer data — the
standard non-parametric circadian measures and activity-fragmentation measures
used in actigraphy and digital-phenotyping research, implemented as small,
well-tested, dependency-light functions.

Every metric takes a plain 1-D activity series (minute-level MIMS, ENMO,
counts) and returns a scalar. No device- or study-specific assumptions, so the
same code works across cohorts and accelerometer types.

## Metrics

Circadian rhythm (Witting 1990; Van Someren 1999):
- `interdaily_stability` (IS) — reproducibility of the 24-h pattern across days
- `intradaily_variability` (IV) — within-day fragmentation of the rhythm
- `relative_amplitude` (RA) — contrast between most-active 10 h and least 5 h
- `l5` / `m10` — mean activity of least-active 5 h / most-active 10 h windows

Fragmentation (Di 2017; Wanigatunga 2019):
- `astp` — Active to Sedentary transition probability
- `satp` — Sedentary to Active transition probability
- `bout_lengths` / `mean_bout_length` — bout duration distribution / mean
- `fragmentation_index` — fraction of adjacent epochs that change state

## Install

    pip install -e ".[dev]"

## Usage

    import numpy as np
    import actrhythm as ar

    activity = np.loadtxt("participant_mims.csv")

    ar.interdaily_stability(activity, epochs_per_hour=60)
    ar.intradaily_variability(activity)
    ar.relative_amplitude(activity, epochs_per_hour=60)
    ar.astp(activity, sed_threshold=10.558)

## Testing

Every metric is pinned to a hand-computed or analytically-derived value
(IV of an alternating series is exactly 4.0; IS of identical repeated days is
exactly 1.0; RA on a known profile is 9/11):

    pytest

## References

- Witting W, et al. Biol Psychiatry (1990).
- Van Someren EJW, et al. Chronobiol Int (1999).
- Di J, et al. bioRxiv 182337 (2017).
- Wanigatunga AA, et al. J Gerontol A (2019).

## License

MIT
