"""X35a -- the residual instrument. `r = z_obs - z_model(visible)`.

CEQ_V15_1_DELTA.md, X35a: onset by a MEMORYLESS comparator on residual energy
(a Shewhart chart), threshold calibrated on NO-PLANT runs, output
`(onset, magnitude, downstream gate estimates)`.

MEMORYLESS, AND WHY NOT CUSUM. `detect` compares each index's residual energy
against the threshold INDEPENDENTLY -- no running sum, no state carried across
indices, so a run of small excursions can never accumulate into an alarm. That
is the whole difference from CUSUM, and it is settled: CUSUM's dependence
inflation killed X26 and the memoryless detector was the one that survived at
matched ARL0 on dependent data. No CUSUM path is implemented here, because
none is needed by any must-fire and a second detector in the file would invite
exactly the isolated comparison the delta rules out.

THE VISIBLE MODEL IS THE ORACLE, AND THAT BOUNDS THE READING. Nothing is
trained (L-LEAN), so `oracle_visible` is the generator's own analytic forward
pass over the VISIBLE variables only -- `bed_k.rebuild(bed, bed["b"])`, which
touches `b` and the kernel and never `bed["plant"]`. The consequence is exact
and worth stating rather than discovering: with an exact visible model

    r  =  observation noise  +  planted latent

identically, so the no-plant residual IS the observation noise and "flat under
no plant" is checkable exactly, before any arm exists. It also means the
oracle residual does not depend on the kernel at all (tests/x35 asserts this),
so the flatness reading here is a statement about the OBSERVATION model and
must be RE-MEASURED the moment `z_model` becomes a trained arm, whose
approximation error is kernel-dependent and is not zero. The detector cannot
outrun the model it subtracts.

`truncated_visible` exists to keep that from being a slogan: it is a
deliberately imperfect visible model (the same kernel truncated to a finite
window, which is the error a limited-context arm makes) and it is what gives
the flatness gate a rejection region rather than a foregone pass
(MISTAKES.md V-10).

CALIBRATION. `calibrate` takes the (1-alpha) quantile of the RUN-MAX residual
energy over a calibration seed block, which targets a per-RUN false-alarm rate
of alpha directly. Every rate reported against it must come from a disjoint
evaluation block: a threshold refitted to the data it judges is not a
threshold (MISTAKES.md M-2), and this module makes that structural by taking
seeds as an explicit argument everywhere instead of drawing them internally.

NO CRB. CEQ_V15_2_DELTA.md (d) asks for distance-to-CRB beside every onset CI
and STRIKES any CRB number quoted before its own noise-sweep must-fire. This
module quotes none. The onset is a discrete index and enters the likelihood
only through it, so the score with respect to it does not exist and neither
does a Fisher information; a CRB-bearing localization bed needs a CONTINUOUS
onset parameter. `run_detect` returns the whole residual field beside the
onset so the delta's competing estimators -- exact source solve `(I - A) r`,
time-reversal `W^T r` -- score the SAME field this onset was read from.

SEEDING DISCIPLINE, matching ceq/beds/bed_k.py: one `seed` per run, and the
observation-noise stream is derived from it through a fixed salt so it is
reproducible, disjoint from the bed's own stream, and never touches global
state.

float64 throughout.
"""
from __future__ import annotations

import numpy as np

from ceq.beds import bed_k

__all__ = [
    "oracle_visible", "truncated_visible", "observe", "run_residual",
    "detect", "run_detect", "calibrate", "false_alarm_rate", "flatness",
]

#: Fixed salt for the observation-noise stream. `default_rng([OBS_SALT, seed])`
#: spawns a stream from a SeedSequence that shares no state with
#: `default_rng(seed)`, so the noise at seed s is not a continuation of the
#: bed's own draws at seed s -- the two channels are independent by
#: construction rather than by hoping the draw counts never line up.
OBS_SALT = 0x35A


# --------------------------------------------------------------------------
# visible models
# --------------------------------------------------------------------------


def oracle_visible(bed: dict) -> np.ndarray:
    """z_model(visible): the generator's analytic forward pass over the
    visible inputs. EXACT -- reads `bed["b"]` and the kernel, never
    `bed["plant"]`. This is the bar-setting model, not a stand-in for an arm.
    """
    return bed_k.rebuild(bed, bed["b"])


def truncated_visible(L: int):
    """A deliberately IMPERFECT visible model: the same kernel with every lag
    beyond `L` dropped. This is the approximation error a finite-context arm
    actually makes on a long-memory bed -- the tail it cannot see -- and it is
    the control that gives the no-plant flatness gate a rejection region.
    Returns a `bed -> z_model` callable so it drops into the same slot as
    `oracle_visible`.
    """
    def model(bed: dict) -> np.ndarray:
        K = bed["K"]
        n = K.shape[0]
        lag = np.arange(n)[:, None] - np.arange(n)[None, :]
        return np.where(lag <= L, K, 0.0) @ bed["b"]
    model.__name__ = f"truncated_visible_L{L}"
    return model


# --------------------------------------------------------------------------
# observation and residual
# --------------------------------------------------------------------------


def observe(bed: dict, noise_sd: float, seed: int) -> np.ndarray:
    """z_obs = z + noise_sd * N(0, 1), on a stream disjoint from the bed's.

    At `noise_sd == 0` this returns `z` bitwise: `0.0 * g` is a signed zero and
    `x + (-0.0) == x` for every finite x. The draw still happens so the code
    path is the same one the noisy case takes.
    """
    rng = np.random.default_rng([OBS_SALT, seed])
    return bed["z"] + noise_sd * rng.standard_normal(bed["n"])


def run_residual(seed: int, *, n: int, kind: str, params: dict, noise_sd: float,
                 model=None, plant=None) -> tuple[dict, np.ndarray]:
    """One run: build the bed at `seed`, observe it, subtract the visible
    model. Returns `(bed, r)` -- the bed so a caller can score against
    `bed["plant"]`, the residual so a caller can run its own estimator on the
    identical field. `model` defaults to `oracle_visible`."""
    bed = bed_k.build(kind, n, seed, plant=plant, **params)
    z_obs = observe(bed, noise_sd, seed)
    z_model = (model or oracle_visible)(bed)
    return bed, z_obs - z_model


# --------------------------------------------------------------------------
# the memoryless comparator
# --------------------------------------------------------------------------


def detect(r: np.ndarray, threshold: float) -> dict:
    """Shewhart onset: the FIRST index whose residual energy exceeds
    `threshold`, with nothing accumulated across indices.

    `magnitude` is the RMS residual from the onset on. Under a plant of size m
    at noise sd s it estimates `sqrt(m^2 + s^2)`, i.e. it is biased UP by the
    noise variance inside the root; the bias is reported rather than
    subtracted, because subtracting an assumed noise variance would put a
    second unverified model inside the instrument.

    `gate_estimates` is the per-index drive the posited latent node must carry
    downstream (X35b re-propagates with it): the residual itself from the
    onset on, exactly 0 upstream of it. `residual` and `energy` are returned
    whole so a competing estimator scores this field, not its own draw.

    Strict `>`: at a threshold of 0.0 (which is what an exact model at zero
    noise calibrates to) a bitwise-zero residual must call nothing.
    """
    energy = r * r
    hits = np.flatnonzero(energy > threshold)
    onset = int(hits[0]) if hits.size else None
    gate = np.zeros_like(r)
    magnitude = None
    if onset is not None:
        gate[onset:] = r[onset:]
        magnitude = float(np.sqrt(energy[onset:].mean()))
    return dict(onset=onset, magnitude=magnitude, gate_estimates=gate,
                residual=r, energy=energy, threshold=float(threshold))


def run_detect(seed: int, threshold: float, *, plant=None, **cfg) -> dict:
    """`run_residual` then `detect`, at one seed."""
    _, r = run_residual(seed, plant=plant, **cfg)
    return detect(r, threshold)


# --------------------------------------------------------------------------
# calibration on no-plant runs, and the rates measured against it
# --------------------------------------------------------------------------


def calibrate(seeds, alpha: float, **cfg) -> dict:
    """Threshold from NO-PLANT runs: the (1-alpha) quantile of the run-max
    residual energy over `seeds`. No `plant` argument is accepted -- calibrating
    on planted data is the failure this whole node is a control for.

    Targeting the RUN-MAX distribution makes `alpha` a per-run false-alarm
    rate directly, which is the quantity must-fire 2 is stated in. Interpolation
    `"higher"` with the strict `>` in `detect` makes the in-sample rate <=
    alpha by construction -- which is exactly why the in-sample rate is not
    evidence and a fresh block is (MISTAKES.md M-2). `maxima` is returned so a
    caller can print the in-sample rate beside the fresh one instead of
    substituting it.
    """
    maxima = np.array([float(np.max(run_residual(s, **cfg)[1] ** 2)) for s in seeds],
                      dtype=np.float64)
    return dict(threshold=float(np.quantile(maxima, 1.0 - alpha, method="higher")),
                alpha=float(alpha), runs=int(maxima.size), maxima=maxima)


def false_alarm_rate(seeds, threshold: float, *, plant=None, **cfg) -> dict:
    """Fraction of runs in which ANY index is called, plus the per-index
    exceedance rate and `ARL0 = 1 / per_index_rate` -- the average run length
    to a false alarm, in indices, which is the axis the memoryless-vs-CUSUM
    comparison is matched on.

    `plant` is accepted so the same counter reads the ZERO-MAGNITUDE plant
    (present in the manifest, absent from the label): its detection rate must
    be this false-alarm rate, not ~1.
    """
    runs = alarms = exceed = indices = 0
    for s in seeds:
        _, r = run_residual(s, plant=plant, **cfg)
        hit = (r * r) > threshold
        runs += 1
        alarms += int(hit.any())
        exceed += int(hit.sum())
        indices += int(hit.size)
    per_index = exceed / indices if indices else float("nan")
    return dict(rate=alarms / runs if runs else float("nan"), alarms=alarms,
                runs=runs, exceedances=exceed, indices=indices,
                per_index_rate=per_index,
                arl0=(indices / exceed) if exceed else float("inf"))


def flatness(seeds, **cfg) -> dict:
    """Is the NO-PLANT residual flat? Two numbers, pooled over `seeds`:

    `corr` -- Pearson correlation of per-index MEAN residual energy against
    index. Flat means 0. Its null sd is `1/sqrt(n-1)` however many runs are
    pooled: more runs shrink the per-index deviations but the correlation is
    scale-invariant, so pooling does not tighten this number and the bar has to
    be set from `n`, not from the run count.

    `half_ratio` -- second-half mean energy over first-half mean energy. Flat
    means 1. This one DOES tighten with run count (relative sd
    `sqrt(2 / (runs * n / 2))`), so it is the sharper of the two.

    An approximation error that grows with index -- a dropped long-memory tail,
    say -- moves both; iid observation noise moves neither.
    """
    E = np.array([run_residual(s, **cfg)[1] ** 2 for s in seeds], dtype=np.float64)
    mean_e = E.mean(axis=0)
    idx = np.arange(mean_e.size, dtype=np.float64)
    half = mean_e.size // 2
    return dict(corr=float(np.corrcoef(idx, mean_e)[0, 1]),
                half_ratio=float(mean_e[half:].mean() / mean_e[:half].mean()),
                mean_energy=float(mean_e.mean()), runs=int(E.shape[0]),
                per_index_mean_energy=mean_e)
