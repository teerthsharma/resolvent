"""R6 -- local readings, global agreement.  VERDICT: DELETE.

REQUIREMENTS.md R6 falsifier, verbatim:

    plant a contradiction between two regions. The module must flag or resist
    it measurably more than an averaging baseline.

The deletion test is `test_holonomy_flags_the_contradiction_better_than_
averaging`. It is RED. The averaging baseline is not beaten; it wins.

Background. The recorded death of the obvious mechanism --

    dim ker Delta_F = d mod 2 on a connected graph with a cycle and generic
    SO(d) restriction maps -- exactly 0 in all 80 draws at d = 2, 4, 6, 8

-- is already bound by `tests/foreman/test_sheaf_kernel_parity.py` and is not
re-measured here. Its cause names an escape that looked good: a generic SO(d)
holonomy fixes an axis if and only if d is odd, so on a COMPACT structure group
the obstruction to a global section is generically invisible to a RANK
statistic. Change the structure group to translations of the line -- readings
are pairwise offsets, "v sits g_e above u" -- and the group is non-compact, the
obstruction is a VALUE rather than a kernel dimension, and it is generically
nonzero on any graph with a cycle. The sheaf died from the wrong instrument,
not the wrong subject.

That escape is real and it still loses. Module and baseline read the SAME
subspace -- the cycle space, the orthogonal complement of the image of the
incidence map -- and differ only in the norm. Measured, AUROC over 200
contradicted against 200 clean draws, 40 nodes in two regions of 20, sigma =
0.3 reading noise, one cross-region edge off by DELTA = 2.0:

    chords  |E|  cycles | LS rms (avg)  LS max|r|  holonomy_z  maxplus eig
         0   44       5 |      0.7741     0.7280      0.7068       0.6545
        12   56      17 |      0.8443     0.7774      0.8562       0.5839
        50   92      53 |      0.8671     0.8841      0.7870       0.7112
       150  188     149 |      0.8589     0.9860      0.9067       0.6841
       400  425     386 |      0.8017     0.9799      0.8918       0.6495

Three things fall out and all three delete R6.

1. The averaging baseline, read in its own natural L-infinity form -- fit one
   global potential by least squares, then look at the LARGEST edge residual --
   dominates the module everywhere at 50 cycles and above, 0.9860 against
   0.9067 at its best. "Averaged away" was the premise; least squares does not
   average the contradiction away, it localizes it to the edge that carries it.

2. The module's statistic never clears the pre-registered AUROC floor of 0.90
   except in one cell, where the baseline is 0.08 ahead. R6 required
   "measurably more than an averaging baseline" and the sign is wrong.

3. The unification is refuted too. The max cycle mean IS the max-plus
   eigenvalue and the log of the Perron root, the same object that certifies
   contraction in `test_r3_perturbation.py`. As a detector it is the WORST
   column on the board at every cycle count, 0.5839 to 0.7112. R3 and R6 do not
   collapse into one mechanism through the Perron root; that hypothesis is
   dead by measurement, not by opinion.

What survives is not a mechanism but a normalization note, recorded in
`perron.holonomy_z`: the max-plus eigenvalue divides a cycle defect by cycle
length because that is what makes it an eigenvalue, and for detection the
denominator has to be sqrt(length) because accumulated reading noise grows as
the square root while a planted contradiction does not grow at all. That
correction is worth +0.27 AUROC (0.5839 -> 0.8562 at 17 cycles) and it is still
not enough.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import perron as pn

N_A, N_B = 20, 20              # two regions
N_NODES = N_A + N_B
N_CROSS = 6                    # edges gluing region A to region B
CHORDS = 12                    # extra within-region edges, so cycles exist
SIGMA = 0.3                    # reading noise
DELTA = 2.0                    # the planted contradiction, on ONE cross edge
N_TRIALS = 200

AUROC_FLOOR = 0.90             # R6: "must flag"
BEATS_AVERAGING_BY = 0.10      # R6: "measurably more than an averaging baseline"


def graph(chords=CHORDS, seed=0):
    """Two regions, each internally connected with chords, glued by N_CROSS
    edges. Every region alone is connected; the contradiction is planted on a
    cross edge so it lives only in the glued cycles."""
    rng = np.random.default_rng(seed)
    e = [(i, i + 1) for i in range(N_A - 1)]
    e += [(N_A + i, N_A + i + 1) for i in range(N_B - 1)]
    for _ in range(chords):
        lo, hi = (0, N_A) if rng.random() < 0.5 else (N_A, N_NODES)
        u, v = rng.integers(lo, hi, 2)
        if u != v:
            e.append((int(min(u, v)), int(max(u, v))))
    first = len(e)
    e += [(int(rng.integers(0, N_A)), int(rng.integers(N_A, N_NODES)))
          for _ in range(N_CROSS)]
    return e, first


EDGES, FIRST_CROSS = graph()


def readings(seed, contradicted, device, edges=None, first=None):
    """Local readings of pairwise offsets, from one true potential plus noise.
    A contradiction is a single cross-region edge reporting DELTA off."""
    edges = EDGES if edges is None else edges
    first = FIRST_CROSS if first is None else first
    rng = np.random.default_rng(1000 + seed)
    phi = rng.normal(size=N_NODES)
    g = np.array([phi[v] - phi[u] for u, v in edges]) + rng.normal(0, SIGMA, len(edges))
    if contradicted:
        g[first] += DELTA
    return torch.tensor(g, dtype=torch.float64, device=device)


def auroc(pos, neg):
    a = np.concatenate([pos, neg])
    r = np.empty_like(a)
    r[np.argsort(a, kind="mergesort")] = np.arange(1, len(a) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def _scores(stat, device, edges=None, first=None, n=N_TRIALS):
    pos = np.array([stat(readings(s, True, device, edges, first)) for s in range(n)])
    neg = np.array([stat(readings(s, False, device, edges, first)) for s in range(n)])
    return auroc(pos, neg)


# ---------------------------------------------------------------- the statistics

def module_stat(g, edges=None):
    """The module: worst cycle defect, standardized by sqrt(cycle length)."""
    return pn.holonomy_z(EDGES if edges is None else edges, N_NODES, g)


def maxplus_stat(g, edges=None):
    """The unifying object: max cycle mean = max-plus eigenvalue = log Perron
    root, the same statistic that certifies contraction for R3."""
    return pn.max_cycle_mean(EDGES if edges is None else edges, N_NODES, g)


def _resid(g, edges):
    b = torch.zeros(len(edges), N_NODES, dtype=torch.float64)
    for i, (u, v) in enumerate(edges):
        b[i, v] += 1.0
        b[i, u] -= 1.0
    gg = g.to(torch.float64).reshape(-1, 1).cpu()
    return (b @ torch.linalg.lstsq(b, gg).solution - gg).flatten()


def averaging_rms(g, edges=None):
    """Averaging baseline, L2: RMS of the least-squares potential residual."""
    return float((_resid(g, EDGES if edges is None else edges) ** 2).mean() ** 0.5)


def averaging_max(g, edges=None):
    """Averaging baseline, L-infinity: the largest single edge residual of the
    same least-squares fit. This is the baseline R6 has to beat and does not."""
    return float(_resid(g, EDGES if edges is None else edges).abs().max())


# ------------------------------------------------------------- R6's deletion test

def test_holonomy_flags_the_contradiction_better_than_averaging(device):
    """RED. This is the test that deletes R6.

    The module and both averaging reads see the same cycle space. Fitting one
    global potential by least squares and reporting its largest edge residual
    localizes the planted contradiction rather than averaging it away, and it
    scores 0.9860 at 149 cycles against the module's 0.9067.
    """
    edges, first = graph(chords=150)
    mine = _scores(lambda g: module_stat(g, edges), device, edges, first)
    base = max(_scores(lambda g: averaging_rms(g, edges), device, edges, first),
               _scores(lambda g: averaging_max(g, edges), device, edges, first))
    assert mine >= AUROC_FLOOR and mine - base >= BEATS_AVERAGING_BY, (
        f"holonomy AUROC {mine:.4f} vs best averaging read {base:.4f} "
        f"(gap {mine - base:+.4f}); need >= {AUROC_FLOOR} and gap >= "
        f"{BEATS_AVERAGING_BY}. R6 deleted."
    )


# ------------------------------------------- what is true, measured and green

def test_the_max_plus_eigenvalue_is_the_worst_detector_on_the_board(device):
    """Kills the collapse hypothesis. The max cycle mean is the log of the
    Perron root -- literally R3's contraction certificate read in the log
    domain -- so if R3 and R6 were one mechanism this would be the detector.
    It is beaten by every other column at every cycle count tested."""
    for chords in (12, 150):
        edges, first = graph(chords=chords)
        mp = _scores(lambda g: maxplus_stat(g, edges), device, edges, first, n=100)
        hz = _scores(lambda g: module_stat(g, edges), device, edges, first, n=100)
        am = _scores(lambda g: averaging_max(g, edges), device, edges, first, n=100)
        assert mp < hz and mp < am, f"chords={chords}: {mp=} {hz=} {am=}"


def test_length_normalization_is_the_whole_gap_the_module_ever_closed(device):
    """The one transferable result. Dividing a cycle defect by its length is
    what makes it a max-plus eigenvalue; dividing by sqrt(length) is what makes
    it a detector, because noise accumulates as sqrt(length) and the planted
    contradiction does not accumulate at all. Worth +0.27 AUROC here."""
    mp = _scores(lambda g: maxplus_stat(g), device)
    hz = _scores(lambda g: module_stat(g), device)
    assert hz - mp >= 0.2, f"maxplus {mp:.4f} -> holonomy_z {hz:.4f}"


def test_a_magnitude_pooling_reader_is_exactly_blind_to_it(device):
    """The one place the module is strictly better, and it is not enough to
    save R6 because the beaten reader is a strawman and is named as one.

    Modeled on `monodromy`'s measured template -- a local statistic reading
    0.368101 against 0.370076 on the two sides of an opposite global answer.
    Here the blindness is exact rather than to three digits: flip the
    ORIENTATION of one region's reading instead of shifting it, and every
    reported magnitude is unchanged, so any statistic pooling magnitudes is
    bitwise identical while the holonomy moves.
    """
    g = readings(0, False, device)
    flipped = g.clone()
    flipped[FIRST_CROSS] = -flipped[FIRST_CROSS]
    pooled = lambda v: (float(v.abs().mean()), float(v.abs().std()),
                        float(v.abs().sort().values.sum()))
    assert pooled(g) == pooled(flipped), "the pooled reader is not actually blind"
    assert module_stat(flipped) != module_stat(g)


def test_each_region_alone_is_consistent_so_the_defect_is_purely_global(device):
    """'Locally fine everywhere, globally identified.' Restricted to either
    region the contradicted readings admit a potential to within noise; the
    defect appears only once the two regions are glued. The construction is
    sound -- it is the detector that loses, not the setup."""
    clean, dirty = readings(0, False, device), readings(0, True, device)
    for lo, hi in ((0, N_A), (N_A, N_NODES)):
        idx = [i for i, (u, v) in enumerate(EDGES) if lo <= u < hi and lo <= v < hi]
        sub = [(EDGES[i][0] - lo, EDGES[i][1] - lo) for i in idx]
        before = pn.holonomy_z(sub, hi - lo, clean[idx])
        after = pn.holonomy_z(sub, hi - lo, dirty[idx])
        assert after == before, (
            f"region [{lo},{hi}) moved {before:.6f} -> {after:.6f}; the "
            f"contradiction is not purely global")
    assert module_stat(dirty) > module_stat(clean), (
        f"glued {module_stat(clean):.4f} -> {module_stat(dirty):.4f}")


def test_the_statistic_is_zero_exactly_when_a_global_potential_exists(device):
    """The certificate half of 'flag or resist'. Noiseless readings from a
    genuine potential score exactly 0. True, cheap, and not sufficient: a
    zero-noise certificate is not a detector under noise, which is the whole
    finding above."""
    rng = np.random.default_rng(7)
    phi = rng.normal(size=N_NODES)
    g = torch.tensor([phi[v] - phi[u] for u, v in EDGES],
                     dtype=torch.float64, device=device)
    assert module_stat(g) < 1e-12, module_stat(g)
    g2 = g.clone()
    g2[FIRST_CROSS] += DELTA
    assert module_stat(g2) > 1e-3, module_stat(g2)
