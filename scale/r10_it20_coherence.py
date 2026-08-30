"""Which `d` and `k` S1 actually runs at, and whether 0.1748 is its constant.

THE RULE UNDER TEST, verbatim from the v-main.3M script, iteration 20:

    capacity line: role coherence floor = 0 for k <= d (Welch), random-role
    coherence ~ 0.1748 at d=256, k=16 (corrected constant, MC CI
    [0.17446, 0.17513]) -- the scramble control's expected residual, not zero.

WHAT THIS MODULE DOES. It measures three things the rule asserts and one it
assumes, at the configuration S1 actually runs at rather than the one the
constant was derived for.

  1. The Welch floor at every shipped `(d, k)`, not only at the cited one.
  2. The random-unit-vector max coherence by Monte Carlo at both `(256, 16)`
     and the shipped `(16, k)`, so the comparison is visible rather than
     asserted.
  3. The coherence of the REAL selected key directions, because the random
     model's applicability to them is a claim, not a definition.
  4. The scramble control's residual on the actual construction, which is what
     any S1 table is read against.

WHY THE CONFIGURATION IS THE WHOLE QUESTION. Coherence has no scale-free value:
its null depends on `d` and on `k`, so a coherence constant is anchored only to
the `(d, k)` it was measured at. That is MISTAKES.md V-17 in its exact form.

WHERE THE SHIPPED GEOMETRY IS FIXED, and it is fixed in one place:
`scale/m3_capability.py:79` sets `D_MODEL = 16` and `:87` sets `K_PIVOTS = 8`.
Keys are `wk(x)` with `wk = nn.Linear(d_model, d_model)` (`:109`), so a key
vector lives in `R^16`. `scale/foreman_signfloor.py:82` sets the same
`DMODEL = 16` for the consequence path, and `scale/foreman_consequence.py`'s
`--d 256` is the TASK width, passed as `d`, with `d_model=SF.DMODEL` supplied
separately at `:427` -- so no run in this repo puts a key vector in `R^256`.

WHAT A "SCRAMBLE CONTROL" IS HERE. `scale/fgreen_matched.py:86 select_random`
draws a fixed uniform sample of positions instead of the shipped top-k, and
`imbalance_table` (`:114`) prints the Hungarian residual it leaves against the
shipped pool. That IS the it.20 matcher: cost `|norm_i - norm_j|`, assignment
by `scipy.optimize.linear_sum_assignment`, verified against the Monge sorted
optimum (`scale/matcher.py:153-192`). Its residual is in KEY-NORM units.
Coherence is a dimensionless cosine. The two are not the same quantity and no
value of one is the expected value of the other.

RUN
    python -m scale.r10_it20_coherence            # demo(), assert-based, ~35 s
    python -m scale.r10_it20_coherence --write    # table + rows, ~60 s

Seeds are declared before any run: Monte Carlo seed 0 (matching
`scale/coherence_floor.py`'s own published cell so the d=256 row is a
reproduction, not a re-pick), eval batch seed 12345, arm seed 0, bootstrap
seed 0. Nothing here reads a clock, a hash seed or a thread count.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scale import matcher                                        # noqa: E402
from scale import m3_capability as m3                            # noqa: E402
from scale.coherence_floor import (monte_carlo, pair_union_coherence,  # noqa: E402
                                   welch_bound)
from scale.negation_scope import make_batch                      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The constant the rule cites, and the CI it cites with it.
CITED = 0.1748
CITED_CI = (0.17446, 0.17513)

#: Declared before the run.
MC_SEED, MC_TRIALS = 0, 20000
EVAL_SEED, ARM_SEED, BOOT_SEED = 12345, 0, 0

#: The shipped eval geometry. `s` and `d` are TASK parameters; `d_model` is the
#: vector dimension and is the only one coherence is a function of.
S_TASK, D_TASK, N_EVAL, N_DRAWS = 64, 24, 512, 64

#: Pivot counts this repo actually runs. 8 is `m3.K_PIVOTS`; 32 and 128 are the
#: `--ks` default of every sweep (`scale/aggregator_matched_filler.py:155`,
#: `scale/arm_s.py:485`, `scale/b1_collapse_test.py:154`, and five more).
SHIPPED_KS = (8, 32, 128)


def welch_row(d: int, k: int) -> dict:
    """The Welch floor and whether the rule's `k <= d` premise holds there."""
    return dict(d=d, k=k, k_le_d=(k <= d), welch=welch_bound(d, k),
                vacuous=(welch_bound(d, k) == 0.0))


def coherence_row(d: int, k: int, *, trials: int = MC_TRIALS,
                  seed: int = MC_SEED) -> dict:
    """Random-unit-vector coherence at `(d, k)`, with the cited value beside it."""
    mc = monte_carlo(d, k, trials=trials, seed=seed)
    return dict(d=d, k=k, trials=trials, seed=seed,
                max_mean=mc["max_mean"], max_ci_lo=mc["max_ci_lo"],
                max_ci_hi=mc["max_ci_hi"], mean_mean=mc["mean_mean"],
                mean_ci_lo=mc["mean_ci_lo"], mean_ci_hi=mc["mean_ci_hi"],
                welch=welch_bound(d, k), mu_pairs=pair_union_coherence(d, k),
                cited=CITED,
                cited_inside_ci=(mc["max_ci_lo"] <= CITED <= mc["max_ci_hi"]),
                ratio_to_cited=mc["max_mean"] / CITED)


def _ci(v: np.ndarray) -> tuple[float, float, float]:
    """Normal 95% CI on a mean over independent draws. Same form as
    `coherence_floor.monte_carlo`, so the two are read on one scale."""
    m = float(v.mean())
    h = 1.96 * float(v.std(ddof=1)) / np.sqrt(v.size)
    return m, m - h, m + h


def shipped_scramble_residual(n_draws: int = N_DRAWS) -> dict:
    """The SHIPPED scramble control's residual, through its own code path.

    `fgreen_matched.imbalance_table` is the producer of the published cell at
    `results/fgreen_matched_it1.txt:11`. Called here so the number in this
    report is reproduced in-process rather than copied out of a text file.

    It differs from `scramble_residual` in one respect worth naming: because
    `select_random` re-seeds its generator inside every call and
    `imbalance_table` calls it one draw at a time, the shipped control applies
    the SAME permutation to all draws. That is one scramble measured 64 times;
    `scramble_residual` draws a fresh one per draw and is the population
    expectation. Both are reported, and they agree.
    """
    from scale import fgreen_matched as fg                       # noqa: PLC0415
    x, _, _, _ = make_batch(N_EVAL, S_TASK, D_TASK, d_model=m3.D_MODEL,
                            seed=EVAL_SEED)
    tbl = fg.imbalance_table(x[:n_draws], m3.K_PIVOTS, ARM_SEED)
    mval, lo, hi = tbl["pivot_random"]
    bval, blo, bhi = tbl["pivot_band"]
    return dict(d=m3.D_MODEL, k=m3.K_PIVOTS, draws=n_draws,
                residual=mval, ci_lo=lo, ci_hi=hi,
                band_residual=bval, band_ci_lo=blo, band_ci_hi=bhi,
                published=0.157792,
                source="results/fgreen_matched_it1.txt:11")


def real_keys(n_draws: int = N_DRAWS) -> np.ndarray:
    """The shipped eval batch's key tensor, `[n_draws, s, d_model]`, float64.

    Untrained on purpose: the it.20 capacity line is about what the GEOMETRY
    permits before training, which is the only regime a coherence floor speaks
    to at all.
    """
    x, _, _, _ = make_batch(N_EVAL, S_TASK, D_TASK, d_model=m3.D_MODEL,
                            seed=EVAL_SEED)
    torch.manual_seed(ARM_SEED)
    arm = m3.Arm("pivot_unsigned", S_TASK)
    with torch.no_grad():
        return arm.wk(x[:n_draws]).double().numpy()


def real_coherence(key: np.ndarray, k: int) -> dict:
    """Coherence of the DIRECTIONS of the shipped top-k keys, per draw.

    The random-unit-vector model assumes independent uniform directions. Real
    keys are one linear map applied to correlated token features, so whether
    the model describes them is measured here rather than assumed.
    """
    mx, mn = [], []
    for row in key:
        take = np.argsort(-np.linalg.norm(row, axis=-1))[:k]
        u = row[take]
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        g = np.abs(u @ u.T)
        iu = np.triu_indices(u.shape[0], 1)
        mx.append(g[iu].max())
        mn.append(g[iu].mean())
    m, lo, hi = _ci(np.array(mx))
    mm, mlo, mhi = _ci(np.array(mn))
    return dict(d=key.shape[-1], k=k, draws=key.shape[0],
                max_mean=m, max_ci_lo=lo, max_ci_hi=hi,
                mean_mean=mm, mean_ci_lo=mlo, mean_ci_hi=mhi)


def scramble_residual(key: np.ndarray, k: int, *, seed: int = BOOT_SEED) -> dict:
    """The it.20 matcher's residual for the scramble control, and its scale.

    `argmin_pi sum |norm_i - norm_pi(i)| / n` between the shipped top-k pool and
    a uniformly drawn k, using `scale/matcher.py` -- the same solver, with its
    Monge oracle verified on the first draw. Reported next to the key-norm sd,
    because a residual is only readable against the spread it failed to remove.
    """
    g = torch.Generator().manual_seed(4242)          # fgreen_matched.RANDOM_SEED
    s = key.shape[1]
    res, norms = [], []
    for i, row in enumerate(key):
        nr = np.linalg.norm(row, axis=-1)
        norms.append(nr)
        a = np.sort(nr)[::-1][:k]
        b = nr[torch.randperm(s, generator=g)[:k].numpy()]
        res.append(matcher.match(a, b, verify=(i == 0)).residual)
    res = np.array(res)
    m, lo, hi = _ci(res)
    flat = np.concatenate(norms)
    sd = float(flat.std(ddof=1))
    return dict(d=key.shape[-1], k=k, draws=key.shape[0], seed=seed,
                residual=m, ci_lo=lo, ci_hi=hi,
                keynorm_mean=float(flat.mean()), keynorm_sd=sd,
                residual_in_sd=m / sd, cited=CITED, ratio_to_cited=m / CITED)


def rows() -> list[dict]:
    """Every row this iteration publishes, in one list, tagged by kind."""
    out: list[dict] = []
    for d, k in [(256, 16)] + [(m3.D_MODEL, k) for k in SHIPPED_KS] + [(16, 16)]:
        out.append(dict(kind="welch", **welch_row(d, k)))
    for d, k in [(256, 16), (m3.D_MODEL, m3.K_PIVOTS), (16, 32), (16, 128)]:
        out.append(dict(kind="mc_random_unit", **coherence_row(d, k)))
    key = real_keys()
    out.append(dict(kind="real_key_directions", **real_coherence(key, m3.K_PIVOTS)))
    out.append(dict(kind="scramble_residual", **scramble_residual(key, m3.K_PIVOTS)))
    out.append(dict(kind="scramble_residual_shipped", **shipped_scramble_residual()))
    return out


def table(rs: list[dict]) -> str:
    w: list[str] = []
    w.append("=" * 78)
    w.append(f"R10 it.20 -- COHERENCE AT THE CONFIGURATION S1 RUNS AT")
    w.append(f"  shipped geometry: d_model={m3.D_MODEL} (m3_capability.py:79), "
             f"k={m3.K_PIVOTS} (:87)")
    w.append(f"  cited constant:   {CITED} at d=256, k=16, CI {list(CITED_CI)}")
    w.append(f"  MC seed={MC_SEED} trials={MC_TRIALS}; eval seed={EVAL_SEED}; "
             f"arm seed={ARM_SEED}")
    w.append("=" * 78)
    w.append("")
    w.append("WELCH FLOOR -- the rule's 'floor = 0 for k <= d' premise, per config")
    w.append(f"  {'d':>5} {'k':>5} {'k<=d':>6} {'welch':>10}  reading")
    for r in [r for r in rs if r["kind"] == "welch"]:
        w.append(f"  {r['d']:>5} {r['k']:>5} {str(r['k_le_d']):>6} "
                 f"{r['welch']:>10.6f}  "
                 f"{'VACUOUS (constrains nothing)' if r['vacuous'] else 'BINDS'}")
    w.append("")
    w.append("RANDOM UNIT VECTORS -- max_{i<j} |<u_i,u_j>| by Monte Carlo")
    w.append(f"  {'d':>5} {'k':>5} {'max':>10} {'95% CI':>24} {'x cited':>9}  0.1748 in CI")
    for r in [r for r in rs if r["kind"] == "mc_random_unit"]:
        w.append(f"  {r['d']:>5} {r['k']:>5} {r['max_mean']:>10.6f} "
                 f"[{r['max_ci_lo']:>10.6f},{r['max_ci_hi']:>10.6f}] "
                 f"{r['ratio_to_cited']:>8.3f}x  {r['cited_inside_ci']}")
    w.append("")
    w.append("THE ACTUAL CONSTRUCTION -- shipped top-k key DIRECTIONS")
    for r in [r for r in rs if r["kind"] == "real_key_directions"]:
        w.append(f"  d={r['d']} k={r['k']} draws={r['draws']}")
        w.append(f"    max  |cos| {r['max_mean']:.6f} "
                 f"[{r['max_ci_lo']:.6f},{r['max_ci_hi']:.6f}]")
        w.append(f"    mean |cos| {r['mean_mean']:.6f} "
                 f"[{r['mean_ci_lo']:.6f},{r['mean_ci_hi']:.6f}]")
    w.append("")
    w.append("THE SCRAMBLE CONTROL'S EXPECTED RESIDUAL -- key-norm units, NOT a cosine")
    for r in [r for r in rs if r["kind"] == "scramble_residual"]:
        w.append(f"  d={r['d']} k={r['k']} draws={r['draws']}")
        w.append(f"    residual   {r['residual']:.6f} "
                 f"[{r['ci_lo']:.6f},{r['ci_hi']:.6f}]")
        w.append(f"    key norms  mean {r['keynorm_mean']:.6f} sd {r['keynorm_sd']:.6f}"
                 f"  -> residual = {r['residual_in_sd']:.4f} sd")
    for r in [r for r in rs if r["kind"] == "scramble_residual_shipped"]:
        w.append(f"  shipped path (fgreen_matched.imbalance_table, one fixed scramble)")
        w.append(f"    pivot_random {r['residual']:.6f} "
                 f"[{r['ci_lo']:.6f},{r['ci_hi']:.6f}]   "
                 f"published {r['published']:.6f} at {r['source']}")
        w.append(f"    pivot_band   {r['band_residual']:.6f} "
                 f"[{r['band_ci_lo']:.6f},{r['band_ci_hi']:.6f}]   (the matched control)")
    w.append("")
    w.append("=" * 78)
    return "\n".join(w)


def demo() -> None:
    """Assert-based. Every assertion is a claim the report makes."""
    # -- 1. The instrument can fail, shown three ways before it is trusted. -----
    # (a) Welch is not vacuous everywhere: above the dimension it binds and is
    #     TIGHT. d=2, k=3 reads 0.5 and three unit vectors at 120 degrees attain
    #     |cos 120| = 0.5. A bound that were always 0 would pass every test in
    #     this file and mean nothing.
    assert abs(welch_bound(2, 3) - 0.5) < 1e-12
    ang = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    v = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    g = np.abs(v @ v.T)
    assert abs(g[np.triu_indices(3, 1)].max() - 0.5) < 1e-12

    # (b) The Monte Carlo is sensitive to the sampler it is fed. Drop the
    #     normalisation and the "coherence" is no longer a cosine: its mean must
    #     leave the CI the normalised sampler produces. If this passed, the
    #     estimator would be reading something other than direction.
    rng = np.random.default_rng(MC_SEED)
    bad_g = rng.standard_normal((2000, 8, 16))
    bad_pair = np.abs(np.einsum("nik,njk->nij", bad_g, bad_g))
    iu = np.triu_indices(8, 1)
    bad_mean = float(bad_pair[:, iu[0], iu[1]].mean())
    good = monte_carlo(16, 8, trials=2000, seed=MC_SEED)
    assert not (good["mean_ci_lo"] <= bad_mean <= good["mean_ci_hi"]), bad_mean

    # (c) The matcher answers zero only when it should. Identical pools give
    #     exactly 0; a shifted pool gives back the shift.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert matcher.match(a, a.copy()).residual == 0.0
    assert abs(matcher.match(a, a + 0.75).residual - 0.75) < 1e-12

    # -- 2. The cited cell reproduces at the (d, k) it was derived for. --------
    at256 = coherence_row(256, 16)
    assert at256["cited_inside_ci"], at256
    assert abs(at256["max_ci_lo"] - CITED_CI[0]) < 1e-4, at256
    assert abs(at256["max_ci_hi"] - CITED_CI[1]) < 1e-4, at256

    # -- 3. And it does NOT survive the move to the shipped dimension. --------
    assert m3.D_MODEL == 16 and m3.K_PIVOTS == 8, (m3.D_MODEL, m3.K_PIVOTS)
    here = coherence_row(m3.D_MODEL, m3.K_PIVOTS)
    assert not here["cited_inside_ci"], here
    assert here["max_ci_lo"] > at256["max_ci_hi"], (here, at256)   # disjoint
    assert here["ratio_to_cited"] > 3.0, here

    # -- 4. The 'k <= d' premise fails at two of the three shipped k. ---------
    assert welch_bound(m3.D_MODEL, 8) == 0.0
    assert welch_bound(m3.D_MODEL, 32) > 0.0
    assert welch_bound(m3.D_MODEL, 128) > 0.0

    # -- 5. The scramble control's residual, on the real construction. --------
    key = real_keys(n_draws=16)
    assert key.shape[-1] == 16, key.shape        # a key lives in R^16, not R^256
    sc = scramble_residual(key, m3.K_PIVOTS)
    assert sc["residual"] > 0.0                  # not zero, which is the rule's point
    assert sc["residual_in_sd"] > 1.0            # and larger than the spread it left

    # -- 6. And the shipped producer reproduces its own published cell. -------
    ship = shipped_scramble_residual()
    assert abs(ship["residual"] - ship["published"]) < 5e-7, ship
    # The two scramble paths agree: fresh-permutation expectation sits inside
    # the shipped fixed-permutation interval.
    full = scramble_residual(real_keys(), m3.K_PIVOTS)
    assert ship["ci_lo"] <= full["residual"] <= ship["ci_hi"], (ship, full)

    print(f"demo(): welch(16,8)=0, welch(16,32)={welch_bound(16, 32):.6f}")
    print(f"demo(): MC max d=256,k=16 {at256['max_mean']:.6f} vs "
          f"d=16,k=8 {here['max_mean']:.6f} ({here['ratio_to_cited']:.2f}x cited)")
    print(f"demo(): scramble residual (16 draws) {sc['residual']:.6f} "
          f"= {sc['residual_in_sd']:.3f} key-norm sd")
    print("demo(): all assertions passed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scale.r10_it20_coherence")
    ap.add_argument("--write", action="store_true",
                    help="write results/r10_it20_coherence.jsonl")
    a = ap.parse_args(argv)

    from scale.vram_gate import preflight
    fits, verdict = preflight(0, name="r10-it20-coherence", host_mib=512)
    print(verdict)
    if not fits:
        print("REFUSED by the gate; no run.")
        return 1

    rs = rows()
    print(table(rs))
    if a.write:
        out = ROOT / "results" / "r10_it20_coherence.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"wrote {len(rs)} rows to {out.relative_to(ROOT).as_posix()}")
    print()
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
