"""Pre-registered claims from the M2 docstring (scale/pivot_probe.py), tested
directly: pivots are CONTENT-selected, not position- or noise-selected.

TEST 1 asks whether the operator built in ceq/bench.py::_causal_tgate_operator
even sees the score select_pivots uses (key.norm). TEST 2 asks whether that
content-based choice does anything a content-BLIND (random) choice of the same
size would not. TEST 3 asks whether the selected pivots are geometrically
special (mutually redundant/collinear) relative to a random token set, once
key direction is separated from key norm by L2-normalisation.

Imports the EXISTING probe and attack modules (scale/pivot_probe.py,
scale/mech_attack.py, ceq/bench.py) rather than reimplementing any operator.
"""
from __future__ import annotations

import math
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ceq import bench
from scale.pivot_probe import select_pivots, clopper_pearson, run_arm
from scale.mech_attack import draws


def test_operator_sees_the_key_norm():
    """A1 = _causal_tgate_operator(q, k, g, tau=1.0) at s=32, d=16. Rescale ONE
    token's key by a positive constant (k2[7] = k[7] * 7.5) and rebuild A2.
    The docstring's pivot score is `key.norm(dim=-1)`; if the operator were
    blind to that quantity, this rescale could not move any entry of A.
    """
    s, d = 32, 16
    gen = torch.Generator().manual_seed(0)
    rnd = lambda *sh: torch.randn(*sh, generator=gen)

    q = rnd(s, d)
    k = rnd(s, d)
    gvec = torch.sigmoid(rnd(s))

    A1 = bench._causal_tgate_operator(q, k, gvec, tau=1.0)

    before_norm = float(k[7].norm())
    k2 = k.clone()
    k2[7] = k[7] * 7.5
    after_norm = float(k2[7].norm())

    A2 = bench._causal_tgate_operator(q, k2, gvec, tau=1.0)

    diff = A2 - A1
    max_abs_diff = float(diff.abs().max())
    col7_max_abs_diff = float(diff[:, 7].abs().max())

    print(f"token 7 key norm: before={before_norm:.6f} after={after_norm:.6f} "
          f"(ratio={after_norm / before_norm:.4f})")
    print(f"max abs diff (A2-A1), all entries: {max_abs_diff:.6e}")
    print(f"max abs diff (A2-A1), column 7 only: {col7_max_abs_diff:.6e}")

    assert max_abs_diff > 1e-6, (
        f"rescaling token 7's key by 7.5x did not move the operator: "
        f"max|A2-A1|={max_abs_diff:.6e} (column 7 max|A2-A1|="
        f"{col7_max_abs_diff:.6e}), key norm before={before_norm:.6f} "
        f"after={after_norm:.6f}"
    )


def _make_random_pivots(seed: int):
    """Content-blind pivot chooser matching the `pivots_fn(kk0, k, exclude) ->
    LongTensor` contract: k indices drawn uniformly from range(1, s-1) minus
    `exclude`, off a PRIVATE torch.Generator seeded for reproducibility. Never
    touches the caller's generator, so plugging this into draws() cannot
    perturb draws()'s own RNG stream.
    """
    gen = torch.Generator().manual_seed(seed)

    def _choose(kk0, k, exclude=()):
        s = kk0.shape[0]
        pool = [t for t in range(1, s - 1) if t not in exclude]
        perm = torch.randperm(len(pool), generator=gen)
        chosen = [pool[i] for i in perm[:k].tolist()]
        return torch.tensor(chosen, dtype=torch.long)

    return _choose


def _flip_rate_from_draws(rec, floor: float = 1e-6):
    flips = sum(
        1 for r in rec
        if (r["a_ij"] + r["bg"] + r["t1"]) * (r["a_ij"] + r["bg"] + r["t2"]) < 0
        and min(abs(r["lo"]), abs(r["hi"])) > floor
    )
    return flips, len(rec)


def test_content_selection_beats_random_selection():
    """scale.mech_attack.draws did not support a custom pivot chooser, so
    scale/mech_attack.py was edited to add `pivots_fn=None` (see reported
    diff). `random_pivots` is the content-blind chooser required by the task:
    it ignores kk0 entirely and returns k uniformly random indices off its own
    seeded Generator.

    Proof step first: with pivots_fn=None, draws() must still reproduce
    run_arm's flip rate exactly (run_arm is untouched and always uses
    select_pivots directly), which is the same equality mech_attack.main()'s
    own CHECK 3 already establishes -- run at a cheap n_draws=64 here purely
    to keep this proof fast; the edit made to the shared `pivots` line does
    not depend on n_draws.

    Then the real measurement: FULL flip rate at s=128, k=8, n_draws=4096,
    seed=0, placement="in_P", protocol="SCALING", content selector vs random.
    `fast=True` skips the autograd call and reconstructs lo/hi analytically
    from (a_ij + bg + t) * wo_sum -- exactly what mech_attack.main() CHECK 2
    proves lo/hi equal, and what its CHECK 4 proves gives the identical flip
    rate to the autograd path -- used here only to keep the 4096-draw sweep
    inside the CPU time budget; it does not change which quantities are
    compared.
    """
    kind, s, k, protocol, placement = "pivot_signed", 128, 8, "SCALING", "in_P"

    # ---- proof: pivots_fn=None reproduces the pre-edit path exactly ----
    proof_n = 64
    proof_rec = draws(kind, s, n_draws=proof_n, k=k, placement=placement,
                       protocol=protocol, seed=0)
    proof_flips, proof_used = _flip_rate_from_draws(proof_rec)
    proof_rate = proof_flips / proof_used
    proof_ref = run_arm(kind, s, n_draws=proof_n, k=k, placement=placement,
                         protocol=protocol, seed=0)
    print(f"PROOF (pivots_fn=None == pre-edit run_arm) @ n_draws={proof_n}: "
          f"draws()-recomputed rate={proof_rate!r} (k={proof_flips}, "
          f"n={proof_used})  run_arm rate={proof_ref['rate']!r} "
          f"(k={proof_ref['k']}, n={proof_ref['n']})")
    assert proof_rate == proof_ref["rate"] and proof_used == proof_ref["n"], (
        f"pivots_fn=None DIVERGED from pre-edit run_arm: "
        f"draws()-recomputed rate={proof_rate!r} n={proof_used}  "
        f"run_arm rate={proof_ref['rate']!r} n={proof_ref['n']}"
    )

    # ---- the real measurement ----
    n_draws = 4096
    rec_content = draws(kind, s, n_draws=n_draws, k=k, placement=placement,
                        protocol=protocol, seed=0, fast=True)
    k_content, n_content = _flip_rate_from_draws(rec_content)
    r_content = k_content / n_content

    random_pivots = _make_random_pivots(seed=12345)
    rec_random = draws(kind, s, n_draws=n_draws, k=k, placement=placement,
                       protocol=protocol, seed=0, fast=True,
                       pivots_fn=random_pivots)
    k_random, n_random = _flip_rate_from_draws(rec_random)
    r_random = k_random / n_random

    lo_c, hi_c = clopper_pearson(k_content, n_content)
    lo_r, hi_r = clopper_pearson(k_random, n_random)
    rel_diff = abs(r_content - r_random) / r_content if r_content else float("inf")

    print(f"CONTENT selector: rate={r_content!r} k={k_content} n={n_content} "
          f"CP95%=[{lo_c:.6f},{hi_c:.6f}]")
    print(f"RANDOM  selector: rate={r_random!r} k={k_random} n={n_random} "
          f"CP95%=[{lo_r:.6f},{hi_r:.6f}]")
    print(f"relative difference |r_content-r_random|/r_content = {rel_diff!r}")

    assert rel_diff > 0.25, (
        f"content vs random pivot selection did not differ by >25% relative: "
        f"r_content={r_content!r} (k={k_content}, n={n_content}, "
        f"CP95%=[{lo_c:.6f},{hi_c:.6f}])  "
        f"r_random={r_random!r} (k={k_random}, n={n_random}, "
        f"CP95%=[{lo_r:.6f},{hi_r:.6f}])  rel_diff={rel_diff!r}"
    )


def test_selected_pivot_keys_are_not_special_after_normalisation():
    """At s=512, d=16: draw x0, wk the same way the probe does (x0 = randn(s,d),
    wk = randn(d,d), kk0 = x0 @ wk), take pivots = select_pivots(kk0, 8,
    exclude=(511, 128)). Compare the selected pivots against 8 uniformly
    random tokens (drawn from the same eligible pool, i.e. excluding 511 and
    128 too, for an apples-to-apples population) on:

      * mean key NORM (expected to differ a lot -- that is the selector's
        entire mechanism)
      * mean pairwise |cosine similarity| of the L2-NORMALISED key (khat),
        i.e. whether the selected tokens' DIRECTIONS are more mutually
        redundant than a random token's directions would be

    Repeated over 32 independent draws; reports mean +/- std for all four
    statistics and a pooled-SE z for the cosine comparison.
    """
    s, d, k = 512, 16, 8
    excl = (511, 128)
    n_reps = 32
    gen = torch.Generator().manual_seed(0)
    rnd = lambda *sh: torch.randn(*sh, generator=gen)

    pivot_norms, all_norms, pivot_cos, rand_cos = [], [], [], []
    pool_all = [t for t in range(s) if t not in excl]

    for _ in range(n_reps):
        x0 = rnd(s, d)
        wk = rnd(d, d)
        kk0 = x0 @ wk
        pivots = select_pivots(kk0, k, exclude=excl)

        perm = torch.randperm(len(pool_all), generator=gen)
        rand_idx = torch.tensor([pool_all[i] for i in perm[:k].tolist()],
                                dtype=torch.long)

        pivot_norms.append(float(kk0[pivots].norm(dim=-1).mean()))
        all_norms.append(float(kk0.norm(dim=-1).mean()))

        khat = torch.nn.functional.normalize(kk0, dim=-1)

        def mean_abs_cos(idx):
            kh = khat[idx]
            sim = (kh @ kh.T).abs()
            iu = torch.triu_indices(len(idx), len(idx), offset=1)
            return float(sim[iu[0], iu[1]].mean())

        pivot_cos.append(mean_abs_cos(pivots))
        rand_cos.append(mean_abs_cos(rand_idx))

    def mstd(xs):
        return statistics.mean(xs), statistics.stdev(xs)

    mp_norm, sp_norm = mstd(pivot_norms)
    ma_norm, sa_norm = mstd(all_norms)
    mp_cos, sp_cos = mstd(pivot_cos)
    mr_cos, sr_cos = mstd(rand_cos)

    se_p = sp_cos / math.sqrt(n_reps)
    se_r = sr_cos / math.sqrt(n_reps)
    pooled_se = math.sqrt(se_p ** 2 + se_r ** 2)
    z = (mp_cos - mr_cos) / pooled_se if pooled_se > 0 else float("inf")

    print(f"over {n_reps} draws (s={s}, d={d}, k={k}, exclude={excl}):")
    print(f"  mean key norm, selected pivots: {mp_norm:.6f} +/- {sp_norm:.6f}")
    print(f"  mean key norm, all tokens:      {ma_norm:.6f} +/- {sa_norm:.6f}")
    print(f"  mean |cosine|, selected pivots: {mp_cos:.6f} +/- {sp_cos:.6f}")
    print(f"  mean |cosine|, random tokens:   {mr_cos:.6f} +/- {sr_cos:.6f}")
    print(f"  pooled SE={pooled_se:.6f}  z=(pivot-random)/pooled_SE={z:.6f}")

    assert z > 3.0, (
        f"pivot cosine did not exceed random-token cosine by >3 pooled SE: "
        f"mean|cos|(pivots)={mp_cos:.6f}+/-{sp_cos:.6f}  "
        f"mean|cos|(random)={mr_cos:.6f}+/-{sr_cos:.6f}  z={z:.6f}  "
        f"(key norm pivots={mp_norm:.6f}+/-{sp_norm:.6f} "
        f"all={ma_norm:.6f}+/-{sa_norm:.6f})"
    )
