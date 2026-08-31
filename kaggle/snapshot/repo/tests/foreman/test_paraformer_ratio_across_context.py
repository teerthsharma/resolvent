"""FOREMAN round 4 -- bind the ParaFormer separation across context length.

Round 3 published a single number: sgate **0.1230** against ParaFormer
**0.000000** on 2048 draws, exact Clopper-Pearson 95% ceiling 0.00146, ">= 84x".
It was measured at ONE context length and reported without an interval on either
side. Both of those are fixed here.

Before any of that: the arm itself is audited. `ceq/bench.py::sign_flip_rate`
drew a `gam` vector for the `paraformer` arm and then never referenced it -- the
arm fell through to `else: a = _softmax_operator(...)` followed by `h = a @ h`,
which is plain single-hop softmax. Every ParaFormer number in the repository was
therefore a softmax number. `test_the_paraformer_arm_is_not_the_softmax_arm`
pins that.

Every test parametrizes over cpu and cuda, skipping cuda when absent.
"""
from __future__ import annotations

import pytest
import torch

from ceq import bench

from _device import DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_the_paraformer_arm_is_not_the_softmax_arm(device):
    """ParaFormer keeping only its k=0 term is `Z = gamma_0 V`, blind to x.

    `hops=0` draws no gammas, so the `paraformer` and `softmax` arms consume the
    IDENTICAL random stream and the two rates are comparable draw for draw. On
    the INPUT path softmax has a live, nonzero rate; ParaFormer truncated to
    `k=0` propagates `V` unchanged, so `d(out_i)/d(x_j)` is exactly zero and no
    draw can flip. A `paraformer` reading equal to the `softmax` reading means
    the hop coefficients are being ignored.
    """
    dev = torch.device(device)
    kw = dict(wrt="x", hops=0, s=8, n_draws=256, seed=0, device=dev)
    softmax = bench.sign_flip_rate("softmax", **kw)
    paraformer = bench.sign_flip_rate("paraformer", **kw)
    assert softmax > 0.0, f"control is dead: softmax input-path rate {softmax}"
    assert paraformer == 0.0, (
        f"paraformer at hops=0 reads {paraformer}, softmax reads {softmax}; "
        "the k=0 term cannot see x, so the arm is not applying its coefficients"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_a_bounded_receptive_field_holds_the_separation_that_global_attention_dilutes(device):
    """The 1/s dilution is a property of a GLOBAL row, not of the signed matrix.

    A third token `c` reaches the pair `(i, j)` only through the `k >= 2` term of
    the path sum, where it is one of the intermediates the sum runs over. Under
    a global causal row that count is ~s, so its share is 1/s. Under a sliding
    window of width `w` -- the standard bounded-receptive-field attention of
    Mistral, Gemma and gpt-oss -- the same count is ~w, which does not move when
    the sequence grows.

    So: hold `w` and the relative offsets of `(i, j, c)` inside the window fixed,
    grow `s` by 32x, and the signed matrix must keep its rate. The unwindowed
    control on the same draws must lose it.
    """
    dev = torch.device(device)
    w = 8
    def at(s, window):
        return bench.sign_flip_rate(
            "sgate", n_draws=512, s=s, i=s - 1, j=s - 1 - w // 2,
            c=s - 1 - w // 4, window=window, seed=0, device=dev)

    windowed = [at(s, w) for s in (32, 128, 1024)]
    globalr = [at(s, 0) for s in (32, 128, 1024)]

    assert globalr[0] > 4 * globalr[-1], (
        f"control is dead: the global row did not dilute, {globalr}")
    assert windowed[-1] > 0.5 * windowed[0], (
        f"windowed rate collapsed with context anyway: {windowed}")
    assert windowed[-1] > 10 * globalr[-1], (
        f"at s=1024 windowed {windowed[-1]} is not clear of global {globalr[-1]}")


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_signed_content_dependent_path_sum_is_not_already_deltanet(device):
    """RED first. The residual claim against the construction that ships.

    DeltaNet's chunkwise form (arXiv:2406.06484, eq:inverse) is

        T = (I + tril(diag(beta) K K^T, -1))^-1 diag(beta)

    -- a strictly-lower-triangular matrix built from K and beta, whose entries
    `-beta_i (k_i . k_j)` carry a sign per entry with no non-negativity anywhere,
    inverted exactly. Because it is strictly triangular it is nilpotent, so that
    inverse IS a finite path sum `sum_k (-M)^k`. Content-dependent, signed,
    multi-hop, and shipping in `fla/ops/utils/solve_tril.py` and sglang's
    `chunk_kda_fwd_kernel_inter_solve_fused`.

    If it reads the same content-conditional sign rate as `sgate` on this probe,
    the residual claim has no cell left to occupy.

    The `deltanet` arm draws its own `beta` before `x0`/`v0`, exactly as the
    `paraformer` arm draws its gammas, so that no other arm's random stream --
    and no previously published number -- moves. The consequence is that this is
    an UNPAIRED comparison: both arms see 2048 iid draws from the same
    distribution, not the same 2048 draws.
    """
    dev = torch.device(device)
    n = 2048
    kw = dict(depth=1, hops=2, n_draws=n, s=8, i=7, j=1, c=4, seed=0, device=dev)
    delta = bench.sign_flip_rate("deltanet", **kw)
    sgate = bench.sign_flip_rate("sgate", **kw)
    assert delta < sgate / 10.0, (
        f"DeltaNet's signed content-dependent triangular path sum reads "
        f"{delta} against sgate's {sgate}, 2048 draws each. Signed + "
        f"content-dependent + multi-hop is occupied by a shipping kernel.")


def _cp(k: int, n: int, conf: float = 0.95):
    """Exact Clopper-Pearson interval. `bench.zero_success_upper_bound` is the
    k = 0 slice of this; the general case needs the Beta quantile."""
    from scipy.stats import beta
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_separation_from_signed_hop_coefficients_survives_context(device):
    """RED first. The 84x, restated as something a reviewer can check.

    Round 3 published "at least 84x" from a single context length and no
    interval on either side. Both rates are binomial, so the honest statement of
    a separation is: the exact Clopper-Pearson 95% LOWER bound on the signed
    matrix clears the exact 95% UPPER bound on the signed coefficients. At s = 8
    it does. This asserts it still does at s = 128 -- a context length below
    anything a language model runs at.

    KILL CONDITION: if the two intervals overlap at s = 128, then 2048 draws
    cannot tell a signed base matrix from signed hop coefficients at any context
    a 300M model uses, and the residual novelty claim has no measurement behind
    it at the only scale that matters.
    """
    dev = torch.device(device)
    n = 2048
    def band(kind, s):
        r = bench.sign_flip_rate(kind, hops=2, n_draws=n, s=s, i=s - 1,
                                 j=s // 4, c=s // 2, seed=0, device=dev)
        return _cp(round(r * n), n)

    near = (band("sgate", 8), band("paraformer", 8))
    far = (band("sgate", 128), band("paraformer", 128))
    assert near[0][0] > near[1][1], f"no separation even at s=8: {near}"
    assert far[0][0] > far[1][1], (
        f"at s=128 sgate CP95={far[0]} and paraformer CP95={far[1]} overlap; "
        f"2048 draws cannot separate a signed base matrix from signed hop "
        f"coefficients at that context, and 128 is short.")


@pytest.mark.parametrize("device", DEVICES)
def test_the_round_4_arms_reproduce_their_published_counts(device):
    """Calibration for round 4, the way `n_draws=128` was calibrated for round 3.

    Every number in README's replacement table comes off these two calls. If the
    probe drifts, this fails before any of the prose is believed. Counts are
    exact on cpu; cuda differs by matmul reduction order, so the assertion is a
    band wide enough to survive that and narrow enough to catch an arm that
    stopped being itself -- which is exactly the failure round 3 shipped.
    """
    dev = torch.device(device)
    n = 2048
    got = {k: bench.sign_flip_rate(k, hops=2, n_draws=n, s=8, i=7, j=2, c=4,
                                   seed=0, device=dev)
           for k in ("sgate", "deltanet", "signmag", "paraformer", "softmax")}
    cpu_counts = {"sgate": 340, "deltanet": 168, "signmag": 216,
                  "paraformer": 61, "softmax": 0}
    assert got["softmax"] == 0.0, f"softmax must be exactly 0, got {got['softmax']}"
    for k, want in cpu_counts.items():
        if k == "softmax":
            continue
        assert got[k] * n == pytest.approx(want, rel=0.15), (
            f"{k} reads {got[k] * n:.0f}/{n}, published {want}/{n}: {got}")
    assert got["deltanet"] > got["paraformer"], (
        f"the ordering that kills the claim is gone: {got}")


@pytest.mark.parametrize("device", DEVICES)
def test_claim_the_context_decay_is_the_path_count_and_not_the_row_normalizer(device):
    """RED first. Round 3's mechanism sentence, isolated.

    Round 3: "the two-hop term sums over ~s intermediates of which c is one, so
    its share is 1/s BY CONSTRUCTION". That sweep moved `j = s/4` and `c = s/2`,
    so the number of intermediates between `j` and `i` grew with `s` at the same
    time as every row normalizer did. Two channels, one measurement.

    Pin the offsets instead -- `j = i-4`, `c = i-2` -- so the path count between
    `j` and `i` is a constant 3 at every context length. If the 1/s share is the
    cause, every arm must now be FLAT in `s`. DeltaNet, whose entries
    `-beta_i (k_i . k_j)` carry no row normalizer at all, is the control.
    """
    dev = torch.device(device)
    def at(kind, s):
        return bench.sign_flip_rate(kind, hops=2, n_draws=1024, s=s, i=s - 1,
                                    j=s - 5, c=s - 3, seed=0, device=dev)

    delta = [at("deltanet", s) for s in (32, 512)]
    sgate = [at("sgate", s) for s in (32, 512)]
    assert delta[1] > 0.5 * delta[0], (
        f"control is dead: the normalizer-free arm decayed too, {delta}")
    assert sgate[1] > 0.5 * sgate[0], (
        f"at a FIXED path count the row-normalized arm still fell "
        f"{sgate[0]} -> {sgate[1]} while the normalizer-free arm held "
        f"{delta[0]} -> {delta[1]}. The decay is the softmax row normalizer "
        f"summing over s tokens, not the two-hop share being 1/s.")
