"""The COGS capability number and the content-conditional-sign theorem have
never been evaluated on a common object.

THE CONTRADICTION THIS FILE IS ABOUT. `CHECKLIST.md:229` and
`ceq/hf/modeling_ceq.py:165` carry one capability measurement: COGS
generalization exact match, softmax 0.0293 (15/512) against sgate 0.0000
(0/512), one-sided Fisher p = 2.7502788939e-05, behind in-distribution too
(0.9258 vs 0.7734), "identical steps / lr / batch / seed / data / eval
subsample". `README.md:892` carries the opposing claim: content-conditional
sign 0.1641 for sgate against softmax's exactly 0.0000. The first says the
operator has no capability; the second says it has one softmax provably cannot
have. Both are read as statements about the same two arms.

They are not. Two substitutions separate them, and each is checked below by
VALUE, never by structure -- nine structure-comparing instruments in this
project gave false readings and three value-comparing ones never did.

  1. THE SIGNED ARM. `ceq/capability.py` reaches `lm.TinyLM` through
     `harness.train_model` and never assigns `lm.RHO`, `lm.SGATE_LAM` or
     `lm.HOPS`, so the COGS run took the committed module globals
     `(0.9, 1.0, 3)` (`ceq/lm.py:29-30`). Every published sign-flip number and
     the 1.0334 parity ratio are at `(1.5, 0.10, 2)` -- the opposite end of the
     `lam` family. At `lam = 1.0` every row sum is exactly zero; at
     `lam = 0.10` every row carries `rho(1-lam)/(1+lam)` of positive mass.
     `DONE.md:3132` records the gap as an `ceq/lm.py` hygiene defect and marks
     it "NOT CORRECTED, deliberately"; nothing anywhere scopes the COGS
     capability number by it.

  2. THE SOFTMAX ARM. The exact 0.0000 that makes the theorem's exclusivity
     claim is measured on a stack with no nonlinearity between the attention
     layers (`ceq/bench.py:398-399`: the GELU is applied only when the arm name
     ends `_gelu`). Every block of the model `ceq.capability` trains has one
     (`ceq/lm.py:214`, `nn.Linear -> nn.GELU -> nn.Linear`), and it trains four.

CONTROLS. Three, and each one is a comparison whose answer is known in advance,
so a green here cannot be a blind instrument: the comparator must read equality
as exactly 0.0, must report a planted difference at exactly its planted size,
and the sign probe must read exactly 0.0 for the arm the theorem says is zero.

CPU only; nothing here allocates on a device.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ceq import bench, harness, lm

SEED = 0
PARITY_RHO, PARITY_LAM = 1.5, 0.10   # tests/chase/test_hub_package_hardening.py:50
PROBE = dict(n_draws=128, s=8, hops=3, seed=0)   # run_calib.py CASES geometry


def _q_k_as_capability_builds_them():
    """The (q, k) block 0 of the COGS model actually sees, on real COGS data.

    Built through `lm.TinyLM` and `harness._batch` rather than from `torch.randn`
    so that whatever `ceq.capability` puts into the operator is what is compared.
    No module global is touched anywhere in this function -- that is the point.
    """
    vocab = harness.build_vocab("cogs")
    pairs = harness.load_pairs("cogs", "train")
    model = lm.TinyLM("sgate", seq=harness.COGS_SEQ, vocab=len(vocab), seed=SEED)
    x, _ = harness._batch(pairs, vocab, list(range(4)), torch.device("cpu"),
                          harness.COGS_SEQ)
    blk = model.blocks[0]
    with torch.no_grad():
        h = model.tok(x) + model.pos(torch.arange(x.shape[1]))[None]
        q, k, _v = blk.attn.qkv_heads(blk.n1(h))
    return blk.attn, q, k


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """The only comparator in this file. One number, no threshold inside it."""
    return float((a - b).abs().max())


# --------------------------------------------------------------- the controls

def test_control_the_comparator_reads_equality_as_exactly_zero():
    """MUST-FIRE CONTROL 1, in the passing direction.

    The same comparator, against the operator built from the globals that are
    actually committed, must return EXACTLY 0.0. If this is not 0.0 the RED
    below means nothing, because the comparator would be reporting a difference
    that is not there.
    """
    attn, q, k = _q_k_as_capability_builds_them()
    with torch.no_grad():
        a_run = attn.operator(q, k)
        a_committed = bench._causal_sgate_operator(q, k, rho=lm.RHO,
                                                   lam=lm.SGATE_LAM)
    assert _max_abs_diff(a_run, a_committed) == 0.0


def test_control_the_comparator_reports_a_planted_difference_at_its_exact_size():
    """MUST-FIRE CONTROL 2. A gate never seen failing is not a gate.

    0.25 is planted into one entry and the comparator must return 0.25 exactly,
    not "greater than zero" -- an inequality here would pass for a comparator
    that reports any nonzero constant, which is the `> 1e-3` defect that let a
    fabricated magnitude survive fourteen tests in this project. 0.25 is a
    binary fraction, so `x + 0.25 - x` is exact in float32 at this magnitude and
    the pin can be an equality rather than a tolerance.
    """
    attn, q, k = _q_k_as_capability_builds_them()
    with torch.no_grad():
        a = attn.operator(q, k)
        b = a.clone()
        b[0, 0, 4, 1] = b[0, 0, 4, 1] + 0.25
    assert _max_abs_diff(a, b) == 0.25


def test_control_the_sign_probe_reads_exactly_zero_where_the_theorem_says_zero():
    """MUST-FIRE CONTROL 3, for the theorem half.

    Two softmax layers with NOTHING between them. The theorem's claim is that
    this is exactly 0.0, and the probe must say so, otherwise the nonzero in
    `test_the_theorem_zero_survives_the_mlp_...` is probe noise rather than the
    nonlinearity.
    """
    assert bench.sign_flip_rate("softmax", depth=2,
                                device=torch.device("cpu"), **PROBE) == 0.0


# --------------------------------------------------------------------- the RED

def test_the_cogs_capability_arm_is_the_operator_every_published_number_is_about():
    """RED. The signed arm `ceq.capability` trains on COGS, entry by entry,
    against `bench._causal_sgate_operator` at the campaign's shipped point.

    `scale/m3_capability.py:79` and `bench.sign_flip_draws`'s own defaults both
    pin that point at `rho = 1.5, lam = 0.10`. If the COGS arm is a different
    matrix, the 0.0000 it scored is not a reading of the operator the 0.1641
    sign-flip rate and the 1.0334 parity ratio are readings of.
    """
    attn, q, k = _q_k_as_capability_builds_them()
    with torch.no_grad():
        a_run = attn.operator(q, k)
        a_parity = bench._causal_sgate_operator(q, k, rho=PARITY_RHO,
                                                lam=PARITY_LAM)
    d = _max_abs_diff(a_run, a_parity)
    assert d == 0.0, (
        f"the COGS capability run trained a DIFFERENT operator: "
        f"max|A_run - A_parity| = {d!r}; "
        f"lm globals in force = (RHO {lm.RHO}, SGATE_LAM {lm.SGATE_LAM}, "
        f"HOPS {lm.HOPS}), parity point = ({PARITY_RHO}, {PARITY_LAM}, 2)")


def test_the_benchmarked_operator_carries_the_positive_row_mass_the_published_one_does():
    """RED, and the structural half of the same substitution.

    At the parity `lam = 0.10` every row with at least one visible key carries
    `rho(1-lam)/(1+lam)` of positive mass. At the committed `lam = 1.0` the two
    softmax halves cancel and every row sum is exactly zero -- so the operator
    that ran on COGS annihilates the constant vector, and is identically zero on
    row 1, where only one key is visible and both halves put all of their mass
    on it.

    The expected value is read off the parity operator in the same dtype rather
    than hardcoded, so this compares two measurements, not a measurement against
    a literal.
    """
    attn, q, k = _q_k_as_capability_builds_them()
    with torch.no_grad():
        a_run = attn.operator(q, k)
        run_rows = a_run[0, 0].sum(-1)
        parity_rows = bench._causal_sgate_operator(
            q, k, rho=PARITY_RHO, lam=PARITY_LAM)[0, 0].sum(-1)
        row1_mass = float(a_run[0, 0, 1].abs().sum())
    assert _max_abs_diff(run_rows, parity_rows) == 0.0, (
        f"row sums differ: as-run rows 1..4 = "
        f"{[float(v) for v in run_rows[1:5]]!r}, "
        f"parity rows 1..4 = {[float(v) for v in parity_rows[1:5]]!r}; "
        f"as-run row-1 L1 mass = {row1_mass!r}")


def test_the_hops_the_campaign_is_about_carry_signal_in_the_benchmarked_arm():
    """RED, and the quantitative mechanism behind the other two.

    The whole construction is `out = v + Av + A^2 v + ...`; the theorem is a
    statement about `I + A + A^2`. This measures what each of those terms is
    worth, as a fraction of `||v||`, in the arm that was actually trained on
    COGS -- block 0, real COGS token ids, `lm.TinyLM` at seed 0, nothing
    rebound. The parity operator on the SAME q, k, v is the comparison, so the
    only difference between the two columns is `(rho, lam)`.
    """
    attn, q, k = _q_k_as_capability_builds_them()
    vocab = harness.build_vocab("cogs")
    pairs = harness.load_pairs("cogs", "train")
    model = lm.TinyLM("sgate", seq=harness.COGS_SEQ, vocab=len(vocab), seed=SEED)
    x, _ = harness._batch(pairs, vocab, list(range(4)), torch.device("cpu"),
                          harness.COGS_SEQ)
    with torch.no_grad():
        h = model.tok(x) + model.pos(torch.arange(x.shape[1]))[None]
        _q, _k, v = model.blocks[0].attn.qkv_heads(model.blocks[0].n1(h))
        nv = float(v.norm())

        def hop_mass(a):
            t, out = v, []
            for _ in range(3):
                t = a @ t
                out.append(float(t.norm()) / nv)
            return out

        run = hop_mass(attn.operator(q, k))
        parity = hop_mass(bench._causal_sgate_operator(q, k, rho=PARITY_RHO,
                                                       lam=PARITY_LAM))
    assert run == parity, (
        f"hop mass ||A^n v||/||v|| for n = 1,2,3 -- as-run {run!r} vs "
        f"parity {parity!r}; hop-2 ratio parity/as-run = "
        f"{parity[1] / run[1]!r}")


def test_the_theorem_zero_survives_the_mlp_the_benchmarked_model_has():
    """RED. The exclusivity claim, measured on the architecture that was
    benchmarked instead of on the probe's MLP-free stack.

    `softmax_gelu` is the same softmax operator with one GELU between the layers
    and nothing else changed, so if this is not 0.0 then the softmax arm of the
    COGS run has the property the theorem reserves for signed operators, and
    "softmax cannot represent it" is a statement about the probe, not about the
    benchmarked model.
    """
    r = bench.sign_flip_rate("softmax_gelu", depth=2,
                             device=torch.device("cpu"), **PROBE)
    assert r == 0.0, (
        f"one GELU between two softmax layers reads {r!r} on the same probe "
        f"that reads exactly 0.0 without it")
