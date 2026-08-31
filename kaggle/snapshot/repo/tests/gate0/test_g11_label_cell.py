"""GATE-0 item G1.1 -- the two defects G0.1 filed and was forbidden to fix.

    D1  `ceq/arm_smprime.py::label_cell` scores EVERY position against
        `ceq/arm_phase.py::chain_label`, whose loop starts at `i = 1` and so
        reserves slot 0 as the chain's drive-free initial state. The (L)
        read-out at slot 0 is `G_00 V_0 = V_0` -- the EMPTY path product. On a
        caller whose slot 0 carries a drive the residual is `max |b_0|` and
        nothing else, with honest settings.

    D2  `V16_ARM_SMPRIME.md` row (e) certifies the softmax corner with
        `Sigma_j |W_ij| = 1.000000` at `beta = 1`. That is an identity for
        every gate whatsoever; it sees `beta` and cannot see `g`.

NOTHING IS INHERITED. `V16_ARM_SMPRIME.md` (a) and (e) and `V17_G01_IDENTITY.md`
are cited only so a reader can diff them against what this file MEASURES. Every
number below is recomputed here.

NO NEW CONSTRUCTIONS (the round's first law). No arm, no oracle, no comparator
of its own: `label_cell`, `readout`, `operator`, `mutate`, `oracle_heads` and
`path_product` are `ceq/arm_smprime.py`'s as shipped, and the corpus is
`scale/negation_scope.py::make_equilibrium_batch` called as shipped.

NOTHING TRAINS (L-LEAN). No gradient, no cell, no NRMSE, no verdict.

THE ROW-(E) SWITCH, AND WHY IT EXISTS. `G11_ROW_E=1` reads the softmax corner
with row (e)'s statistic instead of the replacement. Under it the corner test
goes RED, because the statistic cannot separate the configuration the test
plants. That RED is D2's evidence: the vacuity is shown by making the check
depend on the vacuous quantity and watching the planted negative walk through.

    $ G11_ROW_E=1 python -m pytest tests/gate0/test_g11_label_cell.py -q   # RED
    $ python -m pytest tests/gate0/test_g11_label_cell.py -q -s            # real
"""
from __future__ import annotations

import math
import os

import pytest
import torch

from ceq import arm_smprime as smp
from ceq.arm_phase import chain_label
from scale import negation_scope as ns

DT = torch.float64
CT = torch.complex128

#: G0.1's own bar for the (L) bind, and BIND 2's "1.000000" = six places.
LABEL_BAR = 1e-6
ROW_BAR = 5e-7

#: BED-M's measured gate support, re-asserted on every draw rather than assumed.
BEDM_SUPPORT = (-1.0, 0.0, 1.0)

#: the deliberate blunting -- never on in a reported run.
ROW_E = os.environ.get("G11_ROW_E") == "1"


def _bedm(n: int, s: int, d: int = 24, t_star: int | None = 2, seed: int = 0):
    """BED-M `e3_t2` at float64: `(gates, drives, bed-oracle label)`.

    `t_star=None` is the FULL-LENGTH chain, where `head = 0` and the gate at
    position 1 is live. That setting is what separates the two candidate
    repairs, so it is a parameter here rather than a constant.
    """
    x, _y32, head, p = ns.make_equilibrium_batch(n, s, d, t_star=t_star,
                                                 d_model=16, seed=seed)
    x = x.double()
    return (x[:, :, ns.CH_DRIVE], x[:, :, ns.CH_FLIP],
            ns.equilibrium_oracle(x, head, p))


def _readout(a: torch.Tensor, v: torch.Tensor, u, th, *, beta=0.0,
             route="product") -> torch.Tensor:
    """The (L) read-out at EVERY position, `ceq/arm_smprime.py::readout` as
    shipped. `beta = 0`, QK off, `V` unrescaled -- `#5a`'s path-product corner.
    """
    zero = torch.zeros(*a.shape[:-1], a.shape[-1], 1, dtype=DT, device=a.device)
    return smp.readout(zero, zero, v.unsqueeze(-1), u, th,
                       beta=beta, qk=0.0, route=route)[..., 0]


# ============================================================== D1 (the BOS slot)

def test_the_real_corpus_residual_reads_the_hops_and_not_max_abs_b_zero():
    """THE RED. `label_cell` on BED-M's real corpus with HONEST settings.

    Before the repair this reads `1.440495` -- bit-for-bit `max |b_0|`, because
    `make_equilibrium_batch` zeroes `b[s-1]` and not `b[0]`, so slot 0 carries a
    drive that `chain_label`'s `y_0 = 0` convention does not. After the repair
    the number is the hops' own, and the excluded term is REPORTED rather than
    deleted.
    """
    a, b, y = _bedm(8, 64)

    #: L-DOM: the census before the residual. A draw off `{-1,0,+1}` is a
    #: different bind, and a degenerate label is a void PASS.
    assert sorted(float(v) for v in torch.unique(a)) == list(BEDM_SUPPORT)
    assert int((a == 0).sum()) > 0, "no annihilating gate -- clause 4 untested"
    assert float(y.std()) > 0.1, float(y.std())
    b0 = float(b[:, 0].abs().max())
    assert b0 > 1.0, b0            # slot 0 really does carry a drive

    rec = smp.label_cell(a, b, mutation="none")
    print(f"  D1  real corpus n=8 s=64  residual = {rec['residual']:.6e}"
          f"   residual_bos = {rec['residual_bos']:.6e}   max|b_0| = {b0:.6e}")

    #: the repair: the headline number is a reading of the hops.
    assert rec["residual"] <= LABEL_BAR, rec["residual"]
    #: and it is NOT the drive at slot 0 wearing a residual's name.
    assert abs(rec["residual"] - b0) > 1.0, (rec["residual"], b0)
    #: the excluded term survives as its own column, at its measured value.
    assert abs(rec["residual_bos"] - b0) <= 1e-12, (rec["residual_bos"], b0)


def test_the_repaired_residual_fires_on_a_planted_gate_perturbation():
    """NON-DEGENERACY OF THE PASS HALF. A residual that dropped a position
    could have dropped the bind's teeth with it. The same scorer, on gates
    moved off the oracle by a known amount, with the corpus's label held fixed.
    """
    a, b, _y = _bedm(8, 64)
    ladder = {}
    for eps in (1e-12, 1e-9, 1e-6, 1e-3):
        ap = a.clone()
        ap[:, -1] = ap[:, -1] + eps        # the last hop, which enters t*=2
        #: the LABEL is the honest gates', the READ-OUT the perturbed gates':
        #: a perturbation of the arm against a fixed truth.
        ladder[eps] = float((_readout(ap, b.to(CT), *smp.oracle_heads(ap))[..., 1:]
                             - chain_label(a, b)[..., 1:]).abs().max())
    print("  D1  perturbation ladder (gate + eps, label held): "
          + "  ".join(f"{e:.0e}->{v:.6e}" for e, v in ladder.items()))
    assert ladder[1e-12] <= LABEL_BAR, ladder
    assert not (ladder[1e-3] <= LABEL_BAR), ladder
    assert ladder[1e-3] > ladder[1e-9] > ladder[1e-12], ladder


@pytest.mark.parametrize("mutation", [m for m in smp.MUTATIONS if m != "none"])
def test_the_planted_negatives_still_fire_after_the_repair(mutation):
    """NON-DEGENERACY OF THE PASS HALF, second form. `ceq/arm_smprime.py`
    ships four mutilations of the (L) setting; dropping a position from the
    residual must not have vacated the rejection region. Read on the REAL
    corpus, which is where the defect lived.
    """
    a, b, _y = _bedm(8, 64)
    honest = smp.label_cell(a, b, mutation="none")["residual"]
    r = smp.label_cell(a, b, mutation=mutation)["residual"]
    print(f"  D1  negative {mutation:16s} residual = {r:.6e}"
          f"   (honest {honest:.6e})")
    assert honest <= LABEL_BAR, honest
    assert not (r <= LABEL_BAR), f"{mutation} did not fire: {r}"


def test_the_bos_term_is_a_constant_function_of_the_arm():
    """THE MECHANISM, and the argument for the repair that was taken.

    Slot 0's read-out is `G_00 V_0` and `G_00` is the EMPTY path product, so no
    gate, no `m`, no `theta`, no route and no `beta` enters it. Measured: the
    slot-0 read-out is `b_0` BITWISE under every setting the arm has -- except
    `exp_scan`, where it is `nan` from the BOS's own `log m_0 = -inf`, a
    quantity the path product never reads.

    A term no setting of the arm can move is not a reading of the arm. That is
    why it is excluded from the residual and reported beside it.
    """
    a, b, _y = _bedm(8, 64)
    seen = {}
    for mutation in smp.MUTATIONS:
        u, th, v, beta, route = smp.mutate(*smp.oracle_heads(a), b.to(CT),
                                           mutation)
        o0 = _readout(a, v, u, th, beta=beta, route=route)[..., 0]
        seen[mutation] = float((o0 - b[:, 0].to(CT)).abs().max())
    print("  D1  slot-0 read-out vs b_0, by mutation: "
          + "  ".join(f"{m}={d:.3e}" for m, d in seen.items()))
    for mutation in ("none", "drop_phase", "drop_magnitude", "beta_one"):
        assert seen[mutation] == 0.0, (mutation, seen[mutation])
    #: the one exception, pinned by its mechanism rather than waved away.
    assert math.isnan(seen["exp_scan"]), seen["exp_scan"]
    assert float(a[:, 0].abs().max()) == 0.0     # the BOS gate that logs to -inf


def test_zeroing_b_zero_would_hide_a_real_label_disagreement():
    """THE REFUTATION OF THE OTHER CANDIDATE REPAIR, measured.

    Candidate (b) was "zero `b_0` in the draw". At `t_star = None` the chain is
    full length, `head = 0`, and the gate at position 1 is LIVE -- so
    `G_i0 = prod_{k=1..i} a_k` is `+-1` and slot 0's drive reaches every later
    row. There `chain_label` and BED-M's own `equilibrium_oracle` are genuinely
    DIFFERENT labels, and the arm computes the bed's.

    Zeroing `b_0` moves BOTH sides of the comparison and the disagreement
    disappears. The repaired scorer reports it. A repair that makes a real
    disagreement invisible is not a repair.
    """
    a, b, y = _bedm(8, 64, t_star=None)
    u, th = smp.oracle_heads(a)
    out = _readout(a, b.to(CT), u, th)
    tgt = chain_label(a, b)

    live_column = float(smp.path_product(smp.gate(u, th))[:, 1:, 0].abs().max())
    d_rows = float((out[..., 1:] - tgt[..., 1:]).abs().max())
    d_label = float((tgt[:, -1] - y.to(CT)).abs().max())
    d_arm = float((out[:, -1] - y.to(CT)).abs().max())

    bz = b.clone()
    bz[:, 0] = 0.0
    d_rows_zeroed = float((_readout(a, bz.to(CT), u, th)[..., 1:]
                           - chain_label(a, bz)[..., 1:]).abs().max())

    print(f"  D1  t_star=None  max|G_i0| (i>=1) = {live_column:.6f}"
          f"   rows 1.. = {d_rows:.6e}"
          f"   |chain_label[-1] - bed oracle| = {d_label:.6e}"
          f"   |read-out[-1] - bed oracle| = {d_arm:.6e}"
          f"   rows 1.. with b_0 zeroed = {d_rows_zeroed:.6e}")

    assert live_column == 1.0, live_column           # column 0 is live
    assert d_rows > 1.0, d_rows                      # and the repair reports it
    assert d_label > 1.0, d_label                    # chain_label != bed label
    assert d_arm <= 1e-12, d_arm                     # the ARM computes the bed's
    assert d_rows_zeroed <= LABEL_BAR, d_rows_zeroed  # (b) would read a PASS


def test_the_repair_is_inert_where_the_bos_convention_already_held():
    """`bedm_draw` and `band_draw` both set `b[0] = 0`, honouring
    `chain_label`'s convention, so the excluded term was already exactly zero
    and the headline number is unmoved by the repair."""
    for name, (a, b) in (("bedm_draw", smp.bedm_draw(seed=15, s=64)),
                         ("band_draw", smp.band_draw(seed=15, s=64))):
        rec = smp.label_cell(a, b, seed=15)
        print(f"  D1  {name} s=64  residual = {rec['residual']:.6e}"
              f"   residual_bos = {rec['residual_bos']:.6e}")
        assert float(b[0].abs()) == 0.0
        assert rec["residual_bos"] == 0.0, rec["residual_bos"]
        assert rec["residual"] <= LABEL_BAR, rec["residual"]


def test_the_repair_does_not_move_a_published_manifest_hash():
    """`cell_manifest` fingerprints `(operator, readout, hop, path_product,
    oracle_heads)` and hashes `CONFIG_FIELDS + SMP_FIELDS`. `label_cell` is in
    neither list and `residual_bos` is in neither block, so no published cell's
    identity moves under this repair. Pinned so that a later edit which DOES
    move it has to say so."""
    a, b = smp.bedm_draw(seed=15, s=64)
    rec = smp.label_cell(a, b, seed=15)
    assert "residual_bos" not in rec["manifest"]["smp_values"]
    assert "residual" not in rec["manifest"]["values"]
    seen = {rec["manifest"]["hash"]}
    for mutation in smp.MUTATIONS[1:]:
        h = smp.label_cell(a, b, mutation=mutation, seed=15)["manifest"]["hash"]
        assert h not in seen, f"{mutation} left the hash unmoved"
        seen.add(h)


# ==================================================== D2 (the vacuous corner)

def _corner_inputs(n: int = 64, d: int = 8, seed: int = 3):
    """`tests/gate0/test_g01_identity.py`'s draw, so the two files' corner
    numbers are on the same bytes."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g, dtype=DT),
            torch.randn(n, d, generator=g, dtype=DT),
            torch.rand(n, generator=g, dtype=DT),
            (torch.rand(n, generator=g, dtype=DT) * 2.0 - 1.0) * math.pi)


def _corner_stat(w: torch.Tensor) -> torch.Tensor:
    """The quantity the corner is certified with. Row (e)'s under the switch,
    the replacement otherwise."""
    return w.abs().sum(-1) if ROW_E else w.real.sum(-1)


def test_row_e_is_an_identity_in_beta_and_cannot_see_the_g_switch():
    """D2's RED, and its mechanism.

    `|W_ij| = R_ij e_ij / Z_i^beta` because `|G_ij| = R_ij` is clause 1, and
    `Z_i = Sigma_j R_ij e_ij` BY CONSTRUCTION -- so
    `Sigma_j |W_ij| = Z_i^{1-beta}`, which is `1` at `beta = 1` for EVERY gate,
    every `theta` and every setting of `g`. The gate cancels between the
    numerator and its own normalizer; the statistic is a function of `beta`
    alone.

    Under `G11_ROW_E=1` the corner is certified with that statistic and the
    planted `g` walks straight through, which is the RED this item owes.
    """
    q, k, u, th = _corner_inputs()
    rows = []
    for gv in (0.0, 0.25, 0.5, 0.75, 1.0, 2.0):
        w = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=gv)
        rows.append((gv,
                     float((w.abs().sum(-1) - 1.0).abs().max()),
                     float((w.real.sum(-1) - 1.0).abs().max())))
    for gv, dev_abs, dev_real in rows:
        print(f"  D2  beta=1 g={gv:<5} max|sum |W| -1| = {dev_abs:.6e}"
              f"  (row (e), VACUOUS)   max|sum Re W -1| = {dev_real:.6e}"
              f"  (replacement)")

    #: the identity, pinned across the whole sweep.
    assert max(d for _g, d, _r in rows) <= 1e-12, rows
    #: the corner is a setting of `g`, so a certificate must separate `g != 0`.
    for gv, _dev_abs, _dev_real in rows[1:]:
        w = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=gv)
        dev = float((_corner_stat(w) - 1.0).abs().max())
        assert not (dev <= ROW_BAR), \
            f"the corner statistic did not separate g={gv}: {dev}"


def test_the_replacement_keeps_every_bit_of_row_e_s_beta_sensitivity():
    """NON-DEGENERACY OF THE REPLACEMENT, and why it is a strengthening rather
    than a swap. At `g = 0` the operator is real and non-negative, so
    `Re W = |W|` ENTRYWISE and the replacement reads exactly what row (e) reads
    -- including at `beta = 0.999999`, which is what says the bar is `5e-7`.
    """
    q, k, u, th = _corner_inputs()
    w0 = smp.operator(q, k, u, th, beta=1.0, qk=1.0, g=0.0)
    assert float(w0.imag.abs().max()) == 0.0
    assert torch.equal(w0.real, w0.abs())        # entrywise, not summed
    #: the row sums differ only by float64 reduction order over a strided view.
    gap = float((w0.real.sum(-1) - w0.abs().sum(-1)).abs().max())

    ladder = {}
    for bv in (0.0, 0.5, 0.999999, 1.0):
        w = smp.operator(q, k, u, th, beta=bv, qk=1.0, g=0.0)
        ladder[bv] = (float((w.abs().sum(-1) - 1.0).abs().max()),
                      float((w.real.sum(-1) - 1.0).abs().max()))
    print(f"  D2  g=0: Re W == |W| entrywise; row-sum gap {gap:.6e}   beta ladder "
          + "  ".join(f"{b}->({x:.3e}, {y:.3e})" for b, (x, y) in ladder.items()))

    assert gap <= 1e-15, gap
    assert ladder[1.0][1] <= ROW_BAR, ladder      # the corner PASSES
    for bv in (0.0, 0.5, 0.999999):
        assert not (ladder[bv][1] <= ROW_BAR), (bv, ladder[bv])


def test_row_e_is_blind_to_the_gate_itself_and_the_replacement_is_not():
    """D2's blindness stated at its widest: row (e) is `1.000000` at `beta = 1`
    on gates that have nothing in common, BED-M's own `-1` among them.

    The annihilating gate is recorded as the replacement's OWN limit: there
    every row is a delta, so it really does sum to `1` in both statistics and
    the replacement is not a complete corner certificate on its own either.
    """
    q, k, _u, _th = _corner_inputs()
    gen = torch.Generator().manual_seed(11)
    gates = {
        "m~U(0,1), theta~U(-pi,pi)": (torch.rand(64, generator=gen, dtype=DT),
                                      (torch.rand(64, generator=gen, dtype=DT)
                                       * 2 - 1) * math.pi),
        "m=1, theta=pi  (BED-M's -1)": (torch.ones(64, dtype=DT),
                                        torch.full((64,), math.pi, dtype=DT)),
        "m=0  (annihilating)": (torch.zeros(64, dtype=DT),
                                torch.zeros(64, dtype=DT)),
    }
    read = {}
    for tag, (uu, tt) in gates.items():
        w = smp.operator(q, k, uu, tt, beta=1.0, qk=1.0, g=1.0)
        read[tag] = (float((w.abs().sum(-1) - 1.0).abs().max()),
                     float((w.real.sum(-1) - 1.0).abs().max()))
        print(f"  D2  gate {tag:28s} sum|W|-1 = {read[tag][0]:.6e}"
              f"   sum Re W -1 = {read[tag][1]:.6e}")

    for tag, (dev_abs, _dev_real) in read.items():
        assert dev_abs <= 1e-12, (tag, dev_abs)          # blind to all three
    for tag in ("m~U(0,1), theta~U(-pi,pi)", "m=1, theta=pi  (BED-M's -1)"):
        assert read[tag][1] > 1.0, (tag, read[tag])      # the replacement sees
    #: the limit, on the record rather than discovered later.
    assert read["m=0  (annihilating)"][1] <= ROW_BAR, read["m=0  (annihilating)"]
