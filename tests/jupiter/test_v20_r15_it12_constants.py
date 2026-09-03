"""CEQ v20 R15 it.12 -- THE FIVE UNASSERTED CONSTANTS, AND Q2'S DOMAIN.

`V20_R15_IT11_INSPECTOR.md:552-558` refused the `+6` on one sentence: "four of
its constants are asserted by nothing." This file is the answer. Every constant
below was PUBLISHED IN PROSE BEFORE this file existed, at the path:line named
beside it, and every node here recomputes it from the same object the prose
measured and asserts it to a stated tolerance no wider than the digits the prose
printed.

    constant            published at                       tolerance
    1.0845223424        WITHDRAWN prose 1.084523, see below  5e-10
    0.9746794345        V20_R15_IT6_JUPITER.md:466         5e-11  (prose: 10 dp)
    +0.717647           V20_R15_IT7_JUPITER.md:91          5e-7   (prose: 6 dp)
    -0.032353           V20_R15_IT7_JUPITER.md:92          5e-7   (prose: 6 dp)
    +0.145 ... +0.220   V20_R15_IT8_JUPITER.md:344,352     5e-4   (prose: 3 dp)

PLANTED NEGATIVES. Set ``JUP_IT12_MUTATE`` to one of the names below; the
mutation is applied to the DATA the constant is computed from, never to an
assertion. Nothing here is conjoined with ``False`` -- that is the V-16 class
and this office is on record about it (`V20_R15_IT11_INSPECTOR.md`).

  hull_target -- add 0.01 to the escaping target sample; the hull excess moves
  delay_d     -- build the delay block at d=21; err_1 moves off sqrt(1-1/20)
  qk_rank     -- swap two cells' qk; the qk Spearman moves
  beta_rank   -- swap two cells' beta; the beta Spearman moves
  band_cell   -- nudge one flat-band cell's eval_nrmse; the band endpoints move
  q2_domain   -- pretend the Q2 file imports an arm; the domain census breaks

Read-only. No git write, nothing touches Kaggle.
"""
from __future__ import annotations

import json
import math
import os
import pathlib

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
RETAKE = ROOT / "results" / "v17k_r4_retake.jsonl"
FRESH = ROOT / "results" / "v20_r15_it6_seeds8_15.jsonl"
MUT = os.environ.get("JUP_IT12_MUTATE", "")

FLOOR1 = 0.7071067811865476  #: CEQ_V20_R15_CONTRACT.md:116, preregistered
FLAT_FGA = 0.5032958984375  #: the flat band's frac_gate_annihilated, it.7 :78


# =========================================================== the sixteen cells
def _cells(path: pathlib.Path, kind: str) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("t") == "cell" and rec["kind"] == kind:
            out.append(rec)
    return out


@pytest.fixture(scope="module")
def w1() -> list[dict]:
    """The same sixteen `arm_smprime` cells the it.7 filing measured, loaded the
    same way (`tests/jupiter/test_v20_r15_it7_q3.py:34-47`): the retake journal
    first, the fresh journal only for seeds it does not already carry."""
    by_seed = {c["seed"]: c for c in _cells(RETAKE, "arm_smprime")}
    for c in _cells(FRESH, "arm_smprime"):
        by_seed.setdefault(c["seed"], c)
    cells = [json.loads(json.dumps(c)) for _, c in sorted(by_seed.items())]
    assert [c["seed"] for c in cells] == list(range(16))
    if MUT == "qk_rank":
        a, b = cells[0]["manifest"]["smp_values"], cells[9]["manifest"]["smp_values"]
        a["qk"], b["qk"] = b["qk"], a["qk"]
    if MUT == "beta_rank":
        a, b = cells[0]["manifest"]["smp_values"], cells[9]["manifest"]["smp_values"]
        a["beta"], b["beta"] = b["beta"], a["beta"]
    if MUT == "band_cell":
        next(c for c in cells if c["frac_gate_annihilated"] == FLAT_FGA)["eval_nrmse"] += 0.05
    return cells


def _spearman(x: list[float], y: list[float]) -> float:
    """Pearson on midranks. numpy only -- scipy is installed but this is four
    lines and the tie handling is the whole content of the definition."""
    def rank(v):
        order = np.argsort(np.asarray(v), kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        # midranks for ties, so a tied pair cannot inflate rho
        arr = np.asarray(v, dtype=float)
        for u in np.unique(arr):
            m = arr == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def test_qk_spearman_is_plus_0_717647(w1):
    """CONSTANT 3. V20_R15_IT7_JUPITER.md:91 prints `+0.717647` for
    rho(manifest.smp_values.qk, eval_nrmse) over the sixteen cells. Until this
    node existed, `grep -rn spearman` over tests/ scripts/ ceq/ returned
    nothing (V20_R15_IT11_INSPECTOR.md:554)."""
    qk = [c["manifest"]["smp_values"]["qk"] for c in w1]
    nrmse = [c["eval_nrmse"] for c in w1]
    rho = _spearman(qk, nrmse)
    print("Q3/W1 rho(qk, eval_nrmse), n=16 = %.10f   prose +0.717647" % rho)
    assert abs(rho - 0.717647) < 5e-7, "prose +0.717647, measured %.10f" % rho


def test_beta_spearman_is_minus_0_032353(w1):
    """CONSTANT 4. V20_R15_IT7_JUPITER.md:92 prints `-0.032353`, and it is
    load-bearing: it carries the it.7 s2.2 strike of the M1 descent transfer.
    `beta` appeared zero times in the it.7 test file."""
    beta = [c["manifest"]["smp_values"]["beta"] for c in w1]
    nrmse = [c["eval_nrmse"] for c in w1]
    rho = _spearman(beta, nrmse)
    print("Q3/W1 rho(beta, eval_nrmse), n=16 = %.10f   prose -0.032353" % rho)
    assert abs(rho + 0.032353) < 5e-7, "prose -0.032353, measured %.10f" % rho


def test_q5_w1_flat_band_distance_to_floor1_is_0_145_to_0_220(w1):
    """CONSTANT 5. V20_R15_IT8_JUPITER.md:344 tabulates the flat band's
    `dist_to_floor` as `+0.145 ... +0.220` and :352 grades Q5/W1 F1 on it. The
    band is `eval_nrmse - floor1` over the twelve cells at the flat
    `frac_gate_annihilated`; zero test nodes read it before this one."""
    flat = [c for c in w1 if c["frac_gate_annihilated"] == FLAT_FGA]
    assert len(flat) == 12, "the flat band is twelve cells, got %d" % len(flat)
    dist = sorted(c["eval_nrmse"] - FLOOR1 for c in flat)
    print("Q5/W1 dist_to_floor over the flat band: min=%.6f max=%.6f (n=12)"
          % (dist[0], dist[-1]))
    print("Q5/W1 all twelve = %r" % ["%.6f" % d for d in dist])
    assert abs(dist[0] - 0.145) < 5e-4, "prose +0.145, measured %.6f" % dist[0]
    assert abs(dist[-1] - 0.220) < 5e-4, "prose +0.220, measured %.6f" % dist[-1]


# ================================================== the convex-hull excess
def test_sign_flip_gate_seed0_hull_excess_is_1_0845223424():
    """CONSTANT 1, AND THE PUBLISHED VALUE IS WITHDRAWN.
    V20_R15_IT6_JUPITER.md:427,494 states `dist = 1.084523`. It existed in three
    prose hits and no Python file (V20_R15_IT11_INSPECTOR.md:59). When this node
    was first run it went RED against that value:

        AssertionError: prose 1.084523, measured 1.0845223424
        assert 6.575759790017344e-07 < 5e-07

    The Inspector's diagnosis is confirmed arithmetically: `1.084523` is the
    hand-subtraction of two SIX-DECIMAL DISPLAYS, `2.483118 - 1.398595`, and the
    true excess is `1.0845223424`, which displays as `1.084522`. The published
    digit is wrong. `1.084523` IS WITHDRAWN; this node asserts the measured value
    at 5e-10, ten times tighter than the digits any prose printed.

    The hull of a row of nonnegative weights
    summing to 1 is [min, max] of the values it mixes, so the excess is a
    property of `(a, b)` alone and needs no operator call to state."""
    S = 8
    torch.manual_seed(0)  #: identical draw order to test_v20_r15_it6_q1_exact_class.py:141
    a = (torch.randint(0, 2, (S,)) * 2 - 1).to(torch.float64)
    b = torch.randn(S, dtype=torch.float64)
    t = torch.zeros(S, dtype=torch.float64)
    for i in range(S):
        prev = t[i - 1] if i > 0 else torch.zeros((), dtype=torch.float64)
        t[i] = a[i] * prev + b[i]
    if MUT == "hull_target":
        t[1] += 0.01

    esc = None
    for i in range(S):
        lo, hi = b[: i + 1].min().item(), b[: i + 1].max().item()
        ti = t[i].item()
        if ti < lo - 1e-12 or ti > hi + 1e-12:
            esc = (i, lo, hi, ti, (lo - ti) if ti < lo else (ti - hi))
            break
    assert esc is not None, "the planted negative must escape the hull somewhere"
    i0, lo0, hi0, t0, dist = esc
    print("Q2/W3 first escape at i=%d: t_i=%.6f not in [%.6f, %.6f]" % (i0, t0, lo0, hi0))
    print("Q2/W3 hull excess dist = %.10f   (withdrawn prose: 1.084523)" % dist)
    assert i0 == 1, "prose says the first escape is at i=1, got i=%d" % i0
    assert abs(dist - 1.0845223424) < 5e-10, (
        "withdrawn prose 1.084523; measured value is 1.0845223424, got %.10f" % dist)


# ============================================ the delay-block headline constant
def test_q2_w1_err_1_is_0_9746794345():
    """CONSTANT 2. V20_R15_IT6_JUPITER.md:466 names `err_1 = 0.9746794345` as
    THE constant of the Q2/W1 cell; the it.6 node asserted only
    `np.isfinite(err_1)` (test_v20_r15_it6_q2_outside_bound.py:105).

    THIS NODE DOES NOT RESCUE THE CELL. The Inspector is right that the block is
    `I_d` with a zero row, so `err_1 = sqrt(1 - 1/d)` is a closed form and this
    assertion is a check on arithmetic, not on a construction. It is written
    because a published constant with no node is worse than a published constant
    whose node is weak, and the second line below states the closed form so a
    reader can see exactly how much the node is worth."""
    d = 21 if MUT == "delay_d" else 20
    base = "b" * d
    rows = [base] + ["b" * k + "a" + "b" * (d - 1 - k) for k in range(d)]
    cols = ["b" * j for j in range(d)]

    def f(w: str) -> float:
        return 0.0 if len(w) < d else (1.0 if w[len(w) - d] == "a" else 0.0)

    H = np.array([[f(r + c) for c in cols] for r in rows], dtype=float)
    sv = np.linalg.svd(H, compute_uv=False)
    energy = sv ** 2
    r2_1 = float(energy[0] / energy.sum())
    err_1 = math.sqrt(1.0 - r2_1)
    print("Q2/W1 d=%d: R2_1=%.15f  err_1=%.10f   prose 0.9746794345" % (d, r2_1, err_1))
    print("Q2/W1 closed form sqrt(1-1/20)=%.15f -- the block is I_d with a zero row, "
          "so this node checks arithmetic, not the construction" % math.sqrt(1 - 1 / 20))
    assert abs(err_1 - 0.9746794345) < 5e-11, "prose 0.9746794345, measured %.10f" % err_1


# ======================================================= TASK B: Q2'S DOMAIN
Q2_FILE = "tests/jupiter/test_v20_r15_it6_q2_outside_bound.py"
W1_MODULE = "ceq/arm_smprime.py"   #: W1's arm
W3_MODULE = "ceq/arm_pl.py"        #: W3's arm


def test_q2_reads_no_object_that_either_frozen_wing_is_measured_on():
    """TASK B, THE DOMAIN CENSUS. Q2 is graded on BED-K objects while both
    frozen wings are BED-M, which this office's own it.8 ruling forbids:
    V20_R15_IT8_JUPITER.md:474 -- "BED-K only ... do not cite it on W1/W3 at
    all."

    The census is an emptiness claim and it is checked the way the round's other
    domain censuses are checked (tests/jupiter/test_v20_r15_it6_ldom_census.py):
    by reading the source. Q2's file imports `ceq.hankel` and nothing else from
    the tree, names neither arm, and opens no journal -- so its intersection
    with either wing's domain is empty, and no number it produces can bind
    either wing. The wing files below are asserted to read the journals, so the
    emptiness is a real disjointness and not an artifact of both sides being
    empty."""
    src = (ROOT / Q2_FILE).read_text(encoding="utf-8")
    if MUT == "q2_domain":
        src += "\nfrom ceq import arm_pl\n"

    imports = sorted({ln.strip() for ln in src.splitlines()
                      if ln.startswith(("import ", "from ")) and "ceq" in ln})
    print("Q2 ceq-imports = %r" % imports)
    assert imports == ["from ceq.hankel import NEG_ENTRY, hankel_block, rank_plus_lower, rank_real"], (
        "Q2's ceq surface is not hankel-only: %r" % imports)
    for name in ("arm_smprime", "arm_pl", "arm_phase", "corpus"):
        assert name not in src, "Q2 names %s; the domain claim is void" % name
    assert "results/" not in src and ".jsonl" not in src, (
        "Q2 opens a journal; it is no longer a pure BED-K object")

    #: the other side of the disjointness: both wings ARE journal-measured
    for wing_test in ("tests/jupiter/test_v20_r15_it7_q3.py",
                      "tests/jupiter/test_v20_r15_it8_q4_q5.py"):
        wsrc = (ROOT / wing_test).read_text(encoding="utf-8")
        assert ".jsonl" in wsrc, "%s reads no journal" % wing_test

    #: and BED-K is a different corpus from BED-M, on the module's own word
    bedk = (ROOT / "ceq/beds/bed_k.py").read_text(encoding="utf-8").splitlines()
    assert "BED-K" in bedk[0], bedk[0]
    assert "Companion to ceq/corpus.py's BED-M" in bedk[2], bedk[2]
    print("Q2 domain census: BED-K only; intersection with W1/W3 domain = EMPTY")
