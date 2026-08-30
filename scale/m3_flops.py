"""Analytic FLOP accounting for the five M3 cells.

WHY THIS FILE EXISTS. The K-F gate is a wall-clock gate, and this box is
contended, so the clock is not a measurement. Shapes are not contended: a FLOP
count is arithmetic on the tensor shapes the shipped code actually builds, and
contention cannot move it. Nothing here is timed. Nothing here is a clock.

EVERY formula below carries the file and the line number it was read from.
Where a term's cost is not fixed by the source, the term prints NOT FOUND
rather than a number.
"""
from __future__ import annotations

import pathlib
import sys

import torch

torch.set_num_threads(2)          # pinned HERE, not by the launcher

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.arm_s import neumann_for                              # noqa: E402

# ------------------------------------------------------------------ constants
# D_MODEL / HIDDEN / LR read from scale/m3_capability.py:78-87; the evaluation
# point (n, s, k sweep, t_star) is the one the task fixes.
N, S, D_MODEL, HIDDEN = 8192, 64, 16, 128
KS = (8, 16, 32)
T_STAR = 21
BETA = 0.5

# ARM S settles exactly ONE row per example -- `out[-1] = ...` at
# scale/arm_s.py:254 -- and the M3 readout reads exactly that row,
# `out[:, s - 1]` at scale/m3_capability.py:166. So every ARM S add-on term is
# paid n times, once per batch example.
ROWS_SETTLED = 1


# ------------------------------------------------------------------ the terms
# Each entry: (label, symbolic formula, citation, value function).
# CONVENTION: one multiply-add = 2 FLOPs, so a matmul [m,p]@[p,q] is 2*m*p*q.

BASE_TERMS = [
    ("wq(x), wk(x)", "4*n*s*d_model^2",
     "m3_capability.Arm.forward, self.wq/self.wk "
     "(nn.Linear(d_model, d_model, bias=False))",
     lambda n, s, dm, h, k, t: 4 * n * s * dm * dm),
    ("q @ k^T", "2*n*s^2*d_model",
     "m3_capability.Arm._operator -> bench._softmax_operator (softmax branch; "
     "the signed branch is bench._causal_sgate_operator at the same shape)",
     lambda n, s, dm, h, k, t: 2 * n * s * s * dm),
    ("a @ x", "2*n*s^2*d_model",
     "m3_capability.Arm.forward, the `a = self._operator(q, k)` operator "
     "applied to x",
     lambda n, s, dm, h, k, t: 2 * n * s * s * dm),
    ("mlp(z)", "4*n*s*d_model*hidden",
     "m3_capability.Arm.forward, self.mlp (nn.Sequential, two Linear layers)",
     lambda n, s, dm, h, k, t: 4 * n * s * dm * h),
    ("readout(h)", "2*n*s*d_model",
     "m3_capability.Arm.forward, self.readout (nn.Linear(d_model, 1)); all s "
     "rows are built, and only then is one position indexed out",
     lambda n, s, dm, h, k, t: 2 * n * s * dm),
]

SELECT_TERM = (
    "pivot select (key norm)", "n*2*s*d_model",
    "arm_s.pivots_of -> pivot_probe.select_pivots (the topk itself is "
    "comparisons, NOT COUNTED)",
    lambda n, s, dm, h, k, t: ROWS_SETTLED * n * 2 * s * dm)

SETUP_TERM = (
    "SETUP log_pivot_context", "n*( 2*(k+1)*s*d_model + 2*k^2*s + 2*k*s*d_model )",
    "arm_s.log_pivot_context: the `w = q[want] @ kk^T / sqrt(d)` logit slice, "
    "the chunked `log_gram` logsumexp loop, and the `la_piv.exp() @ v` return; "
    "same decomposition as the shipped counter `f_setup` in arm_s.gate3_cost",
    lambda n, s, dm, h, k, t: ROWS_SETTLED * n * (
        2 * (k + 1) * s * dm + 2 * k * k * s + 2 * k * s * dm))

LOOP_TERM = (
    "LOOP t_star * log_alpha_step", "n*t_star*2*k^2",
    "arm_s.log_alpha_step: one logsumexp over [k,k]; per-step cost as counted "
    "by `f_step` in arm_s.gate3_cost",
    lambda n, s, dm, h, k, t: ROWS_SETTLED * n * t * 2 * k * k)

#: THE PER-ROW TERMS. A `ROW_CELLS` cell writes the pivot reading at every query
#: row, so the query-row-dependent work is paid `s` times and the query-row-
#: INDEPENDENT work is not paid again at all. Splitting them is the whole point
#: of the entry: pricing a row cell by reusing SETUP/LOOP/CONTRACT understates
#: it, and pricing it as `s` copies of the single-row cell overstates it.
#:
#: What is shared and NOT multiplied: the pivot selection, the Gram, and
#: `A_P @ V` depend on the pivot set alone. `m3_quintuple.batched_row_gates`
#: recomputes none of them.
#:
#: What replaces the setup logit slice: the single-row path forms `k+1` rows of
#: `q @ kk^T`; every row is wanted here, so the full `[s, s]` product is formed
#: -- `2*(k+1)*s*d_model` becomes `2*s*s*d_model` for the gate, and the `k`
#: pivot rows are still formed by the shared call, so only the `+1` query row is
#: replaced by `s` of them.
#:
#: THIS ENTRY IS A FLOOR ON THE ROW CELLS' COST AND NOT A PREDICTION OF IT, AND
#: THE GAP WIDENS WITH `s`. The FLOP ratio `settledrow / settled` is 1.700 at
#: every geometry below. The measured wall-clock ratio, same session, one
#: training step including backward and Adam:
#:
#:     s=16 n=512    0.055674 -> 0.193932   ratio  3.48
#:     s=64 n=2048   0.407800 -> 4.560600   ratio 11.19
#:     s=64 n=8192   3.437200 -> 25.54320   ratio  7.43
#:
#: so the model is optimistic by 2.0x at the pilot geometry and by 4.4x to 6.6x
#: at the shipped one. The arithmetic is right; the dispatch is not in it. The
#: settle is a Python loop over `t_max` steps, and the per-row form runs it over
#: an `[n*s, k]` tensor whose Gram copy is 268 MB at s=64, n=8192.
#:
#: **A pilot cost measured at small `s` must not be scaled to full geometry by
#: this term**; doing so understates the bill by 2x to 3x. The 1.5x spread
#: between the two `s=64` rows is contention and not `n`: the same `settled`
#: unit read 2.0775 s/step in an earlier session against 3.4372 s/step here,
#: 1.65x apart on identical code and geometry, which is why every clock in this
#: repository is labelled PROVISIONAL and why the ratios above are quoted only
#: from same-session pairs.
ROW_GATE_TERM = (
    "PER-ROW gate (full logit matrix)", "n*2*s^2*d_model",
    "m3_quintuple.batched_row_gates: the `q @ kk^T` full product in float64, "
    "replacing the single query row of arm_s.log_pivot_context's `k+1`-row "
    "slice. The k pivot rows are unchanged and stay in SETUP_TERM",
    lambda n, s, dm, h, k, t: ROWS_SETTLED * n * 2 * s * s * dm)

CONTRACT_TERM = (
    "alpha @ AV", "n*2*k*d_model",
    "arm_s.arm_s, the `out[-1] = (log_alpha.exp() @ av)` contraction",
    lambda n, s, dm, h, k, t: ROWS_SETTLED * n * 2 * k * dm)


def bwd_spec(n, s, dm, h, k, t, nn_terms):
    """(n_neumann - 1) VJPs, per the `for _ in range(ctx.n_neumann - 1)` loop in
    arm_s.Settled.backward.

    Per-VJP cost 4*k^2 is the shipped figure `f_jvp` in arm_s.gate3_cost.
    """
    return ROWS_SETTLED * n * (nn_terms - 1) * 4 * k * k


def bwd_shipped(n, s, dm, h, k, t, nn_terms):
    """What `flops_backward` in arm_s.gate3_cost multiplies by: n_neumann, not
    n_neumann - 1."""
    return ROWS_SETTLED * n * nn_terms * 4 * k * k


def base_flops(n, s, dm, h, k, t):
    return sum(f(n, s, dm, h, k, t) for _, _, _, f in BASE_TERMS)


def cell_terms(cell, n, s, dm, h, k, t, nn_terms):
    """Per-cell {term: flops}."""
    base = base_flops(n, s, dm, h, k, t)
    sel = SELECT_TERM[3](n, s, dm, h, k, t)
    setup = SETUP_TERM[3](n, s, dm, h, k, t)
    loop = LOOP_TERM[3](n, s, dm, h, k, t)
    contract = CONTRACT_TERM[3](n, s, dm, h, k, t)
    if cell == "softmax":
        return dict(base=base, select=0, setup=0, loop=0, contract=0, bwd=0)
    if cell == "glance":
        # arm_s.arm_s: `if t_max == 0: return out, None` fires BEFORE the
        # `piv = pivots_of(kk, k_piv)` that follows it, so `a @ v` is the whole
        # cost. No setup, no loop, no contract, no backward.
        return dict(base=base, select=0, setup=0, loop=0, contract=0, bwd=0)
    if cell == "settled":
        return dict(base=base, select=sel, setup=setup, loop=loop,
                    contract=contract,
                    bwd=bwd_spec(n, s, dm, h, k, t, nn_terms))
    if cell == "twin":
        # ZERO settling steps: alpha = normalised gate. Setup paid, loop not.
        # No fixed point is solved, so no implicit-gradient backward runs.
        return dict(base=base, select=sel, setup=setup, loop=0,
                    contract=contract, bwd=0)
    if cell == "argmax":
        # alpha one-hot at argmax(gate): the `out[-1] = (log_alpha.exp() @ av)`
        # contraction in arm_s.arm_s degenerates to reading one row of AV -- a gather,
        # zero multiply-adds. argmax itself is comparisons, NOT COUNTED.
        return dict(base=base, select=sel, setup=setup, loop=0, contract=0,
                    bwd=0)
    if cell in ("twinrow", "settledrow"):
        # The gate is per query row; the Gram and A_P@V are not. The setup term
        # keeps its k pivot rows but loses its single query row to ROW_GATE.
        gate = ROW_GATE_TERM[3](n, s, dm, h, k, t)
        shared = setup - ROWS_SETTLED * n * 2 * s * dm
        rows = dict(base=base, select=sel, setup=shared + gate,
                    loop=loop * s, contract=contract * s,
                    bwd=bwd_spec(n, s, dm, h, k, t, nn_terms) * s)
        if cell == "twinrow":
            # Same relationship twin has to settled: no fixed point is solved,
            # so no settle loop and no implicit-gradient backward.
            rows.update(loop=0, bwd=0)
        return rows
    raise ValueError(cell)


CELLS = ("softmax", "glance", "settled", "twin", "argmax")

SYMBOLIC = {
    "softmax": "F_base",
    "glance":  "F_base                                   (== cell 1, exactly)",
    "settled": "F_base + n*[ 2*s*d + 2*(k+1)*s*d + 2*k^2*s + 2*k*s*d "
               "+ t_star*2*k^2 + 2*k*d + (n_neumann-1)*4*k^2 ]",
    "twin":    "F_base + n*[ 2*s*d + 2*(k+1)*s*d + 2*k^2*s + 2*k*s*d + 2*k*d ]",
    "argmax":  "F_base + n*[ 2*s*d + 2*(k+1)*s*d + 2*k^2*s + 2*k*s*d ]",
}


def main() -> int:
    print("=" * 78)
    print("ANALYTIC FLOP ACCOUNTING -- FIVE M3 CELLS")
    print("=" * 78)
    print("NO WALL CLOCK APPEARS IN THIS FILE. Nothing here is timed.")
    print()
    print("--- CONVENTION (stated once) ---")
    print("  ONE MULTIPLY-ADD = 2 FLOPS. A matmul [m,p]@[p,q] costs 2*m*p*q.")
    print("  Counts are MATMUL flops only -- the same convention as the shipped")
    print("  counter at scale/arm_s.py:457-460. Elementwise and transcendental")
    print("  terms are listed at the end with their ELEMENT counts; their")
    print("  per-element FLOP cost is NOT FOUND in the source.")
    print("  d == d_model == %d: scale/m3_capability.py:142 feeds width-d_model"
          % D_MODEL)
    print("    q,k into the operator (nn.Linear(d_model,d_model) at :107-108),")
    print("    so ARM S's `d` and M3's `d_model` are the same number here.")
    print("  k is the REQUESTED pivot count. scale/arm_s.py:108 drops p == 0,")
    print("    so the live count kp <= k: every count below is an UPPER BOUND")
    print("    taken at kp = k.")
    print("  Every ARM S add-on is multiplied by n: ARM S settles ONE row")
    print("    (scale/arm_s.py:254) and M3 reads exactly that row")
    print("    (scale/m3_capability.py:166), so it is paid once per example.")
    print()

    print("--- n_neumann, RUN not assumed ---")
    nn_terms = neumann_for(BETA)
    print("  scale/arm_s.py:317 neumann_for(beta=%s) -> %r  (type %s)"
          % (BETA, nn_terms, type(nn_terms).__name__))
    print("  backward loop runs n_neumann - 1 = %d VJPs "
          "(scale/arm_s.py:310-311)" % (nn_terms - 1))
    print("  NOTE, raw: the SHIPPED counter at scale/arm_s.py:468 multiplies by")
    print("    n_neumann (%d), not n_neumann - 1 (%d). Both totals printed below."
          % (nn_terms, nn_terms - 1))
    print("  t_star = %d, supplied by the task, not read from a run." % T_STAR)
    print()

    print("--- F_base, the shared M3 arm (scale/m3_capability.py:140-166) ---")
    print("  %-22s %-24s %s" % ("term", "symbolic", "citation"))
    for label, sym, cite, _ in BASE_TERMS:
        print("  %-22s %-24s %s" % (label, sym, cite))
    print("  %-22s %s" % ("F_base TOTAL",
                          "4*n*s*d_model^2 + 4*n*s^2*d_model "
                          "+ 4*n*s*d_model*hidden + 2*n*s*d_model"))
    print()

    print("--- ARM S add-on terms ---")
    for label, sym, cite, _ in (SELECT_TERM, SETUP_TERM, LOOP_TERM,
                                CONTRACT_TERM):
        print("  %-24s %s" % (label, sym))
        print("  %-24s   ^ %s" % ("", cite))
    print("  %-24s %s" % ("BWD (spec)", "n*(n_neumann-1)*4*k^2"))
    print("  %-24s   ^ %s" % ("", "scale/arm_s.py:310-311, per-VJP 4*k^2 from "
                                  "scale/arm_s.py:460"))
    print("  %-24s %s" % ("BWD (shipped counter)", "n*n_neumann*4*k^2"))
    print("  %-24s   ^ %s" % ("", "scale/arm_s.py:468"))
    print()

    print("--- SYMBOLIC FORMULA PER CELL ---")
    for i, c in enumerate(CELLS, 1):
        print("  %d. %-8s %s" % (i, c, SYMBOLIC[c]))
    print()

    print("=" * 78)
    print("EVALUATED at n=%d s=%d d_model=%d hidden=%d t_star=%d n_neumann=%d"
          % (N, S, D_MODEL, HIDDEN, T_STAR, nn_terms))
    print("=" * 78)

    base = base_flops(N, S, D_MODEL, HIDDEN, 0, 0)
    print("F_base = %d FLOPs  (independent of k)" % base)
    for label, _, _, f in BASE_TERMS:
        v = f(N, S, D_MODEL, HIDDEN, 0, 0)
        print("    %-22s %18d   %8.4f%% of F_base" % (label, v, 100.0 * v / base))
    print()

    hdr = ("  %-8s %5s %16s %14s %12s %10s %12s %16s %10s"
           % ("cell", "k", "base", "setup", "loop", "contract", "bwd",
              "TOTAL", "ratio"))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in KS:
        for c in CELLS:
            t = cell_terms(c, N, S, D_MODEL, HIDDEN, k, T_STAR, nn_terms)
            setup_col = t["setup"] + t["select"]
            total = (t["base"] + setup_col + t["loop"] + t["contract"]
                     + t["bwd"])
            print("  %-8s %5d %16d %14d %12d %10d %12d %16d %10.6f"
                  % (c, k, t["base"], setup_col, t["loop"], t["contract"],
                     t["bwd"], total, total / base))
        print("  " + "-" * (len(hdr) - 2))
    print("  setup column = SETUP log_pivot_context + pivot-select key norm.")
    print("  ratio = cell TOTAL / cell 1 (softmax) TOTAL = %d." % base)
    print()

    print("--- FORWARD-ONLY totals (backward excluded), and their ratios ---")
    print("  %-8s %5s %16s %10s" % ("cell", "k", "FWD TOTAL", "ratio"))
    for k in KS:
        for c in CELLS:
            t = cell_terms(c, N, S, D_MODEL, HIDDEN, k, T_STAR, nn_terms)
            fwd = (t["base"] + t["setup"] + t["select"] + t["loop"]
                   + t["contract"])
            print("  %-8s %5d %16d %10.6f" % (c, k, fwd, fwd / base))
    print()

    print("--- settled cell under the SHIPPED backward multiplier "
          "(scale/arm_s.py:468) ---")
    print("  %-8s %5s %16s %10s" % ("cell", "k", "TOTAL", "ratio"))
    for k in KS:
        t = cell_terms("settled", N, S, D_MODEL, HIDDEN, k, T_STAR, nn_terms)
        tot = (t["base"] + t["setup"] + t["select"] + t["loop"] + t["contract"]
               + bwd_shipped(N, S, D_MODEL, HIDDEN, k, T_STAR, nn_terms))
        print("  %-8s %5d %16d %10.6f" % ("settled", k, tot, tot / base))
    print()

    print("--- SETUP vs LOOP, side by side (which one dominates) ---")
    print("  %5s %22s %20s %12s %14s"
          % ("k", "SETUP n*O(k*s*d+k^2*s)", "LOOP n*t_star*2*k^2",
             "setup/loop", "setup/F_base"))
    for k in KS:
        setup = SETUP_TERM[3](N, S, D_MODEL, HIDDEN, k, T_STAR)
        loop = LOOP_TERM[3](N, S, D_MODEL, HIDDEN, k, T_STAR)
        print("  %5d %22d %20d %12.6f %14.6f"
              % (k, setup, loop, setup / loop, setup / base))
    print("  SETUP split, per k:")
    print("    %5s %20s %20s %20s"
          % ("k", "logit slice", "Gram logsumexp", "A_P @ V"))
    print("    %5s %20s %20s %20s"
          % ("", "n*2*(k+1)*s*d", "n*2*k^2*s", "n*2*k*s*d"))
    print("    %5s %20s %20s %20s"
          % ("", "arm_s.py:155", "arm_s.py:168-172", "arm_s.py:173"))
    for k in KS:
        a = N * 2 * (k + 1) * S * D_MODEL
        b = N * 2 * k * k * S
        c2 = N * 2 * k * S * D_MODEL
        print("    %5d %20d %20d %20d" % (k, a, b, c2))
    print()

    print("--- CELL 2 == CELL 1, shown rather than asserted ---")
    print("  scale/arm_s.py:236-256: at t_max = 0, arm_s returns at :247 with")
    print("  `a @ v` from :245-246, BEFORE :249 selects any pivot. The docstring")
    print("  at :241-243 states the returned tensor is BITWISE identical to")
    print("  glance(q, kk, v) (:259-261), and g3_bind (:330) measures it.")
    print("  So cell 2's count must equal cell 1's. It does:")
    for k in KS:
        t1 = cell_terms("softmax", N, S, D_MODEL, HIDDEN, k, T_STAR, nn_terms)
        t2 = cell_terms("glance", N, S, D_MODEL, HIDDEN, k, T_STAR, nn_terms)
        s1, s2 = sum(t1.values()), sum(t2.values())
        print("    k=%-3d softmax=%d  glance=%d  equal=%s  difference=%d"
              % (k, s1, s2, s1 == s2, s1 - s2))
        assert s1 == s2, (k, s1, s2)
    print()

    print("--- NON-MATMUL terms: ELEMENT counts, per-element FLOP NOT FOUND ---")
    print("  These are NOT in the totals above. Element counts are exact;")
    print("  what one exp/log/div/tanh costs is NOT FOUND in the source.")
    rows = [
        ("scale by 1/sqrt(d)", "n*s^2", N * S * S, "ceq/bench.py:187"),
        ("softmax over [n,s,s]", "3*n*s^2 (exp, add, div)", 3 * N * S * S,
         "ceq/bench.py:188"),
        ("masked_fill x2", "2*n*s^2", 2 * N * S * S, "ceq/bench.py:188"),
        ("residual add x + a@x", "n*s*d_model", N * S * D_MODEL,
         "scale/m3_capability.py:144"),
        ("mlp biases", "n*s*(hidden+d_model)", N * S * (HIDDEN + D_MODEL),
         "scale/m3_capability.py:110"),
        ("GELU", "n*s*hidden", N * S * HIDDEN, "scale/m3_capability.py:110"),
        ("readout bias", "n*s", N * S, "scale/m3_capability.py:112"),
        ("log_softmax in setup", "n*(k+1)*s", None,
         "scale/arm_s.py:157-158 -- k-dependent, per-k below"),
        ("la_piv.exp() in setup", "n*k*s", None,
         "scale/arm_s.py:173 -- k-dependent, per-k below"),
        ("log_alpha_step O(k) tail", "n*t_star*3*k", None,
         "scale/arm_s.py:179-180, NOT in the 2*k^2 shipped step count"),
    ]
    for label, sym, val, cite in rows:
        if val is None:
            print("  %-26s %-24s %14s  %s" % (label, sym, "see per-k", cite))
        else:
            print("  %-26s %-24s %14d  %s" % (label, sym, val, cite))
    print("  per-k element counts:")
    print("    %5s %18s %18s %20s" % ("k", "log_softmax", "la_piv.exp()",
                                      "step O(k) tail"))
    for k in KS:
        print("    %5d %18d %18d %20d"
              % (k, N * (k + 1) * S, N * k * S, N * T_STAR * 3 * k))
    print("  gate normalise (twin, alpha = gate/gate.sum()): n*k elements, "
          "no matmul")
    print("  argmax(gate) (argmax cell): n*k comparisons, NOT COUNTED as FLOPs")
    print("  topk in pivot select: comparisons, NOT COUNTED as FLOPs "
          "(scale/pivot_probe.py:91)")
    print()
    print("torch %s  torch.get_num_threads()=%d"
          % (torch.__version__, torch.get_num_threads()))
    print("END. No wall clock was taken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
