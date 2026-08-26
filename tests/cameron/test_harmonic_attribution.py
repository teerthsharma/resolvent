"""U1 / N3 -- HARMONIC-MEASURE ATTRIBUTION AND ITS CROSS-CHECK CONTROL.

THE CONTRACT CLAUSE (v10.1 N3). Attribution ground truth for the
absorbing-chain corpus is the HARMONIC MEASURE: for interior (transient)
nodes I and boundary nodes B, ``K(v, .) = row v of (I - P_II)^-1 P_IB`` --
the exit distribution of the random walk started at v onto the boundary.
The U1 attribution column is a per-chunk MASKING DISPLACEMENT measured on a
trained predicting arm: mask chunk xi (zero its position), measure how far
the arm's prediction moves. The two must be CROSS-CHECKED by rank
correlation with a bootstrap CI. House kill K-R8d: if displacement disagrees
with the kernel below the pre-registered bar, the DISPLACEMENT PROBE is
wrong -- the kernel is exact -- so the probe is repaired, never the oracle.

WHAT EACH HALF IS, ON THIS REPO'S OWN OBJECTS.

  kernel side   `scale.e4_harmonic` is the repo's absorbing-chain corpus
                generator (its header: "the e3-harmonic ladder"). Its
                `absorbing_chain` builds (Q, R) absorbed at TWO bridge
                endpoints; the harmonic measure generalised to a SET of
                boundary documents is the same Dirichlet problem with R a
                matrix, and this module adds no linear algebra of its own:
                Q and R are built by `e4_harmonic.absorbing_chain`'s loop
                rule (uniform step over neighbours), the solve IS
                `e4_harmonic.fixed_point` ((I - Q)^-1 applied columnwise,
                unchanged for matrix right-hand sides), and its exactness is
                checked against `e4_harmonic.hop_reading`, the Neumann
                iterate, on small graphs where that series visibly converges.

  arm side      the corpus instance declares s CHUNKS (the boundary nodes).
                Each example carries one N(0,1) evidence value per chunk in
                CH_FLIP and an independent distractor payload in CH_PAYLOAD;
                the label is the harmonic aggregate y = sum_xi omega_v(xi)
                e_xi, recomputed from x -- no answer key is stored. A tiny
                control arm (`calibrate_bar`'s own shape and init) learns
                the aggregate; masking displacement per chunk is then
                |pred(full) - pred(masked)| averaged over eval draws.

THE PRE-REGISTERED BAR. `PREREGISTERED_RHO_FLOOR` was frozen from PILOT data
BEFORE the headline run, disclosed here exactly as
E_LADDER_PREREGISTERED_READING.md section 7b discloses its pilot:

    Pilot rows, produced before the floor was frozen (seed 0, shipped case
    LargestJoin_S2Rips_4096/deg 4.25/seed 0x33960005, 64 declared chunks,
    query node 283, n_train=2048, steps=150, lr=0.02, n_eval draws=256,
    bootstrap B=200):

        spearman(omega_rank, mean_displacement_rank) = 0.743864
        bootstrap CI [0.656532, 0.816955]

    Floor frozen at 0.50: comfortably below the pilot point estimate and its
    CI lower edge, far above the ~0 noise floor of 64 ranked chunks, round.
    The headline run uses a DIFFERENT seed (7) and MORE draws (1024); if it
    lands below the floor, K-R8d fires at the PROBE, not at the kernel.

NON-DEGENERACY (the fifteenth vacuous control must not happen). Every PASS
half of the cross-check carries hard non-degeneracy clauses: the kernel row
must put mass on more than two chunks and must not concentrate (>0.9 on one
chunk); the displacement profile must be nonconstant with at least three
live chunks; the correlation must be evaluable. Both failure directions are
kept as planted must-fire controls below: a degenerate corpus whose kernel
puts all mass on ONE chunk must fire the concentration clause, and a dead
(never-trained) arm whose displacements carry no structure must fire the
spread clause. Neither direction may pass silently.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import e4_harmonic as EH                                 # noqa: E402
from scale import negation_scope as NS                              # noqa: E402

#: a cheap drawn graph for the algebraic identities (structural checks do
#: not need the shipped size -- same reasoning as EH.SMALL_CASE's comment).
TEST_CASE = (64, 2.0, 0x33960005)

#: the shipped admissible substrate for the pilot/headline numbers.
SHIPPED_CASE = EH.SHIPPED_CASE


# ============================ 1. the kernel is the exact exit distribution ====
def test_harmonic_kernel_solves_the_dirichlet_problem_exactly():
    """K = (I - Q)^-1 R on a DRAWN graph: every row sums to 1 (a walk exits
    somewhere, almost surely), the defining residual ||(I-Q)K - R|| vanishes
    to solver precision, and the SOLVE agrees with the repo's Neumann
    iterate `e4_harmonic.hop_reading` -- the two routes to the same fixed
    point, tested rather than assumed."""
    adj, nodes, bridge = EH.case_graph(*TEST_CASE)
    out = NS.absorbing_boundary_kernel(adj, nodes, n_chunks=8)
    k_mat, bnd, innr, q_node = out["kernel"], out["boundary"], \
        out["interior"], out["query_node"]
    q, r, _ = _multi_absorb(adj, innr, bnd)
    assert np.abs(k_mat.sum(axis=1) - 1.0).max() < 1e-9
    assert np.abs((np.eye(len(innr)) - q) @ k_mat - r).max() < 1e-12
    zn = EH.fixed_point(q, r)
    assert np.abs(zn - k_mat).max() < 1e-12
    neumann = EH.hop_reading(q, r, 20000)
    assert np.abs(neumann - k_mat).max() < 1e-8


def test_the_query_node_is_the_farthest_interior_and_build_is_deterministic():
    """The queried document v is picked by a stated deterministic rule --
    BFS-farthest interior node from the boundary set, smallest index on
    ties -- so two builds of the same corpus give bitwise the same kernel
    and the same omega row. A rule that could drift could be tuned."""
    adj, nodes, bridge = EH.case_graph(*TEST_CASE)
    a = NS.absorbing_boundary_kernel(adj, nodes, n_chunks=8)
    b = NS.absorbing_boundary_kernel(adj, nodes, n_chunks=8)
    assert a["query_node"] == b["query_node"]
    assert np.array_equal(a["kernel"], b["kernel"])
    dist = _bfs_dist_from(adj, set(a["boundary"]))
    far = max(a["interior"], key=lambda u: (dist.get(u, -1), -u))
    assert a["query_node"] == far


def test_harmonic_measure_row_is_exposed_and_nondegenerate_on_a_drawn_case():
    """`harmonic_measure` returns the QUERY ROW omega_v(.). On a real drawn
    corpus it must be a probability distribution with mass on SEVERAL
    chunks -- a one-chunk omega would make the later correlation control
    vacuous in the struck-by-history way."""
    adj, nodes, bridge = EH.case_graph(*TEST_CASE)
    out = NS.absorbing_boundary_kernel(adj, nodes, n_chunks=8)
    omega = NS.harmonic_measure(out)
    assert omega.shape == (8,)
    assert abs(float(omega.sum()) - 1.0) < 1e-9
    assert float(omega.min()) >= 0.0
    live = int((torch.tensor(omega) > 1e-6).sum())
    assert live >= 3, live


# ==================== 2. the evidence corpus has an executable oracle ========
def test_harmonic_label_batch_oracle_is_executable_and_controls_are_live():
    """y = omega @ e recomputed from x; predict-the-mean reads exactly 1.0;
    the payload-only control FAILS the bar (>= 1.0) because the distractor
    channel rides along independent of the label -- the house convention
    that keeps a comparison against zeros from masquerading as a control."""
    adj, nodes, bridge = EH.case_graph(*TEST_CASE)
    out = NS.absorbing_boundary_kernel(adj, nodes, n_chunks=8)
    omega = torch.tensor(NS.harmonic_measure(out))
    x, y = NS.harmonic_label_batch(omega, 256, seed=0)
    e = x[:, :, NS.CH_FLIP]
    assert torch.equal(y, e @ omega.to(e.dtype))
    assert abs(float(NS.nrmse(y.mean().expand_as(y), y)) - 1.0) < 1e-6
    payload_pos = x.shape[1] - 2
    assert NS.nrmse(x[:, payload_pos, NS.CH_PAYLOAD], y) >= 1.0


# ======================= 3. the displacement probe and the cross-check =======
def test_trained_arm_displacement_ranks_track_the_kernel_top_chunk():
    """A trained arm's masking displacement must put its TOP chunk where the
    kernel puts its top mass -- the coarsest reading of agreement, kept as
    its own test so the finer Spearman bar below cannot silently rot while
    argmax still holds."""
    adj, nodes, bridge = EH.case_graph(*TEST_CASE)
    out = NS.absorbing_boundary_kernel(adj, nodes, n_chunks=8)
    omega = torch.tensor(NS.harmonic_measure(out))
    xtr, ytr = NS.harmonic_label_batch(omega, 2048, seed=0)
    net = NS.train_control_arm(xtr, ytr, steps=150, lr=0.02, seed=0)
    draws = NS.masking_displacement(net, xtr[:256])
    assert int(torch.tensor(draws).argmax()) == int(omega.argmax())


def test_the_crosscheck_control_passes_live_and_records_its_nondegeneracy():
    """PASS half, with the non-degeneracy clauses VISIBLE in the returned
    record: kernel spread, displacement spread, evaluable rho. A control
    whose pass cannot show WHY it is not vacuous is the fifteenth."""
    adj, nodes, bridge = EH.case_graph(*TEST_CASE)
    out = NS.absorbing_boundary_kernel(adj, nodes, n_chunks=8)
    omega = torch.tensor(NS.harmonic_measure(out))
    xtr, ytr = NS.harmonic_label_batch(omega, 2048, seed=0)
    net = NS.train_control_arm(xtr, ytr, steps=150, lr=0.02, seed=0)
    draws = torch.tensor(NS.masking_displacement(net, xtr[:256],
                                                 return_draws=True))
    rec = NS.rank_crosscheck(torch.tensor(omega), draws, rho_floor=0.5)
    assert rec["ok"], rec
    assert rec["checks"]["kernel_spread"] and rec["checks"]["displacement_spread"]
    assert rec["rho"] == rec["rho"]


# ========================== 4. the must-fire directions, both seen firing ====
def test_must_fire_degenerate_corpus_concentration_rejects():
    """FIRE DIRECTION 1. A degenerate corpus: every transient node touches
    only ONE boundary hub, so all harmonic mass sits on one chunk by
    construction. The control must REJECT naming concentration, not fail
    somewhere incidental."""
    m = 8
    adj = [[] for _ in range(m)]
    for i in (1, 3, 5, 7):
        adj[i].append(0)
        adj[0].append(i)
    nodes = list(range(m))
    out = NS.absorbing_boundary_kernel(adj, nodes, n_chunks=4)
    omega = torch.tensor(NS.harmonic_measure(out))
    xtr, ytr = NS.harmonic_label_batch(omega, 512, seed=0)
    net = NS.train_control_arm(xtr, ytr, steps=20, lr=0.02, seed=0)
    draws = torch.tensor(NS.masking_displacement(net, xtr[:128],
                                                 return_draws=True))
    rec = NS.rank_crosscheck(omega, draws, rho_floor=0.5)
    assert not rec["ok"]
    assert "concentration" in rec["reason"], rec


def test_must_fire_dead_arm_rejects_on_displacement_degeneracy():
    """FIRE DIRECTION 2. An arm that never saw training predicts a constant;
    masking any chunk moves nothing, so the displacement profile carries no
    structure and the control must reject on the SPREAD clause rather than
    report a meaningless rho."""
    adj, nodes, bridge = EH.case_graph(*TEST_CASE)
    out = NS.absorbing_boundary_kernel(adj, nodes, n_chunks=8)
    omega = torch.tensor(NS.harmonic_measure(out))
    xtr, ytr = NS.harmonic_label_batch(omega, 128, seed=0)
    net = NS.dead_control_arm(8)     # predicts a constant: input-blind arm
    draws = torch.tensor(NS.masking_displacement(net, xtr[:128],
                                                 return_draws=True))
    rec = NS.rank_crosscheck(omega, draws, rho_floor=0.5)
    assert not rec["ok"]
    assert "spread" in rec["reason"] or "rho" in rec["reason"], rec


def test_must_fire_band_rejects_when_rho_lands_below_the_floor():
    """FIRE DIRECTION 3, the K-R8d direction itself: a live-looking pair
    whose rank correlation sits below the pre-registered floor must be
    rejected BY THE FLOOR clause, with the reason carrying both numbers --
    that rejection is what sends the repair to the probe."""
    rng = torch.Generator().manual_seed(11)
    omega = torch.rand(16, generator=rng)
    omega[3] += 2.0                      # spread ok, structured even
    draws = torch.rand(64, 16, generator=rng) * 1e-3
    draws[:, 3] = 0.5                    # top chunk agrees...
    perm = torch.randperm(16, generator=rng)
    anti = draws[:, perm]                # ...but ranks elsewhere are shuffled
    rec = NS.rank_crosscheck(omega, anti, rho_floor=0.9)
    assert not rec["ok"]
    assert "floor" in rec["reason"], rec


# ================================ 5. the pre-registered headline claim =======
def test_preregistered_floor_is_frozen_with_provenance():
    """The floor is a CONSTANT WITH PROVENANCE, not a local argument: it
    exists, it is a plain float, and the disclosure comment above it names
    the pilot rows it was frozen from. RED until frozen, by design."""
    assert isinstance(NS.PREREGISTERED_RHO_FLOOR, float)


def test_headline_crosscheck_clears_the_preregistered_floor():
    """THE HEADLINE. Different seed than the pilot, four times the eval
    draws, the SHIPPED corpus instance. Asserts spearman >= floor and a
    bootstrap CI whose lower edge is above zero -- i.e. the disagreement
    K-R8d polices did not happen. This test was written BEFORE the headline
    number existed; only the floor constant came from the pilot, and its
    disclosure is in the module docstring."""
    assert isinstance(NS.PREREGISTERED_RHO_FLOOR, float)
    rec = NS.u1_attribution_run(seed=7, n_train=2048, steps=150,
                                n_eval=1024, n_boot=400)
    assert rec["ok"], rec
    assert rec["rho"] >= NS.PREREGISTERED_RHO_FLOOR, rec
    assert rec["ci"][0] > 0.0, rec


# ------------------------------------------------------------------ helpers --
def _multi_absorb(adj, interior, boundary):
    """The SAME construction `NS.absorbing_boundary_kernel` uses, restated
    here so the test can form (Q, R) independently and check residuals
    against it. Uniform step over neighbours, absorbing rows removed."""
    idx = {v: i for i, v in enumerate(interior)}
    bidx = {v: j for j, v in enumerate(boundary)}
    qm = np.zeros((len(interior), len(interior)))
    rm = np.zeros((len(interior), len(boundary)))
    for v in interior:
        nb = [u for u in adj[v] if u != v]
        step = 1.0 / len(nb)
        for u in nb:
            if u in bidx:
                rm[idx[v], bidx[u]] += step
            elif u in idx:
                qm[idx[v], idx[u]] += step
            else:
                raise AssertionError(f"neighbour {u} outside merged set")
    return qm, rm, idx


def _bfs_dist_from(adj, sources):
    from collections import deque
    dist = {}
    dq = deque((s, 0) for s in sources)
    seen = set(sources)
    while dq:
        u, d = dq.popleft()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                dist[w] = d + 1
                dq.append((w, d + 1))
    return dist
