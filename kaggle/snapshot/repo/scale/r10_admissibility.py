"""SATURN, v-main.3M it.16: the admissibility battery for the it.14/15 corpus.

The script line, verbatim:

> SATURN: C-A split integrity = train/eval same graph family or graph in x
> (impact died at linear-probe train 4.93e-08 vs eval 1.478); C-B linear probe
> within-split must read NRMSE > bar; C-C peak activation bytes =
> 4*n*s*d*heads computed in writing vs card bytes.

Run with

    python -m scale.r10_admissibility            # demo(), assert-based, ~20 s
    python -m scale.r10_admissibility --write    # the battery, writes the .jsonl

WHAT IS ADMITTED. The corpus of `scale/r10_corpus_spec.py` (JUPITER, it.14) and
`scale/r10_dual_oracle.py` (SATURN, it.15): 12 rungs on JUPITER's seeds plus 16
widened rungs plus 3 probes, harmonic labels `u` on a connected attachment tree
with chords, boundary data `g ~ U[0,1]`.

HOW THE INSTANCES ARE RECOVERED, WITHOUT RE-STRATIFYING. `make_rung` consumes its
rng in the order graph -> node order -> `g`, and `stratify` consumes NO rng
(`scale/r10_corpus_spec.py:361-397` takes `adj` and `order` and draws nothing).
The stratifier's only output is the boundary SIZE, and every shipped row prints
it. So an instance is recovered exactly from `(n, mean_deg, seed, boundary_size)`
with zero eigensolves -- 121 + 165 of them are skipped -- and the recovery is
BOUND rather than assumed: `u_absorbing_range` recomputed from the recovered
instance must equal the shipped pair, which `verify_recovery` asserts at 0.0.

THE THREE CLAUSES, AND WHERE EACH ILL-POSEDNESS IS REPORTED RATHER THAN REPAIRED.

C-A. **The corpus declares no train/eval split at all.** There is no split field
in either `.jsonl` and no split rule in either report. The clause is therefore
ill-posed as written, and the smallest amendment is one sentence -- *declare which
partition of the corpus is train and which is eval* -- so both partitions a reader
could actually mean are measured instead of one being chosen quietly:
  - SPLIT-ARM: arm `reproduce` against arm `widen`, the only partition the it.15
    corpus itself names.
  - SPLIT-FILE: `results/r10_it14_corpus_spec.jsonl` against
    `results/r10_it15_dual_oracle.jsonl`, the partition a reader gets by taking
    the two shipped files as two datasets.
"Same graph family" is decided by REGENERATION, not by a resemblance statistic:
an instance is in the family iff `spec.draw_graph` on its own `(n, mean_deg,
seed)` reproduces its edge set bit-for-bit. "Graph in both" is decided by a
SHA-256 over the sorted edge list, and the labelled key adds the boundary and `g`.

C-B. The bar is IMPORTED, not re-picked: `rips_gate.FAIL_BAR = 0.9`, by object
identity, so a local re-definition fails `demo()` step 0. That is the repo's own
"the decoder is doing essentially nothing" bar (`scale/rips_gate.py:58-61`) and
it is already used for exactly this anti-triviality role at
`scale/impact.py:1121` ("local r0 NRMSE ... FAIL_BAR 0.9, must be >=0.9").
The feature ladder is DECLARED here before any number was read, four rungs from
degree-only to the most generous strictly-local set, and the verdict is taken on
the most generous one. Two readings are separated and never averaged: HELD-OUT
(the clause) and IN-SAMPLE (the `4.93e-08` shape that started this).

C-C. `4*n*s*d*heads` is computed in writing at the shapes this corpus is used at
and compared against the card. It is then compared against the ACTUAL peak the
shipped harness builds, which is the `[n, s, s]` operator at
`scale/m3_capability.py:144`, larger by `s/d`. And the resource question is
answered separately, because at it.8 this campaign measured that the capability
harness spends no VRAM at all -- `device=cpu (no .cuda() anywhere in this file)`,
`scale/m3_capability.py:261` (the it.8 filing and `scale/vram_gate.py:37` both
cite this as `:120`; at this commit it is `:261`, and the string is unchanged) --
so the binding resource is HOST RSS, `PEAK_RSS_MIB` at
`scale/r10_it8_pricing.py:46`.

FIVE COUNTS, NOT FOUR. passed / failed / errored / inapplicable / not_reached.
The fifth is the category added at it.15 for `widen-n128-d4-t0.905`, which was
never generated; it is carried through this battery rather than dropped, so a
missing instance cannot read as a pass in a second report either.

EVERY CLAUSE IS SHOWN ABLE TO FAIL. `demo()` plants a violation against each of
the three and asserts it is caught, and plants a control against C-A and C-B that
must NOT fire, because a clause that fires on everything is worth as little as one
that fires on nothing.

COST. numpy only, no torch, dense float64 at `n <= 1024`, one process, no GPU, no
training, no capacity sweep. `vram_gate.preflight(0, name='r10-admissibility',
host_mib=256)` is called by both `demo()` and `main()` and the run is skipped if
it refuses.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from scale import r10_corpus_spec as spec
from scale import r10_dual_oracle as dual
from scale import r10_it8_pricing as pricing
from scale import rips_gate
from scale import vram_gate

__all__ = ["FAIL_BAR", "FEATURE_SETS", "PROBE_SEED", "extra_for", "recover",
           "verify_recovery", "graph_key", "labelled_key", "family_check",
           "split_integrity", "features", "probe", "activation_bytes",
           "operator_bytes", "host_rss_fit", "memory_clause", "tally", "demo",
           "main"]

#: IMPORTED BY OBJECT IDENTITY, not re-picked. `scale/rips_gate.py:61`, value
#: 0.9, described there as "the decoder is doing essentially nothing", and
#: already used in exactly this anti-triviality role at `scale/impact.py:1121`.
#: `demo()` step 0 asserts the identity, so a local re-pick fails the self-check.
FAIL_BAR = rips_gate.FAIL_BAR

#: The other side of the same pair, carried only so the report can say where a
#: reading lands relative to both. `scale/rips_gate.py:60`.
PASS_BAR = rips_gate.PASS_BAR

#: The probe's row shuffle. Declared, printed in every row, and the ONLY source
#: of randomness in C-B: the instances themselves are the corpus's.
PROBE_SEED = 0x3A160000

#: THE FEATURE LADDER, DECLARED BEFORE ANY NUMBER WAS READ. Each rung is a
#: superset of the one above it, so the ladder reads as "what does the next kind
#: of feature buy". Every feature is static and strictly local -- a degree, a
#: ball size, a hop count, or a one-hop read of the boundary data. None of them
#: iterates, none of them solves, none of them sees `u`.
#:
#: The VERDICT is taken on the last rung, the most generous one, because the
#: clause is "a linear probe on the features must NOT already beat the bar" and
#: the honest way to run that is to hand the probe everything a local decoder
#: could have. The other three are printed so the ladder is visible and the
#: verdict cannot be read as a feature set chosen to produce it.
FEATURE_SETS = {
    "F0_degree": ("intercept", "deg"),
    "F1_balls": ("intercept", "deg", "ball1", "ball2", "ball3"),
    "F2_geometry": ("intercept", "deg", "ball1", "ball2", "ball3",
                    "n_boundary_nbrs", "hop_to_B", "inv_hop"),
    "F3_boundary_data": ("intercept", "deg", "ball1", "ball2", "ball3",
                         "n_boundary_nbrs", "hop_to_B", "inv_hop", "gbar"),
}
VERDICT_SET = "F3_boundary_data"

#: The shapes this corpus is actually used at. `s` is the graph size, because a
#: graph instance becomes one sequence: the corpus ships `n in {64, 128, 256,
#: 512, 1024}` (`spec.RUNGS`). `d = 16` is `m3_capability.D_MODEL` at
#: `scale/m3_capability.py:79`, "fixed per task spec". `heads = 1` is not a
#: convention, it is the shipped module: `Arm.__init__` builds ONE `wq` and ONE
#: `wk` (`scale/m3_capability.py:108-109`) and `_operator` returns a single
#: `[n, s, s]` tensor (`:144`). There is no head axis to sum over.
CORPUS_S = (64, 128, 256, 512, 1024)
D_MODEL = 16
HEADS = 1
#: Batch sizes: the three the it.8 RSS fit was measured at, plus the two small
#: ones a 28-instance corpus would actually be run at.
BATCHES = (8, 64, 2048, 8192, 32768)

#: The round's stated host budget: a training job holds ~6 GiB of a 16 GiB box
#: and roughly 2 GiB is free. Declared here because `read_host()` moves minute to
#: minute and a plan made against a momentary reading is not a plan.
ROUND_BUDGET_MIB = 2048.0

HERE = Path(__file__).resolve().parents[1]
IT14 = HERE / "results" / "r10_it14_corpus_spec.jsonl"
IT15 = HERE / "results" / "r10_it15_dual_oracle.jsonl"
OUT = HERE / "results" / "r10_it16_admissibility.jsonl"


# --------------------------------------------------------------------------
# Recovering the shipped instances without re-stratifying
# --------------------------------------------------------------------------
def extra_for(n: int, mean_deg: float) -> int:
    """The chord count, exactly as both generators compute it.

    `scale/r10_corpus_spec.py:524` and `scale/r10_dual_oracle.py:312` both write
    `max(int(n * mean_deg / 2.0) - (n - 1), 0)`. Written once here so a third
    copy cannot drift; the value is checked against the recovered edge count.
    """
    return max(int(n * mean_deg / 2.0) - (n - 1), 0)


def recover(n: int, mean_deg: float, seed: int, boundary_size: int):
    """`(adj, order, boundary, g)` for a shipped row, with ZERO eigensolves.

    The rng order is the generators' own -- graph, node permutation, then `g` --
    and `stratify` draws nothing between the second and the third, so supplying
    the boundary SIZE from the shipped row is the whole of what the skipped
    bisection would have returned.
    """
    rng = np.random.default_rng(seed)
    adj = spec.draw_graph(n, extra_for(n, mean_deg), rng)
    order = [int(v) for v in rng.permutation(n)]
    boundary = order[:boundary_size]
    g = np.array(rng.uniform(0.0, 1.0, size=boundary_size))
    return adj, boundary, g


def solve_u(adj, boundary, g):
    """The absorbing oracle on a recovered instance, via the shipped routes."""
    PII, PIB, interior, deg = spec.blocks(adj, boundary)
    return dual.absorbing_extension(PII, PIB, deg, g), interior, deg


def verify_recovery(row) -> dict:
    """BIND the recovery to the shipped row, or the whole battery is measuring a
    lookalike corpus.

    `u_absorbing_range` is a two-float fingerprint of the label vector that the
    it.15 rows carry, so recomputing it from the recovered instance and diffing
    is a check on the graph, the node order, the boundary set, the boundary data
    and the oracle at once. Anything but 0.0 means the recovery is wrong.
    """
    adj, boundary, g = recover(row["n"], row["mean_deg"],
                               int(row["seed"], 16), row["boundary_size"])
    u, interior, _deg = solve_u(adj, boundary, g)
    got = [float(u.min()), float(u.max())]
    want = row.get("u_absorbing_range")
    return {"id": row.get("id"), "interior_recovered": len(interior),
            "interior_shipped": row.get("interior_size"),
            "range_delta": None if want is None
            else max(abs(got[0] - want[0]), abs(got[1] - want[1]))}


# --------------------------------------------------------------------------
# C-A -- SPLIT INTEGRITY
# --------------------------------------------------------------------------
def graph_key(adj) -> str:
    """Canonical identity of a graph: SHA-256 over the sorted edge list.

    The node LABELS are kept, because both splits are drawn by the same
    generator into the same label space `0..n-1`, so two instances sharing an
    edge set under this key really are the same object and not merely isomorphic
    ones. An isomorphism-invariant key would be a weaker statement and a much
    more expensive one; the stronger claim is the one that is cheap here.
    """
    edges = sorted((u, v) for u in adj for v in adj[u] if u < v)
    return hashlib.sha256(repr(edges).encode()).hexdigest()[:16]


def labelled_key(adj, boundary, g) -> str:
    """Identity of the whole labelled instance: graph, boundary, boundary data.

    Two instances agreeing here have BIT-IDENTICAL labels `u`, because `u` is a
    function of exactly these three. That is what makes the shared count a
    statement about leakage rather than about coincidence.
    """
    h = hashlib.sha256(graph_key(adj).encode())
    h.update(repr(list(boundary)).encode())
    h.update(np.asarray(g, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def family_check(adj, n: int, mean_deg: float, seed: int) -> dict:
    """Is this graph in the declared family? Decided by REGENERATION.

    "Same family" is not a resemblance statistic here. The family is
    `spec.draw_graph(n, extra_for(n, mean_deg), default_rng(seed))` -- a uniform
    attachment tree plus chords -- and an instance is in it iff that call
    reproduces its edge set bit-for-bit. A graph drawn from any other process
    fails on the hash, which is what `demo()` plants.
    """
    ref = spec.draw_graph(n, extra_for(n, mean_deg), np.random.default_rng(seed))
    edges = sum(len(adj[v]) for v in adj) // 2
    return {"in_family": graph_key(adj) == graph_key(ref),
            "edges": edges,
            "edges_expected": (n - 1) + extra_for(n, mean_deg)}


def split_integrity(name: str, train, eval_, *, plant_leak=False) -> dict:
    """C-A on one declared split. `train`/`eval_` are lists of recovered dicts
    `{id, n, mean_deg, seed, adj, boundary, g}`.

    Reports, separately: whether both sides are the same family (by
    regeneration), how many GRAPHS appear on both sides, and how many LABELLED
    instances appear on both sides. The two counts are kept apart because they
    cost different things -- a shared graph under a different boundary leaks
    structure, a shared labelled instance leaks the answer.

    `plant_leak` copies one train instance into eval, which is the must-fire.
    """
    ev = list(eval_) + ([dict(train[0], id=train[0]["id"] + "-LEAKED")]
                        if plant_leak and train else [])
    fam = [family_check(r["adj"], r["n"], r["mean_deg"], int(r["seed"], 16))
           for r in list(train) + ev]
    out_of_family = [f for f in fam if not f["in_family"]]

    tr_g = {graph_key(r["adj"]): r["id"] for r in train}
    tr_l = {labelled_key(r["adj"], r["boundary"], r["g"]): r["id"] for r in train}
    shared_graphs, shared_labelled = [], []
    for r in ev:
        gk, lk = graph_key(r["adj"]), labelled_key(r["adj"], r["boundary"], r["g"])
        if gk in tr_g:
            shared_graphs.append({"eval": r["id"], "train": tr_g[gk],
                                  "graph_key": gk,
                                  "labelled_identical": lk in tr_l})
        if lk in tr_l:
            shared_labelled.append({"eval": r["id"], "train": tr_l[lk]})

    leaked_rows = sum(len(r["adj"]) - len(r["boundary"]) for r in ev
                      if labelled_key(r["adj"], r["boundary"], r["g"]) in tr_l)
    eval_rows = sum(len(r["adj"]) - len(r["boundary"]) for r in ev)
    same_family = not out_of_family
    clean = same_family and not shared_graphs
    return {"kind": "C-A", "split": name, "status": "ok" if clean else "failed",
            "train_instances": len(train), "eval_instances": len(ev),
            "same_graph_family": same_family,
            "out_of_family": [f for f in out_of_family],
            "shared_graphs": len(shared_graphs),
            "shared_labelled_instances": len(shared_labelled),
            "shared_detail": shared_graphs[:16],
            "eval_rows": eval_rows, "eval_rows_leaked": leaked_rows,
            "eval_rows_leaked_frac": (leaked_rows / eval_rows) if eval_rows else None,
            "rule": "graph identity = SHA-256 of the sorted edge list on the "
                    "generator's own node labels; labelled identity adds the "
                    "boundary list and the float64 bytes of g; family membership "
                    "= spec.draw_graph(n, extra_for(n, mean_deg), rng(seed)) "
                    "reproduces the edge set bit-for-bit"}


# --------------------------------------------------------------------------
# C-B -- THE WITHIN-SPLIT LINEAR PROBE
# --------------------------------------------------------------------------
def _hops_to_boundary(adj, boundary, cap: int = 8) -> dict[int, int]:
    dist = {b: 0 for b in boundary}
    q = deque(boundary)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return {v: min(dist.get(v, cap), cap) for v in adj}


def features(adj, boundary, g) -> tuple[dict[str, np.ndarray], list[int]]:
    """The declared columns, one row per INTERIOR node, ascending -- the same
    order `spec.blocks` returns `interior` in, so the rows line up with `u`
    without a reindex step that could itself be wrong.

    `gbar` is the only column that touches the boundary DATA, and it is exactly
    one Neumann term: `(1/deg v) * sum over boundary neighbours of g`. It is on
    the ladder because the clause asks whether a linear probe ALREADY beats the
    bar, and withholding the one local feature that obviously carries signal
    would be choosing a probe that fails.
    """
    bset = set(boundary)
    gof = {b: float(g[i]) for i, b in enumerate(boundary)}
    interior = [v for v in sorted(adj) if v not in bset]
    hop = _hops_to_boundary(adj, boundary)
    cols = {name: np.zeros(len(interior)) for name in
            ("intercept", "deg", "ball1", "ball2", "ball3", "n_boundary_nbrs",
             "hop_to_B", "inv_hop", "gbar")}
    for i, v in enumerate(interior):
        d = float(len(adj[v]))
        nb = sum(1 for u in adj[v] if u in bset)
        cols["intercept"][i] = 1.0
        cols["deg"][i] = d
        for r in (1, 2, 3):
            cols[f"ball{r}"][i] = float(len(rips_gate.ball(adj, v, r)))
        cols["n_boundary_nbrs"][i] = float(nb)
        cols["hop_to_B"][i] = float(hop[v])
        cols["inv_hop"][i] = 1.0 / (1.0 + hop[v])
        cols["gbar"][i] = sum(gof[u] for u in adj[v] if u in bset) / d
    return cols, interior


def _design(cols, names) -> np.ndarray:
    return np.column_stack([cols[n] for n in names])


def _insample(x: np.ndarray, y: np.ndarray) -> float:
    w = np.linalg.lstsq(x.T @ x + 1e-6 * np.eye(x.shape[1]), x.T @ y,
                        rcond=None)[0]
    return rips_gate.nrmse(x @ w, y)


def probe(cols, y: np.ndarray, names, *, seed: int = PROBE_SEED) -> dict:
    """One reading. HELD-OUT is the clause; IN-SAMPLE is printed beside it.

    The split is `rips_gate.fit_eval` -- the repo's own half/half least squares
    at `scale/rips_gate.py:163-174` -- on rows shuffled by a declared seed, so
    the halves are not an artefact of the generator's node labelling. Both halves
    come from the SAME instance (or the same arm), which is what "within-split"
    means: this is not a transfer measurement and is not reported as one.

    IN-SAMPLE is the `4.93e-08` shape and is carried so the two readings can
    never be quoted for each other. A held-out reading at the bar with an
    in-sample reading at `1e-08` is a different object from both readings at the
    bar, and the pair says which.
    """
    x = _design(cols, names)
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(y))
    held = rips_gate.fit_eval(x[p], y[p])
    return {"features": names, "n_features": len(names), "n_rows": int(len(y)),
            "n_train": int(len(y) // 2), "n_eval": int(len(y) - len(y) // 2),
            "nrmse_heldout": held, "nrmse_insample": _insample(x, y),
            "label_sd": float(y.std()), "seed": f"0x{seed:08x}"}


def probe_instance(inst) -> dict:
    """C-B on one instance, every rung of the ladder, verdict on the last.

    `label_sd == 0` is INAPPLICABLE, not a pass: `rips_gate.nrmse` returns NaN
    for a constant label and the caller must say so rather than skip it -- the
    discipline that module's own docstring sets at `:121-124`.
    """
    cols, interior = features(inst["adj"], inst["boundary"], inst["g"])
    u, _int2, _deg = solve_u(inst["adj"], inst["boundary"], inst["g"])
    row = {"kind": "C-B", "id": inst["id"], "n": inst["n"],
           "mean_deg": inst["mean_deg"], "seed": inst["seed"],
           "boundary_size": len(inst["boundary"]), "interior": len(interior),
           "bar": FAIL_BAR, "pass_bar": PASS_BAR,
           "bar_source": "rips_gate.FAIL_BAR (scale/rips_gate.py:61)",
           "scope": "within-instance"}
    if float(u.std()) == 0.0:
        row.update(status="inapplicable",
                   reason="label sd is exactly 0; NRMSE is undefined")
        return row
    row["ladder"] = {k: probe(cols, u, list(v)) for k, v in FEATURE_SETS.items()}
    v = row["ladder"][VERDICT_SET]["nrmse_heldout"]
    row.update(verdict_set=VERDICT_SET, nrmse=v, beats_bar=bool(v <= FAIL_BAR),
               status="ok" if v > FAIL_BAR else "failed")
    return row


def probe_pooled(name: str, instances) -> dict:
    """C-B at corpus level: all rows of one split pooled, then split in half.

    Pooling raw `u` across instances is deliberate and is the honest shape of
    the question -- a consumer is handed a batch of instances and does not get
    told which is which -- but it does mean the pooled label carries
    between-instance variance that no local feature can read, so the pooled
    reading is a WEAKER anti-triviality test than the per-instance one and is
    reported beside it, never instead of it.
    """
    parts, ys = [], []
    for inst in instances:
        cols, _interior = features(inst["adj"], inst["boundary"], inst["g"])
        u, _i, _d = solve_u(inst["adj"], inst["boundary"], inst["g"])
        parts.append(cols)
        ys.append(u)
    cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    y = np.concatenate(ys)
    row = {"kind": "C-B", "id": f"pooled-{name}", "scope": "within-split-pooled",
           "instances": len(instances), "bar": FAIL_BAR, "pass_bar": PASS_BAR,
           "ladder": {k: probe(cols, y, list(v)) for k, v in FEATURE_SETS.items()}}
    v = row["ladder"][VERDICT_SET]["nrmse_heldout"]
    row.update(verdict_set=VERDICT_SET, nrmse=v, beats_bar=bool(v <= FAIL_BAR),
               status="ok" if v > FAIL_BAR else "failed")
    return row


# --------------------------------------------------------------------------
# C-C -- PEAK ACTIVATION BYTES, AND WHICH RESOURCE BINDS
# --------------------------------------------------------------------------
def activation_bytes(n: int, s: int, d: int = D_MODEL, heads: int = HEADS) -> int:
    """The clause's formula, verbatim: `4 * n * s * d * heads`.

    4 bytes is float32. This prices ONE `[n, s, d]` activation per head -- the
    q/k projections' output shape at `scale/m3_capability.py:108-109`. It does
    not price the operator, which is the next function and is the actual peak.
    """
    return 4 * n * s * d * heads


def operator_bytes(n: int, s: int, heads: int = HEADS) -> int:
    """What the shipped harness actually peaks at: the `[n, s, s]` operator.

    `Arm._operator` returns `[n, s, s]` (`scale/m3_capability.py:144`) and
    `forward` then computes `a @ x`, so the operator is live and float32. The
    FLOP table already knows the shape -- `4*n*s^2*d_model` is a base term at
    `scale/m3_flops.py:260` -- so the `s^2` is not a new claim, it is the same
    tensor priced for memory instead of arithmetic. Larger than the clause's
    figure by exactly `s / d`.
    """
    return 4 * n * s * s * heads


def host_rss_fit():
    """`(intercept, slope, holdout_residual_mib)` for peak host RSS vs batch size.

    RECONSTRUCTED, not copied. `pricing.PEAK_RSS_MIB`
    (`scale/r10_it8_pricing.py:46`) is 901.2 / 1567.9 / 4275.1 MiB at
    n = 2048 / 8192 / 32768, and the published line `665.5 + 0.1102*n` is the
    line through the two LARGEST points with the smallest held out -- which is
    why it.8 could report it as accurate "to within 10 MiB on a held-out point".
    An unweighted three-point least squares gives `672.2 + 0.10992*n` instead, a
    different line, so the fit rule is stated here rather than left to be
    inferred from a coefficient that nearly matches.

    `demo()` asserts both the published coefficients and the held-out residual,
    so a change to those three measurements moves this line and fails the
    self-check instead of silently disagreeing with the published one.
    """
    ns = sorted(pricing.PEAK_RSS_MIB)
    hold, (n1, n2) = ns[0], ns[-2:]
    y1, y2 = pricing.PEAK_RSS_MIB[n1], pricing.PEAK_RSS_MIB[n2]
    b = (y2 - y1) / (n2 - n1)
    a = y1 - b * n1
    resid = abs(pricing.PEAK_RSS_MIB[hold] - (a + b * hold))
    return float(a), float(b), float(resid)


def memory_clause(batches=BATCHES, ss=CORPUS_S) -> list[dict]:
    """C-C, one row per shape, both resources priced, each in its own column.

    The VRAM rows are `inapplicable`, not `passed`. The consumer of this corpus
    is the capability harness, which is CPU-only by construction
    (`scale/m3_capability.py:261`: `device=cpu (no .cuda() anywhere in this
    file)`), so a VRAM verdict on it is a true statement about a resource the job
    does not spend -- the exact shape it.8 catalogued when
    `require(4275, name='capacity-sweep')` returned FITS citing 7162 MiB free.
    The HOST rows are the binding ones and they are `passed` or `failed`.
    """
    card, host = vram_gate.read_card(), vram_gate.read_host()
    a, b, _resid = host_rss_fit()
    rows = []
    for n in batches:
        rss = a + b * n
        for s in ss:
            act = activation_bytes(n, s)
            op = operator_bytes(n, s)
            rows.append({
                "kind": "C-C", "batch_n": n, "s": s, "d": D_MODEL,
                "heads": HEADS,
                "activation_bytes_formula": act,
                "activation_mib_formula": act / 2 ** 20,
                "operator_bytes_shipped_peak": op,
                "operator_mib_shipped_peak": op / 2 ** 20,
                "understated_by": op / act,
                "card_bytes": None if card is None else card.total_mib * 2 ** 20,
                "card_free_mib": None if card is None else card.free_mib,
                "vram_fits": None if card is None
                else bool(act / 2 ** 20 * 1.25 <= card.free_mib),
                "vram_status": "inapplicable",
                "vram_reason": "the capability harness spends no VRAM "
                               "(scale/m3_capability.py:261, device=cpu, no "
                               ".cuda() anywhere in this file)",
                "host_peak_rss_mib": rss,
                "host_available_mib": None if host is None else host.available_mib,
                # The VERDICT prices against the round's DECLARED budget, not
                # against `read_host()`. The measured reading moved 1142 -> 6946
                # MiB across seven runs inside twenty minutes -- and 1142 -> 5590
                # inside one thirteen-second run -- so a status taken from it is
                # a statement about the second it ran in. The measured reading is
                # carried in its own column beside it.
                "host_budget_mib": ROUND_BUDGET_MIB,
                "host_status": ("ok" if rss * 1.25 <= ROUND_BUDGET_MIB
                                else "failed"),
                "host_status_measured": ("errored" if host is None else
                                         "ok" if rss * 1.25 <= host.available_mib
                                         else "failed"),
            })
    return rows


def binding_crossover(host_available_mib: float, card_free_mib: float,
                      s: int = max(CORPUS_S)) -> dict:
    """The batch size at which each resource refuses, at the corpus's largest
    `s`. The point of the pair is the ratio between them."""
    a, b, _resid = host_rss_fit()
    host_n = (host_available_mib / 1.25 - a) / b
    per_example_mib = activation_bytes(1, s) / 2 ** 20
    vram_n = card_free_mib / 1.25 / per_example_mib
    return {"s": s, "host_refuses_at_batch": host_n,
            "vram_refuses_at_batch": vram_n,
            "ratio": vram_n / host_n if host_n else None,
            "vram_load_at_host_refusal_mib": host_n * per_example_mib * 1.25,
            "vram_load_at_host_refusal_frac": host_n * per_example_mib * 1.25
            / card_free_mib}


# --------------------------------------------------------------------------
# Counts, five of them
# --------------------------------------------------------------------------
def tally(rows) -> dict:
    """passed / failed / errored / inapplicable / not_reached, each separate.

    `not_reached` is the it.15 category and it is carried, not dropped: an
    instance that was never generated is not a pass in this report either.
    """
    def c(status):
        return sum(r.get("status") == status for r in rows)
    return {"total": len(rows), "passed": c("ok"), "failed": c("failed"),
            "errored": c("errored"), "inapplicable": c("inapplicable"),
            "not_reached": c("not_reached")}


# --------------------------------------------------------------------------
# The corpus, loaded and recovered
# --------------------------------------------------------------------------
def _load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_instances():
    """Every rung of both files, recovered. Returns
    `(it14, reproduce, widen, not_reached_rows, recovery_checks)`."""
    it15 = _load(IT15)
    it14 = _load(IT14)
    checks, out = [], {}
    for tag, rows in (("it14", [r for r in it14 if r["kind"] == "rung"
                                and r["status"] == "ok"]),
                      ("it15", [r for r in it15 if r["kind"] == "rung"
                                and r["status"] == "ok"])):
        recovered = []
        for i, r in enumerate(rows):
            adj, boundary, g = recover(r["n"], r["mean_deg"],
                                       int(r["seed"], 16), r["boundary_size"])
            rid = r.get("id") or f"it14-n{r['n']}-d{r['mean_deg']:g}-#{i}"
            recovered.append({"id": rid, "arm": r.get("arm", "it14"),
                              "n": r["n"], "mean_deg": r["mean_deg"],
                              "seed": r["seed"], "adj": adj,
                              "boundary": boundary, "g": g})
            if r.get("u_absorbing_range") is not None:
                checks.append(verify_recovery(r))
        out[tag] = recovered
    not_reached = [{"kind": "C-A", "id": r.get("id") or r.get("seed"),
                    "status": "not_reached",
                    "reason": r.get("reason", "no instance was ever generated")}
                   for r in it15 if r["kind"] == "not_generated"]
    return (out["it14"],
            [r for r in out["it15"] if r["arm"] == "reproduce"],
            [r for r in out["it15"] if r["arm"] == "widen"],
            not_reached, checks)


# --------------------------------------------------------------------------
# demo -- every clause shown able to fail, and shown not to fire on a control
# --------------------------------------------------------------------------
def demo() -> None:
    fits, why = vram_gate.preflight(0, name="r10-admissibility", host_mib=256)
    print(why, flush=True)
    if not fits:
        print("SKIPPED: the host gate refused. A refusal is the result.")
        return

    # 0. The bar is IMPORTED, by object identity.
    assert FAIL_BAR is rips_gate.FAIL_BAR, "the bar was re-picked locally"
    assert FAIL_BAR == 0.9, FAIL_BAR

    it14, repro, widen, not_reached, checks = load_instances()
    assert len(it14) == 12 and len(repro) == 12 and len(widen) == 16
    assert len(not_reached) == 1, not_reached

    # 1. The recovery is BOUND to the shipped rows, or nothing below is about
    #    the shipped corpus.
    worst = max(c["range_delta"] for c in checks if c["range_delta"] is not None)
    assert worst == 0.0, f"recovery does not reproduce the shipped labels: {worst}"
    print(f"  recovery bound to {len(checks)} shipped rows, "
          f"max u-range delta {worst}")

    # 2. C-A on the two declared splits.
    arm = split_integrity("SPLIT-ARM reproduce|widen", repro, widen)
    fil = split_integrity("SPLIT-FILE it14|it15", it14, repro + widen)
    print(f"  C-A SPLIT-ARM  : family={arm['same_graph_family']} "
          f"shared_graphs={arm['shared_graphs']}")
    print(f"  C-A SPLIT-FILE : family={fil['same_graph_family']} "
          f"shared_graphs={fil['shared_graphs']} "
          f"labelled={fil['shared_labelled_instances']}")
    assert arm["shared_graphs"] == 0, arm
    assert fil["shared_graphs"] == 12, fil

    # 2a. MUST-FIRE: a leaked instance is caught.
    leaked = split_integrity("plant-leak", repro, widen, plant_leak=True)
    assert leaked["shared_graphs"] == 1 and leaked["status"] == "failed", leaked
    assert leaked["shared_labelled_instances"] == 1
    # 2b. MUST-FIRE: an out-of-family graph is caught, by regeneration.
    ring = {v: {(v - 1) % 32, (v + 1) % 32} for v in range(32)}
    alien = [{"id": "plant-ring", "n": 32, "mean_deg": 4.0, "seed": "0x00000001",
              "adj": ring, "boundary": [0, 1], "g": np.array([1.0, 0.0])}]
    fam = split_integrity("plant-family", repro, alien)
    assert not fam["same_graph_family"] and fam["status"] == "failed", fam
    # 2c. MUST-NOT-FIRE: the clean split above did not fire. Both directions seen.
    assert arm["status"] == "ok", arm

    # 3. C-B on one instance, both plants.
    inst = repro[0]
    row = probe_instance(inst)
    print(f"  C-B {row['id']}: ladder " + " ".join(
        f"{k}={row['ladder'][k]['nrmse_heldout']:.4f}" for k in FEATURE_SETS))
    cols, _interior = features(inst["adj"], inst["boundary"], inst["g"])
    names = list(FEATURE_SETS[VERDICT_SET])
    # 3a. MUST-FIRE: a label inside the feature span must be read at ~0.
    planted = 3.0 * cols["deg"] - 1.5 * cols["ball1"] + 7.0
    hit = probe(cols, planted, names)["nrmse_heldout"]
    # 1e-3, not 1e-12: `fit_eval` carries a `1e-6 * I` ridge (rips_gate.py:172)
    # and `ball3` runs to ~60, so an exactly-representable label is recovered to
    # ~1e-05 rather than to machine zero. Four orders under the bar either way.
    assert hit < 1e-3, f"the probe cannot read a label in its own span: {hit}"
    assert hit <= FAIL_BAR, hit
    # 3b. MUST-NOT-FIRE: white noise must sit at the mean predictor.
    noise = np.random.default_rng(PROBE_SEED + 1).normal(size=len(cols["deg"]))
    miss = probe(cols, noise, names)["nrmse_heldout"]
    assert miss > FAIL_BAR, f"the probe read structure in noise: {miss}"
    print(f"  C-B must-fire planted={hit:.3e} (bar {FAIL_BAR}), "
          f"must-not-fire noise={miss:.4f}")

    # 4. C-C in writing, and which resource binds.
    assert activation_bytes(8, 1024) == 4 * 8 * 1024 * 16 * 1
    assert operator_bytes(8, 1024) == 4 * 8 * 1024 * 1024 * 1
    assert operator_bytes(8, 1024) // activation_bytes(8, 1024) == 1024 // 16
    a, b, resid = host_rss_fit()
    assert abs(a - 665.5) < 0.05 and abs(b - 0.1102) < 5e-5, (a, b)
    # 10.1 MiB, measured. it.8 reported this held-out point as accurate "to
    # within 10 MiB"; it is 10.1. The bound is stated at the measured value
    # rather than the published one, so the number is the check.
    assert resid <= 10.1 + 1e-9, resid
    print(f"  C-C host RSS = {a:.1f} + {b:.4f}*n MiB, held-out residual "
          f"{resid:.1f} MiB at n={min(pricing.PEAK_RSS_MIB)}")
    card, host = vram_gate.read_card(), vram_gate.read_host()
    # 4a. MUST-FIRE, host side: a request past free host RAM is refused.
    if host is not None:
        bad, _w = vram_gate.preflight(0, name="plant-host",
                                      host_mib=host.available_mib * 4)
        assert not bad, "the host clause passed a request 4x free host RAM"
    # 4b. MUST-FIRE, card side: a request past the card is refused.
    if card is not None:
        bad, _w = vram_gate.preflight(card.total_mib * 10, name="plant-vram")
        assert not bad, "the VRAM clause passed a request 10x the card"
    # BOTH resources, or neither: `binding_crossover` reads `host.available_mib`,
    # and `read_host()` returns None when psutil is absent
    # (`scale/vram_gate.py:122-125`). Guarding this on the card alone would raise
    # AttributeError on a box with nvidia-smi and no psutil -- unreachable here,
    # since psutil is installed, which is exactly why it needs the guard rather
    # than a run to find it. A module that prices memory must survive being
    # unable to read memory.
    if card is not None and host is not None:
        cross = binding_crossover(host.available_mib, card.free_mib)
        print(f"  C-C crossover at s={cross['s']}: host refuses at batch "
              f"{cross['host_refuses_at_batch']:.0f}, VRAM at "
              f"{cross['vram_refuses_at_batch']:.0f} "
              f"({cross['ratio']:.1f}x later); VRAM load at the host refusal is "
              f"{cross['vram_load_at_host_refusal_frac'] * 100:.1f}% of free")
        # 4c. The clause as written CANNOT fail at any shape this corpus is used
        #     at -- that is the finding, and it is asserted rather than asserted
        #     away.
        assert cross["vram_refuses_at_batch"] > cross["host_refuses_at_batch"], cross

    print("demo OK")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--write" not in argv:
        demo()
        return 0
    fits, why = vram_gate.preflight(0, name="r10-admissibility", host_mib=256)
    print(why, flush=True)
    if not fits:
        print("REFUSED: no battery run. A gate refusal is the reportable result.")
        return 1

    t0 = time.perf_counter()
    it14, repro, widen, not_reached, checks = load_instances()
    rows = [{"kind": "recovery_check", "checked": len(checks),
             "max_u_range_delta": max(c["range_delta"] for c in checks
                                      if c["range_delta"] is not None),
             "status": "ok"}]

    # C-A
    rows.append(split_integrity("SPLIT-ARM reproduce|widen", repro, widen))
    rows.append(split_integrity("SPLIT-FILE it14|it15", it14, repro + widen))
    rows.append(dict(split_integrity("plant-leak (must-fire)", repro, widen,
                                     plant_leak=True), role="must_fire"))
    rows.extend(not_reached)

    # C-B
    for inst in repro + widen:
        rows.append(probe_instance(inst))
    rows.append(probe_pooled("reproduce", repro))
    rows.append(probe_pooled("widen", widen))

    # C-C
    rows.extend(memory_clause())
    card, host = vram_gate.read_card(), vram_gate.read_host()
    if card is not None and host is not None:
        # Two budgets, because the measured one is not stable. `read_host()`
        # returned 1142 / 2351 / 3178 / 4615 / 5229 / 5770 / 6946 MiB available
        # across seven runs of this module inside twenty minutes, and inside the
        # LAST of those runs it moved from 1142 MiB at preflight to an implied
        # 5590 MiB thirteen seconds later. That is the aggregate-occupancy point
        # `scale/vram_gate.py:14-30` makes, arriving on the host side. So the
        # crossover is reported at the reading AND at the round's declared
        # budget, and the declared one is the number a plan is made against.
        rows.append(dict(binding_crossover(host.available_mib, card.free_mib),
                         kind="C-C-crossover", budget="measured available",
                         host_available_mib=host.available_mib, status="ok"))
        rows.append(dict(binding_crossover(ROUND_BUDGET_MIB, card.free_mib),
                         kind="C-C-crossover", budget="round declared 2 GiB",
                         host_available_mib=ROUND_BUDGET_MIB, status="ok"))

    rows.append({"kind": "summary", "seconds": round(time.perf_counter() - t0, 1),
                 "C-A": tally([r for r in rows if r["kind"] == "C-A"]),
                 "C-B": tally([r for r in rows if r["kind"] == "C-B"]),
                 "C-C_host": {"passed": sum(r.get("host_status") == "ok"
                                            for r in rows),
                              "failed": sum(r.get("host_status") == "failed"
                                            for r in rows),
                              "vram_inapplicable": sum(
                                  r.get("vram_status") == "inapplicable"
                                  for r in rows)}})
    OUT.write_text("".join(json.dumps(r, sort_keys=True, default=float) + "\n"
                           for r in rows))
    print(f"wrote {OUT} -- {len(rows)} rows, "
          f"{rows[-1]['seconds']} s")
    print(json.dumps(rows[-1], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
