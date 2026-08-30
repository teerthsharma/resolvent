"""SATURN, v-main.3M it.15: the generator as a corpus, and a dual oracle that HALTS.

The script line, verbatim:

> SATURN: generator + dual oracle; per instance assert
> `|u_absorbing - u_kirchhoff|_inf <= 1e-10` (matrix-tree cofactor route) --
> disagreement halts the corpus, not the reading.

Run with

    python -m scale.r10_dual_oracle            # demo(), assert-based, ~3 s
    python -m scale.r10_dual_oracle --write    # the corpus, ~150 s, writes .jsonl

WHAT IS NEW HERE, AND WHAT IS BORROWED. The generator, `lambda_2`, the
stratifier and the absorbing oracle are JUPITER's (`scale/r10_corpus_spec.py`,
it.14) and are imported, not rewritten -- arm `reproduce` runs his exact seeds so
its `lambda_2` column is byte-comparable against
`results/r10_it14_corpus_spec.jsonl`. What is new is (1) the Kirchhoff route
generalised off the `|B| = 2` subfamily to the whole corpus, (2) the halt, and
(3) the four-way count that keeps an unreachable instance from reading as a pass.

THE KIRCHHOFF ROUTE, GENERALISED. `scale/kirchhoff.harmonic_measure` computes
`omega_x = M_xa / M_aa` on the `b`-grounded Laplacian and applies exactly when
`|B| = 2` and `g = (1, 0)`. The corpus has `|B|` from 4 to 127 and a random `g`,
so the identity is used one boundary node at a time. Ground at `B \\ {b}` --
delete those rows and columns of the SYMMETRIC unnormalised `L = D - A`, keeping
`b` as an ordinary column -- solve `K_b z = e_b` once, and read

    omega^{(b)}_v = z_v / z_b = F(v ~ b ; B) / F(B),

the ratio of the number of spanning `|B|`-forests, one boundary node per tree,
that put `v` in `b`'s tree, to the number of such forests. That is the all-minors
matrix-tree theorem: `det(L` with rows/cols `S` deleted`) = F(S)`, so
`z_b = det K_{B} / det K_{B\\{b}} = F(B) / F(B \\ {b})` and `z_v` is the cofactor
above it. `u_kirchhoff = sum_b g_b omega^{(b)}`, by linearity of the harmonic
extension in its boundary data. `demo` ties the ratio to literal spanning-forest
counts on a 5-node graph with `kirchhoff.brute_force_separating_forests`, and
checks the route reduces to the shipped `harmonic_measure` where that applies.

WHAT IS STRUCTURALLY DIFFERENT, IN ONE SENTENCE. The absorbing route solves the
nonsymmetric row-normalised `I - P_II = D_I^-1 L_II` on the interior alone
against the right-hand side `P_IB g` and reads `u` off directly; the Kirchhoff
route solves the symmetric unnormalised `L` on `interior + {b}` -- a different
matrix, one size larger, differently indexed -- against `e_b`, and recovers each
value as a RATIO of two entries of that one column, a division the other route
never performs.

AND WHAT THEY SHARE, WHICH IS THE HONEST HALF OF THAT SENTENCE. `I - Q = D^-1 L_T`
is a diagonal relation, which `scale/kirchhoff.py:57-62` states about its own
pair. So the two routes share the adjacency data and the degrees read off it, and
a defect that corrupts BOTH identically is invisible to the pair. That is not
hypothesised here, it is measured: the SHARED plant puts the `deg + 1` miscount in
both routes at once and the pair agrees to `1e-16`, which is the blind spot
stated as a number instead of argued away.

THE TOLERANCE IS IMPORTED. `AGREEMENT_TOL = kirchhoff.AGREEMENT_TOL`,
`scale/kirchhoff.py:106`, value `1e-10`, set there from a measured window
(largest gap between two correct oracles `9.636736e-14`; smallest gap the planted
degree defect produces `1.749951e-01`). Nothing here re-picks it, and nothing
here narrows it after seeing a result.

THE HALT IS LITERAL. `build_corpus` scores every instance and then, if any
instance FAILED or ERRORED, raises `CorpusHalted` and returns nothing at all.
It does not return the good rows. It does not return the rows with a flag. `main`
writes no `.jsonl` when it is raised. `demonstrate_halt` plants a disagreement on
one instance of a three-instance corpus and asserts that zero rows come back --
the point being that a corpus which drops what its own oracle cannot confirm is
worse than one that refuses to exist.

Scoring every instance before halting, rather than failing fast on the first
disagreement, is deliberate: fail-fast would make the four counts unknowable for
everything after the first bad instance, and the counts are the deliverable.
Nothing is emitted either way.

WHERE THE SECOND ORACLE CANNOT REACH. `K_b` is nonsingular iff every node of
`interior + {b}` can reach `B \\ {b}`, which is exactly
`r10_corpus_spec.admissible(adj, B \\ {b})` -- so the precondition is checked with
the repo's own guard rather than a new one. It fails on `|B| = 1` (nothing left to
ground at; the absorbing route still answers, `u == g_0`) and on a component
holding only one boundary node. Those instances are recorded
`status="inapplicable"`, `dual_checked=false`, and are counted in their own
column. They are NOT passes. Weight positivity, the third precondition, is
vacuous here: `kirchhoff.laplacian` builds unit conductances only, so a zero or
negative weight cannot be expressed by an instance this corpus can hold.

COST. Dense float64, `n <= 1024`, one process, no GPU, no training. Peak is the
`n x n` Laplacian plus one `(|I|+1) x (|I|+1)` LU workspace -- under 40 MiB at
`n = 1024`. `scale.vram_gate.preflight(0, name='r10-dual-oracle', host_mib=256)`
is called by `main` and the run is skipped if it refuses. The `|B|` separate
solves are the price of the independence: fusing them through a Schur complement
collapses the ratio to `L_II^-1 A_Ib` and turns the second oracle back into the
first one, which is why it is not done.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from scale import kirchhoff
from scale import r10_corpus_spec as spec

__all__ = ["AGREEMENT_TOL", "CorpusHalted", "kirchhoff_applicable",
           "kirchhoff_extension", "absorbing_extension", "check_instance",
           "build_corpus", "make_rung", "reproduce_arm", "widen_arm", "probes",
           "demonstrate_halt", "tally", "demo"]

#: IMPORTED, not re-picked: `scale/kirchhoff.py:106`, value `1e-10`, set there
#: from the measured window quoted in that module's docstring.
AGREEMENT_TOL = kirchhoff.AGREEMENT_TOL

#: JUPITER's rung shapes and seed base, imported so arm `reproduce` is his corpus
#: and not a lookalike. `scale/r10_corpus_spec.py:509` and `:516`.
RUNGS = spec.RUNGS
SEED0 = spec.SEED0

#: Arm `widen`. Same bisection, a DECLARED target ladder in place of the band's
#: upper edge. Seed base is distinct so no draw is shared with arm `reproduce`.
WIDEN_SEED0 = 0x3A150000
WIDEN_TARGETS = (0.905, 0.915, 0.925, 0.935, 0.945)
WIDEN_SHAPES = ((128, 4.0), (256, 6.0), (512, 4.0))
WIDEN_BIG = ((1024, 6.0, 0.905), (1024, 6.0, 0.925))

#: The planted defect, and it is the repo's own: `deg + 1` in place of `deg`, the
#: miscount from counting a node among its own neighbours
#: (`kirchhoff.scratch_chain_with_off_by_one`, `scale/kirchhoff.py:225-260`).
#: `demo` asserts this row-scaling reproduces that function's output exactly.
PLANT = "deg+1: the node counted among its own neighbours"


class CorpusHalted(RuntimeError):
    """Raised instead of returning a corpus. Carries the offending instances."""

    def __init__(self, bad):
        self.bad = bad
        detail = "; ".join(
            f"{r['id']}: status={r['status']} gap={r.get('gap')}" for r in bad)
        super().__init__(
            f"corpus HALTED: {len(bad)} instance(s) the dual oracle could not "
            f"confirm [{detail}]. No corpus is emitted -- not the good rows, not "
            "the rows with the bad one flagged. Find the wrong codepath first.")


# --------------------------------------------------------------------------
# The two routes
# --------------------------------------------------------------------------
def kirchhoff_applicable(adj, boundary) -> str | None:
    """`None` if the matrix-tree cofactor route applies, else why it does not.

    `K_b` (the Laplacian with `B \\ {b}` grounded) is nonsingular exactly when
    every node of `interior + {b}` reaches `B \\ {b}`, which is the condition
    `r10_corpus_spec.admissible` already encodes. Checked with that guard rather
    than a second copy of it.
    """
    if len(boundary) < 2:
        return (f"|B| = {len(boundary)} < 2: grounding at B \\ {{b}} leaves the "
                "full singular Laplacian, so the cofactor ratio F(v,b|B\\b) / "
                "F(b|B\\b) is 0/0. The absorbing route still answers.")
    for b in boundary:
        try:
            spec.admissible(adj, [v for v in boundary if v != b])
        except ValueError as exc:
            return f"grounding at B \\ {{{b}}} is singular: {exc}"
    return None


def kirchhoff_extension(adj, boundary, g, *, diag_defect: float = 0.0):
    """ORACLE K, the matrix-tree cofactor route, for any `|B| >= 2` and any `g`.

    Returns `(u, partition_dev, solves)`. `partition_dev` is
    `max_v |sum_b omega^(b)_v - 1|`, an internal consistency check of this route
    alone -- the absorption probabilities over a full boundary must sum to one --
    reported so the route is not trusted only because the other one agrees.

    `diag_defect` adds a constant to the Laplacian diagonal; it is the planted
    `deg + 1` miscount on THIS side, and is used only by the must-fire.
    """
    nodes = sorted(adj)
    lap, index = kirchhoff.laplacian(adj, nodes)
    if diag_defect:
        lap = lap + diag_defect * np.eye(len(nodes))
    bset = set(boundary)
    interior_idx = [index[v] for v in nodes if v not in bset]
    u = np.zeros(len(interior_idx))
    partition = np.zeros(len(interior_idx))
    for j, b in enumerate(boundary):
        keep = interior_idx + [index[b]]
        rhs = np.zeros(len(keep))
        rhs[-1] = 1.0
        z = np.linalg.solve(lap[np.ix_(keep, keep)], rhs)
        omega = z[:-1] / z[-1]          # the cofactor ratio, F(v ~ b; B) / F(B)
        u += float(g[j]) * omega
        partition += omega
    return u, float(np.max(np.abs(partition - 1.0))), len(boundary)


def absorbing_extension(PII, PIB, deg, g, *, defect: bool = False):
    """ORACLE A, the spec's resolvent, via `r10_corpus_spec.resolvent_oracle`.

    `defect=True` divides by `deg + 1`. `P^bad = (D+I)^-1 W = (D+I)^-1 D P`, so
    the miscount is exactly a row-scaling of the clean blocks by `d/(d+1)` -- no
    second copy of `blocks` to drift out of step with the first. `demo` asserts
    it reproduces `kirchhoff.scratch_chain_with_off_by_one` to the last bit.
    """
    if defect:
        s = (deg / (deg + 1.0))[:, None]
        PII, PIB = PII * s, PIB * s
    return spec.resolvent_oracle(PII, PIB, g)


# --------------------------------------------------------------------------
# One instance, scored into exactly one of four categories
# --------------------------------------------------------------------------
def check_instance(inst, *, plant: str | None = None, reverse_below: int = 0):
    """One corpus row. `status` is one of ok / failed / inapplicable / errored.

    `plant` corrupts the row's OWN absorbing oracle, so a disagreement can be
    forced and the halt demonstrated. It is not how the must-fire is run.

    THE MUST-FIRE RIDES ALONG, on the clean pass, so it costs nothing extra in
    the direction that matters. `deg + 1` in the absorbing route is a row-scaling
    of blocks already formed, so `mustfire.absorbing` is measured on EVERY
    applicable instance. The opposite direction, `deg + 1` on the Laplacian
    diagonal, needs a second Kirchhoff pass and so is measured on instances with
    `n <= reverse_below`; it brings the SHARED plant with it, which is the
    control that matters most: both routes wrong the same way must be shown to
    slip through, or the agreement is being read as evidence it cannot carry.
    """
    adj, boundary, g = inst["adj"], inst["boundary"], inst["g"]
    row = {k: v for k, v in inst.items() if k not in ("adj", "boundary", "g")}
    row.update(boundary_size=len(boundary), status="ok", dual_checked=False,
               plant=plant)
    t0 = time.perf_counter()
    try:
        why = kirchhoff_applicable(adj, boundary)
        PII, PIB, interior, deg = spec.blocks(adj, boundary)
        u_a = absorbing_extension(PII, PIB, deg, g, defect=plant is not None)
        row.update(interior_size=len(interior),
                   u_absorbing_range=[float(u_a.min()), float(u_a.max())])
        if why is not None:
            row.update(status="inapplicable", reason=why)
            return row
        u_k, part_dev, solves = kirchhoff_extension(adj, boundary, g)
        gap = float(np.max(np.abs(u_a - u_k)))
        row.update(dual_checked=True, gap=gap, tol=AGREEMENT_TOL,
                   kirchhoff_solves=solves, partition_dev=part_dev,
                   agrees=bool(gap <= AGREEMENT_TOL))
        if not row["agrees"]:
            row["status"] = "failed"
        if plant is None:
            u_a_bad = absorbing_extension(PII, PIB, deg, g, defect=True)
            mf = {"absorbing": _fires(u_a_bad, u_k)}
            if inst.get("n", 10 ** 9) <= reverse_below:
                u_k_bad, _p, _s = kirchhoff_extension(adj, boundary, g,
                                                      diag_defect=1.0)
                mf["kirchhoff"] = _fires(u_a, u_k_bad)
                mf["shared"] = _fires(u_a_bad, u_k_bad)
            row["mustfire"] = mf
    except Exception as exc:                     # counted, never swallowed
        row.update(status="errored", error=f"{type(exc).__name__}: {exc}")
    row["seconds"] = round(time.perf_counter() - t0, 3)
    return row


def _fires(x, y) -> dict:
    """`(gap, rejected)` for one plant, against the SAME imported tolerance."""
    gap = float(np.max(np.abs(x - y)))
    return {"gap": gap, "rejected": bool(gap > AGREEMENT_TOL)}


def build_corpus(instances, *, plant_at=None, reverse_below: int = 0):
    """THE GENERATOR. Returns the corpus, or raises `CorpusHalted` and returns
    NOTHING -- not the passing rows, not the rows with the bad one flagged.

    `plant_at` is an index (or set of indices) whose instance is built with a
    defect, so the halt can be demonstrated rather than described.
    `reverse_below` runs the second, opposite-direction plant on instances whose
    interior is smaller than it, since it costs a whole extra Kirchhoff pass.
    """
    planted = {plant_at} if isinstance(plant_at, int) else set(plant_at or ())
    rows = [check_instance(inst, plant=PLANT if i in planted else None,
                           reverse_below=reverse_below)
            for i, inst in enumerate(instances)]
    bad = [r for r in rows if r["status"] in ("failed", "errored")]
    if bad:
        raise CorpusHalted(bad)
    return rows


def tally(rows) -> dict:
    """The four counts, each named, plus what was actually dual-checked."""
    return {"instances": len(rows),
            "passed": sum(r["status"] == "ok" for r in rows),
            "failed": sum(r["status"] == "failed" for r in rows),
            "errored": sum(r["status"] == "errored" for r in rows),
            "inapplicable": sum(r["status"] == "inapplicable" for r in rows),
            "dual_checked": sum(bool(r.get("dual_checked")) for r in rows)}


# --------------------------------------------------------------------------
# The instances
# --------------------------------------------------------------------------
def make_rung(n: int, mean_deg: float, seed: int, target: float, arm: str):
    """One stratified rung, drawn with JUPITER's generator and stratifier.

    The rng is consumed in his order -- graph, node order, then `g` -- so arm
    `reproduce` at `target = BAND_HI` reproduces his instances exactly, `g`
    included, and the `lambda_2` column can be diffed against
    `results/r10_it14_corpus_spec.jsonl` instead of eyeballed.
    """
    rng = np.random.default_rng(seed)
    adj = spec.draw_graph(n, max(int(n * mean_deg / 2.0) - (n - 1), 0), rng)
    order = [int(v) for v in rng.permutation(n)]
    k, lam, solves = spec.stratify(adj, order, lo=spec.BAND_LO, hi=target)
    if k is None:
        return {"kind": "rung", "arm": arm, "n": n, "mean_deg": mean_deg,
                "seed": f"0x{seed:08x}", "target": target, "lambda_2": lam,
                "eigensolves": solves, "status": "band_miss",
                "reason": f"bisection crossed from >{target} to <{spec.BAND_LO}; "
                          "no integer |B| lands inside the band for this target"}
    boundary = order[:k]
    g = np.array(rng.uniform(0.0, 1.0, size=len(boundary)))
    return {"kind": "rung", "arm": arm, "id": f"{arm}-n{n}-d{mean_deg:g}-t{target}",
            "n": n, "mean_deg": mean_deg, "seed": f"0x{seed:08x}",
            "target": target, "lambda_2": lam, "eigensolves": solves,
            "in_band": bool(spec.BAND_LO <= lam <= spec.BAND_HI),
            "adj": adj, "boundary": boundary, "g": g}


def reproduce_arm():
    """JUPITER's twelve rungs, his seeds, his target (the band's upper edge)."""
    return [make_rung(n, d, SEED0 + i, spec.BAND_HI, "reproduce")
            for i, (n, d) in enumerate(RUNGS)]


def reproduce_check(instances):
    """Arm `reproduce` diffed against JUPITER's shipped rows, per rung.

    This exists because "I imported his generator" and "I rebuilt something that
    looks like it" are not distinguishable from a clean agreement number, and
    the second one would quietly weaken every figure in this report. If the
    `lambda_2` column is bit-identical to `results/r10_it14_corpus_spec.jsonl`,
    the instances are his -- same graph, same node order, same boundary, same
    `g` off the same rng stream -- and the only new codepath is the second
    oracle, which is the whole point of the round.
    """
    path = Path(__file__).resolve().parents[1] / "results" / \
        "r10_it14_corpus_spec.jsonl"
    if not path.exists():
        return {"kind": "reproduce_check", "status": "unavailable",
                "reason": f"{path.name} absent; the diff cannot be run"}
    theirs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    theirs = [r for r in theirs if r.get("kind") == "rung" and "lambda_2" in r]
    mine = [i for i in instances if i.get("arm") == "reproduce"]
    diffs = [{"n": a["n"], "mean_deg": a["mean_deg"], "seed": a["seed"],
              "lambda_2": b["lambda_2"],
              "lambda_2_delta": abs(a["lambda_2"] - b["lambda_2"]),
              "boundary_size_delta": len(b["boundary"]) - a["boundary_size"]}
             for a, b in zip(theirs, mine)]
    return {"kind": "reproduce_check", "status": "ok", "rungs": len(diffs),
            "max_lambda_2_delta": max((d["lambda_2_delta"] for d in diffs),
                                      default=None),
            "boundary_sizes_identical": all(d["boundary_size_delta"] == 0
                                            for d in diffs),
            "per_rung": diffs,
            "note": "0.0 here means the generator was IMPORTED from "
                    "scale/r10_corpus_spec.py, not reconstructed"}


def widen_arm():
    """The same bisection aimed at a DECLARED ladder of targets across the band.

    The rule change is one argument -- `hi = target` instead of `hi = BAND_HI` --
    and no rule was chosen after seeing where it landed.
    """
    out, i = [], 0
    for n, d in WIDEN_SHAPES:
        for t in WIDEN_TARGETS:
            out.append(make_rung(n, d, WIDEN_SEED0 + i, t, "widen"))
            i += 1
    for n, d, t in WIDEN_BIG:
        out.append(make_rung(n, d, WIDEN_SEED0 + i, t, "widen"))
        i += 1
    return out


def probes():
    """Instances built to hit the class the Kirchhoff route cannot reach.

    The drawn corpus never hits it -- `draw_graph` is connected by construction
    and the stratifier never returned `|B| < 4` -- so a count of zero
    inapplicable instances would say nothing about whether the category is
    handled. These three make it say something.
    """
    ring = {v: {(v - 1) % 6, (v + 1) % 6} for v in range(6)}
    split = {0: {1}, 1: {0}, 2: {3}, 3: {2}}
    return [
        {"kind": "probe", "id": "probe-B1", "n": 6, "adj": ring, "boundary": [0],
         "g": np.array([0.4]),
         "note": "|B| = 1: the absorbing route answers (u == g_0), the cofactor "
                 "ratio is 0/0"},
        {"kind": "probe", "id": "probe-split", "n": 4, "adj": split,
         "boundary": [0, 2], "g": np.array([1.0, 0.0]),
         "note": "two components, one boundary node each: admissible for the "
                 "absorbing route, singular for every grounding of the other"},
        {"kind": "probe", "id": "probe-ring-B2", "n": 6, "adj": ring,
         "boundary": [0, 3], "g": np.array([1.0, 0.0]),
         "note": "the control: same ring, |B| = 2, applicable and expected to "
                 "pass, so the two refusals above are not a broken route"},
    ]


# --------------------------------------------------------------------------
# The must-fires
# --------------------------------------------------------------------------
def demonstrate_halt():
    """The halt, shown rather than described.

    A three-instance corpus is built twice: clean, and with a `deg + 1` defect on
    instance 1. The clean build returns three rows. The planted build returns
    NOTHING -- it raises, and the two good instances do not come back.
    """
    small = [make_rung(64, 4.0, SEED0 + i, spec.BAND_HI, "halt") for i in (0, 1)]
    small.append(make_rung(128, 4.0, SEED0 + 2, spec.BAND_HI, "halt"))
    clean = build_corpus(small)
    emitted, err = None, None
    try:
        emitted = build_corpus(small, plant_at=1)
    except CorpusHalted as exc:
        err = exc
    return {"kind": "halt_demo",
            "clean_rows": len(clean),
            "clean_status": sorted({r["status"] for r in clean}),
            "planted_at": 1,
            "planted_rows_emitted": 0 if emitted is None else len(emitted),
            "halted": err is not None,
            "halt_gap": err.bad[0].get("gap") if err else None,
            "halt_message": str(err)[:400] if err else None,
            "note": "the two passing instances of the planted build are NOT "
                    "returned; the corpus refuses to exist rather than shipping "
                    "the rows its own oracle confirmed"}


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
def demo() -> None:
    """Assert-based self-check. Every claim the report makes has a line here."""
    # 0. The tolerance is the imported one, not a local re-pick.
    assert AGREEMENT_TOL is kirchhoff.AGREEMENT_TOL and AGREEMENT_TOL == 1e-10

    # 1. The generalised route REDUCES to the shipped |B|=2 oracle, and to the
    #    path's closed form.
    adj, boundary, g, _lam_cf, u_cf = spec.path_case(7)
    u_k, part, solves = kirchhoff_extension(adj, boundary, g)
    omega, transient = kirchhoff.harmonic_measure(adj, sorted(adj), (0, 8))
    assert np.max(np.abs(u_k - omega)) < 1e-14, np.max(np.abs(u_k - omega))
    assert np.max(np.abs(u_k - u_cf)) < 1e-14
    assert part < 1e-12 and solves == 2

    # 2. The ratio really is a ratio of SPANNING-FOREST COUNTS, tied to literal
    #    enumeration on a graph small enough to enumerate.
    tiny = {0: {1, 2}, 1: {0, 2, 3}, 2: {0, 1, 4}, 3: {1, 4}, 4: {2, 3}}
    nodes = sorted(tiny)
    den = kirchhoff.brute_force_separating_forests(tiny, nodes, 0, 4)
    brute = np.array([kirchhoff.brute_force_separating_forests(
        tiny, nodes, 0, 4, together=v) / den for v in (1, 2, 3)])
    u_t, _p, _s = kirchhoff_extension(tiny, [0, 4], np.array([1.0, 0.0]))
    assert np.max(np.abs(u_t - brute)) < 1e-12, (u_t, brute, den)

    # 3. |B| > 2 and a non-binary g: the case the shipped oracle cannot take.
    g3 = np.array([1.0, 0.3, 0.7])
    PII3, PIB3, _int3, deg3 = spec.blocks(tiny, [0, 3, 4])
    u_a3 = absorbing_extension(PII3, PIB3, deg3, g3)
    u_k3, part3, _s = kirchhoff_extension(tiny, [0, 3, 4], g3)
    assert np.max(np.abs(u_a3 - u_k3)) <= AGREEMENT_TOL and part3 < 1e-12

    # 4. The plant is the REPO's plant: the row-scaling reproduces
    #    kirchhoff.scratch_chain_with_off_by_one bit for bit.
    PII, PIB, interior, deg = spec.blocks(adj, boundary)
    u_bad = absorbing_extension(PII, PIB, deg, g, defect=True)
    ref = kirchhoff.scratch_chain_with_off_by_one(adj, sorted(adj), (0, 8))
    assert np.max(np.abs(u_bad - ref)) == 0.0, np.max(np.abs(u_bad - ref))

    # 5. MUST-FIRE, both directions -- and the shared plant, which must NOT fire.
    inst = {"kind": "demo", "id": "path-L7", "n": 9, "adj": adj,
            "boundary": boundary, "g": g}
    mf = check_instance(inst, reverse_below=64)["mustfire"]
    assert mf["absorbing"]["rejected"] and mf["absorbing"]["gap"] > 1e-3, mf
    assert mf["kirchhoff"]["rejected"] and mf["kirchhoff"]["gap"] > 1e-3, mf
    assert not mf["shared"]["rejected"] and mf["shared"]["gap"] < 1e-14, mf

    # 6. THE HALT: a planted disagreement emits nothing at all.
    h = demonstrate_halt()
    assert h["clean_rows"] == 3 and h["clean_status"] == ["ok"], h
    assert h["halted"] and h["planted_rows_emitted"] == 0, h

    # 7. Inapplicable is inapplicable, not a pass -- and the control passes.
    rows = [check_instance(p) for p in probes()]
    assert [r["status"] for r in rows] == ["inapplicable", "inapplicable", "ok"], \
        [(r["id"], r["status"]) for r in rows]
    assert all(r["dual_checked"] is False for r in rows[:2])
    assert tally(rows) == {"instances": 3, "passed": 1, "failed": 0, "errored": 0,
                           "inapplicable": 2, "dual_checked": 1}

    print(f"demo ok: |B|=2 reduction {np.max(np.abs(u_k - omega)):.3e}, forest "
          f"tie-down {np.max(np.abs(u_t - brute)):.3e}, |B|=3 gap "
          f"{np.max(np.abs(u_a3 - u_k3)):.3e}, plants "
          f"{mf['absorbing']['gap']:.6e}/{mf['kirchhoff']['gap']:.6e}, shared "
          f"plant {mf['shared']['gap']:.3e} (NOT rejected), halt emitted "
          f"{h['planted_rows_emitted']} rows")


# --------------------------------------------------------------------------
def main(write: bool) -> None:
    from scale.vram_gate import preflight
    fits, verdict = preflight(0, name="r10-dual-oracle", host_mib=256)
    print(verdict)
    if not fits:
        print("REFUSED by the gate; that is the reportable result.")
        return
    t0 = time.perf_counter()
    drawn = reproduce_arm() + widen_arm()
    # A rung whose bisection overshot the band produced NO instance. It was
    # never reached by either oracle, so it is neither a pass nor an
    # inapplicable one; it gets its own count and its own rows.
    misses = [i for i in drawn if i.get("status") == "band_miss"]
    instances = [i for i in drawn if "adj" in i] + probes()
    repro = reproduce_check(instances)
    try:
        rows = build_corpus(instances, reverse_below=256)
    except CorpusHalted as exc:
        print(exc)
        print("NO CORPUS EMITTED. No .jsonl written. That is the deliverable.")
        return
    counts = tally(rows)
    halt = demonstrate_halt()
    checked = [r for r in rows if r.get("dual_checked")]
    worst = max(checked, key=lambda r: r["gap"])
    fired = [r["mustfire"] for r in rows if "mustfire" in r]
    summary = {"kind": "summary", **counts,
               "not_generated": len(misses),
               "rungs_attempted": len(drawn), "probes": len(probes()),
               "max_gap": worst["gap"], "max_gap_at": worst["id"],
               "max_partition_dev": max(r["partition_dev"] for r in checked),
               "tol": AGREEMENT_TOL, "tol_source": "scale/kirchhoff.py:106",
               "eigensolves": sum(i.get("eigensolves", 0) for i in drawn),
               "kirchhoff_solves": sum(r.get("kirchhoff_solves", 0) for r in rows),
               "mustfire_absorbing": {
                   "instances": len(fired),
                   "rejected": sum(f["absorbing"]["rejected"] for f in fired),
                   "gap_min": min(f["absorbing"]["gap"] for f in fired),
                   "gap_max": max(f["absorbing"]["gap"] for f in fired)},
               "mustfire_kirchhoff": {
                   "instances": sum("kirchhoff" in f for f in fired),
                   "rejected": sum(f["kirchhoff"]["rejected"]
                                   for f in fired if "kirchhoff" in f),
                   "gap_min": min((f["kirchhoff"]["gap"]
                                   for f in fired if "kirchhoff" in f), default=None),
                   "gap_max": max((f["kirchhoff"]["gap"]
                                   for f in fired if "kirchhoff" in f), default=None)},
               "mustfire_shared_blindspot": {
                   "instances": sum("shared" in f for f in fired),
                   "rejected": sum(f["shared"]["rejected"]
                                   for f in fired if "shared" in f),
                   "gap_max": max((f["shared"]["gap"]
                                   for f in fired if "shared" in f), default=None)},
               "reproduce_max_lambda_2_delta": repro.get("max_lambda_2_delta"),
               "lambda_2_range": [min(r["lambda_2"] for r in rows if "lambda_2" in r),
                                  max(r["lambda_2"] for r in rows if "lambda_2" in r)],
               "seconds": round(time.perf_counter() - t0, 1)}
    # `kind` LAST: `{"kind": ..., **m}` lets the miss's own "rung" kind win, and
    # a never-generated rung filed under `kind == "rung"` is exactly the
    # confusion between "not reached" and "passed" this round is counting.
    out_rows = rows + [{**m, "kind": "not_generated"} for m in misses] \
        + [repro, halt, summary]
    for r in out_rows:
        print(json.dumps(r, default=float)[:220])
    if write:
        out = Path(__file__).resolve().parents[1] / "results"
        with (out / "r10_it15_dual_oracle.jsonl").open("w", encoding="utf-8") as fh:
            for r in out_rows:
                fh.write(json.dumps(r, sort_keys=True, default=float) + "\n")
        print(f"wrote {out / 'r10_it15_dual_oracle.jsonl'}")


if __name__ == "__main__":
    if "--write" in sys.argv or "--run" in sys.argv:
        main(write="--write" in sys.argv)
    else:
        demo()
