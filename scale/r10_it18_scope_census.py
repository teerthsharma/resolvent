"""MARS attack #2, v-main.3M it.18. The scope-hunt census: R(control) against D.

The script line, verbatim:

    MARS: attack #2 scope-hunt -- for each control, the reachability set
    R(control) vs production domain D; fires iff R n D = None for any control
    (the 14th class, mechanized as a set census).

    python -m scale.r10_it18_scope_census            # demo(), assert-based
    python -m scale.r10_it18_scope_census --write    # the census, writes the .jsonl

HOW R AND D ARE MADE FINITE. A literal intersection over two infinite input
spaces is not computable, so both sides are replaced by named finite objects and
the replacement is the thing to attack, not the arithmetic.

  R(control) is replaced by THE INPUTS THE CONTROL IS ACTUALLY HANDED. Every
  control in the registry is deterministic: its inputs are literals in the source
  or are generated from declared seeds. `R` is therefore a finite, enumerable set
  and no proxy is involved -- this side is exact.

  D is replaced by TWO objects, and every verdict says which one decided it.

    D_shipped  -- the instances the production builder has actually emitted:
                  `dual.reproduce_arm()` + `dual.widen_arm()`, 28 instances,
                  entered through the FRONT DOOR, membership by SHA-256 on the
                  labelled key (`adm.labelled_key`). Exact, both ways.

    D_invariant -- five NECESSARY conditions on membership of D, each read off
                  the production code rather than guessed. A control input that
                  violates one is PROVEN out of D. A control input that violates
                  none is NOT proven in: the conditions are necessary, not
                  sufficient, and saying otherwise is the whole error this
                  census exists to avoid.

  A control input that is in neither -- violates no invariant, matches nothing
  shipped -- is UNDECIDED, and falls back to a SAMPLE: `SEARCH_DRAWS`
  production graph draws at the control's own `n`. A miss there is
  "not observed in 200 draws", never "empty".

THE FIRING RULE, FILED BEFORE ANY CONTROL WAS MEASURED.

    FIRES(control)  iff  no element of R(control) is IN D.
    strength = "proven"  iff every element is out by a violated invariant or by
                            exhaustive membership in a finite domain
             = "sampled" iff any element's out-ness rests only on the draw search

THE ROLE TAXONOMY, ALSO FILED FIRST, AND IT IS THE FINDING. The rule as written
flags a class of control for which out-of-domain is the entire design intent.

    certifying  -- its reading supports a claim ABOUT PRODUCTION. Firing is a
                   defect: the claim rests on inputs production cannot make.
    excluding   -- it exercises a REFUSAL. Production cannot emit what a guard
                   exists to refuse, so firing is expected and is not a defect;
                   the dual obligation is a non-firing in-domain control.
    illustrating-- it exists to demonstrate a hazard. Firing IS the result.

Role is a declared property of each control, not something this census can infer,
and it is declared in the registry below beside the site that owns the control.
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from scale import kirchhoff, vram_gate
from scale import r10_admissibility as adm
from scale import r10_corpus_spec as spec
from scale import r10_dual_oracle as dual

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "r10_it18_scope_census.jsonl"

AGREEMENT_TOL = kirchhoff.AGREEMENT_TOL
BAND_LO, BAND_HI = spec.BAND_LO, spec.BAND_HI

#: The (n, mean_deg) grid the production builder is ever CALLED at. Finite by
#: declaration in the two shipped modules, not by a choice made here.
SHAPES = set(spec.RUNGS) | {(n, d) for n, d in dual.WIDEN_SHAPES} | \
         {(n, d) for n, d, _t in dual.WIDEN_BIG}

#: Seeds for the fallback sample. Declared before the census ran.
SEARCH_SEED0 = 0x3A180000


def _shipped_draws() -> tuple[int, str]:
    """`spec`'s own "how many production draws price a property" number, read
    out of its source: `naive_yield(200, SEED0 + 0x1000)`.

    Lifted rather than copied so a drifted duplicate is impossible, and imported
    rather than re-picked so the budget is not one chosen after seeing a miss.
    """
    src = (HERE / "scale" / "r10_corpus_spec.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "naive_yield"
                and node.args and isinstance(node.args[0], ast.Constant)):
            return int(node.args[0].value), f"scale/r10_corpus_spec.py:{node.lineno}"
    raise LookupError("scale/r10_corpus_spec.py no longer calls naive_yield")


SEARCH_DRAWS, SEARCH_DRAWS_SOURCE = _shipped_draws()


# --------------------------------------------------------------------------
# D -- the production domain, both objects
# --------------------------------------------------------------------------
def d_shipped():
    """Every instance the production builder emits, through the FRONT DOOR.

    `dual.reproduce_arm()` and `dual.widen_arm()` are the two shipped arms; the
    band miss carries no instance and is returned separately so a corpus of 29
    attempts is not reported as a domain of 29.
    """
    rows = dual.reproduce_arm() + dual.widen_arm()
    live = [r for r in rows if r.get("status") != "band_miss"]
    return live, [r for r in rows if r.get("status") == "band_miss"]


def invariants(inst) -> list[dict]:
    """The five NECESSARY conditions, each cited to the line that forces it.

    Necessary, never sufficient. A control passing all five is UNDECIDED, and
    the census says so rather than promoting it to a membership.
    """
    adj, boundary, g = inst["adj"], inst["boundary"], np.asarray(inst["g"])
    n = len(adj)
    edges = sum(len(v) for v in adj.values()) // 2
    shapes_at_n = sorted(d for (m, d) in SHAPES if m == n)
    want = {(n - 1) + adm.extra_for(n, d) for d in shapes_at_n}
    start = next(iter(sorted(adj)))
    seen, stack = {start}, [start]
    while stack:                                    # connectivity of the GRAPH
        for u in adj[stack.pop()]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    try:
        spec.admissible(adj, boundary)
        adm_ok, adm_why = True, None
    except ValueError as exc:
        adm_ok, adm_why = False, str(exc)[:120]
    if adm_ok:
        PII, _PIB, _i, deg = spec.blocks(adj, boundary)
        lam = spec.spectrum(PII, deg)[0]
    else:
        lam = None
    return [
        {"invariant": "I1 shape", "holds": bool(shapes_at_n),
         "detail": f"n={n}; production is called at n in "
                   f"{sorted({m for m, _ in SHAPES})}",
         "cite": "scale/r10_corpus_spec.py:506 RUNGS; "
                 "scale/r10_dual_oracle.py:124-125 WIDEN_SHAPES/WIDEN_BIG"},
        {"invariant": "I2 edge count", "holds": bool(want and edges in want),
         "detail": f"edges={edges}; production emits {sorted(want)} at n={n}",
         "cite": "scale/r10_admissibility.py:168 extra_for; "
                 "scale/r10_corpus_spec.py:138 draw_graph"},
        {"invariant": "I3 connected", "holds": len(seen) == n,
         "detail": f"{n - len(seen)} of {n} nodes outside the component of "
                   f"node {start}",
         "cite": "scale/r10_corpus_spec.py:145-152 attachment tree, "
                 "connectivity by construction"},
        {"invariant": "I4 g in [0,1)", "holds": bool(g.size and g.min() >= 0.0
                                                     and g.max() < 1.0),
         "detail": f"g in [{float(g.min()):.6g}, {float(g.max()):.6g}]"
                   if g.size else "g is empty",
         "cite": "scale/r10_dual_oracle.py:325 rng.uniform(0.0, 1.0)"},
        {"invariant": "I5 lambda_2 in band", "holds": bool(
            adm_ok and lam is not None and BAND_LO <= lam <= BAND_HI),
         "detail": (f"lambda_2={lam:.10f}" if lam is not None
                    else f"admissible refused: {adm_why}")
                   + f"; band [{BAND_LO}, {BAND_HI}]",
         "cite": "scale/r10_corpus_spec.py:361 stratify returns k=None outside "
                 "the band; scale/r10_dual_oracle.py:317-323 emits band_miss"},
    ]


def sample_search(inst, draws: int = None, seed0: int = SEARCH_SEED0) -> dict:
    """The fallback. `draws` production GRAPH draws at the control's own `n`.

    Graph identity is settled by `spec.draw_graph` alone -- the stratifier draws
    nothing -- so no eigensolve is needed and the search is over the object that
    decides membership first. Narrowing to the control's own `n` makes the
    sample CHEAPER and therefore WEAKER, and that is said here rather than in a
    footnote: a miss is evidence about this `n` only.
    """
    draws = SEARCH_DRAWS if draws is None else draws
    n = len(inst["adj"])
    key = adm.graph_key(inst["adj"])
    degs = sorted(d for (m, d) in SHAPES if m == n)
    tried = 0
    for d in degs or [4.0]:
        for i in range(draws):
            rng = np.random.default_rng(seed0 + i)
            g = spec.draw_graph(n, adm.extra_for(n, d), rng)
            tried += 1
            if adm.graph_key(g) == key:
                return {"found": True, "draws_tried": tried, "at_mean_deg": d}
    return {"found": False, "draws_tried": tried, "mean_degs": degs,
            "seed0": f"0x{seed0:08x}"}


def in_d_instance(inst, shipped_keys: dict) -> dict:
    """Membership of one instance in D. Exact when it can be, sampled when not."""
    key = adm.labelled_key(inst["adj"], inst["boundary"], np.asarray(inst["g"]))
    if key in shipped_keys:
        return {"member": True, "strength": "proven",
                "how": f"labelled key matches shipped instance "
                       f"{shipped_keys[key]}"}
    inv = invariants(inst)
    broken = [i["invariant"] for i in inv if not i["holds"]]
    if broken:
        return {"member": False, "strength": "proven",
                "how": "violates " + ", ".join(broken), "invariants": inv}
    s = sample_search(inst)
    return {"member": bool(s["found"]),
            "strength": "proven" if s["found"] else "sampled",
            "how": ("graph reproduced by a production draw"
                    if s["found"] else
                    f"no invariant violated and no match in {s['draws_tried']} "
                    f"production draws at n={len(inst['adj'])}"),
            "search": s, "invariants": inv}


# --------------------------------------------------------------------------
# The three other domains, each decided by its own predicate
# --------------------------------------------------------------------------
def in_d_label(inst, y) -> dict:
    """Is `y` a label production can emit on this instance?

    `u` is the harmonic extension, so `y` is producible iff `(I - P_II) y` lies
    in the column space of `P_IB` at some `g` in `[0,1)`. Least squares gives the
    only candidate `g`; the residual decides. A residual above the imported
    tolerance is a PROOF that no `g` produces `y`.
    """
    PII, PIB, _i, _d = spec.blocks(inst["adj"], inst["boundary"])
    rhs = (np.eye(PII.shape[0]) - PII) @ y
    ghat = np.linalg.lstsq(PIB, rhs, rcond=None)[0]
    resid = float(np.max(np.abs(PIB @ ghat - rhs)))
    ok_range = bool(ghat.min() >= 0.0 and ghat.max() < 1.0)
    return {"member": bool(resid <= AGREEMENT_TOL and ok_range),
            "strength": "proven", "residual": resid,
            "implied_g_range": [float(ghat.min()), float(ghat.max())],
            "how": ("harmonic with an implied g in [0,1)" if resid <= AGREEMENT_TOL
                    and ok_range else
                    f"no g reproduces y: residual {resid:.3e} > {AGREEMENT_TOL}"
                    if resid > AGREEMENT_TOL else
                    f"implied g leaves [0,1): "
                    f"[{float(ghat.min()):.4g}, {float(ghat.max()):.4g}]")}


def production_preflights() -> list[dict]:
    """Every `preflight(...)` in `scale/`, with each argument classified.

    Taken from the AST, so a call wrapped across lines is not missed -- and both
    it.16 resource plants are wrapped, so a line-oriented scan would have found
    zero and reported an empty domain as a clean one.

    D_resource is the set of requests whose arguments are LITERAL CONSTANTS: a
    shipped module prices a job at a declared number. A request computed from a
    live reading is therefore provably not one of them, and that is a property of
    the call and not a judgement about which call sites count as production.
    """
    out = []
    for p in sorted((HERE / "scale").glob("*.py")):
        src = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", getattr(node.func, "id", None))
                    == "preflight"):
                continue
            for arg in list(node.args) + [k for k in node.keywords
                                          if k.arg in ("job_mib", "host_mib")]:
                val = arg.value if isinstance(arg, ast.keyword) else arg
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    continue                        # name=, not a resource
                out.append({"site": f"scale/{p.name}:{node.lineno}",
                            "expr": ast.get_source_segment(src, val),
                            "literal": isinstance(val, ast.Constant)})
    return out


def in_d_resource(request: str, shipped) -> dict:
    """A resource request is in D iff it is a literal request production issues."""
    lits = {r["expr"] for r in shipped if r["literal"]}
    return {"member": request in lits, "strength": "proven",
            "how": (f"{request} is a shipped literal production request"
                    if request in lits else
                    f"{request} is not a literal; production issues only "
                    f"{sorted(lits)} across {len(shipped)} preflight arguments")}


def in_d_file(path: Path) -> dict:
    """A file is in D iff it lives under the repository this campaign writes."""
    inside = str(path).startswith(str(HERE))
    return {"member": inside, "strength": "proven",
            "how": ("under the repository root" if inside
                    else "outside the repository root; nothing production "
                         "writes can land there")}


# --------------------------------------------------------------------------
# THE REGISTRY -- every control, its site, its declared role, its R
# --------------------------------------------------------------------------
def _two_component_n64():
    """A CONSTRUCTED plant: n = 64, the production edge count for mean degree 4,
    and two components.

    It was written to be refused by I3 alone. MEASURED, it is refused by I3 AND
    I5 -- splitting the graph moves `lambda_2` out of the band as well -- so it
    is a two-invariant plant and is reported as one. The single-invariant plant
    on this census is `plant-g-out-of-range`, which trips I4 and nothing else.
    """
    rng = np.random.default_rng(0x3A180001)
    adj = {}
    for half in (0, 32):
        part = spec.draw_graph(32, 0, rng)          # a 31-edge tree
        for v, nb in part.items():
            adj[v + half] = {u + half for u in nb}
    want = (64 - 1) + adm.extra_for(64, 4.0)        # what production emits
    while sum(len(v) for v in adj.values()) // 2 < want:
        half = 0 if rng.integers(0, 2) == 0 else 32
        a, b = int(rng.integers(0, 32)) + half, int(rng.integers(0, 32)) + half
        if a != b and b not in adj[a]:
            adj[a].add(b)
            adj[b].add(a)
    return {"id": "plant-two-components", "adj": adj, "boundary": [0, 32],
            "g": np.array([0.25, 0.75])}


def _recovered(inst):
    """The below-door entry's output for a shipped rung: `adm.recover` replaying
    the rng stream with the boundary SIZE taken from the row."""
    adj, boundary, g = adm.recover(inst["n"], inst["mean_deg"],
                                   int(inst["seed"], 16), len(inst["boundary"]))
    return {"id": inst["id"] + "-recovered", "adj": adj, "boundary": boundary,
            "g": g}


def _relabelled(inst):
    """A CONSTRUCTED plant that violates NOTHING: a real front-door instance with
    two node labels transposed, boundary carried along. Same n, same edge count,
    same connectivity, same spectrum, same `g` -- and a different graph key.

    It exists so the SAMPLED branch of the census is exercised on an input built
    for the purpose, rather than only on whatever the live tree happens to hold.
    """
    a, b = inst["boundary"][0], next(v for v in sorted(inst["adj"])
                                     if v not in inst["boundary"])
    sw = {a: b, b: a}
    adj = {sw.get(v, v): {sw.get(u, u) for u in nb}
           for v, nb in inst["adj"].items()}
    return {"id": "plant-relabelled", "adj": adj,
            "boundary": [sw.get(v, v) for v in inst["boundary"]],
            "g": np.asarray(inst["g"]).copy()}


def registry(shipped):
    """Every control this round runs, with the site that owns it.

    Enumerated from the four round-10 corpus modules by reading their `demo()`
    and must-fire bodies; the site is printed with every row so the list can be
    audited against the source rather than trusted.
    """
    real = shipped[0]
    real64 = next(r for r in shipped if r["n"] == 64)
    ring6 = {v: {(v - 1) % 6, (v + 1) % 6} for v in range(6)}
    ring32 = {v: {(v - 1) % 32, (v + 1) % 32} for v in range(32)}
    pa, pb, pg, _lam, _u = spec.path_case(7)
    star = {0: set(range(1, 64))}
    star.update({v: {0} for v in range(1, 64)})
    probes = {p["id"]: p for p in dual.probes()}
    u_real = spec.resolvent_oracle(*spec.blocks(real["adj"], real["boundary"])[:2],
                                   real["g"])
    cols, _int = adm.features(real["adj"], real["boundary"], real["g"])

    def C(cid, site, role, domain, **kw):
        return {"control": cid, "site": site, "role": role, "domain": domain, **kw}

    return [
        # ---- it.14, scale/r10_corpus_spec.py
        C("it14/must_fire_oracle", "scale/r10_corpus_spec.py:432", "excluding",
          "instance", inputs=[real]),
        C("it14/must_fire_dose", "scale/r10_corpus_spec.py:441", "excluding",
          "instance", inputs=[real]),
        C("it14/admissible-disconnected", "scale/r10_corpus_spec.py:711",
          "excluding", "instance",
          inputs=[{"id": "bad-split", "adj": {0: {1}, 1: {0}, 2: {3}, 3: {2}},
                   "boundary": [0], "g": np.array([0.5])}]),
        C("it14/admissible-isolated", "scale/r10_corpus_spec.py:717", "excluding",
          "instance",
          inputs=[{"id": "bad-isolated", "adj": {0: {1}, 1: {0}, 2: set()},
                   "boundary": [0], "g": np.array([0.5])}]),
        C("it14/path_case", "scale/r10_corpus_spec.py:478", "certifying",
          "instance",
          inputs=[{"id": "path-L7", "adj": pa, "boundary": pb, "g": pg}]),
        # ---- it.15, scale/r10_dual_oracle.py
        C("it15/probe-B1", "scale/r10_dual_oracle.py:390", "excluding",
          "instance", inputs=[probes["probe-B1"]]),
        C("it15/probe-split", "scale/r10_dual_oracle.py:390", "excluding",
          "instance", inputs=[probes["probe-split"]]),
        C("it15/probe-ring-B2", "scale/r10_dual_oracle.py:390", "certifying",
          "instance", inputs=[probes["probe-ring-B2"]]),
        C("it15/plant-absorbing", "scale/r10_dual_oracle.py:217", "excluding",
          "instance", inputs=[real]),
        C("it15/plant-kirchhoff", "scale/r10_dual_oracle.py:217", "excluding",
          "instance", inputs=[real]),
        C("it15/plant-shared", "scale/r10_dual_oracle.py:217", "excluding",
          "instance", inputs=[real]),
        C("it15/demonstrate_halt", "scale/r10_dual_oracle.py:419", "excluding",
          "instance", inputs=shipped[:3]),
        # ---- it.16, scale/r10_admissibility.py
        C("it16/plant-leak", "scale/r10_admissibility.py:661", "excluding",
          "instance", inputs=[real]),
        C("it16/plant-ring32", "scale/r10_admissibility.py:665", "excluding",
          "instance", inputs=[{"id": "plant-ring", "adj": ring32,
                               "boundary": [0, 1], "g": np.array([0.9, 0.0])}]),
        C("it16/control-clean-SPLIT-ARM", "scale/r10_admissibility.py:670",
          "certifying", "instance", inputs=shipped),
        C("it16/probe-F0..F3", "scale/r10_admissibility.py:675", "certifying",
          "instance", inputs=[real]),
        C("it16/plant-linear-label", "scale/r10_admissibility.py:681",
          "excluding", "label", inputs=[real],
          labels=[3.0 * cols["deg"] - 1.5 * cols["ball1"] + 7.0]),
        C("it16/control-noise-label", "scale/r10_admissibility.py:690",
          "excluding", "label", inputs=[real],
          labels=[np.random.default_rng(adm.PROBE_SEED + 1)
                  .normal(size=len(cols["deg"]))]),
        C("it16/plant-host", "scale/r10_admissibility.py:711", "excluding",
          "resource", request="host.available_mib * 4"),
        C("it16/plant-vram", "scale/r10_admissibility.py:715", "excluding",
          "resource", request="card.total_mib * 10"),
        # ---- it.17, scale/r10_it17_battery.py
        C("it17/control_front_door", "scale/r10_it17_battery.py:191",
          "certifying", "instance", inputs=[real]),
        C("it17/control_below_door", "scale/r10_it17_battery.py:203",
          "certifying", "instance", inputs=[_recovered(real)]),
        C("it17/control_hand_built-star", "scale/r10_it17_battery.py:215",
          "illustrating", "instance",
          inputs=[{"id": "star-64", "adj": star, "boundary": [0],
                   "g": np.array([0.5])}]),
        C("it17/plant-dead-bit", "scale/r10_it17_battery.py:", "excluding",
          "instance",
          inputs=[{"id": "plant-dead-bit",
                   "adj": {0: {1, 5}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4},
                           4: {3}, 5: {0}},
                   "boundary": [0, 4, 5], "g": np.array([0.9, 0.0, 0.9])}]),
        C("it17/plant-shuffled-label", "scale/r10_it17_battery.py:", "excluding",
          "label", inputs=[real],
          labels=[np.random.default_rng(0x3A170001).permutation(u_real)]),
        C("it17/planted-key-file", "scale/r10_it17_battery.py:demo step 5",
          "excluding", "file",
          path=Path(tempfile.gettempdir()) / "planted_key.jsonl"),
        C("it17/clean-control-file", "scale/r10_it17_battery.py:demo step 5",
          "excluding", "file",
          path=Path(tempfile.gettempdir()) / "clean.jsonl"),
        C("it17/C-E-ii-live-scan", "scale/r10_it17_battery.py:scan_population",
          "certifying", "file",
          path=HERE / "results" / "r10_it15_dual_oracle.jsonl"),
        # ---- it.18, this module. Constructed for the census's own must-fires.
        C("it18/plant-two-components", "scale/r10_it18_scope_census.py",
          "illustrating", "instance", inputs=[_two_component_n64()]),
        C("it18/plant-g-out-of-range", "scale/r10_it18_scope_census.py",
          "illustrating", "instance",
          inputs=[dict(real64, id="plant-g-7.5",
                       g=np.asarray(real64["g"]).copy() * 0 + 7.5)]),
        C("it18/plant-relabelled", "scale/r10_it18_scope_census.py",
          "illustrating", "instance", inputs=[_relabelled(real64)]),
    ]


# --------------------------------------------------------------------------
# The census
# --------------------------------------------------------------------------
def census_one(ctl, shipped_keys, preflights) -> dict:
    """One control. Every element of R is decided; the control fires iff none of
    them is IN D."""
    per = []
    if ctl["domain"] == "instance":
        per = [in_d_instance(i, shipped_keys) for i in ctl["inputs"]]
    elif ctl["domain"] == "label":
        per = [in_d_label(ctl["inputs"][0], y) for y in ctl["labels"]]
    elif ctl["domain"] == "resource":
        per = [in_d_resource(ctl["request"], preflights)]
    elif ctl["domain"] == "file":
        per = [in_d_file(ctl["path"])]
    members = [p["member"] for p in per]
    fires = not any(members)
    strength = ("proven" if all(p["strength"] == "proven" for p in per)
                else "sampled")
    return {"kind": "census", "control": ctl["control"], "site": ctl["site"],
            "role": ctl["role"], "domain": ctl["domain"],
            "R_size": len(per), "in_D": sum(members),
            "fires": fires, "strength": strength if fires else "n/a",
            "detail": [{"member": p["member"], "strength": p["strength"],
                        "how": p["how"]} for p in per[:4]],
            "status": "failed" if (fires and ctl["role"] == "certifying")
                      else "ok"}


BUCKET = {"ok": "passed", "failed": "failed", "error": "errored",
          "inapplicable": "inapplicable", "not_reached": "not_reached"}


def tally(rows) -> dict:
    out = dict.fromkeys(list(BUCKET.values()) + ["unscored"], 0)
    for r in rows:
        out[BUCKET.get(r.get("status"), "unscored")] += 1
    return out


# --------------------------------------------------------------------------
def demo() -> None:
    fits, why = vram_gate.preflight(0, name="r10-it18-scope-census", host_mib=512)
    print(why, flush=True)
    if not fits:
        print("SKIPPED: the host gate refused. A refusal is the result.")
        return

    assert AGREEMENT_TOL is kirchhoff.AGREEMENT_TOL
    assert SEARCH_DRAWS == 200, SEARCH_DRAWS
    print(f"  SEARCH_DRAWS = {SEARCH_DRAWS} from {SEARCH_DRAWS_SOURCE}; "
          f"seeds 0x{SEARCH_SEED0:08x}+i")
    print(f"  D shape grid: {sorted(SHAPES)}")

    # D_shipped through the front door -- only the twelve, for the self-check.
    shipped = dual.reproduce_arm()
    keys = {adm.labelled_key(r["adj"], r["boundary"], r["g"]): r["id"]
            for r in shipped}
    print(f"  D_shipped (demo subset): {len(keys)} front-door instances")

    # 1. MUST-NOT-FIRE. A front-door instance is PROVEN in D.
    good = in_d_instance(shipped[0], keys)
    assert good["member"] and good["strength"] == "proven", good
    print(f"  must-not-fire: front-door instance -> in D, {good['strength']}")

    # 2. MUST-FIRE, three CONSTRUCTED plants, three different invariants.
    for name, inst, want in (
            ("two components", _two_component_n64(), "I3 connected"),
            ("g out of range", dict(shipped[0], g=np.asarray(shipped[0]["g"]) * 0 + 7.5),
             "I4 g in [0,1)"),
            ("6-node ring", {"adj": {v: {(v - 1) % 6, (v + 1) % 6}
                                     for v in range(6)},
                             "boundary": [0, 3], "g": np.array([0.9, 0.0])},
             "I1 shape")):
        v = in_d_instance(inst, keys)
        assert not v["member"] and v["strength"] == "proven", (name, v)
        assert want.split()[0] in v["how"], (name, want, v["how"])
        print(f"  must-fire: {name:15s} -> out of D, proven, {v['how']}")

    # 3. THE SAMPLED BRANCH, on a CONSTRUCTED input that violates nothing.
    rel = in_d_instance(_relabelled(shipped[0]), keys)
    assert not rel["member"] and rel["strength"] == "sampled", rel
    assert rel["search"]["draws_tried"] >= SEARCH_DRAWS, rel["search"]
    print(f"  sampled branch: relabelled instance -> out of D, SAMPLED only "
          f"({rel['search']['draws_tried']} draws)")

    # 4. The label domain, both directions, proven.
    PII, PIB, _i, _d = spec.blocks(shipped[0]["adj"], shipped[0]["boundary"])
    u = spec.resolvent_oracle(PII, PIB, shipped[0]["g"])
    assert in_d_label(shipped[0], u)["member"], "the real label is not in D_label"
    bad = in_d_label(shipped[0], np.random.default_rng(0x3A170001).permutation(u))
    assert not bad["member"] and bad["strength"] == "proven", bad
    print(f"  label domain: real label in D; shuffled label out, "
          f"residual {bad['residual']:.3e} > {AGREEMENT_TOL}")

    # 5. Role taxonomy: firing is a FAILURE only for a certifying control.
    pf = production_preflights()
    assert pf, "no production preflight call site found"
    exc = census_one({"control": "x", "site": "-", "role": "excluding",
                      "domain": "resource", "request": "host.available_mib * 4"},
                     keys, pf)
    cer = census_one({"control": "y", "site": "-", "role": "certifying",
                      "domain": "resource", "request": "host.available_mib * 4"},
                     keys, pf)
    assert exc["fires"] and exc["status"] == "ok", exc
    assert cer["fires"] and cer["status"] == "failed", cer
    print(f"  role rule: same firing control scores ok as excluding and failed "
          f"as certifying ({len(pf)} production preflight sites scanned)")
    print("demo OK")


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--write" not in args:
        demo()
        return 0

    t0 = time.time()
    fits, why = vram_gate.preflight(0, name="r10-it18-scope-census", host_mib=512)
    print(why, flush=True)
    if not fits:
        OUT.write_text(json.dumps({"kind": "refusal", "gate": why}) + "\n",
                       encoding="utf-8")
        return 0

    shipped, misses = d_shipped()
    print(f"  D_shipped: {len(shipped)} instances, {len(misses)} band miss, "
          f"{time.time() - t0:.1f}s", flush=True)
    keys = {adm.labelled_key(r["adj"], r["boundary"], r["g"]): r["id"]
            for r in shipped}
    pf = production_preflights()

    rows = [{"kind": "provenance", "search_draws": SEARCH_DRAWS,
             "search_draws_source": SEARCH_DRAWS_SOURCE,
             "search_seed0": f"0x{SEARCH_SEED0:08x}",
             "agreement_tol": AGREEMENT_TOL, "band": [BAND_LO, BAND_HI],
             "shapes": sorted(SHAPES), "d_shipped": len(shipped),
             "d_shipped_keys": len(keys), "band_misses": len(misses),
             "production_preflight_sites": len(pf), "gate": why,
             "firing_rule": "FIRES iff no element of R(control) is IN D; "
                            "strength proven iff every element is out by a "
                            "violated invariant or by exhaustive membership",
             "roles": ["certifying (firing is a defect)",
                       "excluding (firing is expected)",
                       "illustrating (firing is the result)"]}]
    rows.extend({"kind": "band_miss", "id": m.get("seed"), "status": "not_reached",
                 "reason": m.get("reason")} for m in misses)

    reg = registry(shipped)
    for ctl in reg:
        try:
            rows.append(census_one(ctl, keys, pf))
        except Exception as exc:                      # counted, never swallowed
            rows.append({"kind": "census", "control": ctl["control"],
                         "site": ctl["site"], "role": ctl["role"],
                         "domain": ctl["domain"], "status": "error",
                         "reason": f"{type(exc).__name__}: {exc}"})

    cen = [r for r in rows if r["kind"] == "census"]
    fired = [r for r in cen if r.get("fires")]
    rows.append({"kind": "summary", "seconds": round(time.time() - t0, 1),
                 "controls_examined": len(cen),
                 "controls_firing": len(fired),
                 "firing_proven": sum(1 for r in fired
                                      if r["strength"] == "proven"),
                 "firing_sampled": sum(1 for r in fired
                                       if r["strength"] == "sampled"),
                 "firing_by_role": {role: sum(1 for r in fired
                                              if r["role"] == role)
                                    for role in ("certifying", "excluding",
                                                 "illustrating")},
                 "by_role": {role: sum(1 for r in cen if r["role"] == role)
                             for role in ("certifying", "excluding",
                                          "illustrating")},
                 "by_domain": {d: sum(1 for r in cen if r["domain"] == d)
                               for d in ("instance", "label", "resource",
                                         "file")},
                 "counts": tally(rows)})

    OUT.write_text("".join(json.dumps(r, sort_keys=True, default=float) + "\n"
                           for r in rows), encoding="utf-8")
    print("wrote", OUT, len(rows), "rows", f"{time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
