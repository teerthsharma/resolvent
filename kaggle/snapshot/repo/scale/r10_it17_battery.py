"""MARS, v-main.3M it.17. C-D / C-E / C-F on the it.14 + it.15 harmonic corpus.

The script line, verbatim:

    SATURN: C-D = exists arm with NRMSE < 1.0 before any failure is scored;
    C-E oracle executable, zero answer keys in files; C-F do()-bit flip moves
    labels with effect >= pre-set delta, control inputs drawn from the
    PRODUCTION batch path (L-SCOPE).

    python -m scale.r10_it17_battery            # demo(), assert-based
    python -m scale.r10_it17_battery --write    # the battery, writes the .jsonl

NOTHING HERE RE-PICKS A CONSTANT. Every threshold is EXTRACTED FROM THE SOURCE
THAT ALREADY SHIPS IT, by parsing that file rather than by copying the number:

  * `DELTA = 0.5`   -- `scale/negation_scope.py:1587`, `bar_verdict`'s shipped
                       one-sided do()-bit clause `flipper_dependence > 0.5`.
  * `MEAN_BAR = 1.0`-- the definition of `rips_gate.nrmse`: the mean predictor
                       is exactly 1.0 (`scale/rips_gate.py:118-127`). C-D's own
                       bar, not a choice.
  * `FAIL_BAR`, `PASS_BAR` -- `rips_gate`, by object identity, as it.16.
  * `AGREEMENT_TOL` -- `kirchhoff.AGREEMENT_TOL`, as it.14 and it.15.
  * `BUILDER`       -- `scale/p1prime.py`'s own production-builder vocabulary,
                       lifted out of its source with `ast` so a drifting copy is
                       impossible. The front-door classifier here is checked
                       against p1prime's SHIPPED list on all 191 tests at its
                       pin before it is used on anything of ours.

WHAT A do()-BIT IS ON THIS CORPUS, DECLARED BEFORE IT IS MEASURED. `g` is drawn
`uniform(0,1)` per boundary node (`scale/r10_dual_oracle.py:325`), so the corpus
ships no bit. The do()-bit is therefore DEFINED here as the two-point
intervention on one boundary coordinate:

    do(g_b := 1)  against  do(g_b := 0),      Delta u = Omega[:, b]

`Omega = (I - P_II)^-1 P_IB` is the harmonic-measure matrix, so `Delta u` is
exactly the absorption probability at `b` and is the quantity
`kirchhoff.harmonic_measure` computes by a second route. This is the LARGEST
single-coordinate intervention available -- the clause is given its best shot.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from scale import kirchhoff, p1prime, rips_gate, vram_gate
from scale import r10_admissibility as adm
from scale import r10_corpus_spec as spec
from scale import r10_dual_oracle as dual

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results" / "r10_it17_battery.jsonl"
EVENTS = HERE / "house-events.jsonl"

#: Imported by object identity, exactly as it.16 does.
FAIL_BAR = rips_gate.FAIL_BAR
PASS_BAR = rips_gate.PASS_BAR
AGREEMENT_TOL = kirchhoff.AGREEMENT_TOL

#: C-D's bar is the definition of the statistic, not a pick. `rips_gate.nrmse`
#: normalises by `sd(y)`, so the mean predictor reads exactly 1.0.
MEAN_BAR = 1.0

#: The k-hop ladder, `scale/rips_gate.py:37`'s shipped `LADDER_KS` entire. `k=0`
#: is dropped because `u_0 = 0` is not an arm, it is the zero predictor. NOTHING
#: ELSE IS TRUNCATED: an earlier cut at `k <= 32` was removed once it turned out
#: to sit on the wrong side of the crossing, which is exactly the shape of
#: choosing a ladder that flatters the verdict.
KHOP_KS = tuple(k for k in rips_gate.LADDER_KS if k >= 1)

#: Declared BEFORE the C-D plant is run. One shuffle seed, one instance.
CD_PLANT_SEED = 0x3A170001

PROBE_SEED = adm.PROBE_SEED


# --------------------------------------------------------------------------
# Constants lifted out of the source that ships them
# --------------------------------------------------------------------------
def _shipped_delta() -> tuple[float, str]:
    """`bar_verdict`'s shipped one-sided do()-bit threshold, read out of the file.

    `scale/negation_scope.py` is read as TEXT and never imported: importing it
    drags `scale.impact` and torch into a box that is holding a training job.
    The clause is `if not (cal["flipper_dependence"] > 0.5)`.
    """
    src = (HERE / "scale" / "negation_scope.py").read_text(encoding="utf-8")
    for i, line in enumerate(src.splitlines(), 1):
        m = re.search(r'cal\["flipper_dependence"\]\s*>\s*([0-9.]+)', line)
        if m:
            return float(m.group(1)), f"scale/negation_scope.py:{i} (bar_verdict)"
    raise LookupError("bar_verdict's one-sided flipper clause is gone")


DELTA, DELTA_SOURCE = _shipped_delta()


def _p1prime_builder():
    """`scale/p1prime.py`'s BUILDER pattern, taken from its AST.

    It is a local inside `main()` there, so it cannot be imported. Lifting the
    literal out of the parse tree is the difference between USING p1prime's
    vocabulary and shipping a second copy of it that drifts.
    """
    src = (HERE / "scale" / "p1prime.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "BUILDER"):
            return re.compile(ast.literal_eval(node.value.args[0])), node.lineno
    raise LookupError("scale/p1prime.py no longer defines BUILDER")


BUILDER, BUILDER_LINE = _p1prime_builder()


# --------------------------------------------------------------------------
# C-F(b) -- L-SCOPE. p1prime's method, verified against p1prime's own answers.
# --------------------------------------------------------------------------
def aliases_of(src: str) -> set[str]:
    """Local names bound to something in the `scale` package.

    Mirrors `scale/p1prime.py:104-115`. Verified against p1prime's shipped
    classification on all 191 tests at its pin by `front_door_binding()`.
    """
    out: set[str] = set()
    for m in re.finditer(r"^\s*from\s+scale(?:\.(\w+))?\s+import\s+([^\n#]+)", src, re.M):
        for piece in m.group(2).replace("(", "").replace(")", "").split(","):
            piece = piece.strip()
            if piece:
                out.add(piece.split(" as ")[-1].strip() if " as " in piece else piece)
    for m in re.finditer(r"^\s*import\s+scale\.(\w+)(?:\s+as\s+(\w+))?", src, re.M):
        out.add(m.group(2) or m.group(1))
    return out


def front_door_calls(src: str, aliases: set[str] | None = None) -> list[tuple]:
    """Lines where a BUILDER token is reached through a `scale` alias.

    Mirrors `scale/p1prime.py:118-129`. WHERE the input enters, not who built it.
    """
    aliases = aliases_of(src) if aliases is None else aliases
    hits = []
    for i, line in enumerate(src.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for a in aliases:
            if re.search(r"\b%s\s*[.\[(]" % re.escape(a), line) and BUILDER.search(line):
                hits.append((i, a, line.strip()[:90]))
                break
        else:
            if aliases and BUILDER.search(line) and any(
                    re.search(r"\b%s\b" % re.escape(a), line) for a in aliases):
                hits.append((i, "direct", line.strip()[:90]))
    return hits


def front_door_binding() -> dict:
    """The classifier above, run over every test at p1prime's PIN and diffed
    against p1prime's SHIPPED front-door list.

    Without this the classifier is a lookalike of p1prime's and the L-SCOPE
    verdict rests on a re-implementation nobody checked.
    """
    tests = p1prime.tree_files(p1prime.PIN)
    src = p1prime.blobs(p1prime.PIN, tests)
    mine = {t for t in tests if front_door_calls(src[t])}
    shipped = set((HERE / "results" / "p1prime_front_door.txt")
                  .read_text(encoding="utf-8").split())
    return {"kind": "C-F-scope-binding", "pin": p1prime.PIN,
            "tests_at_pin": len(tests), "shipped_front_door": len(shipped),
            "mine_front_door": len(mine),
            "disagreements": sorted(mine ^ shipped),
            "status": "ok" if mine == shipped else "failed"}


def head_census() -> dict:
    """p1prime's own census at HEAD, by importing its functions -- not by
    re-running its `main`, which would print and could write."""
    tests = p1prime.tree_files("HEAD")
    src = p1prime.blobs("HEAD", tests)
    callers = [t for t in tests if front_door_calls(src[t])]
    return {"kind": "C-F-scope-census", "rev": "HEAD", "files": len(tests),
            "front_door": len(callers), "no_front_door": len(tests) - len(callers),
            "pin": p1prime.PIN, "pin_shipped": [191, 46, 145],
            "note": "the shipped triple is AUDIT.md's, at the pin; HEAD is drift"}


# --------------------------------------------------------------------------
# The two control entries, side by side. Their SOURCE is the L-SCOPE evidence.
# --------------------------------------------------------------------------
def control_front_door(n: int, mean_deg: float, seed: int):
    """FRONT DOOR. The instance is manufactured by the production builder.

    `dual.make_rung` is the only callable that emits a corpus instance: it draws
    the graph, draws the node order, runs `spec.stratify` -- which runs
    `spec._lambda_at`, which runs `spec.admissible` and `spec.spectrum` -- and
    only then slices `B` and draws `g`.
    """
    return dual.make_rung(n, mean_deg, seed, spec.BAND_HI, "reproduce")


def control_below_door(n: int, mean_deg: float, seed: int, boundary_size: int):
    """BELOW THE DOOR, and it is the round's own path. `adm.recover` reproduces
    the same four objects by replaying the rng stream with the boundary SIZE
    supplied from a shipped row, which is exactly the output of the stage it
    skips. Stages left unexercised: `spec.stratify`, `spec._lambda_at`,
    `spec.spectrum`, `spec.admissible`, and the band check.
    """
    adj, boundary, g = adm.recover(n, mean_deg, seed, boundary_size)
    return {"adj": adj, "boundary": boundary, "g": g}


def control_hand_built():
    """BELOW THE DOOR AND HAND-BUILT. A star: centre in `B`, every leaf interior
    at degree 1. No production builder can emit it -- `lambda_2 = 0`, nowhere
    near the band -- and every leaf absorbs at the centre in one step, so the
    do()-bit effect is the largest the corpus's units allow.
    """
    n = 64
    adj = {0: set(range(1, n))}
    adj.update({v: {0} for v in range(1, n)})
    return {"adj": adj, "boundary": [0], "g": np.array([0.5])}


# --------------------------------------------------------------------------
# The corpus, entered through the front door
# --------------------------------------------------------------------------
def front_door_corpus():
    """`dual.reproduce_arm()` -- 12 rungs, JUPITER's seeds, 121 eigensolves."""
    return [r for r in dual.reproduce_arm() if r.get("status") != "band_miss"]


def label(inst):
    """`(u, interior, deg, PII, PIB)` -- the oracle, recomputed from the input."""
    PII, PIB, interior, deg = spec.blocks(inst["adj"], inst["boundary"])
    return spec.resolvent_oracle(PII, PIB, inst["g"]), interior, deg, PII, PIB


# --------------------------------------------------------------------------
# C-D -- AT LEAST ONE ARM READS NRMSE < 1.0
# --------------------------------------------------------------------------
def khop(PII, PIB, g, k: int) -> np.ndarray:
    """`u_k = sum_{m<k} P_II^m P_IB g` -- k rounds of the message passing the
    architecture is supposed to do, untrained, reading only `(adj, B, g)`."""
    term = PIB @ g
    out = term.copy()
    for _ in range(k - 1):
        term = PII @ term
        out = out + term
    return out


def arms(inst, y=None) -> dict:
    """Every arm's held-out-or-exact NRMSE on one instance.

    `mean` is the control and must read exactly 1.0; an arm set in which the
    control does not read 1.0 is measuring something other than NRMSE.
    """
    u, interior, _deg, PII, PIB = label(inst)
    y = u if y is None else y
    cols, _int = adm.features(inst["adj"], inst["boundary"], inst["g"])
    out = {"mean": rips_gate.nrmse(np.full_like(y, y.mean()), y),
           "probe_F3": adm.probe(cols, y, list(adm.FEATURE_SETS["F3_boundary_data"]),
                                 seed=PROBE_SEED)["nrmse_heldout"]}
    for k in KHOP_KS:
        out[f"khop_{k}"] = rips_gate.nrmse(khop(PII, PIB, inst["g"], k), y)
    return out


def c_d(inst, y=None, *, tag: str = "live") -> dict:
    """C-D on one instance. Verdict: does ANY arm other than the control read
    NRMSE < 1.0."""
    a = arms(inst, y)
    live = {k: v for k, v in a.items() if k != "mean"}
    finite = {k: v for k, v in live.items() if np.isfinite(v)}
    row = {"kind": "C-D", "id": inst.get("id", tag), "tag": tag,
           "n": inst.get("n"), "seed": inst.get("seed"),
           "bar": MEAN_BAR, "bar_source": "rips_gate.nrmse (scale/rips_gate.py:118)",
           "control_mean_arm": a["mean"], "arms": a}
    if not finite:
        row.update(status="inapplicable",
                   reason="label sd is 0; NRMSE is NaN for every arm")
        return row
    best = min(finite, key=finite.get)
    row.update(best_arm=best, best_nrmse=finite[best],
               status="ok" if finite[best] < MEAN_BAR else "failed")
    return row


# --------------------------------------------------------------------------
# C-E(i) -- THE ORACLE IS A FUNCTION, NOT A STORED LABEL
# --------------------------------------------------------------------------
def c_e_executable(inst) -> dict:
    """Three properties a stored label cannot have, all on one instance.

    * SENSITIVE: a different `g` gives a different answer. A cached vector does
      not. This is the property that separates a function from a key.
    * DETERMINISTIC: the same input twice, bitwise.
    * SECOND ROUTE: the Kirchhoff cofactor oracle, which forms a different
      matrix, agrees inside the imported tolerance.
    """
    u, interior, _deg, PII, PIB = label(inst)
    g2 = inst["g"][::-1].copy()
    sens = float(np.max(np.abs(spec.resolvent_oracle(PII, PIB, g2) - u)))
    det = float(np.max(np.abs(spec.resolvent_oracle(PII, PIB, inst["g"]) - u)))
    why = dual.kirchhoff_applicable(inst["adj"], inst["boundary"])
    row = {"kind": "C-E-i", "id": inst.get("id"), "interior": len(interior),
           "sensitivity_to_g": sens, "determinism_dev": det,
           "tol": AGREEMENT_TOL, "tol_source": "kirchhoff.AGREEMENT_TOL",
           "oracle_cites": ["scale/r10_corpus_spec.py:254 resolvent_oracle",
                            "scale/r10_corpus_spec.py:259 averaging_oracle",
                            "scale/r10_dual_oracle.py:170 kirchhoff_extension"]}
    if why is None:
        uk, part, solves = dual.kirchhoff_extension(inst["adj"], inst["boundary"],
                                                    inst["g"])
        row.update(second_route_gap=float(np.max(np.abs(uk - u))),
                   partition_dev=part, kirchhoff_solves=solves)
    else:
        row.update(second_route_gap=None, second_route_reason=why)
    ok = (sens > AGREEMENT_TOL and det == 0.0
          and (row["second_route_gap"] is None
               or row["second_route_gap"] <= AGREEMENT_TOL))
    row["status"] = "ok" if ok else "failed"
    return row


def stored_label_oracle_check(u) -> dict:
    """THE PLANT for C-E(i): an 'oracle' that ignores its input and returns the
    label it was built with. Every downstream number it produces is correct.
    Only the sensitivity property refuses it."""
    cached = u.copy()

    def fake(_PII, _PIB, _g):
        return cached

    sens = float(np.max(np.abs(fake(None, None, None) - cached)))
    return {"kind": "C-E-i-mustfire", "plant": "cached label vector",
            "sensitivity_to_g": sens, "tol": AGREEMENT_TOL,
            "fires": bool(sens <= AGREEMENT_TOL)}


# --------------------------------------------------------------------------
# C-E(ii) -- ZERO ANSWER KEYS IN FILES
# --------------------------------------------------------------------------
#: `lean/.lake` is 35,950 vendored Lake build files (mathlib source and olean
#: artifacts); nothing in this campaign writes there. Excluded and COUNTED as
#: excluded, which is the whole point of reporting a denominator.
SCAN_SKIP_DIRS = (".git", os.path.join("lean", ".lake"))
SCAN_MAX_BYTES = 16 * 1024 * 1024


def scan_population(root: Path) -> tuple[list[Path], dict]:
    """(files to scan, files excluded per declared exclusion).

    The excluded trees are WALKED, not pruned, so the denominator reported is a
    count and not an estimate.
    """
    keep, excluded = [], {s: 0 for s in SCAN_SKIP_DIRS}
    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        hit = next((s for s in SCAN_SKIP_DIRS
                    if rel == s or rel.startswith(s + os.sep)), None)
        if hit is not None:
            excluded[hit] += len(filenames)
        else:
            keep.extend(Path(dirpath) / f for f in filenames)
    return keep, excluded


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(HERE)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _needles(labels: dict[str, np.ndarray]) -> dict[str, str]:
    """Literal renderings of label coordinates, one map literal -> instance.

    Coordinates 0, middle, last, MIN and MAX. `min`/`max` are in because a
    two-float summary of the label vector is the exact shape this corpus ships
    (`u_absorbing_range`), and a scan that could not see it would be reporting a
    zero it had arranged for itself.
    """
    out = {}
    for iid, u in labels.items():
        for v in (u[0], u[len(u) // 2], u[-1], u.min(), u.max()):
            for lit in (repr(float(v)), "%.12g" % v):
                if len(lit) >= 10:
                    out[lit] = iid
    return out


def scan_answer_keys(paths, labels: dict[str, np.ndarray]) -> dict:
    """Two independent routes over the same file population.

    TEXTUAL: any literal rendering of a label coordinate, anywhere.
    STRUCTURAL: any JSON numeric array whose length matches a label vector and
    whose entries match it inside the imported tolerance -- a whole key, stored.
    """
    needles = _needles(labels)
    rx = re.compile("|".join(sorted(map(re.escape, needles), key=len, reverse=True)))
    by_len: dict[int, list[tuple[str, np.ndarray]]] = {}
    for iid, u in labels.items():
        by_len.setdefault(len(u), []).append((iid, u))

    examined = skipped_binary = skipped_large = bytes_read = 0
    textual, structural = [], []
    for p in paths:
        try:
            size = p.stat().st_size
        except OSError:
            skipped_binary += 1
            continue
        if size > SCAN_MAX_BYTES:
            skipped_large += 1
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped_binary += 1
            continue
        examined += 1
        bytes_read += size
        for m in set(rx.findall(text)):
            textual.append({"file": _rel(p), "literal": m,
                            "instance": needles[m]})
        if p.suffix in (".json", ".jsonl"):
            for line in (text.splitlines() if p.suffix == ".jsonl" else [text]):
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except ValueError:
                    continue
                for arr in _numeric_arrays(doc):
                    for iid, u in by_len.get(len(arr), ()):
                        if float(np.max(np.abs(np.asarray(arr) - u))) <= AGREEMENT_TOL:
                            structural.append({"file": _rel(p), "instance": iid,
                                               "length": len(arr)})
    return {"kind": "C-E-ii", "files_examined": examined,
            "files_excluded_by_declaration": None, "skipped_binary": skipped_binary,
            "skipped_oversize": skipped_large, "bytes_read": bytes_read,
            "labels_searched": len(labels), "needles": len(needles),
            "textual_hits": textual, "structural_hits": structural,
            "n_textual": len(textual), "n_structural": len(structural)}


def _numeric_arrays(doc, out=None):
    """Every list of >= 16 numbers anywhere in a parsed JSON document."""
    out = [] if out is None else out
    if isinstance(doc, list):
        if len(doc) >= 16 and all(isinstance(v, (int, float)) and
                                  not isinstance(v, bool) for v in doc):
            out.append(doc)
        else:
            for v in doc:
                _numeric_arrays(v, out)
    elif isinstance(doc, dict):
        for v in doc.values():
            _numeric_arrays(v, out)
    return out


# --------------------------------------------------------------------------
# C-F(a) -- THE do()-BIT EFFECT
# --------------------------------------------------------------------------
def omega(inst):
    """`Omega = (I - P_II)^-1 P_IB`, one LU and `|B|` right-hand sides.

    Column `b` is `Delta u` for `do(g_b := 1)` against `do(g_b := 0)` and is the
    absorption probability at `b`. `Omega.sum(axis=1) == 1` is the partition of
    unity it.15 reports as `partition_dev`, reused here as the internal check.
    """
    PII, PIB, interior, deg = spec.blocks(inst["adj"], inst["boundary"])
    om = np.linalg.solve(np.eye(PII.shape[0]) - PII, PIB)
    return om, interior, PII, PIB


def c_f_effect(inst, *, tag: str = "front_door") -> dict:
    """C-F(a) on one instance: EVERY do()-bit, not a chosen one.

    The statistic is the campaign's own `flipper_dependence`,
    `mean|Delta y| / mean|y|` -- the form asserted at
    `tests/cameron/test_c1_propagate_registration.py:244`. `linf` is carried
    beside it because the corpus has already been bitten once by a bound whose
    norm was never stated (it.14 section 4).
    """
    om, interior, PII, PIB = omega(inst)
    u = spec.resolvent_oracle(PII, PIB, inst["g"])
    mean_u = float(np.mean(np.abs(u)))
    fd = np.mean(np.abs(om), axis=0) / mean_u if mean_u else np.full(om.shape[1], np.nan)
    linf = np.max(np.abs(om), axis=0)
    bset = set(inst["boundary"])
    no_interior_nbr = [b for b in inst["boundary"]
                       if not (set(inst["adj"][b]) - bset)]
    whole = 1.0 - 2.0 * u                       # do(g := 1 - g), exact and affine
    row = {"kind": "C-F-a", "id": inst.get("id", tag), "tag": tag,
           "n": inst.get("n"), "seed": inst.get("seed"),
           "boundary_size": len(inst["boundary"]), "interior": len(interior),
           "delta": DELTA, "delta_source": DELTA_SOURCE,
           "statistic": "flipper_dependence = mean|du| / mean|u|",
           "partition_dev": float(np.max(np.abs(om.sum(axis=1) - 1.0))),
           "fd_min": float(fd.min()), "fd_median": float(np.median(fd)),
           "fd_max": float(fd.max()),
           "bits_clearing_delta": int((fd >= DELTA).sum()),
           "bits_total": int(om.shape[1]),
           "bits_moving_nothing": int((linf <= AGREEMENT_TOL).sum()),
           "boundary_nodes_with_no_interior_neighbour": len(no_interior_nbr),
           "linf_min": float(linf.min()), "linf_median": float(np.median(linf)),
           "linf_max": float(linf.max()),
           "whole_boundary_flip_fd": float(np.mean(np.abs(whole)) / mean_u)
           if mean_u else None}
    row["status"] = "ok" if row["fd_min"] >= DELTA else "failed"
    return row


def plant_dead_boundary_bit():
    """THE PLANT for C-F(a), CONSTRUCTED so its reachability cannot depend on
    live data. Path `1-2-3` between boundary `0` and `4`, plus boundary node `5`
    hung off `0` alone. `5` has no interior neighbour, so `Omega[:, 5] == 0` and
    its do()-bit moves nothing. `spec.admissible` accepts the graph.
    """
    adj = {0: {1, 5}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}, 5: {0}}
    return {"adj": adj, "boundary": [0, 4, 5],
            "g": np.array([1.0, 0.0, 1.0]), "id": "plant-dead-bit"}


# --------------------------------------------------------------------------
# Tally -- five categories, each in its own column
# --------------------------------------------------------------------------
#: The fifth category is SATURN's, added at it.15 when a corpus rung was never
#: generated. A row with no `status` at all is counted as `unscored` rather than
#: folded into a pass, which is the same discipline one level up.
BUCKET = {"ok": "passed", "failed": "failed", "error": "errored",
          "inapplicable": "inapplicable", "not_reached": "not_reached"}


def tally(rows) -> dict:
    out = dict.fromkeys(list(BUCKET.values()) + ["unscored"], 0)
    for r in rows:
        out[BUCKET.get(r.get("status"), "unscored")] += 1
    return out


# --------------------------------------------------------------------------
# demo -- every clause shown able to fail, on CONSTRUCTED inputs
# --------------------------------------------------------------------------
def demo() -> None:
    fits, why = vram_gate.preflight(0, name="r10-it17-battery", host_mib=512)
    print(why, flush=True)
    if not fits:
        print("SKIPPED: the host gate refused. A refusal is the result.")
        return

    # 0. Constants are the shipped ones, checked rather than copied.
    assert FAIL_BAR is rips_gate.FAIL_BAR and PASS_BAR is rips_gate.PASS_BAR
    assert AGREEMENT_TOL is kirchhoff.AGREEMENT_TOL
    assert DELTA == 0.5, DELTA
    assert BUILDER.search("make_batch") and not BUILDER.search("zzz_not_a_builder")
    print(f"  DELTA = {DELTA} from {DELTA_SOURCE}")
    print(f"  BUILDER lifted from scale/p1prime.py:{BUILDER_LINE}")

    # 1. C-F(b): the classifier IS p1prime's, checked on its own 191 answers.
    b = front_door_binding()
    print(f"  scope classifier vs p1prime at {b['pin']}: "
          f"{b['mine_front_door']}/{b['tests_at_pin']} against shipped "
          f"{b['shipped_front_door']}, disagreements {len(b['disagreements'])}")
    assert b["status"] == "ok", b["disagreements"][:5]

    src = Path(__file__).read_text(encoding="utf-8")
    fd_src = _func_source(src, "control_front_door")
    hb_src = _func_source(src, "control_hand_built")
    assert front_door_calls(fd_src, aliases_of(src)), "front-door control is not"
    assert not front_door_calls(hb_src, aliases_of(src)), "hand-built control is"
    print("  L-SCOPE must-fire: front-door control classified front door; "
          "hand-built control classified BELOW the door")

    # 2. One front-door instance carries the rest of the self-check.
    inst = control_front_door(128, 4.0, 0x3A140002)
    u, interior, _deg, PII, PIB = label(inst)
    print(f"  instance {inst['id']}  lambda_2={inst['lambda_2']:.10f}  "
          f"|I|={len(interior)}  |B|={len(inst['boundary'])}")

    # 3. C-D holds, and its control reads exactly 1.0.
    d = c_d(inst)
    assert abs(d["control_mean_arm"] - 1.0) < 1e-12, d["control_mean_arm"]
    assert d["status"] == "ok", d
    print(f"  C-D: best arm {d['best_arm']} NRMSE={d['best_nrmse']:.6f} "
          f"(probe_F3 {d['arms']['probe_F3']:.6f}, mean control "
          f"{d['control_mean_arm']:.6f})")

    # 3b. C-D MUST-FIRE, on a CONSTRUCTED label: the same instance with its
    #     label shuffled by a seed declared in the module header.
    y = np.random.default_rng(CD_PLANT_SEED).permutation(u)
    dp = c_d(inst, y, tag="plant-shuffled-label")
    print(f"  C-D must-fire (label shuffled, seed 0x{CD_PLANT_SEED:08x}): "
          f"best {dp['best_arm']} {dp['best_nrmse']:.6f} -> {dp['status']}")
    assert dp["arms"][f"khop_{max(KHOP_KS)}"] > MEAN_BAR, dp["arms"]
    assert dp["status"] == "failed", dp

    # 4. C-E(i) holds, and a cached-label oracle is refused.
    e = c_e_executable(inst)
    assert e["status"] == "ok", e
    print(f"  C-E(i): sensitivity {e['sensitivity_to_g']:.6e}, determinism "
          f"{e['determinism_dev']:.1e}, second route {e['second_route_gap']:.3e}")
    mf = stored_label_oracle_check(u)
    assert mf["fires"], mf
    print(f"  C-E(i) must-fire: cached-label oracle sensitivity "
          f"{mf['sensitivity_to_g']:.1e} <= {AGREEMENT_TOL} -- REFUSED")

    # 5. C-E(ii) must-fire, bound to a CONSTRUCTED file rather than to the tree.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        key = Path(td) / "planted_key.jsonl"
        key.write_text(json.dumps({"u": [float(v) for v in u]}) + "\n",
                       encoding="utf-8")
        clean = Path(td) / "clean.jsonl"
        clean.write_text(json.dumps({"note": "no labels here"}) + "\n",
                         encoding="utf-8")
        hit = scan_answer_keys([key], {"demo": u})
        miss = scan_answer_keys([clean], {"demo": u})
    assert hit["n_structural"] == 1 and hit["n_textual"] > 0, hit
    assert miss["n_structural"] == 0 and miss["n_textual"] == 0, miss
    print(f"  C-E(ii) must-fire: planted key -> {hit['n_structural']} structural, "
          f"{hit['n_textual']} textual; clean control -> 0 / 0")

    # 6. C-F(a) on the live instance, and the CONSTRUCTED dead-bit plant.
    f = c_f_effect(inst)
    print(f"  C-F(a): fd min {f['fd_min']:.6f} median {f['fd_median']:.6f} max "
          f"{f['fd_max']:.6f} against delta {DELTA} -> {f['status']}; "
          f"linf max {f['linf_max']:.6f}; partition_dev {f['partition_dev']:.2e}")
    dead = plant_dead_boundary_bit()
    spec.admissible(dead["adj"], dead["boundary"])       # the front door accepts it
    fp = c_f_effect(dead, tag="plant-dead-bit")
    assert fp["bits_moving_nothing"] >= 1 and fp["fd_min"] == 0.0, fp
    assert fp["status"] == "failed", fp
    print(f"  C-F(a) must-fire: dead boundary bit moves "
          f"{fp['linf_min']:.1e} in linf -> {fp['status']}")

    # 7. THE L-SCOPE POINT, in numbers: the same clause on a hand-built control
    #    that no production builder can emit.
    star = c_f_effect(control_hand_built(), tag="plant-hand-built-star")
    print(f"  L-SCOPE: hand-built star reads fd_min {star['fd_min']:.6f} -> "
          f"{star['status']}, while the front-door instance reads "
          f"{f['fd_min']:.6f} -> {f['status']}")
    assert star["status"] == "ok" and f["status"] == "failed", (star, f)
    print("demo OK")


def _func_source(src: str, name: str) -> str:
    """The source of one top-level function, by line span from the AST."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(src.splitlines()[node.lineno - 1:node.end_lineno])
    raise LookupError(name)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--write" not in args:
        demo()
        return 0

    t0 = time.time()
    fits, why = vram_gate.preflight(0, name="r10-it17-battery", host_mib=512)
    print(why, flush=True)
    if not fits:
        OUT.write_text(json.dumps({"kind": "refusal", "gate": why}) + "\n",
                       encoding="utf-8")
        return 0

    rows = [{"kind": "provenance", "delta": DELTA, "delta_source": DELTA_SOURCE,
             "fail_bar": FAIL_BAR, "pass_bar": PASS_BAR,
             "agreement_tol": AGREEMENT_TOL, "mean_bar": MEAN_BAR,
             "khop_ks": list(KHOP_KS), "probe_seed": f"0x{PROBE_SEED:08x}",
             "cd_plant_seed": f"0x{CD_PLANT_SEED:08x}", "gate": why}]

    print("front door: dual.reproduce_arm() ...", flush=True)
    corpus = front_door_corpus()
    print(f"  {len(corpus)} rungs in {time.time() - t0:.1f}s", flush=True)

    rows.append(front_door_binding())
    rows.append(head_census())

    for inst in corpus:
        rows.append(c_d(inst))
        rows.append(c_e_executable(inst))
        rows.append(c_f_effect(inst))

    # below-door entry on the SAME rungs: same object, fewer stages
    below = []
    for inst in corpus:
        b = control_below_door(inst["n"], inst["mean_deg"], int(inst["seed"], 16),
                               len(inst["boundary"]))
        gap = max(float(np.max(np.abs(b["g"] - inst["g"]))),
                  float(b["boundary"] != inst["boundary"]))
        below.append(gap)
    rows.append({"kind": "C-F-b", "id": "below-door-vs-front-door",
                 "rungs": len(corpus), "max_instance_gap": max(below),
                 "front_door_callable": "scale/r10_dual_oracle.py:306 make_rung",
                 "below_door_callable": "scale/r10_admissibility.py:178 recover",
                 "stages_skipped": ["spec.stratify (r10_corpus_spec.py:361)",
                                    "spec._lambda_at (:352)",
                                    "spec.admissible (:162, reached only via :356)",
                                    "spec.spectrum (:222)",
                                    "the in_band check (r10_dual_oracle.py:329)"],
                 "status": "ok"})

    # plants, all constructed
    inst0 = corpus[0]
    u0 = label(inst0)[0]
    rows.append(c_d(inst0, np.random.default_rng(CD_PLANT_SEED).permutation(u0),
                    tag="plant-shuffled-label"))
    rows.append(stored_label_oracle_check(u0))
    rows.append(c_f_effect(plant_dead_boundary_bit(), tag="plant-dead-bit"))
    rows.append(c_f_effect(control_hand_built(), tag="plant-hand-built-star"))

    # C-E(ii): the whole repo, once
    print("scanning the tree for answer keys ...", flush=True)
    it14, repro, widen, not_reached, checks = adm.load_instances()
    labels = {}
    for inst in repro + widen:
        labels[inst["id"]] = adm.solve_u(inst["adj"], inst["boundary"],
                                         inst["g"])[0]
    paths, excluded = scan_population(HERE)
    scan = scan_answer_keys(paths, labels)
    scan["files_excluded_by_declaration"] = excluded
    scan["exclusions"] = list(SCAN_SKIP_DIRS)
    scan["recovery_max_range_delta"] = max(
        (c["range_delta"] for c in checks if c["range_delta"] is not None),
        default=None)
    scan["status"] = "ok" if scan["n_structural"] == 0 else "failed"
    rows.append(scan)
    rows.extend({"kind": "C-A-carryover", "id": r["id"], "status": "not_reached",
                 "reason": r["reason"]} for r in not_reached)

    kinds = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r)
    rows.append({"kind": "summary", "seconds": round(time.time() - t0, 1),
                 "per_clause": {k: tally(v) for k, v in kinds.items()},
                 "overall": tally(rows)})

    OUT.write_text("".join(json.dumps(r, sort_keys=True, default=float) + "\n"
                           for r in rows), encoding="utf-8")
    print("wrote", OUT, len(rows), "rows", f"{time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
