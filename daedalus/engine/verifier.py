"""Daedalus verifier: stages V0..V5, run in the registered order V0, V1, V4, V2, V3, V5
(registry/bars.json addendum_1.stage_order: learnability gates every comparative read).

    python verifier.py <candidate_dir> [...]          # prints one JSON verdict per candidate

verify(dir) -> {"name", "verdict": "ACCEPT"|"REJECT", "stage", "law", "reason", "detail"}
The first failing stage rejects. Candidate contract: proposer.py docstring.
Bars come only from registry/bars.json, read once at import; candidates never
write it (sandbox write-deny + before/after sha256 of engine/, registry/, sealed/).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DAEDALUS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(DAEDALUS))
SEALED = os.path.join(DAEDALUS, "sealed")
PROTECTED = [HERE, os.path.join(DAEDALUS, "registry"), SEALED]
sys.path.insert(0, HERE)
import beds  # noqa: E402

import planning_bed as PB  # noqa: E402

BARS = json.load(open(os.path.join(DAEDALUS, "registry", "bars.json")))
ADD = BARS["addendum_1"]
R2 = json.load(open(os.path.join(DAEDALUS, "registry", "bars_r2.json")))["r2_b_ranker"]
R3 = json.load(open(os.path.join(DAEDALUS, "registry", "bars_r3.json")))["r3_d_ranker_v3"]
R4 = json.load(open(os.path.join(DAEDALUS, "registry", "bars_r4.json")))
RBEDS = {**R2["beds"], **R4["r4_c_bed"]["beds"]}  # same inner dicts: r2 beds read exactly as registered
# bars_r4 r4_d: every verdict records the sha256 of the code that produced it and of every bars file it read, as read
PROVENANCE = {rel: hashlib.sha256(open(os.path.join(DAEDALUS, rel), "rb").read()).hexdigest() for rel in
              ("engine/verifier.py", "engine/sandbox_runner.py", "engine/planning_bed.py", "engine/beds.py",
               "registry/bars.json", "registry/bars_r2.json", "registry/bars_r3.json", "registry/bars_r4.json")}
RANKER_INTEGRITY = {"V0", "V0.sandbox", "V0.init", "V1", "V4", "V2.ceiling", "V2.order"}
RANKER_CLAIMS = {"permutation_equivariant", "bounded"}
BASE_RANGE = 1.0  # base = -(distance rank)/(K-1) spans [-1, 0] for every K
FORBIDDEN_KEYS = {"bar", "bars", "threshold", "thresholds", "margin", "opponent", "opponents",
                  "baseline", "baselines", "seed", "seeds", "steps", "lr", "budget", "eval", "test"}
KNOWN_CLAIMS = {"causal", "row_stochastic", "pole_free"}
# Static source scan: candidates are pure modules. Names of protected places, I/O and dynamic code.
SRC_BANNED = [r"\bsealed\b", r"\bregistry\b", r"bars\.json", r"secret\.json", r"verifier", r"sandbox_runner",
              r"house-events", r"\bsocket\b", r"\bsubprocess\b", r"\bctypes\b", r"\burllib\b", r"\brequests\b",
              r"\bhttp\b", r"os\.system", r"__import__", r"\bexec\s*\(", r"\beval\s*\(", r"\bbase64\b",
              r"\bcodecs\b", r"\bmarshal\b", r"\bimportlib\b", r"\bopen\s*\(", r"np\.load", r"torch\.load",
              r"\bpickle\b", r"\bshutil\b", r"\bos\.remove\b", r"\bPath\s*\("]
WIN = 32  # truth-window length for literal / array matching


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree_hash(paths=PROTECTED) -> str:
    h = hashlib.sha256()
    for root in paths:
        for dp, dn, fn in sorted(os.walk(root)):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                p = os.path.join(dp, f)
                h.update(p.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()


def _truth_windows():
    """sha256 set of every WIN-long window of every sealed label array, and whole-file hashes."""
    files, wins = set(), set()
    for f in os.listdir(SEALED):
        p = os.path.join(SEALED, f)
        files.add(_sha(open(p, "rb").read()))
        if f.endswith(".npz"):
            z = np.load(p)
            for k in z.files:
                flat = z[k].astype(np.int64).ravel()
                files.add(_sha(flat.tobytes()))
                for i in range(0, len(flat) - WIN + 1):
                    wins.add(_sha(flat[i:i + WIN].tobytes()))
    # the bed's own transition table is truth too (hard-coding it is hard-coding the answer)
    table, _ = beds._a5()
    flat = table.astype(np.int64).ravel()
    for i in range(0, len(flat) - WIN + 1):
        wins.add(_sha(flat[i:i + WIN].tobytes()))
    return files, wins


_TRUTH = None


def _contains_truth(seq) -> bool:
    arr = np.asarray(seq, dtype=np.int64).ravel()
    if _sha(arr.tobytes()) in _TRUTH[0]:
        return True
    return any(_sha(arr[i:i + WIN].tobytes()) in _TRUTH[1] for i in range(0, len(arr) - WIN + 1))


def _int_literals(src: str):
    """Every list/tuple literal of >= WIN ints in the source, flattened (nested lists included)."""
    out = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            vals = []
            stack = [node]
            while stack:
                n = stack.pop(0)
                if isinstance(n, (ast.List, ast.Tuple)):
                    stack = list(n.elts) + stack
                elif isinstance(n, ast.Constant) and isinstance(n.value, int):
                    vals.append(n.value)
            if len(vals) >= WIN:
                out.append(vals)
    for m in re.finditer(r"(?:-?\d+[\s,;]+){%d,}-?\d+" % (WIN - 1), src):  # ints hidden in strings
        out.append([int(x) for x in re.findall(r"-?\d+", m.group(0))])
    return out


def _walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def _fetched_owner_ids():
    p = os.path.join(DAEDALUS, "owners.json")
    d = json.load(open(p))
    owners = d.get("owners", d if isinstance(d, list) else [])
    return {o["id"] for o in owners if o.get("fetched") is True}


def reject(res, stage, law, reason, **detail):
    res.update(verdict="REJECT", stage=stage, law=law, reason=reason, detail=detail)
    return res


# ------------------------------------------------------------------ V0 static
def v0_static(cdir, res, bed=None):
    mpath = os.path.join(cdir, "manifest.json")
    try:
        man = json.load(open(mpath))
        if bed is not None:  # bars_r4 r4_c: the verifier reads an arm on another registered bed; every check still applies
            man["bed"] = bed
    except Exception as e:
        return None, reject(res, "V0", "L-REPRO", f"manifest unreadable: {e}")
    res["name"] = man.get("name", os.path.basename(cdir))
    if man.get("kind") not in ("operator", "bed", "ranker"):
        return None, reject(res, "V0", "schema", f"kind {man.get('kind')!r}")
    bad = sorted({p for p, k in _walk_keys(man) if str(k).lower() in FORBIDDEN_KEYS})
    if bad:
        return None, reject(res, "V0", "L-FIXED-BAR / L-BR", "manifest sets verifier-owned fields", keys=bad)
    fetched = _fetched_owner_ids()
    unfetched = [c for c in man.get("citations", []) if c not in fetched]
    if unfetched:
        return None, reject(res, "V0", "fetch-before-cite", "citation not fetched this run", citations=unfetched)
    for claim in man.get("claims", []):
        if claim not in (RANKER_CLAIMS if man["kind"] == "ranker" else KNOWN_CLAIMS):
            return None, reject(res, "V1", "L-AUDIT (claim without instrument)", f"claim {claim!r} has no V1 spec")
    if man["kind"] == "operator" and man.get("bed") not in beds.BEDS:
        return None, reject(res, "V0", "schema", f"bed {man.get('bed')!r} not registered")
    if man["kind"] == "ranker":
        if man.get("bed") not in RBEDS:
            return None, reject(res, "V0", "schema", f"ranker bed {man.get('bed')!r} not registered")
        view = man.get("features", "predicted")
        if view not in RBEDS[man["bed"]]["feature_views"]:  # never substitute silently: the author meant this view
            return None, reject(res, "V0", "L-SEALED-TRUTH", f"feature view {view!r} is not registered; executed outcomes are sealed truth")
        eps = man.get("epsilon")
        if "bounded" in man.get("claims", []) and not (isinstance(eps, (int, float)) and not isinstance(eps, bool) and 0 < eps < float("inf")):
            return None, reject(res, "V1", "L-AUDIT (claim without instrument)", f"claims bounded with epsilon {eps!r}")
        if "bounded" in man.get("claims", []) and 2 * eps >= BASE_RANGE:  # bars_r4 r4_b
            return None, reject(res, "V1", "L-AUDIT (vacuous claim)",
                                f"claims bounded with epsilon {eps} but 2 * epsilon >= base range {BASE_RANGE}: every ordering is reachable")
    for dp, _, fn in os.walk(cdir):
        for f in fn:
            p = os.path.join(dp, f)
            raw = open(p, "rb").read()
            if _sha(raw) in _TRUTH[0]:
                return None, reject(res, "V0", "L-SEALED-TRUTH", "shipped file hashes to sealed truth", file=f)
            if f.endswith((".npy", ".npz")):
                try:
                    z = np.load(p, allow_pickle=False)
                    arrs = [z[k] for k in z.files] if hasattr(z, "files") else [z]
                except Exception as e:
                    return None, reject(res, "V0", "L-SEALED-TRUTH", f"unloadable array {f}: {e}")
                if any(_contains_truth(a) for a in arrs if np.issubdtype(np.asarray(a).dtype, np.number)):
                    return None, reject(res, "V0", "L-SEALED-TRUTH", "shipped array contains sealed truth", file=f)
            if f.endswith(".py"):
                src = raw.decode("utf-8", "replace")
                try:
                    py_compile.compile(p, cfile=os.path.join(tempfile.gettempdir(), "daedalus_pyc.tmp"), doraise=True)
                except py_compile.PyCompileError as e:
                    return None, reject(res, "V0", "compiles", str(e)[:200])
                for pat in SRC_BANNED:
                    m = re.search(pat, src)
                    if m:
                        return None, reject(res, "V0", "L-SEALED-TRUTH / sandbox", f"banned source token {m.group(0)!r}", file=f)
                if any(_contains_truth(v) for v in _int_literals(src)):
                    return None, reject(res, "V0", "L-SEALED-TRUTH", "source literal contains truth", file=f)
    return man, None


# ------------------------------------------------------------------ sandbox
class InfraError(RuntimeError):
    pass


def run_sandbox(cdir, man, job, timeout=1800, tries=3):
    """Retry runner deaths; a candidate's own exception is returned in out['error'] and is a verdict."""
    for _ in range(tries):
        out, before, after = _run_sandbox(cdir, man, job, timeout)
        if "infra" not in out or out["violations"]:  # r2: an empty-stderr death has infra "" and is still infra
            return out, before, after
        time.sleep(10)
    raise InfraError(out["infra"])


def _run_sandbox(cdir, man, job, timeout):
    import torch
    before = tree_hash()
    with tempfile.TemporaryDirectory(prefix="daedalus_") as td:
        job = dict(job, cand_dir=os.path.abspath(cdir), repo=REPO, entry=man.get("entry", "candidate.py"))
        jp, op = os.path.join(td, "job.pt"), os.path.join(td, "out.pt")
        torch.save(job, jp)
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-I", "-B", os.path.join(HERE, "sandbox_runner.py"), jp, op],
                               capture_output=True, text=True, timeout=timeout, cwd=td)
            err = p.stderr[-2000:]
        except subprocess.TimeoutExpired:
            return {"infra": "timeout", "error": "timeout", "violations": []}, before, tree_hash()
        vlog = os.path.join(td, "violations.log")
        logged = open(vlog).read().split("\n") if os.path.exists(vlog) else []
        if not os.path.exists(op):  # the runner process died (OOM, driver): infrastructure, never a verdict
            return {"infra": err[-300:], "error": "infra", "violations": [v for v in logged if v]}, before, tree_hash()
        out = torch.load(op, weights_only=True)
        out["violations"] = sorted(set(out.get("violations", [])) | {v for v in logged if v})
        out["wall_s"] = time.time() - t0
    return out, before, tree_hash()


def _job(man, bed, seeds, budget, tag):
    import torch
    gen, vocab, ncls = beds.BEDS[bed]
    b = BARS["beds"][bed]
    trX, trY, evX, evY = [], [], [], []
    for k, s in enumerate(seeds):
        X, Y, band = gen(20000, beds.secret_seed(f"{tag}|train|{s}"))
        Xe, Ye, _ = gen(4000, beds.secret_seed(f"{tag}|eval|{s}"))
        trX.append(torch.tensor(X)); trY.append(torch.tensor(Y)); evX.append(torch.tensor(Xe)); evY.append(Ye)
    probe = torch.tensor(gen(64, beds.secret_seed(f"{tag}|probe"))[0])
    # addendum_2: secret relabeling of input tokens (sigma) and classes (pi) per tag, so task truth
    # baked into a candidate (tables, oracles, delayed oracles) answers in the wrong alphabet
    rr = np.random.default_rng(beds.secret_seed(f"{tag}|relabel"))
    sig, pi = torch.tensor(rr.permutation(vocab)), rr.permutation(ncls)
    trX = [sig[x] for x in trX]; evX = [sig[x] for x in evX]; probe = sig[probe]
    trY = [torch.tensor(pi[y.numpy()]) for y in trY]; evY = [pi[y] for y in evY]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    job = {"mode": "operator", "spec": {"vocab": vocab, "n_classes": ncls, "T": b["T"], "param_budget": budget},
           "seeds": seeds, "train_X": trX, "train_Y": trY, "eval_X": evX, "probe_X": probe,
           "steps": b["train_steps"], "batch": b["batch"], "lr": b["lr"], "device": dev}
    return job, evY, band


def _band_acc(pred, Y, band):
    pred = np.asarray(pred)
    return float((pred[:, band] == Y[:, band]).mean())


_FLOOR_CACHE, _POOL_CACHE = {}, {}
CONTROLS = os.path.join(DAEDALUS, "controls")


def bed_floors(bed):
    if bed not in _FLOOR_CACHE:
        gen = beds.BEDS[bed][0]
        X, Y, band = gen(20000, beds.secret_seed(f"floors|{bed}|train"))
        Xe, Ye, _ = gen(4000, beds.secret_seed(f"floors|{bed}|eval"))
        _FLOOR_CACHE[bed] = beds.floors(X, Y, Xe, Ye, band)
    return _FLOOR_CACHE[bed]


def seeds_for(tag, k=3):
    """bars_r2 r2_c: k + 1 secret run seeds, the first draw whose set has both parities and every residue
    mod 3 and mod 4 -- which residues is decided by the salt, so no residue predicate can be whitelisted.
    Run seeds are never written to results."""
    for j in range(10000):
        s = [beds.secret_seed(f"{tag}|seed{i}|try{j}") for i in range(k + 1)]
        if all({x % m for x in s} == set(range(m)) for m in (2, 3, 4)):
            return s
    raise RuntimeError("no admissible seed set")


def train_eval(cdir, man, budget, tag):
    """One sandboxed run of a candidate at a param budget; returns (out, evY, band, before, after)."""
    bed = man["bed"]
    seeds = seeds_for(tag, BARS["beds"][bed]["v4_seeds"])
    job, evY, band = _job(man, bed, seeds, budget, tag)
    # init-oracle instrument: the runner reports post-training preds; the untrained read is a second run at steps=0
    out, before, after = run_sandbox(cdir, man, job)
    return out, evY, band, before, after


def init_accs(cdir, man, budget, tag):
    bed = man["bed"]
    seeds = seeds_for(tag, BARS["beds"][bed]["v4_seeds"])
    job, evY, band = _job(man, bed, seeds, budget, tag)
    job["steps"] = 0
    out, _, _ = run_sandbox(cdir, man, job)
    if out.get("error") or out["violations"] or "runs" not in out:
        return None
    return [_band_acc(r["pred"], evY[k], band) for k, r in enumerate(out["runs"])]


def pool_accs(bed, budget, names, tag):
    res = {}
    for name in names:
        key = (bed, budget, name, tag)
        if key not in _POOL_CACHE:
            cdir = os.path.join(CONTROLS, name)
            man = json.load(open(os.path.join(cdir, "manifest.json")))
            out, evY, band, _, _ = train_eval(cdir, dict(man, bed=bed), budget, tag)
            if out.get("error"):
                raise InfraError(f"pool member {name} failed: {out['error']}")  # a broken control is not a verdict
            _POOL_CACHE[key] = [_band_acc(r["pred"], evY[k], band) for k, r in enumerate(out["runs"])]
        res[name] = _POOL_CACHE[key]
    return res


# ------------------------------------------------------------------ operator stages
def verify_operator(cdir, man, res, pool=None):
    bed = man["bed"]
    b = BARS["beds"][bed]
    budget = b["param_budget"]
    tag = "m0"  # every candidate at M0 sees the same secret draws, so pool results are comparable
    out, evY, band, before, after = train_eval(cdir, man, budget, tag)
    res["wall_s"] = round(out.get("wall_s", 0), 1)
    # ---- V0 runtime (sandbox)
    if before != after:
        return reject(res, "V0.sandbox", "engine cannot modify the verifier", "protected tree changed during the run")
    if out["violations"]:
        return reject(res, "V0.sandbox", "L-SEALED-TRUTH / sandbox", "sandbox violation", violations=out["violations"][:5])
    if out.get("error"):
        return reject(res, "V0", "runs", out["error"][:300])
    T, C = b["T"], beds.BEDS[bed][2]
    if list(out["probe_shape"].tolist()) != [64, T, C]:
        return reject(res, "V0", "shapes", f"output shape {out['probe_shape'].tolist()} != [64, {T}, {C}]")
    tol = b["param_tolerance_frac"]
    npar = [r["n_params"] for r in out["runs"]]
    if any(abs(n - budget) > tol * budget for n in npar):
        return reject(res, "V0", "L-EQ", f"trainable params {npar} outside {budget} +-{tol:.0%}")
    nhid = max(r["n_hidden"] for r in out["runs"])
    if nhid > 0.10 * budget:
        return reject(res, "V0", "L-EQ (hidden capacity)", f"{nhid} frozen params + buffer entries > 10% of {budget}")
    ia = init_accs(cdir, man, budget, tag)
    fl = bed_floors(bed)
    best_floor = max(fl.values())
    res["floors"] = fl
    if ia is None:
        return reject(res, "V0", "runs", "untrained read failed")
    res["init_acc"] = ia
    if max(ia) > best_floor + 0.05:
        return reject(res, "V0.init", "hard-coded outputs", f"untrained model scores {max(ia):.4f} > best floor {best_floor:.4f} + 0.05")
    # ---- V1 invariants
    tolc = ADD["v1_causal_tol"]
    dev = max(r["causal_dev"] for r in out["runs"])
    res["causal_dev"] = dev
    if dev > tolc:
        return reject(res, "V1", "causal mask", f"future tokens move past logits by {dev:.3g} > {tolc}")
    claims = man.get("claims", [])
    for r in out["runs"]:
        if "row_stochastic" in claims:
            if "mixing_W" not in r:
                return reject(res, "V1", "L-AUDIT", "claims row_stochastic, exposes no mixing_weights")
            W = r["mixing_W"].numpy()
            rs = np.abs(W.sum(-1) - 1).max()
            upper = np.abs(np.triu(W, 1)).max() if W.shape[-1] == W.shape[-2] else 0.0
            if W.min() < -ADD["v1_row_sum_tol"] or rs > ADD["v1_row_sum_tol"] or upper > ADD["v1_row_sum_tol"]:
                return reject(res, "V1", "row sums", f"row-sum error {rs:.3g}, min {W.min():.3g}, future mass {upper:.3g}")
        if "pole_free" in claims:
            if "resolvent_W" not in r:
                return reject(res, "V1", "L-AUDIT", "claims pole_free, exposes no resolvent_params")
            W, g = r["resolvent_W"].numpy(), r["resolvent_gamma"]
            m = g * np.abs(W).sum(-1).max()
            if not m < 1:
                return reject(res, "V1", "pole-freedom (D1)", f"gamma*max rowsum|W| = {m:.4f} >= 1")
    # ---- V4 learnability, seeds, reproduction
    accs = [_band_acc(r["pred"], evY[k], band) for k, r in enumerate(out["runs"])]
    res["band_acc"] = accs
    if min(accs) < b["v4_learnability_in_distribution"]:
        return reject(res, "V4", "L-LEARN", f"band accuracy {['%.4f' % a for a in accs]} < {b['v4_learnability_in_distribution']} on >=1 of {len(accs)} verifier seeds")
    rep = man.get("reported", {}).get("band_acc")
    if rep is not None and abs(rep - float(np.mean(accs))) > b["v4_repro_tolerance_abs"]:
        return reject(res, "V4", "L-REPRO", f"reported {rep} vs measured {np.mean(accs):.4f}")
    # ---- V2 floors
    if np.mean(accs) < best_floor + b["v2_margin_over_best_floor"]:
        return reject(res, "V2", "L-SHORTCUT", f"mean {np.mean(accs):.4f} < best floor {best_floor:.4f} + margin")
    # ---- V3 trained best responses
    pool = pool or ADD["v3_pool_a5"]
    pa = pool_accs(bed, budget, pool, tag)
    best_name = max(pa, key=lambda n: np.mean(pa[n]))
    margin = float(np.mean(accs) - np.mean(pa[best_name]))
    res["v3"] = {"pool": {n: v for n, v in pa.items()}, "best": best_name, "margin": margin}
    if margin < ADD["v3_margin"]:
        return reject(res, "V3", "L-BR", f"margin {margin:+.4f} over trained {best_name} < {ADD['v3_margin']}")
    # ---- V5 ladder (cheapest instance: the next rung doubles params)
    for rung in ADD["v5_rungs"][1:]:
        o2, evY2, band2, _, _ = train_eval(cdir, man, rung, tag)
        if o2.get("error") or o2["violations"]:
            return reject(res, "V5", "L-SCALE", f"rung {rung} failed: {o2.get('error') or o2['violations'][:2]}")
        a2 = [_band_acc(r["pred"], evY2[k], band2) for k, r in enumerate(o2["runs"])]
        p2 = pool_accs(bed, rung, [best_name], tag)[best_name]
        m2 = float(np.mean(a2) - np.mean(p2))
        res.setdefault("v5", {})[str(rung)] = {"acc": a2, "pool": p2, "margin": m2}
        if m2 < ADD["v3_margin"]:
            return reject(res, "V5", "L-SCALE", f"margin {m2:+.4f} at rung {rung}")
    res.update(verdict="ACCEPT", stage="V5", law="", reason="passed V0-V5 (V6 = Inspector + human sign-off)")
    return res


# ------------------------------------------------------------------ ranker stages (bars_r2 r2_b)
PROBE_SCALES = (1.0, 10.0, 100.0, 1000.0)


def ranker_probes(zhat, rng, n_perm=8):
    """Probes at four feature scales and n_perm secret per-row permutations of them."""
    Z = np.concatenate([zhat * c for c in PROBE_SCALES]).astype(np.float32)
    perms = np.stack([rng.permuted(np.tile(np.arange(Z.shape[1]), (len(Z), 1)), axis=1) for _ in range(n_perm)])
    return Z, np.stack([np.take_along_axis(Z, p[..., None], 1) for p in perms]), perms


def equivariance_dev(S, Sp, perms):
    """max |f(Pz) - P f(z)| and its tolerance 1e-4 (1 + max|f(z)|)."""
    want = np.stack([np.take_along_axis(S, p, 1) for p in perms])
    return float(np.abs(Sp - want).max()), 1e-4 * (1 + float(np.abs(S).max()))


def bound_dev(S, Z):
    return float(np.abs(S - PB.base_score(Z)).max())


def shuffle_rows(X, Y, rng):
    """Secret per-row candidate shuffle; returns shuffled features and the relocated best index."""
    perm = rng.permuted(np.tile(np.arange(X.shape[1]), (len(X), 1)), axis=1)
    return np.take_along_axis(X, perm[..., None], 1), (perm == Y[:, None]).argmax(1), perm


def ceiling_breach(hit, bayes_hit, n):
    """None if hit <= Bayes + 4 SE + 0.01 MC slack, else the line it crossed (only sealed truth crosses it)."""
    line = bayes_hit + 4 * np.sqrt(bayes_hit * (1 - bayes_hit) / n) + 0.01
    return None if hit <= line else float(line)


_RDATA, _RFLOOR = {}, {}


def _rdata(bedname, seeds, tag):
    key = (bedname, seeds, tag)
    if key not in _RDATA:
        b = RBEDS[bedname]
        d = {k: [] for k in ("trX", "trY", "evX", "evY", "evXs", "evYs", "evP", "bayes")}
        for s in seeds:
            tr = PB.make_bed(b["n_train"], b["sigma"], beds.secret_seed(f"{tag}|rtrain|{s}"), K=b["K"])
            ev = PB.make_bed(b["n_eval"], b["sigma"], beds.secret_seed(f"{tag}|reval|{s}"), K=b["K"])
            Xs, Ys, P = shuffle_rows(PB.features(ev), ev["best"], np.random.default_rng(beds.secret_seed(f"{tag}|rshuf|{s}")))
            d["evP"].append(P)
            d["trX"].append(PB.features(tr)); d["trY"].append(tr["best"]); d["evX"].append(PB.features(ev))
            d["evY"].append(ev["best"]); d["evXs"].append(Xs); d["evYs"].append(Ys); d["bayes"].append(PB.bayes_pick(ev))
        _RDATA[key] = d
    return _RDATA[key]


def _hit(preds, labels):
    return float(np.mean([np.mean(np.asarray(p) == y) for p, y in zip(preds, labels)]))


def ranker_floors(bedname, d, tag):
    """Candidate-independent V2 floors: latent distance, set position, a budget-sized pointwise MLP."""
    if (bedname, tag) not in _RFLOOR:
        import torch
        import torch.nn.functional as F
        b = RBEDS[bedname]
        slot = int(np.bincount(d["trY"][0], minlength=b["K"]).argmax())
        torch.manual_seed(0)
        h = 128
        m = torch.nn.Sequential(torch.nn.Linear(b["F"] + 1, h), torch.nn.GELU(), torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, 1))
        f = lambda X: m(torch.cat([X, X.norm(dim=-1, keepdim=True)], -1)).squeeze(-1)
        X, Y = torch.tensor(d["trX"][0]), torch.tensor(d["trY"][0])
        opt, g = torch.optim.AdamW(m.parameters(), lr=b["lr"]), torch.Generator().manual_seed(0)
        for _ in range(b["train_steps"]):
            idx = torch.randint(0, len(X), (b["batch"],), generator=g)
            loss = F.cross_entropy(f(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), b["clip"]); opt.step()
        with torch.no_grad():
            pw = [f(torch.tensor(Xe)).argmax(1).numpy() for Xe in d["evX"]]
        _RFLOOR[bedname, tag] = {"latent_distance": _hit([PB.dist_pick(Xe) for Xe in d["evX"]], d["evY"]),
                            "set_position": _hit([np.full(len(y), slot) for y in d["evY"]], d["evY"]),
                            "pointwise_mlp": _hit(pw, d["evY"])}
    return dict(_RFLOOR[bedname, tag])  # r3: a copy; the caller adds its own null


def verify_ranker(cdir, man, res, tag="m0", v3=True):
    """tag = the secret-seed draw (bars_r3 r3_c); v3=False is the pool members' own read."""
    import torch
    bedname = man["bed"]
    b = RBEDS[bedname]
    seeds = seeds_for(tag)
    d = _rdata(bedname, tuple(seeds), tag)
    prng = np.random.default_rng(beds.secret_seed(f"{tag}|rprobe"))
    Z, Zp, perms = ranker_probes(PB.features(PB.make_bed(b["n_probe"], b["sigma"], beds.secret_seed(f"{tag}|rprobe"), K=b["K"])), prng)
    null_Y = d["trY"][0][np.random.default_rng(beds.secret_seed(f"{tag}|rnull")).permutation(b["n_train"])]
    # bars_r4 r4_a: one process per run seed, and the label-shuffle null is one more process of the same shape
    # (seed 0's seed, data and probes; only the train labels differ), so it is build #1 like every real run
    one = lambda k, Y: {"mode": "ranker", "spec": {"K": b["K"], "F": b["F"], "param_budget": b["param_budget"]},
                        "seeds": [seeds[k]], "train_X": [torch.tensor(d["trX"][k])], "train_Y": [torch.tensor(Y)],
                        "eval_X": [torch.tensor(d["evX"][k])], "eval_Xs": [torch.tensor(d["evXs"][k])],
                        "probe_X": torch.tensor(Z), "probe_perm_X": torch.tensor(Zp),
                        "steps": b["train_steps"], "batch": b["batch"], "lr": b["lr"], "clip": b["clip"], "device": b["device"]}
    outs = []
    for job in [one(k, d["trY"][k]) for k in range(len(seeds))] + [one(0, null_Y)]:
        out, before, after = run_sandbox(cdir, man, job)
        res["wall_s"] = round(res.get("wall_s", 0) + out.get("wall_s", 0), 1)
        # ---- V0 runtime (every process, the null's included: it is the candidate's own code)
        if before != after:
            return reject(res, "V0.sandbox", "engine cannot modify the verifier", "protected tree changed during the run")
        if out["violations"]:
            return reject(res, "V0.sandbox", "L-SEALED-TRUTH / sandbox", "sandbox violation", violations=out["violations"][:5])
        if out.get("error"):
            return reject(res, "V0", "runs", out["error"][:300])
        if list(out["probe_shape"].tolist()) != [len(Z), b["K"]]:
            return reject(res, "V0", "shapes", f"score shape {out['probe_shape'].tolist()} != [{len(Z)}, {b['K']}]")
        outs.append(out["runs"][0])
    runs, null_run, budget = outs[:-1], outs[-1], b["param_budget"]
    npar = [r["n_params"] for r in outs]
    if max(npar) > 1.02 * budget:
        return reject(res, "V0", "L-EQ", f"trainable params {npar} > {budget} * 1.02")
    nhid = max(r["n_hidden"] for r in outs)
    if nhid > 0.10 * budget:
        return reject(res, "V0", "L-EQ (hidden capacity)", f"{nhid} frozen params + buffer entries > 10% of {budget}")
    fl = ranker_floors(bedname, d, tag)
    res["floors"] = dict(fl)
    ia = [float(np.mean(r["init_pred"].numpy() == d["evY"][k])) for k, r in enumerate(runs)]
    res["init_hit"] = ia
    if max(ia) > max(fl.values()) + 0.05:
        return reject(res, "V0.init", "hard-coded outputs", f"untrained ranker hits {max(ia):.4f} > best floor {max(fl.values()):.4f} + 0.05")
    # ---- V1 claimed invariants, on every run seed's trained model: on the probes AND on the eval sets
    # (r2_c: probes alone are fingerprintable, r05), whose two presentations are the same sets permuted
    claims = man.get("claims", [])
    for k, r in enumerate(runs):
        S, Sp = r["probe_S"].numpy(), r["probe_perm_S"].numpy()
        Se, Ses = r["eval_S"].numpy(), r["eval_S_shuf"].numpy()
        if "permutation_equivariant" in claims:
            for where, (dev, tol) in (("probe", equivariance_dev(S, Sp, perms)),
                                      ("eval", equivariance_dev(Se, Ses[None], d["evP"][k][None]))):
                res["equivariance_dev"] = max(dev, res.get("equivariance_dev", 0.0))
                if dev > tol:
                    return reject(res, "V1", "permutation equivariance", f"{where}: max |f(Pz) - P f(z)| = {dev:.3g} > {tol:.3g}")
        if "bounded" in claims:
            for where, bd in (("probe", bound_dev(S, Z)), ("eval", max(bound_dev(Se, d["evX"][k]), bound_dev(Ses, d["evXs"][k])))):
                res["bound_dev"] = max(bd, res.get("bound_dev", 0.0))
                if bd > man["epsilon"] + 1e-5:
                    return reject(res, "V1", "bounded correction", f"{where}: max |score - base| = {bd:.4g} > epsilon {man['epsilon']}")
    # ---- V4 learnability (planner order, every run seed)
    hp = [float(np.mean(r["pred"].numpy() == d["evY"][k])) for k, r in enumerate(runs)]
    res["hit_planner"] = hp
    if min(hp) < b["v4_learnability"]:
        return reject(res, "V4", "L-LEARN", f"hit {['%.4f' % h for h in hp]} < {b['v4_learnability']} on >=1 of {len(hp)} verifier seeds")
    rep = man.get("reported", {}).get("hit")
    if rep is not None and abs(rep - float(np.mean(hp))) > b["v4_repro_tolerance_abs"]:
        return reject(res, "V4", "L-REPRO", f"reported {rep} vs measured {np.mean(hp):.4f}")
    # ---- V2 floors (secret-shuffle presentation)
    hs = _hit([r["pred_shuf"].numpy() for r in runs], d["evYs"])
    res["hit_shuffled"] = hs
    hb = _hit(d["bayes"], d["evY"])
    res["bayes_hit"] = hb
    line = ceiling_breach(hs, hb, sum(len(y) for y in d["evY"]))
    if line is not None:
        return reject(res, "V2.ceiling", "L-SEALED-TRUTH", f"hit {hs:.4f} above the Bayes ceiling {hb:.4f} (line {line:.4f}): only truth crosses it")
    if abs(float(np.mean(hp)) - hs) > b["v2_order_tol"]:
        return reject(res, "V2.order", "L-SHORTCUT (set position)",
                      f"hit moves {np.mean(hp):.4f} -> {hs:.4f} under a secret candidate shuffle: reads candidate index order")
    fl["label_shuffle_null"] = float(np.mean(null_run["pred"].numpy() == d["evY"][0]))
    res["floors"] = dict(fl)
    best = max(fl, key=fl.get)
    if hs < fl[best] + b["v2_margin_over_best_floor"]:
        return reject(res, "V2", "L-SHORTCUT", f"hit {hs:.4f} < best floor {best} {fl[best]:.4f} + {b['v2_margin_over_best_floor']}")
    if not v3:
        res.update(verdict="PASS_V2", stage="V2", law="", reason="pool read: V3 not run")
        return res
    # ---- V3 trained best responses (bars_r3 r3_d): the verifier-owned pool on the same draw
    ph = ranker_pool_hits(bedname, tag)
    best = max(ph, key=ph.get)
    margin = hs - ph[best]
    res["v3"] = {"pool": ph, "best": best, "margin": margin}
    if margin < R3["v3_margin"]:
        return reject(res, "V3", "L-BR", f"margin {margin:+.4f} over pool {best} {ph[best]:.4f} < {R3['v3_margin']}")
    res.update(verdict="PASS_V3", stage="V3", law="",
               reason="passed V0, V1, V4, V2, V3; no V5 ladder is registered for rankers, so never ACCEPT")
    return res


_RPOOL = {}


def ranker_pool_hits(bedname, tag):
    """Each registered pool member's pooled secret-shuffle hit on this draw; a broken member is InfraError."""
    out = {}
    for name, sha in R3["pool"].items():
        if (bedname, tag, name) not in _RPOOL:
            cdir = os.path.join(CONTROLS, name)
            if _sha(open(os.path.join(cdir, "candidate.py"), "rb").read()) != sha:
                raise InfraError(f"pool member {name} source differs from its registered sha256")
            man = json.load(open(os.path.join(cdir, "manifest.json")))
            r = verify_ranker(cdir, dict(man, bed=bedname), {"name": name}, tag, v3=False)
            if r.get("verdict") == "REJECT" and r["stage"] in RANKER_INTEGRITY or "hit_shuffled" not in r:
                raise InfraError(f"pool member {name} inadmissible: {r.get('stage')} {r.get('reason', '')[:120]}")
            _RPOOL[bedname, tag, name] = r["hit_shuffled"]
        out[name] = _RPOOL[bedname, tag, name]
    return out


# ------------------------------------------------------------------ bed proposals (the objective slot)
def verify_bed(cdir, man, res):
    line = BARS["beds"]["bed_validity"]["void_if_any_floor_or_shortcut_scores_at_least"]
    fam, prm = man.get("family"), man.get("params", {})
    if fam == "pointer_chase":
        import shortcuts
        v = shortcuts.void_check(prm["generator"], prm["n"], prm["L"], prm.get("instance", 0), prm["band"])
        res["floors"] = v["scores"]
        best, score = v["best"], v["score"]
    elif fam == "group_reset_x3":
        v = beds.x3_recency(**{k: prm[k] for k in ("n", "density") if k in prm})
        res["floors"] = {f"recency_{k}": a for k, a in v["recency_acc"].items()}
        best = max(res["floors"], key=res["floors"].get); score = res["floors"][best]
    elif fam == "custom":
        tag = "bed"
        job = {"mode": "bed", "seeds": [beds.secret_seed(tag + "|tr") % 100000, beds.secret_seed(tag + "|ev") % 100000],
               "n_train": 20000, "n_eval": 4000, "device": "cpu"}
        out, before, after = run_sandbox(cdir, man, job)
        if before != after:
            return reject(res, "V0.sandbox", "engine cannot modify the verifier", "protected tree changed")
        if out["violations"]:
            return reject(res, "V0.sandbox", "L-SEALED-TRUTH / sandbox", "sandbox violation", violations=out["violations"][:5])
        if out.get("error"):
            return reject(res, "V0", "runs", out["error"][:300])
        band = out["band"].numpy().astype(bool)
        if band.sum() == 0:
            return reject(res, "V2", "L-SHORTCUT", "empty scored band (a check that could not fail)")
        res["floors"] = beds.floors(out["Xtr"].numpy(), out["Ytr"].numpy(), out["Xev"].numpy(), out["Yev"].numpy(), band)
        best = max(res["floors"], key=res["floors"].get); score = res["floors"][best]
    else:
        return reject(res, "V0", "schema", f"bed family {fam!r}")
    if score >= line:
        return reject(res, "V2", "L-SHORTCUT", f"bed VOID: {best} scores {score:.4f} >= {line} on the scored band")
    res.update(verdict="ACCEPT", stage="V2", law="", reason=f"bed admissible: best floor/shortcut {best} {score:.4f} < {line}")
    return res


def verify(cdir, pool=None, tag="m0", bed=None):
    global _TRUTH
    if _TRUTH is None:
        _TRUTH = _truth_windows()
    res = {"dir": os.path.relpath(cdir, DAEDALUS), "name": os.path.basename(cdir)}
    t0 = time.time()
    man, rej = v0_static(cdir, res, bed)
    if rej is None:
        try:
            rej = (verify_operator(cdir, man, res, pool) if man["kind"] == "operator"
                   else verify_ranker(cdir, man, res, tag) if man["kind"] == "ranker" else verify_bed(cdir, man, res))
        except InfraError as e:  # not a verdict: counted neither caught nor passed
            rej = dict(res, verdict="ERROR", stage="infra", law="", reason=f"runner died 3x: {str(e)[:200]}")
    rej["wall_total_s"] = round(time.time() - t0, 1)
    rej["provenance"] = dict(PROVENANCE)
    return rej


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(verify(d), default=float))
