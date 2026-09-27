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

BARS = json.load(open(os.path.join(DAEDALUS, "registry", "bars.json")))
ADD = BARS["addendum_1"]
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
def v0_static(cdir, res):
    mpath = os.path.join(cdir, "manifest.json")
    try:
        man = json.load(open(mpath))
    except Exception as e:
        return None, reject(res, "V0", "L-REPRO", f"manifest unreadable: {e}")
    res["name"] = man.get("name", os.path.basename(cdir))
    if man.get("kind") not in ("operator", "bed"):
        return None, reject(res, "V0", "schema", f"kind {man.get('kind')!r}")
    bad = sorted({p for p, k in _walk_keys(man) if str(k).lower() in FORBIDDEN_KEYS})
    if bad:
        return None, reject(res, "V0", "L-FIXED-BAR / L-BR", "manifest sets verifier-owned fields", keys=bad)
    fetched = _fetched_owner_ids()
    unfetched = [c for c in man.get("citations", []) if c not in fetched]
    if unfetched:
        return None, reject(res, "V0", "fetch-before-cite", "citation not fetched this run", citations=unfetched)
    for claim in man.get("claims", []):
        if claim not in KNOWN_CLAIMS:
            return None, reject(res, "V1", "L-AUDIT (claim without instrument)", f"claim {claim!r} has no V1 spec")
    if man["kind"] == "operator" and man.get("bed") not in beds.BEDS:
        return None, reject(res, "V0", "schema", f"bed {man.get('bed')!r} not registered")
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
        if not out.get("infra") or out["violations"]:
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
    return [beds.secret_seed(f"{tag}|seed{i}") % 100000 for i in range(k)]


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
                raise RuntimeError(f"pool member {name} failed: {out['error']}")
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


def verify(cdir, pool=None):
    global _TRUTH
    if _TRUTH is None:
        _TRUTH = _truth_windows()
    res = {"dir": os.path.relpath(cdir, DAEDALUS), "name": os.path.basename(cdir)}
    t0 = time.time()
    man, rej = v0_static(cdir, res)
    if rej is None:
        try:
            rej = verify_operator(cdir, man, res, pool) if man["kind"] == "operator" else verify_bed(cdir, man, res)
        except InfraError as e:  # not a verdict: counted neither caught nor passed
            rej = dict(res, verdict="ERROR", stage="infra", law="", reason=f"runner died 3x: {str(e)[:200]}")
    rej["wall_total_s"] = round(time.time() - t0, 1)
    return rej


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(verify(d), default=float))
