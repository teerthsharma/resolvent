"""Daedalus V2 shortcut library for the Phase K pointer-chase beds (bed_k, bed_k').

A bed is VOID when any BOUND hand-built construction scores >= 0.5 on the bed's scored band
(registry/bars.json: beds.bed_validity; known_verdicts.V5_phase_k_window).

Every construction below is the idealised pointer-level version. Where the record builds a construction as real
causal softmax heads (14-slot kp_w14.py, 30-slot kp_next.py, 62-slot kp_w62*.py), the real build matches the
idealised pointers exactly (0 mismatches, kp_w14.json f64/f32, kp_next.json w30, kp_w62.json w62_4096), so the
idealised version is used. All code is copied, not imported: kp_attack.py imports bed_kp from a dead scratchpad
path, and copying keeps this engine independent of the lane directories.

Predictions are root POSITIONS per token, chance-credited as the record scores them: an unresolved final pointer is
credited with the most recent root at or before it (bed_k.py:40-44, 58-61).
"""
import sys
import time
from functools import lru_cache

import numpy as np
from scipy.special import gammaln

sys.dont_write_bytecode = True

PK = "tests/foreman/phase_k/"
V = 32768            # id vocabulary (K0/cameron/bed_k.py:12)
NULL = V             # parent id of a root (K0/cameron/bed_k.py:13)
LANES = {"bed_k": 16, "bed_kp": 8}   # test lanes: bed_k (bed_k.py:112 make_test M=16), bed_k' (K1/cameron/bed_kp.py:19)
# c_max(L): largest reach factor of any bound real-heads construction (RECORD_K.md:180-190; FOREMAN_REPORT K1):
# L=4: 62-slot, 63 heads x 2 dims, exact at 4096/8192/16384 -> 4.25; L=7: 30-slot, exact at 16384 -> 3.5.
C_MAX = {4: 4.25, 7: 3.5}
VOID_LINE = 0.5      # bars.json beds.bed_validity.void_if_any_floor_or_shortcut_scores_at_least
MIN_BAND_TOKENS = 500  # test_k1fpp.py fpp_band_size / test_kp_next.py: a band with fewer tokens reads nothing


# ---------------------------------------------------------------- beds (exact truth)
def make_bed(n, M, rng, cap=None):
    """Copied from K0/cameron/bed_k.py:17-25 (identical to wald.py:make_bed with its own Generator)."""
    chain = rng.integers(0, M, n)
    parent = -np.ones(n, int); depth = np.zeros(n, int); root = np.arange(n); last = {}
    for i in range(n):
        c = chain[i]
        if c in last and (cap is None or depth[last[c]] < cap):
            p = last[c]; parent[i], depth[i], root[i] = p, depth[p] + 1, root[p]
        last[c] = i
    return parent, depth, root


def _pack(parent, depth, root, rng):
    """Copied from K0/cameron/bed_k.py:100-103: n distinct ids from V, root parent id NULL, target = root's id."""
    n = len(parent); ids = rng.permutation(V)[:n]
    return {"ids": ids, "pids": np.where(parent < 0, NULL, ids[np.maximum(parent, 0)]), "parent": parent,
            "depth": depth, "root": root, "target": ids[root]}


def _bed(name, n, seed, k):
    # One Generator per bed, drawn in order: lanes then ids, exactly as make_test(default_rng([seed, n, k]), n).
    rng = np.random.default_rng([seed, n, k])
    return _pack(*make_bed(n, LANES[name], rng), rng)


def bed_k(n, L, seed, k=0):
    """bed_k test bed: K0/cameron/bed_k.py:110-111 make_test (M = 16), rng = default_rng([seed, n, k])."""
    return _bed("bed_k", n, seed, k)


def bed_kp(n, L, seed, k=0):
    """bed_k' test bed: K1/cameron/bed_kp.py:24-25 make_test (M = 8, no cap), rng = default_rng([seed, n, k])."""
    return _bed("bed_kp", n, seed, k)


BEDS = {"bed_k": bed_k, "bed_kp": bed_kp}


def band_mask(depth, L, band):
    """'beyond': depth > 2^L (RECORD_K.md:113-116, clause 2).
    'far': depth > 2 * c_max(L) * 2^L (Amendment K1-a clause 1', RECORD_K.md:131-136; c_max as C_MAX)."""
    if band == "beyond":
        return depth > 2 ** L
    if band == "far":
        return depth > 2 * C_MAX[L] * 2 ** L
    raise ValueError(f"band must be 'beyond' or 'far', got {band!r}")


def _last_root(parent):
    """Copied from K0/cameron/bed_k.py:40-42: most recent root at or before each position (the chance credit)."""
    n = len(parent)
    return np.maximum.accumulate(np.where(parent < 0, np.arange(n), 0))


# ---------------------------------------------------------------- constructions (pointer level)
def pointer0(parent):
    """Copied from K0/foreman/shortcut.py:40-43: a root points to itself."""
    A = parent.copy()
    A[A < 0] = np.flatnonzero(parent < 0)
    return A


def doubling(parent, L):
    """Copied from K0/foreman/shortcut.py:46-50: one content head per layer, A <- A[A]; reaches 2^L hops."""
    A = pointer0(parent)
    for _ in range(L):
        A = A[A]
    return A


def hybrid(parent, depth, L, W, k, mode, m):
    """C2 hybrid, copied from K0/foreman/shortcut.py:53-75 (m = lane count = mean positional gap per hop)."""
    A = pointer0(parent)
    for _ in range(L):
        v1 = A
        cur = A[v1]
        if mode == "anchored":
            lo, hi = v1 - W, v1
        else:
            c = v1 - m * (depth - depth[v1])
            lo, hi = np.maximum(c - W // 2, 0), np.minimum(c + W // 2, v1)
        live = np.ones(len(A), bool)
        for _ in range(k - 2):
            inw = live & (((cur >= lo) & (cur < hi)) | (cur == v1))
            cur = np.where(inw, A[cur], cur)
            live = inw
        A = cur
    return A


@lru_cache(maxsize=None)
def hpd_start(J, W, M):
    """Copied from K1/foreman/k1b.py:17-36 (pmf + hpd_start), P = 1/M as in K1/cameron/k1fp.py:19-38.
    Only the start is cached, not the pmf (the pmf cache costs ~400 MB at depth 2000)."""
    if J == 0:
        return 1
    P = 1 / M
    smax = int(M * J + 60 * np.sqrt(J) * M ** 0.5 + 400)
    s = np.arange(smax + 1)
    f = np.zeros(smax + 1)
    v = s >= J
    sv = s[v]
    f[v] = np.exp(gammaln(sv) - gammaln(J) - gammaln(sv - J + 1) + J * np.log(P) + (sv - J) * np.log1p(-P))
    c = np.concatenate([[0.0], np.cumsum(f)])
    mass = c[W:] - c[:-W]
    mass[0] = -1.0
    return int(np.argmax(mass))


def hpd_hybrid(parent, depth, L, W, k, M):
    """HPD window, copied from K1/foreman/k1b.py:39-53 (Cameron's hpd_analytic port, k1fp.py:41-55)."""
    A = pointer0(parent)
    for _ in range(L):
        v1 = A
        cur = A[v1]
        J = depth - depth[v1]
        a = np.array([hpd_start(int(j), W, M) for j in range(J.max() + 1)])[J]
        lo, hi = v1 - (a + W - 1), v1 - a + 1
        live = np.ones(len(A), bool)
        for _ in range(k - 2):
            inw = live & (((cur >= lo) & (cur < hi)) | (cur == v1))
            cur = np.where(inw, A[cur], cur)
            live = inw
        A = cur
    return A


def _hpd_starts(J, off, W):
    """Copied from K1/foreman/kp_attack.py:19-36."""
    tab = {}
    if J.size == 0:
        return tab
    o = np.argsort(J, kind="stable")
    J, off = J[o], off[o]
    cut = np.flatnonzero(np.diff(J)) + 1
    for j, grp in zip(J[np.r_[0, cut]], np.split(off, cut)):
        h = np.bincount(grp)
        if h.size <= W + 1:
            tab[int(j)] = 1
            continue
        c = np.concatenate([[0], np.cumsum(h)])
        mass = c[W:] - c[:-W]
        mass[0] = -1
        tab[int(j)] = int(np.argmax(mass))
    return tab


def _sc_step(A, depth, W, t2):
    """kp_attack.py:39-56 (window + step) with W2 = 0: v2 = A(v1) by content, v3 = A(v2) if v2 is in the window."""
    v1 = A
    v2 = A[v1]
    J = depth - depth[v1]
    a = np.array([t2.get(int(j), 1) for j in range(J.max() + 1)])[J]
    lo, hi = v1 - (a + W - 1), v1 - a + 1
    in1 = ((v2 >= lo) & (v2 < hi)) | (v2 == v1)
    return np.where(in1, A[v2], v2)


_CAL = {}


def _calibrate(cal, key, L, W):
    """kp_attack.py:59-77 with W2 = 0: per-layer HPD tables measured on calibration beds for this construction."""
    if (key, L, W) in _CAL:
        return _CAL[(key, L, W)]
    As = [pointer0(b["parent"]) for b in cal]
    tables = []
    for _ in range(L):
        j2, f2 = [], []
        for A, b in zip(As, cal):
            d = b["depth"]
            v1 = A
            m = A[v1] != v1
            j2.append((d - d[v1])[m]); f2.append((v1 - A[v1])[m])
        t2 = _hpd_starts(np.concatenate(j2), np.concatenate(f2), W)
        tables.append(t2)
        As = [_sc_step(A, b["depth"], W, t2) for A, b in zip(As, cal)]
    _CAL[(key, L, W)] = tables
    return tables


def sc_hpd(inst, W):
    tables = _calibrate(inst["cal"], inst["cal_key"], inst["L"], W)
    A = pointer0(inst["parent"])
    for t2 in tables:
        A = _sc_step(A, inst["depth"], W, t2)
    return A


def local_window(parent, L, W, k):
    """No-oracle ALiBi-local window: hunt.py:19-42 hybrid_space(key = positions, anchor_self=True), window [t-W, t)."""
    A = pointer0(parent)
    idx = np.arange(len(parent))
    for _ in range(L):
        v1 = A
        cur = A[v1]
        lo, hi = idx - W, idx
        live = np.ones(len(A), bool)
        for _ in range(k - 2):
            inw = live & (((cur >= lo) & (cur < hi)) | (cur == v1))
            cur = np.where(inw, A[cur], cur)
            live = inw
        A = cur
    return A


def bundle_stale(parent, L, W, k):
    """No-oracle stale bundle, copied from K1/foreman/hunt.py:45-59."""
    A = pointer0(parent)
    prev = None
    for _ in range(L):
        v1 = A
        cur = A[v1]
        if prev is not None:
            live = np.ones(len(A), bool)
            for _ in range(k - 2):
                inw = live & (cur >= v1 - W) & (cur < v1)
                cur = np.where(inw, prev[cur], cur)
                live = inw
        prev, A = A, cur
    return A


def _credit(inst, A):
    return _last_root(inst["parent"])[A]


# ---------------------------------------------------------------- the library
# fn(inst) -> predicted root position per token. bound(L, n) says where the record binds the construction; outside it
# the score is reported as info and never enters the verdict.
LIBRARY = [
    {"name": "position_only_floor", "fn": lambda i: _last_root(i["parent"]),
     "source": PK + "K0/cameron/bed_k.py:45-48 (floor_recency; wald.py:52-58)",
     "bound_in": "RECORD_K.md:111-113 clause 2 (K1.F), 138-140 clause 2' (K1.F''); K1/cameron/k1fpp.json 'recency'"},
    {"name": "doubling_credited", "fn": lambda i: _credit(i, doubling(i["parent"], i["L"])),
     "source": PK + "K0/foreman/shortcut.py:46-50; credit K0/cameron/bed_k.py:58-61",
     "bound_in": "RECORD_K.md:22-24 (0.2988 credited L=6 doubling, Cameron)"},
    {"name": "c2_hybrid_anchored_k3W10",
     "fn": lambda i: _credit(i, hybrid(i["parent"], i["depth"], i["L"], 10, 3, "anchored", i["M"])),
     "source": PK + "K0/foreman/shortcut.py:53-75",
     "bound_in": "RECORD_K.md:18-22 (C2 hybrid 0.3982, real heads 0 mismatches); K1/inspector/PASS1.md:31"},
    {"name": "c2_hybrid_centered_k3W10",
     "fn": lambda i: _credit(i, hybrid(i["parent"], i["depth"], i["L"], 10, 3, "centered", i["M"])),
     "source": PK + "K0/foreman/shortcut.py:53-75",
     "bound_in": "K1/inspector/PASS1.md:31 (K1.F cell 0.4085876, bed_k L7 n4096); RECORD_K.md:124"},
    {"name": "hpd_window_k3W10",
     "fn": lambda i: _credit(i, hpd_hybrid(i["parent"], i["depth"], i["L"], 10, 3, i["M"])),
     "source": PK + "K1/foreman/k1b.py:17-53 (P = 1/M as K1/cameron/k1fp.py:19-55)",
     "bound_in": "RECORD_K.md:122-124 (0.5642 on bed_k); K1/inspector/PASS1.md:38-40 (hpd_beats_line, hpd_reference)"},
    {"name": "schpd_10slot", "fn": lambda i: _credit(i, sc_hpd(i, 10)),
     "source": PK + "K1/foreman/kp_attack.py:19-88",
     "bound_in": "K1/inspector/PASS2.md:62-63 (0.4398 on bed_k' L7 n4096)"},
    {"name": "schpd_14slot", "fn": lambda i: _credit(i, sc_hpd(i, 14)),
     "source": PK + "K1/foreman/kp_attack.py:19-88; real heads K1/foreman/kp_w14.py:30-52",
     "bound_in": "RECORD_K.md:125-128 (0.5108, 15 heads x 8 dims, 0 mismatches f64/f32)"},
    {"name": "schpd_30slot", "fn": lambda i: _credit(i, sc_hpd(i, 30)),
     "source": PK + "K1/foreman/kp_attack.py:19-88; real heads K1/foreman/kp_next.py:26-67",
     "bound_in": "RECORD_K.md:128-129 (0.8096, 31 heads x 4 dims), 188-190 (exact at 16384)"},
    {"name": "schpd_62slot", "fn": lambda i: _credit(i, sc_hpd(i, 62)),
     "bound": lambda L, n: n <= 4096 or L == 4,
     "source": PK + "K1/foreman/kp_attack.py:19-88; real heads K1/foreman/kp_w62.py:18-42",
     "bound_in": "RECORD_K.md:184-190 and K1/foreman/FOREMAN_REPORT.md (w62_real_4096 L4+L7; L4 exact at 8192/16384; "
                 "L7 not realized past 4096: w62_8192 rate 2.3e-4, w62a/w62p4096/w62t RED)"},
    {"name": "local_noorc_k3W30", "fn": lambda i: _credit(i, local_window(i["parent"], i["L"], 30, 3)),
     "source": PK + "K1/foreman/hunt.py:19-42 (anchor_self=True)",
     "bound_in": "K1/foreman/kp_next.py:82-83, test_kp_next.py next_far_band (local_W30); K1/inspector/PASS3.md:67"},
    {"name": "stale_noorc_k3W10", "fn": lambda i: _credit(i, bundle_stale(i["parent"], i["L"], 10, 3)),
     "source": PK + "K1/foreman/hunt.py:45-59",
     "bound_in": "K1/inspector/PASS1.md:34 (no_oracle_ceiling at R0)"},
    {"name": "local_noorc_k3W64", "fn": lambda i: _credit(i, local_window(i["parent"], i["L"], 64, 3)),
     "bound": lambda L, n: False,
     "source": PK + "K1/foreman/hunt.py:19-42 (anchor_self=True)",
     "bound_in": "number bound (0.6393 on bed_k, K1/inspector/PASS1.md:41) but unpriced: 64 slots exceed the "
                 "62-slot maximum at d = 128, so info only"},
]

SKIPPED = {
    "bedk_best_response (Foreman K0, 0.3181)": "struck: declared RED and never run (RECORD_K.md:69-72)",
    "Cameron beyond-2^L best-response band on bed_k (0.041-0.063)": "struck: no bar (RECORD_K.md:69-72)",
    "id-space K0 mechanism (window over ids)": "bound as doubling + 0.0013 (K1/inspector/PASS1.md:32): duplicate of doubling",
    "k4 and split (W1, 10-W1) window configs": "bound but dominated by k3 W10 in every recorded sweep (k1b.json, kp_attack.json)",
    "W12/W13 curve values, far_band2_defeats_all, local reach fields": "struck (K1/inspector/PASS2.md:54, 59, 66)",
    "trained opponent aL4 (131M tokens)": "void at its own gate 0.818 < 0.9 (RECORD_K.md:193-194)",
    "resolvent hand-set head": "the solution, not a shortcut (k1fpp.py resolvent_ok)",
}


def instances(bed_name, n, L, seed, n_beds=8):
    """n_beds test beds default_rng([seed, n, k]) and calibration beds default_rng([seed + 1, n, k]) (the record's
    pairing: [12] bed_k K1.F, [22]/[23] bed_k' kp_attack, [31]/[32] K1.F'')."""
    key = (bed_name, n, seed)
    cal = [_bed(bed_name, n, seed + 1, k) for k in range(8)]
    out = []
    for k in range(n_beds):
        b = _bed(bed_name, n, seed, k)
        b.update(L=L, M=LANES[bed_name], cal=cal, cal_key=key, n=n)
        out.append(b)
    return out


def void_check(bed_name, n, L, seed, band, n_beds=8):
    """Score every library entry on the band (mean over beds of per-bed band accuracy, as the record scores it).
    VOID iff a bound entry scores >= 0.5. Raises if the band holds < 500 pooled tokens (a check that could not fail)."""
    insts = instances(bed_name, n, L, seed, n_beds)
    masks = [band_mask(b["depth"], L, band) for b in insts]
    tokens = int(sum(m.sum() for m in masks))
    if tokens < MIN_BAND_TOKENS or not all(m.any() for m in masks):
        raise ValueError(f"band {band!r} at (n={n}, L={L}) holds {tokens} tokens: nothing to score")
    scores, info, wall = {}, {}, {}
    for e in LIBRARY:
        t0 = time.time()
        s = float(np.mean([(e["fn"](b) == b["root"])[m].mean() for b, m in zip(insts, masks)]))
        wall[e["name"]] = round(time.time() - t0, 2)
        (scores if e.get("bound", lambda L, n: True)(L, n) else info)[e["name"]] = s
    best = max(scores, key=scores.get)
    return {"void": scores[best] >= VOID_LINE, "best": best, "score": scores[best], "scores": scores,
            "info_unbound": info, "band_tokens": tokens, "band_threshold": float(2 ** L if band == "beyond" else 2 * C_MAX[L] * 2 ** L),
            "n_beds": n_beds, "wall_s": wall}


if __name__ == "__main__":
    import json
    import os
    import platform
    from datetime import datetime, timezone
    here = os.path.dirname(os.path.abspath(__file__))
    runs = [("near_bed_k", "bed_k", 4096, 7, 12, "beyond", {"hpd_window_k3W10": 0.5642224409448819}),
            ("near_bed_kp", "bed_kp", 4096, 7, 22, "beyond", {"schpd_14slot": 0.5107702349869452,
                                                                "schpd_10slot": 0.43982539164490864,
                                                                "schpd_30slot": 0.8095626631853785}),
            ("far_bed_kp_control", "bed_kp", 16384, 7, 31, "far", {"line (every family)": 0.1230723284100782})]
    out = {"command": "python daedalus/engine/shortcuts.py", "utc": datetime.now(timezone.utc).isoformat(),
           "machine": platform.processor() or platform.machine(), "numpy": np.__version__, "device": "CPU only",
           "void_line": VOID_LINE, "c_max": C_MAX,
           "library": [{k: v for k, v in e.items() if k in ("name", "source", "bound_in")} for e in LIBRARY],
           "skipped": SKIPPED, "runs": {}}
    T0 = time.time()
    for tag, bed, n, L, seed, band, stored in runs:
        t0 = time.time()
        r = void_check(bed, n, L, seed, band)
        r.update(bed=bed, n=n, L=L, band=band, seeds={"test": [[seed, n, k] for k in range(8)],
                                                      "calibration": [[seed + 1, n, k] for k in range(8)]},
                 stored={k: {"stored": v, "rerun": r["scores"].get(k, max(r["scores"].values())),
                             "abs_diff": abs(r["scores"].get(k, max(r["scores"].values())) - v)} for k, v in stored.items()},
                 wall_clock_s=round(time.time() - t0, 1))
        out["runs"][tag] = r
        print(tag, "VOID" if r["void"] else "not void", r["best"], round(r["score"], 4),
              {k: round(v, 4) for k, v in r["scores"].items()}, r["info_unbound"], f"{r['wall_clock_s']}s", flush=True)
    out["wall_clock_total_s"] = round(time.time() - T0, 1)
    out["expectation_V5"] = {"near_bed_k void": out["runs"]["near_bed_k"]["void"],
                             "near_bed_kp void": out["runs"]["near_bed_kp"]["void"],
                             "far_bed_kp not void": not out["runs"]["far_bed_kp_control"]["void"]}
    out["V5_reproduced"] = all(out["expectation_V5"].values())
    path = os.path.join(here, "..", "results", "shortcuts_phase_k.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("V5 reproduced:", out["V5_reproduced"], "total", out["wall_clock_total_s"], "s ->", os.path.normpath(path))
