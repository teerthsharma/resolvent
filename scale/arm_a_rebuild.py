"""ARM A, REBUILT. The old theorem was about DIFFERENCES; the object is SUMS.

WHY THE OLD ARM DIED. A cyclic Singer (v,k,1)-difference set has every nonzero
residue occurring exactly once as `d_a - d_b`. That gate was GREEN and it was
IRRELEVANT. A causal two-hop path i -> p -> j composes

    (i - p) + (p - j) = i - j

with BOTH hops pointing the same way, so the reachable offset set at two hops is
the SUMSET `D + D`, never the difference set `D - D`. Measured on the same set:
|D-D| = 56/56 exactly, |D+D| = 36/57. The birth gate was true of an object that
was not the one built.

WHAT REPLACES IT. `D` must be an additive basis of order 2 for the causal offset
interval: with

    R(D) = D  u  {a + b : a, b in D}      (a = b allowed: p = i-d, j = p-d)

the requirement is `R(D) >= [1, s-1]`. A causal mask HAS NO WRAPAROUND, so the
mod-v version of the question is not the one that binds; the raw interval is.
Sidon sets are the exactly wrong object here -- they MINIMISE sum collisions and
this arm needs them maximised, see SEVERANCE.

SEVERANCE HAS A CLOSED FORM, asserted here and then checked bitwise against
autograd. With M[i,j] = a[i,j] + sum_p a[i,p] a[p,j], token c influences M[i,j]
iff

    P(j)  = {p : i-p in D and p-j in D}                     (the intermediates)
    reach = (i-j in D) or P(j) nonempty
    infl  = reach and ( [ i-c in D and supp(i) >= 2 ]
                        or  exists p in P(j): (p == c or p-c in D)
                                              and supp(p) >= 2 )

`supp(p) >= 2` is not decoration: a row with ONE unmasked key softmaxes to
exactly 1.0 and its normalisation cannot move. So severance is a SET FUNCTION of
D, and MULTIPLICITY of representations -- |P(j)| -- is what buys it down. That is
the opposite of a Sidon set and it is why a MINIMAL basis is the wrong target:
minimal means multiplicity one means the influencing set stays small.

TWO CORRECTIONS THE PREVIOUS BUILD LACKED, both mandatory:
  * `j` (and `c`) are drawn UNIFORMLY AT RANDOM. Fixing j = s//4 pins the offset
    at 3s/4, unreachable at two hops for any schedule with max offset < 3s/8, so
    EVERY schedule including contiguous reads severance exactly 1.0000. That is
    the discipline `scale/carpet_probe.py:24` demands.
  * The flip rate is read on the X4 valuation instrument, `scale/valuation.py`.
    The float path forms `lo * hi`, and that product flushes to -0.0 while both
    factors are healthy -- a real flip counted as no flip.

NO NAME IS COINED HERE. G1 fired on co-prime spacing (arXiv 2606.28560 compares
a "coprime (anti-gridding) reassignment"; the idea originates in dilated CNNs,
arXiv 1702.08502). Additive bases were NOT searched for in that G1 pass. The
fetch is OWED, not passed.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scale.difference_set_arm import (SINGER, draws, flip_rate_x4,  # noqa: E402
                                      offset_mask, op_masked)
from scale.valuation import float_flip_rate, valuation  # noqa: E402


# --------------------------------------------------------------- (a) the set
def reach2(D, n: int) -> set[int]:
    """R(D) intersect [1, n]. a = b IS allowed: p = i-d, j = p-d is a real path."""
    P = sorted({d for d in D if d > 0})
    r = {d for d in P if d <= n}
    for x in range(len(P)):
        for y in range(x, len(P)):
            t = P[x] + P[y]
            if t <= n:
                r.add(t)
    return r


def cover_value(D, n: int) -> int:
    """Coverage as a VALUE, the way |D-D| = v-1 was asserted."""
    return len(reach2(D, n))


def diff_cover_value(D, v: int) -> int:
    return len({(a - b) % v for a in D for b in D if (a - b) % v != 0})


def search_raw(n: int, k: int, *, stop_on_first: bool = True, node_cap: int = 0):
    """DFS for k distinct positive ints with R(D) covering [1, n].

    PRUNE, and it is what makes this tractable: elements ascend, so once d_i is
    fixed every value <= d_i is unreachable by any later sum (every later element
    already exceeds d_i). Hence the next element obeys d <= c+1 where [1,c] is
    the covered prefix. Bitmask ints carry `cov`; `S << d` is every new sum at
    once.

    Returns (witness_or_None, nodes_visited, hit_node_cap).
    """
    found: list = [None]
    nodes = [0]
    capped = [False]

    def rec(elems, S, cov, rem):
        if capped[0] or (found[0] is not None and stop_on_first):
            return
        if node_cap and nodes[0] >= node_cap:
            capped[0] = True
            return
        nodes[0] += 1
        c = 0
        while (cov >> (c + 1)) & 1:
            c += 1
        if c >= n:
            found[0] = list(elems)
            return
        if rem == 0:
            return
        i0 = len(elems)
        # COUNTING BOUND. The t-th further element adds at most (i0+t+1) newly
        # representable values: itself, its double, and one sum with each element
        # already present. Compare against what is STILL MISSING above c -- NOT
        # against (n - c). Values above c may already be covered, and assuming
        # they are not is a prune that deletes real witnesses: it lost
        # D = [1,3,5,6] for n = 12 and reported k = 5 where k = 4 exists.
        missing = sum(1 for x in range(c + 1, n + 1) if not (cov >> x) & 1)
        if missing > sum(i0 + t + 1 for t in range(1, rem + 1)):
            return
        lo = elems[-1] + 1 if elems else 1
        for d in range(lo, min(c + 1, n) + 1):
            S2 = S | (1 << d)
            elems.append(d)
            rec(elems, S2, cov | (1 << d) | (S2 << d), rem - 1)
            elems.pop()
            if capped[0] or (found[0] is not None and stop_on_first):
                return

    rec([], 0, 0, k)
    return found[0], nodes[0], capped[0]


def counting_bound(n: int) -> int:
    """k singles plus k(k+1)/2 sums are at most k(k+3)/2 distinct values, so no
    k below this can cover [1,n]. A PROOF, not a heuristic -- it is what lets the
    search start above 1 without leaving a gap."""
    k = 1
    while k * (k + 3) // 2 < n:
        k += 1
    return k


def min_k_raw(n: int, kmax: int = 26, node_cap: int = 0):
    """Smallest k with a witness. Returns (k, witness, log, k0).

    `log` records EVERY k tried and how it ended: WITNESS, REFUTED (the search
    ran to completion and found nothing), or CAPPED (it hit `node_cap` first, so
    nothing is refuted and the minimum is NOT established). Reporting a minimum
    off a capped search would be asserting an inequality the run never proved.
    """
    k0 = counting_bound(n)
    log = []
    for k in range(k0, kmax + 1):
        w, nodes, cap = search_raw(n, k, stop_on_first=True, node_cap=node_cap)
        if w is not None:
            log.append((k, "WITNESS", nodes))
            return k, w, log, k0
        log.append((k, "CAPPED" if cap else "REFUTED", nodes))
    return None, None, log, k0


def search_mod(v: int, k: int, *, restarts: int = 200, iters: int = 3000, seed: int = 0):
    """Mod-v coverage: R(D) mod v covering Z_v minus {0}. Hill climb, NOT
    exhaustive. Reported only to answer 'is wraparound doing the work'. A causal
    mask cannot wrap, so this number is descriptive and never the requirement."""
    rng = random.Random(seed)

    def cov(D):
        r = {d % v for d in D}
        for x in range(len(D)):
            for y in range(x, len(D)):
                r.add((D[x] + D[y]) % v)
        r.discard(0)
        return len(r)

    best, bestD = -1, None
    pool = list(range(v))
    for _ in range(restarts):
        D = rng.sample(pool, k)
        cur = cov(D)
        for _ in range(iters):
            a = rng.randrange(k)
            old, new = D[a], rng.choice(pool)
            if new in D:
                continue
            D[a] = new
            nc = cov(D)
            if nc >= cur:
                cur = nc
            else:
                D[a] = old
            if cur == v - 1:
                break
        if cur > best:
            best, bestD = cur, sorted(D)
        if best == v - 1:
            break
    return best, bestD


# ---------------------------------------------- severance, in closed form
def _supp_sizes(s: int, D) -> list[int]:
    Ds = sorted({d for d in D if d > 0})
    return [sum(1 for d in Ds if p - d >= 0) for p in range(s)]


def influencing_set(s, Dset, supp, j):
    """The exact set of c in (j, i) that move M[i,j]. None means UNREACHABLE
    (i-j not in R(D)) -- then every c is severed by construction."""
    i = s - 1
    o = i - j
    P = [i - d for d in Dset if (o - d) in Dset and j < i - d < i]
    if o not in Dset and not P:
        return None
    I = set()
    if supp[i] >= 2:
        I |= {i - d for d in Dset if j < i - d < i}
    for p in P:
        if supp[p] >= 2:
            I.add(p)
            I |= {p - d for d in Dset if j < p - d < i}
    return I


def severance_exact(s: int, D):
    """EXACT severance under the sampling discipline actually used: j uniform on
    [0, i-2], then c uniform on (j, i). Returns (severed, unreachable, mean|P|)."""
    i = s - 1
    Dset = {d for d in D if d > 0}
    supp = _supp_sizes(s, D)
    tot = un = 0.0
    reps = 0.0
    n = 0
    for j in range(0, i - 1):
        m = i - j - 1
        if m <= 0:
            continue
        o = i - j
        reps += sum(1 for d in Dset if (o - d) in Dset)
        I = influencing_set(s, Dset, supp, j)
        tot += 1.0 if I is None else (m - len(I & set(range(j + 1, i)))) / m
        un += I is None
        n += 1
    return tot / n, un / n, reps / n


# ------------------------------------------------------------- (b) the arm
def random_pairs(s: int, n: int, seed: int = 0):
    """j and c UNIFORMLY AT RANDOM, j < c < i = s-1. Never placed by hand."""
    rng = random.Random(seed)
    i = s - 1
    return [(j, rng.randrange(j + 1, i))
            for j in (rng.randrange(0, i - 1) for _ in range(n))]


def severance_empirical(D, *, s, pairs, n_draws=4, seed=0):
    """Bitwise-identical gradient pair = severed. The (j,c) draws are SHARED
    across schedules so the comparison is paired."""
    i = s - 1
    flags = []
    for j, c in pairs:
        # lam=0.10 IS THE SHIPPED VALUE (`ceq/bench.py:192`). The inherited
        # `op_masked` defaults to 1.00, a DIFFERENT operator: at lam=1 a
        # single-key row gives pp = pm = 1 and the entry collapses to exactly
        # 0. Measuring an operator that ships nowhere is instrument #17.
        dr = draws(D, s=s, i=i, j=j, c=c, n_draws=n_draws, seed=seed, lam=0.10)
        flags.append(all(lo == hi for lo, hi in dr))
    return sum(flags) / len(flags), flags


def closed_form_flags(s, D, pairs):
    Dset = {d for d in D if d > 0}
    supp = _supp_sizes(s, D)
    out = []
    for j, c in pairs:
        I = influencing_set(s, Dset, supp, j)
        out.append(I is None or c not in I)
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (ctr - half, ctr + half)


# ------------------------------------------------------------------ families
def coprime(k):                       # odd offsets: the schedule G1 fired on
    return [2 * t + 1 for t in range(k)]


def powers_of_two(k, s):
    return [1 << t for t in range(k) if (1 << t) <= s - 1]


def contiguous(k):
    return list(range(1, k + 1))


def two_scale(m, n):
    """[1..m] plus every multiple of m. Covers [1,n] by construction: any
    o = q*m + r with 1 <= r <= m has q*m in D and r in D."""
    return sorted(set(range(1, m + 1)) | {m * t for t in range(2, n // m + 2)
                                          if m * t <= n})


def redundant(m, g, n):
    """[1..m] plus every multiple of g < m: overlapping representations, which
    is the multiplicity severance needs."""
    return sorted(set(range(1, m + 1)) | {g * t for t in range(1, n // g + 1)
                                          if g * t <= n})


def family_at_k(s: int, k: int):
    """Lowest-severance COVERING schedule of EXACTLY size k, enumerated over the
    two-parameter family [1..m] u {multiples of g}. Enumerated, not tuned, so a
    matched-k row is reproducible from (m, g) alone."""
    n = s - 1
    best = (2.0, None)
    for m in range(1, k + 1):
        for g in range(1, m + 1):
            D = redundant(m, g, n)
            if len(D) != k or cover_value(D, n) != n:
                continue
            v = severance_exact(s, D)[0]
            if v < best[0]:
                best = (v, D)
    return best[1]


def hillclimb(s, k, *, restarts=6, iters=300, seed=0, start=None):
    """Minimise EXACT severance subject to FULL coverage of [1, s-1]."""
    n = s - 1
    rng = random.Random(seed)
    best = (2.0, None)
    seeds = [start] if start else []
    while len(seeds) < restarts:
        seeds.append(None)
    for st in seeds:
        if st and len(st) == k:
            D = sorted(st)
        else:
            D = sorted(set(contiguous(max(2, k // 3))))
            while len(D) < k:
                x = rng.randrange(1, n + 1)
                if x not in D:
                    D.append(x)
            D.sort()
        cur = severance_exact(s, D)[0] if cover_value(D, n) == n else 2.0
        for _ in range(iters):
            a = rng.randrange(k)
            old, new = D[a], rng.randrange(1, n + 1)
            if new in D:
                continue
            D[a] = new
            D.sort()
            val = severance_exact(s, D)[0] if cover_value(D, n) == n else 2.0
            if val <= cur:
                cur = val
            else:
                D[D.index(new)] = old
                D.sort()
        if cur < best[0]:
            best = (cur, sorted(D))
    return best


# -------------------------------------------- the 0.1277 baseline, in its own
# harness. THE BASELINE IS NOT COMMENSURABLE WITH THE ONE ABOVE and quoting it
# across harnesses would be the confound this project already published once.
# 0.1277 came from `tests/cameron/schedule_sweep.py`: s=128, i=127, **j = s//4 =
# 32 FIXED**, c EXHAUSTIVE over 33..126 (94 positions), and the schedule is not
# an offset set at all -- it is a DEPTH-4 STACK of dilated bands of window 8,
# `[1,3,5,7]` being the per-layer DILATIONS, two hops per layer, so eight hops
# of composition. 0.1277 = 12/94 exactly. Reproduced and then re-read with j
# drawn at random, which is the correction the whole rebuild turns on.
def _dilated():
    import importlib.util
    p = pathlib.Path(__file__).resolve().parents[1] / "tests" / "cameron" / "dilated.py"
    spec = importlib.util.spec_from_file_location("_dil", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stack_severance(dilations, *, s, pairs, window=8, n_draws=4, seed=0):
    """UNREACHED / INERT / LIVE, verbatim classification from schedule_sweep."""
    cd = _dilated().composed_draws
    i = s - 1
    un = inert = live = 0
    for j, c in pairs:
        dr = cd(n_draws=n_draws, s=s, d=16, i=i, j=j, c=c,
                dilations=dilations, window=window, seed=seed)
        if any(lo != hi for lo, hi in dr):
            live += 1
        elif any(lo != 0.0 or hi != 0.0 for lo, hi in dr):
            inert += 1
        else:
            un += 1
    n = len(pairs)
    return dict(n=n, unreached=un, inert=inert, live=live,
                severed=(un + inert) / n, support=1.0 - un / n)


# ---------------------------------------------------------------- RED tests
def red_tests(s=128):
    """Three paths that MUST fire. An instrument that cannot fail is not one."""
    out = []

    # RED 1 -- the float path misses a real flip. Both factors healthy, product
    # flushes to zero. This is the reason the arm reads X4 and not the float.
    planted = [(1e-200, -1e-200)] * 8
    f64 = float_flip_rate(planted)
    x4 = flip_rate_x4(planted)
    prod = planted[0][0] * planted[0][1]
    a32 = torch.tensor(1e-30, dtype=torch.float32)
    b32 = torch.tensor(-1e-30, dtype=torch.float32)
    p32 = float(a32 * b32)
    out.append(("RED1 float instrument misses a real flip",
                f"float_flip_rate={f64:.4f}  X4={x4:.4f}  "
                f"lo*hi(f64)={prod!r}  float32 1e-30*-1e-30={p32!r}",
                f64 == 0.0 and x4 == 1.0))

    # RED 2 -- j pinned at s//4 severs everything whose 2-hop reach cannot
    # span the resulting offset. This reproduces the previous build's failure
    # FROM ITS CAUSE, and it checks its own precondition rather than assuming
    # it: max(R(D)) < i-j is what makes the reading 1.0000, so the test asserts
    # that first and only then requires 1.0000. A covering schedule is measured
    # at the SAME pinned j as the contrast -- if it also read 1.0000 the cause
    # would be something other than the pinning.
    i, jf = s - 1, s // 4
    off = i - jf
    pinned = [(jf, c) for c in range(jf + 1, i)][:60]
    rows, ok = [], True
    for name, D in (("contiguous 1..%d" % (s // 8), contiguous(s // 8)),
                    ("co-prime k=4", coprime(4)),
                    ("power-of-two", powers_of_two(4, s))):
        reach_max = max(reach2(D, 4 * s), default=0)
        if reach_max >= off:                      # precondition fails, skip
            rows.append(f"{name}=SKIP(reach {reach_max} >= {off})")
            continue
        fr, _ = severance_empirical(D, s=s, pairs=pinned, n_draws=2)
        rows.append(f"{name}[reach {reach_max}]={fr:.4f}")
        ok &= (fr == 1.0)
    cov_D = two_scale(max(2, int(round((s - 1) ** 0.5))), s - 1)
    frc, _ = severance_empirical(cov_D, s=s, pairs=pinned, n_draws=2)
    rows.append(f"COVERING k={len(cov_D)}={frc:.4f}")
    out.append(("RED2 j pinned at s//4 severs every short-reach schedule",
                f"offset i-j={off}  " + "  ".join(rows), ok and frc < 1.0))

    # RED 3 -- a non-covering schedule MUST fail gate 1 and MUST sever.
    bad = powers_of_two(7, s)
    cv = cover_value(bad, s - 1)
    sv = severance_exact(s, bad)[0]
    out.append(("RED3 non-covering schedule fails gate 1",
                f"D={bad}  coverage={cv}/{s-1}  exact severance={sv:.4f}",
                cv < s - 1 and sv > 0.5))
    return out


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=128)
    ap.add_argument("--probes", type=int, default=240)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--climb", action="store_true")
    ap.add_argument("--skip-exhaustive", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    s = a.s
    t0 = time.time()

    print("ARM A REBUILT -- additive basis of order 2 on the CAUSAL offset "
          "interval.\nNo name is coined. The G1 fetch for this cell is OWED.\n")

    # ---------------------------------------------------------------- (a)
    print("=== (a) THE SET. Sums, not differences. ===")
    v = 57
    Sg = SINGER[57]
    Sp = [d for d in Sg if d > 0]
    print(f"  old object   Singer(57,8,1) D={Sg}")
    print(f"    |D-D| mod 57                       = {diff_cover_value(Sg, v)}   "
          f"(v-1 = {v-1})   <- the GREEN gate")
    print(f"    |R(D)| on raw [1,56]               = {cover_value(Sp, 56)}   "
          f"(need 56)   <- what a causal 2-hop actually reaches")
    mod_sum = len({(x + y) % v for x in Sg for y in Sg})
    print(f"    |D+D| mod 57 (incl. 0)             = {mod_sum}   (v = {v})")

    found = {}
    for n in (56, s - 1):
        cap = 0 if n <= 56 else 600_000
        k, w, log, k0 = min_k_raw(n, node_cap=cap)
        found[n] = (k, w)
        print(f"\n  raw coverage of [1,{n}]  ->  k = {k}")
        print(f"    witness D = {w}")
        print(f"    coverage as a VALUE  |R(D) cap [1,{n}]| = {cover_value(w, n)}"
              f"   (need {n})")
        print(f"    counting lower bound k >= {k0}  (k(k+3)/2 >= {n})")
        for kk, how, nn in log:
            print(f"      k={kk:>2}: {how:<8} {nn:>10} nodes")
        proven = all(h == "REFUTED" for _, h, _ in log[:-1])
        print(f"    MINIMUM {'PROVEN' if proven else 'NOT PROVEN'} -- "
              f"{'every smaller k refuted exhaustively' if proven else 'a smaller k hit the node cap; k is an UPPER BOUND only'}")

    print("\n  wraparound, for contrast only -- a causal mask CANNOT wrap:")
    for k in (8, 9, 10, 11, 12):
        c, D = search_mod(v, k)
        print(f"    mod-57, k={k:>2}: best covered residues = {c:>2}/56"
              f"   {'FULL ' + str(D) if c == 56 else ''}")

    # ---------------------------------------------------------------- (b)
    pairs = random_pairs(s, a.probes, seed=20260825)
    js = [p[0] for p in pairs]
    print(f"\n=== (b) THE ARM at s={s}, hops=2, d=16, {a.probes} random (j,c) "
          f"pairs, {a.draws} draws each ===")
    print(f"  BIRTH GATE 2: j drawn uniformly at random -- min {min(js)} max "
          f"{max(js)} mean {sum(js)/len(js):.1f} distinct {len(set(js))}")
    print("  (j is NOT s//4 = %d; see RED2 for what pinning it does)" % (s // 4))

    n1 = s - 1
    wmin = found[n1][1]
    kmin = len(wmin)
    # MATCHED k AND MATCHED DEPTH. Every row is one layer at hops=2, s=128, so
    # depth cannot confound the schedule -- the confound this project already
    # published once. Rows are grouped by k so the comparison is like for like.
    cands = []
    for k in sorted({4, 7, kmin, 24, 32}):
        cands.append((f"-- k = {k} " + "-" * 8, None))
        cands.append((f"co-prime (odd) k={k}", coprime(k)))
        cands.append((f"contiguous 1..{k}", contiguous(k)))
        pw = powers_of_two(k, s)
        if len(pw) == k:
            cands.append((f"power-of-two k={k}", pw))
        if k == kmin:
            cands.append((f"BASIS minimal k={kmin}", wmin))
        fam = family_at_k(s, k)
        if fam is not None:
            cands.append((f"BASIS family k={k}", fam))

    if a.climb:
        for kk in (kmin, 24, 32):
            sv, D = hillclimb(s, kk, seed=7)
            if D:
                cands.append((f"climbed k={kk}", D))

    print(f"\n  {'schedule':<24} {'k':>3} {'cover':>7} {'meanP':>6} "
          f"{'sev(exact)':>11} {'sev(meas)':>10} {'unreach':>8} {'agree':>9} "
          f"{'viol':>5}")
    print("  sev(exact) = STRUCTURAL severance, closed form. sev(meas) = bitwise.")
    print("  viol = closed form said severed, autograd said live. MUST be 0.")
    results = []
    for name, D in cands:
        if D is None:
            print(f"  {name}")
            continue
        cv = cover_value(D, n1)
        sx, un, mp = severance_exact(s, D)
        sm, flags = severance_empirical(D, s=s, pairs=pairs, n_draws=a.draws)
        cf = closed_form_flags(s, D, pairs)
        # THE CLOSED FORM IS ONE-DIRECTIONAL AND MUST BE READ THAT WAY.
        # Structural zero influence => bitwise identical, always. The converse
        # is FALSE: c can move M[i,j] by less than its last bit, which the
        # bitwise test scores as severed. So the checkable claim is the
        # implication, and `viol` counts its violations -- it must be 0.
        agree = sum(1 for x, y in zip(flags, cf) if x == y)
        viol = sum(1 for e, p_ in zip(flags, cf) if p_ and not e)
        print(f"  {name:<24} {len(D):>3} {cv:>4}/{n1:<2} {mp:>6.2f} "
              f"{sx:>11.4f} {sm:>10.4f} {un:>8.4f} {agree:>4}/{len(pairs)} "
              f"{viol:>5}")
        results.append((name, D, cv, sx, sm, un, agree, viol))

    # ---------------------------------------------------------------- (c)
    print("\n=== (c) THE PRE-REGISTERED KILL ===")
    print("  fires if ANY bitwise-identical gradient pair at a random c,")
    print("  OR severance fraction > 0.1277 (co-prime baseline at s=128).")
    for name, D, cv, sx, sm, un, ag, vi in results:
        k1 = "FIRES" if sm > 0.0 else "clear"
        k2 = "FIRES" if sm > 0.1277 else "clear"
        print(f"  {name:<24} any-sever {k1:<6} (n={int(sm*len(pairs))})"
              f"   >0.1277 {k2:<6} ({sm:.4f})")

    # ------------------------- the baseline, re-read in ITS OWN harness
    print("\n=== THE 0.1277 BASELINE, IN THE HARNESS THAT PRODUCED IT ===")
    print("  tests/cameron/schedule_sweep.py: depth-4 stack of DILATED BANDS,")
    print("  window 8, 2 hops per layer. [1,3,5,7] are DILATIONS, not offsets.")
    i = s - 1
    jf = s // 4
    fixed = [(jf, c) for c in range(jf + 1, i)]
    stacks = [("co-prime [1,3,5,7]", [1, 3, 5, 7]),
              ("power-of-two [1,2,4,8]", [1, 2, 4, 8]),
              ("mixed radix [1,4,16,64]", [1, 4, 16, 64]),
              ("contiguous [1,1,1,1]", [1, 1, 1, 1])]
    print(f"\n  (i) j FIXED at s//4 = {jf}, c exhaustive over {len(fixed)} "
          f"positions -- reproduces the published cell")
    print(f"  {'stack':<26} {'unreach':>8} {'inert':>6} {'live':>5} "
          f"{'severed':>8} {'support':>8}")
    for name, dl in stacks:
        r = stack_severance(dl, s=s, pairs=fixed)
        print(f"  {name:<26} {r['unreached']:>8} {r['inert']:>6} {r['live']:>5} "
              f"{r['severed']:>8.4f} {r['support']:>8.4f}")
    print(f"\n  (ii) SAME stacks, j and c UNIFORMLY AT RANDOM ({a.probes} pairs)"
          f" -- the correction")
    print(f"  {'stack':<26} {'unreach':>8} {'inert':>6} {'live':>5} "
          f"{'severed':>8} {'support':>8}")
    stack_rows = []
    for name, dl in stacks:
        r = stack_severance(dl, s=s, pairs=pairs)
        stack_rows.append((name, r))
        print(f"  {name:<26} {r['unreached']:>8} {r['inert']:>6} {r['live']:>5} "
              f"{r['severed']:>8.4f} {r['support']:>8.4f}")

    # ------------------------------------------------------ gate 3, flips
    print("\n=== BIRTH GATE 3: X4 flip rate, c ON vs OFF the schedule ===")
    # smallest schedule among those at minimum measured severance: a bigger
    # k always helps, so ties must break toward the sparser object.
    best = min((r for r in results if r[2] == n1),
               key=lambda r: (r[4], len(r[1])))
    D = best[1]
    Dset = set(D)
    i = s - 1
    R = reach2(D, n1)
    rng = random.Random(99)
    onp, offp = [], []
    while len(onp) < 16 or len(offp) < 16:
        j = rng.randrange(0, i - 1)
        if (i - j) not in R:
            continue
        on = [c for c in range(j + 1, i) if (i - c) in Dset]
        off = [c for c in range(j + 1, i) if (i - c) not in Dset]
        if on and len(onp) < 16:
            onp.append((j, rng.choice(on)))
        if off and len(offp) < 16:
            offp.append((j, rng.choice(off)))
    print(f"  measured on: {best[0]}  D={D}")
    stats = {}
    for label, pool in (("on-schedule", onp), ("off-schedule", offp)):
        allp = []
        for j, c in pool:
            allp += draws(D, s=s, i=i, j=j, c=c, n_draws=24, lam=0.10)
        fl = sum(1 for lo, hi in allp
                 if valuation(lo)[0] and valuation(hi)[0]
                 and valuation(lo)[0] != valuation(hi)[0])
        lo_, hi_ = wilson(fl, len(allp))
        stats[label] = (fl / len(allp), lo_, hi_, fl, len(allp))
        fp = float_flip_rate(allp)
        print(f"  {label:<13} X4 = {fl/len(allp):.6f}  95% CI "
              f"[{lo_:.6f}, {hi_:.6f}]  ({fl}/{len(allp)})   float path = {fp:.6f}")
    a_, b_ = stats["on-schedule"], stats["off-schedule"]
    overlap = not (a_[2] < b_[1] or b_[2] < a_[1])
    print(f"  CIs overlap: {overlap}   -> gate 3 {'PASS' if overlap else 'FAIL'}")

    # ---------------------------------------------------------------- RED
    print("\n=== RED TESTS (each MUST fire) ===")
    for name, detail, fired in red_tests(s):
        print(f"  [{'FIRED' if fired else 'DID NOT FIRE'}] {name}")
        print(f"      {detail}")

    print(f"\n  elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
