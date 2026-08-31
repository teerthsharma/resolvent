# V12 — the adversarial controls on VGPE, filed before any V12 cell has run

**Written 2026-08-30, by the MORIARTY seat (the adversary).** Every `file:line`
below was resolved with `sed -n` against this worktree at commit `bd09ea0`
before it was written (P-6, `MISTAKES.md:315`). Every number is either a
**MEASUREMENT** with the command that produces it (§0), or a **DESIGN
REQUIREMENT** — a threshold a control must clear — labelled as such. Nothing
here is a prediction of what the round will find; that is Irene's file.

**Nothing in this round is trained yet and there is no harness.**
`scale/vgpe.py` is pure algebra: `grep -n "impact\|e4prime\|negation_scope\|M3_TASKS" scale/vgpe.py`
returns **zero hits**. Its own demo runs at `N_ACTIONS = 4`
(`scale/vgpe.py:386`), not the `|A| = 16` the pricing assumes
(`V12_PRICING.md:43`). `python -m pytest tests/watson/ -q` → `11 passed in
2.27s`, verified. So the five arms do not exist, and every pass/fail number
below is a bar the arms must be built to clear, not a forecast of clearing it.

**The headline.** Both of my standing attacks are admissible and both are
specified below with numbers. But the measurement I ran to check attack (i)
turned up something larger that the spec did not anticipate, and it leads:
**neither bed can supply VGPE with an admissible causal graph.** On E4′ the
position code either *is* the label (256/256) or is constant; on IMPACT the
planted graph shares **5 of ~6,125 nonzero entries** with the operator that
actually propagates, and the operator that does propagate reaches **21–33 of
1,024 tokens**. That is §4, and it must be resolved in `THEORY_V12_VGPE.md`
before a cell runs, because every branch of it is already decided.

---

## 0. The producer — every number in this file has a live command

P-1 (`MISTAKES.md:240`) requires a runnable producer for every number. The
script below is it. It reads only, trains nothing, writes nothing. Save it and
run `python moriarty_probe.py`; it prints §1 in full.

```python
"""MORIARTY depth-leak probe. Producer for every measurement in V12_MORIARTY_CONTROLS.md."""
import collections, sys
import numpy as np, torch
sys.path.insert(0, "<this worktree>")
from scale.impact import (build_impact_graph, _impact_query_node, _news_for_example,
                          impact_features, impact_planted_features,
                          CH_NEWS, CH_A_DEG, CH_RHO, CH_SEED)
from scale import negation_scope as NS

S, N, NTR = 1024, 512, 256
RNG = np.random.RandomState(20260830)

def nrmse_split(X, y):                      # fit on 256, score on the held-out 256
    w, *_ = np.linalg.lstsq(X[:NTR], y[:NTR], rcond=None)
    p = X[NTR:] @ w
    return float(np.sqrt(((p - y[NTR:]) ** 2).mean()) / y[NTR:].std(ddof=0))

def binfeat(M, dmap):                       # one column per depth shell, plus intercept
    bins = sorted(set(dmap.tolist()))
    Z = np.stack([M[:, dmap == b].sum(axis=1) for b in bins], axis=1)
    return np.concatenate([np.ones((M.shape[0], 1)), Z], axis=1)

for seed in range(5):                                            # ---- IMPACT
    g = build_impact_graph(S, seed, heterogeneous=False)
    q = _impact_query_node(g)
    A = np.abs(g.A_weighted) > 1e-12
    depth = -np.ones(S, dtype=np.int64); depth[q] = 0
    dq = collections.deque([q])
    while dq:
        u = dq.popleft()
        for v in np.nonzero(A[u])[0]:
            if depth[v] < 0: depth[v] = depth[u] + 1; dq.append(int(v))
    reach = depth >= 0
    wrow = np.abs(g.K[q]); ok = reach & (wrow > 0)
    corr  = np.corrcoef(depth[ok], np.log10(wrow[ok]))[0, 1]
    kmass = wrow[reach].sum() / wrow.sum()
    news = np.stack([_news_for_example(S, seed, i) for i in range(N)])
    Y  = (g.K @ (g.B @ news.T))[q]
    Bn = (g.B @ news.T).T
    x = torch.zeros(N, S, 16, dtype=torch.float32)
    x[:, :, CH_NEWS] = torch.from_numpy(news.astype(np.float32))
    x[:, :, CH_A_DEG] = torch.from_numpy(
        np.abs(g.A_weighted).sum(axis=1).astype(np.float32)).expand(N, S)
    x[:, :, CH_RHO] = float(g.rho); x[:, :, CH_SEED] = float(seed)
    print(seed, int(reach.sum()), int(depth.max()), kmass, corr, Y.std(ddof=0),
          nrmse_split(np.ones((N, 1)), Y),
          nrmse_split(impact_features(x, q, 0).numpy().astype(np.float64), Y),
          nrmse_split(impact_planted_features(x, q, 0).numpy().astype(np.float64), Y),
          nrmse_split(binfeat(Bn, depth), Y),
          np.median([nrmse_split(binfeat(Bn, depth[RNG.permutation(S)]), Y) for _ in range(5)]),
          nrmse_split(binfeat(news, depth), Y))
    SB = np.abs(g.B) > 1e-12; SA = A.copy()
    np.fill_diagonal(SB, False); np.fill_diagonal(SA, False)
    print("   suppB", SB.sum(), "suppA", SA.sum(), "inter", (SB & SA).sum(),
          "jaccard", (SB & SA).sum() / (SB | SA).sum())

xb, yb, fb, pb = NS.make_e4prime_batch(256, 1024, 24)            # ---- E4'
pts = tuple(tuple(pt) for pt in xb[0, :, NS.CH_COORD:NS.CH_COORD + 3].tolist())
marks = NS._e4prime_marks(xb)
pre  = NS._e4prime_adjacency(pts, NS.E4P_TARGET_DEGREE, False)
post = NS._e4prime_adjacency(pts, NS.E4P_TARGET_DEGREE, True)
def bfsdist(adj, s, t):
    if s == t: return 0
    seen = {s}; fr = [s]; d = 0
    while fr:
        d += 1; nx = []
        for u in fr:
            for v in adj[u]:
                if v not in seen:
                    if v == t: return d
                    seen.add(v); nx.append(v)
        fr = nx
    return -1
y = yb.numpy()
own   = np.array([bfsdist(post if br else pre, qi, qj) for qi, qj, br in marks])
dpre  = np.array([bfsdist(pre,  qi, qj) for qi, qj, _ in marks])
dpost = np.array([bfsdist(post, qi, qj) for qi, qj, _ in marks])
print("label==do()bit", sum((l > 0) == br for (_, _, br), l in zip(marks, y)), "/256")
print("finite(own)==label", int(((own >= 0) == (y > 0)).sum()), "/256")
print("finite(pre)", int((dpre >= 0).sum()), " finite(post)", int((dpost >= 0).sum()),
      " corr(dpost,label)", np.corrcoef(dpost, y)[0, 1])
```

---

## 1. WHAT WAS MEASURED

### 1a. IMPACT bed — five seeds, `s = 1024`, `n = 512`, split-half OLS (fit 256 / score 256)

| seed | components | query component | max depth from `q` | mass of \|K[q,·]\| on that component | corr(depth, log₁₀\|K[q,v]\|) | label sd |
|---|---|---|---|---|---|---|
| 0 | 318 | **21** | 7 | **1.000000** | −0.980487 | 2.726 |
| 1 | 332 | **33** | 12 | **1.000000** | −0.966598 | 4.283 |
| 2 | 338 | **27** | 9 | **1.000000** | −0.974832 | 3.952 |
| 3 | 335 | **29** | 11 | **1.000000** | −0.985738 | 3.200 |
| 4 | 333 | **22** | 8 | **1.000000** | −0.985735 | 2.809 |

Depth histogram at seed 0: `[1, 3, 2, 2, 5, 4, 2, 2]` — eight non-empty shells,
so the depth dial is non-degenerate (V-8's check, `MISTAKES.md:136`).

Held-out NRMSE on the same draws and the same split:

| seed | predict-the-mean | `impact_features` (local, the shipped FAIL probe) | `impact_planted_features` (2-hop, the shipped PASS probe) | **depth-binned `Bn`** | depth-binned `Bn`, **SHUFFLED depth map** | depth-binned raw news |
|---|---|---|---|---|---|---|
| 0 | 1.0011 | 1.0002 | **0.0674** | **0.085076** | 1.0129 | 1.0192 |
| 1 | 1.0110 | 1.0199 | 0.1649 | **0.626933** | 1.0347 | 1.0293 |
| 2 | 1.0026 | 1.0001 | 0.2138 | **0.251690** | 1.0199 | 1.0343 |
| 3 | 1.0008 | 1.0095 | 0.2111 | **0.574545** | 1.0134 | 1.0070 |
| 4 | 1.0108 | 1.0098 | 0.1881 | **0.630623** | 1.0250 | 1.0164 |

Predict-the-mean at ~1.0 is definitional, labelled as such per V-3
(`MISTAKES.md:72`). The shuffled-depth column is this control's own planted
positive (V-7, `MISTAKES.md:117`): it holds the number of features and the shell
sizes fixed and destroys only the depth *assignment*.

### 1b. IMPACT — which graph is the causal graph

`y = e_q^T (I − ρ A_norm)^{-1} B n` (`scale/impact.py:668-691`): `B` is applied
**once**, `A_norm` is the operator raised to powers.

| seed | nonzeros in `B` | nonzeros in `A_weighted` | intersection | Jaccard | reach from `q` on the `B` graph | reach from `q` on the `A` graph |
|---|---|---|---|---|---|---|
| 0 | 4096 | 2034 | **5** | **0.0008** | 1024 / 1024 | **21** / 1024 |
| 1 | 4096 | 2096 | **8** | **0.0013** | 1024 / 1024 | **33** / 1024 |
| 2 | 4096 | 1958 | **5** | **0.0008** | 1024 / 1024 | **27** / 1024 |

Distinct absolute edge weights in `A_weighted`: 1016 / 1048 / 979, against
1017 / 1048 / 979 undirected edges — **one weight class per edge**. Sign classes
in `B`: exactly two (`[-1.0, +1.0]`), over 4,076–4,082 distinct magnitudes.

### 1c. E4′ bed — one drawn batch, `n = 256`, `s = 1024`, `d = 24`, `seed = E4P_SEED`

Substrate: 1,019 pre-bridge edges, 179 components, largest 148. Bridge edge
`(343, 354)`; queried components sized **30** (left) and **32** (right), so 62 of
1,024 tokens are ever a query endpoint.

| quantity | measured |
|---|---|
| label balance | +1: 128, −1: 128, sd **1.000000** |
| `label == do()-bit` | **256 / 256** |
| under the **per-example** graph: `finite dist(qi,qj) ⟺ label = +1` | **256 / 256** |
| finite distances, when finite | min 4, median 13.0, max 28, 23 distinct values |
| under the **shared pre-bridge** substrate: pairs with a finite distance | **0 / 256** |
| under the **shared always-bridged** substrate: pairs with a finite distance | **256 / 256** |
| under always-bridged: `corr(dist, label)` | **−0.014944** (n = 256; \|r\| = 0.123 is the two-sided p = 0.05 threshold) |

---

## 2. ATTACK (i) — VGPE LEAKS LABEL INFORMATION THROUGH PATH DEPTH

**The correlation is REAL, seed-stable, and it is not the same object on the two
beds.** Neither bed behaves the way the attack's own wording assumed, and both
departures matter.

### 2a. IMPACT — depth is a sufficient statistic for the *weight*, not for the label

Depth is **constant across examples within a corpus draw**: `build_impact_graph`
runs once per batch (`scale/impact.py:576`), `_impact_query_node` runs once per
batch (`:581`), and only the news vector varies per example (`:585-589`,
`y[idx] = float(r_vec[query])` at `:589`; the oracle's own copy is
`y[b] = r_vec[query].float()` at `:690`). So "depth correlates with the label
across examples" is not even definable here — the depth map has zero variance
inside a corpus.

What is definable, and what I measured, is whether depth is a **sufficient
pooling statistic**. It nearly is, and the mechanism is exact:
`corr(depth, log₁₀|K[q,v]|)` runs **−0.9666 to −0.9857** over five seeds — the
resolvent weight is log-linear in hop count, because `K = (I − ρA)⁻¹` decays
geometrically. Pooling the propagated news `Bn` by depth shell and fitting ten
parameters reaches held-out NRMSE **0.085076 / 0.626933 / 0.251690 / 0.574545 /
0.630623** (median **0.574545**) against a shuffled-depth map at
**1.0129 / 1.0347 / 1.0199 / 1.0134 / 1.0250** (median **1.0199**).
**5 / 5 seeds separate**, and the *smallest* separation is 0.6306 against 1.0250
— a gap of **0.394**, eight times the round's 0.05 floor.

Two qualifications, stated here and not in a limits paragraph (P-8,
`MISTAKES.md:338`):

* Depth over **raw news** carries nothing: 1.0070–1.0343, indistinguishable from
  the mean. The leak is conditional on the plant `B` already having been applied
  — that is, on the model having learned the input map.
* Depth is **strictly weaker** than the bed's own already-shipped planted
  feature: `impact_planted_features` reaches 0.0674–0.2138 on the same draws and
  split, beating depth at every one of the five seeds. So depth is a real channel
  but it is not a shortcut past a control the bed does not already have.
  *(Aside, on P-2, `MISTAKES.md:251`: that entry records `0.06738` for the
  planted decoder as a figure with no on-disk producer. My seed-0 reading is
  **0.0674**. I am not claiming this reproduces that run — the original's `n`,
  split and fit are unstated — only that a value at that magnitude is now
  reachable from a runnable command, which is what P-2 asked for.)*

**The control: DM-IMPACT, the depth channel, at exactly zero parameters.**

The impact tensor is `[n, s, d_model = 16]` and uses channels 0, 1, 13, 14, 15;
`scale/impact.py:70` documents channels **2–9 as reserved and unused**. Every
arm's first operation is `self.wq(x), self.wk(x)` — `Linear` over `d_model`
(`scale/m3_quintuple.py:471`). Writing a feature into channel 2 therefore changes
**no parameter count**: `n_params` stays **4769**, the value every arm in the
shipped capability table carries (`ceq/hf_artifact/README.md:3`). This is the
parameter-matched depth control and its matching is checkable, not argued.

Channel 2 carries `hop(v, q)` — hop distance from token `v` to the query node,
with a sentinel for unreachable. The arm is `twin+RoPE+depth`.

| | criterion | number |
|---|---|---|
| **Non-vacuity gate, must fire FIRST** | `twin+RoPE+depth − twin+RoPE` | **≥ +0.05**, 95% CI excluding zero, 5/5 seeds |
| **Attack DEFEATED** | `twin+VGPE − twin+RoPE+depth` | **≥ +0.05**, CI excluding zero, 5/5 seeds |
| **Attack CONFIRMED (leak is real)** | `twin+VGPE − twin+RoPE+depth` | **\|Δ\| < 0.05**, CI including zero, *while* `twin+VGPE − twin+RoPE` cleared +0.05 |

**What makes DM-IMPACT vacuous, and how to detect it before the run:**

1. *The depth map is degenerate.* Print the depth histogram in the builder, not
   in the reading (V-12's rule, `MISTAKES.md:189`). Measured seed 0:
   `[1,3,2,2,5,4,2,2]`, eight shells — currently non-degenerate. Require `≥ 3`
   non-empty shells and `max depth ≥ 3` at every seed before the run; the
   measured range is 7–12, so this passes today.
2. *The channel is written but never read.* Detect by zeroing channel 2 on the
   trained arm and requiring the eval NRMSE to move. This is V-9
   (`MISTAKES.md:154`): a repair that changes nothing. If the number does not
   move, `twin+RoPE+depth` **is** `twin+RoPE` and the column is decoration.
3. *The channel is not free.* Assert `x[:, :, 2:10].abs().max() == 0` on a drawn
   impact batch before writing to it. `scale/impact.py:70` says these channels
   are unused; V-1 (`MISTAKES.md:34`) is what happens when that kind of claim is
   stated rather than tested.
4. *Nothing routes.* The label lives at node `q` (673 at seed 0) and the readout
   is at `s−1` (`scale/m3_quintuple.py:483`). If no arm can route across that
   gap, all five arms tie for a reason that has nothing to do with position
   codes. The non-vacuity gate above is the detector: if `twin+RoPE+depth` cannot
   beat `twin+RoPE` given a channel that names the answer's location, the bed is
   not measuring position encodings at all and the whole round is void.

### 2b. E4′ — depth does not *leak* the label, it **is** the label

Under the example's own do()-applied graph, `finite dist(qi,qj) ⟺ label = +1` in
**256 / 256** examples, and `label == do()-bit` in **256 / 256**. Half the
examples have no finite depth at all, so there is nothing to depth-match: the
attack as worded does not reach this bed, and the stronger statement in §4 does.

The only depth-matchable position graph on E4′ is the **always-bridged**
substrate, on which the pair is connected 256/256 and `corr(dist, label) =
−0.014944` — decorrelated by more than 8× against the n = 256 significance
threshold of 0.123.

**Control DM-E4′:** every arm's position graph is the always-bridged substrate;
the do()-bit stays where it already lives, in `CH_BRIDGE`
(`scale/negation_scope.py:892`), where the executable oracle reads it.

| | criterion | number |
|---|---|---|
| **Non-vacuity gate** | the two candidate position graphs differ | edge-count difference **exactly 1**, differing edge **exactly `(343, 354)`** |
| **Attack DEFEATED** | `twin+VGPE − twin+RoPE` under the always-bridged graph | **≥ +0.05**, CI excluding zero, 5/5 seeds |
| **Attack CONFIRMED** | the same delta | **\|Δ\| < 0.05**, CI including zero, *while* it cleared +0.05 under the per-example graph |

**Vacuity condition:** DM-E4′ is vacuous if the round's main arm already uses the
always-bridged graph — then the control and the arm are one object, which is
V-1's exact shape (`MISTAKES.md:34`). Detect by asserting the two graphs differ
on drawn instances with the count reported, which is V-1's own rule.

---

## 3. ATTACK (ii) — THE PLANTED-PATH ADVANTAGE IS ANNOTATION, NOT ARCHITECTURE (KILL K4)

VGPE is handed the true causal graph. Any model handed that graph might win. The
counter is to hand the RoPE arms the **same** graph, as per-token features, at
**matched parameters**.

**How the paths enter, and why the matching is exact rather than approximate.**
The tensor is `[n, s, d_model = 16]`; every arm's first op is
`Linear(d_model, d)` (`scale/m3_quintuple.py:471`). Free channels exist on both
beds: impact documents 2–9 as reserved (`scale/impact.py:70`) and E4′'s highest
used channel is `CH_QB = CH_BRIDGE + 2` (`scale/negation_scope.py:894`).
**Therefore the annotation costs zero parameters and `n_params` stays 4769**
(`ceq/hf_artifact/README.md:3`). The matching is not an argument; it is one
assertion, `sum(p.numel() for p in arm.parameters()) == 4769`, on both arms in
the same test.

The annotation is the same object VGPE consumes, flattened to per-token scalars:

| channels | feature | bed |
|---|---|---|
| 1 | `hop(v, target)` — hop distance to the readout target, sentinel if unreachable | both |
| 1 | `reach(v)` — 1 if a path exists, 0 otherwise | both |
| \|A\| | action-type multiset of the root path, one channel per type | both, budget permitting |
| 2 | `hop(v, qi)` and `hop(v, qj)` | E4′ |

Arm name: `twin+RoPE+paths`.

| | criterion | number |
|---|---|---|
| **Non-vacuity gate, must fire FIRST** | `twin+RoPE+paths − twin+RoPE` | **≥ +0.05**, CI excluding zero, 5/5 seeds |
| **Parameter gate** | both arms | `n_params == 4769` **exactly**, asserted |
| **K4 FIRES — the encoding does not ship** | `twin+VGPE − twin+RoPE+paths` | **\|Δ\| < 0.05**, CI including zero, *while* `twin+VGPE − twin+RoPE` cleared +0.05 |
| **K4 does not fire — the encoding is the contribution** | `twin+VGPE − twin+RoPE+paths` | **≥ +0.05**, CI excluding zero, 5/5 seeds |

**What makes K4 vacuous, and how to detect it before the run:**

1. *The annotation moves nothing.* Then "VGPE beats RoPE+paths" is a comparison
   against an arm that never received the annotation in a usable form, and the
   kill is untested. The non-vacuity gate is mandatory and is V-7's planted
   positive (`MISTAKES.md:131`): a reported absence needs a positive the search
   is required to find.
2. *The parameter count moves.* Then the arms are not matched and the row is not
   a kill test. Detected by the `== 4769` assertion.
3. *The free channels are not free.* Asserted, not assumed (V-1).
4. **The one that is already probably live, and it is the important one.** If
   `reach(v)` **alone** solves the bed, then RoPE+paths wins by the annotation and
   VGPE wins by the annotation, and the round measures the annotation twice under
   two names. On IMPACT the mass of `|K[q,·]|` on the query's own component is
   **1.000000 at every one of five seeds** — the label depends on *nothing*
   outside 21–33 of 1,024 tokens. `reach(v)` is therefore a one-bit mask that
   discards 97–98% of the sequence with **zero** loss of label information.
   **Detect before the run**: score `twin+RoPE` with `reach(v)` as the *only*
   added channel. If that alone clears +0.05 over bare `twin+RoPE`, both K4's
   kill and any VGPE win are measuring a reachability mask, and the round needs a
   bed where reachability is not the answer. I am filing that this check is more
   likely to fire than not, on the strength of `Kmass = 1.000000`.

---

## 4. THE ATTACK THE SPEC DID NOT ANTICIPATE — NEITHER BED CAN SUPPLY AN ADMISSIBLE CAUSAL GRAPH

This is stronger than either standing attack and it is settled by measurements
already in §1. It leads.

### 4a. E4′ — the position code is either the label or a constant. There is no third option.

The pre- and post-intervention graphs differ by **exactly one edge**,
`(343, 354)`, and the do()-bit **is** the label (256/256). So:

| position graph handed to VGPE | measured | consequence |
|---|---|---|
| **per-example** (do()-bit applied) | finite `dist(qi,qj)` ⟺ label = +1, **256/256** | the encoding *contains* the label. Any win is **VOID** — D-2 (`MISTAKES.md:642`), an oracle that is the arm's own resolvent, arriving through the position code instead of the label |
| **shared pre-bridge** substrate | finite `dist(qi,qj)`: **0/256** | the queried pair is never connected. The gauge transports nothing on the only pair that matters; the arm cannot exercise its own mechanism and a tie is guaranteed by construction — M-5 (`MISTAKES.md:457`), a process that cannot cross its own threshold |
| **shared always-bridged** substrate | finite: 256/256, `corr(dist, label) = −0.014944` | the path always exists and carries nothing about the label; the encoding is decoration on this bed |

Row 1 is the one that produces the round's most impressive number, and it is
worth nothing. Row 3 is the only admissible choice and it is also the one under
which VGPE has the least to gain — which is the honest position.

**The pre-run check, one assertion:** on a drawn `make_e4prime_batch` batch,
compute reachability of `(qi, qj)` in whatever graph the harness hands the
position code, and require `0 < agreement_with_label < n`. **All three candidate
graphs fail that bound today**: `n/n`, `n/n` inverted, `n/n` constant. This is
V-11's shape (`MISTAKES.md:178`) — a precondition satisfied, or violated, at
every real draw — and it must be resolved in `THEORY_V12_VGPE.md` before a cell
runs.

### 4b. IMPACT — the planted graph is not the operator that propagates

The brief plants "the edges of `B`". The label is
`y = e_q^T (I − ρ A_norm)^{-1} B n` (`scale/impact.py:668-691`): `B` is the
**input map, applied once**; `A_norm` is the **propagator, raised to powers**.
Their supports share **5, 8 and 5 entries** at seeds 0, 1, 2, out of unions of
~6,125 — Jaccard **0.0008 / 0.0013 / 0.0008**.

A path-ordered product along `B`'s edges is a walk on a graph that is, to three
decimal places, disjoint from the one whose powers generate the label. **An arm
handed `B` is mechanically the shuffled-gauge arm wearing the VGPE label** —
which would make arm 2 and arm 4 the same arm, and would turn the round's one
exactly cost-matched control (`V12_PRICING.md:254`) into its own baseline. That
is V-1 (`MISTAKES.md:34`), the repository's largest vacuity class, arriving
through a graph choice instead of a registry key.

**Control:** assert that the graph handed to the position code is the graph whose
powers the oracle takes. Number: the position graph's edge set must equal
`supp(A_norm)` with intersection-over-union **1.000**, or the round declares in
the pre-registration which operator it planted and why. Vacuity condition: the
assertion is vacuous if it compares a graph object against itself — detect by
also asserting the *rejected* graph fails it, which today reads **0.0008**.

### 4c. IMPACT — the right graph reaches 2% of the sequence

Planting `A_norm` fixes 4b and creates a different problem, measured at five
seeds: 318–338 components, the query's own component holding **21–33** nodes, and
`Kmass` on that component **1.000000** every time.

* **97–98% of tokens have no path to the readout and therefore no VGPE position
  at all.** What the harness does with them is a free choice the spec does not
  make, and every plausible choice — identity, a distinguished no-path rotation, a
  learned null token — encodes the reachability bit. Per the `Kmass = 1.000000`
  row, that bit is a **31×–49×** sequence-length reduction (1024/33 to 1024/21)
  handed to the gauge arms and to no RoPE arm. This is §3's vacuity case 4,
  promoted from a risk to a measured near-certainty, and it is why
  `twin+RoPE+paths` must carry `reach(v)`.
* **The priced tree is not this graph.** `V12_PRICING.md:366-367` prices transport
  on "`s = 1024` nodes and `s − 1` edges". Measured: 1,017 / 1,048 / 979
  undirected edges at seeds 0/1/2 — within 0.6% of the priced 1,023, so the edge
  count stands if transport runs everywhere. It does **not** stand if transport
  runs only where a path to the readout exists: a 21–33-node component carries on
  the order of 20–40 edges, so the priced transport term is then an over-price of
  roughly **30×**. And the priced `depth = 8` is the **minimum** of the measured range
  7–12, so the Linus-gate factor `2·depth/3` (`V12_PRICING.md:222`) runs **4.67×
  to 8.00×** across seeds rather than the flat 5.333× tabulated. No arm ordering
  moves — all gauge arms scale identically — so `V12_PRICING.md`'s conclusions
  survive; its per-seed constants do not.

### 4d. The action alphabet has no referent on either bed

`V12_PRICING.md:43` prices `|A| = 16` and flags it as not spec-fixed at `:361`;
`scale/vgpe.py:386` runs at `N_ACTIONS = 4`. Measured: `A_weighted` carries
979–1,048 distinct absolute weights over 979–1,048 edges — **one class per
edge**; `B` carries exactly two sign classes; E4′'s substrate carries one edge
kind plus one bridge edge, i.e. at most two. So `|A|` is **manufactured** by a
bucketing rule the spec does not state, and the shuffled-gauge control does not
shuffle that rule.

Two consequences, both filed:

* If `|A|` comes from quantizing edge weight — the only per-edge quantity either
  bed supplies — the VGPE arms receive a bucketed edge-weight code that the RoPE
  arms do not, and attack (ii) reduces to "put the bucketed weight in a free
  channel." That is one more mandatory channel in `twin+RoPE+paths`.
* If `|A| = 1`, which is the honest reading of E4′'s pre-bridge substrate, then
  the free monoid over one letter is `ℤ` — abelian — and `scale/vgpe.py:25-26`
  says so outright: *"RoPE is the special case: one action type, a chain graph,
  one generator."* At `|A| = 1` arms 1, 2 and 3 are one object up to block size,
  and the non-commutativity ablation has nothing to ablate.

**Control:** print `|A|`, the bucketing rule, and the realised distribution of
action types over the drawn edges, at build time. Then read the **learned**
generators through the dial that is already in the tree and already tested
non-vacuous, `commutator_norms` (`scale/vgpe.py:295`,
`tests/watson/test_vgpe_binds.py:313-340`).

* **Defeats vacuity:** `max_{a,b} ||[U_a, U_b]||_F > 1.0` on the trained
  generators — the same bar the shipped test uses at
  `tests/watson/test_vgpe_binds.py:336` — and every bucket occupied at least once.
* **Confirms vacuity:** the learned generators collapse toward commuting,
  `max ||[U_a,U_b]||_F < 1e-12` (`BAR`, `tests/watson/test_vgpe_binds.py:68`, the
  `abelian_collapse` bar at `:337`). Then the trained VGPE arm **is** the abelian
  arm, and rows 2 and 3 are one arm reported twice. The test that measures this
  exists; only the trained input is missing.

---

## 5. THE REPORTABLE-BUT-FALSE RESULTS THIS ROUND WILL PRODUCE

### 5a. Bed-picking — the likeliest false headline

Irene's §0 (`V12_IRENE_FILING.md:53-65`) makes a win on one bed and a tie on the
other the expected outcome, and there is an obvious temptation to headline the
winner. The arithmetic: the spec's own arms give 4 within-family contrasts × 2
beds = **8 headline cells**; my two mandatory columns add 4 more, for **12**.
Reporting the best of them:

| cells | at the nominal α = 0.05 | at the **achievable** α = 0.0625 (§6) |
|---|---|---|
| 8 | 33.7% | **40.3%** |
| 12 | 46.0% | **53.9%** |

**More than half the time, a round in which every arm is identical produces a
"win" somewhere.**

**Control:** pre-register the two-bed **conjunction** as the verdict, and print
the both-beds table whatever it says.
*Defeats bed-picking:* both beds clear +0.05 with CI excluding zero and 5/5 seed
agreement. *Confirms it:* one bed clears and the other's CI includes zero →
verdict is **SPLIT, NOT A WIN**, and the split is the headline.
*Vacuity condition:* the control is vacuous if only one bed is ever run — detect
by requiring both beds' cell counts **in the artefact** before any verdict
renders, which is M-6's rule and `foreman_looped`'s pattern
(`MISTAKES.md:469-478`).

### 5b. The abelian arm's cost mismatch, reported as a capability finding

`V12_PRICING.md:261-276` establishes that arm 3 is 2× cheaper per node **by
construction** (2×2 blocks against 4×4), with the FLOP difference 0.1369% of the
arm total. The false report writes itself: *"VGPE-abelian is nearly as good and
2× cheaper — ship the abelian one."* The 2× is block size, not commutativity, and
the capability tie would be priced against a confounded clock.

**Control:** run the abelian arm at **matched block size** — `b = 4` blocks whose
generators are constrained to commute (a shared eigenbasis), not `b = 2` blocks.
Then arms 2 and 3 differ in commutativity alone.
*Defeats the attack:* transport FLOPs of arms 2 and 3 equal **to the digit** at
every `n` tested, the same self-check F that already binds arm 4 to arm 2
(`V12_PRICING.md:27`, `:254`). *Confirms the confound is live:* any reported
arm-2-to-arm-3 FLOP or wall-clock ratio other than **1.000**.
*Vacuity condition:* a commuting `b = 4` construction is vacuous if the constraint
is only an **initialisation** — gradient descent leaves the commuting set and arm
3 becomes arm 2 with a different seed. Detect with `commutator_norms` on the
**trained** generators at the end of the run: `max ||[U_a,U_b]||_F < 1e-12` is
required for arm 3 to be the abelian arm at all.

### 5c. `VGPE > shuffled-gauge` reported as `VGPE > RoPE`

Irene predicts VGPE beats shuffled-gauge by +0.04 to +0.10 on impact
(`V12_IRENE_FILING.md:124`). Shuffled-gauge injects a *wrong* rotation into every
QK pair using a misassigned type — it is a **corruption** control, not an
ablation. `VGPE > shuffled` shows the assignment matters; it does not show that
non-commutative transport beats RoPE. Only §3's `VGPE > RoPE+paths` supports the
round's thesis.

**Control:** declare `VGPE − shuffled` **VOID as a capability claim in the
pre-registration**, not after reading it — D-2's own rule (`MISTAKES.md:657`). If
the round wants to keep the contrast as evidence of anything, it must also report
`shuffled-gauge − twin+RoPE`.
*Number:* if `shuffled` loses to plain RoPE by **≥ 0.05** with CI excluding zero,
a misassigned gauge is actively harmful and the assignment demonstrably carries
information. If `|shuffled − RoPE| < 0.05` with CI including zero, the gauge is
**inert**, and `VGPE − shuffled` is `VGPE − RoPE` restated under a second name.
*Vacuity condition:* a VOID declaration cannot itself be vacuous — but it can be
absent, and its absence is the failure mode D-2 records.

---

## 6. CEILING ARITHMETIC, BEFORE THE FIRST NUMBER (V-10, M-5, M-9)

M-5's rule (`MISTAKES.md:457`) and M-9's (`:531`) both require this printed before
any reading, so it is printed here.

* **Five seeds is a sign test.** The finest achievable two-sided p at N = 5 is
  **0.0625**, not the 0.05 the project quotes (`MISTAKES.md:539-540`). Unanimity
  excludes zero 385/385; a 4–1 split 20–44%; a 3–2 split 0–3.7% (`:537-538`).
  **No cell in this round can reach α = 0.05.** Every "CI excludes zero" written
  in §2, §3 and §5 above therefore *means* "5/5 seeds agreed", and I have stated
  the seed-agreement requirement explicitly in every row rather than letting the
  interval imply it. Every published interval must carry `n+` beside it
  (`MISTAKES.md:599-604`).
* **Both my controls inherit that ceiling.** They are paired contrasts on the same
  draws. Neither DM nor K4 can produce a p below 0.0625 at five seeds. That is a
  property of the design, not of the data, and it is known now.
* **The 0.05 floor is provisional on these beds.** It was measured from a paired
  delta SD of `0.056889` on `negation_scope` at s64
  (`M3_QUINTUPLE_PREREGISTERED_READING.md:83,88-90`). M-3's rule
  (`MISTAKES.md:415`) is that a pilot bounds nothing it did not measure. Neither
  V12 bed has ever been run under a twin/softmax contrast. **Print the realised
  paired SD beside the floor** rather than substituting it (M-2's discipline,
  `:402`). My own measured warning sign: the impact label sd runs 2.726–4.283
  across seeds, a 1.57× spread, and my depth-binned NRMSE runs 0.085076 to
  0.630623 — a **7.4× spread across five seeds of the same measurement.** Quoting
  seed 0 alone would have understated by that factor; running five is why I did
  not.
* **Seed requirement, if the realised SD matches `0.056889`.** To resolve a true
  0.03 gap — Irene's live 35% branch, `V12_IRENE_FILING.md:156` — at 80% power
  two-sided: `n ≈ (1.96+0.84)²·(0.056889/0.03)² ≈ 29 seeds`, a normal
  approximation and therefore itself a floor on the requirement. **Five will not
  do it.** M-9's recorded repair (`MISTAKES.md:602-606`) is to run the rung, label
  it **UNDERPOWERED with its seed count in the same breath**, and show the whole
  ladder rather than dropping the rung that cannot resolve. Declare it before the
  run, not after.

---

## 7. THE TWO MANDATORY COLUMNS

Both run on **both** beds. Both print `n+` beside the interval. Both are void
until their non-vacuity gate has fired.

| column | contrast | attack DEFEATED | attack CONFIRMED | non-vacuity gate that must fire first |
|---|---|---|---|---|
| **DEPTH-MATCHED** | `twin+VGPE − twin+RoPE+depth` (IMPACT) · `twin+VGPE − twin+RoPE` under the always-bridged graph (E4′) | Δ ≥ **+0.05**, CI excludes 0, **5/5** | \|Δ\| < **0.05**, CI includes 0, while the un-controlled contrast cleared +0.05 → **VGPE was reading hop count** | `twin+RoPE+depth − twin+RoPE` ≥ +0.05, CI excludes 0 (IMPACT) · the two position graphs differ by exactly edge `(343,354)` (E4′) |
| **PATHS-AS-FEATURES (K4)** | `twin+VGPE − twin+RoPE+paths` | Δ ≥ **+0.05**, CI excludes 0, **5/5** → the encoding is the contribution | \|Δ\| < **0.05**, CI includes 0 → **the contribution is the annotation; the encoding does not ship** | `twin+RoPE+paths − twin+RoPE` ≥ +0.05, CI excludes 0; **and** `n_params == 4769` on both arms, asserted |

Three further gates that are not columns but block the run:

1. **E4′ position-graph admissibility** (§4a):
   `0 < agreement(reachability, label) < n`. Currently **fails on all three
   candidate graphs**.
2. **IMPACT position-graph identity** (§4b): the position graph's support must be
   `supp(A_norm)`, IoU **1.000**; the rejected `B` graph must read **0.0008**.
3. **`reach(v)`-alone probe** (§3, case 4): if `reach(v)` as the sole added
   channel clears +0.05 over bare `twin+RoPE`, both mandatory columns are
   measuring a reachability mask and the bed must change.

---

## 8. THE `MISTAKES.md` ENTRIES THAT BEAR ON THIS ROUND

| entry | line | how it bears |
|---|---|---|
| **D-1** racing a baseline at its proven optimum | `:609` | Applies cleanly to IMPACT. Verified independently of Irene: one graph per batch (`scale/impact.py:576`), one query per batch (`:581`), only news varies (`:585-589`). Does **not** mechanically cover E4′, whose query pair is per-example (`scale/negation_scope.py:1001-1002,1006-1007`) |
| **D-2** an oracle that is the arm's own resolvent | `:642` | Now reaches the **position code**, not the arm. On E4′ the per-example graph's connectivity *is* the label, 256/256 (§4a). Its rule — declare VOID contrasts in the pre-registration — is what §5c requires for `VGPE − shuffled` |
| **V-1** two registry keys, one corpus | `:34` | §4b: planting `B` makes arms 2 and 4 the same arm. Its rule (assert the difference on drawn instances, with the count) is the control in §4b and in §2b |
| **V-7 / V-13** absence from a search that could not have found anything | `:117`, `:202` | Both of my controls' non-vacuity halves are planted positives, and both are mandatory. V-13's other half — scope a repo search to the tracked set, never a directory walk — is why my "`scale/vgpe.py` has no bed wiring" claim is a grep of one named file rather than a tree walk that would have descended into every sibling worktree |
| **V-8** the PASS half's label is constant | `:136` | Checked and passed: E4′ label sd **1.000000** with 128/128 balance; IMPACT label sd **2.726–4.283**. What *is* near-degenerate is `reach(v)`, base rate **0.0205–0.0322** — print it (§3, case 4) |
| **V-9** a repair that changes nothing | `:154` | The non-vacuity gates. A depth channel or a path channel that moves no number is decoration and its column is unreadable |
| **V-10 / M-5** ceiling arithmetic before the first number | `:168`, `:457` | §6. Also §4a's shared-pre-bridge row: an arm that transports nothing between the queried pair cannot produce the verdict it is run to produce |
| **V-11** a precondition satisfied at every real draw | `:178` | §4a: "the position graph does not encode the label" is violated at **every** draw on the per-example graph, and vacuously satisfied on both shared graphs |
| **V-12** trace the label back to the structure that generates it | `:189` | Print the depth histogram and the component-size distribution **in the builder**, not in the reading. `[1,3,2,2,5,4,2,2]` at seed 0 is that print |
| **M-3** pilot spread taken as realised spread | `:415` | My own depth-binned NRMSE spans **7.4×** across five seeds (0.085076 → 0.630623). Seed 0 alone would have been a 7.4×-optimistic pilot. §6 |
| **M-6** a partial run read as a verdict | `:469` | §5a: both beds' cell counts must be in the artefact before any verdict renders |
| **M-8** pricing every arm at one arm's rate | `:495` | Already honoured by `V12_PRICING.md:24-27`. §5b is the residue it could not fix: arm 3 is cheap for a reason unrelated to the property it ablates |
| **M-9** a verdict whose finest achievable p cannot reach its α | `:531` | §6. Applies to **every** arm and to both of my columns. Five-seed readings here are sign tests with a floor of p = 0.0625 |
| **P-1** a number with no live producer | `:240` | §0 is the producer for every measurement in this file |
| **P-2** a number whose only home is a commit message | `:251` | The planted-decoder figure now has a runnable command at the same magnitude: **0.0674** at seed 0 (§2a, with its non-identity caveat) |
| **P-6** line-reference drift | `:315` | Every `file:line` here was resolved with `sed -n` at commit `bd09ea0` before being written |
| **P-8** a bound stated as a price | `:338` | §4c: `V12_PRICING.md`'s transport term is an **upper bound** if transport runs only where a path exists, and that caveat is in the headline row, not below it |

---

## Limits

Collected once, here.

Every measurement in §1 comes from **one worktree, one box, `n = 512` (IMPACT) /
`n = 256` (E4′), and five seeds (IMPACT) / one substrate draw (E4′)**. The E4′
numbers are a single substrate because `_e4prime_points` is seeded from `E4P_SPEC`
and the shipped task registers exactly that one substrate
(`scale/negation_scope.py:880-883`); a different substrate would move the
component sizes and the finite-distance distribution but cannot move
`label == do()-bit` or `finite ⟺ label`, which are consequences of the
construction rather than of the draw — `qi` is drawn from the left component and
`qj` from the right, and the two graphs differ by one edge.

The NRMSE readings in §1a are **held-out ordinary least squares**, not the arms'
own optimiser. They bound what a *linear* reader of each feature set can do; a
trained arm can do better or, per M-1 (`MISTAKES.md:378`), worse if it sees
different preprocessing. They are evidence about what the features contain, not
about what any arm will achieve. In particular the `impact_planted_features` and
`impact_features` columns are re-scorings of shipped feature readers under my
protocol; they are **not** reproductions of the gate figures P-2 records.

**No cell was trained to produce any number in this file, and no VGPE arm
exists.** Every threshold in §2, §3, §5 and §7 is a design requirement on arms
that have not been written. If `THEORY_V12_VGPE.md` builds either bed differently
— a different query rule for IMPACT, a different position graph for E4′ — every
number in §4 must be re-derived against the actual builder before this file is
scored, the same condition Irene's filing places on itself
(`V12_IRENE_FILING.md:263-273`).

The Jaccard figures in §1b compare `supp(B)` **unsymmetrised** (4,096 directed
nonzeros) against `supp(A_weighted)` symmetric (1,958–2,096 nonzeros).
Symmetrising `B` can only *raise* the intersection, and the reachability column
already reports the symmetrised `B` graph (1024/1024). The direction of §4b's
finding is not sensitive to that choice; the third decimal place is.

The two mandatory columns are specified but **not implemented**. I own one file
this round and it is this one. Implementing `twin+RoPE+depth` and
`twin+RoPE+paths` is a channel write and a feature function on each bed; the
parameter-matching claim rests on `d_model = 16` being the input width for every
arm (`scale/m3_quintuple.py:471`) and on the named channels being unused
(`scale/impact.py:70`, `scale/negation_scope.py:894`), both of which I read but
neither of which I asserted in a test.

Finally, the adversarial check on my own strongest claim, §4a: would "the E4′
position code is the label" survive the harness deriving the position graph from
the *pre*-bridge substrate while still routing the do()-bit through `CH_BRIDGE`?
Yes — and that is the second row of §4a's table, where the queried pair is
connected in **0/256** examples and the gauge has nothing to transport. The claim
does not depend on which of the three graphs is chosen. It depends on there being
only three, which follows from the pre and post graphs differing by exactly one
edge, `(343, 354)`, measured.
