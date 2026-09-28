# P1 bar — hyperbolic consequence, far band (registered 2026-09-27, before any P1 run)

Every run records the sha256 of this file. A result JSON whose `bar_sha256` differs from the
file at that commit is void.

## World (exact truth)

- `tree`: random rooted tree, branching drawn uniformly from {1,2,3,4} per node, depth 8, capped at
  1,500 nodes (breadth-first). `dag`: the same generator, then every node at level >= 2 gets, with
  probability 0.25, a second parent drawn uniformly from level-1 nodes other than its parent.
- Relation R: (u, v) in R iff u is a proper ancestor of v (transitive closure, BFS). hop(u, v) =
  shortest directed path length.
- World seed = run seed. Seeds {0, 1, 2}. A comparative number needs all 3.
- WordNet nouns: **deferred** (the NLTK WordNet corpus is not on this box; fetching it is a data
  download, not taken in this run).

## Split (the far band)

- Train positives: pairs with hop in {1, 2} (the near band). Far positives: hop >= 3, never trained.
- Negatives: every ordered non-related pair (u != v, (u, v) not in R), split 50/50 at random into
  train negatives and held-out negatives (so "not seen as negative" cannot mean "positive").
- Far consequence-set query for u: candidates = far positives of u + held-out negatives of u;
  predicted set = {v : logit(u, v) > 0}; F1 per u; **macro-F1 over u with >= 1 far positive**.
- Far root-cause query: the same with roles swapped (query v, candidate ancestors).
- Near (in-distribution) F1: the same construction on train positives + train negatives.

## Arms (d = 2 primary; all trained with the same loss, batches, steps)

| arm | geometry | logit(u, v) | params |
|---|---|---|---|
| `E1` | Euclidean, width d | b - a*dist(u,v) + c*(r(v)-r(u)), r = norm | n*d + 3 |
| `E2` | Euclidean, width 2d | same | n*2d + 3 |
| `E8` | Euclidean, width 8d | same | n*8d + 3 |
| `H1` | hyperbolic (Lorentz model, tangent-at-origin parametrisation), width d | same, dist = arccosh(-<x,y>_L), r = distance to origin | n*d + 3 |
| `O1` | **House's shortcut**: Euclidean order embedding (Vendrov et al. 2016), width d | b - a*‖relu(x_u - x_v)‖² | n*d + 2 |
| `F-depth` | position-only floor, no training | +1 iff level(v) > level(u) | 0 |
| `F-all` | chance floor | +1 always | 0 |

Loss: BCE with positives and negatives drawn in equal numbers per batch (4,096 + 4,096), Adam,
4,000 steps, float64, CPU. Learning rate per arm chosen from {0.01, 0.03, 0.1} **on near-band F1
only** (never on the far band), seed 0; the chosen rate is then used for seeds 0-2.

## Learned gate (L-LEARN)

An arm whose near-band consequence-set F1 is < 0.90 at any seed is **void** on that world: its
far number is printed but enters no comparison.

## DERIVED before the run (why House's arm is here)

A rooted-tree poset has order dimension <= 2: let p1 be a DFS preorder and p2 the DFS preorder
with every child list reversed. If u is a proper ancestor of v, u precedes v in both. If u, v are
incomparable, they sit in different subtrees of their lowest common ancestor w, and the child of w
visited first in p1 is visited last in p2, so the two orders disagree. Hence x_u = (p1(u), p2(u))
satisfies u < v iff x_u < x_v coordinatewise: an exact **2-dimensional Euclidean** representation of
the full transitive closure, and transitivity holds by construction, so a near band learned exactly
implies far recall exactly. Negative curvature is not needed to *represent* consequence sets of a
tree; it is needed to represent tree *distances* at low distortion (S1). This is why O1 is the
strongest baseline lacking the tested property.

## Bars (tree world is deciding; dag world is reported beside it)

All comparisons on far consequence-set macro-F1, mean over 3 seeds, learned arms only.

- **Prediction (plan):** F1(H1) >= F1(E8) - 0.02.
- **Counter (plan):** F1(E2) >= F1(H1) - 0.02.
- **House kill (registered now):** F1(O1) >= F1(H1) - 0.02. If it fires, the lever for consequence
  sets is order, not curvature, and hyperbolic survives in Canon only for metric distortion.
- **PASS** iff prediction holds, counter does not, House kill does not, and H1, E2, E8, O1 are all
  learned at 3/3 seeds. **KILL** iff counter or House kill holds (with the arms involved learned).
  **OPEN** otherwise (e.g. an arm void, or prediction fails while the counter does not hold).
- Floors printed beside every number: F-depth and F-all. An arm that does not beat F-depth by
  >= 0.05 on the far band is reported as "at the floor".

## Lead prior (D-CALIB: the counter is the point estimate)

Expected: House kill fires on `tree` (O1 near 1.0 by the DERIVED block). Branch that would
surprise: H1 beats O1 by > 0.02.

## Amendment A1 (2026-09-27, after House's shortcut hunt, before any arm run)

Declared: House computed floors and a closure rule on the worlds (no trained arm) before this
amendment.

1. **`F-closure`:** the transitive closure of the train positives is printed as the ceiling (zero
   parameters; House measured F1 = 1.000). Every far number is also printed as a fraction of it. The
   far band therefore tests whether an arm's logit *builds in transitivity*.
2. **Floor:** `F-depth3` (+1 iff level(v) >= level(u) + 3) replaces F-depth. "At the floor" is
   measured against F-depth3.
3. **Threshold:** each arm's decision threshold is the one that maximises near-band consequence F1
   (on train pairs), frozen for the far band. Per-query average precision (threshold-free) is reported
   beside F1.
4. **New arms `E1-exp` and `E2-exp`:** Euclidean at widths d and 2d with x = (exp(|w|) - 1) * w/|w|,
   the same radial growth as H1's tangent parametrisation, so that a curvature win is not a
   parametrisation or step-budget win. **PASS also requires F1(H1) > F1(E2-exp) + 0.02.**
5. **Intervals:** every tie or kill comparison uses a paired bootstrap 95% CI over (seed, query), with
   2,000 resamples. A comparison holds if the whole CI is on its side of the margin, and fails if the
   whole CI is on the other side. Otherwise it is undetermined, and so is the verdict it feeds (OPEN).
   The realised depth is logged.
6. **E2 void while H1 is learned:** the counter scores as not holding.
7. **DAG generator (non-deciding world):** the second parent is drawn uniformly from nodes at lower
   levels that are not already ancestors. If none exists, it is skipped and logged.
