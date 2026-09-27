# Plan A — Shelved Canon: first run report (2026-09-27/28)

**Verdict: NO.** No pillar passed its pre-registered bar. P2's counter fired: the matched transformer is already calibrated past T*, so "honest past the horizon" is a training property, not an architectural one (PLAN §6).

## Per pillar

| pillar | verdict | number | file |
|---|---|---|---|
| P2-F Lorenz-63, skill before T*, REL after T* | **KILL** | REL(tf) = 0.0026 / 0.0033 / 0.0013 at 1.5 / 2.0 / 2.5 T*, counter threshold 0.02; BSS(tf) 0.6630 at 1.5 T*. tf learned gate 0.9865 / 0.9861 / 0.9851. Canon's REL ≤ 0.0012 comes from collapse to climatology: pre-T* skill 0.5906 vs tf 0.9501, MSE ratio to tf 2.08 / 7.65 / 9.12 (bar [0.8, 1.25]); fitted growth rate 2.4533 vs true λ 0.9059. tf-platt kill also fires at every live lead. | `p2/results_p2f.json` (bar sha 48a03f96…, 3 seeds) |
| P2-DJ: do candidate rankings outlive state prediction past T*? | **VOID** + underpowered | N = 300 × 3 seeds (bar registers 2,000). Registered null check failed: hit at 3 T* = 0.294, required 0.25 ± 0.03 (impulse, δ = 1e-2). Late excess over state-blind policy 0.027 / 0.003 / 0.020 / 0.007 (point), 0.010 / −0.006 / 0.003 / 0.009 (windowed). Horizon scaling 5.72 per 100× precision vs Lyapunov law 5.08. The file's own `verdict` field reads PREDICTION; the null-check failure overrides it. | `p2/results_dj_n300.json` |
| P1 hyperbolic consequence, far band | **OPEN** | At d = 2, H1 near-band F1 0.5108 and O1 0.2984, both below the 0.90 learn gate — rows void. Only E8 learned: near 0.9915, far 0.9457; floor F-depth3 0.1638, ceiling F-closure 1.000. | `p1/results_p1.json` (bar sha 8bcff25b…) |
| P3 / P4 / P5 | **OPEN**, not run | Beds, arms, bars and budget registered. | `p345/BAR.md` |

Integration gate (≥ 3 pillars pass) is unreachable from this run.

## What D-JEPA (arXiv 2609.24749) leaves open

D-JEPA owns ranking candidate futures with a bounded, permutation-equivariant correction trained on executed outcomes. Its decision-local gap is empirical (within-start Spearman 0.90 → 0.11 inside the top 4). The paper does not use chaos, Lyapunov, Koopman or hyperbolic language. The void, underpowered DJ read points at the one question it leaves open: *when* a ranking stops being predictable. Past about 1.5× the state horizon the Bayes-optimal ranking of 4 candidates carries no information beyond a state-blind policy, and the ranking horizon moves 5.72 time units per 100× precision against the law's 5.08. The candidate contribution is a horizon certificate for rankings, not a better ranker. Nothing here is claimed until DJ reruns at the registered N = 2,000 with the null check passing.

## Files

- `PLAN.md` — stale cells corrected (§2, §8: τ = 4 → 1.541; MDS means 0.3442 / 0.1066 / 0.0506; 10/50-dim worst cases 6.75e15 / 2.28e14); D-JEPA and ACPC added as owners.
- Bars, registered before runs: `p1/BAR.md`, `p2/BAR_DJ.md`, `p2/BAR_F.md` (each with Amendment A1 from House's shortcut hunt; the amendments record that House ran pilots first), `p345/BAR.md`.
- Code: `p1/p1.py`, `p1/test_p1.py` (3/3 planted checks pass), `p2/dj.py`, `p2/p2f.py`.
- Results: `p1/results_p1.json`, `p2/results_p2f.json`, `p2/results_dj_n300.json`.

Citations: 18 owners fetched; 4 marked [U] (Murphy 1973 and Trotter–Moore 1977 primaries, Lorenz 1975, FineWeb-Edu). One conflict unresolved (D-JEPA's 93.75% attributed to Reacher vs PushT), so it is cited nowhere; only 87.89% PushT (n = 256) is cited.

## Runs

| run | wall clock | outcome |
|---|---|---|
| `sun_maths.py` | 6.8 s | reproduced |
| P1 first attempt | 13 min 11 s | out of memory |
| P1 deciding | 994.7 s | completed |
| P2-F | 1,552 s | completed, peak GPU 0.201 GiB |
| P2-DJ N = 2,000 | 94 min 28 s | lost to host OOM from a parallel duplicate run |
| P2-DJ N = 300 | 481.1 s | completed |

Binding limit was host RAM (16 GB, 2–5 GB free), not the GPU.

## Limits

The deep Koopman arm was never built, so P2 could not have passed outright. P2-DJ ran at 15% of its registered N and failed its null check. P1's hyperbolic and order arms never learned the task, so P1's kill is unread. P4 (26–28 h R1, 71–106 h R2) and P5 (17–19 h R1, 48–71 h R2) need the author's yes on local GPU hours; P1's WordNet world needs the NLTK corpus download.

## Next move

P1 Amendment A2: register a Riemannian optimiser with burn-in for H1, a margin loss for O1, and widths d ∈ {2, 5, 10}, then rerun on CPU (≈ 20 min). Prior: once the order-embedding arm learns, House's kill fires — order, not curvature, is the lever for consequence sets.
