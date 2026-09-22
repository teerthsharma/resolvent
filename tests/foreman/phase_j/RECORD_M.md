# Phase J — Addendum M (the volume): dispatcher pre-check

Received 2026-09-22 ~23:05 while the Addendum L house-mode round was running; the
L round was not redirected. Contract: `ADDENDUM_M_CONTRACT.md` (author's, verbatim).

## M1's registered null cannot pass (producer `m1_null_check.py`, numpy/scipy, default_rng(0))

The contract's null — shuffle the t-axis, coherence must fall below 0.2
everywhere — was run on a planted drift (P = T = 256, v = 0.5) before any row
uses it. Shuffling t manufactures strong gradients along t, which the structure
tensor reads as coherent orientation:

| slice | median coherence (sigma 1 / 2 / 4) | fraction > 0.2 |
|---|---|---|
| planted drift | 0.996 / 0.996 / 0.996 | 1.000 |
| t-shuffled (the contract's null) | 0.928 / 0.909 / 0.904 | 0.998 / 1.000 / 1.000 |
| i.i.d. noise | 0.366 / 0.204 / 0.107 | 0.817 / 0.513 / 0.106 |

"Below 0.2 everywhere" also fails on i.i.d. noise at every scale (max 0.913 /
0.770 / 0.521). The instrument itself works: recovered slope dp/dt = 0.496
against the planted 0.5.

**Replacement (reprice the null, tested):** keep each frame intact and destroy
the motion by a random circular shift of each t-column along p; score slope
consistency (coherence-weighted fraction of pixels within 20% of the dominant
slope), not raw coherence; bar = data above the 99th percentile of 200
surrogates. Planted drift 1.000 vs surrogate p99 0.057 (median 0.022); the
contract's t-shuffle scores 0.033 and noise 0.010 on the same statistic.

## What the addendum depends on that does not exist yet (facts, 2026-09-22)

- §0 "Q1–Q3 unrun" is stale: Q2, Q2-REROUTE and Q3 are rows in
  `record_L.jsonl`; Q1 appears only as a board event. The L round is re-verifying them.
- M2 compares against "the kernel barcode"; no Mode-A kernel was built (K1–K5 never
  ran), and the trained gates' exact-zero fractions were 0.0 / 0.0024 / 0.0 / 0.0 by layer.
- M3 tracks R-CURV saddles over time; R-CURV exists only on one static pinned bed
  (bridge -0.7738, `ref_orc.py`), and its re-run was struck until -0.732 is pinned.
- M4 reads consequence coordinates c_i; no trained c-space exists (DO untested).

Runnable now: M1 with the repriced null on planted chains, and V logging in
training runs (Foreman's reflector row). M2–M4 wait on artifacts above.
