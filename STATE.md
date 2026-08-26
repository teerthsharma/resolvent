# STATE — CEQ v9, ROUND 7, THE CAPABILITY ROUND

| field | value |
|---|---|
| contract | `LOOP_PROMPT.md` (v9). Round 6 archived `LOOP_PROMPT_ROUND6_ARCHIVE.md` |
| promise | **`HILBERT`** — D2 verbatim (needs **S1's +12 branch AND S3**), or an honest `BROKEN` |
| iteration | **2 complete, 3 next** |
| phase | **A — foundations (0–4)** |
| **scoreboard** | **23** — 22 carried, **S5 +1** (Merkle journal live) |
| RULE 2 (repaired) | **e-process LIVE and accumulating by iteration 8.** Breach = no live e-process at audit #1. **A deadline can slip; evidence accumulation cannot be deferred.** |
| register | **caveman, all agents.** Artifacts stay normal English |
| **RULE 5** | **Every kill ships a replacement route** — reroute / reprice / retire. **Now in the skill file, not just here.** |
| note | The stop-hook's prompt string still says *"CEQ v8.2 ROUND 6"*. **The file is round 7 and the file governs.** Stale label, current contract. |

## THE v9 BOARD — nothing on it earned except S5

| | item | pts | state |
|---|---|---|---|
| **S1** | M3 quintuple headline cell, **anytime-valid** | +12 / +6 / −4 | Chase — `scale/eprocess.py` in flight |
| **S2** | Hankel-gap family; one gap task **inside** M3 | +3 | Cameron — `ceq/hankel.py` in flight |
| **S3** | Kaggle run on the signed N1–N6 cert; HF; capability table v0 | +6 | not started |
| **S4** | Probe battery at **TRAINED** projections | +3 | **partly done** — Phase D closed 3/3; **aggregator retention NOT re-run** |
| **S5** | Merkle journal + tamper must-fire | +1 | **EARNED** |
| **S6** | Fused settling step, clock ≤ **1.1×** | +1 | target quantified: **3.0 dispatches/step, removable at zero FLOP cost** |
| **S7** | D1 to acceptance, certificate post-mortem as capstone | +2 | Foreman owns it |

## THE ONE NEXT ACTION (round 7, iteration 3)

**Collect the three fellows. All are live and none has landed.**

  1. **CHASE — the e-process must-fire, read before anything else he says.** A null
     simulation must **NOT** cross 20 in 10,000 replays at `α = 0.05`, **and seen
     failing to fire IS the calibration.** If the empirical null crossing rate
     exceeds `0.05`, **the construction is wrong and nothing anytime-valid may be
     claimed this round** — which would take RULE 2's repair with it.
  2. **CAMERON — the worked example, and K-5 rides on it.** If her rank/rank₊
     machinery disagrees with §1.1 (`rank_ℝ = 2` for the difference series; no
     nonnegative factorisation on raw entries; `rank 2–3 vs Ω(n)` after shifting),
     **the instrument is broken and nothing downstream of it is read.**
  3. **FOREMAN — the near-miss delta first** (it gates the Krohn–Rhodes framing),
     then **whether the supermartingale property actually holds for the grid
     mixture.** `tests/foreman/test_methods_mixture_identity.py` is in flight.
     **Every anytime-valid claim this round rests on that step.**

**Build alongside if none has landed. The two cheapest unowned items:**
  * **S6's fused settling step** — the target is already quantified at **exactly
    3.0 aten dispatches per settling step**, `O(t*)` while the FLOPs are `O(1)`,
    **removable at zero FLOP cost**. Worth **+1** at clock ≤ 1.1×.
  * **The unread `delta_image`** — see Open #2. It is the only non-saturating
    diameter in the entire journal set and **nobody has established what the
    restriction is.**

## Open REDs, carried

1. **Trained `Δ(G)` is readable but geometry-bound.** `κ_G = 0.9893068617` at the
   median trained cell, seed 0 — **a property of the s=64 harness, not of
   training**, which **raises** every diameter measured (`5.6865×` / `9.4530×`
   paired, CIs excluding zero). **The tail grew: `0/512 → 2/512 → 26/512`.**
   `s=256`, `s=1024` **NOT MEASURED**. **`κ = β` is still the only certificate
   that holds everywhere.**
2. **One non-saturating diameter sits unread in a committed journal.**
   `results/hilbert.jsonl`: `delta_image = 2.517898719097161 / 1.8227455242048691 /
   3.2026399097564138`, `kappa_cert_image = 0.5576903857145132 /
   0.42656198036250703 / 0.6644055723849551` — **the only restriction in the file
   that does not saturate, and nobody has established what the restriction IS.**
3. **No within-sequence norm-matched control exists.** Top-k by norm means every
   other within-sequence set is strictly lower; the matcher closed **zero** of the
   gap in **512/512** draws. **Structural, no repair known. No mechanism claim
   about *which* tokens may be made.**
4. **Aggregator retention `2.9% / 13.3% / 58.9%`** still random-init **and**
   rank-matched — upper bounds twice over. Part of S4.
5. **Seed 2's trained weights NOT FOUND on disk.** The free bind reproduces the
   published F-green cell on **seed 0 only**.
6. **Ten standing REDs in `tests/cameron/`** — all **BY DESIGN**, audited. Two
   pairs' findings are recorded in **no document**, and one pair's assertion
   **cannot ever fail** (`sigmoid × softmax` is non-negative by construction).
7. **The row-stochastic tradeoff is measured and written nowhere:** non-negative
   **cannot represent a decrementing loop**; unconstrained **loses the contraction
   guarantee** (`ρ = 1.0010632`). **§1.1 turns this into the Hankel gap — it is now
   S2's subject.**
