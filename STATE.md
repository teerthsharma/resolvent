# STATE — CEQ v10, ROUND 8, THE EQUILIBRIUM-CORPUS ROUND

| field | value |
|---|---|
| contract | `LOOP_PROMPT.md` (v10). Round 7 archived `LOOP_PROMPT_ROUND7_ARCHIVE.md` |
| promise | **`HILBERT`** — output only when completely and unequivocally true |
| iteration | **0 complete, 1 next** |
| **scoreboard** | **23** carried |
| **the round exists to move ONE number** | `done7.md` scores the product at **37 %**: engineering **≈ 80 %**, the scientific claim **≈ 5 %**. **The 5 % is because the deciding measurement has been taken ZERO times.** |
| **RULE 2** | **THE READING COMPLETES BY ITERATION 8.** Not the corpus, not the process — the reading. |
| register | **caveman, all agents.** Artifacts stay normal English |
| **RULE 5** | Every kill ships a replacement route — reroute / reprice / retire |
| House | **IDLE.** Released only on a NEW missing-innovation death. The round has its innovation; it needs execution. |

## THE v10 BOARD

| item | pts | owner | state |
|---|---|---|---|
| **X₁₇** e3-harmonic ladder on E4′ Rips, oracle = absorbing-chain solve, `λ₂` in `0.90–0.95`, both gates + planted controls seen firing | +3 | Cameron | in flight |
| **X₁₈** per-draw Ville e-process, calibrated both directions, ceiling arithmetic printed pre-run | +2 | Chase | in flight |
| **THE READING** — settled vs twin across `t* ∈ {1,2,8,32}`, trained weights, 5 seeds, fidelity column **with Identity ablation** | **+15** / **+6** | Chase | **RULE 2: by iteration 8** |
| **X₁₉** E4′ registered in `M3_TASKS` | +1 | Cameron | not started |
| **X₂₀** Kaggle segment + HF upload **with real weights** (author's say-so gate stands) | +4 | Chase | not started |

## WHY THE ORACLE CHANGED, AND IT IS THE WHOLE ROUND

**The absorbing-chain solve is a genuine equilibrium.** `N = (I − Q)^{-1}`, `B = N R`,
the fixed point of `z ← Q z + R`. **It is not a closed-form function of any bounded
neighbourhood**, so an arm that iterates has something to compute.

**Every task read before this round had a closed-form oracle** —
`x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]` and `x[:, :, CH_FLIP].sum(dim=1) ** 2` —
**which is why the deciding contrast came in at `−0.002959` with an interval covering
zero. That contrast was correct and uninformative.**

**And `λ₂` makes the curve a PREDICTION rather than a description.**
`t_rel = 1/(1−λ₂)`; engineering `λ₂` into `0.90–0.95` puts `t_rel` in `10–20`, so the
ladder `{1,2,8,32}` straddles the transition. Cheeger bounds `λ₂` by conductance from
both sides, so the target is reachable **by construction rather than by search**.

## OPEN, CARRIED

1. **`e3_t*` in the CURRENT registry inherits `e1_anchor`'s rig** — same
   `equilibrium_oracle`, whose label is the signed path sum the resolvent computes.
   **Only `settled − twin` is creditable on those rungs.** Foreman owns stating
   `oracle ≠ resolvent` for the NEW absorbing-chain oracle, in Lean.
2. **E4 as specified is STRUCK**; **E4′ (join the two largest) passes both gates at
   `n = 1024`** — leak `0.1565 → 0.9951` — **and is not yet registered.** Not
   admissible at `n = 64`, which still leaks at `0.0055`.
3. **Gate (a) passing does not mean a task is not locally decidable.** Reachability
   needs the diameter; *deciding* a label may not. **Any substrate needs the decoder
   gate, not just the truncation gate.**
4. **Fourteen vacuous controls struck across five authors**, the fourteenth a PASS
   case whose label was constant. **Every must-fire's PASS half now carries its own
   non-degeneracy check.**
5. **The one-hot control is worse than softmax** (`−0.118456`, CI
   `[−0.134115, −0.102786]`, 0/5) and **fails its own bar at `1.010779`**. The gain is
   the **mixture**; a lookup is worse than nothing.
6. **Settling adds variance and no mean** — `sd 0.064106` vs the twin's `0.016547`,
   **3.874×**, same batches, same init, same params.
7. **HF package built, NOT uploaded, and ships NO weights** — a random-init
   `model.safetensors` measures `1,901,686,656` bytes and was correctly refused.
   3 trained checkpoints exist on disk but are `pivot_unsigned`, not settled/twin.
