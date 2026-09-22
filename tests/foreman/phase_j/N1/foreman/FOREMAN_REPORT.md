H-SINK is STRUCK on the existing checkpoints. The kill fires on P1 and P4. The twin has no first-token sink to relieve. A Qiu output gate makes the twin worse on 5/5 seeds. The win is data-dependent forgetting within 16 bytes, and (f)'s magnitude gate does that job at the same sites FoX's forget gate does.

Bed: 64 held-out windows x 512 bytes (`eval_batches()`, split seed 0), checkpoints `d45_ckpt_{a,f}_ss0_pair` and `phase_j/ckpt_a2F_ss0`. Repo HEAD 492ccff. Intervals are percentile bootstraps over the 64 windows unless marked. Test dir: `SP/phase_j/N1/foreman/` (SP = the session scratchpad). Every RED below was first run against a stub JSON, before its measurement script existed. The RED line for those stub runs was `RED <name>: value is null (stub or not measured)`.

## R-SINK rows

| row | bar | measured | test path | after measurement | verdict |
|---|---|---|---|---|---|
| P1 | lift ≥ 2 at delimiters (bytes 10, 46; `\n\n` is the doc boundary) AND AUC(−m) ≥ 0.7 AND AUC > within-doc shuffle p95 | closures (m < 0.05) = **65 / 98,304** site-layers, all layer 0; **lift 0.0** [0.0, 0.0]; AUC 0.791 [0.788, 0.794] vs null p95 0.515 (per layer 0.782 / 0.889 / 0.827) | `test_rsink.py P1` | still RED: `RED P1: P1 lift 0.0 < 2` | STRUCK |
| P2 | (f)'s \|W_ic\| from later queries in c's segment ≥ 0.3 | **0.0056** over 11,224 pairs / 65 closures (twin on the same pairs 0.0044) | `test_rsink.py P2` | still RED: `RED P2: P2 0.005610662512481213 < 0.3` | STRUCK |
| P3 | median ‖v‖ at closure / median ‖v‖ ≤ 0.5 | **0.843** (mean ratio 0.845) | `test_rsink.py P3` | still RED: `RED P3: P3 0.8432363867759705 > 0.5` | STRUCK |
| P4 | Spearman((f) closure rate, (a) attention received from the next 64 queries, t ∈ [1, 447]) ≥ 0.3 AND shuffle p95 \|ρ\| ≤ 0.05 | **ρ = −0.020** [−0.028, −0.012]; null p95 0.012; per twin layer −0.010 / −0.019 / −0.014; continuous variant (−min m) 0.137 | `test_rsink.py P4` | still RED: `RED P4: P4 rho -0.01985042469162218 < 0.3` | STRUCK |
| First-token mass (a) | max over layers ≥ 0.2 (bar set before measuring; the contract gives none) | **0.01130** [0.01127, 0.01134] / 0.00382 [0.00365, 0.00399] / 0.00343 [0.00322, 0.00362]; uniform = 0.01138 | `test_rsink.py FT_a_sink` | still RED: `RED FT_a_sink: (a) first-token mass max 0.011304507032036781 < 0.2` | the twin has no sink |
| First-token mass (f), FoX | ≤ 0.5× (a) in (a)'s argmax layer | (f) 0.00225 / 0.00163 / 0.00194; FoX 0.00190 / 0.00165 / 0.00130 | `test_rsink.py FT_f_relief`, `FT_fox_relief` | GREEN | VOID: a relief bar whose sink does not exist is worth nothing |
| P5 (from existing records, no training) | (a2) = twin plus a per-head sigmoid output gate after SDPA; recovery (a−a2)/(a−f) < 0.2 AND upper CI < 0.5 | **−0.155** [−0.275, −0.034] (t-CI, split seeds 0–4), negative on 5/5 | `test_p5.py P5_not_sink` | GREEN; `P5_sink_relief` still RED | CERTIFIED: not sink relief |

The contract's kill line is "P1 and P4 at null → H-SINK struck". It fires, and H-SINK is struck. On this bed the premise of H-SINK also fails: layers 1–2 never close (minimum m 0.119 / 0.129), and layer 0's closures are the tail of a hard-concrete floor at m ≈ 0.0497.

## Deeper cause, as far as a RED test binds it

All rows below are eval-only on the same sites. Each RED ran against a stub before its script existed.

| claim | bar | measured | test | verdict |
|---|---|---|---|---|
| (f)'s m_t and FoX's f_t mark the same sites | max-layer Spearman ≥ 0.3, null p95 ≤ 0.05 | **0.634 / 0.541 / 0.311**; null p95 0.0135 | `test_rsink.py X_colocate` GREEN | CERTIFIED |
| (f) lives inside 16 bytes | NLL(w=16) − NLL ≤ 0.01 | **+0.00033** nats/byte (w=64: +0.0, bitwise) | `test_window.py W1_f_local` GREEN | CERTIFIED |
| FoX lives inside 16 bytes | same | **+0.00110** (w=64: 2e−11) | `test_window.py W1_fox_local` GREEN | CERTIFIED |
| data dependence of the decay carries the function (eval) | constant-decay cost ≥ 0.05 | (f) m_t → layer geometric mean (0.433 / 0.429 / 0.502): **+0.316**; FoX log f_t → (layer, head) mean: **+0.381**. Both exceed C_win = 0.233 on these sites | `test_const.py C_datadep_f`, `C_datadep_fox` GREEN | CERTIFIED (eval-only) |
| a constant decay suffices | both costs ≤ 0.02 | +0.316 / +0.381 | `test_const.py C_const_suffices` still RED: `RED C_const_suffices: const cost f 0.3160681627143209 fox 0.38096815585093946 > 0.02` | STRUCK at eval |
| the twin loses to long-range dilution | windowing (a) to 16 recovers ≥ 0.5·C_win | (a) goes 1.276 → **3.002** (w=4 3.195, w=64 2.316); recovery **−7.42** | `test_window.py W2_twin_dilution` still RED: `RED W2_twin_dilution: twin recovery -7.42283652572706 < 0.5` | STRUCK |
| N's convention β = 1 holds on (f) | max\|β−1\| ≤ 0.01 | learned β = **1.061 / 0.930 / 0.900** | `test_beta.py` still RED: `RED N_beta_convention: max \|beta-1\| = 0.10042041540145874 > 0.01` | the convention does not describe our operator |

These rows answer the question of why FoX reproduces 100–111% of C_win with the same quintile profile. On this bed the family's magnitude leg is a forget gate:
- it marks the same sites as FoX's forget gate;
- it works within the same horizon (both ≤ 16 bytes; mean attention distance 1.6–2.0 for (f) and 1.2–1.3 for FoX, against the twin's 113 / 15.5 / 16.8 [descriptive, unbound]);
- it depends in the same way on per-token variation in the decay.

The family's own regime (exact zeros, absorbers, NEVER) fires on 0.066% of site-layers, and only in layer 0. So on this bed the tie of the leaps delivers no capability that a forget gate lacks.

With β ≠ 1, a fully closed row has W_ii = e^{(1−β)s_ii}. That is algebra from the operator's definition, not measured. It follows that D3's "m_i = 0 ⇔ λ_i = 1" does not hold at the learned β.

## Replacement routes

- **H-SINK → reroute to H-FORGET.**
  - Object: two trained twins on this bed at 3,538 steps, split seeds 0–2. One is an ALiBi twin (fixed per-head slopes, 0 extra parameters). The other is FoX, already at 100–111%.
  - Measurement: the fraction of C_win recovered on the same eval sites.
  - Bars: ALiBi ≥ 0.8·C_win means the win is a recency prior. ALiBi ≤ 0.5·C_win means the win is data-dependent forgetting, which FoX owns.
  - Seat: Foreman, about 2.2 min per arm-seed at FoX's 131.9 s clock.
  - Why sharper: P5 is already answered (−0.155), and the constant-decay ablation above is off-distribution at eval. A trained ALiBi twin is the only clean split between a recency prior and data dependence.
- **P2, P3 → reprice onto R-KR.** Object: the A₅ word problem with resets. The closure regime never fires on TinyStories bytes, so absorber rows only mean something on a bed that needs m = 0.
- **W2 → retire.** The twin uses context beyond 64 bytes (2.316 at w=64 against 1.276 full), so eval-time windowing cannot test dilution.
- **β convention → reroute to House (D2/D3 at β ≠ 1).** Chase's R-DIAG must read W_ii at the learned β (1.061 / 0.930 / 0.900), not assume β = 1.

## Limits

Only split seed 0 has weights for (a), (f) and FoX, so every row except P5 is one seed, and its intervals are window bootstraps, not seed CIs. P5 is read from `run_record.json` `final_eval_loss` of (a2), whose headwise form is not Qiu's elementwise G1 form. The P1 closure threshold 0.05 cuts the tail of a distribution floored at 0.0497. The 1.93× lift at the byte after a delimiter rests on 65 events and is unbound. The constant-decay ablation is off-distribution. It bounds from above how much data dependence is needed, and it proves no cause. The rotation leg (θ) was not examined. No nurses were spawned; every number comes from the scripts in this directory. Code SHA-256 prefixes: rsink_measure 80d81cf6c5de, window_measure 199781411039, const_measure 5e94725b5b61, p5_read ccc2965ee768, read_beta 255037e09317, test_rsink a832a81efb4f, test_window 4139324dc569, test_const dd983ca106dc, test_p5 ea87e3da0ce7, test_beta 0b7d3f3ca26d.

## OPEN

- Whether a trained constant-recency twin (ALiBi) recovers the win.
- Whether FoX's delimiter dimming is the mechanism. Its bottom-5% forget sites are enriched 4.9× / 8.1× for delimiters in layers 1–2 [descriptive, unbound].
- Seeds 1–4 for every eval-only row.

Default next move: run the H-FORGET ALiBi twin on split seeds 0–2, about 7 GPU-minutes. It is not started, because this brief said no training.
