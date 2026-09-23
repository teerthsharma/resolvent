# FOREMAN K1: the shortcut hunt on bed_k, bed_k' and the far band

**Verdict: the far band holds, and Cameron's far10 band (depth > 160 at L = 4, > 1280 at L = 7) stands.**

**L = 4.** 2-dim slots raise c_max(4) from 3.0 to **4.25**, which moves the Amendment K1-a band from depth > 96 to > 136.
- 63 real heads of 2 dims each form a 62-slot window (126 dims).
- In fp32 it is exact at every test length:
  - n = 4096: full chain;
  - n = 8192 and n = 16384: layer-wise;
  - n = 16384 again as a full chain with rotary period 4096 plus the twin's own ALiBi slope 2^-5 to break aliases.
- On the new band, every construction scores chance: 0.1248 on 30,576 tokens.

**L = 7.** No 2-dim code I tried realizes at the band length n = 16384.
- Period n fails: 0.435 of rows mismatch, as predicted.
- Period 2048 fails because the alias premise is violated.
- Period 4096 fails with 1120 mismatches.
- The bound c_max(7) therefore stays 3.5 (the 30-slot build is exact at 16384, and its far-band score is 0.1246).
- The idealised c(W62) = 5.0 (band 10·2^7 = 1280) is what Cameron's far10 already covers.

**Trained opponent (item 3): void.** The long-budget L = 4 twin was Cameron's `rdepth.py`, trained on 131M tokens (4× an arm). It fails its registered learned-the-task gate: accuracy on depth ≤ 16 is **0.818**, against a 0.9 bar.
- Its far-band score, 0.0000, is therefore not a finding. It sits below chance because the twin emits a wrong id rather than a guess.
- Unbarred reading: even at 4× budget the twin reaches only depth ≈ 8–12, below the doubling line of 16. There is no sign of a budget-driven shortcut, but a gate-passing long run is still owed.
- The plateau row cannot be measured as I registered it, because `rdepth.py` probes evaluate n = 1024 only.
- The aL7 long run was never queued.

Lane: `SP/phase_k/K1/foreman/` (SP = the session scratchpad). Every bar was run RED against a stub before its producer existed (`red_*.log`, `mtimes_red.txt`); the board holds red before green. K0 and Cameron code is imported read-only.

## Bars (it.K1, all rounds)

| Bar (`foreman.k1.*`) | RED line (stub) | Measured | Status |
|---|---|---|---|
| hunt_machinery, ids_eq_pos, content_closure, no_oracle_ceiling | `FAIL … mismatches : -1` | Cameron's 0.4085876 exact; id window = doubling + 0.0013; content addressing reaches exactly 2^L | GREEN |
| leak_probe_detects, no_leak; kp_no_leak | `FAIL … planted … 0.0` | bed_k 0.0633 ≤ 0.0667; bed_k' 0.1274 ≤ 0.1306; planted 1.0 | GREEN |
| sources_verbatim | `FAIL … STUB QUOTE …` | 8 of 8 quotes verbatim, 4 saved PDFs | GREEN |
| hpd_not_void (bed_k) | `FAIL … nan [stub]` | 0.5642 at (7, 4096) | RED = kill fired (ruled) |
| hpd_beats_line, hpd_reference, local_wide_fires, local_crossing, r1_*, train_dist_*, deep_bed_floor | `FAIL … nan [stub]` | see earlier rounds | GREEN |
| deep_bed_is_deeper, alibi_position_channel (bf16 row), reach_* / reach2_* (L = 4) | stub lines | my arithmetic and definition slips | RED |
| reach3_*_L4 | `FAIL … 0.0` | c(HPD, 4) = 1.5; far band 0.0635 | GREEN |
| kp_attack_line_beaten / kp_attack_void | `FAIL … {'best': 0.0 …}` | 0.4398 at 10 slots, under 0.5 | GREEN / RED |
| kp_pricing_crossing | `FAIL … None` | crosses at 14 slots | GREEN |
| kp_w14_heads_exact_f64/_f32 | `FAIL … mismatches == 0 : -1` | 0 mismatches, 0.5108 (15 × 8 dims) | GREEN: bed_k' void (ruled) |
| w30_heads_exact_f32, next_reach_bounded, next_far_band | `FAIL … {'mismatches': -1 …}` | 0.8096; c = 3.0 / 3.5; band at chance | GREEN |
| w62_real_4096 | `FAIL … L=4 mismatches == 0 : {'mismatches': -1}` | 0 mismatches, L = 4 and 7, 8 beds | GREEN |
| w62_real_8192 | same stub | layer-wise rate 2.3e-4 ≤ 1e-3 | GREEN |
| w62_unreal_16384 | same stub | rate 0.435 ≥ 0.01 (period n cannot resolve at 16k) | GREEN (predicted failure seen) |
| w62_reach / w62_far_band | `FAIL … c_w62 0.0` | c(W62) = 4.25 / 5.0; band 8.5·2^4 / 10·2^7 at chance 0.1246 / 0.1244 | GREEN |
| w30_real_long | same stub | 16384 exact at L = 4 and 7; **8192: 587 / 1011 mismatches** | RED (triangle multiplexer) |
| w30_far_band_long | same stub | 0.1248 / 0.1246 at 16k; 0.1282 / 0.1484 at 8k | GREEN |
| w62a_real_16384 / _8192 / no_alias | `FAIL … L=4 rate <= 1e-3 : {'rate': 1.0 …}` | L = 4: 0 mismatches and 0 violations at both n; **L = 7: rate 0.105 / 0.092, 782k / 702k alias violations** | RED (L = 7) |
| w62p4096_real_16384 | `FAIL … L=4 : {'mismatches': -1 …}` | L = 4: 0 mismatches, band 0.1248; **L = 7: 1120 mismatches** | RED (L = 7) |
| w30t_real_8192 | `FAIL … L=4 mismatches == 0 : {'mismatches': -1 …}` | 3 / 6 mismatches, all at the near-alias Δ = 8146 of period √8192 | RED (my period choice) |
| w62t_real_16384 | same stub | L = 4: 0 mismatches, band 0.1248; **L = 7: 24,893 mismatches** (period 2048) | RED (L = 7) |
| far_opp_gate / quiet / chance / plateau | `FAIL … far_opp_gate aL4 … : 0.0` | gate 0.818 < 0.9, so VOID; band 0.0000 / 0.00002 / 0.00002; plateau crashes (probes are n = 1024 only) | RED (void) |
| twin_* (7, bed_k) | `FAIL … twin_machinery …` | not run, by ruling | RED on stub, unmeasured |

## Kills and replacement routes

1. **bed_k (7, 4096) void**, ruled. Route: bed_k'.
2. **bed_k' (7, 4096) void**, ruled. Route: far-band scoring, Amendment K1-a.
3. **The K1-a band moves at L = 4, from > 96 to > 136**, because c_max(4) is now 4.25 (W62 realizes at every test n). Cameron's far10 (> 160) already covers it.
   - At L = 7 the bound band stays at > 896. Far10 (> 1280) covers the idealised W62.
4. **Realizability is a precision question, not a width question.** One rotary pair of period n resolves adjacent positions in fp32 only while 1 − cos(2π/n) exceeds the fp32 spacing below 1 (about 6e-8): exact at 4k, 2e-4 at 8k, 0.43 at 16k.
   - Shorter periods need the ALiBi slope to kill the aliases below, and a window premise (t − c < period) that fails at L = 7.
5. **My triangle multiplexer was fragile.** A 1e-3 leak in x2, times big = 65536, gave 64-position errors. Replaced by a flat-top bump: 587 → 3 mismatches.
   - The residual 3 sit at the near-alias of my non-integer period √8192. A (128, 8192) code has a logit margin of 29.6 against 16.3 (computed, not run).

## OPEN

- **c_max(7) at 16384 between 3.5 and 5.0.** Untried: 3-dim heads (about 41 slots), other period and slope pairs, and stale bundles doubling the per-head coverage. Far10 covers up to 5.0; anything above 5.0 moves the band again.
- **R1 / R2:** c grows with width, so the band must be re-measured per rung.
- **Item 3:** a gate-passing long-budget opponent (L = 4 and L = 7) on the far band.
  - The 131M-token aL4 run did not learn depth 16 (0.818).
  - One GPU priority race is mine: I took the lock at 06:36:19 in the gap while Cameron's runner restarted, and ran 26 minutes ahead of it.
- **Unbarred fields:** the computed rotary margins; `pilot_cpu_L4.log`.
- **Struck in PASS1 and PASS2, not re-argued:**
  - max depth ≈ 2.2·2^L;
  - max depth ≈ n/16;
  - the HPD floor as a function of n/2^L;
  - the bf16 staircase beyond 39.79;
  - the 48% vs 26% grandparent catch;
  - each of 4 beds > 0.5;
  - "no lower bound covers this bed";
  - ids = positions;
  - far_band2_defeats_all;
  - the local reach fields;
  - the W12–W14 curve values;
  - the twin_L4 run and pilot_cpu2.
