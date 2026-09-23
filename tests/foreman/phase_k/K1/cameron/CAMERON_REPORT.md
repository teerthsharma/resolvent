# CAMERON K1: floor rows and R-DEPTH at R0

**Verdict: R-DEPTH at R0 fails its prediction, and the registered (f_R) hits its learnability kill. The bed holds and the counter stays quiet.**

- **(f_R) as registered (γ fixed at 0.999, 8000 steps)** never learns its training depth. Held-out accuracy is 0.2026 and 0.2354 (seeds 0/1) against the 0.8 kill line, so the contract's learnability kill fires.
- **The γ-annealed (f_R) of amendment K1-b** learns on one seed out of two (0.9931 and 0.2011).
  - The seed that learns still misses the far band. It scores 0.2855 / 0.1454 / 0.0784 on depth > 160 at n = 4k / 8k / 16k, and 0.0322 on depth > 1280 at 16k, against a bar of 0.95.
- **Twins.** The only one that learns its training depth is aL7 (0.9948 / 0.9850). It scores **0.0000 on the far band at 16k** on both seeds, so the counter does not fire.
  - aL4, ass4 and aloop5 fail the learned-the-task gate on both seeds and are not read.
- **The bed holds.** bed_k's generator, scored on the far band (K1.F'' and K1.F'''), puts every construction at chance on every cell: 0.110–0.129 against a chance of 1/8.

**Why (f_R) misses the band.** The trained resolvent layer points exactly at the parent: hop 1 on 4,076–4,078 of 4,096 rows, mean max weight 0.988. The failure at test length is leakage out of the softmax, not the operator.
- Scaling its logit scale ×3 at evaluation, with no retraining, lifts depth > 160 at n = 4096 from 0.2479 to 0.9113.
- That probe is unbarred.

Lane: `SP/phase_k/K1/cameron/` (SP = the session scratchpad). The only repo writes were board lines. The GPU is an RTX 4060 8 GB with torch 2.14.0+cu126, one job per `gpu.lock.d`.

## 1. Floor rows (each bar RED on a stub before its producer existed)

### K1.F on bed_k: kept by my row, then voided by Foreman

Files: `test_k1f.py`, `k1f.py`, `k1f.json`.

| Bar | RED line | Measured | Status |
|---|---|---|---|
| cameron.k1.f_machinery | `FAIL ... k2==doubling mismatches : -1` | 0 mismatches, 0 credit violations, probes match | GREEN |
| cameron.k1.f_not_void | `FAIL ... n=4096 L=4 ... mean 1.0000 [stub]` | L4 0.0839 / 0.0696 / 0.0691; L7 0.4086 / 0.1722 / 0.1131 | GREEN, superseded |
| cameron.k1.f_line | `FAIL ... line None recomputed nan` | line = C2 in every cell | GREEN |

Superseded by Foreman's HPD window: 0.5642 at (L7, 4096). The Dispatcher ruled the bed void. The queued runner was stopped before any arm started.

### K1.F' on bed_k' (`bed_kp.py`, sha 20d50b9d): kept by my row, then voided by Foreman

bed_k' uses bed_k's generator, with 32 lanes and a depth cap of 32 for training, and 8 lanes for testing. Files: `test_k1fp.py`, `k1fp.py`, `k1fp.json`.

| Bar | RED line | Measured | Status |
|---|---|---|---|
| cameron.k1.fp_machinery | `FAIL ... hpd port == Foreman 0.5642224409448819 : nan` | the port reproduces 0.5642224409448819 exactly | GREEN |
| cameron.k1.fp_not_void | `FAIL ... n=1024 L=4 ... 1.0000` | train 0.3219; L4 0.1495 / 0.1397 / 0.1313; L7 0.4085 / 0.2506 / 0.1817 | GREEN, superseded |
| cameron.k1.fp_line / fp_deep / fp_resolvent_feasible | stub fails | recomputed = json; fraction ≥ 0.415; resolvent 1.0000 | GREEN |

Superseded by Foreman's builds: sc-HPD reached 0.4398; a 14-slot window built from real heads reached 0.5108; a 30-slot build reached 0.8096. Amendment K1-a then moved scoring to the far band.

### K1.F'' on the far band at 2·c·2^L (depth > 96 / 896)

Files: `test_k1fpp.py`, `k1fpp.py`, `k1fpp.json`. Every family is included, together with Foreman's sc-HPD (W 1–10, 14, 30, and splits), imported from his lane.

| Bar | RED line | Measured | Status |
|---|---|---|---|
| cameron.k1.fpp_machinery | `FAIL ... == 0.43982539164490864 : nan` | exact | GREEN |
| cameron.k1.fpp_band_size | `FAIL ... : 0` | 8,128–124,864 tokens | GREEN |
| cameron.k1.fpp_not_void | `FAIL ... : nan (argmax stub)` | L4 0.1270 / 0.1230 / 0.1239; L7 0.1104 / 0.1231 | GREEN |
| cameron.k1.fpp_line / fpp_resolvent_feasible | stub fails | recomputed = json; resolvent 1.0000 | GREEN |

### K1.F''' on the far band at 10·2^L (depth > 160 / 1280): the band the arms are scored on

This band was registered before any arm trained. Files: `test_k1fppp.py`, `k1fpp.py --band10`, `k1fppp.json`. The families include sc-HPD at W = 62.

| Bar | RED line | Measured | Status |
|---|---|---|---|
| cameron.k1.fppp_machinery | `FAIL ... == 0.43982539164490864 : nan` | exact; bands 160 / 1280 | GREEN |
| cameron.k1.fppp_band_size | `FAIL ... : 0` | 22,464 / 55,232 / 120,768 (L4); 49,088 (L7, 16k) | GREEN |
| cameron.k1.fppp_not_void | `FAIL ... nan` | L4 0.1260 / 0.1259 / 0.1264; L7 (16k) 0.1285 | GREEN |
| cameron.k1.fppp_line / fppp_resolvent_feasible | stub fails | recomputed = json; resolvent 1.0000 | GREEN |

## 2. R-DEPTH at R0 (far10 scoring; 2 seeds per arm)

**Recipe for every arm.** The recipe was fixed from pilots before any registered arm ran.
- Width d = 128 with 2 heads of 64 dims. The attention is the harness's AlibiAttention with no position table.
- Input x = E[id] + P·E[pid], with frozen random unit-norm id codes (V + 1 rows) and an orthogonal-initialised learned map P. The readout is tied to E.
- lr 3e-3, 8,000 steps of 8,192 tokens, with n cycling 256 / 512 / 1024 on bed_kp train beds.
- Evaluation: 64 held-out train beds [21, k] and the K1.F'' beds [31, n, k].

**Arms.**
- aL7 is parameter-matched to aL4 by setting d_ff to 183: 806,912 vs 805,120 non-embedding parameters.
- aloop5 is one block applied 5 times. That is matched to (f_R)'s clock using Chase's 2.27× cost; by FLOP count alone it would be 4.
- (f_R) uses Chase's hook, sha 3855288d.

**Per-run fields (unbarred):**

| run | params (non-emb) | train s / wall s | gate | held-out acc | depth>160 at 4k / 8k / 16k | depth>1280 at 16k | acc at 16k |
|---|---|---|---|---|---|---|---|
| far_fR_s0 | 4,999,556 (805,124) | 663 / 1129 | 0.3406 FAIL | 0.2026 | 0.0039 / 0.0033 / 0.0032 | 0.0000 | 0.0143 |
| far_fR_s1 | 4,999,556 (805,124) | 499 / 622 | 0.3951 FAIL | 0.2354 | 0.0250 / 0.0155 / 0.0066 | 0.0007 | 0.0169 |
| far_fR_ga_s0 | 4,999,556 (805,124) | 570 / 693 | 0.9963 pass | 0.9931 | 0.2855 / 0.1454 / 0.0784 | 0.0322 | 0.1486 |
| far_fR_ga_s1 | 4,999,556 (805,124) | 569 / 690 | 0.3387 FAIL | 0.2011 | 0.0309 / 0.0048 / 0.0189 | 0.0160 | 0.0287 |
| far_aL4_s0 | 4,999,552 (805,120) | 550 / 892 | 0.3465 FAIL | 0.2066 | 0.0047 / 0.0051 / 0.0015 | 0.0000 | 0.0127 |
| far_aL4_s1 | 4,999,552 (805,120) | 444 / 555 | 0.4008 FAIL | 0.2380 | 0.0235 / 0.0112 / 0.0052 | 0.0000 | 0.0173 |
| far_aL7_s0 | 5,001,344 (806,912) | 579 / 847 | 0.9948 pass | 0.9948 | 0.0014 / 0.0004 / 0.0002 | 0.0000 | 0.0185 |
| far_aL7_s1 | 5,001,344 (806,912) | 478 / 596 | 0.9850 pass | 0.9850 | 0.0000 / 0.0000 / 0.0000 | 0.0000 | 0.0158 |
| far_ass4_s0 | 4,999,560 (805,128) | 459 / 582 | 0.3986 FAIL | 0.2375 | 0.0200 / 0.0103 / 0.0047 | 0.0000 | 0.0157 |
| far_ass4_s1 | 4,999,560 (805,128) | 446 / 560 | 0.4375 FAIL | 0.2593 | 0.0475 / 0.0221 / 0.0113 | 0.0000 | 0.0225 |
| far_aloop5_s0 | 4,408,192 (213,760) | 473 / 598 | 0.5883 FAIL | 0.5883 | 0.0020 / 0.0000 / 0.0000 | 0.0000 | 0.0093 |
| far_aloop5_s1 | 4,408,192 (213,760) | 463 / 582 | 0.3488 FAIL | 0.3488 | 0.0333 / 0.0199 / 0.0102 | 0.0000 | 0.0184 |

**Bars.** The files are `test_rdepth_far10.py`, `test_rdepth_far10_ga.py`, and the band-free rows of `test_rdepth_far.py`. An arm that fails the gate is UNREAD, and an UNREAD row counts as failing.

| Bar | RED line (before any arm) | Measured | Status |
|---|---|---|---|
| rdepth_far_trained (gate) | `FAIL ... aL4 s0 ... : missing` | aL7 0.9948 / 0.9850 pass; aL4 0.3465 / 0.4008, ass4 0.3986 / 0.4375, aloop5 0.5883 / 0.3488, fR 0.3406 / 0.3951 fail | RED |
| rdepth_far_learnable (kill: fR ≥ 0.8) | `FAIL ... fR s0 ... : missing` | 0.2026 / 0.2354 | **RED, the kill fires** |
| rdepth_far_multilen16k (fR ≥ 0.95 at 16k) | `UNREAD ...` | UNREAD (gate) | RED |
| rdepth_far10_prediction (fR ≥ 0.95 on depth > 160 and > 1280) | `UNREAD ... fR s0 ...` | UNREAD (gate) | RED |
| rdepth_far10_counter_quiet (aL ≤ 0.5 on the band) | `UNREAD ...` | aL7 0.0000 / 0.0000 (16k, depth > 1280, N 49,088); aL4 UNREAD | RED (aL4 unread) |
| rdepth_far10_twin_under_line (≤ K1.F''' line + 3 SE) | `UNREAD ...` | aL7 0.0000 vs a bar of 0.1330; aL4 UNREAD | RED (aL4 unread) |
| rdepth_far10_opponents_quiet (ass4, aloop5 ≤ 0.5) | `UNREAD ...` | UNREAD (gate) | RED |
| rdepth_ga_trained / ga_learnable (K1-b arm) | `FAIL ... fR_ga s0 ... : missing` | s0 0.9963 / 0.9931 pass; s1 0.3387 / 0.2011 fail | **RED, the kill fires on s1** |
| rdepth_ga_prediction (≥ 0.95) | `UNREAD ...` | s0: 0.2855 / 0.1454 / 0.0784; 0.0322 at depth > 1280; s1 UNREAD | **RED** |
| rdepth_ga_multilen16k (≥ 0.95) | `UNREAD ...` | s0: 0.1486; s1 UNREAD | RED |

**Retired unrun** (registered, never read):
- `test_rdepth.py`: bed_k, voided.
- `test_rdepth_kp.py` and `test_rdepth_kp_line2.py`: bed_k', voided.
- The band-96/896 rows of `test_rdepth_far.py`: retired by the Dispatcher's PASS4 ruling, which scores the arms only on far10.

**Mechanism probes (unbarred; `diag_fr*.py` on the checkpoints, CPU):**
- **far_fR_s0.** The resolvent never learned a pointer: 1.4–1.7% of its mass sits on the parent. What it gets right (depth ≤ 4) comes from its ALiBi layers.
- **far_fR_ga_s0, n = 4096.** The resolvent points at the parent at 0.988 weight. Accuracy by depth is 1.000 at 0–128, 0.769 at 129–256 and 0.08 beyond 257.
- **Logit-scale probe.** Multiplying the logit scale at evaluation by ×1.5 / ×2 / ×3 / ×5 gives 0.8362 / 0.9006 / 0.9113 / 0.8650 on depth > 160.
- **The seed-0 pilot of the γ-annealed arm** points at the grandparent instead, and gives the same curve: 0.1417 at ×1, rising to 0.8764 at ×2.

## 3. Kills and their replacements

1. **bed_k void at (L7, 4096)** (Foreman HPD 0.5642).
   - Rerouted to bed_k' (8 test lanes), then to far-band scoring (K1-a).
   - The band scoring is bound here twice: K1.F'' and K1.F'''.
2. **bed_k' void at (L7, 4096)** (Foreman's real-heads W14, 0.5108).
   - Rerouted to far10 scoring. On it, every construction up to W = 62 scores chance (K1.F''').
3. **(f_R) learnability kill, as registered** (0.2026 / 0.2354).
   - Rerouted to the γ anneal (K1-b). It learns on 1 of 2 seeds.
   - Next route: γ anneal with **sparsemax weights**. Exact zeros remove the leak that the scaling probe shows is the far-band wall. Sparsemax is already gradchecked in Chase's K0 `resolvent.py` dense path.
   - It needs a fused sparsemax path and a registration before it runs.
4. **K1-b prediction fails on the one learning seed** (0.0784 at 16k).
   - Repriced: training at depth ≤ 32 cannot price the per-hop leak. A leak of 1–2% is harmless at depth 32 and fatal at depth 1000.
   - The same sparsemax route applies. A second option is a logit-scale floor, with a twin copy owed for the SSMax arm.
5. **L = 4 twins do not learn depth 16 in 8,000 steps**, on either seed.
   - Foreman's 16,000-step aL4 reached 0.818.
   - Repriced: a longer shared budget (about 24k steps) for every arm, registered as new run names. Without it the counter at L = 4 stays unread.
6. **Learned-embedding recipe retired.** It never learned depth 2 in 4,000 steps.
   - Run `kp_aL7_s0` was launched under a registered name and renamed afterwards. Its held-out accuracy was 0.1390, and it is reported here as a run.
   - Replaced by frozen codes with an orthogonal P.

**Process slips, all mine and all on the board:**
- Four CPU pilots opened CUDA contexts without the lock between 04:06 and 04:10. My own locked pilot then died of out-of-memory.
- One runner restart came 170 s after a release, not the required 180 s.
- Two board lines needed corrections: a row count of 14, not 8, and a gate value of 0.9963, not 0.9941.
- `rdepth.py` gained a pilot-only `--gamma_sched` flag partway through the series. It is off by default, so the other runs behave the same.

## OPEN

- **The L = 4 counter is unread.** The first unread row is aL4 on depth > 160. A gate-passing aL4 needs a longer budget, which applies to every arm.
- **The (f_R) prediction.** Is the far-band wall gone with sparsemax weights? The unbarred ×3 probe reaches 0.9113, not 0.95. The remaining errors (depth 513+ at 0.55–0.69) are unexplained.
- **K1-b is seed-fragile** (1 of 2 seeds). It is unknown whether a longer budget or a different anneal makes it reliable.
- **aL7 learned depth 32 but falls off right after it at test length.** At n = 4096 it scores 0.071 / 0.000 at depth 33–64 (s0 / s1) and at most 0.0014 on the far band, below the 1/8 chance. So it outputs a specific wrong id rather than a guess. Whether that is a length or a position effect is unmeasured.
- **Rungs.** The far10 band is priced at R0 only; R1/R2 need their own c_max.
- **MFU at R0** (RECORD_K clause 8) and the numerics/BOS clauses (6–7) were not in this lane. The (f_R) checkpoints are at `runs/far_fR*/model.pt` for Chase.

**Standing state:**
- The bed is bound: far10 on bed_k'.
- The twins do not shortcut the band.
- (f_R) as registered fails learnability. The γ-annealed (f_R) learns on one seed out of two and misses the band through softmax leakage.
- Default next step: register (f_R) with γ anneal plus sparsemax, and every arm at a shared 24k-step budget, RED first, before any K2 run.
