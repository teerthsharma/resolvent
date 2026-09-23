# Phase K — PROPAGATOR: process record

Contract: `PHASE_K_CONTRACT.md`. Instances: `instances/` (sha256 in `instances/SHA256SUMS`).
Data: `DATA_MANIFEST.json`. Lane producers and reports: `K0/<seat>/`, `S0/cameron/`.
Audit: `K0/inspector/PASS1.md` (rerun logs in `K0/inspector/rerun/`, checkpoints omitted).

## it.K0 (2026-09-23 01:37 – 02:5x): results as bound

Fellows on opus with nurses; one Inspector pass, 63 audited, 24 struck. Machine: RTX 4060
laptop (8 GB), torch 2.14.0+cu126, no flash attention (SDPA dispatches
EFFICIENT_ATTENTION).

**The author's instances reproduce.** `wald.py` and `yukawa.py` hash to their pinned
sha256 and re-run byte-identical to `wald.out` / `yukawa.out` (Inspector). Cameron's
independent rebuild of the wald bed and of Y3 reproduces every pinned number. Chase's
rebuild of Y1 and Y2 matches to ≤ 8.6e−16 (Y1) and ≤ 1.9e−13 relative (Y2). The
contract's instance kill (§6, last line) does not fire.

**R-DEPTH's prediction line is not a ceiling.**
- A hand-set 6-layer causal softmax construction (Foreman's C2 hybrid: pointer doubling
  plus a positional window) scores **0.3982** at R1 width on the pinned bed against the
  pinned L = 6 line of 0.2539. Its 27 heads are real softmax heads (0 mismatches against
  the idealised construction, margin 0.992).
- With chance credit on unresolved queries, perfect L = 6 doubling alone scores
  **0.2988** against a chance-credited line of 0.2739 (Cameron).
- The counter as written (> 0.5 beyond 2^L) does not fire at R1 width: the anchored
  hybrid scores 0.1934. At a window of the whole context (W = 4096) it scores 1.0.
- Width confounds R-SCALE: the centered-window slope is 1.2207 against the 1.2 edge of
  the doubling band (margin 0.021); a dilated bed gives a ratio of 1.2202 against its bar of 1.25 (Foreman).

**The logit scale is unidentified at one training length.** A resolvent fitted at a
single length (s_train = 11.50) scores 0.2551 at n = 16k, where s_99 = 14.50 is needed
(a = 1.082) (Cameron). The γ window on the pinned bed: γ = 0.9 gives 0.7798;
0.99–0.99999 give 1.0; 1 − 1e−10 gives 0.0605. γ = 0.999 is kept. The centered
resolvent is killed (0.5596 on its own bar; slopes 1.600 vs 1.528).

**The resolvent layer trains with exact gradients** (Chase). x = (1−γ)(I−γW)⁻¹V at β = 1;
backward by the transposed triangular solve, blocked in one reverse pass.
- Forward `fs5c` (pivot rebuilt from off-diagonal mass): 3.04e−6 / 4.08e−6 against
  float64 at S 1024. The unmodified fs5 stays RED at 3.68e−5.
- Backward `fusedc`: dq 2.95e−6, dk 9.74e−6, dv 2.15e−7. The kernel-normaliser adjoint
  stays RED at dq 1.14e−3.
- Gradchecks pass for softmax, sparsemax and the blocked adjoint; mutants fail.
- CUDA-graphed equals eager bitwise on 8 tensors.
- Cost, forward+backward against one SDPA forward+backward, B1 H8 D64: **2.27× at
  S 4096, 2.26× at S 8192**; the Inspector's idle-card rerun gives 2.30× / 2.26×.
  Reported, not scored.

**S0 f-lite** (Cameron, audited this pass). Exact to 2.899e−6 against dense smprime;
55,692 tok/s at 11.2M parameters and 1.63 GiB peak; **0.933×** the twin on the
Inspector's rerun (reported 0.92×; `test_cost` divides by a hard-coded twin figure).
HF patch parity stays RED at 6.690e−5 (gate-gradient precision; open).

**DATA is met** (Wilson). FineWeb-Edu `sample/10BT` files 000–003 at revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, licence ODC-By v1.0 (card line 632);
sha256 of all 4 parquet files matches the Hugging Face LFS record. Tokenised with
tiktoken 0.12.0 `gpt2`: 2,994,211,739 train tokens in 30 shards, 10,312,022 validation
tokens (the first 10,000 documents of file 000). Shards live outside the repository
(`C:/Users/seal/datasets/fineweb_edu/`); the Inspector re-hashed 3 of 31.

**Harness** (`K0/wilson/train_ladder.py`). `--attn alibi` is the ALiBi twin (a_L) with
no learned position path. GPU resume is bitwise only with `--deterministic`
(400/400 step losses, validation 6.742448, reproduced by the Inspector across sessions),
at 14.4% throughput. The default mode diverges at step 139 (max 2.22e−4), before the
resume point: run-to-run nondeterminism, not the resume path.

## Struck by the Inspector (to Open, not re-argued)

- Whether bed_k's random ids remove the shortcut. Foreman's `bedk_best_response`
  (0.3181) was declared RED and never run; Cameron's band for the beyond-2^L best
  response on bed_k (0.041–0.063, board 0.058–0.068) has no bar. The two disagree, and
  neither is bound.
- "The shortcut relies on ids = positions"; "no lower bound covers this bed" (sources
  not saved in the lane); "the counter fires at 48 slots"; Foreman's `id_range_gap`.
- Cameron's multi-length route (a = 1.26 → 1.0), the γ 0.99–0.99999 row at 16k, the
  pinned-seed depth claim, "extra heads do not beat perfect doubling".
- S0's 26M and 725k rows; the fp64-accumulation kill.
- Chase: the γ·ρ(W) < 1 check (it could not fail: W_00 = 1); bf16 rows with
  γ·rowsum ≥ 1 (19.3% / 21.2%, read of V = 1 at 2.53 / 1.91); BOS taking 51–99% of
  row 1023's read at γ = 0.999; tail-mass m > 0 on 16/16 content heads; log-kernel
  m < 0 on 7/8; unit self-weight counts; forward error at S 4096 / 8192 (reproduces,
  no bar); densec 9.41×; the pivot diagnosis of fs5's miss.
- Wilson: the selftest 8/8 (no RED; four checks could not fail); the MFU test
  (`test_mfu.py` edited after its RED; the 14 logs still reproduce every row).

Process notes: Chase dispatched no nurses. An Inspector nurse ran Chase's GPU bar file
during Wilson's `real_R1` run, which coincides with the R1 slowdown Wilson excluded.

## R-DEPTH as registered for it.K1 (before any K1 cell runs)

Each clause replaces a contract clause that K0 broke; the bound fact behind it is named.

1. **Bed.** bed_k: n distinct ids drawn from V = 32,768, parents by id, not position
   (Cameron, `K0_k1_bed`). Replaces the pinned bed, whose ids are positions.
2. **Floor row first (K1.F).** Before any arm trains, on bed_k at R0 width and at each
   test length: the position-only floor, chance-credited L-doubling, and the C2 hybrid
   at its best window. The (a_L) line is the maximum of the three, measured. Replaces
   "stays under 0.254", which a hand-set construction beats (0.3982). If the C2 hybrid
   exceeds 0.5 beyond 2^L on bed_k, the bed is void by the contract's own kill and a
   deeper bed is owed before any arm runs.
3. **Twin.** (a_L) is `train_ladder.py --attn alibi`: ALiBi with no absolute-position
   table, so no arm reads position except through the fixed prior.
4. **Logit scale.** (f_R) trains at n ∈ {256, 512, 1024}. Bar, registered now: (f_R)
   ≥ 0.95 at n = 16k after that training. The single-length fit is known to fail
   (0.2551).
5. **γ = 0.999**, fixed (inside the window at 4k and 16k).
6. **Numerics bar (Chase's replacement for γ·ρ(W)):** γ·max row-sum(W) < 1 and the read
   of V = 1 within 1 ± 1e−5, with the resolvent layer in fp32 outside autocast, on every
   R0 checkpoint.
7. **BOS.** Count BOS's share of each (f_R) head's read on the R0 checkpoints; bar:
   < 0.5 in the resolvent layer. If it fails, the per-head γ_h route is run next,
   registered before it runs.
8. **MFU.** Re-measured at R0 with the bar written before the sweep. Wilson's 14 logs
   (0.163–0.171 at R0) are the prior.
9. Any run compared bitwise to another uses `--deterministic`.

Kills stay as the contract states: (f_R) < 0.8 at the training depth is a learnability
kill.

## Amendment K1-a (registered 2026-09-23, before any R-DEPTH arm trains)

Clause 2's kill fired twice at the same cell, L = 7 and n = 4096. Both numbers are
bound (Inspector K1 PASS1–PASS2; Foreman K1 report):
- **bed_k:** Foreman's HPD window scores 0.5642 beyond 2^L. The best window is the
  maximum over bound windows, so this is above Cameron's 0.4086.
- **bed_k'** (Cameron's replacement, 8 lanes): a 14-slot window scores **0.5108**,
  built at R0 width as real causal softmax heads (15 heads × 8 dims + a 57-unit ReLU
  multiplexer) and matching the idealised window pointer for pointer in float64 and
  float32. A 30-slot build (31 heads × 4 dims) scores 0.8096. The "12 dims per slot"
  pricing understated what d = 128 holds by about 3×.

Every window found multiplies reach by a constant factor c over pointer doubling: 3.0 at
L = 4 and 3.5 at L = 7 for the 30-slot build. Past 2·c·2^L every construction scores
chance (1/8) on more than 100k tokens. Replacement clauses 1–2:

1′. **Bed.** bed_k' generator, scored only on the **far band**: tokens deeper than
    2·c_max(L)·2^L, where c_max(L) is the largest reach factor of any bound real-heads
    construction at that rung's width. Currently 6·2^L at L = 4 and 7·2^L at L = 7. For the
    L = 7 arm with 8 lanes, this needs n ≥ 16384.
2′. **Floor row K1.F''.** On the far band, run the position-only floor, credited doubling
    and every bound window (10-, 14- and 30-slot, HPD, no-oracle), RED first. The twin's
    line is the maximum of these. If any exceeds 0.5 on the band, the band moves out to the
    new c_max, and no arm trains until the floor is below 0.5.
- The free positional oracle stays in the kill rule. The harness's bf16 ALiBi position
  signal is coarse (39.79 positions), but a twin with a sharper position channel would not
  be, and a bed whose defence is a rounding artifact is not a defence.
- The R-DEPTH prediction reads on the far band: (f_R) ≥ 0.95 there at every rung.
- Open, and able to move the band: 2-dim slots (about 62 at d = 128) in fp32; the 30-slot
  build at n = 8192 and 16384; whether the band holds at R1 and R2.

## Amendment K1-b and clause 6 v2 (as registered during it.K1)

- **Clause 6 v2** (Chase, at the dispatcher's instruction). Registered and run RED at
  04:33:52, before any checkpoint was read (Inspector PASS4). γ·max row-sum(W) < 1, and
  the read of V = 1 within 2× the fp32-hook floor, where the floor is the fp32 hook
  against a blocked float64 reference on the same checkpoint's inputs. A power bar
  requires K0's fs5 to exceed 2× the floor in at least one cell. The original 1e-5
  line stays registered and is reported beside it.
- **K1-b** (Cameron): arm far_fR_ga_s{0,1}, with γ annealed 0.5 → 0.9 → 0.99 → 0.999
  (0.999 from step 6000; evaluation at 0.999). Bars `rdepth_ga_*` were registered and
  run RED at 07:50:34; the arms launched at 07:53:45 and 08:08:24. The schedule that
  ran equals the registered one. No setting outside the resolvent differs from the
  twins (Inspector FINAL).

## it.K1 (2026-09-23 02:5x – 09:4x): results as bound

Fellows on opus with nurses. The four lanes shared one GPU through a lock directory,
with a 3-minute re-acquire cooldown from 05:25. Seven Inspector passes audited 340
claims and struck 96 (Cameron 120/39, Chase 50/8, Foreman 129/33, Wilson 41/16). Lane
files: `K1/<seat>/`; audits: `K1/inspector/PASS1–6.md`, `FINAL.md`.

**R-DEPTH at R0 fails its prediction, and the registered (f_R) hits the learnability
kill.** Arms train on bed_k' at depth ≤ 32 (8,000 steps) and are scored on the far10
band (depth > 160 at L = 4, > 1280 at L = 7). An arm that fails the learned-the-task
gate is not read.
- **(f_R) as registered** (γ = 0.999 fixed): held-out accuracy 0.2026 / 0.2354
  (seeds 0 / 1), against the 0.8 kill line. The contract's learnability kill fires.
- **(f_R) under K1-b** (γ annealed): learns on seed 0 (gate 0.9963, held-out 0.9931)
  and not on seed 1 (0.3387 / 0.2011; the kill fires on seed 1). On the far band, seed
  0 scores 0.2855 / 0.1454 / 0.0784 on depth > 160 at n = 4k / 8k / 16k, and 0.0322
  on depth > 1280 at 16k, against a bar of 0.95.
- **Twins:** only aL7 passes the gate (0.9948 / 0.9850), and it scores 0.0000 on the
  far band at 16k on both seeds, against its line of 0.1330. aL4 (0.3465 / 0.4008),
  ass4 (0.3986 / 0.4375) and aloop5 (0.5883 / 0.3488) fail the gate and are not read.
  The L = 4 counter is therefore unread.

**The bed holds.** On the far10 band, floor K1.F''' reads 0.1259–0.1285 (chance 1/8),
and a hand-set resolvent scores 1.0000 on every cell. Foreman's densest bound
construction at L = 4 is a 62-slot window built as 63 real heads of 2 dims, exact in
fp32 at n = 4096 / 8192 / 16384. It has reach factor 4.25, putting the bound band at
depth > 136, inside far10. At L = 7 the proven reach is 3.5 (30-slot build, exact at
16384), and the idealised 62-slot reach of 5.0 is exactly far10's 1280. Every bound
construction scores chance on the band. A trained opponent (aL4, 2× an arm's tokens) is
void at its own gate (0.818 against 0.9).

**The resolvent layer trains** (Chase; hook `K1/chase/resolvent_hook.py`, sha
3855288d…). The layer runs in fp32 outside autocast. The backward forms dP from
x − mean(x): at γ = 0.999 the K0 backward cancelled in fp32 and put da/db at 7.5e-4.
The fix takes parity to y ≤ 3.7e-6 and da/db ≤ 5.2e-5; gradcheck passes, the
deterministic pair is bitwise equal, and `--compile` is barred. On every R0 (f_R)
checkpoint read (3 checkpoints, 4 cells):
- clause6_v2 is GREEN: γ·max row-sum 0.99900036, read error at 1.67 / 1.37 / 0.82 /
  0.62 × the floor. The power bar is GREEN: fs5 sits at 12–439× the floor.
- The original 1e-5 line is RED on 3 of 4 cells (1.75e-5, 9.27e-5, 1.10e-5). The fp32
  floor on the same inputs is 1.05e-5 to 6.75e-5.
- Clause 7 is GREEN: the largest BOS share is 0.187, so the γ_h route did not trigger.
- R-RANGE is RED: no trained (f_R) checkpoint meets its premise. The twin-slope bar is
  RED: 36 of 84 heads are out of band on Cameron's chain-bed pilots, and 16 of 16 in
  band on Wilson's LM twins.
- The K0 tail-mass range fit is killed for heads with a sink, replaced by the band-mass
  fit (Y2 to ≤ 1.9e−13).

**R0 language model and MFU** (Wilson). R0 is 4 × 128 layers, ctx 1024, batch 8, and
144,547,840 tokens per run. Every run shares one validation set.

| arm | seed | val loss | tok/s | hours |
|---|---|---|---|---|
| (a_L) | 1 | 4.997022 | 87,410 | 0.507 |
| (a_L) | 2 | 4.984507 | 85,587 | 0.516 |
| (f_R) | 1 | 5.016564 | 70,764 | 0.631 |
| (f_R) | 2 | 5.008180 | 65,884 | 0.688 |

- (f_R)'s validation loss is higher than (a_L)'s at both seeds. R-CARRY is scored at
  K3.
- MFU at R0, against a peak re-measured in the same session: (a_L) 0.134 (ctx 1024)
  and 0.144 (ctx 4096); (f_R) 0.115 and 0.113, counting no solve FLOPs. The contract
  assumed 0.35.
- The ward bar is RED on `done_clean`: foreign GPU processes appeared during 2 of the
  4 runs.

**Kills and their replacements.**
- **(f_R) learnability, as registered** → rerouted to the γ anneal (K1-b), which learns
  on 1 of 2 seeds and misses the band.
- The claimed mechanism is struck as unbarred: softmax leakage at test length, from an
  evaluation-time logit-scale ×3 probe lifting depth > 160 at 4k from 0.2479 to 0.9113.
  It enters K2 as a registered bar, not as a finding.
- **Twins failing the gate at 8,000 steps** → repriced to a longer shared budget for
  every arm.
- **Clause 6's 1e-5 line** → clause6_v2 (above).
- **K0 tail-mass fit** → band-mass fit.

## K2, as registered here (before any K2 cell runs)

K2 repairs R0 before climbing. The contract's it.K2 (R1 for every arm) waits: R1 starts
only when an (f_R) arm passes learnability (≥ 0.8 at training depth) on at least 2 of 3
seeds at R0.

1. **K2.0, the leakage bar** (evaluation only; checkpoint far_fR_ga_s0).
   - Scale the trained per-head logit scale by κ ∈ {1, 1.5, 2, 3, 5} at evaluation,
     with no retraining.
   - Read depth > 160 at n = 4096 and 16384, and depth > 1280 at 16384, on the far10
     beds.
   - Prediction: some κ lifts depth > 160 at 4096 to ≥ 0.9.
   - Counter: no κ lifts it above 0.5, meaning leakage is not the wall.
   - RED on a stub first.
2. **K2.1, the repaired arm.** (f_R_sp): γ anneal as K1-b, with sparsemax attention
   weights in the resolvent layer (exact zeros).
   - This needs a blocked sparsemax path: the dense path costs 9.41× at S 4096 and
     does not fit at 8192 and above.
   - All arms (aL4, aL7, ass4, aloop5, f_R_ga, f_R_sp) train at one shared budget of
     24,000 steps, 3 seeds each, under new run names.
   - Same far10 band, same gate, same bars.
   - Registered RED before launch; the dispatcher records the exact schedule here
     before the first run.
3. Deciding numbers stay on the local RTX 4060. Kaggle reproduction needs the author's
   yes per launch.
