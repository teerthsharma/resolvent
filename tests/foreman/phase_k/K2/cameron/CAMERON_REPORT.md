# CAMERON K2.0: the leakage bar

**Verdict: K2.0 fails its prediction, and its counter stays quiet.**
- Sharpening the trained logit scale lifts depth > 160 at n = 4096 from 0.2855 to 0.8750 (κ = 3). That is short of the 0.9 line and well above the counter's 0.5, so leakage is most of the 4k wall, but not all of it.
- At n = 16384 no κ passes 0.2621 on depth > 160, or 0.0810 on depth > 1280.
- **K2.1 pilot (sparsemax, 24k steps): BRANCH IDENTITY.** It learns (gate 0.9971), but its hard-walk ceiling is 0.8949 / 0.4513 / 0.2498, below 0.95 on all three cells. Its far10 is 0.3046 / 0.1503 / 0.0829 / 0.0325. The Inspector's K2 PASS2 audit is clean. The Dispatcher released the identity grid, and it has been running as a ward since 11:31:41.
- At 16k the wall is the pointer itself. The hard-walk ceiling of this checkpoint (the exact-zero limit of its own logits) is 0.4498 on depth > 160 and 0.2292 on depth > 1280. So sparsemax on this pointer cannot clear the 16k band.

Evaluation only, on `K1/cameron/runs/far_fR_ga_s0/model.pt` (sha256 e354c013…, hook 3855288d). The GPU is an RTX 4060 8 GB, used under `K1/gpu.lock.d`. There were three jobs: 10:10:48–10:11:08, 10:14:12–10:14:19 (184 s after the first release) and the lanes job 10:23:56–10:24:03 (`lock_releases.log`). No nurses were used. The only repo writes were board lines. Lane: `SP/phase_k/K2/cameron/`.

## Bars

Two test files were each written before their producer existed. Each was run RED twice before any measurement: once with the producer absent, and once on a stub. Neither was edited after its RED (the `.sha256` files confirm this).
- `test_k2_leak.py` (sha 0bc8ea17…). It holds the verbatim K2.0 bar plus the control, gate, profile and γ-horizon rows, all registered together before any K2 cell.
  - RED 1: 37 failing rows.
  - RED 2 on the null-world stub (κ and γ inert): 25 failing rows.
  - Measured: `run_test_k2_leak.log`, 22 failing rows, exit 1.
- `test_k2_walk.py` (sha af19b1b6…). It is the replacement route, registered after K2.0 was read and before its own measurement.
  - RED 1: 6 failing rows.
  - RED 2 on the null-pointer stub: 7 failing rows.
  - Measured: `run_test_k2_walk.log`, 5 failing rows, exit 1.

| Row | Measured | Status |
|---|---|---|
| leak_machinery (κ 1 / 1.5 / 2 / 3 / 5) | sha, κ, γ 0.999, a and b = κ × the trained values, beds = K1's | GREEN |
| leak_control_k1 (κ = 1 vs K1, ±0.0005) | 0.2855 / 0.0784 / 0.0322, per-token agreement 1.000000; gate 0.9963 | GREEN |
| **leak_prediction** (max over κ, depth > 160 at 4k, ≥ 0.9) | x1 0.2855 · x1.5 0.7941 · x2 0.8689 · **x3 0.8750** · x5 0.8546 | **RED** |
| **leak_counter_quiet** (max > 0.5) | 0.8750 | **GREEN: the counter does not fire** |
| leak_gate_at_best (κ = 3, depth ≤ 16 at n = 1024, ≥ 0.9) | 0.9962 | GREEN |
| leak_cell, depth > 160 at 4k (≥ 0.95) | 0.2855 / 0.7941 / 0.8689 / 0.8750 / 0.8546 | RED ×5 |
| leak_cell, depth > 160 at 16k (≥ 0.95) | 0.0784 / 0.1783 / 0.2260 / 0.2555 / 0.2621 | RED ×5 |
| leak_cell, depth > 1280 at 16k (≥ 0.95) | 0.0322 / 0.0476 / 0.0600 / 0.0745 / 0.0810 | RED ×5 |
| err_profile, κ = 3 (κ = 1 in brackets) | see the next table | GREEN (recorded) |
| gamma_numerics (fs5c fp32 vs float64 dense, ≤ 1e-3) | 5.93e-5 at γ 0.999; 8.03e-5 at γ 0.9999 | GREEN |
| gamma_machinery | κ 3, γ 0.9999 | GREEN |
| gamma_explains_513 (bucket 513–1024 at 16k, lift ≥ 0.10) | 0.2275 → 0.3026 (+0.0751) | RED |
| gamma_lift_1280 (depth > 1280 at 16k, lift ≥ 0.10) | 0.0745 → 0.0911 (+0.0166) | RED |
| **gamma_counter_quiet** (lift > 0.02) | +0.0166 | **RED: the counter fires, so γ is not the wall** |
| gamma_cell at γ 0.9999, κ 3 (≥ 0.95) | 0.8989 / 0.2889 / 0.0911 | RED ×3 |
| gamma_gate (≥ 0.9) | 0.9963 | GREEN |
| walk_machinery (κ-invariance of the pointer, causality, beds) | kinv_agree 1.0 | GREEN |
| walk_sane (hard walk on depth ≤ 128 at 4k, ≥ 0.98) | 0.9949 | GREEN |
| walk_explains (agreement of the soft model with the hard walk on depth > 160, ≥ 0.9) | 4k **0.9326**; 16k **0.7995** | 4k GREEN, 16k RED |
| walk_explains_counter_quiet (16k, > 0.7) | 0.7995 | GREEN, weak (see Limits) |
| **walk_ceiling** (hard walk on the three far10 cells, ≥ 0.95) | **0.9347 / 0.4498 / 0.2292** | **RED ×3** |
| walk_link_grows (off-chain link rate at positions 4096–16383 ≥ 2× positions 0–1023) | 0.00598 → 0.00840 (1.40×) | RED |
| lanes_machinery / lanes_events (≥ 30 links per rate) | 1441 / 72 / 1832 / 1779 links | GREEN |
| **lanes_position_driven** (r_pos ≥ 0.67 and r_dep ≥ 1.5) | r_pos 1.271, r_dep 6.325 | **GREEN** |
| lanes_depth_counter_quiet (r_pos > 0.33) | 1.271 | GREEN |

**Error profile at the best κ (κ = 3; row err_profile).** This answers the K1 report's open "513+ at 0.55–0.69".

| depth | 0–128 | 129–256 | 257–512 | 513–1024 | 1025+ |
|---|---|---|---|---|---|
| n = 4096, κ 3 (κ 1) | ≥ 0.9983 | 0.9874 (0.7778) | 0.8418 (0.1268) | **0.6058** (0.0474), N 591, max depth 565 | – |
| n = 16384, κ 3 (κ 1) | ≥ 0.9844 | 0.9733 (0.7122) | 0.7843 (0.1337) | 0.2275 (0.0363) | 0.0702 (0.0323) |

- On 8 beds the 513+ bucket reads 0.6058.
- The γ horizon does not explain it: the lift is +0.0751, against a bar of 0.10.
- At the band level on 4k, the hard walk explains the soft model's errors (agreement 0.9326). Of the 0.125 error on depth > 160:
  - 0.0615 is wrong under the hard walk too. These are off-chain argmax links, an identity error.
  - 0.0636 is right under the hard walk. That is the soft residual, which exact zeros would remove.
- At 16k, 0.1973 of the band is right under the hard walk but wrong in the soft read, and 0.5471 is wrong in both.

**Unbarred diagnostics (not findings):**
- The lane is the unit, not the token. Per-bed depth > 160 at 4k, κ = 3, spans 0.7678–0.9142 (`diag_perbed.log`). Beds 0–1 pooled give 0.9094 on GPU.
- K1's struck ×3 value of 0.9113 was a 2-bed draw of that spread. The 8-bed value is 0.8750.
- The mid-position off-chain rate is 0.00146 (positions 1024–4095), and the late rate is 5.8× that. No row reads this.

## Kills and their replacements

1. **The K2.0 prediction (some κ reaches ≥ 0.9 at 4k) fails at 0.8750.**
   - The counter is quiet, so leakage is a real wall: κ is worth +0.59 at 4k.
   - Replacement: the hard-walk split (`test_k2_walk.py`). The remaining 4k error is about half off-chain pointer identity and half soft residual.
2. **The γ-horizon hypothesis (the root share γ^d caps the far band) is retired.**
   - Its counter fires: eval γ 0.9999 moves depth > 1280 by +0.0166, with numerics GREEN.
   - Replacement: pointer identity, tested by `walk_ceiling`.
3. **K1's route "exact zeros remove the far-band wall" is killed for this pointer.**
   - The exact-zero ceiling on its own logits is 0.9347 / 0.4498 / 0.2292, all below 0.95.
   - Sparsemax is not retired. It is the measured repair for the leak part: +0.59 at 4k via κ, and 0.1973 of the 16k band is recoverable under the hard walk.
   - It is insufficient alone. K2.1 must also repair pointer identity past the training context.
4. **"Off-chain links grow with position" (walk_link_grows) fails at 1.40×.** In the far10 beds depth ≈ position / 8, so depth and position are confounded.
   - Replacement test, registered and run on the Dispatcher's order: `test_k2_lanes.py` (sha 916ffa6e). It is eval only and **GREEN** (`run_test_k2_lanes.log`).
   - At equal position (4096–16383), 32-lane beds are 4× shallower. Their off-chain link rate is 0.00932 against 0.00733 at 8 lanes (r_pos 1.271).
   - At equal depth (129–512), 32-lane beds sit 4× later in position. Their rate is 0.00927 against 0.00146 (r_dep 6.325).
   - `lanes_position_driven` PASSES (r_pos ≥ 0.67 and r_dep ≥ 1.5), and the depth counter is quiet (r_pos > 0.33). The wrong links follow position, not depth. That is the premise of the identity grid.

## K2.1, as registered (Dispatcher order after K2.0; all before the pilot runs)

Three tests are registered. Each was RED twice before any run: once with nothing present, once on a null-world stub. The log is `red_test_k21.log` and the hashes are in `test_k21.sha256`. Each registration is on the board.

1. **The pilot, `test_k2_pilot.py`** (sha 9996d651).
   - Run `pilot_fR_sp_s0`: K1's `rdepth.py` with the hook READY_SP names, `--steps 24000`, and `--gamma_sched 0.5:6000,0.9:12000,0.99:18000,0.999`. That is K1-b's anneal at the same fractions of the budget. The rest of the recipe is K1-b's, with seed 0.
   - GATE rows:
     - config (the hook must be named by READY_SP and sit in K2/chase);
     - trained ≥ 0.9;
     - walk machinery;
     - walk_ceiling ≥ 0.95 on the three far10 cells.
   - The far10 and multilen16k rows are read as usual and do not gate.
   - RED counts: 13 rows with nothing present, 10 on the null stub (K1-b relabelled; ceiling 0.9347 / 0.4498 / 0.2292; BRANCH NONE because READY_SP is absent).
2. **Branch rule** (printed by the pilot test as BRANCH):
   - **GRID**: every gate row GREEN. The 6 × 3 grid runs as registered, on the bars of `test_k2_grid.py sp`.
     - The far10 rows do not gate it, because the grid carries the R1 criterion (an fR arm learnable on ≥ 2 of 3 seeds).
   - **IDENTITY**: the pilot trained, but some ceiling cell is below 0.95.
     - **This covers both the 16k ceiling below 0.5 and the 0.5–0.95 range.** Below 0.95, the pilot's own exact-zero ceiling already fails the band, so the as-registered grid cannot pass its prediction.
     - The identity grid runs on the bars of `test_k2_grid.py id`.
   - **REPEAT_S1**: the seed-0 pilot fails the trained gate. `pilot_fR_sp_s1` runs once under the same file.
     - If seed 1 also fails, the result is **SP_RETIRED**: the identity grid runs with its fR_sp rows UNREAD.
3. **Both grids, `test_k2_grid.py`** (sha a622cd95). One file holds both, so neither is written after the pilot.
   - Arms: aL4, aL7 (d_ff 183), ass4, aloop5, fR_ga (K1 hook 3855288d) and fR_sp. Seeds 0, 1, 2 at 24,000 steps.
   - Branch sp: train_ns 256 / 512 / 1024.
   - Branch id: train_ns up to 4096 through `bed_kp.make_train`, which keeps 32 lanes and depth cap 32. Every arm gets the same schedule.
   - Rows: config, trained, learnable, r1, far10 prediction, multilen16k, walk_ceiling, counter_quiet, twin_under_line, opponents_quiet.
   - RED counts: 133 rows per branch with nothing present, 111 on the id null stub.
4. **Pilot result.** `run_test_k2_pilot.log`; the test is unchanged (the sha256 check passes); 9 rows fail.
   - The hook is `resolvent_sp.py` (sha 36039c3a, named by READY_SP).
   - Timings:
     - The pilot held the lock 10:42:32–11:15:56.
     - The walk ran 11:18:56–11:19:05, 180 s after the pilot released the lock.
     - The pilot ran the schedule registered before READY_SP existed (`0.5:6000,0.9:12000,0.99:18000,0.999`), not the K1-b step counts in READY_SP's usage line. This is noted on the board.

| Row | Measured | Status |
|---|---|---|
| pilot_config (GATE) | hook sha = READY_SP's; recipe as registered | GREEN |
| pilot_trained (GATE, ≥ 0.9) | 0.9971 | GREEN |
| pilot_learnable (≥ 0.8) | 0.9949 | GREEN |
| pilot_walk_machinery (GATE) | ok | GREEN |
| **pilot_walk_ceiling** (GATE, ≥ 0.95) | **0.8949 / 0.4513 / 0.2498** | **RED ×3** |
| pilot_far10 (≥ 0.95) | 0.3046 / 0.1503 / 0.0829 / 0.0325 | RED ×4 |
| pilot_multilen16k (≥ 0.95) | 0.1537 | RED |
| pilot_grid_launch | BRANCH: **IDENTITY** | RED |

   - The registered branch is IDENTITY.
   - The rows and the BRANCH line were posted and held. The Inspector's K2 PASS2 audit is clean on every row, and the Dispatcher then released the identity grid.
   - Two report sentences were struck at PASS2 and are removed: a checkpoint-to-checkpoint ceiling comparison, and a claim about the 16k wall on both checkpoints.
   - What the rows bind:
     - The pilot's soft read sits below its own hard-walk ceiling. At 4k, far10 is 0.3046 against a ceiling of 0.8949.
   - Unbarred hypothesis, no row reads it: at test length the sparsemax support is wide rather than exact-zero sharp. The walk_explains-style agreement and support-size rows would bind this on the pilot checkpoint.

5. **Identity grid, released and running.**
   - It runs as `ward.py` (PID 4252), launched by nurse N1 (haiku) at 11:31:41, from the job list `grid_id_jobs.json`. The jobs are 18 runs, seed-major, each under its launch name.
   - Each training run and each fR walk is its own lock hold with the 180 s cooldown. The ward posts a board line as each run lands.
   - Budget guard: if the projected total exceeds 14 h after any landed run, the ward stops and waits for the Dispatcher.
   - `test_k2_grid.py id` runs once, when all 18 runs have landed.
   - The pre-launch estimate is on the board. It is 2,346 s per run: the pilot's fitted step time at the grid's mean context of 1587, 81.5 ms/step, plus 390 s of overhead. The total is 12.9 h, with a band of 11.1–18.0 h.

## Limits

- One checkpoint (seed 0; seed 1 never learned), evaluation only, 8 beds per n.
- The SE fields in the logs treat tokens as independent. The sampling unit is the lane: 64 per n, with a per-bed spread of 0.7678–0.9142. The 0.025 miss of the 0.9 line sits inside that spread. The registered bar is the pooled value, and it fails.
- `walk_explains_counter_quiet` passed on the null-pointer stub (0.7445, `red_test_k2_walk.log`), because at 16k the soft model is mostly wrong. Its GREEN carries no weight.
- The walk's success rule (land in the token's own chain at depth ≤ 8 = 2^(L−1)) was fixed before its RED. It was never tuned.
- The OR over heads is the optimistic ceiling. Per head, the ceiling is lower: 0.3010 / 0.2971 at 16k on depth > 160.
- The lanes test changes the parent gap along with depth: about 8 positions at 8 lanes, about 32 at 32 lanes, and 32 lanes matches training. Its depth-129–512 rate at 8 lanes rests on 72 links, just above the 30-link floor.
- The pilot stub could not exercise the IDENTITY and GRID branch paths, because READY_SP is absent and that is part of the config gate. Those paths run for the first time on the real pilot.

## OPEN

- **The identity grid is running.** Its bars are `test_k2_grid.py id` (sha a622cd95), registered RED, and they are read once, after all 18 runs land.
- **Why the sparsemax pilot's soft read sits far below its hard ceiling** (0.3046 against 0.8949 at 4k). The unbarred guess is a wide support at test length. Its test would be support size and soft-vs-walk agreement on `runs/pilot_fR_sp_s0`, eval only, registered before reading.
- **What makes position drive the wrong links**: the number of competing keys or the logit-scale law. This is not split yet.
- **K1-b seed fragility.** The grid's r1 row reads it at 3 seeds.

**Standing state:**
- Leakage is real but not the whole wall, and γ is not the wall.
- On the K1-b checkpoint, the pointer's identity caps the 16k band at 0.4498 / 0.2292, and its wrong links follow position.
- The pilot's walk ceiling is 0.8949 / 0.4513 / 0.2498. Its registered branch is IDENTITY, and it is bound by the Inspector's K2 PASS2.
- Default next step: let the ward finish. It is estimated at 12.9 h, and the 14 h guard stops it early if needed. Then read `test_k2_grid.py id` once and report its rows.
