# Inspector, Phase K it.K0, pass 1 (pre-prognosis)

63 claims audited, 24 struck. Each claim has one `audit` line in `house-events.jsonl`, from the `dispatch` line "K0 pass 1" onward. The lanes audited are Cameron K0, Cameron S0, Foreman K0, Chase K0, the pinned instances, and Wilson K0 (engineering claims only; his fetched facts and the license text were not re-verified).

Reruns ran against the fellows' own files. Outputs went to `SP/phase_k/K0/inspector/rerun/`, and nothing was written into a fellow's directory: Python ran with `PYTHONDONTWRITEBYTECODE=1`, Chase's bar file ran with `CHASE_BOARD=0`, and Foreman's generators ran from copies. GPU jobs ran one at a time in fresh processes. The Inspector ran the timing reruns (Chase `cost.py`, S0 `test_cost.py` and twin, Wilson's deterministic pair) on an idle card, with 0 MiB and 0 % before each run.

## Table

| fellow | claim | test | RED seen (where, when) | rerun status / number | verdict | why |
|---|---|---|---|---|---|---|
| Foreman (pinned) | wald.py instance reproduces | sha256 + CPU run | n/a (pinned instance) | sha OK; exit 0; stdout byte-identical to wald.out | clean | hash and output match |
| Foreman (pinned) | yukawa.py instance reproduces | sha256 + CPU run | n/a | sha OK; exit 0; byte-identical to yukawa.out | clean | hash and output match |
| Cameron | wald rebuild reproduces every pinned number; RED on the L8 seed band (13/14) | K0_wald_rebuild | red_test_wald_rebuild.log 01:46:36 (ModuleNotFoundError bed_k), board 5184; bed_k.py 01:47:34 | exit 1; all 14 lines and wald_rebuild.json identical | clean | The band seeds 0..199 include seed 23, which is the pinned bed. |
| Cameron | pinned max depth 269 "ties the shallowest of 200 other draws", median 283 | none | none | not rerun | **struck** | No test computes max depth over seeds. Seed 23 is one of the 200. |
| Cameron | Y3 reproduces; law stays struck; slopes 1.17/1.61 | K0_y3_rebuild | red_test_y3.log 01:46:37, board 5187 | exit 0; 9/9 lines and y3_rebuild.json identical | clean | |
| Cameron | L=6 doubling opponent with guessing scores 0.2988 > 0.2739 | K0_doubling_ceiling_chance_credit | red log 01:46:37, board 5185 | exit 0; 0.298828125 | clean | |
| Cameron | gamma window on the pinned bed: 0.9→0.7798, 0.99–0.99999→1.0, 1−1e-10→0.0605 | K0_gamma_window | red log 01:46:37, board 5186 | exit 0; identical | clean | |
| Cameron | keep γ = 0.999 (inside the window at 4k and 16k) | K0_gamma_window + K0_logit_law field | as above | 4k 1.0000; 16k s_99 14.50 at γ 0.999 | clean | The 16k half rests only on a logit_law field. |
| Cameron | γ 0.99–0.99999 give 0.9964/1/1/1 at 16k (board correction 5202) | none | none | no file holds it | **struck** | No script or output file exists. |
| Cameron | logit scale unidentified: s_train 11.50 → 0.2551 at 16k; s_99 14.50; a = 1.082 | K0_logit_law_unidentified | red log 01:51:20 (k1_floors missing), board 5205 | exit 0; identical | clean | |
| Cameron | multi-length route a = 1.26 → 1.0000/1.0000 | route_multilen.py | none ("measured, no bar") | not rerun | **struck** | unbound |
| Cameron | centered resolvent killed (0.5596; 10.80 vs 11.72; slope 1.600 vs 1.528) | K0_centered_resolvent | red log 01:53:27, board 5230 | exit 1; 3/6 FAIL; centered.json identical | clean | The σ-0 slopes 2.31/1.28 come from rounded log values; full precision gives 1.29. |
| Cameron | K1 bed = bed_k with n distinct ids from V = 32,768 | K0_k1_bed | red log 01:46:37, board 5188 | exit 0; 10/10 identical | clean | |
| Cameron | beyond-2^L best response ≈ chance (0.041–0.063; board: 0.058–0.068) | none (k1_floors.py) | none | not rerun | **struck** | Unbound. Two bands are given for the same quantity. |
| Cameron | "extra heads do not beat perfect doubling" (report line 29) | none | none | n/a | **struck** | Unbound. It silently contradicts Foreman's bound 0.3982. |
| Cameron S0 | f-lite exact to 2.899e-06 against dense smprime | S0_exactness_bar | stub_exactness.log 01:23:00, board 5158 | exit 0; log byte-identical; JSON 81/81 equal; stub exit 1 | clean | |
| Cameron S0 | cost bar at 11.2M (≥ 43,573 tok/s, ≤ 4 GiB) | S0_cost_bar_11M | stub_cost.log 01:24:31 (OOM), board 5160 | exit 0; 55,692 tok/s, 1.63 GiB; stub OOM exit 1 | clean | The GREEN (01:30:49) predates the shipped flite.py (01:34:02). The rerun used the shipped kernel. |
| Cameron S0 | 0.92x the twin at 11.2M | field (measure.py all) | n/a (timing) | twin 59,671 and f-lite 55,692 → 0.933x | clean | test_cost divides by a hard-coded twin of 65,359. |
| Cameron S0 | peak 1.63 GiB | S0_cost_bar_11M | as above | 1.63 GiB | clean | |
| Cameron S0 | HF patch parity still RED at 6.690e-05 | S0_hf_patch_parity | board 5163 (stub 1.370) | two runs 6.690e-05, exit 1; stub 1.370 | clean | The fp64 split 5.46e-05 / 1.23e-05 is printed by no test. |
| Cameron S0 | 26M 0.927x, 2.92 GiB; 725k 9.7x the family | none | none | not rerun | **struck** | Only 11.2M is barred. |
| Cameron S0 | kill: fp64 accumulation makes it worse (1.63e-04) | none | none | code reverted | **struck** | unbound |
| Foreman | doubling sim 0.06640625/0.25390625/0.985595703125 | foreman.k0.doubling_sim | board 5190, ts 01:48:36 (results.json missing) | orig and regenerated exit 0; results.json byte-identical (0/879 leaves differ) | clean | |
| Foreman | no local leak (0.0566/0.0560/0.0625) | foreman.k0.no_local_leak | board 5191 | identical | clean | The report says the bar is 0.0763; the code computes 0.0756. |
| Foreman | hybrid beats 2^L at W = 4096 (1.0 > 0.5) | foreman.k0.hybrid_beats_2L | board 5192 | identical | clean | |
| Foreman | counter does not fire at R1 width (anchored 0.1934 ≤ 0.5) | foreman.k0.hybrid_at_R1_width | board 5193 | identical | clean | The centered 0.3276 is an unbarred field. |
| Foreman | 27 real softmax heads, 0 mismatches, 0.398193/0.193390, margin 0.992 | foreman.k0.softmax_heads_exact | board 01:55:05; bar file 01:55:05 < shortcut3.py 01:55:33 | identical; results3.json byte-identical | clean | This is a reproduction bar for the idealized 0.3982. |
| Foreman | C2 scores 0.3982 at R1 width vs pinned 0.2539 | softmax_heads_exact + doubling_sim | as above | identical | clean | |
| Foreman | R-SCALE confound: centered slope 1.2207 > 1.2 | foreman.k0.rscale_width_confound | board 01:53:27; bar file 01:53:19 < shortcut2.py 01:53:55 | identical; results2.json byte-identical | clean | margin 0.021 |
| Foreman | dilated bed ratio 1.2202 ≤ 1.25 | foreman.k0.dilated_bed | board 01:53:27 | identical | clean | margin 0.030 |
| Foreman | id-range gap: own 0.9375, parent 0.9374 | foreman.k0.id_range_gap | declared "not yet run"; no failing run | exit 0, same numbers | **struck** | The RED was never observed. The own-id half is arithmetic and could not fail. |
| Foreman | bed_k hybrid 0.3181 > 0.068 (n = 4096) | foreman.k0.bedk_best_response | declared stub; no failing run | exit 0, 0.3181; JSON byte-identical | **struck** | The RED was never observed. |
| Foreman | counter fires at 48 slots (centered W48 0.5596) | none | none | field reproduces 0.5595549738 | **struck** | No bar reads it. |
| Foreman | the shortcut relies on ids = positions | none | none | n/a | **struck** | unbound |
| Foreman | bed_k with random ids removes it | none | none | n/a | **struck** | Unbound; Foreman's own OPEN says it is unmeasured. |
| Foreman | no lower bound covers this bed | none (literature) | none | n/a | **struck** | The fetched sources are not saved in the lane. |
| Chase | fs5 unmodified RED 3.68e-5 | chase.k0.fwd_softmax_S1024_vs_f64 | red_stub.log 01:43:55, board 5173 | 3.6769499861787216e-05 verbatim; exit 1 | clean | |
| Chase | naive sparsemax RED 1.78e-5 | chase.k0.fwd_sparsemax_S1024_vs_f64 | board 5174 | verbatim | clean | |
| Chase | kernel-lse adjoint RED dq 1.14e-3, dk 1.05e-3 | chase.k0.fused_bwd_S1024_vs_f64 | board 5177 | verbatim | clean | |
| Chase | fs5c 3.04e-6 / 4.08e-6 | chase.k0.fwd_softmax_S1024_vs_f64_fs5c | red_stub2.log 01:51:18, board 5203 | verbatim; stub rerun RED | clean | |
| Chase | fusedc bwd dq 2.95e-6, dk 9.74e-6, dv 2.15e-7 | chase.k0.fusedc_bwd_S1024_vs_f64 | board 5204 stub; 5216 v1 RED | verbatim; stub rerun RED | clean | The bar stayed at 1e-4. |
| Chase | densec 6.20e-6 / 5.80e-6 | chase.k0.fwd_sparsemax_S1024_vs_f64_pivot | red_stub3.log, board 5224 | verbatim | clean | |
| Chase | graphed = eager, bitwise | chase.k0.fusedcg_matches_fusedc_S1024 | red_stub4.log, board 5233 | 0.0 on 8 tensors | clean | |
| Chase | gradchecks pass (softmax, sparsemax) | chase.k0.gradcheck_* | board 5175/5176 | identical (CPU); the untransposed mutant goes RED | clean | |
| Chase | blocked-adjoint gradcheck passes; mutants fail | chase.k0.gradcheck_blocked_adjoint_S64_f64 + diag_mutant.py | red_stub6.log 01:59:25, board 5253 | GREEN; dv×1.01 and no-push mutants both RED | clean | |
| Chase | Y1/Y2 closed form, pinned, tail instrument | chase.k0.y1_rebuild / y2_* | board 5178–5180, 5248 | CPU lines byte-identical | clean | |
| Chase | γ·ρ(W) < 1 | chase.k0.gamma_rho_lt_1 | board 5181 | 0.99 | **struck** | Could not fail (W_00 = 1). Chase self-flagged it, but the 13-GREEN tally counts it. |
| Chase | 16 bars: 13 GREEN, 3 RED, exit 1 | run_final.log | stubs as above | full GPU rerun: RED/GREEN/SUMMARY identical | clean | 12 of the 13 GREENs carry evidence. |
| Chase | cost 2.27x / 2.26x | chase.k0.cost_report (presence only) | red_stub.log `{}`, board 5182 | cost.py copy, idle card: 2.30x / 2.26x; eager 2.88 / 2.42 (reported 4.95 / 3.60) | clean | Reported, not scored. |
| Chase | bf16 rowsum 19.3% / 21.2%, gain 2.53 / 1.91 | none (breaks.py B4) | none | not rerun | **struck** | unbound |
| Chase | BOS share 51–99% at row 1023, γ .999 | none (breaks.py B1) | none | not rerun | **struck** | unbound |
| Chase | tail mass m > 0 on 16/16 content heads | none (B2t rows) | none | not rerun | **struck** | The y2_tail bar covers only the Y2 head. |
| Chase | log-kernel m < 0 on 7/8 heads | none (B2 rows) | none | not rerun | **struck** | unbound |
| Chase | unit self-weights 1/5/17; sparsemax 5–12, 38–94% one-hot | none (B3 rows) | none | not rerun | **struck** | unbound |
| Chase | fp32 error grows with ln n (1.02e-5 / 1.58e-5) | none (cost.py field) | none | the cost rerun reproduces it | **struck** | No bar at S4096/8192. |
| Chase | densec 9.41x, 3.8 GiB | none | none | not rerun | **struck** | Unbound; 3,792.6 MiB is 3.70 GiB. |
| Chase | the fs5 miss is the pivot, not the kernel (1.8165e-4) | none (diag_fwd.py) | none | no saved output | **struck** | unbound |
| Wilson | selftest 8/8 on CPU | train_ladder.py --selftest | none | 8/8 PASS, exit 0 | **struck** | No RED. The param-count checks could not fail. |
| Wilson | default-mode resume RED: 262/400, first at step 139, max 2.22e-4 | K0_harness_resume_gpu | board 5245 stub; test_resume.py 01:57:22 unchanged | test_resume on logs: same fails, exit 1 (not rerun on GPU) | clean | Evals also differ bitwise. |
| Wilson | `--deterministic` resume GREEN, val 6.742448 | K0_harness_resume_gpu_deterministic | board 5286 stub (before GREEN 5295) | GPU rerun GREEN, val 6.742447757720948; bitwise equal to Wilson's det_R0_A across sessions | clean | |
| Wilson | deterministic mode costs 14.4% (82,501 vs 96,384) | field of the bound pair | n/a (timing) | idle-card rerun 81,821 tok/s | clean | See process note 2. |
| Wilson | MFU table, 14/14 GREEN | K0_mfu_measured | board 5239 stub, appended before 01:58:31 | stub RED; 14 logs reproduce every row | **struck** | test_mfu.py was edited at 01:59:21, after its RED and after the first sweep runs. The board says 4 red runs; there are 3. |
| Wilson | manifest shard hashes (val_000000, train_000000, train_000017) | re-hash | n/a | 3/3 sha256 and bytes match | clean | The other 28 shards were not re-hashed. |

## Counts

**63 audited, 24 struck.**

| fellow | audited | struck |
|---|---|---|
| Cameron K0 | 13 | 5 |
| Cameron S0 | 7 | 2 |
| Foreman K0 | 14 | 6 |
| Pinned instances (cited Foreman) | 2 | 0 |
| Chase K0 | 21 | 9 |
| Wilson K0 | 6 | 2 |

## Same quantity, two numbers (reported, not picked)

- **L=6 best response on the pinned bed, all tokens.**
  - Cameron: 0.2988, chance-credited doubling (`cameron/ceiling_credit.json`, bound).
  - Foreman: 0.3982, C2 hybrid (`foreman/results3.json`, bound).
  - Cameron's report line 29, "Extra heads do not beat this", is struck. The report never mentions C2.
- **Beyond-2^L best response on the K1 bed.**
  - Cameron's report: 0.041–0.063. His board event 5238: 0.058–0.068. Source: `cameron/k1_floors.json`, unbound.
  - Foreman, at n = 4096 with an assumed positional channel: 0.318 (0.302–0.342) centered W26, 0.187 anchored, 0.560 centered W48 (`foreman/results_bedk.json`; its bar is struck).
  - Foreman states the condition. Cameron is silent on it.
- **Doubling ceiling at L=6.** wald.out gives 0.2539; Cameron and Foreman both give 0.25390625. These agree.
- **0.5596 names two different quantities.**
  - Cameron: the centered resolvent at s=8, n=4096, 2292/4096.
  - Foreman: centered W48 at depth > 64, 1710/3056.
  - This is not a contradiction.
- **γ.** Cameron keeps γ = 0.999 on the chain bed with hand-set heads. Chase's (struck) BOS finding concerns ALiBi heads. The two are not the same quantity.

## Process notes

1. **Chase:** "Nurses dispatched: 0". Recorded here as a process note, not a strike.
2. **The Inspector's GPU nurse broke the wait rule.**
   - It started G1 (Chase's full bar file, about 02:16:55–02:17:58) while the card showed 7,481 MiB at 97%. That load was Wilson's `real_R1`, which ran 02:16:45–02:17:44.
   - Wilson's excluded R1 b8 slowdown (55k → 21k tok/s after step ~40) coincides with G1. The "other lane" he suspected was most likely this Inspector nurse.
   - The same nurse's S0 reruns (about 02:18–02:19:45) overlapped his deterministic pair. His per-step rate there stays at 80.7k–83.4k tok/s, and the idle-card rerun gives 81,821.
   - The nurse also reported exit 0 for RED runs. The Inspector re-checked those exit codes directly: 1, 1, 1.
3. **Lane overlap on the card.**
   - Wilson's MFU sweep window (02:00:07–02:03:05) overlaps Chase's cost_sparse.py (02:00:57–02:01:29) and run_final (ending 02:02:08), by file times.
   - The polls gate only the start of each run.
   - Wilson's R2 b8 "memory pressure" rows (0.236 / 0.214) and Chase's 9.41x are therefore possibly contended.
4. **Header clock times run late in both lanes.** Cameron's test headers run 7 s to 6 min 47 s late; Foreman's run up to 2 min 41 s late (test_shortcut2: header 01:56, mtime 01:53:19). File mtimes and board order were used as the record throughout.
5. **Two DATA_MANIFEST.json files exist.** One is in the repo at `tests/foreman/phase_k/` (10,823 B, 30 train entries). The other is in `C:/Users/seal/datasets/fineweb_edu/` (6,050 B).
