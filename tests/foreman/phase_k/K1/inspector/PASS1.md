# Inspector, Phase K it.K1, pass 1 (periodic, fellows still working)

45 claims audited, 13 struck. Each claim has one `audit` line in `house-events.jsonl`, after the `dispatch` line "K1 pass 1". The cut is board line 5459. Events after that line belong to the next pass.

Reruns ran on copies in `SP/phase_k/K1/inspector/rerun/`, and nothing was written into any fellow's directory. Python ran with `PYTHONDONTWRITEBYTECODE=1`. Chase's bars ran with `CHASE_BOARD=0`; the board line count was 5458 before and 5458 after. Wilson's stub temp directory was redirected into the inspector folder.

**No GPU rerun was needed**, and the lock was never taken. Every GREEN on the board re-runs on CPU:
- K1.F, Foreman's hunt, Chase's gradcheck and R-RANGE synthetics are CPU work.
- Wilson's MFU bar reads the sweep logs, not the card.

Nurses: N1 (haiku) checked Foreman's sources. N2 (sonnet) mapped every Foreman number to a bar, a field, or nothing.

## Table

| fellow | claim | test | RED seen (where, when) | re-run status / number | verdict | why |
|---|---|---|---|---|---|---|
| Cameron | K1.F machinery | cameron.k1.f_machinery | red_test_k1f.log 02:42:43, board 5359; test mtime 02:42:12 | exit 0; k1f.json and stdout byte-identical; stub rerun RED, identical to red log | clean | |
| Cameron | bed_k not void at R0: C2 best-window credited beyond 2^L is 0.0839/0.0696/0.0691 at L4 and 0.4086/0.1722/0.1131 at L7 | cameron.k1.f_not_void | as above, board 5360 | identical | clean | |
| Cameron | (a_L) line = the C2 values; tightest cell L7 n4096 | cameron.k1.f_line | as above, board 5361 | identical | clean | |
| Cameron | "crosses 0.5 at W=16 centered (0.601)" | none | none | field reproduces 0.60125 | **struck** | It sits in `meta.unbarred_sweep`, and no bar reads it. |
| Wilson | MFU c1024 b8 0.134 (93,243 tok/s, 5.63 GiB); c4096 b2 0.144 (83,424 tok/s, 5.69 GiB) | wilson.k1.mfu | red_stub_mfu.log 02:43:22 (17/17 checks fire); test mtime 02:43:20; sweep session starts 02:43:58 | on-log rerun exit 0; recomputed 0.1342 / 0.1445; stub RED identical | clean | |
| Wilson | MFU c1024 b4 0.132; c4096 b1 0.127 | wilson.k1.mfu (fit_clean) | as above | rows reproduce | **struck** | fit_clean could not fail on these rows. gpujob samples foreign PIDs first at 30 s, and these jobs ran 23.9 s and 28.7 s. jobs.jsonl also records an unidentified PID 8944 on the card at the end of c4096_b1, and no check reads that field. |
| Wilson | c1024 b16 and c4096 b4 do not fit (abort at 11.09 / 11.15 GiB, rc 3) | wilson.k1.mfu (nofit_*) | as above | rc 3 plus memory abort record in both logs | clean | |
| Wilson | bf16 peak 32.31 TF, median of 5 reps, measured before the sweep | wilson.k1.mfu (peak_*) | as above | peak_pre.json matches: 25.80/32.43/32.84/32.31/32.14 | clean | |
| Wilson | post-sweep peak 29.12 TF; SM clock 2685 → 2280 MHz at 87–88 C | none | none | peak_post.json telemetry holds it | **struck** | No check reads peak_post. |
| Wilson | "against 26.77 the same logs give 0.162 / 0.174" | none | none | recomputed 0.1620 / 0.1744 | **struck** | `mfu_vs_26_77` is a field that no check reads. |
| Chase | hook gradcheck passes; the three 1.01× adjoint mutants fail | chase.k1.hook_gradcheck_S64_f64 | red_stub.log 02:48:13 on stub sha 6dd7c8 (mutants pass); test mtime 02:46:20 | on hook 9d0274: GREEN, exit 0; stub RED, exit 1 | clean | Board GREEN #1 came from run1 on 763d05 (aborted). GREEN #2 is run2, whose log header gives 9d0274, the current file. |
| Chase | band-mass instrument matches ln(g + (1−g)e^λ) on the Y2 head, 6 cells | chase.k1.rrange_band_y2 | red_stub_ckpt.log 02:55:54; ckpt_bars.sha256 02:55:51 logs test sha 66f039 (unchanged); test mtime 02:55:02 | on 9d0274: byte-identical GREEN lines, exit 0 | clean | The GREEN ran at 02:56:14, before 9d0274 was written (02:58:43). instr.py takes only GAMMA and C from the hook. |
| Chase | same instrument with a BOS sink, g' = g(1−σ) | chase.k1.rrange_band_sink | as above | byte-identical | clean | Same timing as band_y2. |
| Chase | K0 tail instrument fooled by the sink (ratio 4.4e-6 … 3.1e-10) | chase.k1.rrange_tail_sink_fooled | as above | byte-identical | clean | Same timing as band_y2. |
| Foreman | space hybrid = shortcut hybrid; Cameron's cell 0.4085876 | foreman.k1.hunt_machinery | red_test_hunt.log 02:56:29 on stub; test mtime 02:55:21 | hunt.json byte-identical; exit 0; stub RED identical | clean | |
| Foreman | id-space K0 mechanism collapses to doubling on bed_k (0.0645/0.0674 vs 0.0644/0.0661; hit rate 3.1e-4/2.8e-4); wald 0.398193 | foreman.k1.ids_eq_pos | as above | identical | clean | |
| Foreman | content addressing resolves exactly depth ≤ 2^L, L = 1..5, with and without overlap | foreman.k1.content_closure | as above | identical | clean | |
| Foreman | no-oracle windows ≤ 0.25 at R0 (max 0.1246) | foreman.k1.no_oracle_ceiling | as above | identical | clean | |
| Foreman | pairwise probe detects a planted leak (1.0) | foreman.k1.leak_probe_detects | red_test_leak.log 02:56:29; test mtime 02:55:32 | leak.json byte-identical | clean | |
| Foreman | no leak: 0.0633 ≤ 0.0667, 0.0657 ≤ 0.0682; ids distinct | foreman.k1.no_leak | as above | identical | clean | |
| Foreman | HPD at J=1 is the anchored window; HPD mass ≥ centered mass | foreman.k1.hpd_machinery | red_test_k1b.log 03:00:35; test mtime 03:00:30; k1b.py 03:01:16 | k1b.json byte-identical | clean | |
| Foreman | HPD 0.5642 > K1.F line + 0.01 | foreman.k1.hpd_beats_line | as above | identical | clean | |
| Foreman | clause 2 fires at (L7, n4096): 0.5642 > 0.5; the other 5 cells hold (0.2290, 0.1359; L4 ≤ 0.0940) | foreman.k1.hpd_not_void | as above; stays RED on real data | same failing row, exit 1 | clean | Reported as RED. |
| Foreman | an independent per-token reference: 0 mismatches, mean 0.5528 | foreman.k1.hpd_reference | red_test_k1f.log 03:12:01; test mtime 03:11:44 | hpd_ref.json byte-identical | clean | |
| Foreman | no-oracle local window W64 = 0.6393 on fresh seeds | foreman.k1.local_wide_fires | red_test_k1b.log 03:00:35 | identical | clean | |
| Foreman | smallest crossing W = 50 | foreman.k1.local_crossing | as above | identical | clean | |
| Foreman | deep-bed floor 0.2141/0.2225/0.2291/0.2326/0.1345/0.0950; local ≤ 0.0846 | foreman.k1.deep_bed_floor | red_test_k1c.log 03:03:43; test mtime 03:03:38 | k1c.json byte-identical | clean | |
| Foreman | deep_bed_is_deeper RED at ρ=64 (0.042…0.015); passes at ρ=128/256 (0.499/0.750) | foreman.k1.deep_bed_is_deeper | as above | identical, exit 1 | clean | Reported as RED. |
| Foreman | channel: slope 1/256; fp32 pair error 1.1e-4; bf16 1.98 > 1.0 (RED) | foreman.k1.alibi_position_channel | red_test_poschan.log 02:56:29; test mtime 02:55:40, unchanged | poschan.json byte-identical, exit 1 | clean | The RED is reported as RED, and the bar was not moved. |
| Foreman | harness channel coarse: 39.79 positions | foreman.k1.harness_channel_coarse | red_test_k1c.log 03:03:43, before poschan_harness.json 03:04:49 | byte-identical | clean | |
| Foreman | R1 rule holds: 0.4179/0.2158/0.1314/0.4334 | foreman.k1.r1_rule_holds | red_test_k1d.log 03:08:36; test mtime 03:08:24 | k1d.json byte-identical | clean | |
| Foreman | ρ=32 fires: 0.9322 | foreman.k1.r1_short_fires | as above | identical | clean | |
| Foreman | HPD on the twin's train beds 0.5616 > 0.5 | foreman.k1.train_dist_voidable | red_test_k1e.log 03:09:55; test mtime 03:09:43 | k1e.json byte-identical | clean | |
| Foreman | no-oracle local on train beds 0.1587 ≤ 0.25 | foreman.k1.train_dist_no_oracle | as above | identical | clean | |
| Foreman | 8 quotes verbatim from 4 papers | foreman.k1.sources_verbatim | red_test_sources.log 02:56:30; test mtime 02:56:15 | GREEN, exit 0 | clean | N1: PDF sha256 matches MANIFEST for 4/4; an independent pdftotext extraction contains all 8 quotes; page-1 identities match the arXiv ids. |
| Foreman | HPD beats K1.F's best window at every cell | hpd_not_void + f_not_void rows | as above | both sides reproduce | clean | Derived from two sets of bound rows. |
| Foreman | "max depth ~2.2·2^L"; "reach gain ~1.5×" | none | none | n/a | **struck** | N2: in no output file; no bar. |
| Foreman | "the HPD floor is a function of n/2^L" | none | none | n/a | **struck** | All four rows sit at one ratio, and no bar tests the dependence. |
| Foreman | "max depth ≈ n/16 = 4·2^L"; "depth ≈ pos/16" | none | none | n/a | **struck** | N2: in no output file; no bar. |
| Foreman | harness bias cast to bf16, "staircase of ~t/128 positions" | none | none | n/a | **struck** | Only 39.79 is barred. |
| Foreman | "ρ must grow with rung; R2 (L+3=11) cannot meet ρ ≥ 16 at V = 32,768" | none | none | n/a | **struck** | Arithmetic on a design constant; no bar. |
| Foreman (report) | HPD catches the grandparent's parent 48% vs 26% centered | none | none | n/a | **struck** | N2: in no output file. |
| Foreman (report) | "each of its 4 checked beds is above 0.5 (0.529–0.573)" | none | none | ref_accs 0.5728/0.5576/0.5285/0.5522 | **struck** | test_k1f bars only the 4-bed mean. |
| Foreman (report) | "no lower bound covers this bed" (K0 struck, re-asserted) | foreman.k1.sources_verbatim (quotes only) | see sources_verbatim | quotes verbatim | **struck** | The K0 strike stands. The bar binds the quotes, not the coverage reading. Hdp ≤ 1.0001 and depth ≈ N/16 have no bar. |
| Foreman (report) | K0 struck "shortcut relies on ids = positions" declared "bound, and reworded" | n/a | n/a | n/a | **struck** | The K0 strike stands, with no appeal. The reworded K1 claims are audited on their own rows (ids_eq_pos, local_wide_fires), both clean. |

## Counts

**45 audited, 13 struck.**

| fellow | audited | struck |
|---|---|---|
| Cameron | 4 | 1 |
| Wilson | 6 | 3 |
| Chase | 4 | 0 |
| Foreman | 31 | 9 |

## Same quantity, two numbers (reported, not picked)

The quantity is the clause-2 best-window C2 at (L7, n4096). The pricing is R0 (k3 W≤10), with the oracle, credited, on Cameron's seeds `[12, n, k]`.
- **Cameron: 0.4086.** Centered window, the best of anchored and centered.
- **Foreman: 0.5642.** HPD window.
- Both are bound. They are different window families.
- Cameron's `f_not_void` is GREEN, and Foreman's `hpd_not_void` is RED at the same cell.

## Checks with no hits

- **`status` vs `state`:** all this round's test events use `status`.
- **`hash()` seeding:** none in any lane file. Seeds are `default_rng` lists or `torch.Generator`.
- **Exit codes:** every bar file exits 1 on a failing row. This was verified on the stub reruns and on the RED reruns of test_k1b, test_k1c and test_poschan.
- **Bars edited after their RED:** no test file has an mtime after its RED log.
- **RECORD_K.md:** unchanged since 02:35:59.
- **The 9 clauses:** no fellow edited one. New constructions went under new test names (hpd_*, deep_bed_*, r1_*, train_dist_*).

## Process notes

1. **Chase's lane wrote bytecode into the repo.** The repo copies are gitignored, and no claim depends on them. The files are:
   - `tests/foreman/phase_k/K0/wilson/__pycache__/train_ladder.cpython-311.pyc` (02:46:36). resolvent_hook_stub.py loads K0 train_ladder through importlib.
   - `K0/chase/__pycache__/resolvent.cpython-311.pyc` (02:47:44).
   - `rrange.cpython-311.pyc` (02:56:07), from instr.py.
   - The other lanes set `dont_write_bytecode`.
2. **Unidentified PID 8944.** A compute process with this PID was on the card at 02:46:56, inside Wilson's MFU lock hold (02:44:25–02:47:23). It was gone by 02:47:15. It is not the process of any fellow that was live then.
3. **One lane holds the card.** Wilson's ward chain held the lock from 02:48:19 to about 03:18, then again from 03:18:46 (a_L_s2). Cameron, Chase's run2 and Foreman's twin are queued on the lock. At 03:09 and 03:11, nvidia-smi listed only Wilson's process.
4. **Wilson's harness.** `train_ladder_k1.py` (4c00be…) differs from K0 `train_ladder.py` (d524b1…) in logging and `--eval_seed` only.
5. **Foreman's header clock times were estimates.** He reports they run 1–8 min late. Mtimes and board order are the record.

## Lanes still in flight

- **Cameron, R-DEPTH at R0.**
  - All 7 `cameron.k1.rdepth_*` bars are RED only (stub 02:48:17).
  - The pilot never got the lock. `jobs_twins.json` (03:18:05) is now queued.
  - The (a_L) arms are rdepth.py models built on the train_ladder copy's AlibiAttention (sha d524b1…), not `train_ladder.py --attn alibi` itself.
- **Chase, hook bars.**
  - run2 (pid 21408, hook 9d0274) is waiting on the lock for hook_parity_S1024, hook_train_ladder_R0_50 and hook_deterministic_R0_50.
  - Each bar calls `load_hook()` at its own start. The sha in the run2 header is read once, at process start.
- **Chase, checkpoint bars.** bf16_leak ×3, c6 ×2, c7 ×2, rrange_fR ×2 and rrange_aL_slope are RED only. They wait on finished checkpoints and the lock.
- **Wilson, R0 LM ward.**
  - `wilson.k1.ward` is RED only (stub 02:45:10).
  - a_L_s1 landed (board: val 4.9970, "not yet checked by wilson.k1.ward"). It is not audited in this pass.
  - a_L_s2 is running under the lock.
- **Foreman, trained twin.** The 7 `foreman.k1.twin_*` bars are RED only (stub 02:56:30). twin.py (pid 8264) is waiting on the lock, and twin_L4.log and twin_L5.log are empty.
