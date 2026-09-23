# Inspector, Phase K it.K1, pass 7 (final, pre-prognosis)

106 claims audited this pass, 35 struck. Board: `dispatch` line 5941, nurse lines 5942–5945 (N9–N12, sonnet), audit lines 5946–6051, `done` after them. The delta covers board lines 5917–5940 (Cameron; Chase, Foreman and Wilson posted nothing after 5916), every lane file modified after 07:43 (Cameron's lane only), and the four final reports.

## Priority items

| item | result |
|---|---|
| K1-b ordering | **RED and registration precede every far_fR_ga launch.** The bar was written at 07:50:26. Its stub RED ran at 07:50:34, and board 5917–5921 were appended in the same shell command. The board held 5922 lines at 07:51:10. far_fR_ga_s0 launched at 07:53:45 (stdout 07:53:45.2, lock 07:53:45.5, run dir 07:53:51); far_fR_ga_s1 at 08:08:24. The stub rerun is identical, 14 failing rows. |
| K1-b schedule | **Registered = run.** Both argv and result.json configs carry `0.5:2000,0.9:4000,0.99:6000,0.999`. `gamma_at` gives 0.5 / 0.9 / 0.99 / 0.999 on steps 1–1999 / 2000–3999 / 4000–5999 / 6000–8000. The final eval runs at the hook's 0.999. Mid-run probes run at the scheduled gamma, and no bar reads them. |
| K1-b fairness | **No non-resolvent knob differs.** lr, warmup, steps, tok, train_ns, probe_every, eval_ns, eval_beds, emb, par_init, bed, dff and layers equal far_fR_s{0,1} and all 8 twin runs. gamma_wrap keeps the hook's fp32, autocast-off path. |
| Cameron bars | **All reproduce byte-identical** on CPU copies: test_rdepth_far, test_rdepth_far10 and test_rdepth_far10_ga (exit 1), and the four floor bars (exit 0). Values: learnable RED 0.2026 / 0.2354; ga_learnable RED on s1 at 0.2011; ga_prediction RED at 0.0784 (depth > 160, 16k); aL7 far band 0.0000 against a bar of 0.1330. Gates: aL7 0.9948 / 0.9850, aL4 0.3465 / 0.4008, ass4 0.3986 / 0.4375, aloop5 0.5883 / 0.3488, fR 0.3406 / 0.3951, fR_ga 0.9963 / 0.3387. |
| ×3 logit-scale probe | **Struck.** The verdict's "Why" section uses it as the reason. It was printed to a terminal only and no bar reads it. Its ×1 value, 0.2479, is from 2 beds, not the bar's 0.2855 on 22,464 tokens. |
| Chase bars | **Confirmed.** clause6_v2 GREEN, c7_bos GREEN at 0.186905, rrange_fR_dressed RED (rerun identical), rrange_aL_slope RED at 36/84 (7/8/6/5/8/2) with 0/16 out of band on LM. The predicates were re-applied to ckpt_rows_v2.jsonl on CPU; pass 5's GPU rerun was already identical. |
| Wilson bars | **Confirmed.** wilson.k1.mfu GREEN on both sessions, and wilson.k1.ward RED with done_clean only (1/17 fired, 2 rows). The final val losses are 4.997022 / 4.984507 / 5.016564 / 5.008180. Every rerun is identical to its log. |

## Cameron: 64 audited, 18 struck

Board 5917–5939: 35 audited, 8 struck.
- 5924: held-out depth ≤ 16 is 0.9941. The gate reads 0.9963. Withdrawn at 5925.
- 5924: acc at 4k / 8k is 0.5041 / 0.2792. These are result.json fields that no bar reads.
- 5924: parent-pointer diagnostics (hop 1 on 4,076–4,078 rows, 0.988; 1.000 / 0.769 / 0.08 by depth). Terminal output only; no check.
- 5924: ×3 logit scale lifts 0.2479 → 0.9113. No bar, and the ×1 value is from 2 beds.
- 5924: the wall is softmax leakage that depth-32 training cannot price. Rests on the struck diagnostics (pass 6, 5905).
- 5930: "plus the seed-0 pilot" counts a pilot as a seed. The pilot's learning was struck at pass 6 (5904).
- 5931: aL7 0.071 / 0.032 / 0.006 at depths 33–256, and ≤ 0.0014 on far10. No check reads these bands; aL7's far10 row reads 0.0000.
- 5939: mechanism, parent at 0.988, ×3 lift, "so the wall is softmax leakage". Repeats the 5924 strikes.

CAMERON_REPORT.md (09:38:48): 29 audited, 10 struck.
- Verdict: "the counter stays quiet". counter_quiet is RED, and all 6 aL4 rows are UNREAD.
- "Why (f_R) misses the band": exact parent pointer, leakage rather than the operator. No file, no check; struck at pass 6.
- The logit-scale probe (×1.5 / ×2 / ×3 / ×5; 0.2479 → 0.9113; depth 513+ at 0.55–0.69). Used as the verdict's reason; no bar.
- Mechanism probe on far_fR_s0: 1.4–1.7% of mass on the parent; depth ≤ 4 comes from ALiBi. Struck at pass 6 (5891).
- Mechanism probe on the pilot: grandparent pointer; 0.1417 → 0.8764. Struck at pass 6 (5904 / 5905).
- "Retired unrun": the band-96/896 rows of test_rdepth_far.py. They ran at 08:59 and 09:36, and 8 aL7 rows print PASS.
- Kill 4: training at depth ≤ 32 cannot price a 1–2% per-hop leak. No check measures a leak.
- Kill 6: the learned-embedding run never learned depth 2; kp_aL7_s0 scored 0.1390. Struck at pass 3 (5605 / 5624) and pass 4 (5739).
- OPEN: aL7 scores 0.071 / 0.000 at depth 33–64, ≤ 0.0014 on the far band, and emits a "wrong id rather than a guess". No check reads any of these.
- Standing state: "the twins do not shortcut the band". Only aL7 is read; every aL4, ass4 and aloop5 row is UNREAD.

## Chase: 12 audited, 4 struck

CHASE_REPORT.md (06:23:55); no board events after 5916.
- Verdict: "the 1e-5 line sits below the fp32 floor on 3 of the 4 cells". The floor is above 1e-5 on all 4 cells (the fourth is 1.438e-5); 3/4 is the RED count.
- Bars-table note on hook_parity: "autocast on = off bitwise". Struck at pass 3 (5659); no check compares the two.
- Kills table: the K0 backward's cause (u ≈ 2e4·gx cancels in fp32; dq 2.7e-4 → 1.4e-5). Struck at pass 3 (5659); diagnostic only.
- Kills table: "content sets their range" for the chain-bed a_L pilots. Struck at pass 5 (5815).

## Foreman: 13 audited, 6 struck

FOREMAN_REPORT.md (07:03:10); no board events after 5916.
- "131M tokens (4× an arm)". Struck at pass 5 (5793): it is 2×.
- "Below chance because the twin emits a wrong id rather than a guess". Struck at pass 6 (5874).
- Unbarred reading: "even at 4× budget … depth ≈ 8–12 … no sign of a budget-driven shortcut". Struck at pass 6 (5874); depth 8–12 has no source.
- w62a row: "782k / 702k alias violations". Struck at pass 5 (5747): the row prints 783,878.
- w30t row and kill 5: "all at the near-alias Δ = 8146 of period √8192". No file records mismatch positions (N12).
- Kill 5: "a 1e-3 leak in x2 × 65536 gave 64-position errors; the flat-top bump took 587 → 3". The diagnosis was struck at pass 4 (5729), and the fix's bar is RED.

## Wilson: 17 audited, 7 struck

WILSON_REPORT.md (05:59:58); no board events after 5916.
- MFU against 26.77 TF (0.162 / 0.174; the vs-26.77 column). Struck at pass 1: the field is read by no check.
- MFU rows (a_L) c1024 b4 0.132 and c4096 b1 0.127. Struck at pass 1: fit_clean could not fail on them.
- Mean Δ +0.021608, |Δ1 − Δ2| 0.004130, and the seed spreads 0.012515 / 0.008385. Struck at pass 5 (5799).
- §2: post-sweep peaks 29.12 / 27.78 TF, their telemetry, drift 0.901 / 0.872, "rep 1 lowest at 1890 MHz", and MFU against the post median. Struck at passes 1 and 4; no check reads them.
- §6: "throughput fell within every run" (first vs last 1,000 steps). The fields are read by no check.
- §6: the mid-run f_R_s1 sample (84–86 °C, 7,705 / 8,188 MiB, PID 26416). No file records it.
- §8: a_L_s2 block medians of 82.8–84.3k against 87.9–92.9k. Struck at pass 5 (5799 / 5801).

## K1 totals

| pass | audited | struck |
|---|---|---|
| 1 | 45 | 13 |
| 2 | 38 | 9 |
| 3 | 34 | 12 |
| 4 | 34 | 5 |
| 5 | 47 | 6 |
| 6 | 36 | 16 |
| 7 (this) | 106 | 35 |
| **it.K1** | **340** | **96** |

| fellow | passes 1–6 | pass 7 | it.K1 |
|---|---|---|---|
| Cameron | 56 / 21 | 64 / 18 | **120 / 39** |
| Chase | 38 / 4 | 12 / 4 | **50 / 8** |
| Foreman | 116 / 27 | 13 / 6 | **129 / 33** |
| Wilson | 24 / 9 | 17 / 7 | **41 / 16** |

Each cell is audited / struck. The passes 1–6 counts equal the Inspector audit lines on the board after line 5354: 234 audited, 61 struck.

## Records

- **Reruns.** All ran on copies in `inspector/rerun7/`, CPU only (`CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`), with `CHASE_BOARD=0`. The Wilson bars ran read-only against the lane directories.
- **What was not run.** No training, no GPU job and no lock. Nothing was written into any lane. `git status` is unchanged, and the only repo writes are board appends.
- **Lock after 07:43** (lockwatch6). Cameron alone held it: 9 jobs, far_aloop5_s0 through far_aloop5_s1, free at 09:36:59.
  - One gap was 169.1 s (07:50:56 → 07:53:45), self-reported at 5923.
  - The other 7 gaps were 180.1 s each.
- **Timing source.** Board lines carry no timestamps. K1-b ordering was fixed from file times and the tool timestamps in Cameron's session transcript.
- **Checkpoints no clause-6 or R-RANGE bar has read.** far_fR_s1, far_fR_ga_s0 and far_fR_ga_s1 landed after Chase's 06:17 read. far_fR_ga_s0 (acc 0.9931) meets the rrange_fR_dressed premise.
- **Two counts off on the board.** Registration 5921 said 8 failing rows; the log says 14 (corrected at 5922). Line 5939 says "12 registered runs + K1-b"; jobs_far_all.json registers 10, so the total is 12 runs.
