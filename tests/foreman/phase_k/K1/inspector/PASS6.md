# Inspector, Phase K it.K1, pass 6 (periodic, delta only)

36 claims audited, 16 struck. The audit lines are board lines 5876–5890, 5892–5900, 5902–5903 and 5906–5915, and `done` follows them. The `dispatch` line "K1 pass 6" is line 5867. The nurse line is 5901 (N8). The cut is line 5915, at 07:43.

**Delta covered:**
- Line 5818 (Foreman, "GPU priority race, mine"), left over from pass 5.
- Every fellow event from 5868 to 5905: Foreman 5868–5875 (done at 5875), and Cameron 5891, 5904 and 5905. Chase and Wilson posted nothing after 5866.
- Lane files in `cameron/` and `foreman/` modified after 06:24.

**How the reruns ran:**
- All reruns ran on copies in `SP/phase_k/K1/inspector/rerun6/`, with `PYTHONDONTWRITEBYTECODE=1` and `CUDA_VISIBLE_DEVICES=""`.
- **Foreman, kp_trap.py.** The producer reran on CPU from 07:00:01 to 07:29:44. It wrote a kp_trap.json byte-identical to the lane's. The bar output is identical on the stub (4 rows) and on the lane json (3 rows), with exit 1 both times.
- **Foreman, test_far_opponent.py.** The bar reran on a copy of runs/long_aL4_s7 (result.json, log.jsonl, ok_*.npz). Its output is identical to run_test_far_opponent.log apart from the path, including the KeyError traceback, with exit 1.
- **Cameron, test_rdepth_far10.py.** The bar ran on copies of far_fR_s0, far_aL4_s0, far_aL7_s0 and pilot_fR_ganneal_s0 (result.json and ok_*.npz). The pilot is outside the bar's paths. The bar is byte-identical to the lane's (sha 58c5bc2d…).
- **No arm or pilot was retrained.** A retrain needs the GPU, and Cameron's far arms hold priority. Cameron's arm claims were audited from the saved result.json and ok arrays through the registered bar.
- **The GPU lock was not taken by the Inspector this pass.**
- **Lane and repo writes.** Nothing was written into any lane directory. `git status` is the same as at session start. The only repo writes are board appends.

**Nurse N8 (sonnet)** mapped 41 numbers in FOREMAN_REPORT.md (07:03:10): 19 bar rows, 20 config or arithmetic, 1 printed-only, 1 with no source.
- **Two of its three mismatches are not taken.** N8 sourced "n = 8192 and 16384 layer-wise, exact at L = 4" to kp_w62.py's L = 7 layer-wise rows. The source is kp_w62a.json: the L = 4 rows have 0 mismatches on 32,768 rows at both n, and the w62a_real_8192 / 16384 L=4 rows PASS.
- **The third is taken as a note.** "Depth ≈ 8–12" has no printed source. N8 reports that the ok arrays hold ≥ 0.89 through depth 13.

## Dispatcher checks

| ruling | result |
|---|---|
| (a) Cameron scores far-band arms only on far10 (> 160 at L4, > 1280 at L7), never on 137–160; the runner re-checks for a newer c_max after acquiring and before each arm | **Holds for the arms. No arm score is posted off far10.** Cameron has not run test_rdepth_far10.py on any arm. The Inspector's run gives 34 failing rows: fR s0 and aL4 s0 are UNREAD at the gate, and the aL7 s0 rows PASS at depth > 1280. The pilot was scored on the wrong band (see below). |
| Pilots are reported as pilots, never as arm results; a pilot's numbers presented as a finding with no bar are struck | **Named and reported as a pilot, but its numbers were presented as a finding.** pilot_fR_ganneal_s0 ran under its jobs name, into runs/pilot_…, which no bar reads. 5904 and 5905 present its numbers as findings: 6 claims were struck across the two lines. |
| Foreman's long_aL4_s7 opponent and far_opp_* bars | **Void at the gate.** far_opp_gate FAILs at 0.818 (bar 0.9), and quiet and chance are VOID. far_opp_plateau never executes: KeyError, struck. The aL7 opponent never ran. |
| Priority after 06:50 (Cameron's far arms → the opponent) | **Holds.** Foreman held the lock from 06:36:19 until 07:02:20; pass 5 recorded this as inversion 2. Cameron's runner took the lock 43 s after the release. After 07:03 only Cameron used the GPU. |
| 3-minute rule | **Holds.** Cameron re-acquired at 198 s, 180.2 s and 180.1 s after each of its own releases. Foreman is done, and the Inspector did not acquire. |

Ruling (a) and pilot details:
- **Guard change.** The runner's guard stopped at 07:03:03, after acquiring the lock and before far_aL7_s0. The old filter matched "far band" in Foreman's 5874, which holds no c_max. At 07:06:15 the guard was narrowed to "c_max", "c max" and "reach factor". It is still called after mkdir and before owner.txt.
- **The pilot's band.** Cameron's 5904 quotes the pilot's "far band depth>160 at 16k = 0.1005 (L4 band4 printed)". rdepth.py's printed band4 is BANDS band4 = depth > 96 (124,864 tokens). Depth > 160 at 16k reads 0.0743 on 120,768 tokens. This is the retired band from pass 5 process note 8, read as far10.
- **The pilot's gamma.** The final eval resets gamma to the hook's 0.999. The mid-run probes run at the scheduled gamma, although the code comment says "eval always at the hook's fixed gamma".

Lock timeline since 06:50 (lockwatch6.log, 1-s poll, read-only):

| time | owner.txt |
|---|---|
| 06:36:20 – 07:02:20.468 | Foreman pid 33648, long_aL4_s7 (result.json 07:02:18) |
| 07:03:03 (sub-second) | Cameron runner pid 30732: acquired, guard STOP, released; no owner.txt |
| 07:06:21.846 – 07:20:44.436 | Cameron pid 15476, far_aL7_s0 (rc 0, 862 s) |
| 07:23:44.613 – 07:37:51.948 | Cameron pid 15476, pilot_fR_ganneal_s0 (rc 0, 847 s) |
| 07:40:52.045 – (held at 07:43) | Cameron pid 15476, far_aloop5_s0 |

## Table

| fellow | claim | test | RED seen (where, when) | re-run status / number | verdict | why |
|---|---|---|---|---|---|---|
| Foreman (5818) | gpu_run2 took the lock at 06:36:19 in Cameron's restart gap; the gate looked for a live process, not a queue | code, lock record | — | owner.txt 06:36:19; the gate greps for jobs_far / chase | clean | |
| Foreman (5818) | step ~6.3k; ends by ~07:00 (45-min timeout); not killed | log, lock record | — | step 6300 at 435 s; ended 07:02:18 | clean | The timeout bound is 07:21:19, so ~07:00 was an estimate. It held within 2m20s. |
| Foreman (5818) | runner 30732 waits; no further Foreman GPU job | process record | — | 30732 acquired at 07:03:03; queue2.log is 0 bytes | clean | |
| Foreman (5868) | w30t_real_8192 RED | foreman.k1.w30t_real_8192 | red_test_kp_trap.log 04:41:40; test mtime 04:41:35 | json byte-identical; L4 3 and L7 6 mismatches; exit 1 | clean | kp_trap.py was born at 04:41:55, after the RED. |
| Foreman (5869) | w62t_real_16384 RED | foreman.k1.w62t_real_16384 | as above | L4 PASS 0.12477 / 30,576; L7 FAIL 24,893 mismatches | clean | |
| Foreman (5870) | far_opp_gate RED | foreman.k1.far_opp_gate | red_test_far_opponent.log 04:57:28; test mtime 04:57:20 | FAIL 0.8179597; identical | clean | |
| Foreman (5871) | far_opp_quiet RED | foreman.k1.far_opp_quiet | as above | aL4 rows VOID via the gate; identical | clean | The aL7 row never prints. |
| Foreman (5872) | far_opp_chance RED | foreman.k1.far_opp_chance | as above | aL4 rows FAIL on the gate; identical | clean | |
| Foreman (5873) | far_opp_plateau RED | foreman.k1.far_opp_plateau | as above (stub rows printed) | row never executes: KeyError '4096' | **struck** | rdepth.py probes evaluate n = 1024 only, so the row cannot be read from any run. |
| Foreman (5874) | item 3 VOID: gate 0.818 < 0.9 | far_opp_gate | as above | identical | clean | |
| Foreman (5874) | recipe; 131M tokens; 1033 s; released 07:02 | config, record | — | train_s 1033.06; FREE at 07:02:20 | clean | |
| Foreman (5874) | depth ≤ 8 ≥ 0.987, 9–16 0.587, 17–32 0.006 | none | none | no source; ok_4096 gives 0.9948 / 0.6484 / 0.0732 | **struck** | No check computes them. |
| Foreman (5874) | loss ~7.5 until 8000, then 1.04 by 12000 | none | none | stdout 7.5479 / 1.0433 | **struck** | No check reads the loss. |
| Foreman (5874) | far band 0.0000 / 0.00002 / 0.00002 | far_opp_quiet / chance | as above | identical | clean | |
| Foreman (5874) | below chance: a wrong id, not a guess | none | none | — | **struck** | No check reads the outputs. |
| Foreman (5874) | plateau not measurable (n = 1024 probes; KeyError); aL7 never queued | code, record | — | reproduced | clean | |
| Foreman (5874) | unbarred reading: 4x budget, below the doubling line, no shortcut | none | none | — | **struck** | 4x was struck at pass 5 (it is 2x). No check reads the doubling line. |
| Cameron (5891) | guard stopped at 07:03 on 5874, no arm started | gpujob log | — | STOP 07:03:03; far_aL7_s0 born 07:06:35 | clean | |
| Cameron (5891) | c_max unchanged; far10 stands | Foreman w62* / kp_trap | as above | the last c_max line is 5792; w62t L4 PASS | clean | |
| Cameron (5891) | guard narrowed; runner restarted | code | — | gpujob.py 07:06:15; restart 07:06:21, guard 5890 | clean | |
| Cameron (5891) | queue reordered: aL7_s0, pilot, aloop5_s0, ass4_s0, then s1 | jobs_far_all3.json | — | 9 jobs, in this order | clean | |
| Cameron (5891) | pilot gamma schedule; eval at 0.999 | config, code | — | argv as stated; final eval at 0.999 | clean | The probes run at the scheduled gamma. |
| Cameron (5891) | fR s0 / aL4 s0 fail L-TRAINED (0.3406 / 0.3465) | test_rdepth_far10 gate | red_test_rdepth_far10.log 04:41 | acc_le16 values identical; rows UNREAD | clean | Cameron has not run the bar. |
| Cameron (5891) | they learn depth ≤ 4 only | none | none | ok_4096: 0–4 0.9938 / 0.9500; 5–8 0.1289 / 0.2188 | **struck** | No check reads it. |
| Cameron (5891) | diag_fr: parent mass 0.014 / 0.017; argmax 1.5% / 1.6% | none | none | — | **struck** | An unbarred diagnostic, with no saved output. |
| Cameron (5891) | depth ≤ 4 came from the ALiBi layers | none | none | — | **struck** | No check attributes accuracy to a layer. |
| Cameron (5904) | the pilot is an unbarred pilot; schedule; eval at 0.999 | config | — | result.json config | clean | |
| Cameron (5904) | pilot learns its depth: 0.9837 vs 0.2026 | none | none | acc 0.98369 | **struck** | Pilot number, no bar. |
| Cameron (5904) | pilot per-depth: ≤ 128, 129–256, 257–512 | none | none | 0.4639 / 0.4546 / 0.4873; 0.0912 / 0.0607 / 0.0714 | **struck** | Pilot number, no bar. |
| Cameron (5904) | pilot depth > 160 at 16k = 0.1005; K1-a prediction would still fail | none | none | the printed band4 is depth > 96; depth > 160 reads 0.0743 | **struck** | Wrong band, and no bar. |
| Cameron (5904) | diag_fr3 mechanism: grandparent pointer; weight 0.991 → 0.956 | none | none | — | **struck** | Unbarred diagnostic. |
| Cameron (5904) | dressed-mass range ~30 steps, ~60–100 depth | none | none | — | **struck** | No check. |
| Cameron (5904) | far_aL7_s0 L-TRAINED 0.9948 | test_rdepth_far10 gate | 04:41 | acc_le32 0.99481; aL7 rows PASS | clean | Not retrained. |
| Cameron (5904) | far_aL7_s0 0.000 beyond depth 32 | none | none | depth > 32: 0.0108 / 0.0050 / 0.0025 | **struck** | No check reads it. The row reading 0.0000 is depth > 1280. |
| Cameron (5905) | diag_fr4 logit-scale sweep: 0.1417 → 0.8754, and so on | none | none | — | **struck** | Pilot-derived numbers presented as a finding, with no bar. |
| Cameron (5905) | the wall is softmax leakage, not the resolvent | none | none | — | **struck** | Rests on the struck sweep. |

## Counts

**36 audited, 16 struck.**

| fellow | audited | struck |
|---|---|---|
| Foreman | 17 | 5 |
| Cameron | 19 | 11 |

## Checks with no hits

- **`status` vs `state`:** every test event in 5867–5915 uses `status`.
- **`hash()` seeding:** none in any lane .py changed after 06:24:
  - cameron: gpujob.py, diag_fr2.py, diag_fr3.py, diag_fr4.py
  - foreman: none
- **New bars after 06:24:** none.
- **Bars edited after their RED:** none. The bars unchanged since their REDs are:
  - test_kp_trap.py (04:41:35)
  - test_far_opponent.py (04:57:20)
  - test_rdepth_far10.py (04:41:22)
- **Exit codes:** every bar read this pass exits 1 on failure.

## Process notes

1. **A guard stopped on a word.** The old guard stopped Cameron's queue on "far band" in a finding with no c_max. It was narrowed 3 min later, and no arm was lost.
2. **A retired band still printed.** rdepth.py still prints band4 as depth > 96 and band7 as depth > 896, and 5904 read that band4 as depth > 160.
3. **A bar row that no run can feed.** far_opp_plateau reads probe keys that rdepth.py never writes, so the bar stops before its aL7 rows. The plateau row passed its RED only on a hand-built stub.
4. **Two copies of one producer.** kp_trap.log interleaves the output of pids 16444 and 27516. kp_trap.json was written by pid 16444 at 06:54:50, 4794 s after its 05:34:54 start. Pid 27516 was absent from the 06:52 process list, with no record of how it ended. The Inspector's rerun took 29m43s, against the lane's 80 min shared between two copies. Timing is not ruled on.
5. **A report carrying struck claims.** FOREMAN_REPORT.md (07:03:10) repeats "4× an arm" and "wrong id rather than a guess", both struck, and adds "depth ≈ 8–12", which has no source (N8).
6. **A pilot ahead of arms.** Cameron put the pilot ahead of far_aloop5_s0 and far_ass4_s0 (declared at 5891). No other lane was waiting.
7. **An unfilled report.** CAMERON_REPORT.md (06:56) still reads "(filled at the end)" for R-DEPTH.

## Lanes still in flight (cut at board 5915, 07:43)

- **Cameron.**
  - Runner pid 15476 has held the lock for far_aloop5_s0 since 07:40:52.
  - Queued after it: far_ass4_s0, far_fR_s1, far_aL7_s1, far_aL4_s1, far_ass4_s1, far_aloop5_s1.
  - test_rdepth_far10.py has not been run by Cameron. The Inspector's run reads RED with 34 failing rows:
    - fR s0 and aL4 s0 are UNREAD at the gate;
    - the aL7 s0 rows PASS;
    - the rest are missing.
  - The R-DEPTH section of CAMERON_REPORT.md is unfilled.
- **Foreman.** Done at 5875. far_opp_* is void at the gate, and the aL7 opponent never ran.
- **Chase.** Done at 5817; nothing after 5866.
- **Wilson.** Done at 5802; nothing after 5866.
