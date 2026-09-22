# Inspector final pass (before the prognosis)

Board: house-events.jsonl. This run starts at line 4719 (Foreman dispatch). Pass 1 wrote lines 4805-4845, and this pass wrote 27 audit events at lines 4849-4875 (events_final.py). **This pass: 27 audited, 10 struck, 17 clean.** Pass 1 (41 audited, 8 struck) is carried unchanged.

GPU re-runs: the card was polled free with design4x5.poll_until_free(600), and nvidia-smi listed no compute process before any run. The runs used copies in SPJ/L1/inspector/runs/cam, with PYTHONPATH=SPJ so that costb_impl imports from SPJ. costb_fused.py is byte-identical to the fellow's 23:11:14 file (md5 edd8af50a5512232fecbfcaebea2448f). No fellow file was written. Raw outputs are in runs/cam/out_*.txt, the costb_*_run1.json files (run 1) and costb_*.json (run 2 for the fused impl), gpu_rerun.log and gpu_rerun2.log. CPU runs used CUDA_VISIBLE_DEVICES="" and PYTHONDONTWRITEBYTECODE=1: runs/rivals_*_rerun.txt and runs/cam_admissible_pass2.txt.

## GPU re-run numbers (RTX 4060 Laptop, torch 2.14.0+cu126, 23:43-23:45)

| Test | IMPL | Logged | Re-run 1 | Re-run 2 | Status |
|---|---|---|---|---|---|
| test_costb_fused.py | author | RED, fold 3.775e+00, scan 1.927e+00, 11.422/4.593 = 2.487x | RED, fold 3.775e+00, scan 1.927e+00, 10.575/6.283 = **1.683x** | not run | RED reproduces |
| test_costb_fused.py | fused | GREEN, fold 2.21e-06, scan 6.63e-05, 4.947/4.754 = 1.041x | GREEN, same errors, 7.498/6.890 = **1.088x** | GREEN, 6.987/6.755 = **1.034x** | GREEN reproduces on the post-edit file |
| test_costb_fwdbwd.py | author | RED, grad 1.634e+00, 37.142/17.574 = 2.113x | RED, grad 1.634e+00, 44.556/25.460 = **1.750x** | not run | RED reproduces |
| test_costb_fwdbwd.py | fused | GREEN, grad 9.87e-07, 19.698/17.567 = 1.121x | GREEN, same grad, 27.964/25.858 = **1.081x** | GREEN, 27.756/25.388 = **1.093x** | GREEN reproduces |

The error values match the log bit for bit. The wall-clock ratios moved because the card ran in a different clock state: SDPA took 6.3-6.9 ms forward against 4.6-4.75 ms when the log was written, with the SM at 1890-2670 of 3105 MHz and the die at 77-84 C. Each ratio still lands on the same side of the 1.3x bar as the log. The ratios are bound to the side of the bar they fall on, not to their exact values.

## Table (this pass)

| # | Claim | Check | Actual output line | Verdict | Why |
|---|---|---|---|---|---|
| 42 | Cameron test_costb_fused author RED | GPU re-run | `RED author: fold: max \|out - contract G_ij fold\| = 3.775e+00 > 1e-4 \| scan: max \|Pi_fp32 - Pi_f64\| = 1.927e+00 > 1e-4 \| ratio: Mode B' fwd 10.575 ms / SDPA 6.283 ms = 1.683x > 1.3x` exit 1 | clean | RED holds; ratio drift noted above |
| 43 | Cameron test_costb_fused fused GREEN (file edited 23:11) | GPU re-run x2 | `GREEN fused: fold 2.21e-06, scan 6.63e-05, ratio 1.088x`; `... ratio 1.034x` exit 0 | clean | the forward GREEN now binds the current file |
| 44 | Cameron test_costb_fwdbwd author RED | GPU re-run | `RED author: grad: max rel err vs contract f64 = 1.634e+00 > 1e-3 \| ratio: fwd+bwd 44.556 ms / SDPA 25.460 ms = 1.750x > 1.3x` exit 1 | clean | RED holds |
| 45 | Cameron test_costb_fwdbwd fused GREEN | GPU re-run x2 | `GREEN fused: grad 9.87e-07, fwd+bwd ratio 1.081x`; `... 1.093x` exit 0 | clean | GREEN holds |
| 46 | F1: cost leg survives, 1.041x forward | board order + re-run | REDs 4729 and 4742 come before GREENs 4730 and 4743, which come before finding 4744 | clean | bound; supersedes pass-1 #34 "not re-run: GPU" |
| 47 | F2: forward+backward 1.121x | board order + re-run | fused fwd+bwd JSON 23:11:39 is after the 23:11:14 edit; re-run 1.081x / 1.093x | clean | bound; supersedes pass-1 #35 |
| 48 | F1 baseline: EFFICIENT_ATTENTION, no flash or cuDNN, MATH about 25 ms | re-run backend dict | `EFFICIENT_ATTENTION 5.890 / 6.661 / 6.718`, `MATH 26.5 / 28.5 / 28.1`, FLASH and CUDNN `No available kernel` | clean | backend identity holds; the 4.55-4.75 ms level did not reproduce at this clock state |
| 49 | F1: Triton 3.7.1 core.py:2770 associative_scan with reverse= | read installed file | `2770 def associative_scan(input, axis, combine_fn, reverse=False, _semantic=None, _generator=None):` | clean | the citation checks out |
| 50 | F1 kill: k22 tile-load fusion retired, 0.136 of 4.72 ms (2.9%) | board | no event | **struck** | report-only; stage times have no test |
| 51 | F3: costb_impl builds Pi_i = q_0...q_i, fold err 3.775; test_costb compared it with itself | board + re-run + read | fold `3.7753911924862122` bitwise; SPJ/test_costb.py:29,80 reference `Pi[i] = q0*...*qi` | clean | RED 4729 comes before finding 4745; agrees with Wilson [V] |
| 52 | F3: CPU S=16 f64 identity check 3.3e-16 | search | `3.3e-16` occurs only in CAMERON_REPORT.md | **struck** | no script and no event |
| 53 | test_bed_admissible RED x3 | CPU re-run | `INADMISSIBLE S3 prefix>=32 (best control multiset 0.3641): A2: twin loss did not plateau \| A2: twin never learned the task (short-prefix acc 0.7945 < 0.90)`, exit 1 x3 | clean | identical to pass 1 (only line endings differ) |
| 54 | F4 floors (chance, multiset, window, twin, longer twin, prefix-8, A5 0.4785) | JSON read | S3/S4: chance 0.3333/0.25; R256 argmax 0.333/0.2523; plugin 0.3641/0.2825; w8 0.3272/0.2392; twin 0.3398/0.2432; long 0.3323/0.2550; prefix-8 0.9175/0.9375; A5 0.4785 | clean | numbers equal the JSONs; RED 4776 comes before finding 4778 |
| 55 | F4: S3 and S4 pass A4 | read test | test_bed_admissible.py:8-9 reads A4 from `fq_construction.json beside the input` | **struck** | rests on fq_construction (pass-1 #40, struck) |
| 56 | F4: board bed retired | CPU re-run | `A3: best floor rule_free_tracker = 0.9968, +0.10 = 1.0968 > 1.0` | clean | carried; equals Wilson [V] 0.996844 |
| 57 | Cameron vs Wilson [V] | cross-read | no contradicting line | clean | costb_impl order, EFFICIENT_ATTENTION, scan 5.97, 0.996844 all agree |
| 58 | Chase rivals stub RED 0/10 | CPU re-run | `{"module": "rivals_stub", "passed": 0, "total": 10}`, all rows NotImplementedError | clean | identical to red_run2.txt; exit status 0 on RED, so RED is readable only from the count |
| 59 | Chase rivals GREEN 10/10 | CPU re-run | `{"module": "rivals_impl", "passed": 10, "total": 10}` | clean | every row identical to green_run.txt |
| 60 | board test events 4797-4799 | schema read (SKILL.md:83-84, 230; house.py:128) | `{"t":"test","agent":"Chase","state":"red","file":...}` | **struck** | no `status` and no `name`; see ruling |
| 61 | finding 4800: a2 = Qwen headwise G1 | board order | RED only as `state` | **struck** | unbound |
| 62 | finding 4801: GDN decay = FoX gate up to a per-head power | board order | RED only as `state` | **struck** | unbound |
| 63 | finding 4802: KDA/GDN spectrum in [0,1]; unit circle not shipped | board order + mtimes | RED only as `state`; T3c row 23:35:03 written after rivals_impl.py 23:33:33 | **struck** | unbound |
| 64 | finding 4803: lightning = exp(ALiBi) = constant-f FoX; softmax sets order-blind | board order | RED only as `state` | **struck** | unbound |
| 65 | report M1-M10 literature facts that have a SOURCES.txt line | grep SOURCES.txt | 6.026/5.761/5.792, 46.7%->4.8%, 9.0x/6.0x/11.6x, 6.5x, 75%/6.3x, licenses, RoPE bases all present | clean | sourced to Chase's restatement; the verbatim nurse fetches are not on disk; correctness not ruled |
| 66 | report: KDA prefill 2.9x; Qwen3-Next "10 times ... over 32K"; "12 x (3 GDN + 1)" | grep SOURCES.txt + board | 0 hits each | **struck** | no source line and no event |
| 67 | report claim 3: no lab ships learned phase, retire | board | no event | **struck** | a negative over ten sampled sources; unbound |
| 68 | Chase rivals vs Wilson [V] | cross-read | no contradicting line | clean | a2_gate form matches design4x5.py:211-213 |

### Ruling on "state" versus "status"
The rivals test events are **unbound**. The schema is `test (status red/green, name)` (SKILL.md:230), and the binding rule is "a finding whose test name never appears with `status: red` is unbound" (SKILL.md:83-84). Events 4797-4799 carry `state` and `file`, with neither `status` nor `name`. The board mirror reads `s.tests[e.name||"?"]=e.status` (house.py:128), so these events never show on the board as red or green. Cameron's events lack `name` too, but they carry `status`, and pass 1 accepted the `test` path as the identifier, so the field that carries the verdict is the one enforced. The re-runs reproduce 0/10 and 10/10 exactly, so the substance is reproducible, but the log does not bind it. The rule forbids re-testing a struck claim until it passes, so these stay struck for this run.

## Combined struck list (pass 1 + this pass): 18

| # | Agent | Claim | Reason |
|---|---|---|---|
| 1 | Chase | Q1, FOLD, FQ-ARM reproduced by independent code | no RED on the board (Chase filed it as Open) |
| 2 | Chase | construction survives: hand-set f_Q A5 500/500 at L64, 50/50 at L4096; pair closes 60/120 | route_handset_a5.py has no event; GREEN Q2-REROUTE-INDEPENDENT has no RED |
| 3 | Chase | FQ-ARM 36 params only at d 8, 260 at d 64 | report-only, no event |
| 4 | Foreman | T2b acc@64 0.0000, T2c 1.183/1.078 eval 0.059, anchored-bed 0.0156 | split seeded by salted hash(); re-run gives 0.0312, 1.176/1.351, 0.0 |
| 5 | Foreman | F3: L1-IAUT bar 0.8120 unreachable by any finite SU(2) image | post-hoc RED re-reads a known JSON; covers 2I/2O only |
| 6 | Foreman | I-AUT key 0 reachable p 0.9999; linear readout 0.2904 | no test event; GREEN has no RED |
| 7 | Cameron | exact SU(2) construction on shell bed 1.000/1.000 | fq_construction.py has no event |
| 8 | Cameron | torch.compile 1.069x; fused ratio vs S 1.132/1.065/1.035/0.978 | costb_compile.py and costb_extra.py have no event |
| 9 | Cameron | k22 tile-load fusion retired (2.9%) | report-only, no event |
| 10 | Cameron | F3 CPU S=16 identity check 3.3e-16 | no script, no event |
| 11 | Cameron | S3 and S4 pass A4 | A4 reads fq_construction.json (#7) |
| 12 | Chase | rivals test events 4797-4799 | `state` not `status`, no `name`: unbound under the schema |
| 13 | Chase | finding 4800: a2 = Qwen headwise G1 | RED only as `state` |
| 14 | Chase | finding 4801: GDN decay = FoX gate | RED only as `state` |
| 15 | Chase | finding 4802: shipped per-token spectrum in [0,1]; unit circle not shipped | RED only as `state` |
| 16 | Chase | finding 4803: lightning = ALiBi = constant FoX; softmax order-blind | RED only as `state` |
| 17 | Chase | KDA prefill 2.9x; Qwen3-Next 10x over 32K; 12 x (3 GDN + 1) | no source line, no event |
| 18 | Chase | learned phase ships nowhere, retire | report-only, no event |

## Not re-run
- Foreman a5_anchor_train.py 512/400 runs (carried from pass 1, cost 282-453 s each). The saved jsonl was audited.
- Cameron shell_bed.py and twin_long.py training runs (150-600 s each). The floors were audited against the saved JSONs, and the bed is RED either way.
- Cameron costb_fused_stages.json stage timings: no test produces them.
- Cameron fq_construction.py, costb_compile.py and costb_extra.py: struck in pass 1, and the rule forbids re-testing them.
- Chase's costb_timing.py (2.149x): outside this pass's scope and not re-run.

## Notes for the prognosis (not verdicts)
- The author costb_impl ratio depends on the clock state: 2.487x became 1.683x forward and 2.113x became 1.750x fwd+bwd once the SDPA baseline slowed to 6.3 ms. Both are still above 1.3x. Chase's "past 2x" strike used a different script (costb_timing.py, 2.149x), and this pass did not re-run it.
- test_rivals.py exits 0 whether rows pass or fail, so its RED/GREEN is readable only from the `passed` count.
