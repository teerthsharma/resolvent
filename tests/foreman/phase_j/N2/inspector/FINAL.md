# Inspector, round N2: audit of FOREMAN_REPORT.md (R-POS) and CHASE_REPORT.md (R-DIAG′)

**20 claims were audited and 5 were struck.** Both headlines bind:
- **R-POS:** PASS. a_rope recovers 0.9632, 0.9971 and 1.0143 of C_win.
- **R-DIAG′:** PAST A POLE, in 1,432/1,536 pairs = 93.2%.

Board events: `house-events.jsonl` lines 5137–5156. Everything re-run here was on CPU, from `N2/inspector`.

## Table

| # | cites | claim | verdict | evidence |
|---|---|---|---|---|
| F1 | Foreman | R-POS PASS: a_rope 0.9632 / 0.9971 / 1.0143 (mean 0.9915) | clean | The `test_rpos.py --stub` re-run exits 1 with the line on board 5094 verbatim. The real run exits 0 and prints 'R-POS verdict PASS', matching board 5131. The recomputation from the grid agrees at 3/3 seeds. |
| F2 | Foreman | a_alibi 1.0884 / 1.0671 / 1.0882; beats (f) by +0.0206 / +0.0160 / +0.0231 nats | clean | These are printed by the same GREEN run. The arithmetic is recomputed and matches. |
| F3 | Foreman | CRN digests and parameter parity | clean | In 7/7 rows the recomputed digest equals the file digest, the grid (a) digest and the grid (f) digest. In 7/7 rows the parameter count is 724,608, the same as grid (a). |
| F4 | Foreman | Bar registered before the run (check 2) | clean | RECORD_N.md was last written at 00:34:22. That is before test_rpos.py (00:37:05) and before the first cell (00:39:45). The PASS clause matches the test. RED 5094 precedes every result. |
| F5 | Foreman | Provenance of the a_rope ss0 cell across the 00:50 edit of rpos.py | clean | The pre-edit `.pyc` (compiled from the 00:36:55 source) has the same arm bytecode as the current source. Only `run()` differs. |
| F6 | Foreman | Controls: (a) re-run and wall-clock ratios | clean | The (a) re-run differs from the grid by 1.937e-6 nats, which is 8.3e-6 of C_win. The timings match the report's table. |
| F7 | Foreman | "FoX's data dependence buys nothing measurable" | **struck** | Unbound; see the struck list. |
| F8 | Foreman | "(a) was the only arm with no relative position; every bar measured that handicap" | **struck** | Unbound; see the struck list. |
| F9 | Foreman | Nurse code audit: Toeplitz to 2.4e-6, ALiBi exact, rebinding restored | **struck** | Unbound; see the struck list. |
| F10 | Foreman | poll_until_free stalls multi-cell loops | **struck** | Unbound; see the struck list. |
| F11 | Foreman | Check 4 against Wilson | clean | No contradiction. |
| C1 | Chase | PAST A POLE: 424 / 496 / 512 of 512; γ* min 0.8286 / 0.7071 / 0.4836 | clean | The stub gives 7/7 RED. The real run gives 5 GREEN and 2 RED, exits 1, and prints the line on board 5115 verbatim. The JSON is identical to Chase's apart from `seconds`. |
| C2 | Chase | Bar set at 00:38, before measuring (check 3) | clean | The test file was last written at 00:38:08 and its sha256 matches the report. The first read is 00:41:37 (run 1 died on the CPU guard). |
| C3 | Chase | Fidelity: NLL 1.0430563, attention-output diff 0.0, learned β | clean | Four GREENs reproduce. Stub REDs 5097–5100 precede them. |
| C4 | Chase | Re(W) identical | clean | The GREEN and RED reproduce verbatim (board 5114 and 5116). |
| C5 | Chase | No absorber other than position 0; unit diagonals 0; excluding position 0, 1,014/1,536; Neumann growth 2.0470 | clean | These are the same bound run's JSON fields, reproduced exactly. |
| C6 | Chase | Pinning β = 1 costs +0.0222 nats, 9.6% of C_win (FREE bar RED) | clean | Stub RED 5122 precedes. The test file was last written at 00:43:47, before the first read at 00:44:48. `beta_pin.json` reproduces identically. |
| C7 | Chase | Ruling on the β-pin GREEN re-run | clean (binds) | See the ruling below. |
| C8 | Chase | Second implementation agrees (board 5128) | **struck** | Unbound; see the struck list. |
| C9 | Chase | Check 4 against Wilson | clean, with one correction | β, qk, gate zeros and min m all agree with Wilson §3. "BOS row" is a mislabel: Wilson §4 found that the bed has no BOS token. |

## Recomputed R-POS recoveries

Recovery is (loss(a) − loss(arm)) / (loss(a) − loss(f)), computed from `design4x5_results.jsonl`.

| arm | ss0 | ss1 | ss2 | mean |
|---|---|---|---|---|
| a_rope | 0.9632 | 0.9971 | 1.0143 | 0.9915 |
| a_alibi | 1.0884 | 1.0671 | 1.0882 | 1.0813 |
| a (re-run) | −0.0000 | | | |

C_win is 0.232628, 0.238993 and 0.261949 at ss0, ss1 and ss2.

## Ruling on the β-pin re-run (check 3)

**GREEN 5125 binds, and GREEN 5123 stays void.**
- Stub RED 5121 [beta_pin_stub] and GREEN 5125 [beta_pin] name the same file and unit. They differ only in the note of what the test ran against, so under N1's name rule the RED binds the GREEN.
- `test_beta_pin.py` has not been edited since the RED (00:43:47). The fix was made only in `beta_pin.py` (00:45:01).
- The bug changed only `trained_gstar_min` (0.934 against 0.4836). The tested `beta1_gstar_min` was 1.0 in both runs.
- The check is a theorem check. At β = 1, W_ii = p_ii ≤ 1 and W_00 = 1, so it establishes only that the implementation is sane.
- The docstring's "Registered 00:45" disagrees with the file time. The mtime shows the bar predates the first measurement.

## Struck

1. **Foreman, board 5134: "FoX's data dependence buys nothing measurable over fixed ALiBi here."**
   - No RED or test event names it, and N2 did not run FoX.
   - No bar comparing ALiBi and FoX was registered, and FoX leads at seed 2.
   - The FoX figures themselves match RECORD.md:60.
2. **Foreman, board 5133 claim text and the report's "Standing consequence": "(a) was the only arm in the grid with no relative-position mechanism; every bar scored against (a) measured that handicap."**
   - No test asserts it.
   - For the record: the grid's a2 (design4x5.py:195–201) is softmax plus a post-SDPA output gate, which is no relative prior either.
3. **Foreman, report "Code audit" (sonnet nurse): RoPE Toeplitz to 2.4e-6, ALiBi exactly −slope·(i−j), rebinding restored.**
   - The nurse left no artifact, and no test event carries these numbers.
4. **Foreman, board 5135: "poll_until_free stalls multi-cell loops."**
   - It is a finding with no RED.
   - The log does show 32 polls at compute_procs=1 after cell 1, so it can be re-filed with a test.
5. **Chase, board 5128: "R-DIAG prime second implementation agrees" (nurse1/wii_check.py).**
   - It is a finding with no RED.
   - The nurse's log does print 1.2068454693, and the headline binds without it.

## Not re-run

- **R-POS training:** 7 GPU cells, excluded by the brief. Verified from `rpos_results.jsonl` against the grid.
- **CRN digests recomputed from the corpus:** checked for equality inside the file and against the grid rows only.
- **`rpos.py --check`:** not run.
- **The sonnet nurse's code audit:** no artifact to re-run.
- **Chase's `nurse1/wii_check.py`:** its log was read, not re-executed.
- **The FoX control rows:** read from RECORD.md:60 only.
- **The reroute and next-row proposals in both reports:** these are unmeasured, so there was nothing to audit.

Scripts, logs and outputs from this audit, all in `N2/inspector`:
- scripts: `recompute_rpos.py`, `pyc_diff.py`, `audit_n2.py`
- logs: `rpos_stub.log`, `rpos_real.log`, `stub_rdiag.log`, `real_rdiag.log`, `stub_pin.log`, `real_pin.log`
- outputs: `rdiag_prime.json`, `beta_pin.json`, `board_local.jsonl`, which holds the board events from the re-runs, redirected there
