# Inspector FINAL, round N1

The Inspector rules on binding only, never on correctness. Checks: 1 = re-run matches; 2 = a named RED precedes it on the board; 3 = no contradiction with a Wilson [V] fact; 4 = the bar was set before the result was seen.

| # | fellow | claim | check | actual output line | verdict | why |
|---|---|---|---|---|---|---|
| 1 | Chase | RED-name binding rule: test_n.py::<name> [n_impl_stub] | 2 | rerun stub: 'RED   T1 NotImplementedError: instance not built yet' ... EXIT 1 (27/27 RED) | clean | Binds. Rule applied to all three fellows: a RED binds a later GREEN/finding if both name the same test file and the same test unit (function or assertion label) and differ only in an annotation of what the test ran against. Chase's ' [n_impl_stub]' suffix, Foreman's 'against' field and Cameron's '--stub' flag are that annotation. The names do not have to match character for character. |
| 2 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T1 CERTIFIED | 1,2 | GREEN T1 | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 3 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T2 CERTIFIED | 1,2 | GREEN T2 | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 4 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T4 CERTIFIED | 1,2 | GREEN T4 | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 5 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T5 CERTIFIED | 1,2 | GREEN T5 | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 6 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T6 CERTIFIED | 1,2 | GREEN T6 (asserts 0<mixed<=transient only; counts not compared) | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 7 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T7 CERTIFIED | 1,2 | GREEN T7 | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 8 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T8 CERTIFIED | 1,2 | GREEN T8 (sparsemax 0 mismatch/0 false mass + leak==law) | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 9 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T9 CERTIFIED | 1,2 | GREEN T9 | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 10 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T10 CERTIFIED | 1,2 | GREEN T10 | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 11 | Chase | scratchpad/phase_j/N1/chase/test_n.py::X1 CERTIFIED | 1,2 | GREEN X1 | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 12 | Chase | scratchpad/phase_j/N1/chase/test_n.py::X2a CERTIFIED | 1,2 | GREEN X2a | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 13 | Chase | scratchpad/phase_j/N1/chase/test_n.py::X3a CERTIFIED | 1,2 | GREEN X3a | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 14 | Chase | scratchpad/phase_j/N1/chase/test_n.py::X3b CERTIFIED | 1,2 | GREEN X3b | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 15 | Chase | scratchpad/phase_j/N1/chase/test_n.py::X4 CERTIFIED | 1,2 | GREEN X4 | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 16 | Chase | scratchpad/phase_j/N1/chase/test_n.py::KR CERTIFIED | 1,2 | GREEN KR | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 17 | Chase | scratchpad/phase_j/N1/chase/test_n.py::PF CERTIFIED | 1,2 | GREEN PF | clean | rerun matches run2.log and board; stub RED precedes GREEN on board |
| 18 | Chase | scratchpad/phase_j/N1/chase/test_n.py::T3 (T3b diagonal unchanged) CERTIFIED | 4 | GREEN T3 (run1.log 00:07:00: RED T3 're_diag_bitwise_equal: False') | **STRUCK** | Post-hoc bar. The first real run (run1.log, 00:07:00, board 4934) was RED on bitwise diagonal equality. test_n.py was then edited to re_diag_max_diff <= 1e-12, with the comment 'bitwise was a defect of this test'. That edit is at mtime 00:08:04, after the RED was seen, and the GREENs at 4964/4977 rest on the loosened bar. |
| 19 | Chase | T8 softmax cells 'CERTIFIED pattern' (0.0083 & 0/448; 0.120 & 697/98,119) | 2 | T8 asserts only the sparsemax cells and leak==law; the softmax cells are recorded, never asserted | **STRUCK** | Unbound. No test event asserts the softmax cells; the report's own column reads 'stub RED -> recorded'. |
| 20 | Chase | scratchpad/phase_j/N1/chase/test_n.py::X2b VOID (author's ordering not reproduced) | 1,2 | RED   X2b AssertionError: {'global': 3.899008222063571e-06, 'segmented': 6.570651501780489e-07, ...} | clean | stub RED 4904 -> RED 4944/4987; rerun identical |
| 21 | Chase | scratchpad/phase_j/N1/chase/test_n.py::B1_D1_unit_count_beta1 CERTIFIED | 1,2 | GREEN B1_D1_unit_count_beta1 | clean | stub RED 4908 -> GREEN |
| 22 | Chase | scratchpad/phase_j/N1/chase/test_n.py::B1_D1_unit_count_trained_beta (D1 STRUCK at beta!=1) | 1,2,3 | RED   B1_D1_unit_count_trained_beta AssertionError: ('1.09', {'unit': 0, 'structural': 3, 'max_diag': 1.0677041954249709, ...}) | clean | stub RED 4909 -> RED. The betas 1.09/0.96/0.99 are the q3 checkpoint's; Wilson [V] gives 1.0907/0.9616/0.9624/0.9913, so there is no contradiction. The d45 (f) betas are 1.061/0.930/0.900. |
| 23 | Chase | scratchpad/phase_j/N1/chase/test_n.py::B5_no_pole_inside_unit_gamma (pole inside gamma<1 STRUCK) | 1,2 | RED   B5_no_pole_inside_unit_gamma AssertionError: ('1.09', {... 'max_diag': 1.0677041954249709, ... 'pole_gamma': 0.9365889956084484 ...}) | clean | Binds for the single draw (gamma* 0.937). The 16/20, 16/20 and 10/20 draw frequencies and the '0/60 over 20 draws' are asserted by no test and produced by no saved script (n_impl.b1 is one draw). They are unbound support, not part of the verdict. |
| 24 | Chase | scratchpad/phase_j/N1/chase/test_n.py::B2_D3_as_written_complex_gate (D3 as written STRUCK) | 1,2 | RED   B2_D3_as_written_complex_gate AssertionError: {'as_written': 1.3314002873135162, 'modulus_rho': 2.220446049250313e-16, ...} | clean | stub RED 4910 -> RED |
| 25 | Chase | scratchpad/phase_j/N1/chase/test_n.py::B2r_D3_modulus_rho CERTIFIED | 1,2 | GREEN B2r_D3_modulus_rho | clean | stub RED 4911 -> GREEN |
| 26 | Chase | scratchpad/phase_j/N1/chase/test_n.py::B3_D5_dense_depth (D = n-1; cost-is-depth STRUCK for dense) | 1,2 | GREEN B3_D5_dense_depth | clean | Binds D = n-1 (stub RED 4912 -> GREEN). B3 does not assert the jump-chain hop counts (22/30, 2,749 Neumann), so those are unbound support. |
| 27 | Chase | scratchpad/phase_j/N1/chase/test_n.py::B4_RDIAG_gate_route CERTIFIED at beta=1 | 1,2 | GREEN B4_RDIAG_gate_route | clean | stub RED 4913 -> GREEN |
| 28 | Chase | scratchpad/phase_j/N1/chase/test_n.py::B4_RDIAG_logit_route_diag (count check STRUCK) | 1,2 | RED   B4_RDIAG_logit_route_diag AssertionError: {'diag_bitwise': False, 'count_equal': True, 'count': 3, 'max_abs_diag_diff': 0.8302004224312057} | clean | stub RED 4914 -> RED |
| 29 | Chase | scratchpad/phase_j/N1/chase/test_n.py::B4_RDIAG_can_fire_at_trained_beta VOID | 1,2 | RED   B4_RDIAG_can_fire_at_trained_beta AssertionError: ('1.09', {'count': 0, 'count_other_theta': 0, 'rdiag_check_passes': True}) | clean | stub RED 4915 -> RED |
| 30 | Chase | Rounding-manufactured unit diagonals (2 in f64, 13 in f32), board 5015 | 2 | no test event; the report says 'recorded inside b4', and no B4 assertion reads it | **STRUCK** | Unbound: no test event, RED or otherwise, carries this finding. |
| 31 | Chase | Lean read-only finding (det_of_lowerTriangular at Block.lean:265; no charpoly-conjugation lemma), board 5016 | 2 | no test event | **STRUCK** | Unbound: no RED precedes the finding. The same facts stand as Wilson [V] (section 5), not as Chase's claim. |
| 32 | Foreman | H-SINK STRUCK (kill: P1 and P4 at null) | 1,2 | RED P1: P1 lift 0.0 < 2 \| RED P4: P4 rho -0.01985042469162218 < 0.3 | clean | Stub RED (4883/4886) -> RED (4917/4920). The rerun matches. The closure count 65 matches Wilson [V]: 65 below 0.05, all in layer 0. |
| 33 | Foreman | P2 STRUCK: (f)'s \|W_ic\| 0.0056 | 3 | RED P2: P2 0.005610662512481213 < 0.3 | **STRUCK** | Contradicts [V]. rsink_measure.f_attn computes '(f)'s \|W_ij\| at beta=1', but Wilson [V] gives (f)'s trained beta as 1.061/0.930/0.900. The measured operator is not (f)'s. |
| 34 | Foreman | P3 STRUCK: value-norm ratio 0.843 | 1,2 | RED P3: P3 0.8432363867759705 > 0.5 | clean | stub RED 4885 -> RED; rerun identical |
| 35 | Foreman | FT_a_sink: the twin has no first-token sink (bar 0.2 'set before measuring') | 1,2,4 | RED FT_a_sink: (a) first-token mass max 0.011304507032036781 < 0.2 | clean | The pre-measurement bar is confirmed by mtime: test_rsink.py 00:01:36 < rsink_measure.py 00:02:57 < rsink.json 00:03:46. Stub RED at 4887. |
| 36 | Foreman | FT_f_relief GREEN ((f) first-token mass 0.00225/0.00163/0.00194) | 3 | GREEN FT_f_relief | **STRUCK** | Contradicts [V]: the (f) masses come from f_attn at beta=1, while (f)'s beta is 1.061/0.930/0.900. Foreman had already marked it VOID. |
| 37 | Foreman | FT_fox_relief GREEN (VOID by Foreman) | 1,2 | GREEN FT_fox_relief | clean | Bound: stub RED 4889 -> GREEN 4923. Foreman's VOID stands. |
| 38 | Foreman | P5 CERTIFIED: output gate is not sink relief (-0.155) | 1,2,4 | GREEN P5_not_sink \| RED P5_sink_relief: recovery -0.15473039156774196 < 0.5 | clean | The 0.2 and 0.5 bars are the contract's (sec.5). The added CI<0.5 only tightens. Stub RED 4957/4958 precede. |
| 39 | Foreman | X_colocate CERTIFIED (0.634/0.541/0.311) | 1,2 | GREEN X_colocate | clean | stub RED 4890 -> GREEN 4924 |
| 40 | Foreman | W1_f_local CERTIFIED (+0.00033) | 1,2,4 | GREEN W1_f_local | clean | stub RED 4925 -> GREEN 4968; test_window.py 00:04:55 < window_measure.py 00:05:16 |
| 41 | Foreman | W1_fox_local CERTIFIED (+0.00110) | 1,2 | GREEN W1_fox_local | clean | stub RED 4926 -> GREEN 4969 |
| 42 | Foreman | W2_twin_dilution STRUCK (recovery -7.42) | 1,2 | RED W2_twin_dilution: twin recovery -7.42283652572706 < 0.5 | clean | stub RED 4927 -> RED 4970 |
| 43 | Foreman | C_datadep_f CERTIFIED eval-only (+0.316) | 1,2,4 | GREEN C_datadep_f | clean | stub RED 4962 -> GREEN; test_const.py 00:07:46 < const_measure.py 00:08:08 |
| 44 | Foreman | C_datadep_fox CERTIFIED eval-only (+0.381) | 1,2 | GREEN C_datadep_fox | clean | stub RED 4963 -> GREEN |
| 45 | Foreman | C_const_suffices STRUCK at eval | 1,2 | RED C_const_suffices: const cost f 0.3160681627143209 fox 0.38096815585093946 > 0.02 | clean | stub RED 4961 -> RED 4965 |
| 46 | Foreman | N_beta_convention: beta=1 does not hold on (f) (board 5007) | 4 | RED N_beta_convention: max \|beta-1\| = 0.10042041540145874 > 0.01 | **STRUCK** | Post-hoc bar. rsink_run.log (00:03:46) already printed beta_f 1.0605748891830444, which alone fails 0.01. test_beta.py was written at 00:09:03. The fact stands as Wilson [V] (1.061/0.930/0.900); the binding is struck. |
| 47 | Foreman | 'layers 1-2 never close (minimum m 0.119 / 0.129)' and 'hard-concrete floor at m ~ 0.0497' | 3 | rsink.json m_quantiles[.][0] = 0.049658 / 0.118885 / 0.129461 (these are the 0.001 quantiles) | **STRUCK** | Contradicts [V]. On the same 64x512 windows (generator 20260921, split 0; medians 0.6730/0.4406/0.5300 identical), Wilson's minimum m is 0.04717 / 0.09669 / 0.09255. The reported 'minimum' and 'floor' are 0.1% quantiles. |
| 48 | Foreman | 'the family's own regime (exact zeros, absorbers, NEVER) fires on 0.066% of site-layers' | 3 | 65/98,304 counts m < 0.05, not m == 0 | **STRUCK** | Contradicts [V]: Wilson's table has 0 exact zeros of m in all three layers. |
| 49 | Cameron | R-COST-N prediction <=1.5x STRUCK (bar declared before any fs number) | 1,2,4 | RERUN: RESULT impl=fs5g_c256 ratio=1.757 rel_err=2.959e-06 cost_bar=FAIL exact_bar=PASS | clean | The rerun ratio is 1.757 (SDPA 4.51 ms, impl 7.92 ms). That is below the reported 1.777-1.797 but on the same side of both bars. The pre-measurement bar is confirmed: test_cost_n.py 00:02:58 < explore_fs.log 00:04:05. RED 5026 precedes finding 5027. |
| 50 | Cameron | R-COST-N counter >3x does not hold | 1 | RERUN ratio 1.757 < 3 | clean | Counter bar from contract sec.6 |
| 51 | Cameron | C4 CERTIFIED: hop scheme ~130x, Chebyshev K=130 not exact | 1,2 | RERUN: RESULT impl=hop_cheb ratio=128.548 rel_err=4.458e+04 cost_bar=FAIL exact_bar=FAIL | clean | Reported 129.538 against 128.548 in the rerun. exact_bar=FAIL is printed by the same RED run, so it binds. |
| 52 | Cameron | R-COST-N side rows: Neumann K=130 128.6x, fs2g 2.27x, fs_block256 3.13x, fs_dense 8.01x | 2 | no test_cost_n.py event for these impls | **STRUCK** | Unbound: the only test events are for hop_cheb and fs5g_c256. |
| 53 | Cameron | Two torch-level routes lost: blocked-doubling 2.14x, inverse-ahead 2.19x (board 5033) | 2 | no test event; explore_fs6/7.log only | **STRUCK** | Unbound: no RED precedes the finding. |
| 54 | Cameron | R-NEVER-LEN P1 CERTIFIED: f_N false mass exactly 0 | 1,2 | RERUN stub: 'R-NEVER-LEN P1 FAIL: f_N false mass max 6.830e-02 > 0' \| real: 'P1 f_N false mass per (n,seed): [0.0 x 15]' | clean | Binds by the name rule: same file and same P1 assertion, with --stub as the against-flag. RED 5021 precedes GREEN 5023. |
| 55 | Cameron | '(a) grows as the leak law says: CERTIFIED' | 2 | test_never_len.py asserts P1, the counter and P2 only; no leak-law assertion | **STRUCK** | Unbound: no test event asserts the leak law (Chase's T8 does, but on his own instance). |
| 56 | Cameron | R-NEVER-LEN P2 STRUCK / counter HOLDS | 1,2,4 | RERUN: AssertionError: R-NEVER-LEN COUNTER HOLDS: (a_s)+threshold within 0.02 (0.0000); NEVER is a certificate only | clean | The 0.10 and 0.02 bars are the contract's. test_never_len.py (00:11:05) precedes never_len_rows.jsonl (00:12:29). RED 5024 precedes finding 5025. |
| 57 | Cameron | R-NEVER-LEN' threshold transfer retired | 1,2 | RERUN stub: '... within 0.02 (-0.9984)' \| real: 'R-NEVER-LEN' COUNTER HOLDS: transferred-tau (a_s) within 0.02 (0.0093)' | clean | stub RED 5030 -> RED 5031; per-seed [1,1,1,0.9534,1] reproduced |
| 58 | Cameron | Dense causal depth: 'D5 as numerical cost STRUCK (26-30 hops)' + structural n-1 (board 5028/5029) | 4 | RERUN: AssertionError: D5-as-framed FAIL at n=256: rel err <= 1e-12 after 26 hops, not n-1 = 255 | **STRUCK** | Post-hoc RED. test_dense_depth.py (00:17:31) was written after dense_depth.py (00:15:55) and after Chase's board finding 5013 had published 30 hops at n=1024. There is no stub RED. D = n-1 stands on Chase's B3, not here. |

## Struck

- **Chase**: scratchpad/phase_j/N1/chase/test_n.py::T3 (T3b diagonal unchanged) CERTIFIED (check 4): Post-hoc bar. The first real run (run1.log, 00:07:00, board 4934) was RED on bitwise diagonal equality. test_n.py was then edited to re_diag_max_diff <= 1e-12, with the comment 'bitwise was a defect of this test'. That edit is at mtime 00:08:04, after the RED was seen, and the GREENs at 4964/4977 rest on the loosened bar.
- **Chase**: T8 softmax cells 'CERTIFIED pattern' (0.0083 & 0/448; 0.120 & 697/98,119) (check 2): Unbound. No test event asserts the softmax cells; the report's own column reads 'stub RED -> recorded'.
- **Chase**: Rounding-manufactured unit diagonals (2 in f64, 13 in f32), board 5015 (check 2): Unbound: no test event, RED or otherwise, carries this finding.
- **Chase**: Lean read-only finding (det_of_lowerTriangular at Block.lean:265; no charpoly-conjugation lemma), board 5016 (check 2): Unbound: no RED precedes the finding. The same facts stand as Wilson [V] (section 5), not as Chase's claim.
- **Foreman**: P2 STRUCK: (f)'s |W_ic| 0.0056 (check 3): Contradicts [V]. rsink_measure.f_attn computes '(f)'s |W_ij| at beta=1', but Wilson [V] gives (f)'s trained beta as 1.061/0.930/0.900. The measured operator is not (f)'s.
- **Foreman**: FT_f_relief GREEN ((f) first-token mass 0.00225/0.00163/0.00194) (check 3): Contradicts [V]: the (f) masses come from f_attn at beta=1, while (f)'s beta is 1.061/0.930/0.900. Foreman had already marked it VOID.
- **Foreman**: N_beta_convention: beta=1 does not hold on (f) (board 5007) (check 4): Post-hoc bar. rsink_run.log (00:03:46) already printed beta_f 1.0605748891830444, which alone fails 0.01. test_beta.py was written at 00:09:03. The fact stands as Wilson [V] (1.061/0.930/0.900); the binding is struck.
- **Foreman**: 'layers 1-2 never close (minimum m 0.119 / 0.129)' and 'hard-concrete floor at m ~ 0.0497' (check 3): Contradicts [V]. On the same 64x512 windows (generator 20260921, split 0; medians 0.6730/0.4406/0.5300 identical), Wilson's minimum m is 0.04717 / 0.09669 / 0.09255. The reported 'minimum' and 'floor' are 0.1% quantiles.
- **Foreman**: 'the family's own regime (exact zeros, absorbers, NEVER) fires on 0.066% of site-layers' (check 3): Contradicts [V]: Wilson's table has 0 exact zeros of m in all three layers.
- **Cameron**: R-COST-N side rows: Neumann K=130 128.6x, fs2g 2.27x, fs_block256 3.13x, fs_dense 8.01x (check 2): Unbound: the only test events are for hop_cheb and fs5g_c256.
- **Cameron**: Two torch-level routes lost: blocked-doubling 2.14x, inverse-ahead 2.19x (board 5033) (check 2): Unbound: no RED precedes the finding.
- **Cameron**: '(a) grows as the leak law says: CERTIFIED' (check 2): Unbound: no test event asserts the leak law (Chase's T8 does, but on his own instance).
- **Cameron**: Dense causal depth: 'D5 as numerical cost STRUCK (26-30 hops)' + structural n-1 (board 5028/5029) (check 4): Post-hoc RED. test_dense_depth.py (00:17:31) was written after dense_depth.py (00:15:55) and after Chase's board finding 5013 had published 30 hops at n=1024. There is no stub RED. D = n-1 stands on Chase's B3, not here.

## Re-runs

- **Chase:** test_n.py was re-run from `inspector/chase_rerun`, with BOARD redirected so that no events were forged under Chase's name.
  - Stub: 27/27 RED, exit 1.
  - Real: every status and printed number is identical to run2.log, exit 1, 5 s wall.
- **Foreman:** all 21 test/JSON pairs (stub and real) reproduce the board lines exactly.
- **Cameron (CPU):** four runs, each matching the board.
  - never_len --stub --quick: RED 6.830e-02.
  - never_len full: P1 [0.0 x 15]; counter RED at gap 0.0000; 59 s.
  - transfer --stub: RED -0.9984.
  - transfer: RED 0.0093; 57 s.
- **dense_depth:** 18/26 and 22/30 hops, the same as reported.
- **GPU:** poll_until_free(600) returned True (0 compute procs), then test_cost_n.py ran once per impl.
  - fs5g_c256: ratio **1.757** (SDPA 4.510 ms, impl 7.924 ms), rel err 2.959e-06, RED.
  - hop_cheb: ratio **128.548** (SDPA 4.558 ms, impl 585.899 ms), rel err 4.458e+04, RED.

## Not re-run

- Foreman's measurement scripts (rsink_measure, window_measure, const_measure, p5_read, read_beta): the tests read their saved JSON, and the saved JSON and logs were audited instead. The scripts write into Foreman's directory by `HERE`. They are not on the cost list.
- Cameron's explore_fs*.py profiles, cheb_check.py and ssmax_leak.py: none is a test, and they back only claims already struck as unbound.
