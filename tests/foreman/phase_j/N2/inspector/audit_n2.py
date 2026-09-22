"""Inspector N2: append one audit event per audited claim to the board."""
import json
BOARD = "C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
E = [
 ("Foreman", "clean", "R-POS PASS: a_rope recovers 0.9632/0.9971/1.0143 (mean 0.9915)",
  "Check 1: test_rpos.py re-run from N2/inspector on CPU. --stub exits 1 with the board's line 5094 verbatim. The real run exits 0 and prints 0.9632/0.9971/1.0143 and 'R-POS verdict PASS', matching board 5131. The recoveries recomputed from design4x5_results.jsonl (inspector/recompute_rpos.py) agree to 4 d.p. at 3/3 seeds."),
 ("Foreman", "clean", "a_alibi recovers 1.0884/1.0671/1.0882; ALiBi twin beats (f) by +0.0206/+0.0160/+0.0231 nats (board 5133, numbers)",
  "The same GREEN run prints these, so they bind. Recomputed loss(f)-loss(a_alibi) = +0.0206/+0.0160/+0.0231. The only registered bar that names ALiBi is KILL (<0.5), and it cannot fire."),
 ("Foreman", "clean", "CRN digests and parameter parity, 7/7 rows of rpos_results.jsonl",
  "For each row, crn_digest_recomputed == crn_digest_file == grid (a) digest == grid (f) digest (d36f6436/e5c8ab9a/ee102d79 by split seed). n_params is 724,608, equal to grid (a), in 7/7 rows. The grid holds no duplicate cells."),
 ("Foreman", "clean", "R-POS bar registered before the run (check 2)",
  "RECORD_N.md mtime is 00:34:22, before test_rpos.py (00:37:05) and the first cell (00:39:45). Its PASS clause (RoPE >= 0.8 at seeds 0-2) matches the test and the report. The test's KILL clause adds 'at all three seeds'; that does not bear on a PASS. Stub RED 5094 precedes every Foreman result, 5131-5135. RECORD_N.md is untracked, so this rests on mtime. The header's 'N1 ... 00:47' is later than the file's own mtime and is a mislabel (N1 FINAL.md is 00:33:09)."),
 ("Foreman", "clean", "a_rope ss0 cell provenance across the 00:50:14 rpos.py edit",
  "The ss0 RoPE cell (00:39:45) ran before rpos.py was rewritten. The only surviving copy of that version is __pycache__/rpos.cpython-311.pyc, compiled from the 00:36:55 source. Its bytecode (inspector/pyc_diff.py) is identical to the current source for rope_cs, rotate, rope_forward, alibi_mask, alibi_forward, check and recomputed_digest. Only run() differs, which is the --cell dispatch."),
 ("Foreman", "clean", "Controls: (a) re-run 1.2756859 vs grid 1.2756839; wall-clock ratios",
  "These are control rows read from rpos_results.jsonl, not findings. The difference is 1.937e-6 nats, or 8.3e-6 of C_win. run_seconds and grid_a_run_seconds match the report's table. The 0.21x-of-(f) clock is 101.8-110.7 s against (f)'s 513.8-519.4 s."),
 ("Foreman", "struck", "FoX's data dependence buys nothing measurable over fixed ALiBi here (board 5134; report 'What the result means')",
  "Unbound. No RED or test event names it, N2 did not run FoX, and no bar for ALiBi vs FoX was registered. The FoX figures (100.2/100.3/110.9%) match RECORD.md:60, but 'nothing measurable' is an untested inference. FoX leads at seed 2."),
 ("Foreman", "struck", "'(a) was the only arm in the grid with no relative-position mechanism; every bar scored against (a) measured that handicap' (board 5133 claim text; report 'Standing consequence')",
  "Unbound: no test event asserts it. For the record, the grid's a2 (design4x5.py:195-201) is (a)'s softmax plus a post-SDPA output gate, which adds no relative prior either."),
 ("Foreman", "struck", "Code audit by sonnet nurse: RoPE Toeplitz to 2.4e-6, ALiBi exactly -slope*(i-j), rebinding restored",
  "Unbound. The nurse left no artifact in N2/foreman, and no test event carries these numbers. (The pyc diff binds only the claim that the arm code did not change.)"),
 ("Foreman", "struck", "poll_until_free stalls multi-cell loops (board 5135)",
  "Unbound: a finding with no RED. rpos_run.log does show 32 polls at compute_procs=1, used 201 MiB, after cell 1. It can be re-filed with a test."),
 ("Foreman", "clean", "Check 4: Foreman vs WILSON_REPORT.md",
  "No contradiction. No R-POS claim touches beta, gate zeros, BOS or the operator conventions Wilson verified. The 724,608 (a) params match the grid."),
 ("Chase", "clean", "R-DIAG prime PAST A POLE: gamma* < 0.99 in 424/496/512 of 512 pairs (1,432/1,536 = 93.2%), gamma* min 0.8286/0.7071/0.4836",
  "Check 1: copies of test_rdiag_prime.py (only the board path changed) re-run on CPU from N2/inspector. The stub gives 7/7 RED NotImplementedError, exit 1. The real run gives 5 GREEN and 2 RED, exit 1, and the RED line matches board 5115 verbatim. The output JSON is identical to Chase's apart from 'seconds'. sha256 prefixes 9dab02a7 and 9e4f1a4e match the report."),
 ("Chase", "clean", "R-DIAG prime bar registered at 00:38 before measuring (check 3)",
  "test_rdiag_prime.py mtime is 00:38:08 and the file is unchanged since (sha matches). The stub RED is at 00:38:13. Run 1 (00:39:46) failed on the CPU guard with empty AssertionErrors. The first read is read_run2.log (00:41:37), and N1 holds no measurement of these gamma* values. The bar precedes its result."),
 ("Chase", "clean", "Fidelity prerequisites: NLL 1.0430563, attn-out max diff 0.0, learned beta 1.0605749/0.9300133/0.8995796",
  "The four GREENs reproduce, and they are preceded by stub REDs 5097-5100 under the name rule. eval_nll 1.0430562864777626 and attn_out_max_abs_diff 0.0 match."),
 ("Chase", "clean", "Re(W) identical: diag bitwise equal, imag 0, ReW RED 424/496/512",
  "test_diag_real_and_ReW_diag_equal is GREEN and test_bar_well_posed_gamma_099_ReW is RED. Both reproduce verbatim (board 5114, 5116)."),
 ("Chase", "clean", "No absorber excl. pos 0; unit diagonals 0/0; excl-pos-0 7/495/512 (1,014/1,536 = 66.0%); Neumann growth 2.0470 = 0.99 x 2.0677",
  "test_no_absorber_but_position_0 is GREEN and reproduces. The remaining values are fields of the same bound run's JSON (excl_pos0, unit_diagonals, worst_f64.neumann_growth_per_step 2.0470052), and the re-run reproduces them exactly."),
 ("Chase", "clean", "beta pinned to 1 is not free: NLL 1.0430563 -> 1.0652995, +0.0222 nats = 9.6% of C_win (test_beta_pin_free RED)",
  "Stub RED 5122 (00:43:47) precedes. test_beta_pin.py mtime is 00:43:47, before the first measurement (pin_run1.log, 00:44:48). The re-run exits 1 with the RED line verbatim, and beta_pin.json is identical. 0.022243/0.232628 = 0.0956. The docstring's 'Registered 00:45' disagrees with the file's mtime; the mtime shows the bar predates the first measurement."),
 ("Chase", "clean", "Ruling: beta-pin GREEN re-run (board 5125) binds; the voided GREEN 5123 stays void",
  "Under the name rule, stub RED 5121 [beta_pin_stub] binds GREEN 5125 [beta_pin]: same file and unit. The test file was not edited after the RED. The fix was to beta_pin.py only (00:45:01). The bug changed only trained_gstar_min (0.934 vs 0.4836); beta1_gstar_min was 1.0 in both runs. The re-run reproduces GREEN. It is a theorem check: at beta = 1, W_ii = p_ii <= 1 and W_00 = 1, so it carries implementation sanity only."),
 ("Chase", "struck", "R-DIAG prime second implementation agrees (board 5128, nurse1/wii_check.py)",
  "Unbound: a finding with no RED. nurse1/wii_check.log does print W_ii = 1.2068454693 (diff 1.6e-11). The headline is bound independently."),
 ("Chase", "clean", "Check 4: Chase vs WILSON_REPORT.md",
  "No contradiction. beta (1.0606/0.9300/0.8996), qk (1.0872/1.3335/1.3895), exact gate zeros 0 and min m 0.0472/0.0967/0.0925 all agree with Wilson section 3. Correction: the report's OPEN calls position 0 'the BOS row'. Wilson section 4 verified that the bed has no BOS token, so position 0 is an arbitrary byte. The numbers concern position 0 and stand."),
]
if __name__ == "__main__":
    with open(BOARD, "a", encoding="utf-8") as fh:
        for cites, v, name, why in E:
            fh.write(json.dumps({"t": "audit", "agent": "Inspector", "cites": cites, "verdict": v,
                                 "name": name, "why": why}) + "\n")
    print(len(E), "audited,", sum(v == "struck" for _, v, _, _ in E), "struck")
