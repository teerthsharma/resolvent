"""Inspector K2 pass 2: append one audit line per claim, then done. Single-line appends only."""
import json
B = "C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
RR = "Inspector CPU rerun of the byte-identical copy (9996d651) on runs/pilot_fR_sp_s0: identical to run_test_k2_pilot.log (9 fails, exit 1); N3 own-code recompute equal to 4 decimals and N"
A = [
 ("Cameron", "clean", "Ack hold: pilot_pipeline.py launches no grid by construction (board 6252)",
  "pilot_pipeline.py (sha 185fb913) runs lockjob pilot, lockjob walk, test_k2_pilot.py, board test lines, then exits; no grid argv; no id_*/sp_* run dir under cameron/runs; no rdepth process at 11:24; K1/gpu.lock.d absent"),
 ("Cameron", "clean", "pilot_config GREEN (board 6253)",
  RR + "; config = REG on every key; hook K2/chase/resolvent_sp.py sha 36039c3a on disk = config hook_sha256 = READY_SP"),
 ("Cameron", "clean", "pilot_trained GREEN 0.9971 (board 6254)", RR),
 ("Cameron", "clean", "pilot_learnable GREEN 0.9949 (board 6255)", RR),
 ("Cameron", "clean", "pilot_walk_machinery GREEN (board 6256)",
  RR + "; meta ckpt_sha256 9cde1891 = model.pt; ptr causal at both n; N3 regenerated beds bed_kp.make_test([31,n,k]) equal root and depth 8/8 at 4096 and 16384"),
 ("Cameron", "clean", "pilot_walk_ceiling RED 0.8949 / 0.4513 / 0.2498 (board 6257)",
  RR + "; N5 CPU fp32 re-extraction of bed 0 at 4096 from model.pt agrees with the saved ptr on 8191/8192 (diagnostic)"),
 ("Cameron", "clean", "pilot_far10 RED 0.3046 / 0.1503 / 0.0829 / 0.0325 (board 6258)", RR),
 ("Cameron", "clean", "pilot_multilen16k RED 0.1537 (board 6259)", RR),
 ("Cameron", "clean", "pilot_grid_launch RED (board 6260)", RR),
 ("Cameron", "clean", "finding: pilot read, sha 9996d651 unchanged, 9 failing rows, gate and read row values (board 6261)", RR),
 ("Cameron", "clean", "BRANCH: IDENTITY (walk ceiling 0.8949 / 0.4513 / 0.2498) (board 6261, run_test_k2_pilot.log)",
  "rerun identical; branch code equals RECORD_K K2.1 table (8e1bd52, unchanged at HEAD); N4 ran every path on synthetic worlds with the byte-identical copy: perfect pointer GRID (ceiling 1.0 x3, so the gate can pass), one cell below IDENTITY, real arrays IDENTITY, untrained REPEAT_S1, s1 untrained SP_RETIRED, bad config NONE, bad ckpt sha NONE"),
 ("Cameron", "clean", "test_k2_pilot.py sha 9996d651 unchanged since its RED",
  "mtime 10:21:12; test_k21.sha256 10:22:42; RED 1 10:22:42, RED 2 10:23:12; pilot launch 10:42:32; sha on disk now 9996d651"),
 ("Cameron", "clean", "the branch rule the test applies = RECORD_K 'K2.1, registered RED before any run' (8e1bd52)",
  "GRID iff every gate GREEN; IDENTITY iff trained and some ceiling < 0.95; REPEAT_S1; SP_RETIRED: same in both; the test also gates config and walk machinery and adds NONE, which RECORD_K's table does not list; both GREEN here, so both give IDENTITY; RECORD_K K2.1 text identical at 8e1bd52 and c9daf85"),
 ("Cameron", "clean", "pilot ran the registered hook (36039c3a, resolvent_sp.py:ResolventAttention) and anneal 0.5:6000,0.9:12000,0.99:18000,0.999",
  "result.json config = log.jsonl config (one config line): that hook, sha, class and gamma_sched, 24000 steps (last step record 24000, end record), seed 0; N5: gamma_wrap(ResolventAttention) MRO G > ResolventAttention, gamma 0.5 / 0.9 change the output (6.1e-1 / 2.2e-1), 0.999 equals None exactly, gamma_at switches at 6000 / 12000 / 18000; :FRSP would have bypassed the schedule"),
 ("Cameron", "clean", "pointer-walk extraction under the GPU lock after the 180 s cooldown, from the final checkpoint",
  "pilot released 11:15:56.17, walk acquired 11:18:56 (lockjob waits 180 s from the lane's last release, then atomic mkdir K1/gpu.lock.d), released 11:19:05; rdepth.py has one torch.save, after the step loop and before the final eval; model.pt (11:15:49) is the only checkpoint; meta ckpt_sha256 = model.pt 9cde1891, hook 36039c3a"),
 ("Cameron", "clean", "test_k2_grid.py a622cd95 unchanged since its RED; id rows exist; no grid run dir",
  "sha on disk a622cd95 (mtime 10:21:52); CPU rerun 'id' on cameron/runs: 133 rows, 133 fail (= RED 1): config 18, trained 18, learnable 6, r1 1, prediction 24, multilen16k 6, walk_ceiling 18, counter_quiet 12, twin_under_line 12, opponents_quiet 18; no id_*/sp_* dir under cameron/runs (stub_grid/ only, 10:23:11)"),
 ("Cameron", "clean", "CAMERON_REPORT verdict: 'K2.1 pilot: BRANCH IDENTITY', learns 0.9971, ceiling 0.8949 / 0.4513 / 0.2498, far10 0.3046 / 0.1503 / 0.0829 / 0.0325, hold in force (11:19:41)", RR),
 ("Cameron", "clean", "CAMERON_REPORT K2.1 item 4: test unchanged, 9 rows fail, hook resolvent_sp.py sha 36039c3a named by READY_SP (11:19:41)", "rerun, config and READY_SP"),
 ("Cameron", "clean", "CAMERON_REPORT timings: lock 10:42:32-11:15:56; walk 11:18:56-11:19:05, 180 s after release (11:19:41)", "pilot_fR_sp_s0.stdout, k2_pilot_walk.log, lock_releases.log"),
 ("Cameron", "clean", "CAMERON_REPORT: the pilot ran the schedule registered before READY_SP, not READY_SP's usage step counts; noted on the board (11:19:41)", "config gamma_sched 0.5:6000,...; test registered 10:21:12, READY_SP 10:41:49 with 0.5:2000,...; board 6169 carries the note"),
 ("Cameron", "clean", "CAMERON_REPORT pilot table, 8 rows (11:19:41)", RR),
 ("Cameron", "clean", "CAMERON_REPORT: registered branch IDENTITY, the Dispatcher's hold applies, no grid launched (item 4 and standing state, 11:19:41)", "BRANCH rerun identical; no grid run dir; board 6252 and 6261"),
 ("Cameron", "struck", "CAMERON_REPORT 'What the rows bind': 'Sparsemax training did not repair pointer identity. The 16k ceilings (0.4513 / 0.2498) sit within 0.03 of the K1-b softmax pointer's (0.4498 / 0.2292)' (11:19:41)",
  "no row compares the pilot with K1-b; the 0.03 line was set after both were read; it leaves out the 4k cell, which moved -0.0398 (0.8949 vs 0.9347); the pilot also differs from K1-b in steps (24,000 vs 8,000), so no row attributes the gap to sparsemax training. Bound form: the pilot's own hard-walk ceiling is below 0.95 on all three cells"),
 ("Cameron", "clean", "CAMERON_REPORT: 'did not bring the soft read up to its own ceiling', far10 0.3046 vs ceiling 0.8949 at 4k (11:19:41)", "both are rows on the same 22464 tokens (depth > 160, n 4096)"),
 ("Cameron", "clean", "CAMERON_REPORT: 'sparsemax support is wide at test length' stays labelled (item 4 'Unbarred hypothesis, no row reads it'; OPEN 'the unbarred guess') (11:19:41)", "labelled in both places, absent from the board finding; not counted as a finding"),
 ("Cameron", "clean", "CAMERON_REPORT Limits: the pilot stub could not exercise the IDENTITY and GRID paths; they ran first on the real pilot (11:19:41)", "PASS1 RED-2 record (stub BRANCH NONE, READY_SP absent); N4 has now exercised every path"),
 ("Cameron", "struck", "CAMERON_REPORT standing state: 'The pointer's identity is the 16k wall on both the softmax and the sparsemax checkpoint (0.4498 / 0.4513), and its wrong links follow position' (11:19:41)",
  "re-asserts PASS1's struck 'At 16k the wall is the pointer itself' (walk_explains RED at 16k, 0.7995 < 0.9); no walk_explains row ran on the pilot, whose 16k soft read (0.0829) sits far below its ceiling (0.4513); lanes ran on K1's far_fR_ga_s0 only. Bound form: the hard-walk ceiling caps depth > 160 at 16k at 0.4498 (K1-b) and 0.4513 (pilot)"),
]
with open(B, "a", encoding="utf-8") as f:
    for cites, verdict, name, why in A:
        f.write(json.dumps({"t": "audit", "agent": "Inspector", "cites": cites, "verdict": verdict, "name": name, "why": why}) + "\n")
    f.write(json.dumps({"t": "done", "agent": "Inspector"}) + "\n")
print(len(A), sum(v == "struck" for _, v, _, _ in A))
