# Inspector, Phase K it.K2, pass 2

27 claims audited this pass, 2 struck: Cameron board 11 / 0, process 5 / 0, CAMERON_REPORT 11 / 2.

**The BRANCH line is bound. `BRANCH: IDENTITY (walk ceiling 0.8949 / 0.4513 / 0.2498)` re-runs to the same status and numbers. The rule that prints it equals the one registered in RECORD_K at 8e1bd52. It also prints GRID on a world where the gate passes, so this IDENTITY is not a check that could not pass.**

Scope: board lines 6252–6262 (after pass 1's `done` at 6251); files in `K2/cameron/` modified after 10:56 (CAMERON_REPORT.md 11:19:41, pilot_pipeline.log, run_test_k2_pilot.log, k2_pilot_walk.log, lock_releases.log, `runs/pilot_fR_sp_s0/`). Board: `dispatch` 6263, nurse lines 6264–6266 (N3, N4, N5, sonnet), audit lines 6267–6293, `done` 6294.

## Priority items

| item | result |
|---|---|
| test_k2_pilot.py rerun | **Identical.** The inspector copy (sha 9996d651, the same as the lane file) ran on CPU over `cameron/runs/pilot_fR_sp_s0`. Its output equals `run_test_k2_pilot.log` line for line (CR ignored): config, trained 0.9971, learnable 0.9949 and walk_machinery PASS; walk_ceiling 0.8949 (N 22464) / 0.4513 (N 120768) / 0.2498 (N 49088) FAIL; far10 0.3046 / 0.1503 / 0.0829 / 0.0325 FAIL; multilen16k 0.1537 FAIL; `BRANCH: IDENTITY`; 9 failing rows, exit 1. The pilot's input files hash the same before and after the rerun. |
| Independent recompute (N3) | **Equal to 4 decimals and in N** for all ten values. N3 wrote its own walk code and did not read the test. The meta ckpt sha equals model.pt (9cde1891). ptr is causal at both n. The beds regenerated from `bed_kp.make_test(default_rng([31, n, k]))` equal the saved root and depth on 8 of 8 beds at 4096 and at 16384. Per head, not ORed: 0.7486 / 0.7877 at 4k, 0.2868 / 0.3128 and 0.1307 / 0.1342 at 16k. |
| Branch paths (N4) | **Every path prints as registered.** N4 ran the byte-identical copy on synthetic worlds in `nurse4/`: a perfect pointer gives ceilings 1.0 ×3 and GRID; the real arrays give IDENTITY; a perfect 4k with the real 16k gives IDENTITY; trained set to 0.5 gives REPEAT_S1, and at seed 1 SP_RETIRED; a bad gamma_sched gives NONE; a bad ckpt sha gives NONE. The GRID world also shows the far10 and multilen16k rows do not gate: they fail there, and GRID still prints. |
| Branch rule vs RECORD_K | **Equal on this outcome.** Both give GRID when every gate is GREEN, IDENTITY when the pilot learned and some ceiling is < 0.95, REPEAT_S1 at seed 0 and SP_RETIRED at seed 1. The test is stricter than RECORD_K's table: it also gates config and walk machinery, and it has a NONE path. Both of those gates are GREEN here. The K2.1 text is identical at 8e1bd52 and HEAD c9daf85, and the working tree has no diff. |
| Hook and anneal | **Registered.** result.json config equals the single config line in log.jsonl: hook `K2/chase/resolvent_sp.py:ResolventAttention`, hook_sha256 36039c3a (still 36039c3a on disk), gamma_sched `0.5:6000,0.9:12000,0.99:18000,0.999`, 24,000 steps (the last step record is 24000, followed by an end record), seed 0. N5 checked that the schedule reaches the layer. gamma_wrap(ResolventAttention) has MRO G → ResolventAttention. γ 0.5 and 0.9 change the output (max abs 6.1e-1 and 2.2e-1), and γ 0.999 equals the unscheduled output exactly. gamma_at switches at 6000, 12000 and 18000. With `:FRSP`, `__new__` returns a plain ResolventAttention and the schedule is bypassed; the pilot did not use it. |
| Walk extraction | **Held the lock, waited the cooldown, read the final checkpoint.** The pilot released the lock at 11:15:56.17, and the walk acquired it at 11:18:56 and released it at 11:19:05. lockjob waits 180 s from the lane's last release, then takes K1/gpu.lock.d by atomic mkdir. rdepth.py has one `torch.save`, after the step loop and before the final eval. model.pt (11:15:49) is the only checkpoint in the run dir. meta.json records ckpt 9cde1891, equal to model.pt, and hook 36039c3a. N5 re-extracted bed 0 at 4096 from model.pt on CPU in fp32; it agrees with the saved ptr on 8191 of 8192 positions. The GPU path ran the lower layers under bf16, so this is a diagnostic, not a bar. |
| test_k2_grid.py | **Unchanged; id rows present; no grid run.** The sha is a622cd95 (mtime 10:21:52). The CPU rerun of `id` on `cameron/runs` gives 133 rows and 133 fails, equal to RED 1. Rows by family: config 18, trained 18, learnable 6, r1 1, prediction 24, multilen16k 6, walk_ceiling 18, counter_quiet 12, twin_under_line 12, opponents_quiet 18. There is no `id_*` or `sp_*` dir under `cameron/runs`; the only `id_*` dirs are the RED-2 stubs in `stub_grid/` (10:23:11, nothing newer). K1/gpu.lock.d was absent at 11:24, and no rdepth process was running. |

## Cameron board: 11 audited, 0 struck

6252 (ack of the hold: pilot_pipeline.py launches no grid). pilot_pipeline.py has sha 185fb913 and no grid argv.

6253–6260 (the eight test lines). Every family's status matches its rerun rows.

6261 (the finding). Its values, and the BRANCH line counted as a claim of its own.

## Process: 5 audited, 0 struck

test_k2_pilot.py is unchanged since its RED. The branch rule equals RECORD_K's. The pilot ran the registered hook, class and anneal. The walk ran under the lock after the cooldown, on the final checkpoint. test_k2_grid.py is unchanged, its id rows exist, and no grid dir exists.

## CAMERON_REPORT.md (11:19:41): 11 audited, 2 struck

Struck:
- Item 4, "What the rows bind": "Sparsemax training did not repair pointer identity. The 16k ceilings (0.4513 / 0.2498) sit within 0.03 of the K1-b softmax pointer's (0.4498 / 0.2292)."
  - No row compares the pilot with K1-b, and the 0.03 line was set after both were read.
  - The sentence leaves out the 4k cell, which moved −0.0398 (0.8949 against 0.9347).
  - The pilot also differs from K1-b in steps (24,000 against 8,000), so no row attributes the gap to sparsemax training.
  - Bound form: the pilot's own hard-walk ceiling is below 0.95 on all three cells.
- Standing state: "The pointer's identity is the 16k wall on both the softmax and the sparsemax checkpoint (0.4498 / 0.4513), and its wrong links follow position."
  - This re-asserts pass 1's struck "At 16k the wall is the pointer itself" (walk_explains RED at 16k, 0.7995 < 0.9).
  - No walk_explains row ran on the pilot. Its 16k soft read (0.0829) sits far below its ceiling (0.4513).
  - The lanes test ran on K1's `far_fR_ga_s0` only, so "its wrong links follow position" does not bind the sparsemax checkpoint.
  - Bound form: the hard-walk ceiling caps depth > 160 at 16k at 0.4498 (K1-b) and 0.4513 (pilot).

Clean:
- the verdict's K2.1 bullet;
- item 4's sha, fail count and hook;
- the timings (10:42:32–11:15:56; 11:18:56–11:19:05, 180 s);
- the schedule note (board 6169 carries it);
- the eight-row pilot table;
- "registered branch IDENTITY, hold applies, no grid launched";
- "did not bring the soft read up to its own ceiling" (0.3046 against 0.8949, both rows on the same 22,464 tokens);
- the Limits line saying the stub could not exercise the IDENTITY and GRID paths (N4 has now exercised them).

The wide-support guess stays labelled. Item 4 calls it "Unbarred hypothesis, no row reads it", OPEN calls it "the unbarred guess", and the board finding does not carry it. It is not counted as a finding. "About 10 GPU-hours" in OPEN is an estimate, not a finding.

## Process notes

- **Three pass-1 strikes are still in the report text at 11:19:41:**
  - verdict bullet 1, "so leakage is most of the 4k wall";
  - verdict bullet 4, "At 16k the wall is the pointer itself … So sparsemax on this pointer cannot clear the 16k band";
  - Kill 3, "Sparsemax … is the measured repair for the leak part".

  They stay struck. The report was edited after pass 1 without amending them, and the new standing-state sentence restates the second one.
- **READY_SP's usage line gives `--gamma_sched 0.5:2000,0.9:4000,0.99:6000,0.999`.** The pilot ignored it and ran the schedule the test registered at 10:21:12, and the config gate would have failed on any other value (N4 world F). Cameron declared this at board 6169.
- **The pilot's probes ran at the scheduled γ**, because rdepth sets `HOOK["gamma"] = None` only before the final eval. No row reads the probes.

## Records

- **Reruns.** All ran in `K2/inspector/`, CPU only (`CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`). The tests ran as byte-identical copies, checked by sha. Logs:
  - `rerun2/pilot_meas.log` and `rerun2/grid_id_now.log`;
  - input hashes in `rerun2/pilot_inputs_{before,after}.sha`;
  - `nurse3/recompute.{py,log}`;
  - `nurse4/*.log` and `nurse4/build_worlds.py`;
  - `nurse5/{ptr_cpu,gamma_check}.{py,log}`.
- **Hashes after all runs.** test_k2_pilot.py (both copies) 9996d651, test_k2_grid.py a622cd95, resolvent_sp.py 36039c3a. Every file under `runs/pilot_fR_sp_s0/` is unchanged, and nothing under `K2/cameron/` or `K2/chase/` is newer than the rerun.
- **GPU.** The Inspector and the nurses used none, and did not touch the lock.
- **Repo.** No repo writes except board appends, and no git writes.

## Standing state

- The pilot rows are bound: IDENTITY is the registered branch, and it re-runs on CPU from the saved outputs.
- The two struck report sentences do not touch any row or the BRANCH line.
- Nothing in this pass blocks the `test_k2_grid.py id` launch. The Dispatcher's release is the only thing left.
