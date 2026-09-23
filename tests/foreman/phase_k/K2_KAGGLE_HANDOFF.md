# K2 identity grid on Kaggle: handoff

Written by Wilson (Phase K it.K2), 2026-09-23, 15:40 IST, before the laptop restart. It covers the registered
identity-grid jobs that had not landed locally, as listed in Cameron's `jobs_remaining.json`, plus one control
job. These jobs run on two Kaggle kernels. Every claim cites a command output or a file.
`SP` = `C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad`.
It is a temp directory, and it might not survive the restart (unverified).

## Kernels

All three kernels are private, with internet off, `machine_shape` `NvidiaTeslaT4`, and pushed with `--accelerator NvidiaTeslaT4`.
Each push printed `Kernel version 1 successfully pushed` (`SP/phase_k/K2/wilson/push_{smoke,a,b}.log`).
The runtime check found 2 x Tesla T4 with 15360 MiB each, driver 580.159.04, torch 2.10.0+cu128, CUDA 12.8,
cuDNN 91002, Python 3.12.13 and numpy 2.0.2. `torch.cuda.is_bf16_supported()` returned True.

| slug | version | status at 15:39 IST | lane (GPU) | jobs, in order | estimated lane wall |
|---|---|---|---|---|---|
| `melowdramtic/k2-id-grid-smoke` | 1 | COMPLETE (77 s) | 0 / 1 | smoke only, see below | n/a |
| `melowdramtic/k2-id-grid-a` | 1 | RUNNING (lanes started 10:06:26 UTC) | 0 | id_fR_ga_s0, id_fR_ga_s0_walk, id_fR_sp_s0, id_fR_sp_s0_walk, id_aL4_s0 (CONTROL) | 31,897 s (8.86 h) |
| | | | 1 | id_fR_ga_s1, id_fR_ga_s1_walk, id_fR_sp_s1, id_fR_sp_s1_walk, id_aL4_s1 | 31,897 s (8.86 h) |
| `melowdramtic/k2-id-grid-b` | 1 | RUNNING (lanes started 10:06:27 UTC) | 0 | id_fR_ga_s2, id_fR_ga_s2_walk, id_fR_sp_s2, id_fR_sp_s2_walk, id_aL7_s1 | 32,045 s (8.90 h) |
| | | | 1 | id_ass4_s1, id_aloop5_s1, id_aL4_s2, id_aL7_s2, id_ass4_s2, id_aloop5_s2 | 32,868 s (9.13 h) |

The two kernels hold 21 of the 21 entries in `jobs_remaining.json`: 15 training runs (14 grid runs and 1 control) and 6 walks.
The live logs (`kaggle kernels logs -f`) show 10/10 `SHA OK` and 7/7 `FP OK` in both kernels, followed by the start of both lanes.
The earliest possible finish is about 19:00 to 19:15 UTC (00:30 to 00:45 IST on 24 Sep), if the estimates hold.

The kernels run each job as follows:
- **Argv.** Each job runs with Cameron's registered argv. The only change is the paths: `SP` maps to `/tmp/sp`, and the cwd is `/tmp/sp/phase_k/K2/cameron`.
- **Outputs.** `/tmp/sp/phase_k/K2/cameron/runs` is a symlink to `/kaggle/working/runs`, so every run lands at
  `/kaggle/working/runs/<launch name>/`.
- **Control.** The control job keeps its launch name `id_aL4_s0`. Its `--out` points at `runs_kaggle_control/id_aL4_s0`, which puts it
  at `/kaggle/working/runs_kaggle_control/id_aL4_s0/`. Cameron's note says it must never go into `runs/`.
- **Files kept.** Each run keeps `result.json`, `log.jsonl`, `ok_{4096,8192,16384}.npz` and `hardware.json`.
  - fR runs also keep `model.pt` and `walk/{ptr_4096,ptr_16384}.npz, meta.json`.
  - The control keeps `model.pt` because Cameron's `bring_back` list names it.
  - `model.pt` is deleted for the other twins.
  - `runs/<name>.stdout` holds each job's stdout.
  - `/kaggle/working/kernel_status.json` holds the sha checks, the fingerprints, and the per-job rc, wall and status.
- **`hardware.json`.** It records the GPU name, torch, CUDA, cuDNN, nvidia-smi output, kernel slug and
  `kernel_version_expected` (1, which matches the push output), plus the start and end times, rc and wall.
- **Time guards.** A job is skipped, and recorded as skipped in `kernel_status.json`, if elapsed time plus its estimate is over 41,400 s.
  A running job is killed at 42,300 s (11.75 h), so the kernel can exit and save its outputs.
  A walk is skipped if its training job failed.

## Smoke (`k2-id-grid-smoke` v1): all green

The smoke ran the registered argv with `--steps 50` for four jobs. Two ran concurrently on each lane:
- smoke_aL4_s1
- smoke_fR_sp_s1, with its walk
- smoke_fR_ga_s1, with its walk
- smoke_fR_ga_g999_s1 (`--gamma_sched 0.999`, for measurement only)

Results, from `SP/phase_k/K2/wilson/out/smoke_v1/kernel_status.json` (the smoke is also downloadable from Kaggle):
- **Integrity checks.** 10/10 source sha256 matched. 7/7 of Cameron's bed fingerprints matched: far10 n=4096/8192/16384,
  heldout_1024, and train seeds 0/1/2 steps 1-5.
- **Jobs.** 6/6 jobs exited with rc 0. That includes both walks, which exercise the fR_ga fs5c path and the fR_sp sparsemax path.
- **Peak memory.** The peak was 4.88 GiB at eval n=16384.
- **Kaggle test.** `test_k2_grid_kaggle.py id` was run against copies of these smoke runs. The `config_hw` row failed on `steps`
  only, which is expected for a 50-step run. It printed `(device Tesla T4)`, and it accepted the recorded hook sha and file name.
- **Walk checks.** For both fR smoke runs, `walk/meta.json` `ckpt_sha256` equals the sha of `model.pt`, the ptr depth arrays equal
  K1 `far_fR_ga_s0`'s, and the pointer ranges are valid.

## Slowdown factor and estimates

Rates are the mean s/step from step 1 to step 50 in `log.jsonl`. On Kaggle, two jobs ran concurrently, one per T4.
Locally, the jobs ran one at a time on the RTX 4060.

| arm | Kaggle T4 | local RTX 4060 | ratio |
|---|---|---|---|
| aL4 (twin) | 0.1772 | 0.0599 (id_aL4_s0) | **2.96x** |
| fR_ga, gamma 0.5 | 0.1912 | 0.0783 (id_fR_ga_s0) | 2.44x |
| fR_ga, gamma 0.999 from init | 0.2014 | n/a | n/a |
| fR_sp at grid shape | 0.1973 | unmeasured locally at grid shape | n/a |

**Factor used: 2.96.**

Per-job estimates:
- **Twins.** Local wall x 2.96: aL4 5,577 s, aL7 5,725 s, ass4 5,218 s, aloop5 5,565 s.
- **fR_sp.** 0.1973 x 24,000 x 1.055 x 1.24 = 6,200 s. The last two factors come from the pilot's local ratios:
  overall rate / first-50 rate, and wall / train.
- **fR_ga.** 20,000 s = the local projected wall (about 6,750 s) x 2.96.
  - The local id_fR_ga_s0 ran at train_s 5,952 at step 23,150. Its rate was 0.071 s/step up to step 12,000 and 0.44 to 0.47 s/step after.
  - At gamma 0.999 from random init, the Kaggle rate was only 0.2014 s/step. So the local late-phase cost comes from the trained
    state, and the smoke could not measure its T4 cost (see Limits).
- **Walks.** 60 s each; the smoke walks took 8.6 s and 11.5 s.

## Quota

`kaggle quota` read 0.69 h used before the smoke, 0.71 h after it, and 0.82 h about 3.3 min after kernels a and b both started.
The step from 0.71 h to 0.82 h is 0.11 h. That equals 2 sessions x 3.3 min, which is consistent with quota counting session time at 1x
rather than per GPU.

Estimated GPU-hours: a = 8.86 h, b = 9.13 h, about 18.0 h in total. The worst case, where both kernels hit the 11.75 h guard, is 23.5 h.
At 15:39 IST the remaining quota was 29.18 of 30.00 h, and it refreshes 2026-09-26T00:00:00.

## Pushed files (sha256)

Kernel sources: `SP/phase_k/K2/wilson/kernels/<name>/`. `kaggle kernels pull <slug> -p <dir>` also returns the exact source.

| file | sha256 |
|---|---|
| k2-id-grid-a/run.py | beda22d406e4fd6e170c1f79df4e7c20393cfd5d52f394b9987547324c25a3f2 |
| k2-id-grid-a/kernel-metadata.json | 382654a7998b706cdd68f07c7835e349d1b8fc9f7b1ba262e561d824d099c75a |
| k2-id-grid-b/run.py | ef85d9792cf2d59baa69305363c77e519e07408853c55d83729aa72cc81f7b1b |
| k2-id-grid-b/kernel-metadata.json | 785203a483caec86720c56b77c691a0ff6e372f71105c2037c9663ac3b1b5b93 |
| k2-id-grid-smoke/run.py | 5ce161c3dcab9317e47ea9bfbd116e4d8a10ceb5781500d4dce7c512293e2598 |
| k2-id-grid-smoke/kernel-metadata.json | dbb764e66078ab591438125a9def06c01ee8582b418177f8881587c9bcd566ab |
| generator `SP/phase_k/K2/wilson/mk_kernels.py` | 2d1bbed26bb21541e74bb102bb38a375c29066f7ee2e621b40a0c5ec4e74b203 |
| template `SP/phase_k/K2/wilson/driver_template.py` | a19c1df813b8b6bdd716e00f1eae2008f1e7bb28186e93eec60f8645a01ee4f0 |

Each `run.py` embeds the following sources byte-identical. The kernel checks each sha on disk before it runs anything:

| embedded source | sha256 |
|---|---|
| SP phase_k/K1/cameron/rdepth.py | 6a413119a322a1f22d0488b88326eba7ee7fbc295f8205bf3bf6150119c764df |
| SP phase_k/K1/cameron/train_ladder.py | d524b13005d86bd49ae1d7692d41de4e0030542f7a19a212fb562de916758d93 |
| SP phase_k/K1/cameron/bed_k.py | aa60e6d963307d57a4db7ce99a785ec45b75814e6ebdcd8c8650a2174ecb320f |
| SP phase_k/K1/cameron/bed_kp.py | 20d50b9d71248cd39aff74219b831b2c513d6044dfc0ca7ecc744a7a6a323355 |
| SP phase_k/K2/cameron/k2_ptr.py | b02a81360c87da78cbc0ec6891cb46559addea70ae3477dbd0cf372686ef5151 |
| SP phase_k/K1/chase/resolvent_hook.py (fR_ga) | 3855288d7da7acdebf54b9b1979d9b0b5afebb3dca6d8ccefd684812d33451e2 |
| SP phase_k/K2/chase/resolvent_sp.py (fR_sp) | 36039c3a4189189840d04b345fa3c54d43e6d32bcc8a176ce5e8aac135bfd7bc |
| repo tests/foreman/phase_k/K0/chase/resolvent.py | f850c568f3454c36224325d7b28df6a081a0e32069b860188882b6a6374ceccd |
| repo tests/foreman/phase_k/K0/wilson/train_ladder.py | d524b13005d86bd49ae1d7692d41de4e0030542f7a19a212fb562de916758d93 |
| repo tests/foreman/phase_j/N1/cameron/cost_n.py | 9ab22197eaad779a51a1b8e1cff294babe7811f12bca2aba53ea2d481fbec93a |

`cost_n.py` is missing from Cameron's `dependency_sha256` list, but the kernels need it: K0 `resolvent.fs5c` imports it through the relative
`CAMERON` path. It is placed at `<cwd>/C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_j/N1/cameron/cost_n.py`, next to the
two K0 files. All three repo files are git-tracked and had no local modifications (`git status --short` was empty for them).

## After the restart: status, download, placement, test (Git Bash)

```bash
SP="C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
kaggle kernels status melowdramtic/k2-id-grid-a
kaggle kernels status melowdramtic/k2-id-grid-b
# optional, live: kaggle kernels logs -f melowdramtic/k2-id-grid-a
```

**0. If SP is gone.** Restore the test's inputs and the four landed local runs from the repo copy:

```bash
B="C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/K2/cameron_local_runs"
mkdir -p "$SP/phase_k/K2/cameron/runs" "$SP/phase_k/K1/cameron/runs/far_fR_ga_s0"
cp -n "$B/_test_inputs/K2_cameron/"* "$SP/phase_k/K2/cameron/"
cp -n "$B/_test_inputs/K1_cameron/k1fppp.json" "$SP/phase_k/K1/cameron/"
cp -n "$B/_test_inputs/K1_cameron/runs/far_fR_ga_s0/"*.npz "$SP/phase_k/K1/cameron/runs/far_fR_ga_s0/"
for r in id_aL4_s0 id_aL7_s0 id_ass4_s0 id_aloop5_s0; do cp -rn "$B/$r" "$SP/phase_k/K2/cameron/runs/"; done
```

**1. Download.** Run this only once both kernels read `KernelWorkerStatus.COMPLETE`:

```bash
mkdir -p "$SP/phase_k/K2/kaggle_out/a" "$SP/phase_k/K2/kaggle_out/b"
kaggle kernels output melowdramtic/k2-id-grid-a -p "$SP/phase_k/K2/kaggle_out/a"
kaggle kernels output melowdramtic/k2-id-grid-b -p "$SP/phase_k/K2/kaggle_out/b"
```

**2. Place.**
- `runs/<name>` goes to `SP/phase_k/K2/cameron/runs/<name>`.
- The control goes to `SP/phase_k/K2/cameron/runs_kaggle_control/id_aL4_s0`, never to `runs/`.
- A local run that has landed by the ward's own rule is never overwritten. The ward's rule is `result.json`, plus `walk/meta.json` for fR runs.
- A local partial directory is renamed `<name>.local_partial`. The test reads only the exact `id_<arm>_s<seed>` names, so the renamed directory is ignored.

```bash
python - <<'EOF'
import shutil
from pathlib import Path
K2 = Path("C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K2")
for k in "ab":
    for sub in ("runs", "runs_kaggle_control"):
        for d in sorted((K2 / "kaggle_out" / k / sub).glob("*")):
            dst = K2 / "cameron" / sub / d.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if d.is_file():
                shutil.copy2(d, dst if not dst.exists() else dst.with_name(d.name + ".kaggle")); continue
            if (dst / "result.json").exists() and ("fR" not in d.name or (dst / "walk" / "meta.json").exists()):
                print("SKIP, landed locally:", dst); continue
            if dst.exists():
                dst.rename(dst.with_name(dst.name + ".local_partial"))
            shutil.copytree(d, dst); print("placed", dst)
EOF
```

**3. Test.** Cameron's rule in `jobs_remaining.json` is: run once, after all 18 grid runs are in `runs/`.
Use `test_k2_grid_kaggle.py`, sha 3ff92f9f2c55db15b401c134d0deac5085cc6a70a79318fe97320108831d7366, with the same env the ward used:

```bash
cd "$SP/phase_k/K2/cameron" && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python test_k2_grid_kaggle.py id runs
```

Besides `runs/`, the test reads `SP/phase_k/K1/cameron/k1fppp.json` and
`SP/phase_k/K1/cameron/runs/far_fR_ga_s0/ok_{4096,16384}.npz`. Step 0 restores both.

## Local runs copied into the repo

`tests/foreman/phase_k/K2/cameron_local_runs/` (23 MB) contains:
- **Four landed grid twins.** id_aL4_s0, id_aL7_s0, id_ass4_s0 and id_aloop5_s0, each with `result.json`, `log.jsonl` and
  `ok_{4096,8192,16384}.npz`, plus each run's `.stdout`. Their `model.pt` files are not copied.
  The local walls were 1884, 1934, 1763 and 1880 s (`ward_grid_id.log`).
- **`pilot_fR_sp_s0`.** A complete copy, including `model.pt` and `walk/`, plus its `.stdout`.
- **`_test_inputs/`.** `test_k2_grid.py` (a622cd95), `test_k2_grid_kaggle.py` (3ff92f9f) and its `.sha256`, `jobs_remaining.json`,
  `grid_id_jobs.json`, `ward.py`, `ward_grid_id.log`, `k2_ptr.py`, `mk_jobs_remaining.py`, K1 `k1fppp.json`, and K1
  `runs/far_fR_ga_s0/ok_{4096,8192,16384}.npz`.

The local id_fR_ga_s0 had not landed when this file was written. At 15:39 IST its directory held `log.jsonl` only, and
`ward_state.json` read `landed: 4`. Kernel a runs its own id_fR_ga_s0. If the local run and its walk land before the restart, the
placement step keeps the local copy and skips the Kaggle one.

## Limits

- **fR_ga estimate.** The 20,000 s fR_ga estimate is unverified: the smoke could not reproduce the local late-phase cost.
  If fR_ga overruns by more than about 1.5x, the time guard skips the jobs after it in that lane, and they appear as `skipped` in
  `kernel_status.json`. A kernel whose total goes past 11.75 h loses only the job that is running at that point.
- **12 h limit.** That Kaggle stops batch GPU sessions at 12 h is from memory, not checked in this session.
- **Output on timeout.** Whether Kaggle keeps `/kaggle/working` when a session is stopped externally is unverified.
- **Quota at 1x.** That quota counts session time at 1x is inferred from one 0.11 h step, not from Kaggle documentation.
- **T4 numerics.** bf16 autocast on the T4 (sm_75) ran without error. No bar compares T4 numerics with the RTX 4060 runs yet.
  Cameron's control, id_aL4_s0 in `runs_kaggle_control/`, exists for that comparison, and his note says a bar must be registered
  before it is read.
- **Duplicate fR_ga_s0.** If the local id_fR_ga_s0 lands, there are two copies of it. This file does not decide which one the grid reads.
- **SP survival.** Whether SP survives the restart is unverified.
