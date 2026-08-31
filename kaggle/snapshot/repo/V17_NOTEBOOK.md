# V17_NOTEBOOK.md — what `kaggle/ceq_v17k.ipynb` does, cell by cell

This file documents the Kaggle notebook, its dependencies, and the launch
checklist the author runs through by hand. It does not launch anything —
nobody who wrote it called the `kaggle` CLI, pushed a kernel, or touched
`~/.kaggle`. The author launches, on his own say-so, only after Gate 0 is
green.

**Governing rule, restated because it is the point of the whole notebook:**
the certified local RTX 4060 DECIDES every research number in round v17-K.
Kaggle only REPRODUCES. Every Q1/Q2/Q4 cell prints its own number beside the
local one with a `|Delta|` column and is labelled REPRODUCTION, never a
verdict.

---

## 1. The cell sequence, in order

Each numbered cell refuses to proceed if the one before it did not — either
by raising through a `gate(name, ok, msg)` helper (defined in cell 1 and
reused by every later cell) or simply by letting a real exception (a failed
`assert`, a `MissingPin`, a `HygieneViolation`, a non-zero `!` shell exit
captured in `_exit_code`) propagate, which halts a Kaggle "Run All" at the
cell that actually broke.

| # | cell | what it does | halts on |
|---|---|---|---|
| 1 | Environment | prints python/torch version, CUDA availability, per-device free/total VRAM | no CUDA device visible |
| 2 | Clone | resumes an already-cloned `/kaggle/working/ceq-project`, else copies a code-snapshot Dataset, else `git clone` + `checkout PINNED_SHA` (needs Internet ON); asserts `git rev-parse HEAD == PINNED_SHA` | `PINNED_SHA` unset, no code source configured, or the SHA does not match |
| 3 | Install | checks `transformers`/`datasets`/`huggingface_hub`/`accelerate`/`chess` against `requirements.txt`'s pins; installs only what is missing (needs Internet ON if anything is) | — (installs are best-effort; later cells fail loudly if the environment is still wrong) |
| 4 | Smoke | `python -m ceq.hf.smoke` — 7 CPU-only checks, no GPU needed | any of the 7 checks fails |
| 5 | Dataset paths | prints the `/kaggle/input/...` mount points from the author's attach list | (informational only) |
| 6 | Hash verify | `ceq.kdata.verify_file` / `verify_digest` on every attached file; `ceq.kdata.bed_signature` for BED-M/BED-K/BED-1, all three regenerated, none uploaded | any digest mismatch, or `MissingPin` on the three sources that start unpinned |
| 7 | Real-data smoke | runs `ceq.kdata`'s five hygiene guards against the first ~100 REAL Lichess games and the first few real enwik8 `<page>` boundaries — the guards' own test suite (`tests/gate0/test_g05_data.py`) only ever sees a 36-game synthetic fixture and a 24-article synthetic enwik8 shape | a real game a guard cannot parse/split/replay, or an enwik8 boundary a guard rejects |
| 8 | K-CERT | runs `scripts/k_cert.py` fresh on this Kaggle GPU, compares its `bar.worst.delta_over_tol` against the local certified reading (`results/k_cert_local.json`, committed at `PINNED_SHA`) | the local certificate is missing/malformed, the Kaggle run HALTs internally, or its worst `delta/tol` clears 50% (`k_cert.HALT_FRACTION`) |
| 9 | Q1/Q2 | `scripts/v15_r1.py --device cuda`, diffed cell-by-cell (`kind`, `seed`) against the local `results/v15_r1.jsonl` reading | local reference file missing |
| 10 | Q3 calibration | builds `/kaggle/working/corpus.txt` from the L1-verified enwik8 train split; runs `ceq.hf.train.preflight` then 50 real training steps to measure `s/step`; derives `CHUNK_STEPS` from an 11h chunk cap | the shape does not fit `T4-16GB`'s budget |
| 11 | Q3 chunk | auto-detects the newest `/kaggle/working/ceq-run-NNN` checkpoint and resumes from it (fresh run if none); trains one chunk; tees stdout to `/kaggle/working/train.log` | `ceq.hf.train.train`'s own preflight/`resume_from==out_dir` guards |
| 12 | Q4 | `scale/m3_quintuple.py --device cuda`, then `python -m scale.capability_table --journal ... --version kaggle_v17k`, journal preserved and diffed against the local `results/m3_quintuple_v2_cuda.jsonl` | local reference journal missing |

---

## 2. Dependencies — status as measured against this repo at authoring time

This section moved twice while the notebook was being written, because six
other agents were landing files in the same tree concurrently. What follows
is the state as last checked; **re-run the commands in each row before
trusting it**, because more of this may have changed since.

| dependency | status when this notebook was authored | check it yourself |
|---|---|---|
| `ceq/kdata.py` (G0.5 K-DATA) | **Implemented.** All 29 of its own tests pass. | `python -m pytest tests/gate0/test_g05_data.py -q` |
| `scripts/k_cert.py` (G0.6 K-CERT) | **Implemented** as a module — its pure functions (`fit_power_law`, `delta_over_tol`, `halt_on_bar`, `zero_step_gate`, `determinism_verdict`, …) pass their own unit tests. | `python -m pytest tests/gate0/test_g06_kcert.py -q -k "not certificate"` |
| `results/k_cert_local.json` | **Present but incomplete against the schema the tests (and this notebook) expect.** The file that exists has top-level keys `schema, when, box, min_r2, halt_fraction, cell_ns, arms, git, throughput, elapsed_s, git_end` — `bar.worst` is `null`, and `memory`, `zero_step`, `determinism` are absent entirely. 4 of 13 certificate-reading tests fail against it (`test_certificate_carries_both_refits_...`, `test_certificate_worst_delta_over_tol_clears_the_fifty_percent_line`, `test_certificate_zero_step_gate_fired_...`, `test_certificate_reports_determinism_in_both_regimes`). This notebook's K-CERT cell checks `bar.worst` is a `dict` before indexing into it and gives a named error rather than a bare `TypeError` if it is not — but it cannot make the launch decision on a certificate this shape, and neither should the author. | `python -m pytest tests/gate0/test_g06_kcert.py -q` |
| `ceq/hf/train.py` — atomic checkpoint write (G0.3 K-PERSIST) | **Solved.** `_atomic_torch_save` (tmp file + `fsync` + `os.replace`) now backs `trainer_state.pt`; `train()` also refuses `resume_from == out_dir`. 7/7 `tests/gate0/test_g03_persist.py` pass. | `python -m pytest tests/gate0/test_g03_persist.py -q` |
| `ceq/hf/train.py` — periodic mid-chunk checkpoint | **Still absent.** `train()`'s signature has no `save_every` (or equivalent) parameter; the checkpoint write happens exactly once, at the end of the full `steps` loop, atomically but only once. A session killed *before* the loop finishes still loses the entire chunk, not just its tail. The Q3 chunk cell (`kaggle/ceq_v17k.ipynb` cell 17) is written against `T.train(..., save_every=N)` and uses it via `inspect.signature` detection if it ever lands; until then it warns loudly and runs unsafely. **Keep `CHUNK_STEPS` well under the 11h line, not right up against it, until this lands.** | `python -c "import inspect; from ceq.hf import train as T; print('save_every' in inspect.signature(T.train).parameters)"` |
| `ceq/autopilot.py` (G0.8/G0.9, out of this notebook's scope) | Exists — a merkle-sealed, tier-1/2/3 decision envelope for supervising a long multi-chunk run from a polled JSONL training journal. This notebook does **not** emit that journal format or call `ceq.autopilot.decide()`; the task that produced this notebook scoped it to the L1/L2/L3 gate structure only. Flagging as a real, related capability that a future revision of this notebook could wire into the Q3 cell (emit one `PollEvent`-shaped JSON line per checkpoint) — not built here to avoid a construction outside the given scope. | `python -m pytest tests/gate0/test_g08_autopilot.py tests/gate0/test_g09_mars.py -q` |

### What the author must confirm (not verifiable from this repo alone)

- **Q1/Q2's instrument is an inference, not a name given in the task.**
  `scripts/v15_r1.py --device cuda` is the only in-tree, CUDA-capable runner
  that already produced a certified-device reading (recent commits
  "Certify CUDA, refute the 5.8x gap", "Price the deciding cells on the
  certified device") and its record schema (`t="cell"` rows keyed by
  `(kind, seed)`, fields `eval_nrmse`, `eval_h_hat`, `conservation_drift`,
  `a_hat_max`) is what the Q1/Q2 cell reads. If v17-K's actual Q1/Q2 name a
  different deciding measurement, swap the command and `NUMERIC_FIELDS` in
  that cell.
- **`results/v15_r1.jsonl`, committed today, is `device: "cpu"`.** No
  `device: "cuda"` v15_r1 reading exists anywhere in `results/` as of this
  writing. The Q1/Q2 cell's `LOCAL_Q1Q2_JSONL` points at this file with a
  loud comment to confirm it is actually the certified CUDA reading before
  the notebook is trusted — if the certified reading lives under a
  different filename or tag, repoint that variable.
- **Filenames inside each attached Kaggle dataset.** `arevel/chess-games`,
  the pinned `lichess/chess-evaluations` shard, and
  `alexkarev/tinystories-train-ready` all have their in-dataset filenames
  left as `<pinned-filename>` placeholders in cells 7–9 — a Kaggle Dataset's
  internal layout is not knowable until it is attached. Fill these in from
  the real directory listing after attaching.

---

## 3. The pinned SHA slot

`kaggle/ceq_v17k.ipynb` cell 3 (`# 2. L1 -- repo cloned at a pinned SHA`)
defines:

```python
PINNED_SHA = ""          # <-- AUTHOR FILLS THIS IN BEFORE PUSHING. Required.
REPO_URL = ""             # e.g. 'https://github.com/<you>/<repo>.git'
CODE_DATASET_DIR = ""     # e.g. '/kaggle/input/ceq-repo-snapshot', if attached
```

An empty `PINNED_SHA` halts the notebook at cell 3 (`gate("PINNED_SHA is
set", ...)`). At minimum `results/k_cert_local.json` (schema-complete),
`results/k_data_manifest.json` (all three Kaggle-side pins filled in), and
`results/v15_r1.jsonl`/`results/m3_quintuple_v2_cuda.jsonl` (confirmed as
the certified readings) must exist **at the commit `PINNED_SHA` names** —
pin a SHA before those land and every downstream gate halts by design.

---

## 4. The launch checklist (run through by hand before pushing anything)

- [ ] `results/k_cert_local.json` is schema-complete: `python -m pytest
  tests/gate0/test_g06_kcert.py -q` is 13/13, not 9/13.
- [ ] `results/k_data_manifest.json`'s three Kaggle-side sources
  (`lichess_chess_games`, `lichess_chess_evaluations`, `tinystories_cdla`)
  are `status: PINNED` with a real `sha256`, not
  `UNPINNED_AWAITING_KAGGLE` — this needs one dry run of the L1 hash cell
  against the real attached files first (it will halt with the digest
  printed; paste that digest in, commit, re-run).
- [ ] the Q1/Q2 local reference (`results/v15_r1.jsonl` today) is confirmed
  as the certified **CUDA** reading, not the CPU-tagged file currently
  committed under that name.
- [ ] `git rev-parse HEAD` at commit time is pasted into `PINNED_SHA`.
- [ ] the four struck dataset slugs are NOT attached
  (`robikscube/this-week-in-chess-archive`,
  `dimitrioskourtikakis/gm-games-chesscom`,
  `thedevastator/tinystories-narrative-classification`,
  `nightfury1103/enwik8`).
- [ ] BED-M is **not** uploaded as a dataset — it, BED-K and BED-1 all
  regenerate from `ceq.corpus.build` / `ceq.beds.bed_k` / `ceq.beds.bed_1`
  at a pinned seed. (An earlier draft of this notebook had this wrong; see
  §5.)
- [ ] enwik8 is attached as a private Dataset containing the extracted
  100,000,000-byte `enwik8` file (pinned sha256
  `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`) —
  the zip download's own integrity, if re-fetched from
  `http://mattmahoney.net/dc/enwik8.zip`, is a separate check the author
  runs before upload, outside this notebook.
- [ ] GPU accelerator selected in Notebook Settings (T4 x2 or P100); this
  notebook only ever uses one device (`device="cuda"`, the default CUDA
  device) — no multi-GPU code path exists in `ceq/hf/train.py` and none was
  added here.
- [ ] decide the clone path for cell 3: attach a code-snapshot Dataset (no
  network needed for the rest of the session) or plan to toggle Internet ON
  for cells 3–4 only, then off again.
- [ ] `CHUNK_STEPS` in the Q3 cells is sized conservatively — periodic
  mid-chunk checkpointing does not exist yet (§2), so a chunk that dies
  before it finishes loses all of it.

---

## 5. What in `TRAINING.md` / `colab/train_ceq.ipynb` is stale or wrong

Findings from reconciling both documents against the current tree, some
independently confirmed by `V17_G04_ENV.md` (a parallel environment audit):

- **`colab/train_ceq.ipynb` cell 1 pins `transformers>=5.0`.**
  `requirements.txt` and `TRAINING.md` §6.1 both pin `transformers==5.3.0`
  tight, for a stated reason (`ceq/hf/modeling_ceq.py` sets six undocumented
  `PreTrainedModel` internals a later release, including a patch, can
  rename). The Colab notebook's unbounded `>=5.0` can install a
  transformers that silently drops one of the six. `V17_G04_ENV.md` "B3"
  independently names this same defect. `kaggle/ceq_v17k.ipynb`'s install
  cell uses the tight pin.
- **`colab/train_ceq.ipynb` cell 1 also installs `accelerate`, which no
  module in this repo imports.** `ceq/hf/train.py` hand-rolls its training
  loop (`torch.save` of the optimizer state directly); it does not use
  `transformers.Trainer`. `V17_G04_ENV.md` "B3" and `requirements-kaggle.txt`
  both independently name this a dead install line. Not carried into the
  Kaggle notebook.
- **`TRAINING.md` §6.4's data cell (`load_open_text(...,
  streaming=True)`) cannot run under this contract's "internet OFF" rule.**
  `datasets.load_dataset(..., streaming=True)` fetches shards over the
  network at iteration time — `V17_G04_ENV.md` "B4" measured this
  independently and calls it a hard blocker. `kaggle/ceq_v17k.ipynb`'s Q3
  cell instead builds its corpus from the L1 hash-verified enwik8 train
  split (already read into memory for the L1/real-data-smoke cells) — no
  second read, no network.
- **`TRAINING.md` §6.3 says "On Kaggle's P100 pass `gpu="T4-16GB"` anyway —
  it is the more conservative budget of the two and `sizing.GPUS` has no
  P100 entry."** Confirmed still true: `ceq/sizing.py`'s `GPUS` dict has
  keys `A100-40GB`, `A100-80GB`, `L4-24GB`, `T4-16GB`, `RTX4060L-8GB` only.
  Carried into the Kaggle notebook's preflight cell as written.
- **`TRAINING.md` §4's chunk-sizing rule (`chunk_steps = floor(0.80 *
  session_cap_seconds / r)`) uses the 12h session cap.** This task's own
  spec is stricter — "chunked ≤11h" — so `kaggle/ceq_v17k.ipynb` applies
  the same `0.80` margin on top of an 11h cap instead of TRAINING.md's
  9.6h-effective figure (`0.80 * 12h`), which is slightly more
  conservative and matches the literal "≤11h" requirement rather than
  deriving it.
- **Neither document anticipated `ceq/kdata.py`, `scripts/k_cert.py`, or
  `results/k_data_manifest.json`, all three of which now exist and change
  the correct shape of the data/install cells.** Not a defect in either
  document — they predate this round — but a reason not to copy their
  cells verbatim; §1 above is the reconciled sequence.

---

## 6. What this file does not establish

- Nothing here was run on an actual Kaggle GPU. Every claim about what a
  cell does is a static read of its source plus a local, CPU-side
  verification that every symbol it calls resolves in this repo (see the
  task's "VERIFY WHAT YOU CAN, LOCALLY" instruction) — never an execution
  of the notebook end to end.
- The tree this notebook was authored against changed repeatedly while it
  was being written (six other agents landing files concurrently, as
  `V17_G04_ENV.md` also documents about its own measurement window). §2's
  "Dependencies" table is a snapshot, not a standing guarantee — re-run its
  "check it yourself" commands before trusting any row.
