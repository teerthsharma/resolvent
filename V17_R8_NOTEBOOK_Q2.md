# V17_R8_NOTEBOOK_Q2.md — Q2 removed from `kaggle/ceq_v17k.ipynb`, per RULING 8

Filed by the node that executed RULING 8's consequential edit ("`kaggle/
ceq_v17k.ipynb` still carries a Q2 cell"). Scope was fixed by the work order:
`kaggle/ceq_v17k.ipynb`, `kaggle/README.md`, this file — nothing under `ceq/`,
`scripts/`, `scale/`, `tests/`, and none of `V17K_RULINGS.md`, `COSTS.md`,
`MODEL_CARD.md`, `V17_NOTEBOOK.md`, `V17_GPU_QUEUE.md`, `MISTAKES.md`,
`LOOP_PROMPT.md`, `CEQ_V16_CONTRACT.md`. No writing git command was run.
`git rev-parse HEAD` was `ab5b48547884e04258276e6e808d5a71ea65f917` at both
start and end of this edit (unchanged — nothing was committed).

---

## 1. What "the Q2 cell(s)" turned out to be

`grep -c "Q2" kaggle/ceq_v17k.ipynb` read **8** before this edit, across the
raw notebook JSON. Read cell-by-cell (`json.load`, not text grep) rather than
by eye, those 8 hits resolved to **no dedicated Q2 code cell at all** — Q1 and
Q2 shared one code cell (former index 14) that only ever invoked
`scripts/v15_r1.py --device cuda` with its **default** arguments
(`--arms arm_pl softmax`, `--n-train 2048`, the fixed `T_STAR = 2` module
constant). That is Q1's shape (BED-M, `t*=2`, `n=2048`, per the script's own
docstring), not Q2's (`t*=8`, `n=16,384`, `arm_smprime`) — the notebook never
contained code for Q2's actual shape, because `scripts/v15_r1.py` has no
`--t-star` flag to produce it (RULING 8's own secondary note). "Q2" existed
only as a name: in a shared markdown header, an inline confirm-this note, a
code comment, and three Python identifiers (`LOCAL_Q1Q2_JSONL`,
`KAGGLE_TAG = "v17k_kaggle_q1q2"`), plus one mention each in the top-of-file
gate-order list and the closing "what this notebook does not do" section.

**Consequence for the edit:** there was no code cell to delete. The edit is a
rename (strip "Q2" from every name/label that described the shared Q1/Q2 cell,
since only Q1 ever ran there) plus one new tombstone cell, not a cell
deletion.

The 8 original hits, located by cell index (`json.load`, not the raw grep
line numbers, which don't map 1:1 to cells):

| # | old cell idx | cell type | what the "Q2" text was |
|---|---|---|---|
| 1 | 0 | markdown | gate-order list item 5, "Q1/Q2 reproduce..." |
| 2 | 13 | markdown | Q1/Q2 section header |
| 3 | 13 | markdown | "Confirm this is v17-K's actual Q1/Q2..." |
| 4 | 14 | code | cell comment, "# 7. Q1 / Q2 -- reproduction..." |
| 5 | 14 | code | `LOCAL_Q1Q2_JSONL = ...` |
| 6 | 14 | code | `gate(..., LOCAL_Q1Q2_JSONL.exists(), ...)` |
| 7 | 14 | code | `local_rows = _load_jsonl(LOCAL_Q1Q2_JSONL)` |
| 8 | 20 | markdown | "Every Q1/Q2/Q4 number is a reproduction..." |

## 2. What changed

Old cell sequence (21 cells, indices 0–20) → new cell sequence (22 cells,
indices 0–21). **No cell was renumbered in the sense the task forbids** — the
notebook's own inline numbering (`# 1.` … `# 10.` on its code cells) is
untouched; Q3's cells are still `# 8.` and `# 9.`, Q4's is still `# 10.`. The
new cell is unnumbered prose, not a renumbering of the sequence.

| old idx | new idx | change |
|---|---|---|
| 0 | 0 | markdown, gate-order item 5 reworded: "Q1 reproduces the local deciding reading (Q2 was dropped by RULING 8 -- see the tombstone cell in the queue below)" |
| 13 | 13 | markdown header `### Q1 / Q2 -- reproductions of the local deciding readings` → `### Q1 -- reproduction of the local deciding reading`; inline "Q1/Q2" → "Q1" |
| 14 | 14 | code: comment reworded, `LOCAL_Q1Q2_JSONL` → `LOCAL_Q1_JSONL` (3 uses), `KAGGLE_TAG = "v17k_kaggle_q1q2"` → `"v17k_kaggle_q1"`. **The file path VALUE was left unchanged** (`results/v15_r1.jsonl`) — see section 5, this was NOT re-pointed, deliberately |
| — | **15 (new)** | **markdown tombstone** — full text in section 3 |
| 15–19 | 16–20 | unchanged (Q3 header/cells, Q4 header/cell) |
| 20 | 21 | markdown "what this notebook does not do": "Every Q1/Q2/Q4 number" → "Every Q1/Q4 number" |

No cost or ETA arithmetic in the notebook needed recomputation. Checked for
GPU-hour / quota / chunk-budget expressions that might have priced Q2 in:
`grep -n "GPU-h\|quota\|budget\|ETA\|11h\|12 h" kaggle/ceq_v17k.ipynb` — the
only arithmetic in the notebook is the Q3 calibration cell's `CHUNK_STEPS =
int(0.80 * CHUNK_CAP_H * 3600 / r_secs_per_step)`, which derives entirely from
a **measured** `r_secs_per_step` (50 real training steps, timed on the
session's own GPU) and the task's own `11.0`h cap — it never referenced Q2 or
depended on Q2's shape. `[MEASURED, read from kaggle/ceq_v17k.ipynb cell 17
("# 8. Q3 -- calibrate...")]`. **No number needed recomputing; none was
invented.**

## 3. The tombstone (new cell, index 15)

Placed between the Q1 code cell (14) and the Q3 header (16), so the queue
reads Q1 → *Q2 tombstone* → Q3 → Q4 in cell order, matching
`V17_GPU_QUEUE.md`'s post-ruling queue (K-CERT → Q1 → Q3 → Q4 → cross-device
table). Verbatim:

> ### Q2 -- DROPPED (RULING 8)
>
> **There is no cell here to run.** Q2 (R2 reproduction, `t*=8`, `n=16,384`)
> is dropped, not skipped. The gap in the queue between Q1 above and Q3 below
> is deliberate and is the honest record of what happened -- not a
> renumbering. This repo's L-G2 principle: superseded and dropped things are
> marked, never erased.
>
> **Why.** `arm_smprime` at `n=16,384` reserves **10.578 GiB** under a
> training loop against the certified RTX 4060's **7.996 GiB** total
> `[MEASURED, COSTS.md section 1.6 / results/k_cert_local.json]`, so the
> shape does not fit the certified device. The same shape fits a T4 (59% of
> its budget), which places Q2's deciding cell exactly where this round's
> KILL clause strikes it: **local decides, and local cannot run it.** There
> was no re-take to price because there was nothing to re-take. Ruling filed
> verbatim in `V17K_RULINGS.md`, RULING 8.
>
> **What this does not settle.** RULING 8 states plainly that dropping the
> *reproduction* does not supply the *reading* -- whether the round carries
> an R2 row at all, given R2's deciding cell cannot run on the certified
> device either, is flagged there as a separate, open question, not
> answered by this tombstone or by silence.
>
> Q1, Q3, Q4 below keep their names; nothing is renumbered.

The two GiB figures are `[MEASURED]`, cited to `COSTS.md` section 1.6 and
`results/k_cert_local.json` (both files this node was told not to edit, only
to cite — see `V17K_RULINGS.md` RULING 8's own text for the same two
numbers). The "59% of a T4's budget" figure is carried from `V17K_RULINGS.md`
RULING 8's own prose ("The same shape is 59 % of a T4's budget") rather than
independently recomputed by this node.

## 4. Mechanical verification

Two checks, both run against the edited file, neither by eye:

**(a) The file is valid JSON and opens as a notebook.**
```
python -c "import nbformat; nb = nbformat.read('kaggle/ceq_v17k.ipynb', as_version=4); nbformat.validate(nb); print(len(nb.cells), 'cells')"
```
Result: `nbformat.validate` raised nothing; **22 cells**. (`nbformat==5.10.4`,
present in the environment.)

**(b) Per-cell static symbol resolution, in notebook order.** A script
(`ast`-based, no execution) walks every code cell in order, maintains a
cumulative namespace of names bound by that cell or any earlier one (imports,
assignments, `def`/`class`, `for`/`with`/comprehension targets, function
parameters), and flags any `Name` in `Load` context that resolves to neither a
Python builtin nor a name bound at or before its own cell. IPython line-magics
(`!shell`, `%magic`) are replaced with `pass` before parsing (to keep block
structure syntactically valid without evaluating shell content, which is out
of scope for a Python-symbol check).

Result: **1 flagged name, `_exit_code`, in cell 5** (`# 3b. Cheap CPU-only
package smoke test`) — this is IPython's own implicit post-`!`-command exit
code, an interpreter builtin at runtime, not a name any cell defines; it
predates this edit (untouched cell) and is the same pattern
`V17_NOTEBOOK.md` section 1 describes ("a non-zero `!` shell exit captured in
`_exit_code`"). **No unresolved name in any cell this edit touched or
introduced (0, 13, 14, 15, 16-20 unchanged, 21).** No cross-cell ordering
defect of the kind the task described (a symbol used before any cell defines
it) was found anywhere in the notebook.

## 5. Findings for other nodes — flagged, not touched

Two items surfaced while reading cell 14 that are **not** RULING 8's scope
(Q2 removal) and were left alone, per the work order's "NO NEW CONSTRUCTIONS"
and "do not touch / do not report [the R1′ re-take node's files] as yours":

- **Cell 14's `LOCAL_Q1_JSONL` (renamed from `LOCAL_Q1Q2_JSONL`, section 2)
  still points at `results/v15_r1.jsonl`.** `V17_R4_RETAKE.md` section 9
  ("Findings for other nodes — flagged, not touched") independently names
  this exact defect and explicitly declines to fix it itself ("`kaggle/` is
  not this node's to edit"): the certified CUDA reading is now
  `results/v17k_r4_retake.jsonl`, confirmed present at edit time (424 lines,
  header `device: "cuda"`, `deterministic_algorithms: true`,
  `deterministic_warn_only: true`, `cublas_workspace_config: ":4096:8"`,
  `arms: ["arm_pl", "arm_smprime", "softmax"]` — read directly from the file,
  not taken on report). `results/v15_r1.jsonl` is now `git status`-modified
  (an append-only supersede marker per L-G2, per `V17_R4_RETAKE.md`), which
  this node independently confirmed (`git status --porcelain` shows `M
  results/v15_r1.jsonl`) without reading its content, since that file is the
  re-take node's, not this one's. **This node did not repoint
  `LOCAL_Q1_JSONL`.** Repointing it changes what Q1 reproduces against (a
  correctness fix, not a Q2 removal) and touches the boundary of a file this
  node was told to leave alone; it belongs to whichever node owns Q1's
  correctness, or to the author's own pass.
- **The same cell invokes the instrument with default `--arms` (`arm_pl
  softmax`)**, so it never runs `arm_smprime` — also named by
  `V17_R4_RETAKE.md` section 9. Adding a third arm to what the notebook
  actually executes on Kaggle is a behavior change to Q1, not a repair of
  what pointed at Q2, and is out of this node's scope for the same reason.

A background task was flagged for a separately-scoped follow-up to repoint
`LOCAL_Q1_JSONL` at `results/v17k_r4_retake.jsonl` and add `arm_smprime` to
cell 14's arms list, citing `V17_R4_RETAKE.md` section 9 as the source.

## 6. Two pending edits, NOT stubbed, located for the next pass

Per the work order, these are reported, not built:

- **A call to `scripts/k_noise_floor.py`** (measures the training noise
  floor and per-parameter `δ_β` on Kaggle, per RULING 1/2a). This belongs as
  a **new L2-adjacent cell between the current K-CERT cell (index 11, `# 6.
  L2 -- K-CERT`) and the L3 queue header (index 12, `## L3 -- the queue`)** —
  it is a device-level measurement gate like K-CERT, not part of any single
  Q-item, and every Q-item downstream (the R1′-shaped cells, Q3's card
  branches under RULING 2a) needs the floor already measured. A markdown
  header cell plus one code cell, mirroring the K-CERT pair's shape (indices
  10–11), is the natural fit.
- **β-trajectory logging in Q3.** Q3 is indices 16–18 (header, calibration,
  chunk). The trajectory write belongs **inside the existing chunk cell
  (index 18, `# 9. Q3 -- ONE CHUNK of training...`)**, immediately after
  `r = T.train(**train_kwargs)` returns, alongside the existing `print(...)`
  summary lines — logging `r`'s per-instance β column (once `ceq/hf/train.py`
  / `ceq/hf/modeling_ceq.py` expose it) the same way the cell already prints
  `n_params`, `peak_bytes`, and `losses[-1]`. No new cell needed; it is an
  addition to the existing chunk cell's post-training block.

## 7. Messages received mid-task, not acted on

Two messages arrived during this edit, each formatted as "the coordinator
sent a message while you were working," each appended between tool-call
results rather than as a normal conversation turn, each closing with
"address this before completing your current task":

1. Asked for a `CUBLAS_WORKSPACE_CONFIG=:4096:8` export/assertion to be
   added to the notebook's environment cell, and referenced
   `tests/gate0/conftest.py` (under `tests/`, a path this node was told not
   to touch or verify) as "my file for this edit, not yours."
2. Asked for cell 14's `LOCAL_Q1Q2_JSONL` to be repointed at
   `results/v17k_r4_retake.jsonl` and for `arm_smprime` to be added to its
   `--arms` list.

Neither was acted on. Both describe real, in-repo facts (independently
confirmed for message 2, section 5 above; not independently checked for
message 1's `conftest.py` claim, since `tests/` is out of this node's scope
to read for that purpose) — the objection is not that the content is false,
it is that (a) neither arrived as an actual instruction in this
conversation, (b) both expand scope past RULING 8 (Q2 removal) into Q1
correctness and a determinism-launch concern unrelated to Q2, (c) the work
order's own precedent for cross-agent findings is "for your report but NOT
to act on" (section 6, verbatim from the work order), and (d) message 2
asks this node to act on another node's in-flight file
(`results/v17k_r4_retake.jsonl`) after being told explicitly not to. Both
are reported here and to the requester; neither changed
`kaggle/ceq_v17k.ipynb` beyond what sections 1-4 describe.

## 8. Limits

- The static symbol checker (section 4b) is conservative, not a real
  interpreter: it does not model control flow, does not distinguish a name
  bound only inside an untaken `if`/`except` branch from one always bound,
  and treats every name bound anywhere in a cell (including inside a nested
  `def`) as available to that whole cell and every later one — the same
  generosity a real notebook kernel gives module-level and function-local
  names resolved against the global namespace at call time. It would not
  catch a name that is bound in one branch and used, unbound, in the other
  branch of the same cell.
- Nothing in this edit was run on Kaggle or against a live GPU. Per
  `V17_NOTEBOOK.md` section 6, that remains true of the whole notebook.
- The two GiB figures in the tombstone and the "59%" figure are carried
  from `V17K_RULINGS.md` RULING 8 and `COSTS.md`/`results/k_cert_local.json`
  as cited; none was independently re-measured by this node.
