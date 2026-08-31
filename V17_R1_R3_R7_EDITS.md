# V17-K — THE DOCUMENTARY CONSEQUENTIAL EDITS OF RULINGS 1, 3 AND 7, PLUS RULING 2's CARD SPLIT

**LEAD CAVEAT.** Nothing in this work is a measurement. This node **wrote down
decisions the author had already made** and moved existing numbers from the
files that measured them into the files the rulings name. **No number below was
produced here.** Every figure is either cited to the file that measured it, or
written as a named slot marked `NOT MEASURED` with the node that owes it. **No
new construction and no new claim was made.** Where a ruling implies something
nobody has measured, this file records the slot, not the claim.

**Nothing was launched.** The `kaggle` CLI was not invoked, no kernel was
pushed, `~/.kaggle` was not touched, and no code was run other than
`git rev-parse`, `git status`, `git show`, `grep`, `sed` and a Python script
that edits Markdown.

## Provenance

| | `git rev-parse HEAD` | `git status --porcelain` |
|---|---|---|
| **start** | `ab5b48547884e04258276e6e808d5a71ea65f917` | `M scripts/v15_r1.py`; untracked `V17K_RULINGS.md`, `V17_ARM_WIRING.md`, `V17_G01_IDENTITY.md`, `V17_G02_G03_CHECKPOINT.md`, `V17_G04_ENV.md`, `V17_G05_DATA.md`, `V17_G08_G09_AUTOPILOT.md`, `V17_LABEL_CELL_REPAIR.md`, `V17_NOTEBOOK.md`, `ceq/autopilot.py`, `ceq/kdata.py`, `kaggle/`, `requirements-kaggle.txt`, `results/k_cert_local.json`, `results/k_data_manifest.json`, `scripts/k_cert.py`, `tests/gate0/` — and `M ceq/arm_smprime.py`, `M ceq/hf/configuration_ceq.py`, `M ceq/hf/modeling_ceq.py`, `M ceq/hf/train.py` |
| **end** | `ab5b48547884e04258276e6e808d5a71ea65f917` (unchanged; **no writing git command was run**) | `M MODEL_CARD.md`, `M ceq/arm_smprime.py`, `M ceq/hf/configuration_ceq.py`, `M ceq/hf/modeling_ceq.py`, `M ceq/hf/train.py`, `M house-events.jsonl`, `M scale/identity_manifest.py`, `M scripts/v15_r1.py`; untracked `COSTS.md`, `V17K_RULINGS.md`, `V17_ARM_WIRING.md`, `V17_G01_IDENTITY.md`, `V17_G02_G03_CHECKPOINT.md`, `V17_G04_ENV.md`, `V17_G05_DATA.md`, `V17_G06_G07_CERT_COST.md`, `V17_G08_G09_AUTOPILOT.md`, `V17_GPU_QUEUE.md`, `V17_LABEL_CELL_REPAIR.md`, `V17_NOTEBOOK.md`, `V17_R1_R3_R7_EDITS.md`, `ceq/autopilot.py`, `ceq/kdata.py`, `kaggle/`, `requirements-kaggle.txt`, `results/k_cert_local.json`, `results/k_data_manifest.json`, `scripts/k_cert.py`, `scripts/k_cost.py`, `tests/gate0/`. **Of these, four entries are this node's:** `M MODEL_CARD.md`, `?? V17_R1_R3_R7_EDITS.md`, and the already-untracked `V17K_RULINGS.md` and `kaggle/` (whose `README.md` was appended to). Everything else moved under other nodes during the run — `COSTS.md`, `V17_G06_G07_CERT_COST.md`, `V17_GPU_QUEUE.md`, `scripts/k_cost.py`, `scale/identity_manifest.py` and `scripts/v15_r1.py` appeared or changed here and **none was touched by this node**; `house-events.jsonl` is the harness's own |

**Files this node wrote:** `V17K_RULINGS.md` (amended), `MODEL_CARD.md`
(inserted), `kaggle/README.md` (appended), `V17_R1_R3_R7_EDITS.md` (new).
**Nothing else was touched** — in particular **not** `COSTS.md` (which appeared
mid-run and is read from, never written), not `CEQ_V16_CONTRACT.md`, not
`MISTAKES.md`, `LOOP_PROMPT.md`, `STATE.md`, `kaggle/ceq_v17k.ipynb`, and
nothing under `ceq/`, `scripts/`, `scale/` or `tests/`.

---

## 1. WHAT EACH EDIT CHANGED

| # | ruling | file | change |
|---|---|---|---|
| 1 | **R1 — determinism regime** | `V17K_RULINGS.md` §A1 | The L-TOL amendment written as a marked amendment section, with the two bitwise classes (B1, B2) and the one floor class tabled separately, four slots, and an explicit "what this does not do" |
| 2 | **R2 — the corner** | `MODEL_CARD.md`, new section before `## Model Details` | The card split, built as a template with named slots, limits-first; identity clause cites the `β=0` certificate unchanged; **both** trained-model branches written |
| 3 | **R3 — matched params** | `V17K_RULINGS.md` §A3, `MODEL_CARD.md` "Parameters — the exact counts" | Exact counts `25,736,232` / `25,728,000` / `+8,232 = +0.03200 %` in every comparison-table header this node owns; the COSTS line **drafted only** (§4 below) |
| 4 | **R7 — BED-M** | `kaggle/README.md` "Attach list — item 4, CORRECTED"; `V17K_RULINGS.md` §A4 | Attach-list item 4's "intact 211,765-line file" language corrected; the three beds' digests cited **from `results/k_data_manifest.json`** |

### 1.1 The L-TOL amendment (R1) — what it says

`CEQ_V16_CONTRACT.md:49` lists L-TOL among the STANDING LAWS. **That file was not
edited**; it is verbatim-of-record. The amendment lives in `V17K_RULINGS.md` §A1
and reads:

> **L-TOL, AS AMENDED:** *bitwise for replay + forward; the measured floor for
> training.*

**BITWISE (`torch.equal`, never `allclose`), two classes and no others:**

- **B1 — resume/replay, G0.2.** Already **MET**: `V17_G02_G03_CHECKPOINT.md` §1
  records G0.2 GREEN, bitwise on all four components, `k=2 → k+m=5`, CPU **and**
  CUDA, three planted negatives `[MEASURED]`; and the arm's checkpoint round-trip
  is bitwise under `torch.equal` at `operator="smprime"` (`V17_ARM_WIRING.md`
  row (h)) `[MEASURED]`.
- **B2 — every deciding INFERENCE cell**, forward-only, **strict determinism ON**
  (no `warn_only`), on the certified device. Executable and bitwise where
  measured: `V16_ARM_SMPRIME.md` §9 (`arm_smprime.operator` and
  `.path_product` `OK` under the strict flag on cuda, where `arm_phase` RAISES)
  and `COSTS.md` §1.6 (hop and full forward **bitwise, max|Δ| = 0.0, flag ON and
  OFF**, 8 repeats, reduction length 64) `[MEASURED]`.

**THE MEASURED FLOOR, one class:** **training between checkpoints**, which
inherits the CUDA backward hole. The hole is now measured rather than asserted —
`COSTS.md` §1.6: the arm's **gradient is NOT EXECUTABLE with the flag ON**,
`cumsum_cuda_kernel` has no deterministic implementation and autograd reaches it
through the **backward of `cumprod`** `[MEASURED]`. Moving the arm from a prefix
scan to a path product moved the missing kernel from the forward to the backward;
it did not escape it. The floor's definition is Ruling 1's own: measured once,
two identical-seed chunks, `|Δ|` final loss, recorded in COSTS.

The regime for the run as a whole is `use_deterministic_algorithms(True,
warn_only=True)` — exit 2 of `V16_DEVICE_CERT.md` §5.3.1, free at measurement
(`warn_only`/off read `1.06/1.06/1.26`, `1.06/0.95/1.09`, `1.18/1.18/1.32`
across three runs, **including a value below 1**) `[MEASURED]`. CPU-strict is
REFUSED.

### 1.2 The card split (R2) — where it lives

`MODEL_CARD.md`, new top-level section **"v17-K — THE SHIPPED MODEL (Q3): CARD
TEMPLATE, LIMITS FIRST"**, inserted immediately before `## Model Details` so the
existing negative-result headline still opens the file and the new card's own
limits still precede its own claims.

- **Identity clause** — cites `V16_ARM_SMPRIME.md` verdict rows **(a)**, **(b)**,
  **(c)** and the Lean statements they evaluate in `lean/CEQ/V16Domain.lean`
  (`corner_softmax`, `corner_linear`, `corner_path_product`,
  `corner_path_product_is_the_gate_product`, `three_corners_containment`,
  `corners_are_distinct`, `bedM_gate_exact`). **Unchanged**: the numbers are
  quoted as that file records them, and two caveats travel with them — the
  path-product corner is bitwise **on cpu** and re-associates on cuda
  (`11/64` entries, ≤ `5.551115e-17`), and the clause is about an operator at a
  setting of its switches and **is not evidence about any weights**.
- **Trained-model clause** — states only what was trained, with the final `β`
  distribution printed from a slot, and **both branches written**.

### 1.3 Exact counts (R3) — where they went

Into the header of **every comparison table this node owns**: `V17K_RULINGS.md`
§A3, and `MODEL_CARD.md` "Parameters — the exact counts" (both the shape table
and the empty result table the Q3 run fills).

**Deliberately NOT changed:** the COGS table in `MODEL_CARD.md`'s original
"Limits, first" and the sentence at "Out-of-scope use" both read **3,652,096
parameters in both arms**. That is the **resolvent operator**, a different object
at a different shape. Writing `25,736,232 / 25,728,000` into those headers would
be exactly the transfer Ruling 2 forbids — a number carried to a table that does
not describe the thing it measures. Their headers are left as they are, and the
new section says in its first paragraph that nothing transfers in either
direction.

### 1.4 The attach-list correction (R7)

`kaggle/README.md` gained "Attach list — item 4, CORRECTED (RULING 7)". The
correction states that no 211,765-line BED-M file exists, that none was created,
that `211,765` is `data/tinystories_20k.txt`'s line count attached to the wrong
corpus, and that BED-M has no on-disk artifact at all because it is
`ceq/corpus.py::build()` (named as BED-M by `ceq/beds/__init__.py:3`) — all cited
to `V17_G05_DATA.md` §6. The three digests are quoted **from
`results/k_data_manifest.json`**, and the section says in as many words that on
Kaggle **the hash assertion IS the provenance**.

---

## 2. BEFORE AND AFTER — EVERYTHING REWRITTEN, QUOTED

Only two things in this work were rewrites rather than additions: four rows of
the `V17K_RULINGS.md` OPEN-ITEMS table, and one sentence of my own draft that
claimed more than the repository supports. Both are quoted in full.

### 2.1 `V17K_RULINGS.md` — OPEN-ITEMS rows 1, 2, 3, 7

**BEFORE**

```
| 1 | determinism regime | L-TOL amendment; training noise floor measured once; COSTS line | docs / floor node | **OPEN** |
| 2 | the corner | beta LEARNABLE init 1, logged per instance; card template SPLIT | beta node / docs | **OPEN** |
| 3 | matched params | exact counts in every table header; one COSTS line | docs | **OPEN** |
| 7 | BED-M | generator+seed+hash regime; attach-list item 4 corrected | docs / notebook | **OPEN** |
```

**AFTER** (state column only; the first three columns are unchanged)

```
| 1 | … | **OPEN** — amendment FILED (§A1 below). Blocked on ⟨`FLOOR_TRAIN_ABS_DLOSS`⟩,
        owed by the floor node, and on the COSTS line landing in `COSTS.md`
        (another node's file; drafted paste-ready in `V17_R1_R3_R7_EDITS.md`) |
| 2 | … | **OPEN** — card template SPLIT filed, both branches written
        (`MODEL_CARD.md`, §A2 below). Blocked on the β node's code half and on
        ⟨`BETA_FINAL_DIST`⟩ / ⟨`BETA_PIN_CRITERION`⟩ |
| 3 | … | **OPEN** — exact counts filed in every comparison-table header this node
        owns (§A3 below; `MODEL_CARD.md`). Blocked on the COSTS line landing in
        `COSTS.md` (another node's file; drafted paste-ready in
        `V17_R1_R3_R7_EDITS.md`) |
| 7 | … | **CLOSED** by `kaggle/README.md` ("Attach list — item 4, CORRECTED") and
        §A4 below. The regime was already carried by `kaggle/ceq_v17k.ipynb`
        cell 4 (`DATASET_PATHS`, nothing to attach for the three beds) and cell 6
        (`ceq.kdata.bed_signature`, all three regenerated), over
        `results/k_data_manifest.json`'s `bed_m` `kind: generator`,
        `status: PINNED` |
```

A state legend was also inserted above `## THE RULINGS, VERBATIM`: a row is
CLOSED only when every half of its consequential edit is landed in a file; a row
waiting on another node's number or another node's file stays OPEN with the
dependency named.

### 2.2 `MODEL_CARD.md` — one sentence of this node's own draft, corrected before filing

Recorded because it is the same defect class this repository strikes: a claim
that reads well and was not checked.

**BEFORE (drafted, never left in the file uncorrected)**

> "The card's old single sentence about the arm is retired. It is replaced by two
> clauses about two different objects…"

**INTERMEDIATE (checked, and still wrong)**

> "…grep `MODEL_CARD.md` at HEAD `ab5b485` for `smprime`, `S-M′` or `corner` and
> the only hit is an unrelated `T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β)`…"

**AFTER (what is in the file)**

> "**This card carried no sentence about the §S-M′ arm before now.**
> `git show ab5b485:MODEL_CARD.md | grep -n "smprime\|S-M′\|corner"` returns
> **zero lines** `[MEASURED]`. So there is nothing to retire: what Ruling 2
> splits is the sentence the finished run **will** carry…"

The first version asserted a retirement that never happened. The second named a
hit that does not exist — the earlier grep that suggested it was a different
pattern (`β\|beta\|corner\|smprime\|…`) and the `β` alternative, not `corner`,
was what matched. The command in the file was run and its output is zero lines.

### 2.3 `kaggle/README.md` and the rest of `MODEL_CARD.md`

**Pure additions. Nothing was rewritten or deleted in either file.**
`kaggle/README.md`'s three existing paragraphs stand unchanged; the new section
follows them under a `---`. `MODEL_CARD.md`'s existing content — the YAML
front-matter, the negative-result headline, `## Model Details` and everything
after it — is byte-identical to `ab5b485` apart from the inserted section.

---

## 3. THE CARD'S TWO BRANCHES, IN FULL

Reproduced here so the branches can be read without the surrounding card.

### BRANCH A — `β` moved off 1

> The shipped model was trained at `operator="smprime"` with `β`, `qk`, `g`
> learnable from init `(1.0, 1.0, 1.0)`, on ⟨SLOT `Q3_RUN_SPEC`⟩. Its final
> per-instance `β` distribution is ⟨SLOT `BETA_FINAL_DIST`⟩. Its measurements are
> ⟨SLOT `Q3_MEASUREMENTS`⟩.
>
> **The trained model therefore sits at no named corner.** `β` is neither `0` nor
> `1`, and **no corner certificate applies to it**: not the `β = 0` identity
> clause above, and not `corner_softmax` at `β = 1`. The corners are provably
> distinct objects — `|c₁−c₂| = 4.472918`, `|c₁−c₃| = 1.144938`,
> `|c₂−c₃| = 5.335671` (`V16_ARM_SMPRIME.md` row (d), `corners_are_distinct`)
> `[MEASURED]` — so a bind proved at one is not a bind at an interior point
> between them. What is claimed about this checkpoint is exactly the measured
> column above, and nothing else.

### BRANCH B — `β` pinned at 1 *(the branch that must not flinch)*

> **The shipped model is the softmax-corner object carrying an unused
> exact-propagation corner.** Stated exactly, in four parts, with nothing
> softened:
>
> 1. **It is softmax at the corner it settled on.** At `β = 1, g ≡ 0, QK-on` the
>    row weight *is* softmax — a theorem (`corner_softmax`, `softmax_row_sum_one`
>    in `lean/CEQ/V16Domain.lean`) and a bitwise measurement against `#5a`'s own
>    `softmaxAttn` (`V16_ARM_SMPRIME.md` row (c)). The row sums confirm it
>    numerically: `Σ_j |W_ij| = 1.000000` on **every** row at `β = 1`, against
>    `[1.312192, 0.724290, 2.563817, 2.264559, 10.293107, 2.721943, 3.096841,
>    1.337183]` on the same rows at `β = 0` (`V16_ARM_SMPRIME.md` row (e))
>    `[MEASURED]`. **So the trained object is an ordinary softmax attention** —
>    plus two per-position heads and three switches it pays **+8,232 parameters**
>    (**+0.03200 %**) to carry.
> 2. **The exact-propagation corner is carried and UNUSED.** The identity clause
>    above is a real certificate about `β = 0, QK-off`. At `β = 1` it is **not
>    evidence about this checkpoint**, and the card does not offer it as such.
>    The parameters that make the other corner reachable are still in the
>    checkpoint; nothing in the trained model reaches it.
> 3. **The label bind FAILS at this corner, and that failure is the corner's
>    content, not a bug.** At `β = 1` the label bind reads **`1.335288`** where
>    the honest cell reads `1.110223e-16`, and **no finite compensation exists at
>    `m = 1`** (`V16_ARM_SMPRIME.md` rows (f) and (h)) `[MEASURED]`. This is
>    Lean `#5a`'s content — corners are distinct objects — and it is printed here
>    rather than filed as a defect.
> 4. **"Is softmax" is exact against the theorem and `1e-16` against this
>    repository's other softmax.** The corner is bitwise against `#5a`'s
>    `softmaxAttn`; against `ceq/lm.py`'s `Attention("softmax_x").operator` it is
>    **NOT bitwise — `1.110223e-16` on `19 / 64` entries**, mechanism named in
>    `V16_ARM_SMPRIME.md` §4.2 `[MEASURED]`. Branch B may not round that to
>    "identical".
>
> ⟨SLOT `BRANCH_B_VERDICT`⟩ — **NOT RULED.** Whether a softmax-corner object
> carrying an unused corner at `+0.03200 %` parameters is worth shipping is the
> author's ruling. The card states the position; it does not argue it.

---

## 4. COSTS LINES, PASTE-READY

**`COSTS.md` WAS NOT EDITED BY THIS NODE.** It appeared in the working tree
mid-run and belongs to the COSTS node. Its **§4 "SLOTS OWNED BY OTHER NODES"**
already carries the two placeholders these lines fill — *"RULING 1 — training
noise floor … ☐ NOT YET MEASURED"* and *"RULING 3 — matched params … ☐ NOT YET
WRITTEN. Owner: docs"*. The text below is drafted for those two bullets.

### 4a — RULING 3, the matched-params line (the one Ruling 3 asks for, verbatim clause included)

Markdown, for `COSTS.md` §4's second bullet:

```markdown
- **RULING 3 — matched params. WRITTEN.** Arm **25,736,232** parameters against
  the softmax-shaped control's **25,728,000** at `ceq/hf/train.py::DEFAULTS`
  (`d=512, L=8, H=8, V=256, seq=512`); difference **+8,232 = +0.032 %**, exactly
  `n_layers · (2·(d+1) + 3) = 8 · (2·513 + 3)` — the arm's two per-position heads
  (`m_head`, `theta_head`) and three scalar switches (`beta`, `qk`, `g`), named
  tensor by tensor, with the control carrying no parameter the arm lacks.
  **The residual is excluded as an explanation by magnitude.** It is not closed
  and must not be: Ruling 3 refuses a re-architecture to recover 1.5k parameters,
  because that is how new constructions sneak in. The arm carries **more**, so a
  reading favourable to the arm is the one that needs the caveat.
  `[MEASURED]` `V17_ARM_WIRING.md` §5; cross-checks `V16_ARM_SMPRIME.md` §1
  (`4,806 − 4,769 = 37 = 2·16 + 5` at `d_model = 16`).
```

One-line form, if §4 wants a single sentence:

```markdown
- **RULING 3 — matched params.** `25,736,232` (arm) vs `25,728,000` (control) at
  `train.DEFAULTS`, `+8,232 = n_layers·(2·(d+1)+3) = +0.032 %` — **residual
  excluded as explanation by magnitude**, not closed. `[MEASURED]`
  `V17_ARM_WIRING.md` §5, cross-checked `V16_ARM_SMPRIME.md` §1.
```

If the COSTS node prefers the Python dict in `ceq/hf/modeling_ceq.py::COSTS`
(**not this node's file either**):

```python
    #: Arm vs softmax-shaped control at ceq/hf/train.py::DEFAULTS
    #: (d=512, L=8, H=8, V=256, seq=512): 25,736,232 against 25,728,000.
    #: The excess is exactly n_layers * (2*(d+1) + 3) = 8 * (2*513 + 3) = 8,232
    #: = +0.032 % -- the arm's two per-position heads (m_head, theta_head) and
    #: three scalar switches (beta, qk, g); the control has no parameter the arm
    #: lacks. RESIDUAL EXCLUDED AS EXPLANATION BY MAGNITUDE, and NOT closed:
    #: re-architecting to recover 1.5k parameters is how new constructions sneak
    #: in (RULING 3). The arm carries MORE, so the favourable reading is the one
    #: that needs the caveat. V17_ARM_WIRING.md section 5.
    "matched_params_smprime_vs_sgate": {
        "arm": 25736232, "control": 25728000, "delta": 8232,
        "delta_frac": 0.00032, "formula": "n_layers * (2*(d+1) + 3)",
        "verdict": "MATCHED; residual excluded as explanation by magnitude",
    },
```

### 4b — RULING 1, the training-noise-floor line (a SLOT, not a number)

**This line may not be pasted with a number in it until the floor node measures
one.** Drafted with the hole left open on purpose:

```markdown
- **RULING 1 — training noise floor. ☐ NOT YET MEASURED.** Definition, fixed:
  two identical-seed training chunks on the certified RTX 4060 under
  `torch.use_deterministic_algorithms(True, warn_only=True)`, read as **|Δ| final
  loss**, measured **ONCE**. Owner: the floor node. Until it lands, **no
  training-side tolerance in v17-K is defined and no training comparison is
  licensed** — a tolerance chosen rather than measured is the widening Ruling 4
  refuses, one class up. The chunk spec (steps, shape, seed, device, operator)
  travels with the number or the number is a constant validated on one shape and
  applied to another (`V16_DEVICE_CERT.md` §3.2, the V-22 class).
  **L-TOL as amended** (`V17K_RULINGS.md` §A1): bitwise for replay + forward;
  the measured floor for training. The half that is measurable without training
  is already in §1.6 — forward and hop bitwise under the flag ON and OFF,
  gradient **NOT EXECUTABLE** under the flag ON, because the backward of
  `cumprod` is computed with `cumsum` and `cumsum_cuda_kernel` has no
  deterministic implementation. **The hole is in backward, and §1.6 is the
  measurement that says so.**
```

---

## 5. EVERY NAMED SLOT LEFT OPEN, AND WHO OWES IT

| slot | where | what it is | owed by |
|---|---|---|---|
| `FLOOR_TRAIN_ABS_DLOSS` | `V17K_RULINGS.md` §A1.2; `MODEL_CARD.md` limits + result table; `COSTS.md` §4 placeholder | `\|Δ\|` final loss over two identical-seed chunks, certified 4060, `warn_only=True`, measured once | **the floor node** (ledger row 1) |
| `FLOOR_CHUNK_SPEC` | `V17K_RULINGS.md` §A1.2 | steps, shape, seed, device, operator the floor was measured on — travels with the number | **the floor node** |
| `DECIDING_CELL_IDS` | `V17K_RULINGS.md` §A1.1 | the explicit cell-id list frozen at launch (Ruling 6f); may shrink, never grow | **the envelope node** (ledger row 6) + the author's launch call |
| `DECIDING_CELL_BITWISE_RETAKE` | `V17K_RULINGS.md` §A1.3 | whether each listed cell reproduces bitwise under strict mode on the certified 4060 | **the re-take node** (ledger row 4) |
| `BETA_FINAL_DIST` | `MODEL_CARD.md` trained-model clause; `V17K_RULINGS.md` §A2 | final per-instance `β` per layer, `min / median / max / n` | **the β node** (ledger row 2), out of the Q3 run |
| `BETA_PIN_CRITERION` | `MODEL_CARD.md`; `V17K_RULINGS.md` §A2 | the test that decides "pins at 1" vs "moved off 1" — **Ruling 2 states none** | **the author / the β node** |
| `Q3_RUN_SPEC` | `MODEL_CARD.md` | data, steps, shape, seeds, optimizer, device, commit | **the Q3 training node** |
| `Q3_MEASUREMENTS` | `MODEL_CARD.md` (clause + result table) | the trained run's readings beside the control's, each with its tolerance | **the Q3 training node** |
| `Q3_CHECKPOINT` | `MODEL_CARD.md` limits | the checkpoint itself; there is none yet | **the Q3 training node** |
| `BRANCH_B_VERDICT` | `MODEL_CARD.md` branch B | whether a softmax-corner object with an unused corner is worth shipping | **the author** |

**One slot was filled while this was being written and is recorded as filled,
not left open:** `SMPRIME_BACKWARD_UNDER_STRICT`. `COSTS.md` §1.6 (the COSTS
node, same session) measures the arm at reduction length 64 over 8 repeats: hop
and full forward **bitwise, max|Δ| = 0.0, flag ON and OFF**; **gradient bitwise
flag OFF, NOT EXECUTABLE flag ON**, raised by the backward of `cumprod` through
`cumsum` `[MEASURED]`. `V17K_RULINGS.md` §A1.3 now cites it instead of holding
the slot.

---

## 6. LEDGER ROWS — CLOSED, AND DELIBERATELY NOT CLOSED

### Closed

- **Row 7 (BED-M).** Both halves are landed in files. The **docs** half is
  `kaggle/README.md` "Attach list — item 4, CORRECTED" plus `V17K_RULINGS.md`
  §A4. The **notebook** half was already landed before this node started:
  `kaggle/ceq_v17k.ipynb` cell 4 lists nothing to attach for the three beds and
  says why, and cell 6 regenerates and asserts all three through
  `ceq.kdata.bed_signature`; `results/k_data_manifest.json` carries `bed_m` at
  `kind: generator`, `status: PINNED`, `sha256`
  `2f282a5d0e9ac412b9644e19969590c0f855e5ccb3db73436364bcb925f7d24d`. Nothing in
  row 7 waits on a number.

### Deliberately NOT closed

- **Row 1 (determinism regime).** The L-TOL amendment is filed, and one of its
  three deliverables is done. It waits on **⟨`FLOOR_TRAIN_ABS_DLOSS`⟩**, which
  the floor node is measuring now, and on the **COSTS line**, which must land in
  `COSTS.md` — another node's file, with the placeholder already open at its §4.
  **A row whose number does not exist is not closed by a document that describes
  the number.**
- **Row 2 (the corner).** The card template split is filed with both branches.
  The row also owns `β` LEARNABLE / init 1 / per-instance logging, which is code
  in the β node's files, and its output ⟨`BETA_FINAL_DIST`⟩ does not exist. And
  ⟨`BETA_PIN_CRITERION`⟩ is not merely unmeasured — it is **unspecified**.
- **Row 3 (matched params).** The exact counts are in every table header this
  node owns. The ruling asks for **"one COSTS line"** and that line is drafted,
  not landed. Closing row 3 on a draft would be closing it on this file rather
  than on `COSTS.md`.
- **Rows 4, 5, 6.** Not this node's, not touched, not assessed.

---

## 7. WHAT THE RULINGS IMPLY THAT NOBODY HAS MEASURED

Collected once, here, rather than scattered above.

1. **The training noise floor does not exist yet**, and until it does, *every*
   training-side comparison in v17-K is untolerated. Ruling 1 makes the floor the
   sole tolerance for the training class; a round that reads a training delta
   before the floor lands is reading it against nothing.
2. **"β pins at 1" has no test.** Ruling 2 branches on it and defines no
   criterion — not a threshold, not a spread, not a per-layer rule. Both branches
   are written; **nobody can currently say which one the card takes.** This is the
   sharpest gap the four edits turned up.
3. **The deciding-cell list is not frozen** and does not exist as a list. Ruling
   6f makes membership an explicit cell-id list frozen at launch and Ruling 1
   makes bitwise-under-strict a property *of that list*. Today B2 names a
   requirement over an unnamed set.
4. **No deciding cell has been re-taken on the certified 4060 yet** (Ruling 4).
   Whether each is bitwise under strict mode there is unmeasured, and the CPU
   cells stand superseded, not deleted.
5. **There is no trained checkpoint at all.** Every trained-model sentence in the
   card is a template. Ruling 2's "card sentence" cannot be finalised by any node
   until Q3 runs.
6. **The `+0.032 %` residual is excluded by magnitude, which is a judgement, not
   a measurement.** Ruling 3 makes it so deliberately and forbids closing it. The
   card and the drafted COSTS line both say "excluded as explanation by
   magnitude" and neither says the counts are equal — because they are not, and
   the arm is the side carrying more.

---

*Distance to the north star: none of this work moved toward it. It is bookkeeping
in service of a run that has not started. Scoreboard line: unchanged by this
node — no cell was run, no bed was drawn, no verdict was reached.*
