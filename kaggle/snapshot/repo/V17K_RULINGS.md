# CEQ v17-K — G0.10 RULING LEDGER

Filed verbatim from the author's adjudication of the seven questions raised by
the Gate-0 nodes. **Gate 0 closes only at open-rulings = 0.** This file is the
register; each ruling's consequential edit is owned by a named node and is not
discharged until its row reads CLOSED with the file that closed it.

## OPEN-ITEMS TABLE

**G0.10 CONTAINS A CIRCULARITY AND IT IS NOT RESOLVED HERE.** Gate 0 closes at
open-rulings = 0, and nothing launches until Gate 0 is green. But rows 1 and 2
are blocked on numbers that **only the Kaggle run produces** — the training
noise floor (measured on the device the training runs on) and the beta
distribution of a **trained** model. As written, Gate 0 blocks the run that
would close it. The rows are marked with which half is pre-launch and which is
post-run; **whether that split is legitimate, or whether the gate's text should
change, is the author's ruling and it is not taken here.**

| # | ruling | consequential edit | owner | state |
|---|---|---|---|---|
| 1 | determinism regime | L-TOL amendment; training noise floor measured once; COSTS line | docs / floor node | **OPEN** — amendment FILED (§A1 below). Blocked on ⟨`FLOOR_TRAIN_ABS_DLOSS`⟩, owed by the floor node, and on the COSTS line landing in `COSTS.md` (another node's file; drafted paste-ready in `V17_R1_R3_R7_EDITS.md`) |
| 2 | the corner (criterion now **RULING 10'**, not 2a's `5*delta_beta`) | beta LEARNABLE init 1, logged per instance; card template SPLIT | beta node / docs | **OPEN** — card template rebuilt to RULING 2a's THREE branches + census word-gate, all three written (`MODEL_CARD.md`, §A2 below). `BETA_PIN_CRITERION` is now SPECIFIED by RULING 2a itself (`\|β_i,final − 1\| ≤ 5·δ_β,i`, k=5 heuristic scale); what remains open is its **evaluation**. Blocked on two measurements, neither landed: ⟨`DELTA_BETA_PER_PARAM`⟩ — per-parameter `δ_β,i` owed by the **floor node** (`scripts/k_noise_floor.py`, which today measures only the loss-space floor over the Ruling-1 identical-seed pair, not a per-parameter one) — and ⟨`GRADIENT_CENSUS`⟩ — per-parameter integrated `\|∂L/∂β_i\|` owed by the **β node** (`ceq/hf/modeling_ceq.py`, a new gradient accumulator; `beta_summary`'s `max_abs_grad` is a max over logged steps, not the needed integral). `BETA_FINAL_DIST` and `BRANCH_C_LOCATIONS` (layer-level only — no per-head `β` exists to locate) wait on the same two |
| 3 | matched params | exact counts in every table header; one COSTS line | docs | **OPEN** — exact counts filed in every comparison-table header this node owns (§A3 below; `MODEL_CARD.md`). Blocked on the COSTS line landing in `COSTS.md` (another node's file; drafted paste-ready in `V17_R1_R3_R7_EDITS.md`) |
| 4 | CPU cells | re-take on the certified 4060; journal supersede marks | re-take node | **CLOSED** by `V17_R4_RETAKE.md`. 24 cells on the 4060 in `results/v17k_r4_retake.jsonl`, **2.79 GPU-min** against the `~2 GPU-h` estimate (43x over-book, measured). No verdict moves: 0 sign-flips in 16, worst device delta `9.522e-03` on `arm_pl` seed 7 — **not** on the control as the probe's 6-cell sample suggested. `results/v15_r1.jsonl` 25 -> 26 lines, first 25 bytes identical, append-only supersede marker (L-G2) |
| 5 | Q1 instrument | hash into identity manifest; one L-SCOPE check | manifest node | **CLOSED** by `V17_R5_INSTRUMENT.md`. Digest `77429cdb...35498` over **two** components (file bytes + a bytecode fingerprint of 15 reached callables), so a helper edit in another module moves it. L-SCOPE measured, not asserted: both mains run with a recording spy, tensors **bitwise identical** at the same seed. Every cell now carries `instrument_hash` automatically |
| 6 | envelope text | six edits to `ceq/autopilot.py` and its tests | envelope node | **CLOSED** by `V17_R6_ENVELOPE.md`. 48 -> **62 tests**. 6a/6b/6e/6f needed code; **6c and 6d were already true** and were verified and covered rather than staged as fixes. Structural `note`-blindness survives, re-verified over the two new helpers |
| 7 | BED-M | generator+seed+hash regime; attach-list item 4 corrected | docs / notebook | **CLOSED** by `kaggle/README.md` ("Attach list — item 4, CORRECTED") and §A4 below. The regime was already carried by `kaggle/ceq_v17k.ipynb` cell 4 (`DATASET_PATHS`, nothing to attach for the three beds) and cell 6 (`ceq.kdata.bed_signature`, all three regenerated), over `results/k_data_manifest.json`'s `bed_m` `kind: generator`, `status: PINNED` |

**State legend.** A row is CLOSED only when every half of its consequential
edit is landed in a file. A row whose edit waits on a number another node is
still measuring, or on a file another node owns, stays OPEN with the
dependency named. Amendments and the slots they leave open are in
"AMENDMENTS" at the foot of this file.

---

## THE RULINGS, VERBATIM

**RULING 1 — DETERMINISM REGIME: CUDA with warn_only=True.** Bitwise is
REQUIRED for: resume/replay (G0.2) and every deciding INFERENCE cell
(forward-only, strict mode ON — the hole is in backward). Training between
checkpoints inherits the hole; its noise floor is MEASURED ONCE (two
identical-seed chunks, |Δ| final loss) and recorded in COSTS. L-TOL amended:
bitwise for replay+forward; the measured floor for training. CPU-strict is
REFUSED (loses the certified device; that is a different round).

**RULING 2 — THE CORNER:** the β=1 label-bind failure (1.335288) is Lean #5a's
content, not a bug — corners are distinct objects. Q3 trains with β LEARNABLE,
INIT 1, per-instance β logged as a column. The card sentence SPLITS: identity
clause cites the β=0 certificate (unchanged, real); trained-model clause states
only what was trained, with the final β distribution printed. If β pins at 1:
the shipped model is the softmax-corner object carrying an unused
exact-propagation corner — say exactly that. NO sentence transfers across
corners without a bind at the corner it describes.

**RULING 3 — MATCHED PARAMS:** 0.032% residual is MATCHED. Print exact counts
in every table header + one COSTS line ("residual excluded as explanation by
magnitude"). Do NOT re-architect to close 1.5k params — that is how new
constructions sneak in.

**RULING 4 — CPU CELLS: RE-TAKE, never widen.** R1′ and every deciding cell
re-run on the certified 4060 before launch (~2 GPU-h). CPU cells stay in the
journal as `device:cpu`, superseded — never deleted (L-G2). Tolerance widening
to absorb an unmeasured CPU↔CUDA delta is REFUSED.

**RULING 5 — Q1/Q2 INSTRUMENT:** `scripts/v15_r1.py` is ADOPTED as the named
instrument. Hash into the identity manifest; one L-SCOPE check (its batch path
== production path); Q1/Q2 cite it by hash. Optional rename
`scripts/q1_r1prime.py`; the hash is the identity either way.

**RULING 6 — ENVELOPE TEXT** (six ambiguities, now law):
- **6a** stale journal = no write for 2 CONSECUTIVE POLLS.
- **6b** "session <20 min" = KAGGLE'S reported remaining time.
- **6c** simultaneous triggers: highest tier wins; within tier, table order.
  Tier-3 co-firing with a Tier-1 fix ⇒ HALT, not fix.
- **6d** Tier-2 lineage scoped PER ROOT CELL: two Tier-2 on any cell descended
  from one root ⇒ Tier 3 (no laundering via chains).
- **6e** OOM keyed per (cell, shape): second OOM at same shape after halving ⇒
  Tier 3, not a second halving.
- **6f** deciding-cell membership = explicit cell-id list, FROZEN at launch; may
  shrink mid-flight, never grow.

**RULING 7 — BED-M:** generator + seed + expected hash, regenerated and
asserted in-notebook (same regime as BED-K/BED-1). Attach-list item 4's "intact
file" language corrected in the same commit; the hash assertion IS the
provenance on Kaggle.

---

## LAUNCH ORDER (unchanged)

G0.1–G0.10 green → K-CERT first on Kaggle (δ/tol < 50% or HALT) → Q1 → Q2 →
Q3 (β learnable, init 1) → Q4 → cross-device table → HF package on explicit
say-so.

---

# AMENDMENTS — THE CONSEQUENTIAL EDITS, AS FILED

Each section below is a consequential edit of a ruling above, written down here
rather than in the file the ruling is about, wherever that file is
verbatim-of-record. **Nothing in this part is a new claim.** Every figure is
cited to the file that measured it, or written as a named slot marked
`NOT MEASURED` with the node that owes it. A slot is not a prediction.

Filed by the R1/R3/R7 documentary node at HEAD `ab5b485`. Deliverable and
before/after quotations: `V17_R1_R3_R7_EDITS.md`.

---

## A1 — L-TOL, AS AMENDED BY RULING 1

**Where this lives, and why it is not in the contract.** L-TOL is listed among
the STANDING LAWS at `CEQ_V16_CONTRACT.md:49`. That file is filed verbatim from
the author's message of 2026-08-31 and is verbatim-of-record; **it was not
edited.** This section is the amendment, and this text is what binds for round
v17-K.

> **L-TOL, AS AMENDED:** *bitwise for replay + forward; the measured floor for
> training.*

The amendment is a consequence of Ruling 1's determinism regime — **CUDA with
`torch.use_deterministic_algorithms(True, warn_only=True)`**, which is exit 2 of
the three priced in `V16_DEVICE_CERT.md` §5.3.1, chosen there because it is the
only exit that leaves the bar's certification regime and the run's regime the
same flag at the same setting, and because it is free: the `warn_only`/off ratio
read `1.06/1.06/1.26`, `1.06/0.95/1.09`, `1.18/1.18/1.32` across three runs,
**including a value below 1**, which a real cost cannot produce
(`V16_DEVICE_CERT.md` §5.3.1) `[MEASURED]`. **CPU-strict is REFUSED** — it loses
the certified device, and that is a different round.

### A1.1 — HELD TO BITWISE (`torch.equal`, never `allclose`)

Two claim classes, and no others, are held to bitwise equality:

| # | claim class | scope of the bitwise requirement | status today |
|---|---|---|---|
| **B1** | **resume / replay — G0.2 K-RESUME** | the resumed run reproduces the un-interrupted run on all four checkpointed components; and a checkpoint round-trip (`save_pretrained` → `from_pretrained`) reproduces the logits | **MET.** `V17_G02_G03_CHECKPOINT.md` §1 records G0.2 **GREEN** — bitwise on all four components, `k=2 → k+m=5`, CPU **and** CUDA, three planted negatives `[MEASURED]`. The arm's own checkpoint round-trip is bitwise under `torch.equal` at `operator="smprime"` (`V17_ARM_WIRING.md` verdict row (h)) `[MEASURED]` |
| **B2** | **every deciding INFERENCE cell** | forward-only, **strict determinism ON** (`use_deterministic_algorithms(True)`, no `warn_only`), on the certified device | **EXECUTABLE, MEMBERSHIP NOT FROZEN.** The shipped arm's forward runs under strict mode and is bitwise there: `arm_smprime.operator` and `arm_smprime.path_product` read `OK` under `use_deterministic_algorithms(True)` on cuda, where `arm_phase.operator` and `arm_phase.scan_phase` **RAISE** (`V16_ARM_SMPRIME.md` §9) `[MEASURED]`; and `COSTS.md` §1.6 reads hop and full forward **bitwise, max\|Δ\| = 0.0, flag ON and OFF**, 8 repeats at reduction length 64 `[MEASURED]`. What is **not** settled is which cells are on the list — see the slot below — and whether each reproduces bitwise on the certified 4060 after Ruling 4's re-take |

**Why B2 is stated forward-only.** The hole is in **backward** (Ruling 1's own
words). A deciding inference cell never takes a backward pass, so strict mode is
both affordable and executable for it; a training step is not, which is A1.2.

**B2's membership is a list, not a description.** Ruling 6f: deciding-cell
membership is an **explicit cell-id list, FROZEN at launch**, and may shrink
mid-flight, never grow.

> ⟨SLOT `DECIDING_CELL_IDS`⟩ — **NOT MEASURED / NOT FROZEN.** The explicit
> cell-id list frozen at launch under Ruling 6f. Owed by the envelope node
> (ledger row 6) together with the author's launch call. Until it exists, B2
> names a requirement and not a set.

### A1.2 — HELD TO THE MEASURED FLOOR

| claim class | regime | why it cannot be bitwise |
|---|---|---|
| **training between checkpoints** | `use_deterministic_algorithms(True, warn_only=True)`; the run is compared against the **measured noise floor**, never against `0` | It **inherits the CUDA backward hole, and the hole is now measured.** `COSTS.md` §1.6: the arm's **gradient is NOT EXECUTABLE with the flag ON** — `cumsum_cuda_kernel` has no deterministic implementation and autograd reaches it through the backward of `cumprod` `[MEASURED]`. So the round runs `warn_only=True`, where that kernel warns instead of raising (`V16_DEVICE_CERT.md` §5.3.1), and a claim of bitwise training under that regime would be a claim about kernels nobody selected |

**The floor's definition, from Ruling 1 verbatim:** MEASURED ONCE, as **two
identical-seed chunks**, read as **`|Δ|` final loss**, and recorded in COSTS.

> ⟨SLOT `FLOOR_TRAIN_ABS_DLOSS`⟩ — **NOT MEASURED.** `|Δ|` final loss between
> two identical-seed training chunks on the certified RTX 4060 under
> `warn_only=True`. Owed by the **floor node** (ledger row 1 owner). It is being
> measured as this is written; **no number is written here, and none is
> guessed.** Until it lands, every training-side tolerance in round v17-K is
> undefined and no training comparison is licensed. `COSTS.md` §4 already holds
> the placeholder this number lands in (“**☐ NOT YET MEASURED**”).

> ⟨SLOT `FLOOR_CHUNK_SPEC`⟩ — **NOT MEASURED.** The chunk the floor is measured
> on (steps, shape, seed, device, operator) travels with the number, or the
> number is a constant validated on one shape and applied to another — the V-22
> class `V16_DEVICE_CERT.md` §3.2 names. Owed by the same node.

### A1.3 — WHAT THIS AMENDMENT IMPLIES THAT NOBODY HAS MEASURED

Written as slots, not as claims, per this round's first law.

~~⟨SLOT `SMPRIME_BACKWARD_UNDER_STRICT`⟩~~ — **FILLED WHILE THIS SECTION WAS
BEING WRITTEN, by the COSTS node.** `COSTS.md` §1.6 measures the workhorse arm
`ceq/arm_smprime.py` at the arms' reduction length 64, 8 repeats, both regimes:
hop and full forward **bitwise, max|Δ| = 0.0, under the flag ON as well as OFF**;
**gradient bitwise with the flag OFF and NOT EXECUTABLE with the flag ON** —
`RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation`,
raised by the **backward** of `cumprod`, which autograd computes with `cumsum`
`[MEASURED]`. **Moving the arm from a prefix scan to a path product did not
escape the missing kernel; it moved it from the forward to the backward**
(`COSTS.md` §1.6). This is the direct measurement behind Ruling 1's "the hole is
in backward", and it is why B2 is forward-only and A1.2 is a floor.

> ⟨SLOT `DECIDING_CELL_BITWISE_RETAKE`⟩ — **NOT MEASURED.** Whether each cell on
> the frozen list actually reproduces bitwise under strict mode on the certified
> 4060. Owed by the **re-take node** (ledger row 4): R1′ and every deciding cell
> re-run on the certified device before launch. CPU cells stay in the journal as
> `device:cpu`, superseded, never deleted (L-G2); tolerance widening to absorb an
> unmeasured CPU↔CUDA delta is REFUSED.

### A1.4 — WHAT THE AMENDMENT DOES NOT DO

- It does not widen any tolerance. The floor is a **measurement**, not a bar
  chosen to admit a reading.
- It does not license `allclose` anywhere in B1 or B2.
- It does not change the flag. `warn_only=True` is the regime for the whole run;
  the strict forward cells set strict mode **for themselves**, and that is the
  only place strict mode appears.
- It does not touch `CEQ_V16_CONTRACT.md`.

---

## A2 — RULING 2, THE DOCUMENTARY HALF: WHERE THE CARD SPLIT LIVES

The card's split sentence is built in `MODEL_CARD.md`, section **"v17-K — THE
SHIPPED MODEL (Q3): CARD TEMPLATE, LIMITS FIRST"**, as a template with named
slots and **both branches written** — the text if `β` moves off `1` and the text
if `β` pins at `1`. The identity clause there cites the existing `β=0`
certificate **unchanged**: `V16_ARM_SMPRIME.md` verdict rows (a)–(c) and the
statements in `lean/CEQ/V16Domain.lean` they evaluate.

The code half of Ruling 2 — `β` LEARNABLE, INIT 1, per-instance `β` logged as a
column — is the **β node's** and is not discharged here.

> ⟨SLOT `BETA_PIN_CRITERION`⟩ — **NOT SPECIFIED.** Ruling 2 branches on whether
> `β` "pins at 1", and **states no test for it**. The card's two branches are
> both written; which one ships is a decision nobody has defined a rule for.
> Owed by the author / the β node, before Q3's card is finalised.

---

## A3 — RULING 3, THE EXACT COUNTS

`V17_ARM_WIRING.md` §5 `[MEASURED]`, this box, CPU, float32:

| shape | `operator="smprime"` (arm) | `operator="sgate"` (softmax-shaped control) | difference |
|---|---|---|---|
| `d=32, L=2, H=4, V=32, S=16` | **27,914** | **27,776** | **+138** = `2·(2·(32+1)+3)` |
| `ceq/hf/train.py::DEFAULTS` — `d=512, L=8, H=8, V=256, seq=512` | **25,736,232** | **25,728,000** | **+8,232** = `8·(2·(512+1)+3)` = **+0.03200 %** |

Formula `n_layers · (2·(d+1) + 3)`: the arm's **two per-position heads**
(`m_head`, `theta_head`, each a `[1,d]` weight plus a `[1]` bias) and **three
scalar switches** (`beta`, `qk`, `g`) — named tensor by tensor in
`V17_ARM_WIRING.md` §5, with the assertion that the control has **no** parameter
the arm lacks. Cross-checks `V16_ARM_SMPRIME.md` §1: `4,806 − 4,769 = 37 =
2·16 + 5` at `d_model = 16` `[MEASURED]`.

**Which way it cuts, recorded because it is the caveat that matters:** the arm
carries **more** parameters, so a reading favourable to the arm is the one that
needs the caveat (`V17_ARM_WIRING.md` §5, FINDING FOR THE AUTHOR).

The COSTS line Ruling 3 requires is **drafted, not landed** — `COSTS.md` belongs
to another node and was being written while this was filed. That file's §4
(“SLOTS OWNED BY OTHER NODES”) already carries the placeholder — *“RULING 3 —
matched params … **☐ NOT YET WRITTEN.** Owner: docs”* — and the paste-ready text
for it is in `V17_R1_R3_R7_EDITS.md`, section “COSTS LINES, PASTE-READY”.
**No edit to `COSTS.md` was made by this node.**

---

## A4 — RULING 7, ATTACH-LIST ITEM 4

Attach-list item 4 asked for BED-M "as the intact **211,765-line** file (not
regenerated; hash-pinned)". **That file does not exist and none was created**
(`V17_G05_DATA.md` §6 `[MEASURED]`: every file in the working tree over 1 MB was
line-counted; exactly one has 211,765 lines and it is `data/tinystories_20k.txt`,
TinyStories, not a chain corpus). BED-M has no on-disk artifact at all — it is
`ceq/corpus.py::build()`, named as BED-M by `ceq/beds/__init__.py:3`.

**Regime, per Ruling 7:** generator + seed + expected hash, regenerated and
asserted in-notebook, **same as BED-K and BED-1**. On Kaggle **the hash
assertion IS the provenance.** Digests cited from `results/k_data_manifest.json`,
not retyped from memory; corrected language filed in `kaggle/README.md`.

---

# RULING 2a — "PINNED" DEFINED (closes the Ruling-2 hole)

Filed after the docs node reported that Ruling 2 branched on "β pins at 1" and
never defined it. Ledger row 2 stays OPEN until the β node and the floor node
land the two measurements below.

**BASIS.** The Ruling-1 identical-seed pair yields `δ_β = |β⁽ᵃ⁾ − β⁽ᵇ⁾|` **per
β-parameter** at end of training — the noise floor measured in β's OWN units,
from runs already paid for. The loss-space floor was one instance of a general
rule, now stated:

> **A QUANTITY'S NOISE FLOOR IS MEASURED IN THAT QUANTITY'S UNITS FROM THE
> IDENTICAL-SEED PAIR.**

The unit-transfer weakness raised against the loss-floor proposal is thereby
**dissolved, not answered**.

**CRITERION.** `β_i` is PINNED iff `|β_i,final − 1| ≤ 5·δ_β,i`. **`k = 5` is
labeled HEURISTIC SCALE, not a CI** — one pair is one draw of the noise, and
the label says so on the card.

**QUANTIFIER** (the second hole the first was hiding). "β pins" is
**per-parameter**; the card reports the DISTRIBUTION (min, median, max,
fraction pinned). The original two branches were a **false dichotomy**. Three:

| branch | condition | card says |
|---|---|---|
| A | **≥95 % pinned** | "the shipped model is the softmax-corner object carrying an unused exact-propagation corner" |
| B | **≤5 % pinned** | "training left the softmax corner; the model is the interpolated object, β distribution printed" |
| C | **else — MIXED** | the likely outcome and the most informative one: print the distribution **and WHERE the moved β's live** (which layers / heads), because a model that leaves the corner only in some heads is a finding about **where exactness pays** |

**GRADIENT CENSUS** (distinguishes two meanings of "pinned"). Per pinned `β_i`,
the **integrated `|∂L/∂β_i|` over training**, against the same statistic for
moved β's.

- **PINNED-WITH-SIGNAL** — gradient arrived and β returned to 1; softmax is
  genuinely preferred there.
- **PINNED-WITHOUT-SIGNAL** — the dial was never exercised; the claim weakens to
  **"unused"**, never **"preferred"**.

**The card sentence must use the census's word, not the flattering one.**

**PRECEDENCE.** This is the DEFAULT criterion. If the β node's proposal arrives
with a defensible basis that differs, it is adjudicated against this one by
**which claims less** — the criterion that licenses the weaker sentence wins
ties. (L-SIGN's direction, applied to definitions.)

**COST.** Zero new runs (the pair exists); one extra logged tensor (β trajectory
+ its gradient accumulator); one card template edit (two branches → three, plus
the census word).

---

# RULING 8 — Q2 DROPPED

**Q2 (R2 reproduction, `t*=8`, `n=16,384`) is DROPPED.**

The finding that forced it: `arm_smprime` at `n=16,384` reserves **10.578 GiB**
under a training loop against the 4060's **7.996 GiB**, so the shape **does not
fit the certified device**. The same shape is 59 % of a T4's budget, so it fits
Kaggle — which places Q2's deciding cell exactly where the KILL clause strikes
it: *local decides, and local cannot run it.* There was no re-take to price
because there was nothing to re-take.
(Secondary, and not the reason: `T_STAR = 2` is a module constant at
`scripts/v15_r1.py:124` with no `--t-star` flag.)

**Queue after this ruling:** K-CERT → **Q1** → **Q3** → **Q4** → cross-device
table → HF package on explicit say-so.

**CONSEQUENCE, FLAGGED NOT DECIDED.** R2's deciding cell cannot run on the
certified device either, by the same memory fact. Dropping the *reproduction*
does not supply the *reading*. Whether the round carries an R2 row at all is a
separate question this ruling does not answer, and it should not be answered by
silence.

**CONSEQUENTIAL EDIT OWED:** `kaggle/ceq_v17k.ipynb` still carries a Q2 cell.

---

# RULING 9 — THE REGIME FOR R1′ CELLS, DERIVED FROM THE NORTH STAR

**The question.** An R1′ cell takes **150 gradient steps**, so it is not
forward-only and strict determinism is **not executable** on it (autograd
differentiates `cumprod` with `cumsum`). Ruling 1's two clauses — bitwise for
replay+forward, measured floor for training — did not anticipate a *deciding*
cell that trains.

**The north star decides it** (`CEQ_V16_CONTRACT.md`, immutable):

> Attention that is **EQUAL to self-attention on its own ground**, built FROM
> softmax and AdamW, and capable on ground they cannot occupy.

R1′ exists to adjudicate an **equality claim**. An equality claim is only as
strong as the noise it is read against: *equal to within a tolerance nobody
measured is not a result.* So R1′ cells fall under the **measured-floor
clause**, and the floor is not optional decoration — **no equality or
non-equality statement may be made about an R1′ cell without its floor printed
beside it.**

**The floor for these cells, by Ruling 2a's general rule** (*a quantity's noise
floor is measured in that quantity's units from the identical-seed pair*):
`δ_nrmse` = `|eval_nrmse⁽ᵃ⁾ − eval_nrmse⁽ᵇ⁾|` over an **identical-seed repeat on
the certified device under the ruled regime** (`warn_only=True`). It is NOT the
seed-to-seed spread, which is a different quantity answering a different
question.

**Note for the measurement:** a flag-OFF identical-seed repeat during pricing
came back **bitwise identical** on `eval_nrmse`. If that survives under
`warn_only=True`, the floor is ~0 and the equality claims are held very tight —
which is the outcome the north star wants, and is exactly why it must be
measured rather than assumed.

---

# RULING 10′ — "PINNED" BY LIKELIHOOD RATIO

**Supersedes Ruling 10. The ulp criterion is STRUCK as engineering wearing a
ruling's hat.** Ruling 2a's `5·δ_β` threshold retires from the criterion with
it; `δ_β` survives as the diagnostic it always was.

**THE CRITERION.** Pinned is a **1-dof nested-model comparison**, decided by the
canonical constants:

    Λ = 2·[ LL_eval(β_final) − LL_eval(β ≡ 1) ]

computed by **two deterministic forward passes on the held-out eval split** — no
training, no pairs, no floors.

| Λ | verdict |
|---|---|
| **≤ 3.841** (χ²₁ at 0.95, Wilks `[V]`) | **PINNED** — the data cannot reject that the shipped model is the softmax-corner object |
| **> ln n** (BIC `[V]`, `n` = eval count; `[RUN: 8.29 at n = 4000]`) | **MOVED** — the dial buys its description length |
| **3.841 < Λ ≤ ln n** | the card prints the **interval verdict** — *"rejected at 0.95, below description-length"* — both constants, no interpolation, no invented `k` |

**WHY THE DEGENERATE FLOOR DISSOLVES RATHER THAN NEEDING A PATCH.** Wilks'
randomness is over the **DATA**, not the optimizer. A bitwise training pair is
simply **irrelevant** to the test. (The measured `δ_nrmse = 0.0` on the certified
device, which broke the `5·δ_β` criterion by making its tolerance exactly zero,
therefore stops mattering rather than needing a floor-of-the-floor.)

**THE UNITS OBJECTION, CLOSED BY THE METRIC.** The Wald form
`(β_final − 1)²·I_β ≷ 3.841`, with `I_β` the observed Fisher information in β, is
the asymptotically equivalent parameter-space statement. **Fisher information IS
the converter between parameter displacement and likelihood displacement.** The β
node's original objection was correct and is **answered by the metric, not
waived**.

**HONEST POWER, PRINTED.** The verdict ships with its minimum detectable
departure:

    |β − 1|_min  ≈  √( 3.841 / (n · I_β) )

`[RUN: a real 2 % departure is detected only 32 % of the time at n = 4000]` — so
**"PINNED" means "indistinguishable at this resolution", and the resolution is a
number on the card, never an implication of exactness.**

**UNCHANGED:** the three-branch card (2a's quantifier), the **gradient census**
(pinned-with-signal vs pinned-without-signal — still the scientific payload),
per-parameter reporting.

**COST.** Two forward passes and one Hessian-diagonal read. Zero new runs, zero
invented constants — `3.841`, `ln n` and `2` were on the shelf the whole time.
