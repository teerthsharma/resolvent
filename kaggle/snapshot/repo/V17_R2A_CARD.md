# V17-K — RULING 2a's CARD EDIT: THE FALSE DICHOTOMY REPLACED WITH THREE BRANCHES

**LEAD CAVEAT.** Nothing in this work is a measurement. This node rewrote a
template to carry a decision the author already made in `V17K_RULINGS.md`,
RULING 2a. **No new claim was made, no number was invented, and no run was
started or launched.** Every figure quoted below is cited to the file that
measured it (its original branch-A/B card text, filed under the pre-2a
Ruling 2); every new figure needed by 2a's finer criterion is a named slot
marked `NOT MEASURED`, with the node that owes it. `kaggle` was not invoked;
`~/.kaggle` was not touched; no writing git command was run.

## Provenance

| | `git rev-parse HEAD` | `git status --porcelain` |
|---|---|---|
| **start** | `ab5b48547884e04258276e6e808d5a71ea65f917` | `M MODEL_CARD.md`, `M ceq/arm_smprime.py`, `M ceq/hf/configuration_ceq.py`, `M ceq/hf/modeling_ceq.py`, `M ceq/hf/train.py`, `M house-events.jsonl`, `M scale/identity_manifest.py`, `M scripts/v15_r1.py`; untracked `COSTS.md`, `V17K_RULINGS.md`, `V17_ARM_WIRING.md`, `V17_G01_IDENTITY.md`, `V17_G02_G03_CHECKPOINT.md`, `V17_G04_ENV.md`, `V17_G05_DATA.md`, `V17_G06_G07_CERT_COST.md`, `V17_G08_G09_AUTOPILOT.md`, `V17_GPU_QUEUE.md`, `V17_LABEL_CELL_REPAIR.md`, `V17_NOTEBOOK.md`, `V17_R1_R3_R7_EDITS.md`, `ceq/autopilot.py`, `ceq/kdata.py`, `kaggle/`, `requirements-kaggle.txt`, `results/k_cert_local.json`, `results/k_data_manifest.json`, `scripts/k_cert.py`, `scripts/k_cost.py`, `tests/gate0/` |
| **end** | `ab5b48547884e04258276e6e808d5a71ea65f917` (unchanged; no writing git command run) | same set, **plus** `?? V17_R2A_CARD.md` (this file), **plus** two entries that appeared mid-session from other nodes working the same tree and are not this node's: `?? V17_R4_RETAKE_PRICE.md`, `?? results/r4_price_probe.json`. `MODEL_CARD.md` and `V17K_RULINGS.md` remain `M` / already-untracked-and-edited, both edited further by this node |

**Files this node touched:** `MODEL_CARD.md` (the trained-model clause
rewritten; the identity clause and everything else untouched), `V17K_RULINGS.md`
(ledger row 2's dependency text only), `V17_R2A_CARD.md` (new, this file).
**Nothing else.** Not `COSTS.md`, not `CEQ_V16_CONTRACT.md`, not `kaggle/*`,
not `MISTAKES.md`, `LOOP_PROMPT.md`, `STATE.md`, and nothing under `ceq/`,
`scripts/`, `scale/`, `tests/` — those were read for citation only.

---

## 1. WHAT RULING 2a CHANGED, AND WHY THE OLD TWO BRANCHES WERE WRONG TO KEEP

The pre-2a card (filed by the R1/R3/R7 node, see `V17_R1_R3_R7_EDITS.md` §3)
had two branches: "`β` moved off 1" and "`β` pinned at 1," with Branch B
written as an unconditional four-part argument that the shipped model *is*
softmax — bitwise against `#5a`'s `softmaxAttn`, row sums exactly `1.000000`.
Ruling 2a rules that this was a **false dichotomy** for two independent
reasons, both now fixed:

1. **No test existed for "pins."** The old card's `⟨SLOT BETA_PIN_CRITERION⟩`
   read "NOT SPECIFIED." Ruling 2a supplies one: `β_i` is PINNED iff
   `|β_i,final − 1| ≤ 5·δ_β,i`, `δ_β,i` being that parameter's own spread
   across the Ruling-1 identical-seed pair — a quantity's noise floor measured
   in that quantity's own units.
2. **The criterion is per-parameter, not global**, and a real run will not
   cleanly split its layers into "all near 1" or "all elsewhere" — the
   likely and most informative outcome is a mix. A two-branch card had no slot
   for that outcome at all.

A subtler defect in the old Branch B was corrected while rewriting it, not
merely inherited: **"pinned" (within `5·δ_β` of 1) is not "exactly at 1."**
The old Branch B's row-sum and bitwise claims (`Σ_j|W_ij| = 1.000000`, bitwise
against `softmaxAttn`) are properties of the **operator at `β = 1` exactly** —
true, measured, and still cited — but Ruling 2a's pin criterion admits layers
that are merely *near* 1 within a heuristic noise band. Filing the old
Branch B's exact-corner claims unconditionally under the new, looser
"pinned" condition would itself be a transfer-across-corners error of exactly
the kind this card's own law forbids. The rewritten Branch A below keeps the
citations (they are real measurements about the theorem's corner) but adds the
missing sentence: **pinned-within-`5δ_β` is its own, weaker bind than
exactly-at-the-corner, and the branch does not claim bitwise equality for the
trained layers.** This is a correctness fix made in service of Ruling 2a's own
text, not a new claim about anything measured.

---

## 2. THE THREE BRANCHES, AS WRITTEN (verbatim from `MODEL_CARD.md`)

### BRANCH A — `≥ 95 %` of `β` parameters PINNED

> The shipped model is the softmax-corner object carrying an unused
> exact-propagation corner.

Two claims share the sentence and are kept separate in the card: (1) a
**structural** claim — the `β = 0, QK-off` corner's parameters are carried and
unreached, true regardless of any census; (2) a **census-gated** claim — the
sole point of entry for the word "preferred," `⟨SLOT GRADIENT_CENSUS_WORD⟩`
(mechanism in §3). The branch explicitly states what `≥95 %` pinned does
**not** claim: bitwise equality with the softmax theorem. Fraction pinned is
read off `⟨SLOT BETA_FINAL_DIST⟩`.

### BRANCH B — `≤ 5 %` of `β` parameters PINNED

> Training left the softmax corner; the model is the interpolated object, `β`
> distribution printed.

Card text: neither corner certificate applies to an interpolated checkpoint
(cites `corners_are_distinct`, `V16_ARM_SMPRIME.md` row (d), `[MEASURED]`); no
"preferred"/"unused" language attaches here — Ruling 2a's word-gate is not
wired to this branch, and the card does not add one on its own initiative.

### BRANCH C — else: MIXED

> Some layers are pinned near `β = 1`, some are not, and the finding is WHERE
> the split falls.

Card text requires the full distribution **and** the moved `β`'s locations
(`⟨SLOT BRANCH_C_LOCATIONS⟩`). Filed here with the honest limit Ruling 2a's
own phrasing runs into: it asks for locations "which layers, which heads,"
and **this architecture has no per-head `β` to locate** — `β` is one scalar
per layer, shared across all heads in that layer
(`ceq/hf/modeling_ceq.py::CEQAttention.__init__`; `beta_column`'s own
docstring: *"a `[n_layers, n_heads]` beta would be a construction the arm
does not have"*). The card reports layer-level locations only and says
explicitly why head-level ones are not reachable without a new construction
the round's first law already refuses. This is flagged rather than silently
dropped, and is the most consequential finding of this edit — see §5.

Branch selection itself is a slot, not a free choice: `⟨SLOT
BRANCH_VERDICT⟩` states plainly that the fraction-pinned figure in `⟨SLOT
BETA_FINAL_DIST⟩` is the only input, thresholds `≥0.95` / `≤0.05` / else, and
no branch may be filled before that figure exists.

---

## 3. HOW THE FLATTERING WORD WAS MADE UNREACHABLE WITHOUT THE CENSUS

The word **"preferred,"** applied to the softmax corner, is given exactly one
point of entry in the card: `⟨SLOT GRADIENT_CENSUS_WORD⟩`, in Branch A's
sentence 2. That slot is not free text — the card defines it as a **table
lookup**, with exactly two legal fills:

| `⟨GRADIENT_CENSUS⟩` over the pinned set | the ONLY legal fill |
|---|---|
| unanimous PINNED-WITH-SIGNAL | *"a genuine preference — the gradient repeatedly returned `β` to the corner"* |
| any PINNED-WITHOUT-SIGNAL present | *"not established as a preference — the dial was never exercised for at least one pinned layer, so the corner's word is 'unused'"* |

Three things make the word unreachable by a careless fill rather than merely
discouraged:

1. **The slot's legal values are enumerated in the card itself**, as a lookup
   table keyed by the census classification — not a blank a writer fills from
   judgment. There is no third value, and no way to write "preferred" as free
   prose elsewhere in the clause: the card states outright that "preferred"
   "may not be written anywhere in this clause from any source other than
   `⟨GRADIENT_CENSUS⟩`'s classification."
2. **The default on disagreement is the weaker word.** A mixed census
   (some pinned `β_i` WITH-SIGNAL, some WITHOUT) resolves to "not established
   as a preference," never to "preferred." This extends Ruling 2a's own
   PRECEDENCE clause ("the criterion that licenses the weaker sentence wins
   ties") from its stated use — choosing between competing pin-criteria — to
   an analogous tie inside a single criterion's output. **This extension is
   this document's own, not a quotation of Ruling 2a**, and is flagged as an
   interpretive choice in the card and here, for the author to confirm or
   override.
3. **The upstream slot the table reads from does not exist.** `⟨SLOT
   GRADIENT_CENSUS⟩` — the integrated `|∂L/∂β_i|` per pinned parameter — has
   no measurement behind it yet, and the card states the slot is "NOT
   FILLABLE" until it does. So today, literally nothing can fill
   `GRADIENT_CENSUS_WORD` at all; the gate fails closed, not open.

Branches B and C carry no "preferred"/"unused" language, so the gate's only
job is guarding Branch A, and it guards the single sentence where Ruling 2a's
own vocabulary could tempt a flattering fill.

---

## 4. EVERY NAMED SLOT, AND THE NODE THAT OWES IT

| slot | what it is | owed by |
|---|---|---|
| `DELTA_BETA_PER_PARAM` | `δ_β,i` for every layer — that layer's `β` spread across the Ruling-1 identical-seed pair | **the floor node**, `scripts/k_noise_floor.py` (today measures the loss-space floor only; a per-parameter `β` statistic off the same two chunks does not exist in that file) |
| `BETA_FINAL_DIST` | `β_i,final` per layer as `min/median/max/n`, **plus fraction PINNED** once the criterion is evaluated against `DELTA_BETA_PER_PARAM` | **the β node**, `ceq/hf/modeling_ceq.py::beta_column`/`beta_summary` (computes `final_min/median/max` today; applies no pin criterion — there is no `δ_β,i` to apply one against) |
| `GRADIENT_CENSUS` | per pinned `β_i`: integrated `\|∂L/∂β_i\|` over training, and its PINNED-WITH-SIGNAL / PINNED-WITHOUT-SIGNAL classification | **the β node**, `ceq/hf/modeling_ceq.py` — a new per-parameter gradient accumulator; `beta_summary`'s existing `max_abs_grad` is a max over logged steps, not the needed running integral, and does not fill this slot |
| `GRADIENT_CENSUS_WORD` | the single legal point of entry for "preferred"/"unused," filled only by table lookup against `GRADIENT_CENSUS` | derived, not independently owed — NOT FILLABLE until `GRADIENT_CENSUS` lands |
| `BRANCH_C_LOCATIONS` | which **layers** carry a moved `β`; explicitly **not** head-level, since no per-head `β` exists | **the β node**, `ceq/hf/modeling_ceq.py`, out of `BETA_FINAL_DIST` |
| `BRANCH_VERDICT` | which of the three branch texts the finished card carries — selected mechanically by `BETA_FINAL_DIST`'s fraction-pinned figure, `≥0.95`→A / `≤0.05`→B / else→C | derived — no node owes a "decision," only the figure that decides it |
| `Q3_RUN_SPEC`, `Q3_MEASUREMENTS`, `Q3_CHECKPOINT` | unchanged from the pre-2a card — data/steps/shape/seed/device/commit; the trained run's readings; the checkpoint itself | **the Q3 training node** (unchanged ownership) |

**Not a new slot, but worth stating precisely:** `BETA_PIN_CRITERION` is no
longer an open slot for a **rule** — Ruling 2a supplies the rule verbatim
(`|β_i,final − 1| ≤ 5·δ_β,i`, `k=5` a heuristic scale). What remains open is
the rule's **evaluation**, which is exactly `DELTA_BETA_PER_PARAM` and
`BETA_FINAL_DIST` above — no separate slot is needed for it and the card does
not carry one.

---

## 5. FINDING FOR THE AUTHOR: THE "WHICH HEADS" GAP

Ruling 2a's Branch C text asks the card to report, for the mixed case,
"WHERE the moved β's live (which layers, which heads)." The current
parametrization makes only the first half answerable. `β` is one scalar
`nn.Parameter` per attention layer in `CEQAttention`, **shared across every
head in that layer** — stated as a deliberate design choice, not an
oversight, in `beta_column`'s own docstring: *"GRANULARITY, STATED RATHER
THAN CHOSEN... A `[n_layers, n_heads]` beta would be a construction the arm
does not have."* Reporting a per-head location for a moved `β` would require
building exactly the widened parametrization that file already refuses.
`MODEL_CARD.md`'s Branch C states this gap explicitly rather than silently
reporting layers only and letting a reader assume heads were considered and
found uniform. This is not a defect this node introduced or can close — it is
a mismatch between Ruling 2a's phrasing and `ceq/hf/modeling_ceq.py`'s
existing, deliberate design, surfaced for the author's and the β node's
attention.

---

## 6. IDENTITY CLAUSE — UNCHANGED, VERIFIED BY DIFF

`git diff --stat -- MODEL_CARD.md` reads **327 insertions, 0 deletions**
against HEAD `ab5b485`, but that alone does not prove the identity clause was
untouched — the pre-2a card's own additions were themselves uncommitted, so a
diff against HEAD cannot distinguish "not touched by this node" from
"happened to also be an addition." The direct check: this node's two `Edit`
calls had `old_string` boundaries starting at `### Trained-model clause —
STATES ONLY WHAT WAS TRAINED` and at the closing bullet of `## Limits, first
(v17-K)`, respectively — neither overlaps `## The split sentence (RULING 2)`'s
opening paragraph or `### Identity clause — CITES THE β = 0 CERTIFICATE,
UNCHANGED` (rows (a)/(b)/(c), the Lean statement list, or its two travelling
caveats). A post-edit read of that section (`MODEL_CARD.md` lines 113–144)
was compared line-by-line against the pre-edit read (lines 105–136 before the
Limits-list addition shifted it by 8 lines) and is **byte-identical**.

---

## 7. WHAT THIS EDIT DOES NOT DO

- It does not close ledger row 2. `V17K_RULINGS.md`'s OPEN-ITEMS table row 2
  still reads **OPEN**, with its dependency text rewritten to name the two
  measurements it now waits on (`DELTA_BETA_PER_PARAM`, `GRADIENT_CENSUS`)
  instead of the pre-2a slot names.
- It does not touch `COSTS.md`, `CEQ_V16_CONTRACT.md`, `kaggle/*`,
  `MISTAKES.md`, `LOOP_PROMPT.md`, `STATE.md`, or anything under `ceq/`,
  `scripts/`, `scale/`, `tests/`.
- It does not invent `δ_β,i`, a gradient integral, a fraction pinned, or a
  branch verdict. Every one of those is a named slot.
- It does not resolve the "which heads" gap in §5 — it names it.
- It does not run `kaggle`, touch `~/.kaggle`, or launch anything.

---

*Nothing here moved toward a trained checkpoint. Bookkeeping in service of a
run that has not started — the same distance-to-north-star line the R1/R3/R7
node filed, still true of this edit.*
