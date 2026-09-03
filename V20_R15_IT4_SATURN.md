# V20 R15 it.4 — SATURN (WATSON): THE FREEZE

The list is frozen at `V20_R15_WING_MANIFEST.md` with **`FROZEN-N = 2`**,
`FREEZE-SHA256 = fbf17e07e6cab495bd1fdbb4392ab88e92f5b0610e40d2b8a283d8bcb9c542ff`,
and **W2 `arm_phase` struck and retired to UNPRODUCED at 5.114 s per cell**.
**The coordinator rules on N**; this office rules only that a third wing cannot
be defended on the record as it now stands, and states exactly what it would
cost to put it back.

New node `tests/saturn/test_v20_r15_freeze_manifest.py` — **18 GREEN**, RED
first, RED logged as its own event. No git write. Nothing touched Kaggle. Zero
moons: everything here is a local read or a local run.

---

## 0. THE C14 REPAIR, FIRST, BECAUSE IT IS WHAT WAS STRUCK

it.2's strike was a **log** defect, not a content defect: one event at
`house-events.jsonl:11719` with **no `t` field at all**, so RED and verdict
arrived in one atom and nothing in the log established RED-first.

This iteration's log, by line:

| line | `t` | what |
|---|---|---|
| `11759` | **`red`** | the freeze bind, RED, **before the manifest existed** |
| `11761` | `finding` | `FROZEN-N = 2`, the strike, the census |
| `11762` | `finding` | the digest and its planted negative |
| `11763` | `finding` | zero P-6 drift at HEAD |
| `11764` | `done` | close |

And the repair is bound rather than promised —
`test_every_it4_saturn_event_carries_a_t_field` and
`test_the_it4_red_is_its_own_event_and_precedes_every_finding` read the log back
and assert both properties, with a calibration node that shows the detector can
see a `t`-less dict.

**A finding that came out of building it, and it is not mine alone.** `[RUN]` a
`t`-field audit over all `11,764` log lines: **116 events carry no `t`**. They
are not malformed — they use *other* schema keys. `:399` and `:993` (round-1
`chase`) use `kind`; **`:11744`, appended THIS round, uses `event`**. So the
log has three competing schemas and no bind on any of them, and a fourth
`t`-less event landed *after* the Inspector struck mine for exactly that. The
node above is scoped to this office's it.4 events deliberately: the log is
append-only, so a bind over the history would be a RED that can never go green
because the fix is forbidden. **The round-wide version is the coordinator's to
impose, and it is one line of pytest.**

---

## TASK A — THE FREEZE MANIFEST, AND WHAT DISTINGUISHES A FOUND WING FROM A NAMED ONE

### The discriminator, stated so it runs

> A wing is **FOUND** iff `results/` holds at least one journalled record whose
> `kind` field is the wing's arm name. Otherwise it is **NAMED**.

That is clause (b) with the charity removed, and it is the only reading under
which clauses (b) and (d) ask different questions: **(d) asks what a cell would
cost; (b) asks whether a cell was ever produced.** A named wing can satisfy (a),
(c), and a *derived* (d) — `arm_phase` satisfies all three — and still have
produced nothing. The it.1 rubric already contained this; what it lacked was the
requirement that clause (b) resolve **into `results/`** rather than into a
document. The frozen manifest now requires exactly that
(`test_the_frozen_list_names_only_found_wings`), which is why both surviving
wings' clause-(b) citations MOVED at the freeze.

### The census `[RUN]` — every `results/**/*.jsonl`, records by `kind`

| `kind` | records | files |
|---|---|---|
| `arm_pl` | **191** | 3 |
| `softmax` | 182 | 4 |
| `arm_smprime` | **182** | 2 |
| `arm_phase` | **0** | **0** |

Calibrated on both sides before the zero is used as a verdict (`MISTAKES.md:804`,
V-15): the counter must read `0` for `arm_phase` **and** `> 0` for `softmax`.

### W2 verified rather than taken from the brief

Each of the four claims in the dispatch was re-read at HEAD, and each holds:

1. **Zero trained cells anywhere** — the census above.
2. **`make_arm` has no `arm_phase` branch** — `scripts/v15_r1.py:179-184` branches
   on `arm_pl` and `arm_smprime` only and falls through to `Arm(kind, s)`.
   `[READ]` the fall-through terminates at **`ceq/bench.py:373` `raise
   ValueError(kind)`** — the brief said `scripts/v15_r1.py` raises it; it does
   not, it *delegates* to the file that does. Same verdict, corrected provenance.
   `--arms` (`scripts/v15_r1.py:552`) carries **no `choices=`**, so
   `--arms arm_phase` is accepted at the CLI and dies inside the bench.
3. **`scripts/v15_r1.py:146` `GATED_ARMS = ("arm_pl", "arm_smprime")` and `:149`
   `ARM_MODULES`** exclude it, so even a hand-built instance emits no gate
   columns.
4. **`ceq/arm_phase.py:476`** — *"NOT TRAINED HERE and not by this node
   (L-LEAN). No optimizer, no gradient."*

MERCURY's criterion-(3)-undefined is the same fact reached from the arena side.

**W2 is STRUCK. It is not refuted** — the closed-magnitude phase gate may well be
a distinct primitive, and its it.1 citations all still resolve. It has never been
produced, and the round's binding kill is that a wing named rather than found is
struck.

### The retirement, priced

| item | cost | source |
|---|---|---|
| one W2 cell, matched params, 150 steps | **5.114 s** (`arm_pl` basis) | `V20_R15_IT3_MERCURY.md:237` |
| the same on the expensive basis, as a ceiling | **19.470 s** | `V20_R15_IT3_MERCURY.md:238` |
| eight cells, the width the other two carry | **≈ 41 s** on basis, **≈ 156 s** on ceiling | `DERIVED`, 8 × the above |
| `make_arm` branch | 3 lines at `scripts/v15_r1.py:179-184` | `READ` |
| `ArmPhase` optimizer path + `GATED_ARMS`/`ARM_MODULES` entries | `scripts/v15_r1.py:146,149` | `READ` |
| `PHASE_FIELDS` emit | new, beside the existing per-arm emit | `READ` |
| strictness | non-strict under `warn_only=True` — **the regime all 24 existing arena cells already ran under** | `V20_R15_IT3_MERCURY.md` §B.1/§B.3 |

**Correction to the dispatch's arithmetic, and it favours the wing.** The brief
carries "~130 GPU-s for eight cells". Eight cells at the `arm_pl` basis is
`8 × 5.114 = 40.9 s`; at the `arm_smprime` ceiling it is `8 × 19.470 = 155.8 s`.
`130` is neither. **The true floor is ~41 GPU-seconds.** W2's gate is well under
a minute of wall clock on this box, and it has never been a cost decision.

---

## TASK B — TAMPER-EVIDENCE

`FREEZE-SHA256` is `sha256` over the eight `wing|clause|path:line|anchor` rows,
normalised and sorted, joined by `\n`.

**Covers:** which wings are listed, which line each clause cites, what string
that line must carry. **Does not cover:** the *content* of the cited files, the
manifest's prose, the strike section, or its prices. That separation is
deliberate — content is the anchor node's job, and it must be re-run at every
HEAD rather than inherited from a freeze.

**The planted negative was run against the file on disk, not simulated.**
`V20_R15_WING_MANIFEST.md`'s W1 clause-(a) row was edited `144 → 145`, the suite
re-run, then reverted:

```
FAILED ...::test_the_declared_hash_matches_the_rows
FAILED ...::test_every_frozen_citation_resolves_at_head[a]
2 failed, 13 passed in 0.61s

E   AssertionError: citation 'ceq/arm_smprime.py:145': anchor
    'def path_product(a: torch.Tensor)' is NOT on that line.
E     line 145 actually reads: '"""`G_ij = prod_{k=j+1}^{i} a_k` for `j <= i`, ...'
```

**Two failures, and both were demanded.** `test_moving_one_citation_by_one_line_fires_both_binds`
requires the digest to move *and* the moved citation to stop resolving; a tamper
that fired only one of the two would show which check is asleep. After revert:
**18 passed**.

---

## TASK C — THE CITATIONS AT HEAD

**Zero drift.** All twelve it.1 citations — W2's four included, since they are
withdrawn as a *wing*, not as citations — still resolve with their anchors at
HEAD `207e7b9`, after the tree gained ten files:

```
python -m pytest tests/saturn/test_v20_r15_wing_rubric.py -p no:randomly -q
1 failed, 20 passed in 2.97s
```

The single failure is it.1's **K6** annex-producer bind, which is not a citation
node — and **it has shrunk**: from four orphan tokens at it.1 to **two**
(`{'M9-F1 Cantelli/Boole cutoff': 'd=65', 'M2 gate-landscape theta': '1.3e-3'}`).
`d=20 at the same` and `0.492 vs KL 0.519` have gained producers since it.1. K6
is being paid down by someone; it stays RED until the last two are.

The whole directory: **`8 failed, 86 passed`**, all eight pre-existing (three
`test_journal_path_is_discoverable`, three `test_r10_it2_spotcheck_reds`, K6, and
it.3's own standing `>= 0.30` RED). **None from the new node.**

---

## RED, VERBATIM

```
python -m pytest tests/saturn/test_v20_r15_freeze_manifest.py -p no:randomly -q
11 failed, 4 passed in 0.63s

E   AssertionError: V20_R15_WING_MANIFEST.md does not exist. The list is not
    frozen, so N is undefined and every wing in this round is a named wing.
```

Logged at `house-events.jsonl:11759` with `"t": "red"`, **before** any it.4
finding. The **4 GREEN on that run are the discriminator itself** — the census
nodes do not read the manifest, so the found-vs-named verdict was already
standing when the list was still empty. That ordering is the point: the strike
of W2 does not depend on anything this office wrote afterwards.

---

## LIMITS

1. **`N` is a ruling, not a measurement, and it is not mine.** What is measured
   is `journalled_cells(arm_phase) == 0`. Whether that strikes W2 depends on
   reading clause (b) as *produced*, not merely *documented*. Read the other way,
   `N = 3` and W2 keeps its it.1 citations, all four of which still resolve.
2. **The discriminator is one-way.** It proves `arm_phase` was never journalled;
   it does not prove it *cannot* be. MERCURY's ruling is that it can, for 5.114 s.
3. **The hash is over rows, not over the tree.** A frozen list whose cited files
   are rewritten keeps its digest and loses its anchors. Only re-running
   `test_every_frozen_citation_resolves_at_head` detects that, and nothing forces
   that node to run at any particular HEAD.
4. **Clause (d) is still not GPU-seconds-to-floor** for either surviving wing —
   it.1's **K5**, unrepaired, and it applies to W1 and W3 equally.
5. **The `t`-field bind is scoped to five events.** 116 historical events remain
   `t`-less by three schemas, and one of those (`:11744`) was appended this round.
6. **`results/` was censused by `kind`, not read.** A record whose arm identity
   lives under a different key would be invisible to the counter; the calibration
   only shows the key works for the four arms that use it.

---

**Distance to the north star.** Unmoved, and the freeze narrows rather than
advances: two wings, both parametrizations of one softmax-class head, neither
standing on ground softmax cannot occupy. `workdonenewseal.md:526` still reads
*"It claims no capability advantage over softmax. None has been measured."*

**Scoreboard.** The `+2` for *"the list FROZEN with N wings, four citations
each"* is claimable on this filing: the list is frozen, every wing carries four
citations, every citation resolves with its anchor at HEAD, and the freeze is
hash-bound with a negative that was seen to fire. **The coordinator rules on N.**
