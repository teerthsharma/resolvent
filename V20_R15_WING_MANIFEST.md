# V20 R15 — THE FROZEN WING MANIFEST

Frozen at it.4 on branch `v17k-gate0`, repo HEAD `207e7b9`. This document is
the list; `tests/saturn/test_v20_r15_freeze_manifest.py` is the check. Nothing
here is meant to be read charitably — every claim below is a node.

**The coordinator rules on N. This manifest states what survives the rubric and
what the record can no longer support.**

```
FROZEN-N = 2
FREEZE-SHA256 = fbf17e07e6cab495bd1fdbb4392ab88e92f5b0610e40d2b8a283d8bcb9c542ff
RETIRED = arm_phase PRICE 5.114 s
```

---

## 1. THE FROZEN LIST

| # | wing | the primitive | (a) distinct PRIMITIVE | (b) EVIDENCE IN THE LEDGER | (c) a KILL it accepted | (d) COST FOOTPRINT |
|---|---|---|---|---|---|---|
| **W1** | **ARM S-M′** (`arm_smprime`) | the **path product** `G_ij = Π m_k e^{iθ_k}` as the hop, with `β`/`QK`/`g` switches | `ceq/arm_smprime.py:144` | `results/v17k_r4_retake.jsonl:161` | `V16_ARM_SMPRIME.md:529` | `V17_R4_RETAKE_PRICE.md:194` |
| **W3** | **ARM PL** (`arm_pl`) | the **real prefix scan** `C = scan(g)` entering key-side only, over a value-zero BOS sink | `ceq/arm_pl.py:88` | `results/v17k_r4_retake.jsonl:25` | `V15_LEDGER.md:786` | `V17_R4_RETAKE_PRICE.md:195` |

Wing ids are held at their it.1 values — `W1`, `W3` — so the strike of `W2` is
visible as a gap rather than hidden by renumbering.

```freeze-manifest
# wing | clause | path:line | anchor that must occur on that exact line
W1|a|ceq/arm_smprime.py:144|def path_product(a: torch.Tensor)
W1|b|results/v17k_r4_retake.jsonl:161|0.9260818361
W1|c|V16_ARM_SMPRIME.md:529|exp_scan
W1|d|V17_R4_RETAKE_PRICE.md:194|15.970
W3|a|ceq/arm_pl.py:88|def scan(g: torch.Tensor)
W3|b|results/v17k_r4_retake.jsonl:25|0.6446726192
W3|c|V15_LEDGER.md:786|ARM PL with the parity claim RETIRED
W3|d|V17_R4_RETAKE_PRICE.md:195|1.614
```

**Clause (b) moved, and the move is the whole point of this iteration.** At it.1
both surviving wings cited a *document* for ledger evidence
(`workdonenewseal.md:91`, `V15_R1.md:250`). Both still resolve at HEAD, and both
are still true — but a document is exactly what a NAMED wing can also produce.
The frozen list cites a **journalled cell**: line `161` is `arm_smprime` seed 0
(`eval_nrmse 0.9260818361`, `secs 16.164`) and line `25` is `arm_pl` seed 0
(`eval_nrmse 0.6446726192`, `secs 1.884`), both `{"t": "cell", ...}` records in
`results/v17k_r4_retake.jsonl`. `test_the_frozen_list_names_only_found_wings`
now *requires* every clause-(b) citation to live under `results/`.

**Clause (d) carries the it.1 reservation unchanged.** These are GPU-seconds per
cell at a matched 150-step budget, not GPU-seconds-to-floor: no arm crosses the
floor (`V15_R1.md:250-251`), so the contract's criterion (3) statistic does not
exist for any wing, W1 and W3 alike. That is it.1's **K5** and it is unrepaired.

---

## 2. THE STRIKE — W2 `arm_phase`, and the mechanical difference between FOUND and NAMED

**The discriminator, stated so it can be run rather than argued:**

> A wing is **FOUND** iff `results/` holds at least one journalled record whose
> `kind` field is the wing's arm name. Otherwise it is **NAMED** — it exists in
> source and in prose, and the ledger has never seen it.

That is clause (b) with the charity removed, and it is the only reading under
which clause (b) and clause (d) are different questions: (d) asks what a cell
costs, (b) asks whether a cell was ever produced. A named wing can satisfy (a),
(c) and a *derived* (d) — `arm_phase` does, all three — and still have produced
nothing.

`[RUN]` the census over every `results/**/*.jsonl`, counting records by `kind`:

| `kind` | journalled records | files |
|---|---|---|
| `arm_pl` | **191** | 3 |
| `softmax` | 182 | 4 |
| `arm_smprime` | **182** | 2 |
| `arm_phase` | **0** | 0 |

The counter is calibrated on both sides in
`tests/saturn/test_v20_r15_freeze_manifest.py`: it must read `0` for `arm_phase`
and `> 0` for `softmax`, so a zero is a measurement and not a broken search
(`MISTAKES.md:117`, V-7).

**Four corroborating reads, each of which alone would be argument and together
are a mechanism:**

1. `results/` has zero `arm_phase` cells (above). Its `theta` is drawn
   `U(−π,π]`, never fit.
2. The runner cannot produce one. `scripts/v15_r1.py:179-184` `make_arm` has
   branches for `arm_pl` and `arm_smprime` only; anything else falls to
   `Arm(kind, s)`, whose bench dispatch ends at `ceq/bench.py:373`
   `raise ValueError(kind)`. `--arms` (`scripts/v15_r1.py:552`) has no
   `choices`, so `--arms arm_phase` is accepted at the CLI and dies inside.
3. `scripts/v15_r1.py:146` `GATED_ARMS = ("arm_pl", "arm_smprime")` and `:149`
   `ARM_MODULES = {...}` — the two places the file enumerates arms that carry a
   gate head. `arm_phase` is in neither, so even a hand-built instance would
   emit no gate columns.
4. The module says so itself: `ceq/arm_phase.py:476` — *"NOT TRAINED HERE and
   not by this node (L-LEAN). No optimizer, no gradient."*

And MERCURY finds criterion (3) **undefined** for it, which is the same fact
arriving from the arena's side.

**W2 is STRUCK from the wing list and RETIRED to UNPRODUCED.** Not refuted — the
closed-magnitude phase gate may well be a distinct primitive. It has never been
produced, and this round's binding kill is that a wing named rather than found
is struck.

### The replacement route, priced

A retirement without a price is an unfinished report (`STATE.md:12`). The route
back to the list is short, and it is arithmetic, not a research programme:

| item | cost | source |
|---|---|---|
| one W2 training cell, matched params, 150 steps, BED-M shape | **5.114 s** (`arm_pl` basis) | `V20_R15_IT3_MERCURY.md:237` |
| the same cell on the expensive basis, as a ceiling | **19.470 s** | `V20_R15_IT3_MERCURY.md:238` |
| eight cells, the width the other two wings carry | **≈ 130 GPU-s** on the `arm_smprime` ceiling; ≈ 41 GPU-s on the `arm_pl` basis | `DERIVED`, 8 × the rows above |
| a `make_arm` branch | `scripts/v15_r1.py:179-184`, three lines | `READ` |
| an `ArmPhase` optimizer path + entry in `GATED_ARMS`/`ARM_MODULES` | `scripts/v15_r1.py:146,149` | `READ` |
| a `PHASE_FIELDS` emit so the cell journals gate columns | new, alongside the existing per-arm emit | `READ` |
| strictness | the cell is `cumsum`-based and non-deterministic under `warn_only=False`; MERCURY's cheapest keeping route is to journal it as non-strict under the `warn_only=True` regime **every existing arena cell already ran under** | `V20_R15_IT3_MERCURY.md` §B.1, §B.3 |

**The W2 gate is under half a minute of wall clock on this box.** It was never a
cost decision. If the coordinator wants `N = 3`, the eight cells buy it, and the
manifest re-freezes with a new `FREEZE-SHA256` and W2's clause (b) pointing at
its own journalled cell like the other two.

---

## 3. WHAT THE FREEZE COVERS, AND WHAT IT DOES NOT

`FREEZE-SHA256` is `sha256` over the eight `wing|clause|path:line|anchor` rows
of the `freeze-manifest` block, normalised and sorted, joined by `\n`.

**It covers:** which wings are on the list, which line each clause cites, and
what string that line must carry. Add a wing, drop a wing, move a citation by
one line, or soften an anchor, and the digest moves.

**It does not cover:** the *content* of the cited files. Rewriting
`V16_ARM_SMPRIME.md:529` leaves the digest untouched — that is deliberate, and
it is why the anchor check is a separate node
(`test_every_frozen_citation_resolves_at_head`) that must be re-run at every
HEAD rather than trusted from the freeze. It also does not cover this prose, the
strike section, or the prices in it.

**The planted negative.** `test_moving_one_citation_by_one_line_fires_both_binds`
takes the clause-(a) row, adds `1` to its line number, and demands *two*
failures: the digest changes, and `resolve()` raises on the moved citation. One
failure without the other would show which of the two checks is asleep.
`[RUN]` GREEN.

**Re-verification at HEAD.** All twelve it.1 citations — including W2's four,
which are not withdrawn as citations, only as a wing — still resolve with their
anchors at HEAD `207e7b9`: `tests/saturn/test_v20_r15_wing_rubric.py`
`20 passed, 1 failed`, the one failure being the it.1 **K6** annex-producer bind,
which is not a citation node. **Zero P-6 drift.** The tree has gained ten files
since it.1 and no cited line has moved.
