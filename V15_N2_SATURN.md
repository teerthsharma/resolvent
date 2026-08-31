# V15 N2 — SATURN: re-date, L-EQ evidence, author-owned strikes

Node n2, CEQ v15 composition round. Measures only; repairs nothing (per task
brief). All three hard-constraint files touched: this file,
`results/v15_loop_suite5.txt`, and an append to `MISTAKES.md`.

---

## TASK A — re-dating the 15 standing loop failures against the live tree

**Command run:**

```
python -m pytest tests/loop -q --no-header -p no:cacheprovider
```

Full output saved to `results/v15_loop_suite5.txt`. Result line:

```
15 failed, 501 passed, 3 warnings in 42.49s
```

**The membership did not move.** All 15 node ids filed in
`V13_D3_LOOP_FORENSICS.md` (`:194-201`, dated against `r10_loop_suite4.txt`,
2026-08-31 00:02:24) are byte-for-byte the same 15 that failed just now. Zero
went green, zero are new. `grep -c "^FAILED"` on the new artifact returns 15,
and every line matches a filed entry exactly — verified by diff, not by count
alone (the forensics doc's own warning: "the count is stable and the
membership is not" does not apply this time — this re-run moved neither).

| # | failure (filed form) | filed status (V13_D3_LOOP_FORENSICS) | live status (`v15_loop_suite5.txt`) |
|---|---|---|---|
| 1-10 | `test_conftest_import_is_order_dependent.py::test_no_test_file_imports_conftest_as_a_bare_module[…]` — 10 parametrised cases: `tests/chase/{test_ceq_hub_package, test_hf_shipping, test_hub_package_hardening, test_kernel_contracts, test_rollback_flex_attention, test_schedule_rebuild, test_stochastic_P}.py`, `attic/tests/chase/{test_multizoom_cost, test_multizoom_kernel, test_multizoom_r5}.py` | STANDING, deferred by decision (`R10_GUARD_REPAIRS.md:132`) | **STILL RED**, all 10, identical parametrisations |
| 11 | `test_corpus_is_recoverable_and_verifiable.py::test_some_countable_unit_of_the_corpus_equals_the_readme_figure` | STANDING, blocked upstream (README states 20,000 lines, file has 211,766) | **STILL RED** |
| 12 | `test_corpus_is_recoverable_and_verifiable.py::test_the_readme_derivation_matches_the_corpus_it_describes` | STANDING, same block | **STILL RED** |
| 13 | `test_every_boundary_node_can_propagate.py::test_no_corpus_instance_has_a_boundary_node_that_cannot_propagate[reproduce-n1024-d4-t0.95]` | STANDING, route stated in-file only (amendment B on `admissible` deferred — corpus regen cost) | **STILL RED** |
| 14 | `test_manifest_refuses_an_absence_it_has_not_earned.py::test_no_weight_record_omits_a_declared_identity_field` | STANDING by contract decision, half-repaired (`device` absent from 60/60 weight records) | **STILL RED** — same assertion text, `over 60 records, declared CONFIG_FIELDS are absent: device in 60/60` |
| 15 | `test_the_bar_control_is_scored_out_of_sample.py::test_the_bar_control_is_not_scored_on_the_tensor_it_trained_on` | STANDING, bears on a live verdict (`calibrate_bar` trains and scores on the same `feats` tensor) | **STILL RED** — same assertion text, `calibrate_bar trains the positive control on ['feats'] and scores it on ['feats']` |

**Movers: none.** Green count: 0. New count: 0. Turnover: 0/15.

**What did move:** passed count, 499 -> 501 (+2), not part of the filed 15 —
two additional tests now exist in the suite and pass; this does not touch the
standing-failure membership. Total collected tests: 514 -> 516.

**Answer to (a):** live count is **15 failed / 501 passed**, identical
membership to the 15 filed at `r10_loop_suite4.txt` (00:02:24, 08-31). The
prior round's caveat — "the current live count may differ from 15" — did not
materialize; it is still 15, and it is still the same 15.

---

## TASK B — L-EQ: measured fraction of pre-v13 section-5 strikes that were `[V]`-as-theorem

**Contract claim** (`CEQ_V15_CONTRACT.md:54`): *"two-thirds of the pre-v13
section-5 strikes were `[V]`-as-theorem."*

**Method.** Read `attic/workdonenew.pre-v13.md` section 5, "CLAIMS DISPROVED,
WITHDRAWN, OR STRUCK" (`:160-211`). The table holds **44 rows**
(`sed -n '166,212p' … | grep -c '^| \*\*'` = 44). Of those, 4 rows record a
claim that *held* rather than one that was struck — the section title is
"disproved, withdrawn, or struck," and these do not fit any of the three:

- `:190` "the hop-2 prediction" — **4/4 HELD**
- `:204` X32 §2's three-word verdict — **WORKS**
- `:205` X31's horizon arithmetic — **CONFIRMED**
- `:179` "the seed spread is the noise model" — **ALREADY GUARDED**

That leaves **40 genuine strikes**. `[V]`-as-theorem, per `CEQ_V15_CONTRACT.md
:51-52`, is specifically: a source is cited (its page fetched, its intro or
description matched) and treated as licensing a claim, when the source's own
theorem — read with its hypotheses — does not support the claim as used. This
is narrower than "any wrong claim involving a formula": an internal arithmetic
slip (sizing factors, thread floors, CI widths) is not a citation failure, and
a theorem correctly cited but misapplied to a mislabeled quantity is not
either (the source's equation *does* support what it actually says).

Applying that test to all 40 strikes, four qualify:

| strike | line | what was cited | what its own equation/hypotheses actually require |
|---|---|---|---|
| "ARL₀ from the closed form is the false-alarm rate" | `:177` | Siegmund's closed-form ARL₀ | valid only under i.i.d. nulls; measured autocorrelation `φ̂=0.709` makes it wrong by **24.1×** (simulated ARL₀ 41.5 vs nominal 1000) |
| "X₂₇a's box-counting `d` feeding `κ = λ(1−d)`" | `:196` | the Kantz–Grassberger relation | every fetched statement of it uses the **information** dimension `D₁`, not box-counting `D₀`; substituting `D₀` under-predicts `κ` by a margin that grows with the branch's `‖f′‖` spread (0.49% to 24.5% across the worked cases) |
| X32 "`ρ_P = √2` for a planted antisymmetric A" | `:200` | the random-matrix asymptotic `E‖J−Jᵀ‖²/E‖J‖² → 2` | that value is the ensemble-average for a **generic random** J; a matrix that is **exactly** antisymmetric gives `ρ_P = 2`, not `√2` — measured `2.000000` at n = 4, 16, 64, 256 |
| X32 "Poincaré–Hopf gives Σ index = 1 for the replicator" | `:201` | the Poincaré–Hopf theorem | requires the vector field **transverse** to the domain boundary; at `μ=0` the field is tangent everywhere (`dz₀ = −0.000000e+00`, measured), so the hypothesis fails and the index sum is not pinned to χ |

**Measured fraction: 4 / 40 = 10%** (one in ten), not two-thirds (66.7%).

**Rows considered and excluded, with reasons** (so the count is checkable, not
asserted):

- X27b "no predictor can exceed exit accuracy `1−c·f(ε)`" (`:180`) —
  *excluded*: the entry states "no such theorem exists… and no source fixes
  `c`" — nothing was cited for the false claim in the first place, so this is
  fabrication-without-citation, not citation-with-unhonoured-hypotheses.
- X27c "H and α̂ are two coordinates" (`:182`) — *excluded*: Daza et al. is the
  source that *corrects* the claim, not a source the original claim leaned on;
  no citation is named for the "two coordinates" belief being struck.
- X27d "κ̂ is a second, independent rate theory" (`:183`) — *excluded*: the row
  ends "Citation owed (`[U]`)" — explicitly *no* citation existed to misapply.
- X32 "Fisher gives `dfbar/dt = Var_z(f)`" (`:202`) — *excluded*: the row's own
  verdict is "**CORRECT THEOREM**, WRONG QUANTITY" — the cited equation does
  hold as stated; the error was conflating `P` and `f̄`, not the citation.
- X28c "three instruments certifying each other" (`:208`) — *excluded*: the
  cited relation `κ=λ(1−D₁)` (Drótos et al.) is correctly read and does hold
  for hyperbolic systems; the strike is that agreement is then tautological,
  not that the source failed to back the claim.

**Answer to (b):** measured fraction is **4/40 (10%)**, not the two-thirds
the contract states. The contract's number does not match the document it
cites as its own evidence. This correction is written into the `MISTAKES.md`
entry below (P-10) rather than silently reconciled.

---

## TASK C — the six author-owned strikes

Searched `attic/workdonenew.pre-v13.md`, `V13_CLAIM_AUDIT.md`, and
`STRUCK.md` for strikes attributed to the author's own hand — a number that
entered the record with no instrument behind it, or an instrument-voice claim
("[RUN]") attached to a producer that does not exist — as opposed to a strike
caused by a bug in a script, a stale parameter, or a data artifact.

`attic/workdonenew.pre-v13.md` and `V13_CLAIM_AUDIT.md` contain no strikes
phrased this way (`grep -iE "author|FABRICATED|no producer|no live
producer|RUN voice|has no possible source"` returns nothing in either file
except one incidental phrase at `V13_CLAIM_AUDIT.md:217`, "cannot be checked
by anyone, including its author" — commentary on an unreproducible throughput
figure, not a strike attribution). All six live in `STRUCK.md`, which lists 12
struck **constants** collapsing into 7 struck **claims** (CI endpoints are
struck together with their point estimate). One of the 7 — the M2 decay
exponent `-1.389` (`STRUCK.md:22`) — is instrument-caused: a `floor = 1e-6`
artifact in the measurement pipeline, not something the author asserted
without a producer. The other six are:

1. **`-0.4654`** (K1 "live rows only" slope, with CI `-0.5173`/`-0.4160`) —
   `STRUCK.md:24-26`. Published in `D1.md` and `done5.md`; **no producer
   exists in any `.py`/`.json`/`.jsonl`/`.txt` in the tree** — the only script
   that computes a live-rows slope emits a different number entirely and
   exits 1. Mechanism: a number was written into prose narration with no
   instrument ever having produced it.
2. **`-1.826`** (M2 slope "as first reported") — `STRUCK.md:27`. Contradicted
   by the iteration-20 measurement (`-1.298`); **"propagated in a RUN voice at
   iteration 17"** — carried forward as if independently verified when it was
   the author's own earlier, superseded report.
3. **`1.471448`** (M5 tail norm, s=128) — `STRUCK.md:28`. **FABRICATED** —
   an exhaustive 1,800-setting sweep produced zero instances of it; measured
   values are `0.880500`/`0.882030`.
4. **`1.343174`** (M5 tail norm, s=512) — `STRUCK.md:29`. Same fabrication
   event as #3, different `s`; measured value `1.292741`.
5. **`0.743864`** (U1/N3 pilot Spearman rho, with CI `0.656532`/`0.816955`) —
   `STRUCK.md:30-32`. The strongest instance: **"NO PRODUCER HAS EVER
   EXISTED"** — `git log -S` across every ref finds zero commits defining any
   of the nine functions the battery is supposed to call. "The number has no
   possible source in any state this repository has ever been in."
6. **`5.4944e-13`** (Karcher residual, float64) — `STRUCK.md:33`. **Asserted
   in a `[RUN]` voice with no live producer** — it lived only in a code
   comment and in prose. (Its numeric origin — a throwaway script's `nan`-
   riddled output — is instrument-caused; what makes *this* a strike is the
   author's claim that it was a verified run, which it never was. This is
   also `MISTAKES.md` P-1.)

**Answer to (c): exactly six**, all in `STRUCK.md`, none in the other two
files searched. The contract's "the six author-owned strikes" (it.29 line)
matches the count found here exactly — unlike Task B's two-thirds figure,
this one holds up.

---

## Summary — answers to the three return items

**(a)** Live: **15 failed, 501 passed** (`results/v15_loop_suite5.txt`).
Membership unchanged from the 15 filed at `r10_loop_suite4.txt` — 0 resolved,
0 new, 0 turnover. The only movement is +2 passing tests outside the standing
set.

**(b)** Measured: **4/40 (10%)** of the pre-v13 section-5 strikes were
`[V]`-as-theorem, against the contract's stated two-thirds (66.7%). The four:
the Siegmund ARL₀ closed form (i.i.d. hypothesis dropped), the
Kantz–Grassberger relation (wrong dimension type substituted), the
random-matrix `ρ_P=√2` value (misapplied to an exact-antisymmetric case), and
Poincaré–Hopf (transversality hypothesis unchecked at μ=0).

**(c)** **Six** author-owned strikes found, all in `STRUCK.md`: the K1
"live rows" slope, the M2 slope "as first reported," two M5 tail-norm
fabrications (s=128, s=512), the U1/N3 pilot Spearman rho, and the Karcher
residual `5.4944e-13`.
