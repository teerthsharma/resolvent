# V20 R15 it.1 — SATURN: THE WING LIST, MINED

Phase A discovery under `CEQ_V20_R15_CONTRACT.md:76-84` (L-FIND). Every wing
below carries four `path:line` citations, one per clause, each verified by
`tests/saturn/test_v20_r15_wing_rubric.py` — which resolves the file, the line,
**and an anchor string that must occur on that exact line**. A citation that
merely points somewhere real does not count.

**Nothing was trained. No git write was run. Kaggle was not touched.**
Repo HEAD `207e7b9`, branch `v17k-gate0`.

**N = 3.**

---

## 1. THE WING LIST

| # | wing | the primitive | (a) | (b) | (c) | (d) |
|---|---|---|---|---|---|---|
| **W1** | **ARM S-M′** | the **path product** `G_ij = Π_{k=j+1..i} m_k e^{iθ_k}` as the hop, with `β`/`QK`/`g` switches | `ceq/arm_smprime.py:144` | `workdonenewseal.md:91` | `V16_ARM_SMPRIME.md:529` | `V17_R4_RETAKE_PRICE.md:194` |
| **W2** | **ARM PHASE** | the **closed-magnitude phase gate** `a = m·e^{iθ}`, `m = clamp(u,0,1)` — `0` and `1` attainable values, not limits | `ceq/arm_phase.py:96` | `MISTAKES.md:2154` | `V16_ARM_SMPRIME.md:415` | `V16_DEVICE_CERT.md:1087` |
| **W3** | **ARM PL** | the **real prefix scan** `C = scan(g)` entering key-side only, over a value-zero BOS sink | `ceq/arm_pl.py:88` | `V15_R1.md:250` | `V15_LEDGER.md:786` | `V17_R4_RETAKE_PRICE.md:195` |

What each clause reads, verbatim:

- **W1 (a)** `def path_product(a: torch.Tensor) -> torch.Tensor:` — **(b)** `oracle gates ⇒ label  5.919777e-16  1e-6  arm_smprime` — **(c)** `| **(c) must-fire: the `exp_scan` route** | **`0.000000`** | **`0` predicted positives**` — **(d)** `| `arm_smprime` | **15.970** | 17.839, 14.101 |` (s / 150 steps, CUDA, mean of 2).
- **W2 (a)** `def gate(u: torch.Tensor, theta: torch.Tensor, *, cap: bool = True)` — **(b)** `| zero-step `ArmPhase`, 8 seeds | `0.300689` |` — **(c)** `CLOSED lower endpoint — MORE than the `50.08%` that kills `arm_phase` at step 0` — **(d)** `  arm_phase  0.02088 s/step   x2.085`.
- **W3 (a)** `def scan(g: torch.Tensor) -> torch.Tensor:` — **(b)** `| **ARM PL** | 8 | **0.829151** | 0.253673 | **[0.617075, 1.041227]** | … | **NO** | 0.232077 |` — **(c)** `ARM PL with the parity claim RETIRED` — **(d)** `| `arm_pl` | **1.614** | 1.622, 1.605 |`.

**The three wings are the three built arms, and that is not a coincidence — it
is clause (d).** Every other candidate in this repository dies on cost, not on
merit. The only per-arm cost measurements that exist anywhere in the tree are
the device-certificate multipliers (`V16_DEVICE_CERT.md:1085-1087`, three arms)
and the R4 re-take timings (`V17_R4_RETAKE_PRICE.md:194-196`, three arms). No
other approach in `ARSENAL.md`, `ceq/`, or the X-items has ever been timed.

```l-find-citations
# wing | clause | path:line | anchor that must occur on that line
W1|a|ceq/arm_smprime.py:144|def path_product(a: torch.Tensor)
W1|b|workdonenewseal.md:91|5.919777e-16
W1|c|V16_ARM_SMPRIME.md:529|exp_scan
W1|d|V17_R4_RETAKE_PRICE.md:194|15.970
W2|a|ceq/arm_phase.py:96|def gate(u: torch.Tensor, theta: torch.Tensor
W2|b|MISTAKES.md:2154|0.300689
W2|c|V16_ARM_SMPRIME.md:415|kills `arm_phase` at step 0
W2|d|V16_DEVICE_CERT.md:1087|arm_phase  0.02088 s/step
W3|a|ceq/arm_pl.py:88|def scan(g: torch.Tensor)
W3|b|V15_R1.md:250|0.829151
W3|c|V15_LEDGER.md:786|ARM PL with the parity claim RETIRED
W3|d|V17_R4_RETAKE_PRICE.md:195|1.614
```

### The clause-(d) reservation, and it is load-bearing

The contract asks for **GPU-seconds-to-floor** (`:81`, `:121`, and criterion (3)
at `:127`). What the three wings have is **GPU-seconds**, not GPU-seconds *to
floor* — because **no arm has ever crossed the floor.** `V15_R1.md:250` reads
`crosses? NO` for ARM PL and `V15_R1.md:251` reads `NO` for softmax, and the R4
re-take on the certified device moved no verdict (`V17_R4_RETAKE.md:194`, *"Every
`dist_to_floor` keeps its"* sign, zero sign-flips in 16). **A time-to-an-event
that never occurs is not a measurement**, so clause (d) is satisfied here in its
cost-footprint sense and is `NOT MEASURED` in its to-floor sense for all three
wings alike. This is priced as **KILL K5** below, with its route.

---

## 2. THE NEAR-MISS TABLE

| # | approach | has | MISSING | cheapest thing that closes it |
|---|---|---|---|---|
| **NM-1** | signed path-sum bench family — `signed`, `sgate`, `tgate`, `tgatex`, `signmag`, `deltanet`, `paraformer` | (a) `tests/loop/test_arms_distinct.py:53` + measured distinctness `results/iter06_pivot_arms_distinct.txt:2` (*7 arms, 21 pairs, 0 collisions*); (b) `ARSENAL.md:262` frustration table `[RUN]`; (c) `ARSENAL.md:255` *"the kill does NOT fire"* | **(d)** | One run of `scripts/v16_device_probe.py` with its `ARMS` tuple (`:80`, today `("softmax","arm_pl","arm_phase")`) extended over the bench family. The probe already emits s/step per arm; this is minutes, not hours. |
| **NM-2** | A2 non-normal / numerical-range guard `w(A) ≤ ρ` | (a) `ceq/nonnormal.py:1`; (b) `ARSENAL.md:71` `[RUN]` `w(A) = 1.499315`; (c) `ARSENAL.md:99` *"Kill: enforce the constraint…"* | **(d)** | Same probe. The object is a shipped operator, so it is timeable as-is. |
| **NM-3** | B2 discrete Morse / free-face eviction **as a certificate** | (a) `ceq/eviction.py:1`; (b) `lean/CEQ/Refcount.lean:160` `free_face_floor_unchanged` (Lean, `0 sorry`); (c) `ARSENAL.md:209` | **(d)** | A certificate is not an arm and may never get a cost. If it enters the arena as an arm component, price it there; otherwise reclassify as an instrument under contract `:87-88`. |
| **NM-4** | the residual gate / adoption path (`out = stock_attention + Σ(αA)^h v`, bitwise stock at `α=0`) | (a) `ceq/hybrid.py:1` | **(c) and (d)** | (c) first: it has **no kill anywhere in the record** (search below). Register one — the obvious pre-registered kill is *"α trained away from 0 buys no NRMSE at matched params."* |
| **NM-5** | the scoped hop cache (exact incremental decode, K rows/token) | (a) `ceq/hopcache.py:1` | **(c) and (d)** | Same shape: **no kill in the record**. Register *"decode output not bitwise against prefill"*, which the module already claims and a test can bind. |
| **NM-6** | X₃₅′ hidden-cause detector | (b) `V15_X35A_RESIDUAL.md:144`/`:147` must-fires `[RUN]`; (c) `V15_X35A_RESIDUAL.md:301` the explicit kill | **(a) and (d)** | (a) is the real gap: it is an **instrument**, not a primitive for the label class. Contract `:87-88` files instruments and training methods as MULTIPLIERS, not arms. Reclassify rather than close. |
| **NM-7** | X₃₇ topological certificates (`Z` winding, `β₁`, Euler–Poincaré) | (b) `workdonenewseal.md:265`; (a-partial) `ceq/certs/topological.py`, `ceq/rips.py` | **(a) and (d)** | Same reclassification as NM-6. |
| **NM-8** | X₃₈ lane coupling / the composed arm | (b) `V16_PRICING.md:612` prices it and marks it `NOT MEASURED` | **(a), (c), (d)** | **Not built.** No anti-alias / lane-coupling module exists under `ceq/` or `scripts/`. Building it is contract v16 it.15 work, never discharged. |
| **NM-9** | the scan skyline (`m=93`, 4,769 matched) | (b) `CEQ_V16_CONTRACT.md:166` | **(a)** | **Not built**, and stated so in shipped code: `scripts/v15_r1.py:763-770` sets `dist_to_skyline = None` with the reason *"no v15/v16 scan-skyline module exists (R-SKY)"*. Contract `:85-86` files the skyline as the **cost floor, not a contender**, so it should not become a wing even once built. |
| **NM-10** | the fractional / GL-weights head (X₃₄) | (b) `workdonenewseal.md:465` (prior art `[V-eq]`) | **(a), (c), (d)** | Not built as an arm. `fractional` resolves only to `ceq/beds/bed_k.py` (the *bed*) and `scripts/v15_n1_probes/p07_arfima_gl.py` (a prior-art probe). |

```l-find-absence
# near-miss | regex searched across root *.md for a co-occurring kill word
NM-4|hybrid|residual gate
NM-5|hopcache|hop cache
```

Both searches return **0 hits**, and the search is calibrated: run against
`ΔFloor|DeltaFloor` it returns the kill recorded at `ARSENAL.md:303`. An absence
search that cannot find a present kill is void (`MISTAKES.md:117`, V-7), so the
control is part of the instrument rather than a courtesy.

---

## 3. THE X-REGISTRY FINDING

**It is not a file, and no single register file exists.** The contract names it
at `CEQ_V20_R15_CONTRACT.md:82-83` as a mining target beside `results/`, `lean/`,
`MISTAKES.md` and `STRUCK.md` — four things that *are* paths. It is the only one
of the five that is not.

Evidence, all `[RUN]` this session:

1. The literal strings `X-registry`, `X registry`, `X_REGISTRY`, `xregistry`,
   `X-ITEM`, `X item`, `X-items` occur **nowhere** in the tree in any file type,
   including `kaggle/` and excluding only `.git/`. The single case-insensitive
   hit is the unrelated phrase *"six-item"* at `V16_CALIBRATION.md:70`.
2. `find . -iname "*registr*"` returns only `PREREGISTRATION_HOLE_AUDIT.md` and
   three `tests/cameron/test_*_registration.py` files. None is an X-register.
3. The X-items are **scattered across at least seven documents** and are
   numbered non-contiguously: X1–X4, X6–X8, X11, X17–X22, X26–X37 with letter
   suffixes (X27b/c/d, X28c, X29a/b, X31a/b/c, X35a/p), plus unicode-subscript
   forms `X₃₂`–`X₃₈` used only in the v16 contract. X5, X9, X10, X12–X16,
   X23–X25 have no referent anywhere.

**The closest thing to a register, and it is partial.** `V13_DAG_TASKLIST.md:9-28`
is the one table in the tree with a row per X-item carrying an output file and a
status (`DONE` / `STRUCK` / `PARTIAL` / `RUNNING`) — but it covers only the X₂₅–X₂₉
band worked in v13. `CHECKLIST.md:353-387` is a second partial table covering
X1–X4 and X6. Neither reaches X₃₀–X₃₈, which live only in prose inside
`CEQ_V15_CONTRACT.md`, its three deltas, and `CEQ_V16_CONTRACT.md`.

**Consequence for L-FIND, and it is the reason this matters rather than a
bookkeeping note.** Clause (b) admits *"an X-item with a run instance"* as
ledger evidence. With no register, there is no enumeration to check a claimed
X-item against, and no status column that would say whether its run instance
still stands. Two X-items in the only table that exists are already marked
**STRUCK** (`V13_DAG_TASKLIST.md:15`, X₂₆ CUSUM/ARL; and X₂₇b inside the A6 row),
so "an X-item with a run instance" is not automatically live evidence. **Until a
register exists, clause (b) should be satisfied by a measured cell or a Lean
theorem** — the two limbs that resolve to a path — and an X-item limb should be
required to cite the run instance's own file rather than the X-number.

---

## 4. KILLS, EACH WITH ITS REPLACEMENT ROUTE

Under `STATE.md:12` — *"Every kill ships a replacement route — reroute / reprice
/ retire."*

**K1 — Nash / QRE / ESS / MFG (`ceq/nash.py`) is not a wing.**
Recorded dead at `ARSENAL.md:298`: *"the equilibrium they select was deleted
(W7: 1/5 seeds; DEQ multiplicity moot)."*
**RETIRE.** Unreachable, because the object it selected among — multiple
equilibria — does not exist for a nilpotent operator whose settle is a direct
solve. What to want instead: the *ordinal* property `nash.py:9-11` was actually
after (composition of sign flips is ordinal, not metric) is already carried by
W1's path product, whose zeros are exact rather than learned
(`lean/CEQ/V16Domain.lean`, `no_prefix_scan_represents_a_zero_gate`). Want the
exactness, not the game.

**K2 — Multi-zoom (`ceq/multizoom.py`, `ceq/mz_kernel.py`) is not a wing, and is
live-tree orphaned.**
`ARSENAL.md:293`: *"Multigrid, spectral decimation, mixed-curvature manifolds —
R5 deleted; locality beat multiscale."* Independently, the import census `[RUN]`
finds `ceq/mz_kernel.py` imported by **nothing in the live tree** and
`ceq/multizoom.py` imported only by `mz_kernel`; the only importers are under
`attic/tests/chase/`.
**RETIRE, and it is purge inventory.** Both modules are candidates for it.43's
tarball-then-delete with their attic tests. What to want instead is already
measured and kept: the locality result that beat them.

**K3 — ΔFloor-by-eviction as a SELECTOR.**
`ARSENAL.md:303-304`: *"Round 12, calibrated instrument, both pre-registered
kills fired, CI at s=1024 entirely below chance."*
**REROUTE — same goal, different object.** The goal was a principled eviction
ranking. The ranking is dead; the **certificate** is not. `ARSENAL.md:207`
already states it: *"Morse upgrades the CERTIFICATE, not the ranker."* Route it
into NM-3, where the Lean theorem `free_face_floor_unchanged`
(`lean/CEQ/Refcount.lean:160`) is the surviving asset.

**K4 — Refcount-priced schedule as a selector.**
`ARSENAL.md:305-306`: *"killed by its own Lean proof: refcount is constant within
a sequence, so the score carries zero bits."*
**REROUTE** to the same place as K3, and note the mechanism is worth keeping in
`MISTAKES.md`: a quantity proved constant within the unit of analysis is a
vacuous selector by theorem, which is the cheapest possible kill and should be
run *before* any selector is built, not after.

**K5 — "GPU-seconds-to-floor" is not measurable for any wing today.**
The arena's criterion (3) (`CEQ_V20_R15_CONTRACT.md:127`) ranks by lowest
GPU-seconds-to-floor. No arm crosses the floor: `V15_R1.md:250` `crosses? NO`,
`V15_R1.md:251` `NO`, and `V17_R4_RETAKE.md:194` records zero sign-flips in 16
on the certified device.
**REPRICE, and here is the arithmetic.** The measurable quantity is
**GPU-seconds per cell at fixed budget**, which exists for all three wings:
`arm_smprime` 15.970 s, `arm_pl` 1.614 s, `softmax` 1.497 s per 150 steps
(`V17_R4_RETAKE_PRICE.md:194-196`), i.e. the shipped arm costs **9.90×** softmax
and **10.68×** the incumbent-matched `arm_pl` on the same six-cell run. Read
against the scoreboard line *"the winner's cost ≤1/10 incumbent"*
(`:270`), **W1 is currently at 9.90× the incumbent, not 0.1× — a 99× gap to that
+4.** Criterion (3) should be restated as *seconds-per-cell at the matched
budget, reported beside distance-to-floor*, until some arm crosses; the moment
one does, the original statistic becomes available and nothing is lost.

**K6 — The Mathematics Annex's `[RUN]` instances are not yet clause-(b)
evidence.**
Four distinctive annex tokens have **no producer in the tree**: `d=65` (M9-F1
Cantelli/Boole cutoff), `d=20 at the same` (M9-F1′ Azuma), `1.3e-3` (M2
gate-landscape θ), `0.492 vs KL 0.519` (M13 W1-vs-KL). Searched across every
`.py`/`.json`/`.jsonl`/`.txt`/`.log` in the tree, excluding `.git`, `kaggle`,
`attic`, `__pycache__`, this test file, and `house-events*.jsonl`.
**This is NOT a fabrication finding.** The contract is dated 2026-09-02 and is
filed verbatim from the author's own message; those instances were plainly run
in his session, on his box. The finding is narrower and it is about *this*
round's evidence rules: under `MISTAKES.md:289` (P-1, *a number with no live
producer*) they are **not reproducible here**, so they cannot serve as clause-(b)
ledger evidence for any wing.
**REPRICE.** The route is cheap and it is Jupiter's it.2 work anyway: land one
script per annex item that re-emits its `[RUN]` figure into `results/`. Sixteen
items, each a few lines, and the annex then satisfies its own L-FLOOR
requirement instead of asserting it. `tests/saturn/test_v20_r15_wing_rubric.py::
test_every_annex_run_instance_has_a_producer_in_the_tree` is the standing bind
and stays RED until they land.

---

## 5. RED TEST EVIDENCE

**Path:** `tests/saturn/test_v20_r15_wing_rubric.py`
**Command:** `python -m pytest tests/saturn/test_v20_r15_wing_rubric.py -p no:randomly -q`

### 5.1 The wing-list bind — RED before this report existed

```
E   Failed: V20_R15_IT1_SATURN.md does not exist. The wing list has not been
    filed, so no wing in this round carries four citations and L-FIND is
    unsatisfied by default.

FAILED ...::test_every_wing_cites_all_four_clauses
FAILED ...::test_every_citation_resolves_to_a_line_that_carries_its_anchor[a]
FAILED ...::test_every_citation_resolves_to_a_line_that_carries_its_anchor[b]
FAILED ...::test_every_citation_resolves_to_a_line_that_carries_its_anchor[c]
FAILED ...::test_every_citation_resolves_to_a_line_that_carries_its_anchor[d]
FAILED ...::test_no_wing_cites_the_same_line_for_two_different_clauses
FAILED ...::test_each_near_miss_claiming_no_kill_really_has_none_in_the_record
7 failed, 5 passed in 0.59s
```

The 5 that passed on that run are the **calibration** cases, and they are the
reason the 7 REDs mean anything: a bad path raises, a line past EOF raises, an
anchor absent from its line raises, a true citation resolves, and the clause-(c)
absence search finds the `ΔFloor` kill that is known to be at `ARSENAL.md:303`.

### 5.2 The annex-producer bind — RED, and it stays RED

```
E   AssertionError: annex items marked [RUN] whose figure NO
    .py/.json/.jsonl/.txt/.log in this tree can emit:
    {'M9-F1  Cantelli/Boole cutoff': 'd=65',
     "M9-F1' Azuma cutoff": 'd=20 at the same',
     'M2     gate-landscape theta': '1.3e-3',
     'M13    W1 vs KL': '0.492 vs KL 0.519'}.
    Under MISTAKES.md:289 (P-1, a number with no live producer) these are not
    reproducible here, so they cannot yet serve as L-FIND clause (b) ledger
    evidence.

8 failed, 6 passed in 4.43s
```

### 5.3 Three defects in my own instrument, caught before shipping

Recorded because this round's whole discipline is that an instrument which has
never been observed failing is not an instrument.

1. **The producer search matched its own token literals and went GREEN by
   quoting itself.** First run of `test_every_annex_run_instance_has_a_producer_in_the_tree`
   PASSED; inspecting `producers_for` showed the only hit for all four tokens was
   `tests/saturn/test_v20_r15_wing_rubric.py`. That is `MISTAKES.md:72` (V-3, *the
   assertion is an algebraic identity of your own construction*). Fixed by
   excluding the test file, with the reason written into the docstring.
2. **The clause-(c) absence search scanned this report and refuted itself.**
   Once the near-miss table was filed, `test_each_near_miss_claiming_no_kill_
   really_has_none_in_the_record` went RED — because the table names `hybrid`
   and `hopcache` on the same lines as the word "kill", and the search scans
   every root `*.md`. Fixed by excluding the report: a document cannot be
   evidence for its own claim. Worth recording as a general shape — **an absence
   search over a corpus that the claim is then written into will always
   eventually find itself.**
3. **My first calibration control was itself unproduced.** The control was
   `0.371583` (`MISTAKES.md:2153`, the re-run zero-step gate-`R²`), and the
   calibration test FAILED — that number lives only in prose. A producer search
   calibrated on a number that has no producer is void. Replaced with
   `0.7071067811865476`, `floor_1` as journalled in `results/*.jsonl`, and the
   control now additionally asserts the hit is under `results/`.

All three were found by looking at what the green *was*, not by trusting that it
was green — which is the same move that caught the ParaFormer defect
(`tests/loop/test_arms_distinct.py:1-10`).

### 5.4 Final state with this report filed

```
1 failed, 13 passed in 4.36s
FAILED ...::test_every_annex_run_instance_has_a_producer_in_the_tree
```

The wing-list bind is **GREEN**: twelve citations, each resolving to a file, a
line inside it, and an anchor present on that line. The one remaining RED is
**K6**, and it is meant to stay red until the annex's `[RUN]` instances have
producers.

---

## 6. WHAT I COULD NOT VALIDATE

Every unresolved item, collected here and nowhere else.

1. **`L-FIND` and `wing` are contract vocabulary with no repo referent.**
   Neither string occurs anywhere in the tree outside `CEQ_V20_R15_CONTRACT.md`
   and this file. `[RUN]` grep over `*.md` and `*.py`. The mapping from "wing" to
   "a built arm" is **mine**, argued from clause (a) (*a distinct PRIMITIVE for
   the label class*) and from contract `:85-88`, which separates arms from
   skylines and multipliers. If the author means something broader by wing —
   e.g. an `ARSENAL.md` tier-1 item — then N = 3 is too small and NM-1/NM-2
   should be promoted the moment they are timed. **This is the single assumption
   the whole list rests on and it should be confirmed at it.4 before the freeze.**

2. **Clause (c) is read as *"the approach has accepted a kill that is on the
   record"*, not *"the approach has been killed"*.** Read the second way, the
   wing list would be empty and the near-miss table would be the whole report.
   The first reading is what makes the rubric constructive, and it is what
   `ARSENAL.md`'s `Kill:` lines are for, but the contract sentence
   (`:80`) admits both. `GUESS` — not asserted.

3. **W3's clause (c) is a claim-level retirement, not an arm-level kill.**
   `V15_LEDGER.md:786` retires ARM PL's *parity claim*, not ARM PL. It is the
   strongest clause-(c) citation ARM PL has: `V15_ARM_PL.md` contains **no**
   `Kill`/`must-fire`/`struck`/`dies`/`DEAD` line at all `[RUN]`. If clause (c)
   requires an arm-level kill, W3 drops to a near-miss.

4. **I did not verify that the three wings are behaviourally distinct from each
   other.** `tests/loop/test_arms_distinct.py` binds the seven *bench* arms, not
   `arm_pl` / `arm_phase` / `arm_smprime`. Their distinctness is argued from
   source (`cumsum` for PL/PHASE vs `cumprod` for SM′, `ceq/compat.py:393-394`)
   and from the mutilation table at `workdonenewseal.md:113-122`, **not measured
   as a fingerprint**. Cheapest closure: extend the existing
   `assert_arms_distinct` helper over the three. This is a real hole — the
   ParaFormer defect is exactly "two names, one code path", and these three arms
   have never been run through the bind that exists to catch it.

5. **`results/` was inventoried, not read.** 321 files, 50 `.jsonl`. I read the
   headers and first records of `v15_r1.jsonl`, `v17k_r4_retake.jsonl`,
   `arm_a.jsonl`, `arm_s.jsonl`, and the cost fields of `k_cert_local.json`
   (`elapsed_s = 523.9`, per-arm medians for `softmax`/`arm_smprime`). The other
   ~45 journals are unexamined; a clause-(b) cell better than the ones cited may
   be in them.

6. **`results/k_cert_local.json` carries a *better* clause-(d) source than the
   one I cited, and I did not use it.** It holds per-`n` median seconds-per-step
   for `softmax` and `arm_smprime` across `n ∈ {2048…32768}` — a cost *law*, not
   a point. I cited the R4 point measurement instead because it covers all three
   wings on one run at one shape, which is the comparable quantity. Mercury
   should prefer the law at it.5 pricing.

7. **The annex's other twelve `[RUN]` items were not checked for producers** —
   only the four whose tokens are distinctive enough that a substring search
   cannot collide with an unrelated float. Short decimals like `0.865` and
   `0.0375` occur in a dozen journals as substrings of longer numbers, so a
   search on them proves nothing in either direction. Those twelve are
   **unresolved, not cleared.**

8. **`X₃₈` I never resolved to a definition line.** It appears only in
   `CEQ_V16_CONTRACT.md:4`, `:109`, `:214`, `:275` and in `V16_PRICING.md:612` /
   `V16_DEVICE_CERT.md:1553`, always as a name in use. The nurse census returned
   `NOT FOUND` for a defining line, and I did not find one either.

---

**Distance to the north star.** Unmoved by this iteration. Discovery lists what
exists; it measures no capability. The three wings are three parametrizations of
one softmax-class head, and the north star's third clause — *capable on ground
softmax cannot occupy* — has no wing standing on it: `workdonenewseal.md:526`
still reads *"It claims no capability advantage over softmax. None has been
measured."*

**Scoreboard.** 0 of 48. The `+2` for *"wing list frozen, four citations each"*
is not claimable at it.1 — the freeze is it.4 (`CEQ_V20_R15_CONTRACT.md:90`),
and the list must survive MARS at it.3 first.
