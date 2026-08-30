# CONFIRMED FINDINGS — audit of `feat/ox-alpha` @ 74e5590, now merged to master

Every line below was verified by direct read this session. Class is `READ`
unless marked. **Do not re-research these — build on them.** If you find one
of them is wrong, say so loudly; a wrong "this exists" is worse than "unknown".

## A. Live defects in the tree right now

| # | Defect | Location | Evidence |
|---|---|---|---|
| A1 | **`impact_hetero` is a vacuous control.** It registers `make_impact_batch`, not `make_impact_hetero_batch`. Both keys return byte-identical 4-tuples. A heterogeneous-plant arm drawing the homogeneous corpus cannot differ from its baseline. | `scale/negation_scope.py:1081` | `make_impact_hetero_batch` exists at `scale/impact.py:601` (**line cite corrected from `:594`**), imported at `scale/negation_scope.py:59`, **never used** |
| A2 | **`run_bucket` counts the whole journal, not the requested units.** `len(done)` and `len(final)` include other tasks' records since `--task` landed. A run asking for 10 against a journal holding 25 printed `25/10 units already journalled` and counted from `[26/10]`; `acc["remaining"]` went negative. | `scale/bucket.py:149`, `:184-185` | RUN: confirmed present on master this session |
| A3 | **PARTLY RETRACTED — Mercury corrected this.** `E_T_STAR` does lack entries for `impact`, `impact_hetero`, `e4prime`. But `scale/e_ladder.py:143` is unguarded **and unreachable**: `RUNGS` is a module constant no CLI touches. **Only `scale/etask_k5e.py:117` reaches a real `KeyError`.** Mercury deliberately left `e_ladder` alone — guarding it would convert an impossible crash into a reachable `None` inside the pre-registered ladder. **FIXED** via `negation_scope.e_t_star(task, s)` returning `None`, `etask_k5e` printing `-`. No dial was invented: IMPACT's diameter is a property of the draw, and a wrong dial reaches the pre-registered table. | `scale/negation_scope.py:1017-1025` | RUN, `tests/mercury/test_r9_mechanical_fixes.py` |
| A4 | **`scale/impact.py` is registered despite its own docstring forbidding it.** `:23` says *"Register as IMPACT in M3_TASKS only past all four gates + planted controls."* It is registered at `:1077` and `:1081`. Its four gates have never executed. | `scale/impact.py:23` (**line cite corrected from `:26`**) | Zero test files import `scale.impact`; zero `results/impact*` artifacts |
| A5 | **`impact_decoder_gate` ships a placeholder.** `planted_sd` is computed and never returned; the comment says `placeholder; actual planted label sd not needed`. Lines 805-810 contain open questions in comment form. | `scale/impact.py:816` | |
| A6 | **Unreachable code after `return`** in `make_impact_batch`, and a no-op guard `if d < 1 or d >= s: pass` where every sibling builder raises. **FIXED** — guard now raises `ValueError`, unreachable block deleted. | `scale/impact.py:645-648` (**line cite corrected by Mercury from `:641-643`**) | RUN |
| A7 | **`e2_consequence` bar is broken at `steps=150`** (`2.446646`), calibrated at 600. Cause is diagnosed and unfixed: `calibrate_bar` trains its control on **raw** `y` while `run_arm` trains arms on **standardised** `y`. | `STATE.md:73-76` | Zero `e2_consequence` rows in any `results/*.jsonl` — it has **never been trained** |

## B. False and stale claims in the documentation

| # | Claim | Where | Truth |
|---|---|---|---|
| B1 | *"the one live direction that has not yet been measured here"* — about iterating the representation | `LOOP_PROMPT.md:53-55` | **Stale.** `scale/foreman_looped.py` is a complete 305-line instrument with a bind battery (`bind_loops_one_is_shipped:128`, `operator_is_nilpotent:159`) and an in-file pre-registration (`:53-70`). `results/foreman_looped.jsonl` holds **2 of 6** required cells |
| B2 | *"U2 mujoco contact-graph DSU n≥1024 and U3 Tonnetz lattice graphs — scaffolding code exists as stubs"* | `BOARD.md:288` | **False.** `git grep -i tonnetz` hits `BOARD.md` (2) and `DONE.md` (1) only. **Zero `.py` anywhere.** No U2 contact-graph builder exists |
| B3 | `settled_plus` result never recorded | `PIVOT_EXCLUSION_FALSIFIER.md:156` says *"`settled_plus` was still running when this was written"* | It finished. `results/etask_k5e_plus.txt` records `settled_plus 0.956787` — **also worse**. Document never updated |
| B4 | **NARROWED. The original entry was wrong and is withdrawn.** It claimed `scale/arm_s.py` asserts a `gram_audit` symbol exists and called it unfixed doc rot. It does not, and it is not: the docstring **carries its own retraction in place**, verified by direct read at `74e5590` — *"no function or test named `gram_audit` exists in this repository, and an earlier revision of this sentence claimed one did (doc rot, recorded as STATE.md item 39)"*. The surrounding paragraph then names the nearest real artefacts and says what they do not cover. **What survives, and all that survives:** *"No measurement of pivot-coordinate underflow is on record"* — `a_p[0]` can reach exactly `0` in float32 and **nothing measures that**. The nearest artefacts are analogies, not a census of `a_p[0]`: `positivity_audit` (`scale/foreman_hilbert.py:305-334`) counts exact zeros-on-support over the **whole** float32 softmax matrix — `16861/523776` entries at `s=1024` (`CHECKLIST.md:775`) — and the shipped settle sidesteps the question arithmetically by running in the log domain, which never forms `exp()`ed probabilities. Owed only if the probability-domain path (`pivot_context`) is ever shipped. | `scale/arm_s.py:50-65` | Corrected by Saturn (it1) and Neptune (it1), who further established the file was already repaired by commit `54148e6` and that `STATE.md:230` was the stale sentence, not `arm_s.py` |
| B5 | IMPACT gate numbers (`truncation k2 0.3243`, `decoder local 1.0039 vs planted 0.06738`, `sign gate degrade 0.8793 CI [0.8300,0.9145]`) | commit `74e5590` body and `DONE.md` | **HALF RETRACTED — this entry was wrong and Mars corrected it.** The "no results file, no test, no CLI" half **stands**. The "nothing reproduces them" half is **false**: Mars ran all four gates for the first time and they reproduce the commit body exactly — `1.0039067318` / `0.0673856682` / `0.3243630511` / `0.8793857278` CI `[0.8300754114, 0.9145472963]`. RUN, `tests/mars/test_mars_r9_iteration1.py`. The numbers were never fabricated; they were merely never journalled |
| B6 | `§U`, `X₂₁`, `X₂₂`, "contract v10.1", "amendment v10.4" | `BOARD.md:282,284,307`, `DONE.md` | Journal prose only. `LOOP_PROMPT.md` has 17 headings, **none is §U**. No R10 plan or criteria exists |
| B7 | Widespread line-reference drift | `scale/m3_quintuple.py:335` cites `arm_s.pivots_of (line 97)` → really `106`; `scale/m3_flops.py:44-84` cites nine stale lines; `M3_QUINTUPLE_PREREGISTERED_READING.md:38` cites `:236` → really `256-257` | Do not trust in-tree docstring line numbers |
| B8 | `scale/capability_table.py` says *"`m3_quintuple.py` … has no `--task` flag"* | `TASK_SOURCE` string | **False** — `--task` was added at `scale/m3_quintuple.py:617`. Also `render()` still emits *"This package carries NO trained weights"* and `write_artifact` (`:443`) would **clobber the weight-manifest sync** from commit `0162bdd` on the next table cut |

## C. The structural finding — why the deciding number reads zero

| # | Fact | Evidence |
|---|---|---|
| C1 | **Every label in the repo is a scalar point prediction at position `s-1`.** | `scale/m3_quintuple.py:311` — `self.readout(h).squeeze(-1)[:, s - 1]` |
| C2 | **Softmax is provably Bayes-optimal on exactly that shape.** One softmax layer attains Bayes risk where linear attention provably cannot. | `arXiv:2410.01537`, ICLR 2025; conceded at `LOOP_PROMPT.md:34-38` |
| C3 | **The repo already knows this and never acted on it.** *"every task here asks for a point prediction — which is exactly the single-location problem where one softmax layer is provably Bayes-optimal"*; the novelty claim is *"UNTESTED because no vector-valued label exists here."* | `STATE.md` items 27-28 |
| C4 | **`impact_truncation` builds the entire propagation vector and throws it away.** `acc` accumulates `Σ_h (ρA)^h Bn` over the whole graph, then returns `y[b] = acc[query]` — one coordinate. | `scale/impact.py:737-756` |
| C5 | **The `e3_t*` oracle is the arm's own resolvent**, so `settled − softmax` is VOID on that ladder; only `settled − twin` is creditable, and that reads `−0.002959`, CI `[−0.042903, +0.031557]`. | `results/e_ladder_reading.txt`; `LOOP_PROMPT.md §1.7d` |

**Consequence:** dropping the `[:, s-1]` index yields an `[n, s]` vector label
for **zero new parameters** — the same `readout` applied at every position.
That leaves the regime where softmax is proven optimal. It must go in a
**separate lane** (own journal, own weights dir) because changing `forward`
voids the `PUBLISHED_SOFTMAX_8192` reproduction gate at
`scale/m3_quintuple.py:696-708`.

## D. Unfinished instruments — cheapest science available

| # | Instrument | State |
|---|---|---|
| D1 | `scale/foreman_looped.py` | **2 of 6 cells.** Have: `e3_t2/steps=150/seed0` for `softmax` (`0.9704371404137836`) and `looped3` (`1.0188051091704295`). Missing: `e3_t8/looped3@150`, and all three cells at `steps=600`. `falsifier()` returns `complete: False` and refuses a verdict on a partial table |
| D2 | CUDA lane | **22 of 60 cells.** `softmax` t1+t32 complete; `settled` t1 complete, t32 only seeds 0-1; `twin` t1 only. Whole `t2` and `t8` columns absent. Stopped mid-`e3_t32` after `settled` seed 1 |
| D3 | CPU ladder reading | **COMPLETE** — `results/e_ladder_reading.txt` ends `ladder complete: True`, row G fires. `STATE.md:21` "RUN IN FLIGHT" is stale for the CPU lane |
| D4 | `ceq/nash.py` | Full QRE ordinal-stance module with the composition argument. Measured tier ladder attention `5.8198` / APPNP `4.2107` / signed `2.6151` OOD NRMSE — **every arm above 1.0**, worse than predicting the mean, on held-out composition of two sign flips. Never revisited |
| D5 | `ceq/rips.py` | The author's own port of `island_benchmark_test.cc` from **google-deepmind/mujoco PR #3396**, seeds `0x3396xxxx`. Six beds spanning the S² connectivity transition, component count 178 → 1. Plus two `REROUTED_CASES`. This is the graph corpus |

## E. Constraint surface for a new arm

- `CELLS = ("softmax", "glance", "settled", "twin", "argmax")` at `scale/m3_quintuple.py:86`. **`tests/chase/test_pivot_exclusion_lift.py:139` asserts this exact 5-tuple** — adding a name to `CELLS` fails that test. Precedent is `PLUS_CELLS` (`:93`): a **new tuple beside** `CELLS`.
- Dispatch is a three-branch `if/elif` in `QuintArm._alpha` (`:282-300`) returning `[n, d]`. `softmax`/`glance` never reach it — they branch in `forward` (`:302-311`).
- Arm name is enumerated in **14 places** including `scale/m3_flops.py:103-131` (raises `ValueError` on an unregistered cell), `scale/e_ladder.py:55` `HOP_BUDGET`, `scale/capability_table.py:110-118`, `tests/gpu/test_cuda_parity.py:44-45`.
- Parameters are built **only** at `scale/m3_capability.py:108-113`. `QuintArm.__init__` calls `super().__init__("softmax", s)` and adds nothing. 4769 = 256+256+2048+128+2048+16+16+1.
- A cell name containing `_sd`, `_task`, `_k` or `_b` **breaks the journal key parsers** (`capability_table.read_journal`, `eprocess._parse_key`). That suffix collision already took down 14 of 16 tests once — see `scale/m3_quintuple.py:561-577`.
- Pivot selection is `key.norm(dim=-1)` top-k, `scale/pivot_probe.py:80-91`. **Non-differentiable — zero gradient flows through selection.** Nothing teaches the model which pivots to pick.

## F. Gates — enforced by hand-written tests only, never at registration

`M3_TASKS` is a plain dict literal; **nothing validates an entry on import.**
The admission bundle precedent is `tests/cameron/test_e4prime_registration.py`
(5 tests, 295 lines). Bars: `PASS_BAR = 0.5`, `FAIL_BAR = 0.9`
(`scale/rips_gate.py:57-61`). Decoder gate fits on a **held-out half** with
ridge `1e-6` (`:163`).

Truncation law: `ceiling(t_star, hops) = sqrt((t*-hops)/t*)`,
`scale/e_ladder.py:61`. Checked against drawn batches, closed form high by at
most `+0.012088`. For a **contraction** label (which `(I−ρA)⁻¹` is) the
nilpotent form does not apply — the repo's own precedent is the relative form
`got <= zeroth * L**k + 1e-9` (`tests/cameron/test_m3_etasks.py:208`), because
NRMSE normalises by `std(y)` and not by initial error.

Flipper dependence is a **two-sided band** (tol 0.05), not a threshold — it is
the *wrong-task* check, not the anti-vacuity check. `e4prime` and `impact` both
declare exactly `0.0`. **A task declaring `0.0` must ship a do()-bit movement
test** or the zero is indistinguishable from the vacuous kind that already got
`unshocked_equilibrium` rejected (`tests/cameron/test_m3_etasks.py:356`).

## G. Statistics

Paired percentile bootstrap, `n_boot=10000`, seed 0, strict at zero
(`scale/m3_synthetic_settled.py:172-182`). Ville e-process:
`ALPHA_FAMILY = 0.05`, `N_DIRECTIONS = 2`, `THRESHOLD = 40.0`
(`scale/eprocess.py:209-219`). **Ceiling arithmetic must be printed before the
first number** — the old `t=5` unit had ceiling `3.80169140625 < 40.0` and
could not cross whatever the data said.

Pre-registration structure: `E_LADDER_PREREGISTERED_READING.md`, §6 is the
outcome table, rows A-H, three columns `# | what the data show | what it
licenses`. Every row has a matching branch in `scale/e_ladder.py:217 verdict()`.
Rows must be **exhaustive before the data** — row H was added because A-G all
conditioned on settled winning somewhere and a strictly-losing ladder fell
through the whole table.
