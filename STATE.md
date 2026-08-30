# STATE — CEQ v10, ROUND 8, THE EQUILIBRIUM-CORPUS ROUND

| field | value |
|---|---|
| contract | `LOOP_PROMPT.md` (v10). Round 7 archived `LOOP_PROMPT_ROUND7_ARCHIVE.md` |
| promise | **`HILBERT`** — output only when completely and unequivocally true |
| iteration | **PHASE R8 ITERATION 0 — READING STARTED (RULE 2 clock: complete by iter 8)** — prev 10 complete, winding down archived |
| **scoreboard** | **25** — 23 carried + **X₁₈ +2 EARNED** — X₁₇ scaffold advanced this iter (covariates + gates re-measured, not yet +3) |
| **the round exists to move ONE number** | `done7.md` scores the product at **37 %**: engineering **≈ 80 %**, the scientific claim **≈ 5 %**. **The 5 % is because the deciding measurement has been taken ZERO times.** PHASE R8 exists to take it on the discrete-Dirichlet E4′ corpus (u_I=(I−P_II)⁻¹P_IB g). |
| **RULE 2** | **THE READING COMPLETES BY ITERATION 8.** Not the corpus, not the process — the reading. **Clock STARTS NOW (iter 0).** |
| register | **caveman, all agents.** Artifacts stay normal English |
| **RULE 5** | Every kill ships a replacement route — reroute / reprice / retire |
| House | **OFF.** House mode is stopped by instruction. No further parallel dispatches; agents in flight finish or do not, and nothing new is spawned. |

## THE v10 BOARD

| item | pts | owner | state |
|---|---|---|---|
| **X₁₇** e3-harmonic ladder on E4′ Rips, oracle = absorbing-chain solve, `λ₂` in `0.90–0.95`, both gates + planted controls seen firing | +3 | Cameron | **ITER 0 SCAFFOLD ADVANCED: discrete-Dirichlet E4′ (u_I=(I−P_II)⁻¹P_IB g) gates re-measured on LargestJoin_S2Rips_1024; covariate table `results/covariate_table_phaseR8_iter0.json` with per-instance λ₂ (exact eig, 60×60 Q) and δ̂ (Gromov four-point) printed; truncation gate PASS (k1 1.4128>1.0, k32 0.0000 monotone), decoder gate PASS at target n (1.0001/0.9951≥0.9 FAIL, planted sum 2e-08 <1e-6, median 0.5530<0.70 gap 0.4471>0.30, both classes nonempty), δ̂ controls FIRING (tree 0.000000, cycle 8.000000 n=32) — K-R8c not void. λ₂ natural 0.9984623637 (t_rel 650.35) misses band, tuned 0.9250000000 (α=0.9264245040) hits band 13.33; K-R8a conditional fire (re-draw not re-label) flagged with reroute route. E-family still 6 tasks; TRAINED capability column not yet claimed. BOARD.md §6 with file:line.** |
| **X₁₈** per-draw Ville e-process, calibrated both directions, ceiling arithmetic printed pre-run | +2 | Chase | **BUILT, 6/6 GREEN.** `scale/eprocess_perdraw.py`; `tests/chase/test_eprocess_perdraw.py`. Ceiling printed pre-run: old unit `t=5` ceiling `3.80169140625 < 40.0` **cannot cross**; new unit `t=2048` ceiling `10**359.6349`, `t=10240` ceiling `10**1802.1745`, **both can cross**. `eprocess.max_attainable` OVERFLOWS past `t=1748`, so it is read in log space and left unmodified. Planted 0.20 effect crosses **20/20**; null crosses at or below `ALPHA_FAMILY 0.05` over 400 reps; PASS half carries its own non-degeneracy check. **Estimand caveat in the file:** per-draw conditions on the trained weights. |
| **THE READING** — settled vs twin across `t* ∈ {1,2,8,32}`, trained weights, 5 seeds, fidelity column **with Identity ablation** | **+15** / **+6** | Chase | **INSTRUMENT BUILT, RUN IN FLIGHT.** `--task` ported and per-cell weights saved (`tests/chase/test_m3_ladder_task.py` **10/10**). Pre-registration `E_LADDER_PREREGISTERED_READING.md` written before the first `e3` number, outcome rows **A–H**, fallback included. Reader `scale/e_ladder.py` prints the curve and **refuses rows A/C/F on a partial ladder**. Measured unit cost on the contended box **375.4 s / 379.3 s** (settled, `n_train=2048`), so one rung is ~77 min and the four-rung ladder ~5 h. Rungs run **endpoints first**: `e3_t1 → e3_t32 → e3_t8 → e3_t2`. **RULE 2: by iteration 8 — PHASE R8 clock now iter 0→8.** |
| **X₁₉** E4′ registered in `M3_TASKS` | +1 | Cameron | **REGISTERED (scale/negation_scope.py:1028, tests/cameron/test_e4prime_registration.py 6/6) but X₁₇ +3 not yet earned — training capability column pending. U1 RAG double-registration scaffold present (rag_multihop_t* bitwise deterministic, measurement-only until R10).** |
| **X₂₀** Kaggle segment + HF upload **with real weights** (author's say-so gate stands) | +4 | Chase | not started |

## WHY THE ORACLE CHANGED, AND IT IS THE WHOLE ROUND

**The absorbing-chain solve is a genuine equilibrium.** `N = (I − Q)^{-1}`, `B = N R`,
the fixed point of `z ← Q z + R`. **It is not a closed-form function of any bounded
neighbourhood**, so an arm that iterates has something to compute.

**Every task read before this round had a closed-form oracle** —
`x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]` and `x[:, :, CH_FLIP].sum(dim=1) ** 2` —
**which is why the deciding contrast came in at `−0.002959` with an interval covering
zero. That contrast was correct and uninformative.**

**And `λ₂` makes the curve a PREDICTION rather than a description.**
`t_rel = 1/(1−λ₂)`; engineering `λ₂` into `0.90–0.95` puts `t_rel` in `10–20`, so the
ladder `{1,2,8,32}` straddles the transition. Cheeger bounds `λ₂` by conductance from
both sides, so the target is reachable **by construction rather than by search**.

## OPEN, CARRIED

1. **`e3_t*` in the CURRENT registry inherits `e1_anchor`'s rig** — same
   `equilibrium_oracle`, whose label is the signed path sum the resolvent computes.
   **Only `settled − twin` is creditable on those rungs.** Foreman owns stating
   `oracle ≠ resolvent` for the NEW absorbing-chain oracle, in Lean.
2. **E4 as specified is STRUCK**; **E4′ (join the two largest) passes both gates at
   `n = 1024`** — leak `0.1565 → 0.9951` — **and is not yet registered.** Not
   admissible at `n = 64`, which still leaks at `0.0055`.
3. **Gate (a) passing does not mean a task is not locally decidable.** Reachability
   needs the diameter; *deciding* a label may not. **Any substrate needs the decoder
   gate, not just the truncation gate.**
4. **Fourteen vacuous controls struck across five authors**, the fourteenth a PASS
   case whose label was constant. **Every must-fire's PASS half now carries its own
   non-degeneracy check.**
5. **The one-hot control is worse than softmax** (`−0.118456`, CI
   `[−0.134115, −0.102786]`, 0/5) and **fails its own bar at `1.010779`**. The gain is
   the **mixture**; a lookup is worse than nothing.
6. **Settling adds variance and no mean** — `sd 0.064106` vs the twin's `0.016547`,
   **3.874×**, same batches, same init, same params.
7. **HF package built, NOT uploaded, and ships NO weights** — a random-init
   `model.safetensors` measures `1,901,686,656` bytes and was correctly refused.
   3 trained checkpoints exist on disk but are `pivot_unsigned`, not settled/twin.

8. **THE AUTHOR'S RULING, and it frames the round:** *"if it is beaten by softmax the
   row H is new row A"* — **an honest negative is the headline, not a footnote.**
   Recorded before the ladder completed.
9. **THE HOP WALL is the most actionable defect open.** At `t* = 8` the best arm reads
   `1.112208` against a **2-hop ceiling of `0.866025`** — **`+0.246` short of what its
   own budget allows.** The binding constraint at that rung is neither the settling nor
   the task. **Measurable fork:** hand an arm the `k=2` truncation as a feature — gap
   closes ⇒ the **readout** binds; gap holds ⇒ the **budget** binds.
10. **`e2_consequence` has never been trained, and it is the one rung the theory
   actually predicts on** (`t* = 31` ≫ hop budget 2). **And its bar is BROKEN at
   `steps=150`** (`2.446646`), CALIBRATED at 600 — because `calibrate_bar` trains its
   control on RAW `y` while `run_arm` trains arms on STANDARDISED `y`.
11. **Rank is RETIRED as a difficulty column, with a proof.** `z' = a·z + b` is a
   2-dimensional linear recursion, so ring-automaton rank is **exactly 2 across the
   whole ladder while `t*` runs 1 → 63.** **Constant rank, spanning difficulty.**
   Replacement metadata: `t*`.

12. **TWO HYPOTHESES FROM `NeMo-Relay#481`, unbound, needing a fellow and a RED test.**
   (a) **The ultrametric order-independence argument** — `d(x,y) = 2^-lcp(x,y)`, under
   which every point of a closed ball is a centre, so a ball is fixed by its members
   rather than by traversal order. **That property is FALSE in Euclidean geometry**, so
   a flat metric cannot prove what it proves. It instantiates RULE 9, and it was
   written upstream before RULE 9 was written here. Nearest live target: the ragged
   pivot-set tie-break in `arm_s.pivots_of`, which currently needs a declared deviation
   because top-k drops index 0 and the batched path backfills it.
   (b) **`A = Σ c_i(c_i−1) / (N(N−1))`**, a **graded** agreement rate that is `1`
   exactly when an analysis is seed-independent. **This project checks determinism by
   bit-identity, which is binary**, and has already hit a case it cannot describe — a
   replay where every `eval_nrmse` matched bitwise and only a wall-clock field drifted.

13. **THE SHARPEST OPEN FINDING: the arms fail where their architecture is provably
   sufficient.** At `t* = 1`, `ceiling(1, 2 hops) = 0.000000` — **measured**, not
   algebra — yet settled reads `0.994399` and twin `0.976975`. **The whole shortfall is
   `f` and the hop budget explains none of it.** Whatever binds at that rung is not
   hops, not the task, and not the settling.
14. **The ladder is PRE-ASYMPTOTIC at every rung** (`t* ∈ {1,2,8,32}` against mode
   times `20.5615` and `169.3116`). **`λ₂^t*` is the wrong predictor AT THE RUNGS even
   though `λ₂` is the right asymptotic rate.** The closed form
   `q_t − q = (u ⊙ Q^t s − s ⊙ Q^t u)/(s ⊙ s_t)` is the right predictor and costs one
   solve. **Reading the curve with `λ₂^t*` makes it look wrong for arithmetic reasons,
   not architectural ones.**
15. **`f → 1` is DERIVED; the approach to 1 is NOT.** The excess is set by both
   spectrum and geometry, and geometry can flip its sign — `f(160) = 0.99034549` at one
   placement, **below one**. Only the limit is proved.
16. **Cheeger relocates rather than evades** — `α`-scaling moves the reach ceiling from
   3 hops to about 13. `α^3 = 0.7951152826` of the walk's weight is inside a radius-3
   ball, and `α^29 = 0.1090161839` survives past the diameter. **A ceiling at 13 is
   still a ceiling.**
17. **`results/m3_capability.txt:1212` is a STALE TRAP.** It records
   `e3_t8 k=2 NRMSE=0.815162`, predating `b[:, s-1] = 0.0` in `make_equilibrium_batch`,
   and describes a corpus that no longer exists. **Reading `f` off that log gives the
   wrong answer.**

18. **THE COMPARISON IS CONFOUNDED BY CONSTRUCTION — this supersedes every reading of
   `settled` vs `twin` vs `softmax` on the E-tasks.** `scale/arm_s.py:107` excludes
   `s-1` then drops `0`; `scale/m3_quintuple.py:129` excludes `(0, s-1)`. Max pivot is
   `s-2`, and `tril(-1)` makes that row read `j ≤ s-3`. **So no pivot row can put
   `v[s-2]` on the value path, and the `t*=1` label is `a[s-1]·b[s-2]`.** Measured:
   perturbing `s-2` moves softmax by **`101.6983`** and the pivot `av` by **`3.263746`**.
   **The numbers are not evidence about settling.**
19. **The root cause is a docstring rationale, not a bug.** *"A pivot reading of the row
   being settled is not an independent reading of it"* — a defensible independence
   argument that became a capability ceiling nobody recognised. **Lifting it is MOVE 1.**
20. **The settled arm iterates a reparameterisation inside the twin's own family**
   (`alpha` on an 8-simplex over `{alpha @ av}`). **It can reallocate, not add** —
   structural explanation for `0.002190` with `3.874×` the variance.
21. **The hop wall is DEPTH.** `arXiv:2402.09268` Thm 4.2: `hop_k` needs
   `L = ⌊log₂ k⌋ + 2`. **Our arms are depth 1; `t*=8` needs ≈5.** At `t*=8` the arm
   trains to `0.860972`, **on its 2-hop ceiling `0.866025`**, and evaluates `1.112208` —
   **memorising noise.**
22. **Softmax is Bayes-optimal on the `t*=1` shape** (`arXiv:2410.01537`) **and its one
   step is already a converged Hopfield update** (`arXiv:2008.02217`, *"converges with
   one update"*). **Iterating the normalisation is known to buy little** (Sinkformer,
   `arXiv:2110.11773`); **DEQ reaches parity, not superiority** (`arXiv:1909.01377`);
   **when iteration pays, the iterated object is the REPRESENTATION**
   (`arXiv:2311.12424`).

23. **THE DELTA SURVIVES, and is now precisely bounded.** `arXiv:2607.21607` is real,
   its formula and all three `R²` are exactly as attributed, and the provenance flag was
   a **false alarm** (block-wide over five neighbours). **But it is a GCN — `Transformer`
   0, `GAT` 0, `GIN` 0, `GraphSAGE` 0, and `attention` never an architecture under
   test — its headline is on engineered graphs (surveyed fall to `R² 0.440`), and its
   label is trained accuracy, with `equilibrium` 0 and `fixed point` 0.** Attention, and
   the equilibrium as the label, are both still unoccupied.
24. **Its own Proposition 1 (iii): at fixed depth, SFC and `γ` induce identical
   rankings** — SFC is a monotone transformation of `λ₂`. **So `0.910 / 0.881 / 0.863`
   are `λ₂`'s numbers relabelled.** The paper's real gain is pooled-depth `0.884`
   against `0.858` — **2.54 %**. Anyone citing that paper as occupying a λ₂
   dose-response must cite `0.884 vs 0.858`, not `0.910`.
25. **Effective resistance there is a BEATEN BASELINE** (`R̄ = tr(L⁺)/n`, a global mean
   over a pseudoinverse trace), **not the two-boundary Dirichlet problem this oracle
   solves.** The `arXiv:2206.11941` impossibility concerns a different object again.

26. **THE ROUND IS WINDING DOWN.** House mode **off**; remaining effort is
   **documentation and mathematics**. **`MATHEMATICS.md` is the theory of record**, with
   an evidence class (`RUN` / `READ` / `CITED` / `DERIVED`) on every load-bearing claim
   and limits collected once at the end.
27. **THE THESIS, and the corpus has been testing its opposite.** The module should
   predict the **shape** the sequence settles into, not the next symbol. But
   `equilibrium_oracle` (`scale/negation_scope.py:274`) returns **`z*_{s-1}`, one
   coordinate** of the fixed point, and `nrmse` is a squared error on that scalar. **So
   every task here asks for a point prediction — which is exactly the single-location
   problem where one softmax layer is provably Bayes-optimal.** The corpus was built as
   the one task softmax cannot lose.
28. **THE NOVELTY CLAIM, narrow and UNTESTED:** one softmax step computes a convex
   mixture **per query row, independently of the other rows**, so nothing in it lets the
   value at position `i` constrain the value at position `j`. **A jointly determined
   configuration is not what that computes.** Not a claim about single coordinates (a
   theorem says softmax is optimal there); not a claim about depth (separately governed,
   and stacked layers get it too). **Untested because no vector-valued label exists
   here.**
29. **Two target settings, sharing the structure:** embodied control, where joint angles,
   contacts and forces are **coupled by constraints** and determined simultaneously; and
   retrieval, where a false passage is an **intervention on the context** whose
   consequence **propagates** through claims conditioned on it, producing answers locally
   plausible everywhere and globally wrong.
30. **The `e3_t2` looped pre-registration is INCOMPLETE and declared so**, not redefined
   to what was affordable. The 600-step half is unrun and `150` is measured to
   undertrain. **A near-tie at one seed is UNDECIDED** — there is an interval on each arm
   and none on the difference.

31. **MOVE 1 IS DEAD, PRE-REGISTERED AND CHEAP.** Lifting the pivot exclusion made the
   arm **worse**: `twin_plus 0.938728` against a threshold of `0.871391`, gap closed
   **`−0.1509`**, `0.015610` worse than `twin`. **So the exclusion really does hide the
   answer token AND removing it does not help — whatever costs the pivot cells
   `0.103453` at `t*=1` is not pivot access.**
32. **THE FIX RELAYED FROM THIS SEAT WAS INERT AND WAS CAUGHT BEFORE SHIPPING.**
   `exclude=(0,)` only sets `-inf` in a top-k over key-norm, so `s-1` became eligible and
   was selected in **0 of 32** examples; the permitted set was `torch.equal` to the
   shipped one. **It would have been a vacuous control shipped as the repair for a
   vacuous-control problem.** The working form reserves the slot unconditionally, and a
   test is kept so the inert version cannot be re-proposed.
33. **THE `e3` LADDER MAY HAVE NO USABLE RUNG.** `t* = 1` is where softmax is provably
   Bayes-optimal (`arXiv:2410.01537`); `t* ≥ 8` needs depth ≈5 against depth-1 arms
   (`arXiv:2402.09268` Thm 4.2). **Creditable and informative may not overlap on this
   family.** The reading's home is the absorbing-chain corpus.
34. **The pilot spread was `2.18×` optimistic** — realised `sd 0.109199` against a piloted
   `0.050146`, so the true five-seed half-width is `≈0.0957`, not `0.043955`. **Every
   sizing decision made on the pilot figure is correspondingly under-powered.**

35. **`looped3` LOSES DECISIVELY AT 150 STEPS AND IT IS NOT A DEPTH VERDICT.**
   `softmax 0.9704371404137836` CI `[0.9536301088926267, 0.9877856292539604]` against
   `looped3 1.0188051091704295` CI `[1.0051942033844927, 1.0334716650310674]` —
   **disjoint, and the looped interval sits entirely above the mean predictor.** But
   `150` is the budget already measured to undertrain, `looped3` does **3× the operator
   work per step**, and the `0.0484` gap **sits inside the measured seed spread**
   (`sd 0.064106` for settled). **Disjoint one-seed bootstrap intervals rule out
   resampling noise, not seed noise.** `falsifier()` returns `complete: false`.
36. **THE DISTINCTION THAT IS EASIEST TO LOSE AND MOST EXPENSIVE TO LOSE:** a bootstrap
   interval computed on a single seed measures **resampling** variability only. It says
   nothing about **seed** variability, which is the larger quantity here. Any future
   single-seed reading must state which of the two its interval covers.

37. **THE FAILING TESTS ARE NOT BROKEN — THEY ARE THE RECORD REPRODUCING.** 146 confirmed
   failures across `chase`/`cameron`/`foreman`, **every one a plain `AssertionError`**,
   with **zero import errors, zero collection errors, zero stale-symbol crashes**. Their
   docstrings predict the exact failure in advance (*"RED ON PURPOSE"*,
   *"VERDICT: DELETE"*). **Fixing any of them deletes a finding.** Zero were fixed, which
   was correct.
38. **ONE REAL FLAG, NOT ACTED ON:**
   `tests/chase/test_scale_sizing.py::...[cuda-1024]` returned `XPASS(strict)` — the
   `KNOWN_RED` ledger records `3.13×`, this run measured `≤ 2.0×`. **Measured on a box
   running multiple concurrent GPU suites**, so the likely cause is contention slowing
   softmax's own baseline. **Needs a clean re-measurement on a quiet GPU before the
   ledger is edited.**
39. **DOC ROT — the `gram_audit` half is FIXED; the measurement gap it named is not, and
   the generated tables are not.** `scale/arm_s.py` no longer cites `gram_audit`:
   commit `54148e6` rewrote the paragraph to state that no such function or test exists,
   that an earlier revision claimed one did, and that the nearest real artefact is
   `foreman_hilbert.positivity_audit`, which counts exact zeros-on-support over the whole
   float32 softmax matrix rather than censusing `a_p[0]`. **What remains open is the
   measurement, not the sentence:** no reading of float32 underflow in the pivot
   coordinate is on record, and the shipped settle sidesteps the question by running in
   the log domain (`log_pivot_context`), so the debt falls due only if the
   probability-domain path `pivot_context` is ever shipped.
   **Still open, separately:** `results/capability_table_v0.*` is stale at
   `journal_commit 9629616`, and its `Limits` string — like `capability_table_v1.*` and
   `ceq/hf_artifact/capability_table_v0.json` — still carries the clause saying
   `scale/m3_quintuple.py` has no `--task` flag. **The generator was corrected this
   round and the artifacts were not**, because regenerating them is a table cut and a
   cut is a measurement decision, not a documentation one. The next cut clears all three
   files; `tests/neptune/test_capability_table_truth.py` fails if the generator ever
   reverts.
40. **One audit hit on the pushed history is a FALSE POSITIVE** — commit `7844d12`'s body
   names the harness directory path inside the paragraph explaining the ignore rule. Not
   an authorship trailer. Left rather than rebased across four commits other agents were
   appending to.
41. **CITE SYMBOLS, NOT LINES — the convention, now measured rather than asserted.**
   Commit `916141d` already cited one test by name instead of a line; this round found
   the general case. Of the line numbers flagged as drifted in `scale/m3_flops.py`,
   **every one had moved by exactly +9** (`124→133`, `155→164`, `168-172→177-181`,
   `173→182`, `176→185`, `179→188`, `254→263`) — one insertion upstream silently
   invalidated nine citations at once. The correction list circulated with the audit was
   itself wrong on two of them (`155→156`, really `164`; `107→106`, really `116`, the
   `select_pivots` call rather than the enclosing `def`), so a sweep applied from that
   list would have replaced stale numbers with different stale numbers. Adjacent
   citations into `scale/m3_capability.py` had drifted by one and nobody had noticed.
   **The rule: cite a `def` by its symbol name alone; cite an expression inside a
   function by the enclosing symbol plus a distinctive quoted fragment — `the
   "out[-1] = (log_alpha.exp() @ av)" contraction in arm_s.arm_s`. Both forms are
   greppable and neither drifts.** Keep a bare line number only where there is no
   enclosing symbol to name. `scale/m3_flops.py`, `scale/m3_quintuple.py` and
   `M3_QUINTUPLE_PREREGISTERED_READING.md` were converted this round; **the rest of the
   tree was not swept**, and in-tree line citations elsewhere remain untrustworthy.
