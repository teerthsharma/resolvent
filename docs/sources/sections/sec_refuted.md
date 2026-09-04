# 3. What the record refuted

*Section 3 of the shape paper. Written by MARS holding the MORIARTY role, 2026-09-03,
against HEAD `207e7b9`. Every load-bearing claim carries `RUN` (executed this session on
this box, torch 2.5.1, float64), `READ` (`path:line`, quoted or paraphrased from the
tree), `CITED` (`[V]` abs page fetched this session, `[U]` reached through the record
only), or `DERIVED` (steps written out). Numbers without one of those tags do not appear.*

The campaign that produced this repository ran fifteen rounds and shipped no attention
claim. It shipped 169 machine-checked declarations, 65 catalogued mechanisms of its own
failure, and a struck-constant registry with twelve entries (`READ STRUCK.md:18`). This
section is the paper's account of what died, by which number, against which control — and
then the taxonomy of *how* it died, because the shape proposed in §4 is not exempt from any
mechanism listed here. Part C turns the taxonomy on the thesis itself.

## 3.0 The headline defect, stated first

Every bed in the tree asks for one real at position `s−1`. `equilibrium_oracle` returns
`z*_{s−1}`, *"the last coordinate of `(I − A)^{-1} b`"* (`READ scale/negation_scope.py:286-304`);
`ArmSMPrime.forward` returns `readout(h).squeeze(-1)[:, seq-1]`, shape `[n]`
(`READ V20_R15_THEORY_TABLE.md:214`). That is the single-location regression shape on which
one attention layer is asymptotically Bayes-optimal — Marion, Berthier, Biau, Boyer,
*Attention layers provably solve single-location regression*, arXiv:2410.01537 `CITED [V]`.
The record files this as `MISTAKES.md` D-1, *"the largest one in the repository and it
subsumes most of the null results"* (`READ MISTAKES.md:677-679`), and its own §17.5 adds the
caveat the paper must carry beside it: the theorem is asymptotic under `d→∞, L=o(d)`, its
predictor is `erf`, not softmax, and the repository ran `L/d = 4.00` — *"'Provably' is not
earned at this geometry"* (`READ MATHEMATICS.md:1046-1056`). So D-1 is a mechanism, not a
theorem about the campaign: the label class put softmax at a point where the record could
not have distinguished "cannot be beaten" from "was not beaten", and 0 of 39 scoreboard
items were earned (`READ workdonenew.md:44`).

## Part A — the dead programmes, each with the number that killed it

| # | programme | the number | interval / control | where |
|---|---|---|---|---|
| A1 | signed strictly-causal path sum (`sgate`) | sign-flip rate `0.16511 / 0.02732 / 0.00000 / 0.00000` at `s = 8/32/128/512`, slope **`−1.298`** vs bar `−0.3` | softmax control exactly `0.0000`, `0/1024` flips; kill survives `floor = 0` at `−1.1150` (R² `0.9350`) | `READ PROGNOSIS.md:40-63` |
| A1′ | the same, downstream | COGS-gen **`0/512`** vs softmax `15/512`, one-sided Fisher `p = 2.75e−05`; in-distribution `0.7734` vs `0.9258` | `3,652,096` params matched on both arms; `zero_success_upper_bound(512) = 0.005834` | `READ RESEARCH.md:178-194` |
| A2 | pivot routing | dense slope `−1.088` vs pivot `−1.298`: routing makes decay *worse* | at `s = 8` routing restricted nothing, `A[:,P]A[P,:] = A@A` to `1.86e−09` | `READ PROGNOSIS.md:44-45, 79-85` |
| A2′ | the exclusion confound | max legal pivot `s−2`; label `a[s−1]·b[s−2]` sits on the hidden position; `+100` perturbation moves softmax `101.6983`, pivot path `3.263746` | lifting it: `twin_plus 0.938728` vs threshold `0.871391`, `0.015610` *worse* than twin; the inert version selected `s−1` in `0/32` draws (V-9) | `READ MATHEMATICS.md:329-347`; `READ D1.md:385-398` |
| A3 | settled mixture weights over pivots | `settled − twin = −0.002959`, exact 95% CI **`[−0.042903, +0.031557]`** over all `3125` paired resamples, `126` atoms | settled sd `0.064106` vs twin `0.016547`, `3.874×`; `argmax − softmax = −0.118456` CI `[−0.134115, −0.102786]` | `READ MATHEMATICS.md:355-380` |
| A4 | the hop-2 term on the softmax operator | prediction filed blind, `4/4` held: `0.976488` (sd `0.004039`, CI `[0.973932, 0.979036]`) vs softmax `0.975371`, contrast **`+0.001117`**; `ĥ = 0.372`; `floor₂ = 0.866025` not approached | five routes agree: K sweep, γ sweep (no `γ` beats `0`), deflation, Wiener (`979×` attenuation), gated hop `0.966692` worse than both | `READ V13_PREDICTION_HOP2.md:35-46`; `READ workdonenew.md:266-294` |
| A5 | VGPE, path-ordered non-abelian transport as PE | occupied outright: PaTH §2.1, `A_ij ∝ exp(k_jᵀ(∏ H_s)q_i)`, RoPE as `H_s = R` | Hankel rank is `d` regardless of alphabet | `CITED [V]` arXiv:2505.16381; `READ PRIOR_ART.md:742-778, 1004-1013` |
| A6 | the original parity clause `g ≡ 0 ⇒ bitwise attention` | row `i` sums to **`i + 1`**; smallest witness `i = 1`, row `(1, 1)` | Lean `gate_zero_row_sum`, `gate_zero_not_stochastic`; repaired by the telescope, `2.2e−16` at the oracle setting | `READ workdonenew.md:47-124`; `READ lean/CEQ/V15.lean` via `MISTAKES.md:2052` |
| A7 | "the normalizer obstructs path products" | false; dissolves uniquely at `γ_j = 1/(1 − a_j)` | additive-logit repair computes `y_i / R_i` to `4.44e−16` | `READ workdonenew.md:118-124`; `READ README.md:208-212` |
| A8 | X₃₅ hidden-cause inference | Basseville & Nikiforov 1993 §7.2.4, *"equation for equation"*; both must-fires discharged by the 1993 equations on the first attempt | learned-model form arXiv:2604.25655 Thm 3.1 `[U]` | `READ README.md:214-219`; `READ workdonenewseal.md:461-463` |
| A9 | closed-magnitude phase gates | S4D ReLU variant: `\|Ā\| = 1.0` exactly on **`32.93%`** of a standard sample, 2022 | `m = 0` attainable: modReLU, `14.70%` | `READ workdonenewseal.md:459-460`; `READ V15_X36_PRIOR_ART.md:396` |
| A10 | R1, the deciding measurement (BED-M, `t* = 2`, N = 8) | ARM PL mean `0.829151`, sd `0.253673`, CI **`[0.617075, 1.041227]`** straddles `floor₁ = 0.7071`; `ĥ = 0.625017` | `5 of 8` cross, `3 of 8` NO READING; `â_max ∈ [1.1029, 285.0719]`, `1/(1 − â_max)` undefined at all eight seeds | `READ V15_R1.md:51-56` |
| A11 | `floor₁ = √((t*−1)/t*)` as an information floor | violated by **`13 of 40`** banked cells (12 `arm_pl`, 1 `arm_smprime`); `s = 64` on `40 of 40` | `floor₁` is where `ĥ = 1`; the real floor on BED-M is the exact oracle at `0.0` | `READ V20_R15_THEORY_TABLE.md:69-91`; corrections C11, C15 (`READ V20_R15_JOURNAL.md:47,51`) |
| A12 | Q6, the state metric | **F4, domain empty**: both wings return `[n]`; `0 of 40` cells journal a distribution; marginal `W1` reads `0.0` on the oracle's own values permuted while NRMSE reads `1.421901` (`> √2`) | metric and bed rank two predictors in opposite orders by `14.465×` | `READ V20_R15_THEORY_TABLE.md:211-225` |
| A13 | M6 vs M2 (`IMPOSSIBLE.md` I1) | `w(A) = 1.499315 > 1`, so `2w^h` grows (`2.999 → 10.11`) while `‖A^h‖` falls (`2.838 → 0.210`) over `h = 1..4` | every eigenvalue guard vacuous: spectrum `{0}`; no denominator-free magnitude certificate for the signed sum was found | `READ IMPOSSIBLE.md:34-76` |
| A14 | Theory 2 (T1 spectral gap, T2 Hodge, T3 ESS) | T1: `0 of 256` eigenvalues above `1e−12`, gap `0.000e+00` on every input; T2: `4.39e12`-entry 1-Laplacian at seq 2048; T3: exactly one fixed point by `occupancy_is_exact_inverse` | the only non-commuting pair on the board was two heads, `‖[A₁,A₂]‖_F = 0.668` | `READ THEORY2.md:21-28, 105-131` |
| A15 | the two-spheres round (Fisher–Rao displacement) | K1 slope `−0.4137` CI `[−0.4579, −0.3704]` vs bar `≥ −0.10`; `θ = arcsin √TV` exactly, so K3 compared two aggregations of one number | sphere loses to a held-out `TV^p` at 8 pivots, `−0.0643 [−0.0981, −0.0159]`; Karcher mean non-unique on `48.3%` of draws at `k = 128` | `READ D1.md:40-97, 432-436` |

Three notes the table cannot carry.

**A1's exponent is not a measurement; its verdict is.** The `−1.298` is fitted on two
nonzero points; the two solid points alone give `−1.2977`, the zeros move it by `0.0003`
(`READ PROGNOSIS.md:357-364`). The same operator on a different geometry reads `−0.958`
(R² `0.9990`) at `floor = 0` and the Inspector's `−1.221` does not reconcile with its own
five rates — recorded as unresolved (`READ RESEARCH.md:147, 275`). The paper carries the
kill (`< −0.3` by a factor of four under every floor tried) and *no* exponent.

**A2 has a second, deeper reading.** *"`pivot_signed` was `pivot_unsigned` wearing a
name"*: at the harness geometry `_causal_sgate_operator(lam=0.10)` is entrywise
non-negative, min entry exactly `0.0`, so the paired test that credited sign compared two
non-negative operators (`READ workdonenew.md:379`). Every M2 and S2 headline was measured on
`tgate`, which ships nowhere (`READ PROGNOSIS.md:112-144`, instrument #17). The negative on
signed attention is therefore also a negative on the record's ability to tell which operator
it had measured — the bind `tests/loop/test_measured_operator_is_shipped.py` exists because
of it.

**A4's mechanism is the one that transfers to the shape.** *"The label composes values along
paths; the arms composed weights across positions"* (`READ MISTAKES.md:1786-1787`). The BED-M
label is a path product `a_{s−1}···a_{s−t}·b_{s−1−t}`; attention supplies weighted sums; no
`K`, gain or deflation moved the term because it was the wrong *shape* — and the corpus
itself *"asks for what a gated linear scan computes natively"*, so the softmax control was the
wrong primitive too (`READ workdonenew.md:281-293`). The shape's I2 (the path product *is* the
sub-diagonal resolvent, `RUN` this session: max abs `0.0`, last row vs `equilibrium_oracle`
`6.2e−15`) is the algebraic statement of the same fact. It is why §4 does not add a hop to
softmax; it makes the resolvent the operator.

## Part B — the taxonomy by mechanism

Four classes, ordered as the record orders them by cost (`READ MISTAKES.md:9-23`). One line
per mechanism, with the instance the record filed. Line numbers are those of `MISTAKES.md`
at `207e7b9` unless another file is named; the record's own P-6 says to cite the symbol, so
each line also carries the heading text.

### V — vacuous controls (a control that cannot fail)

- **V-1** two registry keys, one corpus — `impact_hetero` bound to `make_impact_batch`, byte-identical tuples (`:34-56`)
- **V-2** hand-built minimal example where right and wrong coincide — 2×2 greedy; repaired to `111/400` drawn (`:58-70`)
- **V-3** the assertion is an algebraic identity of its own construction — `pooled < tail` `400/400`; `nrmse(t,t)` in `calibrate_bar` (`:72-86`)
- **V-4** fired, but on the wrong cause — row 0 of `tril(-1)` gives NaN, `torch.equal` False (`:88-96`)
- **V-5** slice-to-nothing — `X in ""[:0] + X` (`:98-106`)
- **V-6** the branch under test never ran — `converged=True, steps=1` (`:108-115`)
- **V-7** a search structurally incapable of finding anything, read as absence — `journal_scan`, true `22 of 68` (`:117-134`)
- **V-8** the PASS half's label is constant — one-component graph, label sd `0.0`, `1024 same / 0 different` (`:136-152`)
- **V-9** a repair that changes nothing — `exclude=(0,)`, `s−1` selected in `0 of 32` (`:154-166`)
- **V-10** a gate whose threshold is satisfied by construction — `0.1 ≤ 1.0`, `30/30` (`:168-176`)
- **V-11** a precondition satisfied at every real draw — `~0.115` vs measured median `1.2454e−20` (`:178-187`)
- **V-12** a single absorbing target makes the label constant — constant to `1.11e−14` (`:189-198`)
- **V-13** a search whose walk includes nested checkouts — `12` hits, `10` phantom worktree copies (`:202-235`)
- **V-14** the control that validates the matcher and never the reach — `366` candidate files became `0` (`:238-285`)
- **V-14a** the scope test that condemns every refusal guard — `19 of 31` fired, `17` not defects; role must be declared (`:770-802`)
- **V-15** the condemning rule with no planted negative — seven of R10's first fourteen condemn (`:804-826`)
- **V-16** the instrument that cannot measure, reporting a pass — `int(out) if out.isdigit() else 0` (`:828-853`)
- **V-17** the threshold imported out of its units — `δ = 0.5` from the chain task; `fd_max × |B| ≈ 3.741` (`:855-889`)
- **V-18** the guard that only guards its callers — inline regroup read `N=9/10` for `8` seeds (`:891-926`)
- **V-19** a wait-gate polling for work that had already finished — `bc` absent, `4h22m` of polling (`:928-959`)
- **V-20** a cap requested in prose, and a sentinel that reads it as infinity — `MAX_ITERATIONS = 0` (`:987-1024`)
- **V-21** a tail read as if it were the file — last 600 bytes generalised (`:961-985`)
- **V-22** a pre-registered constant carried in from another system — `α = 0.2` from the damped pendulum, prefactor set to 1 (`:1140-1170`)
- **V-23** a plural claim whose own central member is the counterexample — *"carriers conserve mass"* vs `i + 1` (`:1626-1656`)
- **V-24** an identity bind whose rejection region is empty — the two-branch escape passes parity for the label itself (`:1658-1721`)
- **V-25** a theorem whose hypothesis no draw in the corpus satisfies — `∀k, 0 < a k` on `0 of 2,048` sequences (`:1954-2035`)
- **V-26** a marginal assertion standing in for a joint claim — two counts cannot express "same seeds" (`:2190-2235`)

### P — provenance failures (a number with no live producer, or a claim true once)

- **P-1** a number with no live producer — `5.4944e−13` in a comment only (`:289-298`)
- **P-2** a number whose only home is a commit message — IMPACT gate figures, `74e5590` (`:300-314`)
- **P-3** a stale claim never retracted — *"still running"* for a finished `settled_plus 0.956787` (`:316-330`)
- **P-4** claimed scaffolding that does not exist — Tonnetz stubs, zero `.py` (`:332-341`)
- **P-5** doc rot pointing at nothing — `gram_audit` (`:343-362`)
- **P-6** line-reference drift — `arm_s.py:97` is `:106` (`:364-373`)
- **P-7** vocabulary with no referent — `§U`, `X₂₁`, "contract v10.1" (`:375-383`)
- **P-8** a headline that states an upper bound as a price — `≈ 29.6 h` measured `5.9 h`, `5×` (`:387-422`)
- **P-9** a commit message that contradicts the commit before it — `67162a6` vs `ecbedf7`, 28 s apart (`:1026-1048`)
- **P-10** a source's intro was cited as its theorem — `[V]`-as-theorem is `4/40 = 10%`, not two-thirds (`:1382-1428`)
- **P-11** a contract citing its own `[M]`-tagged theorem as settled — *"replicator by Lean #8"* (`:1597-1624`)

### M — measurement failures (the instrument measured, but not the thing the verdict names)

- **M-1** train and eval saw different preprocessing — raw vs standardised `y`, control read `2.446645` (`:427-449`)
- **M-2** a threshold refitted to the data it judges — `0.027260` frozen with provenance; the one correct site (`:451-462`)
- **M-3** pilot spread taken as the realised spread — `2.18×`; cost `3.48 → 7.43–11.19` at `s = 64` (`:464-493`)
- **M-4** a single-seed bootstrap interval read as seed variability (`:495-504`)
- **M-5** a process that cannot cross its own threshold — ceiling `3.80` vs `THRESHOLD 40.0` (`:506-516`)
- **M-6** a partial run read as a verdict — `2 of 6`, `22 of 60` (`:518-527`)
- **M-7** a pre-registration with a hole — rows A–G all conditioned on `settled` winning (`:529-541`)
- **M-8** pricing every arm at one arm's rate — `3.0×` over; twin/settled ratio inverts across tasks (`:544-578`)
- **M-9** a verdict whose finest achievable p cannot reach the α it quotes — `0.0625` at N = 5; `126` atoms (`:580-672`)
- **M-10** a finding filed without grepping for the guard that already existed — thread floor `2.345e−3` (`:1072-1107`)
- **M-11** whitening checked at lag 1, integrated at frequency zero — ARL₀ `41.5` vs nominal `1000` at `φ̂ = 0.709` (`:1109-1138`)
- **M-12** a calibrated model whose untested branch went unchecked — checkpointed branch constant in `L` (`:1172-1208`)
- **M-13** an equivalence margin registered without a reachability check — TOST half-width `0.8807σ` vs `0.5σ` at N = 8; power `0.80` first at **N = 70** (`:1210-1247`)
- **M-14** a gate whose threshold sits on the edge of its own null — softmax clears `1.0` by `5.6e−4` (`:1249-1287`)
- **M-15** a descriptive statistic reported without its null — `0.7550` is the null to four decimals, `z = −0.30` (`:1289-1344`)
- **M-16** the thread lane broken by the person who filed the rule — `threads=12` then `14` (`:1346-1380`)
- **M-17** a census that classified the correction record as the defect — `6.63` mention vs use (`:1430-1481`)
- **M-18** a pre-registered kill-diagnostic whose value the corpus fixes — `log|a|` identically `0` on the band; and the order-audit correction inside it (`:1483-1562`)
- **M-19** a dynamical invariant estimated on a collapsed float64 orbit — `~52` steps for a tent map (`:1564-1595`)
- **M-20** a pre-registration that predicts both outcomes, in two sections that never met — BED-K transposed (`:1723-1774`)
- **M-21** a diagnostic prescribed by its statistic — `sign(a)` probe `p₀ = 0.943741` at 0 steps, trained gain `−0.025787` (`:2101-2188`)

### D — design-level failures (the answer was fixed before any data arrived)

- **D-1** racing a baseline at its proven optimum — `out[:, s − 1]` (`:677-708`)
- **D-2** an oracle that is the arm's own resolvent — `e3_t*`, `settled − softmax` VOID (`:710-725`)
- **D-3** a difficulty dial that does not vary — rank exactly `2` across `t* = 1..63` (`:727-738`)
- **D-4** registration without admission — `E_T_STAR` missing keys; `KeyError` vs silent degrade (`:740-753`)
- **D-5** declaring `0.0` without a movement test (`:755-766`)
- **D-6** the repair that was written and never started — `/tmp/waveB2.sh` (`:1050-1070`)
- **D-7** a prediction filed without its counter, in a document whose errors have a sign — `7 of 8` optimistic, `p = 0.0352` (`:2037-2099`)

### B.2 The ten most likely to recur in a paper about a NEW shape, and the rule each imposes

Ranked by (i) how many of the record's own rounds committed it and (ii) how directly the
shape's definition invites it. The rule column is what §4's binds, beds and predictions must
satisfy; each is a design-against clause, not advice.

1. **D-2 — the oracle is the arm's own resolvent.** The shape's state is literally
   `z = (I − γP)^{-1}V`; a bed whose label is `(I − γ*P*)^{-1}V*` makes every contrast a
   reproduction check. **Rule:** the label is generated by a latent environment chain `P*`
   the arm never sees; the pre-registration lists which contrasts are VOID *before* any cell
   runs; E1's registration comment (`READ scale/negation_scope.py:1033-1041`) is the model.
2. **D-1 — racing the optimum.** **Rule:** no bed in the shape paper scores a scalar at one
   position. The label is the vector `z*` (length `s`), the displacement field `Δz`, or the
   argmin over candidate moves; the paper states the regime in which the softmax skyline is
   optimal and shows the bed outside it, with the `L/d` geometry printed (§3.0).
3. **V-24 — an identity bind with an empty rejection region.** I1 (`γ = 0 ⇒ P V` bitwise)
   is proved by one fact — `γ` multiplies the second term — and mentions nothing about `P`,
   `𝒜_k`, or the solve. **Rule:** every bind ships the mutilation battery: drop the absorbing
   rows, drop the solve, replace `P` with Gaussian noise, substitute the label for `z`; the
   bind is admissible only if the honest construction passes and each mutilation fails at
   `O(1)`, counts printed, as `V15_JUPITER2_FORK.md` did (`0.9749 / 0.9165 / 1.000 / 0.4845`).
4. **V-8 / V-12 — the constraint sets make the label constant.** A single absorbing target
   collapsed a label to `1.11e−14`; a one-component graph gave `sd 0.0`. **Rule:** the builder
   prints, at construction, `sd(q^{(k)}) > 0`, `0 < frac(argmin = a) < 1` over candidate
   moves, both classes non-empty, discards counted; `BOARD.md:269`'s triple, for every `k`.
5. **V-25 — a theorem whose hypothesis no draw satisfies.** I4 needs `P` non-negative and
   row-stochastic with `γ < 1`. **Rule:** a domain census beside every gating theorem — the
   measured support of `γ` and of `row-sum(P)` on drawn cells, the fraction admitted at the
   quantifier level the theorem uses; `0%` blocks the gate.
6. **M-13 / M-9 — a margin or α the design cannot reach.** The parity half needs N = 70 for
   TOST power `0.80`; at N = 8 two bit-identical arms return NO VERDICT. **Rule:** parity is
   claimed by identity bind (with rule 3's battery), never by TOST at N < 70; every seed CI
   prints `n+`, the seed-agreement count.
7. **M-21 / M-18 — a prescribed diagnostic whose value the corpus fixes.** **Rule:** the
   paper names the *discrimination* (trained vs zero-step on the same seeds) and admits a
   statistic only after three readings: corpus-alone ceiling, zero-step floor, headroom
   printed. `Var(z) = 0` on the corpus blocks registration.
8. **D-3 / P-8 — a dial that does not vary, a price that is an upper bound.** `s = 64` on
   `40 of 40` cells, exponent unidentified; `≈ 29.6 h` was `5.9 h`. **Rule:** the cost law is
   measured at `s ∈ {64, 256, 1024, 4096}` with `torch.cuda.synchronize()` and randomised
   execution order; every headline price carries its direction (`≤`) and the implementation
   it assumes.
9. **D-7 / L-SIGN — a prediction without its counter.** Nine contract statements, seven
   optimistic. **Rule:** every prediction in §8 ships a counter of equal specificity in a
   separate commit, and a calibration column (checked / wrong / sign) is kept across rounds.
10. **V-3 / V-10 — an assertion that is an identity of its own construction.** Part C6 shows
    I4's "certificate" is one. **Rule:** before any `A ≤ B` enters a verdict, state what set
    each side ranges over; if `B` is computed from `A`'s hypotheses alone, label it
    *definitional* in the output, as `predict_the_mean` now is.

Three more that are one step behind: **V-17** (a threshold in un-anchored units — any
committor threshold must be stated in NRMSE or in Fano-floor units, never in `fd`); **M-15**
(a statistic reported without a sharpness-matched null — any "β₀ persistence of the influence
graph" number needs its random-operator null first); **M-2** (a threshold refitted after the
data — the `⟨CLAUSE_1_TAIL⟩` lesson, `READ V20_R15_THEORY_TABLE.md:265-268`).

## Part C — attacks on the thesis itself

The thesis (`BRIEF.md` §1) is a bounded resolvent `z = (I − γP)^{-1}V` over a row-stochastic
causal `P`, with absorbing constraint sets `𝒜_k`, the committor read `q^{(k)}`, the
interventional displacement `Δz`, and the safest move `argmin_a max_k q^{(k)}(do a)`. Twelve
attacks follow. Each names the mechanism it instantiates, the value that would be measured,
and the number that fires the kill. Two of them (C6, C11) are findings against the brief
as written, not hypotheticals.

**C1. The oracle equals the arm's own resolvent (D-2).** If the bed's label is
`(I − γ*P*)^{-1}V*` and the arm's read is `P̂(I − γ̂P̂)^{-1}V̂`, the contrast is a
reproduction check, exactly `e3_t*` (`READ MISTAKES.md:712-719`). *Measure:* `‖P̂ − P*‖_∞`
per cell at trained weights, and the NRMSE of the *softmax* arm on the same label.
*Kill:* if the paper's pre-registration does not list `shape − softmax` as VOID on any bed
whose oracle is a resolvent of a matrix recoverable from `x`, the bed is struck; if
`‖P̂ − P*‖_∞ < 1e−3` on ≥ 6/8 seeds the arm has copied the environment and no capability
number is creditable.

**C2. Feature leak of the transition matrix into the context (M-21, V15_SATURN2).** The
record's gate `a` was written into `x[:, :, CH_DRIVE]` for the entire campaign; a corpus-alone
least-squares probe reads gate `R² = 1.000000` with no arm (`READ MISTAKES.md:2133-2143`).
A bed that places the rows of `P*` (or the absorbing membership bits) in the context is the
same defect. *Measure:* corpus-alone linear probe from `x` to (a) rows of `P*`, (b) the
indicator of `𝒜_k`, (c) `q^{(k)}` itself. *Kill:* `R² ≥ 0.99` on (a) or (b) makes the bed a
copy task; `R² ≥ 0.5` on (c) at order 0 means the label is reachable without a solve, and the
shape's one-read claim is void on that bed.

**C3. The deeper-softmax skyline at matched parameters.** `hop_k` needs depth
`⌊log₂ k⌋ + 2` (Sanford, Hsu, Telgarsky, arXiv:2402.09268 `CITED [V]`; the `Ω(log k)` lower
bound is conditional, `READ MATHEMATICS.md:284-296`). A depth-5 softmax stack at 4,769
matched parameters computes the `t* = 8` label. *Measure:* the skyline column, `Δ_sky`, at
every rung, per `CEQ_V16_CONTRACT.md` R-SKY (`READ :209-211`). *Kill:* if the skyline is within
`Δ_res` of the shape at every rung, the shape has no capability claim on that bed — only the
exactness (printed δ), one-read consistency and boundary-condition claims survive, and those
are stated as *properties*, never as "beats".

**C4. The absorbing sets trivialise the label (V-8, V-12).** With `K = 1` the safest move is
whichever candidate lowers one committor; with a single absorbing node the committor is
constant to `1.11e−14` (`READ MISTAKES.md:191-192`). *Measure, in the builder:* `sd(q^{(k)})`
per `k`, the fraction of draws on which the argmin is unique, the fraction on which the
lexicographic and max-form safest moves disagree. *Kill:* any `sd = 0`; argmin-unique
fraction outside `(0.05, 0.95)`; disagreement fraction `0` (then the two forms are one form
and the paper may name only one).

**C5. `γ` pinned at 0 by training — the arm is softmax wearing a name.** Precedent: the
hop-2 gain sweep, `γ ∈ {0, .05, .10, .25, .50, 1.0}`, where no `γ` beat `0` and the CONTENT
branch held (`READ workdonenew.md:276`); and `pivot_signed` being `pivot_unsigned` wearing a
name (`READ workdonenew.md:379`). *Measure:* trained `γ̂` per seed with its CI; the ablation
`(I − γ̂P̂)^{-1} → I` at trained weights (V-9's delete-in-process). *Kill:* `|γ̂| < 0.05` on
≥ 6/8 seeds, or the ablation moving NRMSE by less than one seed sd. The mirror kill: `γ̂ → 1`
sends the I4 factor `1/(1 − γ)` to infinity and the certificate becomes vacuous — print
`1/(1 − γ̂)` beside every δ, exactly as `1/(1 − â_max)` was printed and found undefined at all
eight R1 seeds (`READ V15_R1.md:56`).

**C6. The certificate delta is an identity, and can still be exceeded on the object that
ships (V-3, V-10, P-8).** `RUN` this session, the coordinator's own I4 block: at
`K = 1, 2, 4, 8, 16`, `err_∞ = 1.6333 / 1.1433 / 0.5602 / 0.1345 / 0.00775` against bounds
`1.6333 / 1.1433 / 0.5602 / 0.1345 / 0.00775` — **equal to 15 digits at every `K`**.
`DERIVED`: for `P ≥ 0` row-stochastic, `(γP)^k` is non-negative with row sums `γ^k`, so the
tail `Σ_{k>K}(γP)^k` has row sums `Σ_{k>K}γ^k = γ^{K+1}/(1−γ)` exactly, and its `∞`-norm *is*
that number. The "certificate" is the row-sum identity; it carries no information about `P`
beyond the two hypotheses, and its planted negative (rows scaled to `1.5`, `err 119.37` vs
bound `1.143`, `RUN`) fires on the hypothesis, not on the resolvent. That is honest — an exact
truncation formula is better than a bound — but it is **not** L-CERT's certificate for a mask,
because the object that ships is the *sparsified* or *masked* operator, not the truncated
series. *Measure:* on the shipped mask, `‖O_full − O_mask‖_∞` over 1,024 drawn cells, against
the printed δ. *Kill:* exceeded on any draw; or `δ ≥ sd(label)` (an uninformative
certificate); or the row-sum identity presented as evidence about `P`.

**C7. `W1` is permutation-blind on the state metric (Q6 F4, refuted).** A predictor
returning the oracle's own values in the wrong order scores `W1 = 0.0` while NRMSE reads
`1.421901`, worse than predict-the-mean; against `oracle + 0.1σ` the metric and the bed rank
in opposite orders by `14.465×` (`READ V20_R15_THEORY_TABLE.md:221`). Any vector-valued or
displacement-field metric inherits this the moment it is marginal. *Measure:* the metric on
(i) the oracle permuted, (ii) the oracle plus `0.1σ` noise, (iii) the oracle with `Δz`
negated. *Kill:* (i) reading `0`, or (i) preferred to (ii), or (iii) not distinguished from
the oracle. The state metric must be joint — per-coordinate NRMSE plus a position-aware
transport or Procrustes cost — and declared before the corpus run (`READ MATHEMATICS.md:95-99`).

**C8. The safest-move label is decidable at zero hops — a third static task.** The
record's static beds (product of two entries, a sum squared) gave an iterating arm nothing to
compute toward (`READ D1.md:286-291`). If `argmin_a max_k q^{(k)}(do a)` is a function of the
candidate's own features, it is static. *Measure:* a 0-hop predictor (per-position MLP on the
candidate's context row) and a 1-hop softmax, both at matched parameters, on the argmin label,
with exact McNemar paired on identical draws (`READ MATHEMATICS.md:389-395`). *Kill:* the
0-hop arm's argmin accuracy within the Fano floor's resolution of the shape's, or McNemar
`p > 0.05` against the 1-hop softmax at N = 8 — then BED-S is a third static task and is
struck before a number is quoted.

**C9. The cost law never varies `s` (D-3, THEORY_TABLE §0.3).** `s = 64` on `40 of 40`
cells; the exponent in `s` is *"not poorly estimated, unidentified"*; the per-cell timer is
un-synchronised host wall clock whose strongest correlate is run order, `ρ = +0.7029`
(`READ V20_R15_THEORY_TABLE.md:93-109`). The shape's I5 (`s²d` forward substitution vs `2s²d`
for `QKᵀ`) is a FLOP count, not a price; the record's FLOP model already understated a bill by
`2×–3×` (`READ MISTAKES.md:469-476`). *Measure:* wall-clock at `s ∈ {64, 256, 1024, 4096}`,
synchronised, randomised order, N = 8. *Kill:* the fitted exponent's CI excludes the claimed
`2` in the direction of `3`; or the triangular solve does not beat `QKᵀ` at the `s` where the
paper claims it; or any price quoted before that run.

**C10. Both softmax obstructions are cited past their hypotheses (P-10 / L-EQ).** "Softmax
cannot follow" rests on a depth lower bound that is conditional on the 1-vs-2-cycle
conjecture (`READ MATHEMATICS.md:290-292`) and on a composition result — Peng, Narayanan,
Papadimitriou, *On Limitations of the Transformer Architecture*, arXiv:2402.08164 `CITED [V]`
— whose hypothesis is a domain size, with no instance run here. "Softmax is optimal" rests on
an asymptotic `erf` predictor at `L = o(d)` while the tree runs `L/d = 4` (§3.0). *Measure:*
for each of the four obstructions, the theorem's hypotheses as a predicate on the bed, the
fraction of drawn cells satisfying it, and one numeric instance. *Kill:* any obstruction
stated in the paper without its `[V-eq]` line and instance; `0%` admitted on the bed.

**C11. The operator as written is occupied.** The record's own archive lists InfSA —
`(I − γA)^{-1} − I`, content-adaptive Neumann series, `A = ReLU(QKᵀ)` normalised,
non-negative (`READ DONE_ARCHIVE_ROUND1.md:1315-1317`); the abs page confirms a discounted
Neumann series over attention matrices — Roffo, Abdelkawy, Lavie, Palmer, arXiv:2603.00175
`CITED [V]`. APPNP is `(I − (1−α)Â)^{-1}` personalised-PageRank propagation — Gasteiger,
Bojchevski, Günnemann, arXiv:1810.05997 `CITED [V]` — already named tier 2 in the
`ceq/attention.py` header (`READ :7-8`). Dayan's successor representation is
`M = (I − γP)^{-1}` (`READ THEORY.md:22,234` `[U]`). *Rule:* the bounded resolvent over a
non-negative row-stochastic `P` is cited before it is named; the paper's delta is stated
narrowly as *absorbing boundary conditions + the committor read + the interventional
re-solve*, and Part D bounds even that. *Kill:* any sentence calling the resolvent itself new.

**C12. I3 as scripted does not reproduce, and the reason is a V-16 in the coordinator's own
script.** `RUN` of `$SCRATCH/shape_identities.py` reads `I3_committor_eq_resolvent_maxabs =
0.858236511380934`, not `0.0`. `READ shape_identities.py:64-65`: the script asks the bed for
`basin_A` / `basin_B`, which `bed_1.build` does not return (its keys are `A`, `B`,
`READ ceq/beds/bed_1.py:167`), so `.get(..., [0])` and `.get(..., [n−1])` silently substituted
`A = [0]`, `B = [10]` for the bed's `A = [0]`, `B = [1]`. Re-run with the bed's own sets
(`RUN`, one-liner): `A [0] B [1]`, row sums `1.0 / 1.0`, **`I3 resolvent vs bed q maxabs
0.0`**, `max|Lq|` on the interior `1.04e−17`, `q` sd `0.2441`, `9` interior nodes. I3 holds
exactly; the brief's `READ + prior RUN` class for it was correct and the script's `0.858` is an
unreadable-key-reported-as-a-number. *Rule:* the identity scripts in §4 raise on a missing
key; a default value in an identity check is a V-16.

## Part D — DeltaNet and SSD are the native skyline, and a "beats native" sentence is banned

The record established, by direct LaTeX fetch, that DeltaNet's WY matrix
`T = (I + tril(diag(β)KKᵀ, −1))^{-1} diag(β)` is the exact inverse of a strictly lower
triangular, content-dependent, signed operator — nilpotent, hence a finite path sum —
shipping in `fla/ops/utils/solve_tril.py` (`READ DONE_ARCHIVE_ROUND1.md:1224-1233`; Yang,
Wang, Zhang, Shen, Kim, arXiv:2406.06484 `CITED [V]`). On the record's own probe it *"leads at
every s ≥ 16, decays the slowest, and is the only arm still nonzero at s = 512"*, with
non-overlapping wins at `s = 32, 128, 256` (`READ :1236-1248`); the S2 margin was later struck
as a floor artefact, and at `floor = 0` the ordering stands, `1.08×`, overlapping
(`READ :2510-2518`). Gated DeltaNet (Yang, Kautz, Hatamizadeh, arXiv:2412.06464 `CITED [V]`)
adds the decay gate; the record notes a companion writing `(I − A)^{-1} = Σ_n A^n` explicitly
with `A ∈ [0,1]` (`READ :1319-1320`, `[U]`). Mamba-2's structured state-space duality (Dao, Gu,
arXiv:2405.21060 `CITED [V]`) is the statement that these are decompositions of semiseparable
matrices — the triangular-solve class I5 belongs to.

Three consequences.

1. **The shape at its `β = 0`, QK-off, `γ`-general corner is a gated linear recurrence**
   (`gate_zero_beta_zero_is_linear_attention`, `READ lean/CEQ/V16Domain.lean:483`), and BED-M's
   label is *"what a gated linear scan computes natively"* (`READ workdonenew.md:288-290`).
   The native skyline for a resolvent-shaped read is therefore the chunked semiseparable
   solve — SSD's algorithm or DeltaNet's `solve_tril` — at matched parameters, not softmax.
   The paper's architecture table follows R4: each bed against *its* native skyline, and
   *"the honest sentence is whatever both columns permit"* (`READ CEQ_V16_CONTRACT.md:219-221`).
2. **"Beats native" is banned for the same reason "beats softmax" is banned on BED-M**
   (R-SKY, `READ CEQ_V16_CONTRACT.md:209-211`): the skyline is a fellow approximator of the
   same function class at the same cost, and the record's one attempt at such a sentence
   (S2, `1.39×`) was struck. What the shape may claim against the native skyline is a
   `Δ_sky` column with an interval, and the *mechanism* difference: DeltaNet and SSD carry no
   absorbing rows, no committor read, and no re-solve under `do(a)`.
3. **That difference is NOT FOUND, not new.** The record's sweep records no absorbing-boundary
   attention (`READ PRIOR_ART.md:634-657`, the NOT FOUND list, which names committor
   regression from molecular configurations, arXiv:1802.10275, arXiv:1906.06285 `[U]`, as the
   nearest occupants). No search for "absorbing" or "Dirichlet" attention was run *this
   session*; the paper writes `NOT FOUND — sweep owed` and never "novel".

## Limits of this section

All `RUN` numbers are from one seed on one CPU box in float64 and are identity checks, not
capability readings; none carries an interval because none is a statistic. Every `READ` line
number is at `207e7b9`; the record's P-6 applies. Six citations are `[V]` by abs-page fetch
only — titles and author surnames matched, equations *not* transcribed this session — so each
is `[V]` and not `[V-eq]` under L-EQ until §7 runs the instance. The Basseville–Nikiforov,
S4D, Dayan, InfSA-equation and companion-DeltaNet claims are carried from the record as `[U]`.
The Part B line numbers for `MISTAKES.md` are ranges read this session; V-20 and V-21 are
filed out of numerical order in the source and are listed here by number. Part C proposes
kills; none has been run, and every threshold in it (`0.05`, `0.99`, `6/8`, `1,024` draws) is
a pre-registration candidate to be frozen with provenance before data, not a measured null.
