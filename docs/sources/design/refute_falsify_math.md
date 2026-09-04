# REFUTATION — `design_falsify.md` under the MATHEMATICS lens

*MARS / MORIARTY, 2026-09-03, HEAD `207e7b9`. Target: `$SCRATCH/design/design_falsify.md`
(498 lines, VENUS). Evidence classes: `RUN` (executed this session on this box, numpy
float64, stdin one-liners, no file written), `READ path:line`, `CITED [V]` by canonical key in
`$SCRATCH/references.bib` (one fetch this session is marked `[V-fetched]`), `DERIVED` (steps
shown). Every KEEP / REPAIR / KILL carries a number, a bib key, or a `file:line`, and names the
`MISTAKES.md` mechanism it instantiates. Default under uncertainty is KILL; each such verdict
says what would reverse it. Regime vocabulary is `THESIS_NOTES.md` §"THE FAMILY": regime N =
strictly causal `W` (diagonal excluded, nilpotent, any `γ`); regime S = softmax corner (diagonal
kept, `γ < 1`, Neumann).*

---

## 0. The four findings that change the document (summary)

1. **Bets B and C are mutually exclusive at the design point** (`READ design_falsify.md:126-198`).
   The committor label is the undiscounted absorption probability `q^{(k)}` (oracle at
   `γ = 1` on the `Q`-block), but the arm reads `E_i[γ̂^{τ−1} 1_k]` at its trained `γ̂`. On a chain
   with `E[τ] ≈ 7…16` (the `t* = 8` design point) the read sums to `0.010…0.268` at `γ = 0.5`
   and `0.30…0.65` at `γ = 0.9`, and even after renormalisation misses `q` by `0.308 / 0.134 /
   0.019 / 0.002` at `γ = 0.5 / 0.9 / 0.99 / 0.999` (`RUN`, §3.12). Reading `q` to `1 %` at
   `τ = 8` needs `γ̂ ≥ 0.99857`, i.e. `1/(1−γ̂) ≈ 697` — which is Bet C's own *mirror kill*
   (`READ :186-189`, "vacuous certificate"). Bet B cannot hold while Bet C's certificate stays
   non-vacuous unless the label is re-declared as the discounted vector at a registered `γ`.
   Mechanism: V-17 (the label and the read are in different units), D-2 (a label that depends
   on the arm's `γ̂`).
2. **The Fano "floor" at `k = 0` sits `0.208` below chance and is not an information floor**
   (`READ :74, :140-141, :378`). At `I = 0` and a uniform argmin over `m = 8` the exact
   zero-information floor is `1 − 1/m = 0.875`; the printed `1 − ln 2/ln m = 0.6667` is the
   loose Fano form (`ln m` for `ln(m−1)`, `H(a*) = ln m` assumed). The design's own kill "a
   shape at chance (`0.875`) refutes by `+0.208`" measures distance to a number that no arm,
   informed or not, can be below. Under the balance guard as written (`READ sec_beds.md:211`,
   "none above `1 − 1/m + tol`") a class at `0.875` is admitted, `H(a*) = 0.620` nats, the true
   zero-information floor is `0.125`, and the printed `0.6667` is then *above* it — a floor that
   an uninformed arm violates. Mechanism: V-10, L-FLOOR (C15).
3. **The document predates `THESIS_CORRECTIONS_2.md` F1/F2 and is unsound on a causal `P`
   without them** (`READ THESIS_CORRECTIONS_2.md:37-47`): with BOS undeclared, `ρ(Q) = 1.0` and
   `I − Q` has rank `6 of 7` (`RUN`, §3.11); every `Σ_k q^{(k)} = 1` line in §2.2, §4 and Bet B's
   object is false on the arm's own operator until BOS is placed in a boundary set. With
   BOS-as-goal (the instrument's R6 convention) "reach the goal" collapses to "descend to
   position 0 without hitting a constraint", which is the complement of the constraint
   committors — the two heads `argmax q^{(0)}` and `argmin Σ_{k≥1} q^{(k)}` become one rule
   by construction. Mechanism: V-25, V-3, C4 of `sec_refuted.md`.
4. **ChaCAL removes the diagonal inside the inverse** (`CITED [V-fetched] fagnou-2024-chacal`,
   `arxiv.org/html/2410.05565` fetched this session: "we remove the diagonal part of the
   adjacency matrix in the inverse"). ChaCAL's resolvent is therefore regime N (nilpotent, a
   finite sum) under an outer full `A`; its read `(1−γ)A(I−γA_strict)^{-1}V` is **not**
   row-stochastic (`(I − γA_strict)^{-1}𝟙 ≠ 𝟙/(1−γ)`, DERIVED §3.15). The design's "ChaCAL
   control (same operator, no boundary rows) — DERIVED (identical arithmetic)" (`READ :66`),
   the `(1−γ)` row-sum sentence of §0.4, and the "ChaCAL + same `γ`" planted negative (`READ
   :147-153, :207-217`) all assume the diagonal-kept operator. The control must fix the
   diagonal convention before it is a control. Mechanism: P-10 (a source's equation cited from
   a summariser), V-24 (a planted negative that is not the published object).

---

## 1. Verdict table

| # | target (`design_falsify.md`) | verdict | one-line reason (evidence in §3) |
|---|---|---|---|
| 1 | §0.4 `(1−γ)` factor and row sums | REPAIR | bare `P(I−γP)^{-1}` has row sums `1/(1−γ) = 2.5` at `γ = 0.6`; `Π_γ` sums to `1` incl. absorbing rows (`RUN`); the document uses both forms without choosing (`:36`, `:90-91`, `:169-171`) |
| 2 | §2.3 `E[γ^{τ−1}]` identity and `γ ↑ 1` limit | KEEP (identity) / REPAIR (statement) | holds for `i ∈ T`, `V = 1_{𝒜_k}`, **with** the `(1−γ)` factor; at `i ∈ 𝒜_k` the read is `1`, not `γ^{−1}`; the limit equals `q^{(k)}` but `Σ_k q^{(k)} = 1` needs every absorbing state declared (F1) |
| 3 | §3.6 counter: `I − Q` invertibility needs a Perron certificate | REPAIR | on the arm's causal class with BOS declared, `Q` is lower-triangular with `ρ(Q) = max_T P_ii = 0.536 < 1` (`RUN`) — `lower_triangular_isUnit` `[M]`; the Perron route is needed only for the non-causal environment chain (BED-1, `ρ = 0.9409`) — two classes conflated |
| 4 | `(I − γP)^{-1}` with the diagonal and absorbing rows | KEEP | diagonal `1 − γP_ii ≥ 1 − γ > 0` for `γ < 1`; absorbing rows give `(1−γ)·e_i`; `sec_cost.md:9` |
| 5 | §5 rank 2 "keep regime N (`γ = 1`, nilpotent) exact without a dial" | KILL for component (e) | an absorbing row breaks strict lower-triangularity: `rank(I − A') = 7 of 8` (`RUN`); regime N at `γ = 1` cannot carry boundary rows; the survival route cannot carry the mechanism it is meant to save |
| 6 | §3.7 F0 endpoint = `segmentation_blockdiag` on BED-S (softmax corner) | REPAIR | the `[M]` target is `StrictlyLower` (regime N, `READ sec_proved.md:553-555`); on the softmax corner the zero is a `−∞` mask and no iff theorem exists in the tree (`no_prefix_scan_represents_a_zero_gate`, `V16Domain.lean:165`); the block-inverse half (`resolvent_fromBlocks`) does transfer and absorbing rows respect any block structure (DERIVED) |
| 7 | §3.10 EMC proposition | KILL (retire to a remark) | for `γ < 1` the post-intervention fixed point is unique, so both orders coincide on **cyclic** `P` too (`4.4e-16`, `RUN`); the `> 0.1` "violation" is the suffix-only re-solve, an algorithm valid only for triangular `P` (`0.0678`, `RUN`); "by nilpotency" is wrong — it holds by triangularity on the non-nilpotent softmax corner (`5.6e-17`, `RUN`) |
| 8 | §2.1 / §4 C6 displacement identity | KEEP | `max|Δz − (z' − z)| = 2.0e-16` (`RUN`); zero before the intervened row by triangularity (`RUN`) |
| 9 | §4 C6-plant "`V = 1 ⇒ Δz ≡ 0` exactly `0.0`" | REPAIR | `|Δz| = 1.1e-16` via the identity route, `2.2e-16` at an absorbing row (`RUN`); the exact `0.0` is a float accident of one code path, not an identity property; the rejection region (`1.127`, `0.1096`) carries the bind, not the zero |
| 10 | §3.2 Sherman–Morrison pricing | KEEP (algebra) / REPAIR (scope) | closed form vs re-solve `4.4e-16` (redirect) and `5.6e-16` (absorbing) (`RUN`); denominator at an absorbing row is `(1−γ)(1 + γ(PM)_rr) > 0` (`RUN` `0.511918 = 0.511918`); but a **key-side** edit of token `j` changes every row `i ≥ j` after renormalisation — `rank(ΔP) = 6 of 8` (`RUN`) — so "one column + one inner product per candidate" holds for query-side (row) moves only |
| 11 | §3.4 Neumann tail equality | KEEP (equality) / REPAIR (object) | equality to 15 digits with absorbing rows at `K = 1, 4, 8` (`RUN`); rows `1.5` read `5.9049` vs `0.1944` (`RUN`); but "the shipped mask" is a truncated series, not a sparsified `P` — a sparsified `P` needs the F1 union bound of §3.8, which §3.4 omits |
| 12 | C4's DET / parallel-depth sentence (inherited at §0.3) | REPAIR | `⌈log₂ s⌉` squarings are exact only in regime N; regime S needs `⌈log₂(log(δ(1−γ))/log γ)⌉` squarings for accuracy `δ`, each an `O(s³)` matmul; a "round" is a matmul, not a circuit level (depth `O(log² s)`) |
| 13 | §3.1 "identity rows change the *support* of the mixing matrix" | KILL as stated | on a dense causal softmax base the support changes **only on the absorbing rows themselves** (`RUN`: rows changed `= [3]`); the mechanism is a change of weights on rows downstream, not of support; "later positions" is the wrong direction on a causal `P` |
| 14 | §1 / §2.2 / §4 Fano floor `1 − ln 2/ln m` | KILL as an information floor | exact zero-information floor `1 − max_a π(a*)`: `0.875` at `m = 8` uniform; loose Fano `0.6667`, tight Fano `(ln m − ln 2)/ln(m−1) = 0.7124` (`RUN`); `m = 2` refused for the wrong reason (true floor `0.5`) |
| 15 | §2.2 Bet B object `Σ = 1`, F2 | REPAIR | needs BOS in a declared set (`RUN` §3.11); constraints after the query read `q = 0.0` exactly (R10) — a discard rule, not a label value |
| 16 | Bets B + C at the design point | KILL as a pair | §0.1 above; `RUN` §3.12 |
| 17 | Bet B "readable in one resolvent solve" | REPAIR | true of the oracle on the environment chain; a symmetric-support `Q` (BED-1, `oracle_ne_resolvent` hypotheses) is not triangular under any ordering, so the arm's causal resolvent can equal it only for DAG-with-self-loop chains |
| 18 | Bet A statistics (McNemar on the field cosine; MDE `0.039827`) | REPAIR | McNemar is a paired-binary test (`CITED [V] mcnemar-1947-correlated`); the cosine is continuous; the MDE quoted is BED-M's NRMSE paired sd (`READ sec_beds.md:310`) imported into cosine units — V-17 |
| 19 | §1 row 6, §2.2 second bet, §3.1 — ChaCAL "identical arithmetic" | REPAIR | §0.4 above (`[V-fetched]`) |
| 20 | §1 prices, §5 "4.7 min", MDE, `1/(1−γ)`, `δ = 1.526e-05` | KEEP | every arithmetic line re-derived (§3.16) |
| 21 | §3.5 / §3.8 killers marked NOT MEASURED | KEEP | honest; no number claimed |

---

## 2. Regime map — where the document conflates N and S

| line | as written | regime the statement is true in | what BED-S runs |
|---|---|---|---|
| `:345` "EMC holds for the causal class by nilpotency" | N | both (by triangularity) | S |
| `:290-294` F0 endpoint via `segmentation_blockdiag` | N (`StrictlyLower`) | N; S only under a `−∞` mask with `resolvent_fromBlocks` | S |
| `:393` "keep regime N (`γ = 1`) exact without a dial" | N | N **without** absorbing rows | S with absorbing rows |
| `:66` "ChaCAL control … identical arithmetic" | S assumed | ChaCAL's inverse is N (`[V-fetched]`) | S |
| `:169-171` `γ = 0` bitwise softmax; `γ ↑ 1` basin | S | `γ ↑ 1` is a limit; `I − P` is singular at every absorbing row | S |
| `:279-282` `‖Q‖_∞ = 1` blocks the one-file proof | non-causal chain | on the causal arm class `Q` is triangular | both (oracle vs arm) |

Mechanism for the whole table: `sec_proved.md` §2.8 (5) already states that `pow_card_eq_zero`
does not transfer to the diagonal-kept corner (`one_not_nilpotent`, `Nilpotent.lean:105`);
the design cites that warning at `:286-288` and then uses regime-N theorems on the softmax
corner three times. C8 of `THESIS_CORRECTIONS.md` calls this a theorem boundary; V-25.

---

## 3. The attacks, one per target, with the numbers

### 3.1 The `(1−γ)` factor and row sums (target 1)

`RUN` (`s = 8`, `γ = 0.6`, causal softmax `P`, row 3 set to `e_3`): bare `P(I−γP)^{-1}` row sums
`2.5` on every row `= 1/(1−γ)`; `Π_γ = (1−γ)P(I−γP)^{-1}` row sums `1.0` on every row including
the absorbing one; min entry `0.0`. DERIVED: for row-stochastic `P` (identity rows are
row-stochastic) `(I−γP)^{-1}𝟙 = Σ_k γ^k P^k 𝟙 = 𝟙/(1−γ)`, so `P(I−γP)^{-1}𝟙 = 𝟙/(1−γ)`; the
absorbing row `a` of `Π_γ` is `(1−γ)e_aᵀ(I−γP)^{-1} = e_aᵀ`. Both statements in §0.4 are true —
of two different operators. The document's object (`BRIEF.md:64-65`, `:90-91` `Δz`, the whole
price list) is the bare read; Bet C's `E[γ^{τ−1}]` (`:170`) is the normalised read
(`THESIS_NOTES.md:360-362`). `THESIS_CORRECTIONS_2.md:46-47` orders a choice; the document does
not make it. REPAIR: state `O = (1−γ)P(I−γP)^{-1}V` once, in §0.4, and carry the factor into the
certificate (`δ_Π = γ^{K+1}`, not `γ^{K+1}/(1−γ)`). Mechanism: V-17, V-23.

### 3.2 `E[γ^{τ−1}]` and its limit (target 2)

DERIVED (`THESIS_NOTES.md:369-374` re-checked): `z_i = Σ_t γ^t P_i(τ_k ≤ t) = E_i[γ^{τ_k}]/(1−γ)`
for all `i` (at `i ∈ 𝒜_k`, `τ_k = 0`, `z_i = 1/(1−γ)`); the normalised read
`O_i = (1−γ)Σ_j P_ij z_j = E_i[γ^{τ_k − 1}]` uses `τ_k ≥ 1` from `i ∈ T`. Two corrections.
(i) At `i ∈ 𝒜_k` the read is `(1−γ)z_i = 1`, not `γ^{−1}`; the document's `:170` drops the
`i ∈ T` restriction. (ii) `lim_{γ↑1} O_i = P_i(τ_k < ∞)` holds by monotone convergence with no
hypothesis, but the sentence "reads basin membership `argmax_k q^{(k)}`" presumes
`Σ_k q^{(k)} = 1`, which fails whenever an undeclared absorbing state exists (§3.11). KEEP
the identity; REPAIR the two sentences. Mechanism: V-25.

### 3.3 Invertibility of `I − Q` and `I − γP` (targets 3, 4, 5)

`RUN`: causal softmax `P`, `s = 8`, constraint `{3}`. BOS undeclared: `ρ(Q) = 1.000`,
`rank(I − Q) = 6 of 7`. BOS declared: `Q` is lower-triangular (`np.allclose(Q, tril(Q)) = True`),
`ρ(Q) = 0.536 = max_{i∈T} P_ii`. DERIVED: on the causal class every `i ≥ 1` has `P_ii < 1`
(softmax over `j ≤ i` with `exp > 0`), so `I − Q` is triangular with positive diagonal —
`lower_triangular_isUnit` (`READ sec_proved.md:530-531`, graded `[M]`) closes
`committor_is_resolvent_read (b)` in one line for the *arm's* class. The Perron route
(`isUnit_one_sub_of_perron`, `[S]`) is needed only for the environment chain, where
`SymmSupport` holds and `ρ(Q) = 0.9409` (`READ sec_proved.md:100-102`). The document's counter
(`:279-282`) conflates the two and predicts a failure on the class where the proof is trivial.
REPAIR: split I3(b) into (b-causal) `[M]` and (b-chain) `[S]`.

`RUN`: strictly lower `A` (`s = 8`), one row replaced by `e_r`: `rank(I − A) = 8`,
`rank(I − A') = 7`. Regime N at `γ = 1` is singular the moment an absorbing row is added, so
the survival route at `:393` ("keep regime N, exact without a dial") retires the boundary-row
mechanism together with `γ`. KILL that cell for component (e); it stands for BED-M only.
Mechanism: V-25 (`one_not_nilpotent` in the other direction), P-3.

### 3.4 Block-diagonal under a zero gate with absorbing rows (target 6)

DERIVED: if `P_ij = 0` for all `j < c ≤ i` then `I − γP` is block lower-triangular with a zero
off-diagonal block; an absorbing row `P_a = e_a` has no off-diagonal mass, so it is consistent
with the block structure whichever block `a` lies in; `resolvent_fromBlocks` (`READ
sec_proved.md:560-561`, `[M]`) gives the block inverse with `IsUnit` of each block from
`lower_triangular_isUnit`. That half transfers to regime S. The half that does not: the *iff*
"a zero gate ⇔ a zero block" is `pathProd_eq_zero_iff` on the complex path product
(`V16Domain.lean:129`, `READ sec_proved.md:245-246`), and `segmentation_blockdiag` is stated
for `StrictlyLower A` (`READ sec_proved.md:553-555`). On the softmax corner the gate enters
through `scan g` with `g = log m`, where a zero is `Real.log 0 = 0` junk
(`lean_log_junk_makes_the_scan_form_silently_false`, `V16Domain.lean:147`) and
`no_prefix_scan_represents_a_zero_gate` (`:165`) says no exponential prefix scan carries it.
So the F0 count the §3.7 prediction anchors on has no producing theorem on BED-S's operator;
it needs a `−∞` mask convention and a new target (`masked_softmax_blockdiag`). REPAIR: cite
`resolvent_fromBlocks` for the inverse and mark the iff `[S]` on regime S. Mechanism: P-11 (an
`[M]` tag on a declaration that does not cover the operator), V-25.

### 3.5 EMC (target 7)

`RUN` (`s = 8`, `γ = 0.5`, row `r = 4` redirected to `e_2`):
- cyclic dense row-stochastic `P`: "manipulate then equilibrate" `(I−γP')^{-1}V` versus
  "equilibrate then manipulate" (start from the old `z`, iterate `x ← V + γP'x` 2000 times):
  `max|diff| = 4.4e-16`;
- the same cyclic `P`, suffix-only re-solve (rows `≥ r` re-solved with rows `< r` frozen at old
  `z`): `max|diff| = 0.0678` against the full re-solve;
- causal softmax `P` **with** the diagonal (`ρ(P) = 1.000`, not nilpotent): suffix re-solve
  versus full `5.6e-17`.

DERIVED: for `γ‖P‖_∞ < 1` the map `x ↦ V + γP'x` is a contraction with one fixed point, so the
order of "manipulate" and "equilibrate" is immaterial for every `P`, cyclic or not — the
design's §3.10 "planted cyclic `P` reads `> 0.1`" is only the statement that a suffix-only
solve is not a solve when `P` is not triangular. Dash's EMC failure (`CITED [V] dash-2005-emc`,
Thm 1) concerns a *reduced* model whose equilibrated form hides feedback through the
manipulated variable; the shape's explicit linear fixed point has no such reduction. The
document's own counter (`:342-343`: "EMC is filed as a definition, not a proposition") is the
correct outcome, and it is decidable now, at 0 GPU-s, without the bet. The mechanism name
"by nilpotency" (`:345`) is also wrong: the suffix re-solve is exact by *triangularity* (the RUN
above on the non-nilpotent corner), which is why the document's own `1.33e-15` was obtained on
regime S. KILL the proposition; keep one remark citing `dash-2005-emc` and `momennejad-2017-sr`
/ `russek-2017-predictive` for why a *cached* resolvent needs the re-solve. Mechanism: V-3 (an
identity of the construction), V-24 (the "rejection region" is an artefact of a wrong
algorithm, not of the class).

### 3.6 Sherman–Morrison at an absorbing row (target 10)

`RUN`: with `M = (I−γP)^{-1}`, `u = newrow − P_r`, `Δz = γ(Me_r)(uᵀz)/(1 − γuᵀMe_r)`
(`sweep_safety.md:116-131` form) matches the re-solve to `4.4e-16` (redirect `r → e_2`) and
`5.6e-16` (absorbing `r → e_r`); `Δz[:r] = 0.0` exactly in both; the absorbing-row denominator
equals `(1−γ)(1 + γ(PM)_rr)` (`0.511918` both ways). DERIVED: `M = I + γPM` gives
`M_rr − (PM)_rr = 1 + (γ−1)(PM)_rr`, hence `1 − γ(M_rr − (PM)_rr) = (1−γ)(1 + γ(PM)_rr) > 0` for
`γ < 1` — no singularity when a row is made absorbing. KEEP the algebra. The scope claim at
`:220-223` fails: a move realised on the *key/value* side of token `j` changes the logit column
`j` in every row `i ≥ j` and every such row's normaliser, so `ΔP` is not rank one — `RUN`
`rank(ΔP) = 6 of 8` for a key-side edit of token 2 versus `1` for a query-side edit of token
4. The `≤ 2 × 1.041 ms` price holds only for row (query-side) surgery, which is what
`sec_beds.md:158-160` specifies; the document must say so, and price key-side moves as
`m` re-solves. The column `Me_r` itself costs one forward substitution, `s²/2` MACs per
candidate (or one `trsm` with `m` right-hand sides), not `O(sd)`; the `O(sd)` is the inner
product. Mechanism: M-8, P-8.

### 3.7 The `V = 1` plant (target 9)

`RUN`: `V ≡ 1`, `Δz` via the C6 identity `1.1e-16` (redirect) and `2.2e-16` (absorbing);
`uᵀ𝟙 = 4.2e-17 / 1.1e-16`. The identity `uᵀ𝟙 = 0` holds for any row-stochastic-to-row-stochastic
edit, so `Δz ≡ 0` is exact in reals; in floats it is `O(ε)` and reads `0.0` only on a code path
that rounds both solves identically (`THESIS_CORRECTIONS_2.md:42`). A bind that is passed by a
bitwise zero from one code path is V-3 in miniature; the bind is carried by the O(1) rejection
region (`1.127`, `:94`; `0.1096`, `THESIS_CORRECTIONS_2.md:43`), which stands. REPAIR the words
"exactly `0.0`" to "`≤ 1e-15`".

### 3.8 The Neumann tail (target 11)

`RUN` (`γ = 0.6`, absorbing row 3): `‖(I−γP)^{-1} − Σ_{k≤K}(γP)^k‖_∞` equals
`γ^{K+1}/(1−γ)` to 15 digits at `K = 1, 4, 8` (`0.9`, `0.1944`, `0.02519424`); rows scaled to
`1.5` read `5.9049` against `0.1944`. DERIVED: the tail is entrywise non-negative with row sums
`Σ_{k>K}γ^k`, equality of the induced `∞`-norm; for signed `P` with `‖P‖_∞ ≤ 1` the same tail is
bounded by the triangle inequality and equality can still occur (`RUN`: one sign-flipped row
still read `0.1944`), so "attained iff non-negative" would be false — the document does not
say it, `THESIS_NOTES.md:404` says `≤`, fine. KEEP. The object slip: `:245-247` binds the
certificate to "the shipped mask" and quotes a *truncation* residual (`6.13e-05` vs
`δ‖V‖_∞ ≈ 7.6e-05`); a mask that sparsifies `P` has error governed by dropped mass (F1), and the
union certificate lives at `:311` only. REPAIR §3.4 to name the truncated series, and refer
masks to §3.8's union bound. Mechanism: V-17, L-CERT.

### 3.9 DET-class and parallel depth (target 12)

DERIVED. Regime N: `(I − N)^{-1} = Π_{k=0}^{⌈log₂ s⌉−1}(I + N^{2^k})` exactly, `⌈log₂ s⌉`
dense matmuls of `O(s³)` work each; each matmul is depth `O(log s)` as a circuit, so the
circuit depth is `O(log² s)`, not `O(log s)`. Regime S: the same product truncated after `K`
squarings leaves the tail `γ^{2^K}/(1−γ)`, so accuracy `δ` needs `K = ⌈log₂(log(δ(1−γ))/log γ)⌉`
squarings — independent of `s`, `K = 5` at `γ = 0.5, δ = 10^{-6}` — again `O(s³)` per squaring
against the serial `s²d/2`. Cook's `DET` placement (`CITED [V] cook-1985-taxonomy`) is for
iterated matrix product and inversion; weighted DAG path-counting, which `(I−N)^{-1}` computes,
is in that class too, so the DET sentence (C4) stands as a class statement. KEEP with the
regime qualifier; the document inherits C4 at `:31-40` without it. Mechanism: V-17 (two units
of "depth"), P-8.

### 3.10 Transitive-closure support (target 13)

`RUN` (`s = 8`, `γ = 0.6`): support of `Π_γ` before and after making row 3 absorbing differs
on row `3` only; row 5's support is `[1 1 1 1 1 1 0 0]` in both. DERIVED: for non-negative `P`
the support of `Σ_t P^{t+1}` is the reachability relation of `supp(P)`; on a dense causal
softmax every `j ≤ i` is reachable in one step, so the closure is the causal triangle with or
without boundary rows; an absorbing row only removes its own off-diagonal support. The
prediction at `:208-210` ("change the *support* … mass that would have reached later
positions") is false on the base the bed uses, and its direction is inverted (walks descend to
`j ≤ i`, `THESIS_CORRECTIONS_2.md:44`). What boundary rows change is the *weight* on
`j < a` for rows `i > a` (mass redirected to `a`). KILL the support sentence; REPAIR to a
weight statement with a printed `‖Π_shape − Π_ChaCAL‖_∞` on downstream rows (the instrument's
B-E1 already does this, `READ design_instrument.md:88-91`). Mechanism: V-3, P-7.

### 3.11 Sum-to-one, BOS, F2 (targets 15, 3)

`RUN` (§3.3): BOS undeclared ⇒ `ρ(Q) = 1.0`, `I − Q` singular; declared ⇒ `Q` triangular. With
BOS in `𝒜_0` (instrument R6/E2, `READ design_instrument.md:58, :147`), every walk from a
transient `i` that avoids `𝒜_{k≥1}` is absorbed at BOS with probability one (position 0 is
reachable from every `i` by descent), so `q^{(0)} = 1 − Σ_{k≥1}q^{(k)}` by construction and the
goal set `𝒜_0 \ {0}` is reached only as a *stop before BOS*. Consequences: (i) the §3.3
prediction "the Chebyshev and `argmax q^{(0)}` forms disagree on a non-zero fraction" reduces
to `argmin Σ_{k≥1}` versus `argmin max_{k≥1}` — legitimate at `K = 2`, but "reach the goal" is
not a third object; (ii) the goal reachability census (`:143`) must print `q^{(BOS)}` separately
from `q^{(goal)}`, or the bed cannot tell "reached the goal" from "fell off the prompt". F2
(`R10`, `q = 0.0` exactly for constraints after the query) makes any such draw a discard by
guard 2, not a label value; the document's D-3 dial (`:241-242`) must vary constraint
positions *before* the query. REPAIR Bet B's object and §4's C5 row. Mechanism: V-12, V-25,
C4 (`sec_refuted.md:260-266`).

### 3.12 The discounted read against the undiscounted label (target 16 — the fatal one)

`RUN` (birth–death chain on `{0..8}`, `0` constraint, `8` goal, `p = ½` each way, `E_i[τ] =
7…16` on `T` — the `t* = 8` regime): committor `q_goal = 0.125…0.875`; the arm-side read
`E_i[γ^{τ}1_k]` (one `(1−γ)`-normalised solve with `V = 1_{𝒜_k}`) sums over `k` to
`[0.010, 0.268]` at `γ = 0.5`, `[0.302, 0.649]` at `0.9`, `[0.858, 0.937]` at `0.99`,
`[0.984, 0.993]` at `0.999`; the **normalised** read `E[γ^τ1_k]/E[γ^τ]` misses `q_goal` by
`0.308 / 0.134 / 0.019 / 0.002`; the raw read misses it by `0.678 / 0.381 / 0.073 / 0.008`.
DERIVED: `E[γ^{τ−1}] ≥ 0.99` at `τ = 8` needs `γ ≥ 0.99^{1/7} = 0.99857`, `1/(1−γ) = 697`.

Consequences for the document. Bet B (`:136-137`) predicts committor NRMSE below the one-hop
budget at *trained* `γ̂`; Bet C (`:186-189`) prints `1/(1−γ̂)` beside every `δ` and calls
`γ̂ → 1` the mirror kill. On a `t* = 8` bed an undiscounted label is readable only at
`1/(1−γ̂)` in the hundreds, so either B fails or C's certificate is declared vacuous; the §5
dependency paragraph (`:419-422`) sees the `γ̂ = 0` coupling and misses this one. The
instrument (`READ design_instrument.md:153-156`, R11) prints the "delay share" but keeps the
undiscounted label. Three repairs, any one of which reverses the KILL: (a) declare the label as
the discounted reach-avoid vector at a *registered* bed `γ_env` (Fisac's discounted safety
value, `CITED [V] fisac-2019-bridging`) and test whether `γ̂ → γ_env` — then Bet C's `γ̂` is the
recovery of a bed constant and must vary with the draw (D-3); (b) keep `q` and read at
`γ = 1` on the `Q`-block only (the oracle's own solve), dropping the `(I−γP)` operator over
absorbing rows and the certificate claim on that head; (c) keep both and re-price `t*` so that
`γ̂^{t*}` is near `1` — `t* = 2` at `γ̂ = 0.9` gives `0.81`, still `19 %` short. Mechanism:
V-17, D-2, D-3, and the document's own Ruling 10′ line.

### 3.13 "Readable in one resolvent solve" (target 17)

DERIVED: `SymmSupport Q` with one off-diagonal positive entry cannot be triangular under any
permutation (both `(i,j)` and `(j,i)` would sit in the triangle); the arm's `(I−γP)^{-1}` with
causal `P` is triangular in position order. So on BED-S's environment chain as `sec_beds.md`
§6.C.1 specifies it (Rips / `bed_1` graphs, cyclic, `ρ(Q) = 0.9409`, `Q^{11} ≠ 0`), the arm's
resolvent can never *equal* the oracle's; it can only approximate a cyclic-chain committor by
a DAG-with-self-loop path sum over token positions. That is the intended D-2 separation, but
the sentences "the reach-avoid vector is readable in one resolvent solve" (`:157-158`) and
"read jointly by one resolvent solve" (`:114-115`) describe the oracle, not the arm. If instead
the bed's chain *is* the arm's causal `P` over positions (the instrument's R6–R10 reading),
the oracle is the arm's own operator class and C1 of `sec_refuted.md` (`:233-240`) applies.
REPAIR: state which chain is the environment and print `‖P̂ − P_env‖` only when the two are
comparable. Mechanism: D-2.

### 3.14 Bet A statistics (target 18)

`READ :96-98`: the field cosine is paired "by McNemar"; McNemar (`CITED [V]
mcnemar-1947-correlated`) tests paired binary outcomes and applies to the sign column, not to a
continuous cosine. At `N = 8` the finest paired-binary `p` is `2^{-7} = 0.0078`
(`READ sec_measured.md:268`), so the McNemar column is a sign test. The MDE `0.039827`
(`READ sec_beds.md:310`) is the `n = 8` cell for a *NRMSE* paired sd on BED-M; applied to a
cosine difference it is V-17. REPAIR: paired `t`/Wilcoxon on the cosine with its own realised
sd; McNemar on the sign column only.

### 3.15 The ChaCAL diagonal (target 19)

`CITED [V-fetched] fagnou-2024-chacal` (`arxiv.org/html/2410.05565`, this session): Eq. 5
`Y = (1−γ)·A(I−γA)^{-1}V`; "in practice we remove the diagonal part of the adjacency matrix in
the inverse … as it is trivial and improves learning"; `γ = 0` "reverting … to Equation 2"; "a
triangular system … does not require explicitly computing the inverse". DERIVED: with
`A_s = A − diag(A)`, `(I − γA_s)^{-1}` is a finite sum (`A_s^s = 0`, `occupancy_is_exact_inverse`)
and `(I−γA_s)^{-1}𝟙 = Σ_k γ^k A_s^k 𝟙 ≠ 𝟙/(1−γ)` because `A_s𝟙 = 𝟙 − diag(A) < 𝟙`; hence
`(1−γ)A(I−γA_s)^{-1}` is sub-stochastic and `sweep_resolvent.md:197-201`'s "ChaCAL's read is
row-stochastic" is a statement about the diagonal-kept operator, not the published one. The
design's `:66` "identical arithmetic", `:147-153` "ChaCAL with the same `γ`", and `:212-214`
"ChaCAL-with-sink matches within the TOST margin" all need the convention fixed: ChaCAL-as-
published is a regime-N solve under a regime-S outer read, and the "same `γ`" control differs
from the shape by the diagonal at every `γ > 0`. `sweep_expressivity.md:356` carried this as
`[U]`; it is now `[V]`. REPAIR: two control arms, ChaCAL-published (`A_s` in the inverse) and
ChaCAL-diag (the shape without boundary rows), and say which one the TOST is against.
Mechanism: P-10, V-24, R-SKY.

### 3.16 Arithmetic re-derived (target 20) — all KEEP

`0.010162 + 0.001041 = 0.011203`; `× 150 = 1.680`; `× 8 + 4.0 = 17.4`; `0.010162 × 150 = 1.524`,
`× 8 + 4 = 16.2`; `17.4 + 16.2 = 33.6 ≈ 34`; `0.094107 × 150 = 14.12`, `× 8 + 4 = 116.9`,
`+ 16.2 = 133.1`; `70 × 3.204 + 4 = 228.3`; `17.4 + 34 + 228 + 2.6 = 282 s = 4.7 min`;
`1024 × 2.514 ms = 2.57 s`; `0.5^{17}/0.5 = 1.526e-05`; `1.524 × 5 = 7.62`;
`1200³/3 = 5.76e8` (MACs, not FLOPs under the house convention — `sec_beds.md:339` says
"flops"; minor); InfSA cell `0.010162 − 0.001473 + 0.025409 = 0.034098`, `× 150 = 5.11`;
`⌊log₂ 8⌋ + 2 = 5`; Fano M11 instances `1 − (1 + ln 2)/ln 8 = 0.1858`,
`1 − (2 + ln 2)/ln 32 = 0.2229` (arithmetic right, floor form loose per §3.10).

---

## 4. What would reverse each KILL

| KILL | reverses if |
|---|---|
| Bets B+C pair (16) | the label is re-declared discounted at a registered `γ_env`, or the committor head is read at `γ = 1` on the `Q`-block with the certificate claim withdrawn for that head |
| Fano floor (14) | the floor is printed as `1 − max_a π̂(a*)` from the realised class distribution, with tight Fano `(H(a*) − I − ln 2)/ln(m−1)` only where `I(X_{≤k}; a*) > 0` |
| EMC (7) | a *reduced* (equilibrated-away) model of the bed is exhibited on which Dash's Thm 1 hypotheses hold; otherwise it is a remark |
| Support sentence (13) | the base has exact zeros (a masked or F0-segmented `P`) so that boundary rows can cut reachability; on a dense softmax base it cannot |
| Regime N survival for (e) (5) | an absorbing row is realised as a value-zero column sink plus `γ < 1`, i.e. regime S — which is not the cell as written |

---

## 5. Limits

All `RUN` numbers are one draw, `s ≤ 9`, float64 numpy on this CPU, identity checks without
intervals; none is a capability reading. The ChaCAL quotation is a summariser's rendering of the
HTML page fetched this session (three phrases under 15 words each); the equation number and the
diagonal sentence should be re-read against the PDF before the paper prints them. The
birth–death chain of §3.12 is a stand-in for BED-S, whose chain does not exist; the `t* = 8`
regime was matched by `E[τ]`, not by the bed's tolerance dial. The DET remarks are class
statements, not timings. No code file was written; no git write was made; one external fetch.
