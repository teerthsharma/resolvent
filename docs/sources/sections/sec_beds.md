# 6. Beds, labels, floors, metrics and the pricing rule

*Section owner: VENUS (IRENE — competing predictions, floors, metrics). Repository
`<repo root>` at `207e7b9`. Evidence classes: `RUN` (executed
this session, this box, torch 2.5.1, float32 unless stated), `READ path:line`, `CITED`,
`DERIVED`. Every design element names the `MISTAKES.md` mechanism it is built against.*

## 6.0 What this section decides

The record's verdict on itself (`MATHEMATICS.md:27-43`, READ) is that every bed asked
for a scalar at one position, which is the regime in which one softmax layer is
asymptotically Bayes-optimal (`arXiv:2410.01537`, CITED [U], caveats at
`MATHEMATICS.md` §17.5). The shape of §4 returns a whole configuration `z`, its
displacement `Δz` under `do()`, and a decision over moves scored against `K` absorbing
constraint sets. None of the four registered beds carries such a label. This section
(A) audits each bed as it stands, (B) fixes the vector label and the state metric that
the thesis needs, (C) designs BED-S, (D) fixes the price of a cell and the seed count a
number needs before it may be written, and (E) files three predictions with counters.

## 6.A The existing beds, one row each

Column conventions. *Label* is what the arm is trained to emit. *Oracle* is what
computes it. *Floor* is the lowest error any predictor can reach; the column says
whether the record's named floor is one. *Skyline* is the strongest fellow approximator
the bed must be scored against (`CEQ_V16_CONTRACT.md` R-SKY). *Decides* is what a
reading on that bed can and cannot say about the shape.

| bed | generator | label | oracle | information floor | native skyline | what it decides about the shape |
|---|---|---|---|---|---|---|
| **BED-M** chain (E1/E3) | `make_equilibrium_batch`, `scale/negation_scope.py:354-433` (READ): sub-diagonal `A[i,i-1]=a_i` Rademacher, drivers `b_i ~ N(0,1)`, `a` zeroed at and before `head = s-1-t*`, `b[s-1]=0` | **scalar** `z*_{s-1}` (`:286-304`), `N(0,t*)` exactly; RUN this session `n=2048,s=64`: Var(y) `2.0186 / 7.8994 / 31.7124` at `t*=2/8/32` | exact forward scan `z ← a_i z + b_i` (`:300-304`); equals the last row of `(I−A)^{-1}b` to `6.2e-15` (I2, brief §1) | **exact oracle at `0.0`** is the only information floor (`V20_R15_THEORY_TABLE.md:87-91`, READ). `floor₁ = √((t*−1)/t*)` is a one-hop **capability threshold**, violated by `13 of 40` banked cells (`:69-83`, READ) — not a floor. The k-hop truncation law `√((t*−k)/t*)` (`:307-331`) is a *budget* ceiling; RUN `k=0,1,2` at `t*=2`: `1.000058 / 0.712039 / 0.0`, at `t*=8`: `1.000094 / 0.936621 / 0.874109`, at `t*=32`: `1.000029 / 0.984178 / 0.967377` against closed form `1.0 / 0.707107 / 0.0`, `1.0 / 0.935414 / 0.866025`, `1.0 / 0.984251 / 0.968246` | a softmax stack of depth `⌊log₂ t*⌋+2` = `3 / 5 / 7` at `t*=2/8/32` (DERIVED from `arXiv:2402.09268` Thm 4.2, CITED [U]) computes every hop; one depth-1 softmax reads at the 2-hop ceiling `0.860972` vs `0.866025` and memorises (`MATHEMATICS.md:299-306`, READ) | **Cannot decide the shape.** D-1: scalar at one position, softmax's proven regime. D-2: the label is the arm's own resolvent (I2 max abs `0.0`), so `shape − corner-3` on this bed is a reproduction check, never a capability. It stays as the **parity bind** (I1, `γ=0` bitwise) and as the truncation-law must-fire |
| **BED-M** consequence (E2) | `make_consequence_batch`, `:803-850` (READ): 6-player potential game in tokens `0..5`, intervening token at `f=s−1−d`, `τ = safe_tau(margin 1.25)` per example, Lipschitz `0.8` enforced with a raise | **scalar** contrast `z*_{5}(do s₀:=v) − z*_{5}(do s₀:=1−v)` (`:735-767`); label sd `≈0.062` (`:276-277`, READ); RUN `E2_T_STAR = 31`, `E_LIPSCHITZ = 0.8` | 200 damped best-response sweeps with player 0 **clamped** — graph surgery, not a bias nudge (`:718-732`, READ) | exact oracle at `0.0`; shock-blind reading is identically `≥ 1.0` (`:752-756`); truncation `L^k`: `1.000067 / 0.410387 / … / 0.001017` (`MATHEMATICS.md:274-276`, READ); measured crossing of `1e-3` at `k=16` against the bound's `31` (`:269-270`, READ) | depth-1 softmax already reads consequence fidelity `0.807843` CP `[0.754044, 0.854329]` on the sign task (`MATHEMATICS.md:396-406`, READ) — the sign of one scalar is not the ground | **The only interventional channel in the tree**, and it is a scalar coordinate of one player. It decides nothing about `Δz` as a field (§6.B) but its two design moves — mirror contrast (flipper exactly `2.0` by antisymmetry, `:853-856`) and the raw-label kill (`0.194150` shock-blind, `:752-755`) — are the guards BED-S inherits |
| **BED-M** executed programs | `ceq/corpus.py:build`, `:88-130` (READ): 12-token programs, `VOCAB=17`, CPython is the oracle; held-out cell `('-',1)` composes two sign flips never seen together | **scalar** integer `acc`; intervention = one-token edit re-executed (`do()`, not a correlation, `:14-22`) | `exec` on the source (`:60-65`), cross-checked by `execute_subprocess` (`:68-72`) | NOT MEASURED — no cell on this corpus is journalled at HEAD with a floor. The record's own warning (`:95-99`, READ): the literal-split version measured embedding coverage, `attention 0.954 / appnp 0.779 / signed 0.777` near predict-the-mean | any arm whose embedding rows for the held-out cell were trained | decides the **negation-gate** composition question (`if flag: acc = −acc`) that the signed operator was built for and lost; the shape moves that veto into boundary conditions, so this bed is the **planted negative for the boundary mechanism**, not a capability bed |
| **BED-K** delay / power-law | `ceq/beds/bed_k.py:build`, `:194-234` (READ): `z = K b`, `K` delay (`K[i,i−d]=1`) or GL fractional-integral weights `ψ_k`, `H∈(0.5,1)` refused outside (`:92-112`) | **vector over positions**, but each coordinate is a *linear* functional of `b`: no joint determination | `K` itself; `bump(p) = K[:,p]` exact (`:256-269`), cross-checked by central difference on `rebuild` (`:271-279`) — two routes, V-3 | Hankel ceiling `1/d` for one state (`CEQ_V20_R15_CONTRACT.md:116-117`, READ): best-one-state NRMSE `err₁ = 0.9746794345` (`V20_R15_LEAP_LEDGER.md:323`, READ) `= √(1−1/20)` at `d=20` (DERIVED, agrees to 10 digits) | **attention-native by construction**: `hard_delay_attention` reproduces the delay to `(n−1)e^{−45}` (`:311-336`, READ); scan-blind by theorem (`V15Kernel.first_order_cannot_delay`, brief §2) | decides that the shape is **not worse than one head at a fixed offset** — the R-SKY control; a resolvent with `γ>0` has no delay advantage here and must not claim one. **No cell of BED-K's shape has ever been run** (`V20_R15_LEAP_LEDGER.md:323`; `V20_R15_THEORY_TABLE.md:304-306`, READ) |
| **BED-1** committor | `ceq/beds/bed_1.py:build`, `:123-185` (READ): 11-node landscape, basins `A,B,C`, saddles `S_lo(1.0)`, `S_hi0..4(2.0)`, `S_ac,S_cb(1.5)`, Metropolis rates, `T` the dial | **vector** `q ∈ [0,1]^{11}` (splitting probability) plus a **categorical** channel label `lo/hi/trap` by reactive flux (`:228-241`) | Dirichlet solve `(Lq)_int = 0, q|_A=0, q|_B=1` (`:188-198`); RUN `T=0.5`: `q = [0, 1, .5, .5, .5, .5, .5, .5, .5, .25, .75]` = the closed form `(0, ¼, ½, ¾, 1)` on the node set; `harmonic_residual = 4.16e-17`; `T* = 0.6213349345596119`; `label_by_committor = lo = label_by_barrier` below `T*` | exact oracle at `0.0`; **single absorbing target makes the label constant to `1.11e-14`** (V-12, `MISTAKES.md:189-199`) — with `|A|=|B|=1` the label here is fixed by the landscape, and varies only through `T` and jitter | any solver of a 9×9 linear system; a depth-1 softmax cannot form the 3-hop path `A→S_ac→C→S_cb→B` | the **oracle BED-S reuses** (I3); as a capability bed it has too few draws — the landscape is one fixed graph. No arm has run on it (registered at `ceq/kdata.py`, cells `0`) |
| **E4′** harmonic on Rips | `scale/e4_harmonic.py:case_graph / absorbing_chain`, `:103-176` (READ): S² Rips at degree `4.25`, one bridge joins the two largest components (`605 / 597` nodes), absorbing at the bridge endpoints | **scalar per node**: column of `B = N R` toward the first endpoint; sd `0.499989` mean `0.503333` at `kill=0`, collapsing to sd `0.038445` mean `0.004334` at `kill=1/16` (`:74-89`, `:425-434`, READ) | `np.linalg.solve(I−Q, r)` (`:184-190`) **and** the Kirchhoff forest ratio (`scale/kirchhoff.py:1-90`); agreement `9.636736e-14` at `1200` transient nodes, tolerance `1e-10`, planted off-by-one rejected at `≥1.749951e-01` (`MATHEMATICS.md:585-604`, READ) | exact oracle at `0.0`; hop ladder against the fixed point is **flat** — a 32-step reading still reads `1.402462` (`:350-354`, READ) because `ρ(Q)=0.999974225` and the Cheeger floor is `t_rel ≥ vol(S)/2 = 1372.50` (`MATHEMATICS.md:455-461`, READ); local-feature decoder reads the fixed point at `0.973819` (above `FAIL_BAR=0.9`, `scale/rips_gate.py:61`) | none reachable: no finite hop budget approaches the label | decides that **the bottleneck that makes a label global is the quantity that puts `t*` out of reach** — the design constraint BED-S's difficulty dial must respect (§6.C.6) |

Two rows the table must not be read without. First, the Q6 census: `0 of 40` banked
cells journal a prediction vector, histogram, quantile or density
(`V20_R15_THEORY_TABLE.md:213`, `:223`, READ), so no state metric has ever been computed on a
real prediction in this repository. Second, `vector_readout` — the `[n, s]` readout at
zero new parameters — exists at `scale/m3_quintuple.py:368` and `:483` and defaults to
`False` (`results/r9_maths_survey.md:190-200`, READ); the plumbing for a vector label is
built, and the label is what is missing.

## 6.B The vector-valued label and the state metric (Q6)

### 6.B.1 The label

The thesis needs three objects, and each is a vector or a field, never a coordinate:

```
    z*        ∈ R^{s×d}    the fixed point   (I − γP) z* = V            [the shape]
    Δz(a)     ∈ R^{s×d}    z*(do a) − z*                                 [the consequence]
    q^{(k)}(a)∈ [0,1]^{n_T} committor into constraint set k under move a  [the constraint reading]
```

`equilibrium_oracle` already computes all of `z*` and discards all but one entry
(`MATHEMATICS.md:33`, READ). The label change on BED-M is therefore
`return z` in place of `return z[:, s-1]` — but D-1's amendment is explicit that this
voids the `PUBLISHED_SOFTMAX_8192` reproduction gate and must live in a separate lane
with its own journal (`MISTAKES.md:701-708`, READ). Mechanism: **D-1** (the scalar label
is the proven-optimum regime; a vector label leaves it) and **M-1** (train and eval must
see the same readout shape — the lane rule).

### 6.B.2 Why marginal W1 cannot score it

The record's Q6 default was `W1` between predicted and oracle state distributions
(`CEQ_V20_R15_CONTRACT.md:235-238`, READ). Two findings kill it as a regression metric,
and both were re-run this session (`tests/jupiter/test_v20_r15_it11_q6_oracle.py`, RUN:
`5 passed in 2.86s`):

1. **Permutation blindness.** A predictor returning the oracle's own values in the wrong
   order scores `W1 = 0.0` exactly while `NRMSE = 1.421901019003236` — `√2 =
   1.4142135623730951` plus `7.687e-03` — on `equilibrium_oracle` at `n=4096, s=64,
   d=24, t*=2, seed=4096` (`V20_R15_THEORY_TABLE.md:221`, READ). That is worse than
   predict-the-mean. `√2` is the DERIVED value: two independent `N(0,t*)` draws differ
   by `N(0, 2t*)`, and NRMSE normalises by `√t*`.
2. **Ranking inversion.** Against `oracle + 0.1σ` noise, `W1` prefers the permutation
   (`0.0` vs `0.012049103155732155`) while `NRMSE` prefers the noise (`1.421901` vs
   `0.098296619951725`), by `14.465410797679917×` (`:221`, READ).

The mechanism is V-26 in metric form: a marginal (the pooled distribution of values)
standing in for a joint (which value sits at which position). Any metric invariant under
a symmetry the label does not have will be defeated by that symmetry. This rules out, in
order: pooled `W1` (invariant under permutation of positions); Procrustes distance
(invariant under orthogonal rotation of the coordinate frame — `z*`'s coordinates are
labelled positions, not a point cloud); OT without a position cost (the same defect as
pooled `W1`). Mechanism: **V-26**, **L-14**.

### 6.B.3 The default, and why

**Default state metric for regression heads (z\*, Δz):** the position-matched
per-coordinate NRMSE vector,

```
    e_i = RMSE_i / sd_i(y)   for each coordinate i,     reported as the vector, its max and its mean,
```

plus one **joint-consistency term** that no per-coordinate metric can see:

```
    r(ẑ) = ‖ (I − γP_env) ẑ − V ‖_∞ / ‖V‖_∞        the harmonic residual of the prediction
```

computed with the *environment's* operator, which the bed owns and the arm never sees
(D-2). `r` is `0` iff `ẑ` is the fixed point; it is not permutation-blind (permuting `z*`
changes `(I−γP)z`), and it is the one number that scores "jointly determined" rather
than "right on average" — the claim `MATHEMATICS.md:106-127` records as UNTESTED. For
BED-S's committor vectors it is exactly `bed_1.harmonic_residual` (`:201-206`), which
reads `4.16e-17` on the oracle (RUN) and is therefore a calibrated zero. DERIVED; the
residual is computable on every draw because the oracle is exact.

Reasons for the choice, each against a mechanism: (i) `e_i = 1.0` at every coordinate
for the mean predictor, so the harness's `1.0` bar and its 0-step RED gate
(`GATE_TOL = 1e-3`, `COSTS.md:137-139`, READ) transfer unchanged — **V-10** (a bar
satisfied by construction is avoided because the bar's value is the same identity it
always was); (ii) the coordinate vector is what `nrmse` already computes when handed
`[n, m]` labels — `calibrate_bar`'s `payload_only` clause was rewritten to broadcast a
`[n]` payload against a `[n, m]` label (`scale/negation_scope.py:1496-1499`, READ), so the
bar is already vector-ready; (iii) the residual term is refused as a *training* loss
and used only as a *score*, because a loss that reads the environment's `P` is a leak
(**D-2**).

**For the displacement field Δz:** the same coordinate vector on `Δz`, and additionally
the field cosine `⟨Δẑ, Δz⟩ / (‖Δẑ‖‖Δz‖)` and the magnitude ratio `‖Δẑ‖/‖Δz‖`, which
generalise `MATHEMATICS.md` §6's sign fidelity and slope from one scalar to the field.
The sign task stays as a column because softmax already reads it at `0.807843`
(`MATHEMATICS.md:396-406`, READ) and a field metric that did not reduce to it at `s=1` would be a new
unit (**V-17**).

**For probability-valued heads (committors):** scored in the Fisher–Rao coordinate
`φ(p) = 2 arcsin √p` (`MATHEMATICS.md:69-104`, READ), declared before the run; the record
measured the Euclidean/Fisher–Rao ratio at `1.418962` at rung 8 and `1.209472` at rung
32, so choosing the readout after the curve would be choosing the answer (**M-2**).

**For distributional heads, if any arm ships one:** CRPS. It is proper, it reduces to
absolute error for a point mass, and it is what L-13 names as the field that would say
how such a head is scored (`V20_R15_LEAP_LEDGER.md:130`, READ). No arm in the tree has a
distributional head (`0 of 40`, no `*logits`/`*prob*` parameter, `V20_R15_THEORY_TABLE.md:213-216`),
so CRPS is registered as the metric and not yet computed — **P-1** guard: no number is
quoted for it.

## 6.C BED-S — the safest move under K constraints

The author's second message names the capability: *predict safest move under multiple
constraint*. BED-S is that, on a latent chain, with the record's own oracle.

### 6.C.1 Generator

A draw is a triple `(G, 𝒜, M)`:

- `G`: a random walk on a connected graph of `n` nodes with transient set `T` and `K ≥ 2`
  disjoint absorbing sets `𝒜_1..𝒜_K` (rows of `P` on `𝒜_k` are identity). The substrate is
  the E4′ Rips family (`ceq/rips.py`, S² points, geodesic Rips edges) at a size where
  `t*` lands (§6.C.6), or `bed_1`'s energy-graph family jittered (`bed_1.build(jitter)`,
  `:120-160`) when a landscape with named basins is wanted. `K` absorbing sets replace
  `bed_1`'s two.
- `M`: `m` candidate moves, each a `do()` on the context: move `a` **clamps** one
  transient node's outgoing row (a redirect: `P_a[v_a, :] ← e_{u_a}`), or deletes an edge,
  exactly the graph surgery `_br_sweeps` performs by clamping player 0
  (`scale/negation_scope.py:718-732`, READ) and that the maths survey names as SCM
  surgery reusing `fixed_point` verbatim (`results/r9_maths_survey.md:313`, READ).
- `x`: the token list the arm sees — node tokens (id, degree, membership flag *of the
  absorbing sets only*), edge tokens (endpoint ids), move tokens (move id, `v_a`, `u_a`),
  and a query token carrying the start node `s₀` and nothing else. **The transition matrix
  never appears in `x`** — the arm must assemble `P` from edge tokens (§6.C.4).

### 6.C.2 Label

```
    q(a) ∈ [0,1]^K,   q_k(a) = P_{s₀}( absorbed in 𝒜_k | P_a )      for a = 1..m
    a*   = argmin_a  max_k q_k(a)                                     the safest move
```

Two heads: the `[m, K]` committor tensor (regression, scored by §6.B.3 in the Fisher–Rao
coordinate, plus the harmonic residual) and the argmin (classification over `m`, scored
by accuracy with a Clopper–Pearson interval, as `MATHEMATICS.md` §6 does). The weighted
form `argmin_a Σ_k w_k q_k(a)` and the lexicographic form are the same bed with a
different reduction and are registered as columns, not as separate beds (**V-1**: one
corpus, one registry key).

### 6.C.3 Oracle — the Dirichlet solve the record owns

For each `(a, k)`: `(I − Q_a) q^{(k)}(a) = R_{a,k} 1` — `bed_1.committor` with `A` set to
`∪_{j≠k} 𝒜_j` and `B` set to `𝒜_k` (`ceq/beds/bed_1.py:188-198`, READ), `m·K` solves of a
`|T|×|T|` system per draw. I3 (brief §1) is the identity that this is the resolvent read
with absorbing rows. The second route, `scale/kirchhoff.py:harmonic_measure`, is written
for two single-node boundaries `a, b`; extending the grounded Laplacian to a multi-node
boundary set is one index-map change, and until that is in the tree the instrument law
`assert_oracles_agree` applies only to `K=2, |𝒜_k|=1` draws — recorded, not assumed
(**V-3**: two routes must not be the same code; **P-4**: the extension is claimed
scaffolding until it exists). `Σ_k q_k(a) = 1` on every transient node is the
conservation row (**V-23**: printed with its residual, including when it fails).

### 6.C.4 The four guards, each against a named mechanism

1. **Zero-hop control (V-10, D-5).** The record earned this the hard way: BED-M's first
   builder gave the query token a driver, `1/(t*+1)` of the label was legible at zero
   hops, and the harness's own 0-step RED gate aborted three of five rungs; `b[s-1] = 0`
   restored the property (`scale/negation_scope.py:399-415`, READ; `MATHEMATICS.md:258-264`).
   BED-S's analogue: the query token carries `s₀` only, and `s₀` is drawn uniformly from
   `T` **independently of the argmin** — checked at construction by `I(s₀; a*) = 0` up to
   the plug-in estimate's sampling error, and by the untrained arm reading argmin
   accuracy within `1/m ± CP` and committor NRMSE `≥ 1 − GATE_TOL` (the 0-step RED gate,
   `COSTS.md:137`). A move token that named its own `q` would be the same leak one field
   over; move tokens carry `(v_a, u_a)` and nothing derived from the solve.
2. **Constant-label guard (V-8, V-12).** With a single absorbing set the committor is
   identically `1` (V-12's `1.11e-14`); with `K ≥ 2` it can still degenerate when a move
   disconnects `s₀` from all but one set. The builder prints, per draw batch: the
   per-coordinate label sd of `q` (must exceed `0`), the argmin class balance over `m`
   (each class non-empty, none above `1 − 1/m + tol`), and the **discard count** — draws
   rejected because the argmin is tied to within `1e-9` or because some `q_k` is
   constant across moves. The discard fraction is a property of the corpus and is
   journalled beside every reading. `nrmse` returns `nan` on a constant label and
   `nan >= 1.0` is `False` in Python (`MISTAKES.md:149-153`, READ), so the sd print is not
   optional.
3. **Leak guard (D-2).** The oracle runs on the environment chain `P_a`; the arm sees
   edge tokens. The planted positive control: hand a linear decoder the rows of `Q_a`
   as features and require it to read the committor below `PASS_BAR = 0.5`
   (`scale/rips_gate.py:60`) — the leak detector must fire when the leak is planted.
   The planted negative: the same decoder on the strictly-local features E4′ uses
   (degree, ball sizes to radius 5, absorbing-endpoint-in-ball flags —
   `scale/e4_harmonic.py:220-244`, READ) must read above `FAIL_BAR = 0.9`, as it does on
   E4′'s fixed point at `0.973819`. Both halves fire or the bed is not admitted
   (**V-24**: a bind with an empty rejection region).
4. **Registration with admission (D-4).** BED-S enters `M3_TASKS` with its `E_T_STAR`
   entry, its flipper-dependence closed form, and its feature function for
   `calibrate_bar` clause 5, on the same commit — `calibrate_bar` is run through the
   BED-S hooks before any arm is trained, and the trained positive control is handed the
   oracle's *inputs* (edge list, move, `s₀`), never a partial answer (`:789-800`, READ).

### 6.C.5 Information floors (L-FLOOR)

Every BED-S number ships beside two floors, and the section says which is which,
because the record spent a round grading against a threshold it had called a floor
(`V20_R15_THEORY_TABLE.md:69-91`).

**Committor vector.** The information floor is the exact oracle at `0.0`. The
*budget* ceilings are the hop readings `z_k = Q_a z_{k−1} + R` (`e4_harmonic.hop_reading`,
`:194-199`) scored against the fixed point, one per `k` — the analogue of BED-M's
`√((t*−k)/t*)`, but numerical rather than closed-form because the chain does not
terminate (`OracleSeparation.truncation_never_exact`, brief §2). They are printed at
construction, per draw batch, before any arm runs (**M-2**).

**Argmin.** Fano (`CEQ_V20_R15_CONTRACT.md:226-229`, READ):

```
    P(err) ≥ 1 − ( I(X_view ; a*) + ln 2 ) / ln m
```

With the full context `X` and a deterministic oracle, `I(X; a*) = H(a*)` and the bound is
`≤ 0` — Fano says nothing about an arm that sees everything. The informative floors are
Fano against **restricted views**: `X_{≤k}` = the tokens within `k` hops of `s₀` and the
move endpoints, which is exactly what a `k`-hop arm can read. `I(X_{≤k}; a*)` is
computable because the oracle is exact: sample the latent chain conditional on the
visible window, re-solve, and estimate the plug-in mutual information. At `k=0`
guard 1 forces `I = 0`, so the zero-hop Fano floor is `1 − ln 2 / ln m` — RUN:
`0.0` at `m=2`, `0.5` at `m=4`, `0.6667` at `m=8`, `0.75` at `m=16`. The record's own two
M11 instances reproduce under the same formula (`0.1858` at `m=8, I=1`; `0.2229` at
`m=32, I=2`, RUN, against the annex's `0.19` and `0.22`). **`m = 2` is refused**: Fano is
vacuous there, so no floor could be printed (**V-10**). Default `m = 8`.

### 6.C.6 The difficulty dial

Four knobs, and only the first is the registered dial (**D-3**: a dial that does not
vary describes nothing):

- **`t*`** — the smallest `k` at which `NRMSE(z_k, z*) < 1e-3`, E2's convention
  (`E2_DIAL_TOL`, `scale/negation_scope.py:264`), labelled a *tolerance* dial, not a hop
  count. It is set by the chain's geometry, and the record has already measured the
  constraint: a one-edge bridge has conductance `1/vol(S)`, so `t_rel ≥ vol(S)/2`
  (`e4_harmonic.cheeger_t_rel_floor`, `:247-263`, READ; `1372.50` on the shipped case).
  A `t*` in `{2, 8, 32}` therefore needs a bottleneck cut of `c` edges with
  `vol(S)/(2c)` of that order — small components or multi-edge cuts, never a kill rate,
  which collapses the label (`sd 0.499989 → 0.038445`, `:425-434`, READ).
- **The rate is `ρ(Q_a)`, not `λ₂`.** The ergodic chain's second eigenvalue is a
  different number from the transient block's Perron root — measured `0.9964078857`
  against `0.9984623637` on `LargestJoin_S2Rips_1024` (`PRIOR_ART.md:703-707`, READ) — and
  with `K` absorbing sets the full `P` has eigenvalue `1` with multiplicity `≥ K`
  (`e4_harmonic.spectral` docstring, `:207-218`, READ; RUN `ρ(P) = 1.0000000000000013`
  on `bed_1`). The pre-asymptotic rungs do not decay at `ρ^{t}` either
  (`MATHEMATICS.md:437-449`), so `t*` is read off the hop ladder, never predicted from a
  spectrum. Mechanism: **P-8** (a bound stated as a price).
- **`K`** and **`m`**: `K ∈ {2, 3, 4}`, `m ∈ {4, 8, 16}`; `m` moves the Fano floor
  (§6.C.5), `K` moves the number of solves and the conservation row.
- **Softmax skyline** at depth `⌊log₂ t*⌋ + 2` = `3 / 5 / 7` for `t* = 2 / 8 / 32`
  (DERIVED from `arXiv:2402.09268` Thm 4.2; the `Ω(log k)` lower bound is conditional on
  the 1-vs-2-cycle conjecture, Cor. 4.3, and is cited as conditional). The skyline is the
  honest control at *unmatched* depth; the shape's claim is at matched depth and
  parameters (`4,769`, `CEQ_V20_R15_CONTRACT.md:119`).

## 6.D The pricing rule

1. **`N = 8` minimum, deduplicated by seed.** `REQUIRED_SEEDS = 8`
   (`scale/it11_verdict.py:59`, READ); `verdict()` refuses below it rather than reporting
   a CI over the seeds it has, and counts distinct seeds, not rows, because the journal
   carries bit-identical duplicates (`:27-38`, READ). At `N = 5` the finest two-sided
   `p` is `0.0625` and "the CI excludes zero" is a sign test (M-9, `MISTAKES.md:580-590`,
   READ). Mechanism: **M-9**, **V-5**.
2. **Minimum detectable effect, paired, `α = 0.05` two-sided, power `0.80`.** The
   record's exact noncentral-`t` column at `n = 5 / 10 / 20` (`MATHEMATICS.md:979-993`,
   READ) was reproduced this session by the same guarded bisection (RUN, all six record
   entries agree to the sixth decimal), and the `n = 8` and `n = 16` columns were read
   off the same solve:

   | paired sd | provenance | n=5 | **n=8** | n=10 | **n=16** | n=20 |
   |---|---|---|---|---|---|---|
   | 0.010101 | BED-M `t*=2, n=2048` seed sd at `N=8` (`MISTAKES.md:1096`) | 0.016990 | **0.011677** | 0.010061 | **0.007570** | 0.006671 |
   | 0.022345 | record row 1 | 0.037584 | **0.025832** | 0.022256 | **0.016747** | 0.014758 |
   | 0.034451 | record row 2 | 0.057946 | **0.039827** | 0.034313 | **0.025820** | 0.022753 |
   | 0.050146 | `negation_scope` pilot sd (M-3) | 0.084345 | **0.057971** | 0.049946 | **0.037583** | 0.033119 |
   | 0.082152 | record row 4 | 0.138179 | **0.094971** | 0.081824 | **0.061570** | 0.054257 |
   | 0.109199 | realised sd, `2.18×` the pilot (M-3) | 0.183672 | **0.126238** | 0.108762 | **0.081841** | 0.072120 |

   The rule the record wrote for it stands: any contrast smaller than the `n = 8` cell of
   its realised paired sd is unfalsifiable at `N = 8` and may not be claimed (`:1010-1016`,
   READ). BED-S's paired sd is NOT MEASURED; the first eight seeds fix it, and the row is
   filled before any prediction in §6.E is scored (**M-3**: a pilot bounds nothing it did
   not measure — the realised sd is printed beside the pilot).
3. **TOST parity needs `N = 70`.** The 90% CI half-width of a two-sample contrast is
   `t_{.95,2N−2} √(2/N)` in units of `σ`: RUN `0.8807` at `N=8`, `0.6001` at `N=16`,
   `0.4955` at `N=23`, `0.2799` at `N=70`, against a margin of `0.5σ`; power at a true
   difference of zero first clears `0.80` at `N = 70` (`0.7985` at `69`, `0.8059` at `70`,
   `V13_CLAIM_AUDIT.md:35`, READ; three routes sharing no code). Parity at `N = 8` is
   therefore by **identity bind** (I1, `γ = 0` bitwise) and never by TOST (**M-13**).
4. **Thread-count floor.** The harness is deterministic only at a fixed thread count;
   cross-thread drift `2.345e-3` (`scale/it11_verdict.py:133-137`, READ; M-10). Every seed
   interval is built at one thread count, the lane is read from the journal and never
   from the machine (**M-16**), and a margin below `2 × 2.345e-3 ≈ 4.7e-3` cannot be
   defended (`MISTAKES.md:1093-1100`, READ). On CUDA the backward of `cumprod` has no
   deterministic kernel (`COSTS.md:143-156`, READ), so BED-S cells are journalled with the
   regime flag and the bitwise bar of `V17K_RULINGS.md` A1.1 is not claimed for them.
5. **Per-cell cost on the certified device** (RTX 4060 Laptop, 7.996 GiB, torch
   2.5.1+cu121, `COSTS.md:53-60`): one 150-step cell at `n = 2048, s = 64` costs
   `15.970 s` for `arm_smprime`, `1.614 s` for `arm_pl`, `1.497 s` for `softmax`
   (`V17_R4_RETAKE_PRICE.md:187-197`, READ, mean of two seeds); 24 cells plus fixed cost
   `158.74 s = 2.65 GPU-min` (`:203-206`). A capped cell reads `1.884 s`
   (`V20_R15_THEORY_TABLE.md:176`, READ). The BED-S oracle adds `m·K` dense solves of size
   `|T|`: at `|T| = 1200` that is about `|T|³/3 ≈ 5.8e8` flops per solve (DERIVED), which
   is under a second in float64 on this host and is paid once per draw, not per step.
   The clock is not stationary (`14.101 … 17.839 s` for one cell, `:140-150`), so prices
   are quoted in GPU-minutes to two decimals, never to the second (**M-8**, **P-8**).
6. **L-FLOOR.** Every capability number in the paper is printed as a distance to its
   information floor — the exact oracle at `0.0` for regression heads, the restricted-view
   Fano floor for the argmin — and never as a distance to `floor₁` or to any capability
   threshold (`CEQ_V20_R15_CONTRACT.md:60-63`, READ; `V20_R15_THEORY_TABLE.md:87-91`).
   A number without its floor is refused at assembly.

## 6.E Three competing predictions on BED-S, with counters (L-SIGN)

Filed before any BED-S cell exists. Each prediction has a counter of equal specificity
and one number that decides between them. The record's own scored predictions ran
`7 of 8` in the over-crediting direction (`MISTAKES.md:2037-2072`, READ, D-7), so the
sign of every miss below is logged, not discarded. The design point is `t* = 8, m = 8,
K = 2, N = 8, 4,769` parameters, 150 steps, one thread lane, one BH family.

**P1 — the joint read beats the restricted view.**
*Prediction:* the shape's argmin error, CP-upper at `N = 8`, sits below the one-hop
Fano floor `1 − (I(X_{≤1}; a*) + ln 2)/ln 8` computed at construction, and its committor
NRMSE sits below the one-hop budget ceiling `NRMSE(z_1, z*)`.
*Counter:* the shape's CP-lower error is at or above the one-hop Fano floor — the
resolvent read is no better than what a one-hop window licenses, and the joint-read
claim of `MATHEMATICS.md` §0.3 dies on the bed built for it.
*Deciding number:* `CP_upper(err_shape) − Fano_floor(k=1)`; negative confirms, non-negative
refutes. The zero-hop floor at `m = 8` is `0.6667` (RUN), so a shape reading at chance
(`0.875` error) would refute by `+0.208`.

**P2 — the deeper softmax skyline reaches the same number.**
*Prediction:* a softmax stack at depth `⌊log₂ 8⌋ + 2 = 5` and *unmatched* parameters
reaches the shape's argmin accuracy within the `N = 8` MDE of the realised paired sd
(§6.D.2) — the honest control holds, and the shape's claim is exactness, the printed
Neumann `δ = γ^{K+1}/(1−γ)` (I4), and one-read joint consistency, not accuracy.
*Counter:* the depth-5 skyline falls short of the shape by more than the MDE — which
would say the bed needs something the hop construction of `arXiv:2402.09268` does not
supply, and the paper would have to state a capability gap it did not predict.
*Deciding number:* `acc_shape − acc_skyline(depth 5)` against the MDE cell at `n = 8` of
the realised sd; inside the cell confirms, outside in either direction refutes, with
the sign logged.

**P3 — the harmonic residual separates the arms and the accuracy does not.**
*Prediction:* at matched depth 1 and `4,769` parameters, the shape's committor
prediction has harmonic residual `r(q̂) ≤ 10 δ` with `δ` the Neumann certificate at its
truncation order, while depth-1 softmax reads `r(q̂) ≥ 0.1` (of order the label sd), by
a ratio above `10×`; and the two arms' per-coordinate mean NRMSE differ by *less* than
the MDE. That is: the shape's advantage is joint consistency, visible in the residual,
and not in the marginal error where softmax is near-optimal per coordinate (D-1).
*Counter:* softmax's residual is within `2×` of the shape's — a per-row mixture can be
made jointly consistent by training, and "joint determination in one read" is not a
differentiator; the sentence at `MATHEMATICS.md:106-127` is withdrawn.
*Deciding number:* `r(q̂_softmax) / r(q̂_shape)`, paired by seed and draw; `≥ 10` confirms,
`≤ 2` refutes, `(2, 10)` is SPLIT and reported as such (`MATHEMATICS.md` §12's
three-way branch, `:628-633`).

All three are void if any guard in §6.C.4 fails at construction; a bed that is not
admitted produces no reading and no prediction is scored on it (**D-4**).

## 6.F Limits of this section

BED-S has no cell, no paired sd, no measured `t*`, no `I(X_{≤k}; a*)`; every BED-S number
above is a floor formula or a design constant, and is labelled so. The Kirchhoff second
oracle covers only `K = 2` with single-node boundaries until the grounded Laplacian is
extended. The MDE table's `n = 8` and `n = 16` columns are RUN through the record's
formula, not read from the record. `arXiv:2410.01537` and `arXiv:2402.09268` are cited
as the record cites them (`[U]`); no abstract page was fetched this session. CRPS is
registered, never computed. The chess witness is not a registered bed
(`V20_R15_THEORY_TABLE.md:224`) and is not used here.
