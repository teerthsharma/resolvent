# The it.14 corpus spec: harmonic extension, its resolvent, and the dose theorem

JUPITER, v-main.3M script iteration 14. The script line, verbatim:

> corpus spec: labels `u` solve `u(v) = (1/deg v) * sum_{w~v} u(w)` interior, `u|_B = g`;
> oracle `u_I = (I - P_II)^-1 P_IB g` (the proved resolvent); dose theorem
> `err(t) <= lambda_2^t` with `lambda_2` computed exactly per instance (one eig,
> `n <= 1024`), rungs stratified to `lambda_2` in `[0.90, 0.95]`.

Code `scale/r10_corpus_spec.py`, rows `results/r10_it14_corpus_spec.jsonl`.
Reproduce every number below with

```
python -m scale.r10_corpus_spec            # demo(), assert-based, ~4 s
python -m scale.r10_corpus_spec --write    # the corpus, 72.7 s, writes the .jsonl
```

Python 3.11.9, numpy 1.26.4, Windows 11 x86-64, branch `feat/r9-causal-consequence`
at `bf2a769`. Every rung seed is `0x3a140000 + index` and is printed in its row; the
naive-yield draw is seed `0x3a141000`. Re-run and the `.jsonl` is byte-identical:
nothing here reads a clock, a hash seed or a thread count.

`scale.vram_gate.preflight(0, name='r10-corpus-spec', host_mib=512)` returned
`HOST FITS -- needs 640 MiB, 4484 MiB available` before the corpus ran. Two
processes, no training, no GPU. Peak was two dense `1024 x 1024` float64 matrices
and one `eigh` workspace.

---

## 0. THE HEADLINE, BEFORE THE EVIDENCE

**The spec is well-posed on a class it does not name, and its dose theorem is not a
norm-free statement.** Both are stated here as amendments rather than applied
quietly, and both were found by the instruments rather than by reading.

1. **`(I - P_II)` is singular unless every interior vertex can reach `B`.** The
   excluded class is named in §1 and the guard is shown firing.
2. **`err(t) <= lambda_2^t` is FALSE in `l_inf` on 6 of 12 rungs**, worst observed
   ratio `1.290631` at `t = 64`, and is false by an exact `1.0000000000` at `t = 1`
   on the `g == 1` instances where no adversarial construction is involved. It is
   true on 12 of 12 in the degree-weighted 2-norm, and tight there — worst-case
   slack `7.00e-04` in rate. The spec does not say which norm. §4.

Everything else held. Dual-oracle max disagreement over the whole corpus is
`9.992007e-16`, four orders inside the repo's own tolerance. 12 rungs ok, 0 refused,
0 errored.

---

## 1. F1-F8 — WHAT THE SPEC RESTS ON, CITED OR NAMED

"Fetch" here means: located in this repo and cited by file and line, or declared
absent and named. Nothing was fetched from the internet. **Seven of the eight are
cited by file and line; F8 is absent and is stated as a standard assumption. Three
further results the build leans on are also absent, and are named after the table
rather than folded into it.**

| | Result | Status |
|---|---|---|
| **F1** | **`lambda_2 := rho(P_II)`, not the literal second eigenvalue of `P`.** `P = [[P_II, P_IB], [0, I]]` is block triangular, so `spec(P) = spec(P_II) u {1,...}` and the literal second eigenvalue is `1`. | **CITED.** `scale/foreman_lambda2.py:24-25`; `MATHEMATICS.md:426`. |
| **F2** | **The resolvent oracle.** `N = (I - Q)^-1`, `B = N R`, the fixed point of `z <- Q z + R`; solved, not inverted. | **CITED.** `MATHEMATICS.md:423`; `scale/foreman_lambda2.py:317-323`. |
| **F3** | **The exact truncation identity `u - u_t = Q^t u`.** The discarded Neumann tail factors as `Q^t` times the whole sum — exact, not an approximation. This is what makes `err(t)` a statement about `Q^t` and therefore about `rho(Q)` at all. | **CITED.** `MATHEMATICS.md:431`; `scale/foreman_lambda2.py:398-402`. |
| **F4** | **The dose theorem itself, in the norm where it is a theorem.** A non-negative `A` with a Perron certificate `w > 0`, `A w <= rho w`, contracts by `rho` in `\|v\|_w = max_i \|v_i\|/w_i`; iterating gives `rho^t`. | **CITED, and it is a machine-checked proof.** `lean/CEQ/Contraction.lean:72` (`weighted_contraction`) and `:99` (`weighted_contraction_iterate`), with the certificate structure at `:59`. **The same file already carries the refutation of the norm-free reading**: `expander_expands_l2` at `:151` is a row-stochastic matrix that expands the Euclidean norm, and the header at `:12-15` says in terms that row-stochasticity bounds the `inf`-norm and not the 2-norm. §4 is that warning arriving at the absorbing block. |
| **F5** | **The Kirchhoff / matrix-tree second oracle.** `omega_x = M_xa / M_aa = F(x, a \| b) / F(a \| b)` from the `b`-grounded Laplacian; one solve `L_b z = e_a`, symmetric unnormalised `D - A`, independent of the chain solve in matrix, normalisation, right-hand side and index map. | **CITED.** `scale/kirchhoff.py:1-90` (derivation), `:187-213` (`harmonic_measure`); `MATHEMATICS.md:542-609`. |
| **F6** | **The agreement tolerance `1e-10` and the window that sets it.** Largest gap between two correct oracles `9.636736e-14`; smallest gap the planted degree defect produces `1.749951e-01`; twelve orders of daylight; `1e-10` three orders above the former and nine below the latter. | **CITED, and imported rather than re-picked.** `scale/kirchhoff.py:98-106`; `MATHEMATICS.md:590-604`. `scale/r10_corpus_spec.py:118` is `AGREEMENT_TOL = kirchhoff.AGREEMENT_TOL`. |
| **F7** | **The band `[0.90, 0.95]`, and the measured fact that the obvious dial does not reach it.** Bridge conductance moves the ERGODIC `lambda_2`, not `rho(Q)`; `rho(Q)` is monotone INCREASING in bridge weight and reaches `[0.9789, 0.9999]` and `[0.9985, 0.9999]` on the two shipped graphs, never the band. | **CITED.** `scale/foreman_lambda2.py:110` (`TARGET_LO, TARGET_HI = 0.90, 0.95`), `:38-66` (the refutation), `results/foreman_lambda2.txt:6-30` (the sweep). The consequence for §3 is direct: the band has to be reached by **choosing the boundary set**, and that is what stratification means here. |
| **F8** | **`rho(P_II) < 1` iff every interior vertex reaches `B`**, i.e. the resolvent exists exactly on the class of §1. | **NOT IN THIS REPO. ASSUMED, standard.** The standard name is the *substochastic Perron–Frobenius / absorbing-chain fundamental-matrix* result: `Q` substochastic with every state having a path to a deficient row has `rho(Q) < 1`, so `N = sum_k Q^k` converges and equals `(I - Q)^-1`. The nearest thing in the repo is `lean/CEQ/OracleSeparation.lean:148` (`not_isNilpotent`), which proves `Q` is never nilpotent — the opposite corner of the same square, and not this. No line was invented for it. |

Three further results are used and are **not in this repo**; they are named here
rather than cited, for the same reason.

- **Reversibility of the interior block.** `S := D_I^{1/2} P_II D_I^{-1/2} =
  D_I^{-1/2} W_II D_I^{-1/2}` is symmetric and similar to `P_II`, so the spectrum is
  real and `\|P_II^t\|_D = rho^t` exactly in the degree-weighted 2-norm. **ASSUMED,
  standard** (*symmetrisability of a reversible chain*). `scale/foreman_lambda2.py:289`
  uses exactly this conjugation, but for the **ergodic** walk and never for the
  transient block; and `PRIOR_ART.md:692-702` cites Levin–Peres–Wilmer Thm 13.10 for
  reversible chains only to record that it is about the wrong matrix. So the repo
  has the technique and not the statement. This result is what §4's `l2_deg` column
  is, and it is the one that carried the corpus.
- **Monotonicity of the Perron root under principal-submatrix deletion**, which is
  what makes the §3 bisection exact rather than a search. **ASSUMED, standard**
  (*Perron–Frobenius monotonicity*). Absent from the repo.
- **Gelfand's formula**, `\|A^t\|^{1/t} -> rho(A)`, which is why `r_eff` in §4 is the
  right sensitivity statistic and why a constant-factor violation persists rather
  than washing out. **ASSUMED, standard.** Absent from the repo.

---

## 2. THE ORACLE, IMPLEMENTED AND CHECKED THREE WAYS

`u_I = (I - P_II)^-1 P_IB g` is **oracle A**, `np.linalg.solve` on `(I - P_II)`.

**Oracle B** is the defining averaging map iterated from the **adjacency lists**
(`scale/r10_corpus_spec.py:259-278`). It forms no matrix at all — no `P_II`, no
`P_IB`, no `D`, no index map into a dense array — so a transposed block, a mis-slotted
row or a degree read off the wrong axis moves one route and not the other. That is
the independence argument `scale/kirchhoff.py:51-62` makes for its own second route,
applied one level down.

It is run `ceil(log(1e-17)/log(lambda_2)) + 64` sweeps (737–827 on this corpus),
and **the size of the final update is reported per rung**: `0.0e+00` on all twelve.
The iteration is stationary in float64 before it stops, so the residual left in the
comparison is conditioning and not a stopping rule. An iterative oracle stopped early
would fail this tolerance for its own reasons; reporting the last move is what
separates the two.

**Oracle C** is the repo's own — `kirchhoff.harmonic_measure`, the matrix-tree /
grounded-cofactor route (F5) — which applies where `|B| = 2` and `g = (1,0)`, exactly
`MATHEMATICS.md` §11's harmonic measure. It is joined on the path family by two
closed forms.

### The measured disagreement

| pair | max abs disagreement, over the whole corpus |
|---|---|
| **A vs B, 12 rungs, `n` 64…1024** | **`9.992007e-16`** |
| A vs B, 4 path cases | `6.661338e-16` |
| **A vs C (Kirchhoff), 4 path cases** | **`2.220446e-16`** |
| **A vs closed form `u_x = 1 - x/(L+1)`** | **`2.220446e-16`** |
| `lambda_2` vs `cos(pi/(L+1))`, 4 path cases | `2.220446e-16` |
| `eigvalsh(S)` vs `eigvals(P_II)`, 7 rungs `n <= 256` | `3.774758e-15` |

Per-rung A-vs-B gaps run `2.220e-16` (`n=64`) to `9.992e-16` (`n=1024`), growing with
size as conditioning says they must — the same shape `scale/kirchhoff.py:69-71`
records for its own pair.

**The tolerance is `1e-10`, imported from `scale/kirchhoff.py:106` and not re-chosen.**
It is held to because the window it was set from is the window this corpus is in:
the largest clean gap here is `9.99e-16`, five orders below the `9.636736e-14` that
window was built around, and the planted defect below produces `4.07e-01` at its
weakest — so the daylight is wider here than where the constant was calibrated, not
narrower. Re-deriving a tighter constant from these numbers would be fitting a
tolerance to a result already seen.

### The oracle check is shown able to FAIL

Oracle B is re-run dividing by `deg + 1` — the repo's own planted defect
(`kirchhoff.scratch_chain_with_off_by_one`, `scale/kirchhoff.py:225-260`), the
miscount you get by counting a node among its own neighbours. It is survivable by
construction: the map stays a strict contraction, converges, and returns numbers in
`[0,1]`, and nothing downstream would notice.

**It is rejected on 12 of 12 rungs.** Gaps `4.067603e-01` (min, `n=256 deg=8`) to
`6.301502e-01` (max, `n=512 deg=4`) — nine orders above the tolerance at the weakest.
`demo()` asserts the rejection on the path case at `6.041033e-01`.

---

## 3. LAMBDA_2 EXACTLY, AND WHAT STRATIFICATION COSTS

**One eigendecomposition per instance, and it is symmetric.** `S = D_I^{1/2} P_II
D_I^{-1/2}` has entries `W_ij / sqrt(d_i d_j)`, is symmetric and similar to `P_II`.
`eigvalsh` orders the real spectrum exactly instead of sorting complex moduli, and
`S >= 0` entrywise puts the spectral radius at the top of the ascending spectrum —
`vals[-1]`, no `abs`. The same call returns the Perron vector, so §4's weighted norm
costs nothing extra. The similarity claim is checked, not asserted: against
non-symmetric `eigvals(P_II)` on the seven rungs `n <= 256`, max gap `3.774758e-15`;
against `cos(pi/(L+1))` on the path family, max gap `2.220446e-16`.

### The naive distribution — and it is nowhere near the band

Prior declared before any number was read (`scale/r10_corpus_spec.py:401-412`), and
it is the one a person writing "generate some graphs and label them" would type:
`n` uniform on `{64,128,256,512}`, mean degree uniform on `[3,8]`, `|B|` uniform on
`{1,...,n/2}`, `B` uniform without replacement. 200 draws, seed `0x3a141000`, one
eigensolve each, no retry.

| quantile | `lambda_2` |
|---|---|
| min | `0.5625137473` |
| 25% | `0.7239621002` |
| **median** | **`0.8304923343`** |
| 75% | `0.8913555751` |
| max | `0.9943936326` |

**16 of 200 land in `[0.90, 0.95]`. Naive yield 8.0%, 0 errors.** The median sits
`0.07` below the band and the spread is enormous — `0.56` to `0.99` — because `|B|`
was drawn uniformly and `lambda_2` is a steep function of it.

**So the stratification is doing real work, and 8% is the honest price of not doing
it.** It is also the second half of F7: `foreman_lambda2` already measured that the
conductance dial cannot reach this band on the shipped graphs, and the alpha-killing
dial that can reach it changes the operator (`Q = alpha P_TT` is a *killed* walk,
whose fixed point does **not** satisfy `u(v) = mean of neighbours`). Under this
spec's mean-value constraint there is no dial at all: `P_II` is determined by the
graph and `B`, so the only free variable is `B`.

### The stratification, and it is a bisection rather than a search

`B` is grown along a **fixed** node order drawn once per rung. Each added boundary
node deletes a row/column pair from `P_II`, and the Perron root of a non-negative
matrix is monotone non-increasing under that deletion, so `lambda_2(k)` is monotone
and bisection finds the crossing in `ceil(log2 n)` eigensolves. Drawing a fresh
random `B` per `k` would destroy the monotonicity and turn this into a search; that
is why the order is drawn once and held (`scale/r10_corpus_spec.py:361-395`).

**121 eigensolves bought 12 rungs — 10.1 per rung, 8 at `n=64` up to 12 at `n=1024`,
0 band misses, 0 errors.** Against the naive route's 12.5 draws per hit at a wider
`n`-distribution, and with the naive route offering no control over *where* in the
band it lands.

### The corpus

| # | `n` | mean deg | `\|B\|` | interior | `lambda_2` | `lambda_3` | deg range | eigensolves | s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 4 | 7 | 57 | `0.9434514476` | `0.749057` | 1–11 | 8 | 0.07 |
| 2 | 64 | 6 | 4 | 60 | `0.9466824744` | `0.665643` | 2–12 | 8 | 0.14 |
| 3 | 128 | 4 | 11 | 117 | `0.9455579636` | `0.803928` | 1–11 | 9 | 0.42 |
| 4 | 128 | 6 | 7 | 121 | `0.9490558907` | `0.705824` | 2–12 | 9 | 0.57 |
| 5 | 256 | 4 | 19 | 237 | `0.9489302427` | `0.854000` | 1–10 | 10 | 1.55 |
| 6 | 256 | 6 | 16 | 240 | `0.9461843337` | `0.715950` | 1–19 | 10 | 1.82 |
| 7 | 256 | 8 | 17 | 239 | `0.9455168387` | `0.629075` | 2–17 | 10 | 1.33 |
| 8 | 512 | 4 | 41 | 471 | `0.9499393658` | `0.852129` | 1–12 | 11 | 4.49 |
| 9 | 512 | 6 | 27 | 485 | `0.9464959515` | `0.715682` | 1–16 | 11 | 5.00 |
| 10 | 512 | 8 | 31 | 481 | `0.9493823268` | `0.666077` | 1–18 | 11 | 5.93 |
| 11 | 1024 | 4 | 80 | 944 | `0.9494274830` | `0.873311` | 1–14 | 12 | 17.52 |
| 12 | 1024 | 6 | 64 | 960 | `0.9498399931` | `0.719035` | 1–15 | 12 | 18.35 |

All twelve in band, `[0.9434514476, 0.9499393658]`. **The bisection lands in the
upper half of the band every time** — `lambda_2(k)` steps down past `0.95` and the
first `k` that clears it is still near the top. Reaching the lower half needs a
different rule, and none was applied after seeing this.

The path family gives four more, with `lambda_2 = cos(pi/(L+1))` in closed form:
`L=6` `0.9009688679`, `L=7` `0.9238795325`, `L=8` `0.9396926208` — all in band — and
`L=9` `0.9510565163`, **out of band and recorded as out of band** rather than
rounded in.

---

## 4. THE DOSE THEOREM, TESTED — AND IT FAILS IN THE OBVIOUS NORM

`err(t) := ||u - u_t|| / ||u||` with `u_0 = 0` and `u_t = sum_{m<t} P_II^m P_IB g`,
i.e. exactly what a `t`-hop budget computes. `err(0) = 1 = lambda_2^0` by
construction, so `t = 0` is excluded from every "tightest" reading below; `t` runs to
`64`.

**`err(t) <= lambda_2^t` is not a norm-free statement.** The spec does not say which
norm, so all four a reader could mean were measured. Over the 12 rungs:

| norm | held | max `err(t)/lambda_2^t` | worst-case rate slack |
|---|---|---|---|
| `l_inf`, `max_i \|x_i\|` | **held on 6 of 12 — FAILS on 6** | **`1.290631`** at `t=64`, rung 11 | **`-0.011696`** (rung 6) |
| `l_2`, unweighted | 12 of 12 | `0.998206` at `t=1`, rung 7 | `+0.000045` |
| **`l_2` degree-weighted, `sqrt(sum d_i x_i^2)`** | **12 of 12** | `0.994175` at `t=1`, rung 7 | `+0.000108` |
| `sup` Perron-weighted, `max_i \|x_i\|/w_i` | 7 of 7 where defined, **5 undefined** | `0.827314` at `t=1`, rung 6 | `+0.006595` |

"Rate slack" is `lambda_2 - r_eff` with `r_eff := max_{1<=t<=T} err(t)^(1/t)`; it is
negative exactly when the bound is violated, and it is the check's exact sensitivity
(§4.3).

### 4.1 The violation, and it needs no adversarial construction

**Six of twelve rungs violate `err(t) <= lambda_2^t` in `l_inf`.** Worst is rung 11
(`n=1024`, mean deg 4, seed `0x3a14000a`, `lambda_2 = 0.9494274830`) at ratio
`1.290631`, `t = 64`. Rungs 8 (`1.214256`), 5 (`1.246437`), 6 (`1.012361`, at
`t = 1`), 4 (`1.026051`) and 2 (`1.014192`) follow.

The ratio **grows** with `t` on the large rungs, which is what a constant-factor
violation looks like pre-asymptotically: `||P_II^t x||_inf ~ C lambda_2^t` with
`C > 1`, so the bound is not approached from below and later `t` does not rescue it.
The `l_inf` reading of the dose theorem is wrong permanently, not transiently.

**The `g == 1` instances make it exact and remove every degree of freedom.** With
`g == 1` the harmonic extension is `u == 1` (measured deviation `6.66e-16` and
`1.11e-15`), so `err(t)` in `l_inf` is literally `max_i P_i(survive t steps)`. Any
interior vertex with no boundary neighbour has `P_i(survive 1) = 1`, so

```
    err(1) = 1.0000000000  >  lambda_2 = 0.9415127885    (n=128,  seed 0x3a140100)
    err(1) = 1.0000000000  >  lambda_2 = 0.9451165844    (n=256,  seed 0x3a140101)
```

and the ratios reach `1.404412` and `1.224972` by `t = 64`. **Every instance in the
band has such a vertex**, necessarily: a graph in which every interior vertex touches
`B` absorbs in one step and has `lambda_2` nowhere near `0.9`. So the `l_inf`
violation is not a pathology of these draws — it is a property of the band.

### 4.2 The norm where it is a theorem, and it is not the one the repo formalised

The bound is **exactly** true in the degree-weighted 2-norm, for a reason:
`S = D_I^{1/2} P_II D_I^{-1/2}` is symmetric, so `||S^t||_2 = rho(S)^t = lambda_2^t`
with no constant, and `||P_II^t x||_D = ||S^t D^{1/2} x||_2 <= lambda_2^t ||x||_D`.
**12 of 12, and tight**: worst-case rate slack `7.00e-04`, i.e. the measured decay
rate is within `0.074%` of `lambda_2` on the loosest rung and within `0.011%` on the
tightest. This bound is not vacuous — it is very nearly attained.

**The unweighted `l_2` column held 12 of 12 and that is luck, not a theorem.**
`||P_II^t||_2 = ||D^{-1/2} S^t D^{1/2}||_2 <= sqrt(d_max/d_min) * lambda_2^t`, exact
only when the graph is regular; these rungs carry degree ranges as wide as `1–19`, so
the available violation factor is `sqrt(19) = 4.36` and it simply was not realised by
these `g`. Nothing here licenses reading the unweighted `l_2` version as safe.

The Perron-weighted sup norm is where the repo's own machine-checked theorem lives
(F4, `lean/CEQ/Contraction.lean:72`), and two things showed up there.

- **It is looser, by one to two orders**: slack `0.006595` to `0.026786` against
  `l2_deg`'s `0.000108` to `0.000700`. The formalised norm is the pessimistic one on
  this operator class.
- **It is undefined on 5 of 12 rungs.** The certificate needs `w > 0`, which is
  irreducibility of `P_II`; a boundary set that disconnects the interior leaves the
  Perron vector supported on one piece and zero on the others, and `max_i |x_i|/w_i`
  is then not a norm. Rungs 5, 8, 9, 11, 12 are in that state. **They are recorded
  as `null`, not as passes** — the `.jsonl` carries `"perron_positive": false` and
  `"held": null`, and the counts above say `7 of 7 where defined, 5 undefined`
  rather than `12 of 12`.

Where it is defined, the certificate is checked entrywise on the actual matrix rather
than assumed: `max_i |(P_II w - lambda_2 w)_i| <= 2.914335e-16` across all seven.

`lean/CEQ/Contraction.lean` already contains the warning this section is: its header
at `:12-15` says row-stochasticity bounds the `inf`-norm and not the 2-norm, and
`expander_expands_l2` at `:151` is the witness. §4.1 is that same fact arriving from
the other side — the absorbing block bounds the 2-norm (weighted) and **not** the
`inf`-norm.

### 4.3 The dose check is shown able to FAIL

The bound at rate `r` is violated at some `t <= T` iff `r < r_eff`, so the check
catches any understatement of `lambda_2` larger than `slack = lambda_2 - r_eff` and
catches nothing smaller. That identity gives the instrument's exact sensitivity with
no search, and two plants are run against it.

- **Fixed plant, `lambda_2 - 0.02`, declared as a constant.** In `l2_deg` it **fires
  on 12 of 12** (slack `1.08e-04`…`7.00e-04`, all far under `0.02`). In `sup_perron`
  it fires on **6 of the 7 where it is defined** and does **not** fire on rung 3,
  whose slack is `0.026786` — the only rung on this corpus whose slack exceeds the
  plant.
- **That non-firing is not an instrument failure and is not glossed as one.** It is
  the measured statement that the Perron-weighted bound on rung 3 is too loose to
  notice a 2%-wrong `lambda_2` within 64 steps. The adaptive plant `slack + 0.005`
  **fires on 7 of 7 wherever the norm is defined**, and 12 of 12 in `l2_deg`. That
  separates "the check works and the bound is loose" from "the check is broken".
  Both readings are in the `.jsonl` (`dose_mustfire.fires`, `.fires_adaptive`,
  `.detect_threshold`).

`demo()` asserts all of it on the path case: `sup_perron` and `l2_deg` hold,
`l_inf` **does not**, `err(1) > lambda_2` in `l_inf`, and the adaptive plant fires.

---

## 5. THE AMENDMENT THAT MAKES THE SPEC WELL-POSED

**As written the spec is ill-posed on a class it does not exclude, and the fix is one
clause.**

> `(I - P_II)` is invertible iff every interior vertex has a path to `B`.

If an interior component is disjoint from `B`, `P_II` restricted to it is
row-stochastic, `rho = 1`, and `(I - P_II)` is singular — and the underlying problem
genuinely has no unique answer there: any constant may be added on that component and
the mean-value property still holds. Isolated vertices fail one step earlier, at
`1/deg v`. `admissible` (`scale/r10_corpus_spec.py:162-193`) refuses both and says
which; `demo()` shows both refusals firing on planted graphs
(`{0-1, 2-3}` with `B = {0}`, and a graph with a degree-zero vertex). **This corpus
never hit the excluded class** — the generator draws connected graphs by construction
— so the count is 0 refusals of 12, and the guard's value is entirely in having been
shown to fire.

Two smaller notes of the same kind, recorded rather than repaired:

- **`lambda_2 = rho(P_II)` need not be a *simple* eigenvalue** once `P_II` is
  reducible, which is the 5-of-12 case of §4.2. The rate is still correct; the
  Perron *vector* is not unique and the weighted sup norm is not available.
- **`err` needs a norm.** The minimal amendment is one clause: *"dose theorem
  `err(t) <= lambda_2^t` in the degree-weighted 2-norm"*. That version is a theorem,
  held 12/12, and is tight to `7e-04` in rate. Any `l_inf` version is false on this
  corpus and false by construction on any instance in the band.

---

## 6. ERRORS, AND THEY ARE COUNTED SEPARATELY

| kind | rows | ok | refused | error |
|---|---|---|---|---|
| `rung` | 12 | 12 | 0 | 0 |
| `kirchhoff` (path) | 4 | 4 | 0 | 0 |
| `constant_g` | 2 | 2 | 0 | 0 |
| `naive_yield` | 1 | 200 draws | — | 0 |

Every row carries `status`. A rung that overshoots the band returns
`status="refused"` with the crossing `lambda_2`; a rung that raises returns
`status="error"` with the exception text. Neither is summed into a pass. The
`sup_perron` `null` readings of §4.2 are the same discipline one level down: a norm
that is undefined reports `null`, never `held: true`.

---

## 7. LIMITS

Collected here rather than scattered.

The graph family is one family — uniform attachment tree plus random chords — chosen
so connectivity is by construction and the §3 yield prices the *band* rather than a
connectivity retry. Nothing here bears on Rips, grid or expander families, and the
`l_inf` violation rate of 6/12 is a property of this family at these degrees, not a
universal one; the `g == 1` argument, which is family-free, is the part that
generalises. Twelve rungs is not a sample size for anything but a bound check.
`t` stops at 64, so a violation first appearing after 64 steps would be missed —
though F3 makes that unlikely, since the residual is exactly `P_II^t u` and its shape
is settled well before then. The naive-yield prior is one declared prior; a different
one gives a different 8%. The bisection lands in the upper half of the band on every
rung and no rule was applied to spread it after that was seen. The `sup_perron`
column is missing on 5 of 12 rungs and the `l_inf`-vs-`l2_deg` comparison there rests
on 7. `lambda_2` for the naive draws was computed with the boundary set *as drawn*,
so its distribution is conditional on `|B|` uniform, which is the point but also the
whole of the point. No claim is made here about what a model can or cannot learn from
this corpus; this iteration builds and attacks the labels only.
