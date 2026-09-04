# 4. What the record proved, measured and refuted

*The plan's justification, compressed: this section keeps only what §5's cards rest on or are designed against, and every OPEN item
names the card that discharges it. Evidence classes — **RUN** executed on this box this session (2026-09-03, HEAD `207e7b9`, branch
`v17k-gate0`, torch 2.5.1 float64, Lean `leanprover/lean4:v4.7.0`); **READ** `path:line` at that HEAD; **CITED** `[V]`/`[U]`;
**DERIVED** with steps shown. Struck constants are named by the quantity they were, never by their numeral; the numerals live in
`STRUCK.md`, rendered from `tests/loop/test_no_struck_constant_ships.py:47`, and a document reprinting one fails that test.*

---

## 4.1 Proved

**The inventory, RUN this session.** `grep -cE '^(theorem|lemma) ' lean/CEQ/*.lean` returns **169** declarations across **12** files
(Contraction 5, Nilpotent 4, Occupancy 3, OracleSeparation 12, OrbitBound 5, Refcount 10, V15 23, V15Fork 11, V15Kernel 30, V15Phase
26, V15Source 7, V16Domain 33; the import root contributes 0). `lake build` exits **0** on a warm cache; the record calls a warm
green *weak evidence* and backs it by force re-elaboration (`READ V16_LEAN_DOMAIN.md:394-399`), which this session did not repeat,
so card **J-L18** rebuilds cold. `grep -n '\bsorry\b' lean/CEQ/*.lean` returns six hits, every one the prose phrase "No `sorry`" in
a file header (`V15.lean:41`, `V15Fork.lean:44`, `V15Kernel.lean:72`, `V15Phase.lean:72`, `V15Source.lean:60`, `V16Domain.lean:75`):
**zero** `sorry` terms. The axiom set `[propext, Classical.choice, Quot.sound]`, with no `sorryAx`, is **READ** from
`README.md:101-104` and declaration-by-declaration for `V16Domain` from `V16_LEAN_DOMAIN.md:417-465`; Mathlib is pinned at
`lean/lake-manifest.json:7`. Mechanism **P-11** (`MISTAKES.md:1597`, a contract citing its own `[M]`-tagged item as settled): the
paper cites only declarations that build, and §4.3 correction (3) names three that do not. Where a row's census is empty the theorem
is decoration — every magnitude theorem in `V15Phase` carries $\forall k,\ 0<m_k$ and so is decoration on BED-M's $a=0$ draw (`READ
V16_LEAN_DOMAIN.md:151-160`), its closed-support replacement being the `pathProd_*` clauses at 3 of 3. That substitution answers
**V-25** (`:1954`) and is kept as the law **L-DOM**, discharged per target by **J-L18** and per bed by **S-12**.

**Table 4.1 — the load-bearing declarations**, transcribed from the `.lean` text, not paraphrased.

| declaration | file:line | exact hypotheses | certifies | measured instance | domain census, BED-M / BED-K |
|---|---|---|---|---|---|
| `Nilpotent.pow_card_eq_zero`, `occupancy_is_exact_inverse` | `Nilpotent.lean:77`, `:96` | `StrictlyLower A` (`:46`: $A_{ij}=0$ whenever $i\le j$) | A strictly causal operator is nilpotent at the cardinality, and the finite $n$-term occupancy sum **is** its exact two-sided inverse — no sign, magnitude or norm hypothesis. | `matrix_power(A,64)` max-abs $0.0$; $(I-A)^{-1}$ equals `ceq/arm_smprime.py:144 path_product` to max-abs $0.0$ (**RUN**); `tests/w2/test_w2_nonnormal.py:124` asserts $\rho(A)<10^{-12}$ | BED-M full (the chain is strictly sub-diagonal); BED-K empty |
| `Nilpotent.one_not_nilpotent` | `Nilpotent.lean:105` | $0<n$ | Weakening `StrictlyLower` to lower-triangular destroys the theorem: $1^{k}\ne 0$. | — (a refusal; its instance is the softmax corner's diagonal) | binds wherever the shape keeps the diagonal (regime S) |
| `Occupancy.occupancy_telescope` (+ `_eq_inverse_of_nilpotent`, `:83`) | `Occupancy.lean:52` | any `[Ring R]`; **none on $A$** | $(1-A)\sum_{k<N}A^{k}=1-A^{N}$: the truncation residual is exactly $A^{N}$, with no convergence hypothesis. | at $K=16$, $\gamma=0.7$: residual $0.007754350466241788$ against $\gamma^{17}/(1-\gamma)=0.007754350466240224$ (**RUN**) | vacuous by design; the census belongs to its instance |
| `Contraction.weighted_contraction`, `rowStochastic_perron` | `Contraction.lean:72`, `:118` | `PerronCertificate A w ρ` (`:59-62`); then `RowStochastic P`, $0\le\gamma$ | $\lvert(Av)_i\rvert\le\rho Mw_i$, and $\gamma P$ carries such a certificate at weight $\mathbf 1$, rate $\gamma$ — the sup-norm contraction the shape's certificate rests on. | causal softmax rows read $1$ to $2.220446\times10^{-16}$ (**READ** `V16_ARM_SMPRIME.md:29`) | 3 of 3 (the hypothesis is on $P$, not the corpus); `expander_expands_l2` (`:151`) refuses the 2-norm reading |
| `V15.chain_path_product`; `prefix_logit_mask` | `V15.lean:74`, `:128` | $m\le n$, $a,b$ arbitrary; then $\forall k,\ 0<a_k$ with $j\le i$ | Every chain coefficient is a **product** of consecutive gates along the path, and an exponential prefix scan of log-gates reproduces it. | last row against `scale/negation_scope.py:286 equilibrium_oracle`, max-abs $6.217248937900877\times10^{-15}$ (**RUN**) | chain: BED-M full. Scan: **1 of 3** (`bedM_overlap_old_two = 1`, `V16Domain.lean:302`, `by decide`); BED-K empty for both |
| `V15Fork.Asink_row_sum`; `Asink_computes_chain` | `V15Fork.lean:81`, `:140` | none on $a$; then $\forall k,\ a_k\ne 1$ | The telescope head's rows sum to $1$ unconditionally, and one causal row-stochastic softmax head with key-side bias $s_j=\log(1-a_j)$ and a value-zero BOS sink reproduces the chain label exactly. | $6.661338147750939\times10^{-16}$ at $s=8$, $7.549516567451064\times10^{-15}$ at $s=64$ (**RUN** `results/v15_r1.jsonl`); ARM PHASE $9.155133597044475\times10^{-16}$, ARM SMPRIME $5.919777\times10^{-16}$ (**READ** `workdonenewseal.md:91-96`) | row sum universal; the chain clause **2 of 3**, the excluded point being the band value $+1$, where the $\beta=1$ bind reads $1.335288$ |
| `V15Fork.no_row_stochastic_with_drive_values` | `V15Fork.lean:195` | $\forall k,\ 0<a_k$; row sum $1$; exactness on **all** $b$ | No row-stochastic head reproduces the chain on unrescaled drives: the label leaves the convex hull of its own drives. | on $b\equiv 1$ the label exceeds $1$ (`chain_one_gt_one`, `:169`) | the two-branch escape it refuses is **V-24** in proof form |
| `V16Domain.three_corners_containment`; `corners_are_distinct` | `V16Domain.lean:433`, `:445` | $g_0\equiv 0$, $g$ and $qk$ arbitrary; then a decidable witness | Softmax ($\beta=1$), linear attention ($\beta=0$) and the path product ($\beta=0$, QK off) are three settings of one operator, and they differ at $(1,0)$. | $\beta=1$ bitwise against `softmaxAttn`; row sums $1.000000$ at $\beta=1$ against $[1.312192,\dots,10.293107]$ at $\beta=0$; $\lvert c_1-c_2\rvert=4.472918$, $\lvert c_1-c_3\rvert=1.144938$, $\lvert c_2-c_3\rvert=5.335671$ (**READ** `:28-32`, `:333`) | 3 of 3 |
| `V16Domain.pathProd_eq_zero_iff`; `no_prefix_scan_represents_a_zero_gate` | `V16Domain.lean:129`, `:165` | **none**; then $\exists k\in[j{+}1,i],\ m_k=0$ | A zero gate annihilates the path product exactly and only a zero gate does, and **no** exponential prefix scan can carry one. | $96.78\%$ of BED-M causal pairs annihilate, annihilation MCC $1.000000$ corpus-alone; the `exp_scan` plant reads `nan` on $133{,}120/133{,}120$ causal pairs at $n=128$, $s=64$ (**READ** `V16_ARM_SMPRIME.md:519-527`) | **3 of 3** (`bedM_overlap_new_two = 3`, `:304`); BED-M proper 2 of 2 (`:310`) |
| `V15Kernel.first_order_cannot_delay`; `delay_forces_state_injective`; `linear_first_order_cannot_delay_beyond_state_dim` | `V15Kernel.lean:140`, `:253`, `:360` | $1\le d$ with $f$ **arbitrary**; any state type, no linearity; linear $A,B,C$ with $k<e+1$ | No scalar-state first-order recurrence delays by $d\ge1$; delay forces an injective state map; linear delay $d$ needs state dimension at least $d$. | the fitted first-order recurrence reaches $R^{2}=-0.000170$ on the delay bed where the same code recovers AR(1) at $R^{2}=1.000000$ (**READ** `V15_N4_BEDK.md:194`); bracketed in-file by `delay_realizable_at_dimension_d` (`:235`) | BED-K covers $d\ge1$; $d=0$ is legal and *necessarily* outside, exhibited by `V16Domain.delay_zero_is_first_order` (`:339`) |
| `V15Kernel.first_order_cannot_powerlaw` | `V15Kernel.lean:521` | $0<\alpha<1$ | The Grünwald–Letnikov power-law kernel is not geometric, so no affine first-order recurrence reproduces it exactly. | the registered box is $\alpha=H-\tfrac12\in(0,\tfrac12)$; the *approximation* gap is open — the bed fits at $R^{2}=0.604$ ($H=0.75$), $0.755$ ($H=0.9$) (**READ** `workdonenewseal.md:189`) | BED-K; debt **D-APPROX**, discharged by no card and carried in Limits |
| `V15Source.source_is_first_order_difference` | `V15Source.lean:108` | `StrictlyLower A`; $r=\mathrm{occ}(A)h$ | Inverting the resolvent costs one mat-vec and a subtraction, $O(\mathrm{nnz}\,A)$, against a forward map that fills in. | propagated-field error $8.882\times10^{-16}$, two planted sources; $\mathrm{nnz}(1-A)=251$ against $\mathrm{nnz}(W)=1703$ at $n=128$, $d=5$ (**READ** `V15Source.lean:40-42`) | an X35′ instrument, not a bed |
| `OracleSeparation.oracle_ne_resolvent`; `truncation_never_exact` | `OracleSeparation.lean:166`, `:180` | `StrictlyLower A`; `Nonneg Q`; `SymmSupport Q`; $0<Q_{ij}$ | An undirected-walk transient block is **never** a strictly causal operator, at the operator level before any label is drawn, and every truncation rung on it leaves a residual. | BED-1's chain, `bed_1.build(T=0.25, jitter=0.05, seed=11)`: `Nonneg True`, `SymmSupport True`, $\rho(Q)=0.9408612510154677$, $\max\lvert Q^{11}\rvert=0.47739216729378814\ne0$ (**RUN**) | BED-1, E4′; **D-2** (`:710`) as a theorem, and the reason the committor head ships as an exact solve with $\delta=0$ to rounding |

**The regime boundary.** The shape's default operator is the $\beta=1$ corner: a causal softmax row over $j\le i$, so $P_{ii}>0$ and
$\gamma P$ is lower-triangular **with** a diagonal. `pow_card_eq_zero` needs `StrictlyLower`, and `one_not_nilpotent`
(`Nilpotent.lean:105`) is the file's own warning that the strictness is load-bearing. $(I-\gamma P)^{-1}$ is therefore an *infinite*
Neumann series, not the $n$-term sum, so "the resolvent computes all $s$ hops in one operator by nilpotency" holds only for the
diagonal-free operator. The paper names two regimes and says which theorem carries which: **regime N**, the strictly causal corner,
where `occupancy_is_exact_inverse` applies verbatim; **regime S**, the softmax corner, where §4.2's certificate pays instead. On the
non-nilpotent corner the masked route reads $1.99\times10^{-7}$ against the unmasked $6.27\times10^{-13}$ and the naive
$7.96\times10^{-8}$ (**RUN**, ledger row P11), so the masked identity is a value-channel statement conditioned on $V_0=0$, not a
matrix identity, and the default keeps the diagonal. Cards **J-L3**, **J-L5**, **J-L16**. Mechanism **P-3** (`:316`): the nilpotency
sentence was true of the record's operator and would otherwise have been carried forward unqualified.

**The refutations proved in-file.** Fourteen declarations ship the statement they are *not*, each with a witness in the same file —
the structural answer to **V-3** (`:72`) and **V-10** (`:168`). Four are load-bearing. `V15.gate_zero_not_stochastic`
(`V15.lean:237`, hypothesis $1\le i$) kills the round-11 parity clause in the multiplicative reading: row $i$ sums to $i+1$,
smallest witness $i=1$ with sum $2$, so $g\equiv0$ lands on **linear** attention, and only the additive-logit reading
(`gate_zero_logit_identity`, `:251`) gives parity. `V16Domain.no_prefix_scan_represents_a_zero_gate` (`:165`) forecloses the whole
prefix-scan route to exact segmentation, which is why the F0 certificate is the product itself or a segmented scan resetting at
zeros (`READ ceq/arm_smprime.py:157-161`). `V16Domain.lean_log_junk_makes_the_scan_form_silently_false` (`:147`) records that
relaxing $0<a$ to $0\le a$ while keeping the $\log$ form is *silently* wrong — `Real.log 0 = 0` is junk, and the file exhibits `Wp 0
0 1 0 = 1` beside `pathProd 0 0 1 0 = 0`, which is why the zero-gate machinery is written on `pathProd`.
`V15Source.inverse_identity_is_vacuous` (`:83`) refuses reading the contract's Lean #15 as $MM^{-1}=1$: true of every unit of every
ring, dependent on no axioms, used nowhere. Beside them `zero_not_a_counterexample` (`OracleSeparation.lean:193`) records that
dropping $0<Q_{ij}$ leaves the statement *true* — a vacuous control caught in proof form, and the template every Lean card follows
(**J-L0** to **J-L18**).

## 4.2 Measured

**Table 4.2a — the identity binds.** A bind whose rejection region is empty is **V-24** (`MISTAKES.md:1658`), so every row ships a
planted mutilation failing at $O(1)$; the binds run on BED-M's real support $\{-1,0,+1\}$ rather than a toy draw, which is **V-25**.

| bind | residual | bar | planted negatives (residual) | source |
|---|---|---|---|---|
| oracle gates give the label, path-product corner | $5.919777\times10^{-16}$, real part exactly $0.000\times10^{0}$ | $10^{-6}$ | `drop_phase` $1.934830$; `drop_magnitude` $0.466267$; `beta_one` $1.335288$; `exp_scan` `nan` on $133{,}120/133{,}120$ | `READ V16_ARM_SMPRIME.md:25,195-199` |
| label bind, `arm_pl` telescope head | $6.661338147750939\times10^{-16}$ at $s=8$; $7.549516567451064\times10^{-15}$ at $s=64$ | $10^{-6}$ | `drop_key_bias` $0.97494590151405114$; `drop_value_rescale` $0.9165274652308163$; `drop_bos_sink` $1$; `half_key_bias` $0.48449311856985267$ — four for four | **RUN** `results/v15_r1.jsonl`; `READ V15_ARM_PL.md:183-200` |
| label bind, `arm_phase` | $9.155133597044475\times10^{-16}$ at $s=8$; **`nan` on BED-M's band** ($\lvert a\rvert=1$ gives $\log(1-1)=-\infty$) | $10^{-6}$ | the band failure *is* the planted negative, and is not adjusted away | `READ V15_ARM_PHASE.md:36` |
| softmax corner against its own `softmaxAttn` | **bitwise**; against `ceq/lm.py::Attention("softmax_x")` $1.110223\times10^{-16}$ on $19/64$ entries — a fused kernel subtracting the row max, a named mechanism and not a tolerance | bitwise | wrong switch: $\beta=0$ at the softmax corner reads $\max\lvert\mathrm{gap}\rvert>0.5$ | `READ V16_ARM_SMPRIME.md:27,266-293` |
| DAG resolvent against brute-force path sums | $2.7755575615628914\times10^{-17}$ (light gates); $3.5527136788005009\times10^{-15}$ absolute, $2.1572667249528595\times10^{-16}$ relative (heavy) | $1.1\times10^{-16}$ | the two routes share **no code** — dense LU against enumeration of increasing vertex sequences, a **V-3** guard | `READ V15_ARM_PL.md:153-177` |
| $\gamma=0$ parity of the shape's read | `torch.equal` returns `True` against $PV$ | bitwise | rejection region at $\gamma=0.5$: max-abs $2.3002850040264393$ | **RUN** |
| Neumann truncation certificate | equality to $10^{-15}$ at $K\in\{1,2,4,8,16\}$, $\gamma=0.7$, $s=64$ | attained | a non-stochastic $P$ (rows scaled to $1.5$) reads $119.37486584159647$ against the bound $1.1433333333333329$ | **RUN** |
| corner 3 is the sub-diagonal resolvent | max-abs $0.0$ entrywise; label $6.217248937900877\times10^{-15}$ | exact | $A^{s}=0$ exactly | **RUN** |
| committor is the resolvent read with absorbing rows | max-abs $0.0$ on BED-1's **real** sets ($A=[0]$, $B=[1]$, $\lvert T\rvert=9$); harmonic residual $1.0408340855860843\times10^{-17}$ | exact | a must-fire perturbation drives $\max\lvert Lq\rvert$ to $1.000000\times10^{-6}$, a ratio of $9.608\times10^{10}$ | **RUN**; `READ V15_BED1.md:123-124` |
| triangular solve against the dense inverse | $1.7763568394002505\times10^{-15}$ at $\gamma=0.5$, $s=64$, $d=16$ | $10^{-12}$ | — (the cost claim, not the identity, carries the risk; rule 8 below) | **RUN** |

The two-branch form $\mathrm{softmax}(q)V_1+\lambda XV_2$ passes parity bitwise for the honest gate, for Gaussian noise, **and for
the label itself** — an empty rejection region, filed **V-24** (`READ workdonenewseal.md:124-126`). The plan inherits the battery,
not the claim: cards **S-20** to **S-27**, every plant journalled as a FOUND cell with its own `kind` (**S-28**, against **V-7**,
`:117`).

**Table 4.2b — the deciding measurement R1, and its re-take on the certified device.** The row that matters is the split, never the
mean: a marginal standing in for a joint claim is **V-26** (`MISTAKES.md:2190`).

| quantity | CPU, `results/v15_r1.jsonl` | CUDA re-take, `results/v17k_r4_retake.jsonl` |
|---|---|---|
| cell | BED-M, `e3_t2`, $t^\star=2$, $s=64$, $d=24$, $d_{\rm model}=16$, $n_{\rm train}=2048$, 150 steps, 8 threads, $N=8$ | the same, `device="cuda"`, instrument hash `5d41a63d...9a309` on all 24 cells, `deterministic_algorithms: True`, `warn_only: True` |
| $\mathrm{floor}_1=\sqrt{(t^\star-1)/t^\star}$ | $0.7071067811865476$ | the same |
| `arm_pl` | mean $0.829151$, sd $0.253673$, CI95 $[0.617075,\,1.041227]$, $\hat h=0.625017$, `crosses False` | mean $0.830200$, sd $0.255414$, CI95 $[0.616669,\,1.043731]$, $\hat h=0.621535$, `crosses False` |
| `softmax` | mean $0.951767$, sd $0.011824$, CI95 $[0.941881,\,0.961652]$ | mean $0.952561$, sd $0.011452$, CI95 $[0.942987,\,0.962135]$ |
| `arm_smprime` | no CPU counterpart exists | mean $0.822193$, sd $0.250321$, $\hat h=0.647997$; **1 of 8** below the floor (seed 2 at $0.20391993939877656$) |
| **the bimodal split** | crossed **5 of 8** ($0.634002$ to $0.662021$; gate-$R^{2}$ $0.97$ to $0.99$; $\hat a_{\max}$ $1.10$ to $1.51$); NO READING **3 of 8** ($1.113403$, $1.139404$, $1.152430$; gate-$R^{2}$ $0.011$, $0.627$, $0.047$; $\hat a_{\max}$ $20.31$, $49.66$, $285.07$); **nothing between $0.663$ and $1.113$** | identical membership; **zero** verdict sign flips across 16 paired cells |
| paired contrast (PL minus softmax) | $-0.122616$, sd $0.257560$, resolution $\Delta=t_{.975,7}\,\mathrm{sd}/\sqrt{8}=0.215326$ | $-0.122360$, sd $0.260159$, $\Delta=0.217498$; the device moves the contrast by $+2.56\times10^{-4}$, $+1.0\%$ |
| worst CPU-to-CUDA delta | — | $9.522\times10^{-3}$ at `arm_pl` seed 7 ($1.139404\to1.148927$), $4.06\times$ the thread floor; two other non-learning seeds read $1.505\times10^{-4}$ and $6.329\times10^{-5}$, so "did not learn" is not the mechanism |
| identical-seed floor | — | $\delta_{\rm nrmse}=0.0$ bitwise on 6 of 6 pairs, `results/v17k_r4_floor.jsonl`; the seed-to-seed spread is at least $10^{4}$ times larger |
| cost | $1358.3225734233856$ s for 16 cells | $2.79$ GPU-min measured against $2.65$ projected ($+5.3\%$); peak $0.9739$ GiB, $12.2\%$ of $7.996$ GiB |

**The mean $0.829151$ is a value no seed produced.** That is why the plan reports every capability number as a split with $n^{+}$,
the seed-agreement count, beside it (**V-26**; card **S-61**, the adjudicator specification written before data). The recorded
verdict is NOT CROSSED with the interval straddling, and two campaign firsts sit inside the negative: $\hat h=0.625$ against a
nine-cell ceiling of $0.389$, and five seeds below a floor no prior arm had crossed at any cell. Deduplicated across the three
journals (**RUN**, 43 rows to 40 unique on `(kind, seed)`), **13 of 40** banked cells sit below $\mathrm{floor}_1$,
13 read $\hat h>1.0$, and $s=64$ on 40 of 40. The corners are measured distinct at the distances tabled above, with the switches moving the
operator by $0.673101$ ($g$) and $3.522037$ (QK) — mechanism **V-2** (`:58`), a containment whose corners coincide being decoration.

**Table 4.2c — the device certificate (RTX 4060 Laptop, sm_89).** Capability $8.9$, $8{,}585{,}216{,}000$ B ($7.996$ GiB), torch
`2.5.1+cu121`, Python `3.11.9`, Windows-10, eight threads, matmul TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8` (**RUN**
`results/k_cert_local.json`).

| law or constant | value | $R^{2}$ / verdict | note |
|---|---|---|---|
| `softmax` throughput | $\mathrm{s/step}=\exp(-12.18515596)\,n^{0.99625107}$ | $0.9999975$ | 5 points, $n\in\{2048,\dots,32768\}$ |
| `arm_smprime` throughput | $\mathrm{s/step}=\exp(-10.01874582)\,n^{1.00258069}$ | $0.99999987$ | **3** points; at $n=16384$ and $32768$ the allocator reserves $10.578$ and $13.969$ GiB from a $7.996$ GiB card and pages over PCIe, and admitting them bends the exponent to $1.2341$ at $R^{2}=0.976933$. A throughput law fitted through swap is not a throughput law (**P-8**, `:387`) |
| memory constants, fp32 | `C_RESIDUAL` module $18$, re-solved $17.874$; `C_OPERATOR` module $3.9$, re-solved $3.823$ | $0.996373$, both CONFIRMED at $-0.7\%$ and $-2.0\%$ | the module holds where it was fitted |
| bf16-autocast residual | module $2.2$, re-solved $2.383$ | $0.999830$, **WRONG, $+8.3\%$, optimistic** | at the Q3 chunk shape $2.715$ against $2.738$ GiB ($+0.85\%$), `max_batch` $45$ either way — no decision moves, so it is reported and not repaired |
| the complex arm (`arm_phase`) | measured over predicted $1.842$; `C_OPERATOR` $7.50$ at 8 B/element, $4.29\times$ softmax's | — | the module breaks exactly where it is new; `C_RESIDUAL` NOT IDENTIFIED at $s=64$ |
| determinism | `cumprod` has a deterministic CUDA kernel, so `arm_smprime`'s forward is bitwise ($\max\lvert\Delta\rvert=0.0$, 8 repeats), but autograd differentiates it through `cumsum`, which has none, so **the hole moved and did not close**: the strict flag raises and gradients are bitwise only with it off. Run to run, $\delta_{\rm nrmse}=0.0$ on 6 of 6 identical-seed pairs, and 6 of 6 flag-OFF cells reproduce flag-ON **bitwise** (`==`, not `allclose`) | — | `cumsum` is the whole of `arm_pl` and `arm_phase`; Ruling 1; `READ COSTS.md:149-154` |
| cpu against cuda `cumprod` | $11/64$ entries move by at most $5.551115\times10^{-17}$ (sequential against parallel scan); **zeros identical on both devices**; on BED-M's own support $0/289$ entries move | — | association order does not matter where it would change a verdict |
| bar re-certification | `BAR CALIBRATED` on four rungs and both devices; worst $\delta$ $8.580024779547557\times10^{-8}$ against a $10^{-6}$ tolerance ($8.58\%$, headroom $5.83\times$ to the 50 % HALT line); `oracle` exactly $0$ everywhere | — | 30 doubles bit-identical in IEEE-754 hex against the previous revision |

Mechanisms **M-8** (`:544`, pricing every arm at one arm's rate) — one law per arm with its $R^{2}$; **V-22** (`:1140`, a constant
certified under conditions the reading does not reproduce) — the flag regime is journalled on the header; **P-8** — the $R^{2}$ gate
refuses paged points. What the certificate does **not** contain is an exponent in $s$: $s=64$ on 40 of 40 cells, and the per-cell
timer is un-synchronised host wall clock whose strongest correlate is run order, $\rho=+0.7029$, $p=0.0024$, above the gate
correlation $+0.5197$ it would have to be separated from. That is **D-3** (`:727`), discharged only by card **S-66** — the $s$-sweep
with `synchronize()` and randomised order, priced at $206$–$537$ GPU-s, band only, with the run-order confound *surviving* those
edits unless a fifth (blocked order) is made, which no office has priced.

**The calibration column.** Round 11: **9 checked, 9 adverse** on the filed rows (at least 17 checked, 10 adverse, with the
confirmations the table omits). Signs: **7** optimistic, **1** pessimistic (a sizing repair already made, reported as outstanding),
**1** unsigned. The one-sided sign test on the eight signed rows gives $7/8$, $p=0.0352$ — direction established at $\alpha=0.05$ —
and the Wilson 95 % interval on the optimism fraction is $[0.5291,\,0.9776]$, which excludes $0.5$ and fixes nothing else, so it
**licenses an ordering only**. The round's own headline "six wrong, all six optimistic" gives $5/6$, $p=0.1094$ on the honestly
signed rows: right about the direction, wrong about its evidence. The single sub-census with a real denominator reads 11 checked, 4
wrong, $36.4\%$, Wilson $[0.152,\,0.646]$, so $9/9$ is a count and not a rate. The discount rule §5 adopts verbatim: **D-CALIB-1**
the counter is the point estimate; **-2** a bare prediction is blocked, never discounted; **-3** the licence is a *sign*, never a
size, so **no numeric shrink factor is authorised**; **-4** the cheapest refutation of the optimistic half runs first; **-5** the
row is appended whether or not it flatters. Mechanisms **D-7** (`:2037`), **M-2** (`:451`); cards **V-0**, **V-1**, **V-19**.

**Numbers this paper must not use.** The registry holds **12** struck constants (`READ STRUCK.md:18-33`, rendered at `aa82df7`),
named here by quantity.

| struck quantity (registry rows) | why it was struck | what the paper writes instead |
|---|---|---|
| the M2 decay exponent and its $R^{2}$ (1–2) | a `floor = 1e-6` artefact; the two candidate repairs disagree ($R^{2}$ $0.9990$ against $0.9662$) | **delete the claim**; no replacement is published |
| the "live rows only" K1 slope and both interval endpoints (3–5) | UNVERIFIED — the interval exists in no `.py`, `.json`, `.jsonl` or `.txt`, and the only live producer emits a different number and exits 1 | the as-computed slope $-0.4137$, CI $[-0.4579,\,-0.3704]$, reproducing from `scale/arm_a_k1.py` |
| the M2 slope as first reported (6) | contradicted by measurement, and propagated in a `[RUN]` voice | the shipped operator's reading, carried in §4.3 as a *verdict* and not as an exponent |
| the M5 tail norms at $s=128$ and $s=512$ (7–8) | FABRICATED — an 1,800-setting sweep produced neither | measured $0.880500$ (hops $=2$), $0.882030$ (hops $=4$) at $128$; $1.292741$ at $512$ |
| the U1/N3 pilot rank correlation and both CI edges (9–11) | NO PRODUCER HAS EVER EXISTED — nine named functions defined in no commit on any ref | nothing; the item is void |
| the Karcher residual in float64 (12) | asserted `[RUN]` from a throwaway script whose own output was `nan` | $7.481\times10^{-9}$, $8.155\times10^{-9}$, $8.405\times10^{-9}$ at published settings; $7.307\times10^{-13}$, $7.958\times10^{-13}$, $8.405\times10^{-13}$ at `--tol 1e-15 --steps 400` |

**The named stale traps, with their corrected figures.** "No arm crosses $\mathrm{floor}_1$" becomes 6 of 24 at it.2 and **13 of
40** at it.14 (C1, C15); "6 of 34 cells violate the floor" becomes **13 of 40** (**RUN**); "12 of 16 cross" becomes "12 of 16 sit
below the floor, and the pooled verdict is `crosses: false`", the flip turning on seed 9 alone (C13). $p=6.730\times10^{-4}$ was
**unpaired** — the honest paired figure on the banked record is $p=0.012821$, a factor of 19, repaired by measurement at it.10 to
**8 of 9 against 0 of 9**, paired, over three draws (C14). Every GPU-second and cost ratio derived from `scripts/v15_r1.py`'s `secs`
is void, the K-cert laws being the replacement (C17). The "$5.8\times$ wall-clock gap" and softmax at $71.32$ s per 150 steps do not
reproduce: the same arm, corpus, thread pin, steps and statistic re-read $12.3$–$13.6$ s on a quiet host, and what survives is
E-core placement plus host contention, $41$–$83\%$ of the gap in log terms with the residual unexplained. The "$13.6\times$
CPU-to-GPU decision" becomes $11.94\times$ at the point estimate, $29.0\times$ at the pessimistic end.
$\sum_j\lvert W_{ij}\rvert=1.000000$ as the softmax-corner certificate equals $Z_i^{1-\beta}$ and is blind to $g$; write
$\sum_j\mathrm{Re}\,W_{ij}$, at $2.220446\times10^{-16}$ from $1$. "A pole precisely on the unit circle" is `torch.clamp(u, 0, 1)`
at `ceq/arm_smprime.py:113` — a ceiling, not a converged pole (C9). "The arms return a state distribution" is false: 0 of 40 banked
cells journal one. And "beats softmax" on any BED-M cell is banned by **R-SKY**; the paired contrast with its resolution statement
is what the paper writes.

## 4.3 Refuted, by mechanism

**Table 4.3a — the dead programmes, each with the number and interval that killed it.**

| programme | the number | the control / interval | source |
|---|---|---|---|
| the signed strictly-causal path sum (`sgate`), and its downstream task | sign-flip rate $0.16511$, $0.02732$, $0.00000$, $0.00000$ at $s=8,32,128,512$ — a decay of order $1/s$, killed against a bar of $-0.3$ by a factor of four under **every** floor tried; downstream, COGS-gen **$0/512$** against softmax $15/512$, one-sided Fisher $p=2.75\times10^{-5}$, in-distribution $0.7734$ against $0.9258$ | softmax control exactly $0.0000$, $0/1024$ flips; $3{,}652{,}096$ parameters matched on both arms; $\texttt{zero\_success\_upper\_bound}(512)=0.005834$. The fitted exponent is **not carried**: it rests on two nonzero points, a different geometry reads $-0.958$ at $R^{2}\,0.9990$, and the audit's third reading reconciles with none of its own five rates | `READ PROGNOSIS.md:40-63,357-364`; `READ RESEARCH.md:178-194` |
| pivot routing | dense slope $-1.088$ against pivot $-1.298$: routing makes the decay **worse**. The paper carries the ordering, not the exponents | at $s=8$ routing restricted nothing — $A[:,P]\,A[P,:]$ equals $AA$ to $1.86\times10^{-9}$ | `READ PROGNOSIS.md:44-45,79-85` |
| the exclusion confound beneath it | the maximum legal pivot is $s-2$ while the label $a_{s-1}b_{s-2}$ sits on the hidden position; a $+100$ perturbation moves softmax by $101.6983$ and the pivot path by $3.263746$ | lifting it: `twin_plus` $0.938728$ against a threshold of $0.871391$, which is $0.015610$ **worse** than the twin; the inert version selected $s-1$ in $0/32$ draws (**V-9**) | `READ MATHEMATICS.md:329-347`; `READ D1.md:385-398` |
| settling the *mixture weights* over pivots | $\text{settled}-\text{twin}=-0.002959$, exact 95 % CI $[-0.042903,\,+0.031557]$ over all $3125$ paired resamples with $126$ atoms | settled sd $0.064106$ against twin $0.016547$, $3.874\times$; $\text{argmax}-\text{softmax}=-0.118456$, CI $[-0.134115,\,-0.102786]$ | `READ MATHEMATICS.md:355-380` |
| the hop-2 term on the softmax operator | filed blind, $4/4$ held: $0.976488$ (sd $0.004039$, CI $[0.973932,\,0.979036]$) against softmax $0.975371$, a contrast of $+0.001117$; $\hat h=0.372$ | **five independent routes agree**: a $K$ sweep; a $\gamma$ sweep in which no $\gamma$ beats $0$; deflation; a Wiener route at $979\times$ attenuation; a gated hop at $0.966692$, worse than both | `READ V13_PREDICTION_HOP2.md:35-46`; `READ workdonenew.md:266-294` |
| VGPE, path-ordered non-abelian transport as positional encoding | occupied outright by PaTH §2.1, $A_{ij}\propto\exp\!\big(k_j^{\top}(\prod_s H_s)q_i\big)$, with RoPE recovered as $H_s=R$ | the Hankel rank is $d$ regardless of alphabet | `CITED [V]` `yang-2025-path`; `READ PRIOR_ART.md:742-778` |
| the original parity clause, "$g\equiv 0$ gives bitwise standard attention" | row $i$ sums to $i+1$; smallest witness $i=1$, row $(1,1)$, sum $2$ | `gate_zero_row_sum`, `gate_zero_not_stochastic`; repaired by the telescope at $2.2\times10^{-16}$ | `READ workdonenew.md:47-124` |
| "the normalizer obstructs path products" | false: it dissolves uniquely at $\gamma_j=1/(1-a_j)$ | the additive-logit repair computes $y_i/R_i$ to $4.44\times10^{-16}$ | `READ workdonenew.md:118-124` |
| $X_{35}$ hidden-cause inference | occupied since **1993**: Basseville and Nikiforov §7.2.4 in closed form, equation for equation, with **both** must-fires discharged by the 1993 equations on the first attempt | the learned-model form is `[U]` and is the record's own analogy; correction (5) below | `CITED [U]` `basseville-1993-detection`; `READ README.md:214-219` |
| closed-magnitude phase gates | occupied: S4D's ReLU variant reads $\lvert\bar A\rvert=1.0$ exactly on **$32.93\%$** of a standard sample, in 2022 | $m=0$ is attainable too — modReLU, at $14.70\%$ | `CITED [U]` `gu-2022-s4d`; `READ V15_X36_PRIOR_ART.md:396` |
| R1, the deciding measurement | the CI $[0.617075,\,1.041227]$ **straddles** $\mathrm{floor}_1$; 5 of 8 cross, 3 of 8 are NO READING | $1/(1-\hat a_{\max})$ is undefined at all eight seeds; the capability clause was not earned | `READ V15_R1.md:51-56` |
| $\mathrm{floor}_1$ read as an **information** floor | violated by **13 of 40** banked cells (12 `arm_pl`, 1 `arm_smprime`) — a lower bound is never violated, and this one is | it is the locus of $\hat h=1$: a one-hop **capability threshold**. The real BED-M floor is the exact oracle at $0.0$, and no annex theorem predicts either wing's distance to it (best $0.203920$, modally about $0.85$) | `READ V20_R15_THEORY_TABLE.md:69-91`; C11, C15 |
| Q6, the state metric | **F4, the domain is empty**: both wings return $[n]$ scalars and 0 of 40 cells journal a distribution | marginal $W_1$ reads $0.0$ on the oracle's own values permuted while NRMSE reads $1.421901$; metric and bed rank two predictors in opposite orders by $14.465\times$ | `READ V20_R15_THEORY_TABLE.md:211-225` |
| the M6-against-M2 tension (`IMPOSSIBLE.md` I1) | $w(A)=1.499315>1$, so $2w^{h}$ grows from $2.999$ to $10.11$ while $\lVert A^{h}\rVert$ falls from $2.838$ to $0.210$ over $h=1..4$; every eigenvalue guard vacuous, the spectrum being $\{0\}$ | **resolved by Proposition 4**, not repaired: with a row-stochastic $P$ the normalizer pays for a denominator-free certificate carrying **no sum over $s$**, and the sign capability bought by dropping the normalizer is not recovered — it is replaced by absorbing rows | `READ IMPOSSIBLE.md:34-76`; ledger row P4(i) |

Three readings the table cannot carry. Pivot routing's deeper one — *`pivot_signed` was `pivot_unsigned` wearing a name*: at the
harness geometry the operator is entrywise non-negative with minimum entry exactly $0.0$, so the paired test that credited sign
compared two non-negative operators, and every M2 and S2 headline was measured on `tgate`, which ships nowhere; card **S-01**
extends the identity manifest for that reason. The hop-2 mechanism transfers — *the label composes values along paths; the arms
composed weights across positions* — which is the algebraic content of the shape's second identity and the reason §2 makes the
resolvent the operator instead of adding a hop to softmax. And the native skyline is DeltaNet's WY matrix, the exact inverse of a
strictly lower-triangular, content-dependent, signed operator shipping in `fla/ops/utils/solve_tril.py` (`CITED [V]`
`yang-2024-deltanet`, `yang-2024-gateddeltanet`, `dao-2024-ssd`): "beats native" is banned for the reason "beats softmax" is, so the
shape may claim a $\Delta_{\rm sky}$ column with an interval plus the *mechanism* difference — no absorbing rows, no committor read,
no re-solve under $\mathrm{do}(a)$ — written `NOT FOUND — sweep owed`, never "novel".

**The taxonomy.** `MISTAKES.md` carries **66** mechanism headings (**RUN** `grep -cE '^### [VPMD]-' MISTAKES.md`; the record's own
65 omits the sub-entry `V-14a` at `:770`) in four classes, ordered by the cost the record assigns them: **V**, vacuous controls, a
control that cannot fail (26); **P**, provenance, a number with no live producer or a claim true once (11); **M**, measurement,
where the instrument measured but not the thing the verdict names (21); **D**, design, where the answer was fixed before any data
arrived (7). The largest is **D-1**, *the largest one in the repository, and it subsumes most of the null results*.

**Table 4.3b — the ten most likely to recur here, the rule each imposes, and its card.**

| # | mechanism | why the shape invites it | the rule it imposes | card |
|---|---|---|---|---|
| 1 | **D-2** (`:710`) the oracle is the arm's own resolvent | the shape's state *is* $z=(I-\gamma P)^{-1}V$, so a bed labelled $(I-\gamma^\star P^\star)^{-1}V^\star$ makes every contrast a reproduction check | the label is generated by a **latent environment chain** $P^\star$ the arm never sees; the VOID-contrast list is registered before any cell runs; $\lVert\hat P-P_{\rm env}\rVert_\infty$ is a learnability reading, never a capability one | **S-13**, **V-9** |
| 2 | **D-1** (`:677`) racing a proven optimum | every bed in the tree scored one real at position $s-1$ | **no bed scores a scalar at one position**: the label is the vector $z^\star$, the displacement field $\Delta z$, or the argmin over candidate moves; the regime in which the softmax skyline is optimal is stated with the $L/d$ geometry printed | **S-11**, **S-16** |
| 3 | **V-24** (`:1658`) an identity bind with an empty rejection region | $\gamma=0$ giving $PV$ is proved by one fact and says nothing about $P$, the boundary sets, or the solve | every bind ships the mutilation battery — drop the absorbing rows, drop the solve, replace $P$ with noise, substitute the label for $z$ — admissible only if each mutilation fails at $O(1)$ with counts printed | **S-20**..**S-27**, **V-4** |
| 4 | **V-8** (`:136`) / **V-12** (`:189`) the boundary sets make the label constant | one absorbing target collapsed a label to $1.11\times10^{-14}$; a one-component graph gave label sd $0.0$ | the builder prints at construction $\mathrm{sd}(q^{(k)})>0$ for every $k$, class frequencies inside $(0.05,0.95)$, the argmin-uniqueness fraction, the disagreement fraction, the discard count | **S-12** |
| 5 | **V-25** (`:1954`) a theorem whose hypothesis no draw satisfies | the certificate needs $P$ non-negative row-stochastic with $\gamma<1$; the record's own $0<a_k$ theorems admitted 1 of 3 | a domain census beside every gating theorem — the measured support of $\gamma$ and of $\mathrm{rowsum}(P)$ on drawn cells, at the quantifier level the theorem uses; $0\%$ blocks the gate | **J-L18**, **S-42** |
| 6 | **M-13** (`:1210`) / **M-9** (`:580`) a margin or $\alpha$ the design cannot reach | parity needs $N=70$ for TOST power $0.80$, and at $N=8$ two bit-identical arms return NO VERDICT | parity is claimed by identity bind with rule 3's battery, never by TOST at $N<70$ (paired $N\approx36$ only once $\mathrm{sd}_d$ is measured — currently NOT MEASURED); every seed CI prints $n^{+}$ | **S-65**, **V-18** |
| 7 | **M-21** (`:2101`) / **M-18** (`:1483`) a diagnostic whose value the corpus fixes | the gate was written into the context for the whole campaign, and a corpus-alone probe reads gate $R^{2}=1.000000$ with **no arm** | the contract names the *discrimination*, never the statistic; a statistic is admitted only after three readings — corpus-alone ceiling, zero-step floor, headroom printed; $\mathrm{Var}(z)=0$ on the corpus blocks registration | **S-14**, **S-15** |
| 8 | **D-3** (`:727`) / **P-8** (`:387`) a dial that does not vary, a price that is an upper bound | $s=64$ on 40 of 40 cells with the exponent unidentified; a headline of about $29.6$ h was $5.9$ h | the cost law is measured at $s\in\{64,256,1024,4096\}$ with `synchronize()` and randomised order; every headline price carries its direction and the implementation it assumes | **S-66**, **V-21** |
| 9 | **D-7** (`:2037`) / L-SIGN a prediction without its counter | nine contract statements, seven optimistic, $p=0.0352$ | every prediction ships a counter of equal specificity, filed and hashed **before** the first arena cell; the calibration column is appended whether or not it flatters | **S-60**, **V-1**, **V-19** |
| 10 | **V-3** (`:72`) / **V-10** (`:168`) an assertion that is an identity of its own construction | the Neumann bound is *attained*, not slack: for non-negative row-stochastic $P$ the tail's row sums are exactly $\gamma^{K+1}/(1-\gamma)$ (**DERIVED**), so "error at most bound" is a row-sum identity | before any inequality enters a verdict, state what set each side ranges over; where the right side is computed from the left's hypotheses alone, label it *definitional*; the certificate is measured on the **shipped mask** in vector units, never on the truncated series | **S-41**, **S-43**, **V-20** |

Three more sit one step behind: **V-17** (`:855`, a threshold out of its units — a committor threshold is stated in NRMSE or floor
units, never raw feature distance); **M-15** (`:1289`, a statistic without its sharpness-matched null — any $\beta_0$ persistence
number needs its random-operator null first, card **S-71**); **M-2** (`:451`, a threshold refitted after the data — every §5
threshold is frozen with provenance beforehand, which is why the round declined to pick its own tail ruling).

**The eight corrections to the record found while writing this paper**, each a defect in the repository's own documents found by
reading them against their sources, and filed here rather than by editing a verbatim-of-record file.

1. **The contract's M10 lineage identifiers are all misattributed.** arXiv:1905.12200 is Bruel-Gabrielsson et
   al., 1904.09378 is PersLay (`carriere-2019-perslay`), 2011.05804 is Corcoran and Deng
   (`corcoran-2020-ph-gradient-regularization`); the intended papers are Hofer 1906.09003
   (`hofer-2019-connectivity-optimized`), Moor 1906.00722 (`moor-2020-topological-autoencoders`), Carrière 2010.08356
   (`carriere-2021-optimizing-ph`). **P-5** (`:343`) compounded by **P-10** (`:1382`).
2. **The Cheeger line is stated on a non-reversible chain.** `MATHEMATICS.md:455` cites Levin–Peres–Wilmer Thm
   13.10 (`levin-2017-markov-mixing`, read at p. 183: $\Phi_\star^{2}/2\le\gamma_{\rm gap}\le2\Phi_\star$),
   whose hypothesis is **reversibility**; a causal $P$ with $P_{i0}>0$ has $P_{0i}=0$, so the line is a **V-25** exposure. It is
   restated on the transient block's lazy symmetrisation $Q_s:=(Q+Q^{\top})/2$, whose Cheeger bound controls that surrogate's
   spectral gap and **not** $\rho(Q)$, and moved to the oracle cross-check chains (BED-1, E4′). Card **J-D6**.
3. **Three Lean declarations the contract tags `[M]` do not exist.** `CEQ_V20_R15_CONTRACT.md:179`, `:185`,
   `:213`, `:260` cite Lean **#18**, **#19**, **#22** as machine-checked; **RUN**,
   `grep -rn 'segment' lean/CEQ/` returns one hit, a prose comment at `V16Domain.lean:53`, and no declaration
   carries that content. The `[M]` is the contract's *grade*, not a build, and the F0 instance the same line
   prints ($8.9\times10^{-16}$, $595\times$) has **no located producing file**. **P-11** (`:1597`) realised;
   re-registered as work in cards **J-L6** and **J-L7**.
4. **`THEORY.md:22` calls the successor operator new.** Its audit column already contradicts its claim column,
   and the operator is Dayan 1993 (`dayan-1993-successor`, whose eq. 3.1 is $(I-Q)^{-1}$ with **no** $\gamma$,
   per `Occupancy.lean:6-17`). **P-3** (`:316`).
5. **arXiv:2604.25655 does not cite Basseville and Nikiforov.** "The learned-model form of §7.2.4" is the
   record's own analogy, not the source's claim, and is carried `[U]`. **P-10**.
6. **The journal's head phase table is stale**, still reading *it.15 DONE, it.16 next* for Phase C and *not
   started* for Phases D and E while the body carries entries through it.35 — **P-3 sitting inside the block
   built to catch P-3**. The plan reads its start state from the it.35 entry.
7. **Rulings 11 and 12 have no text in the tree**, although `CEQ_V20_R15_CONTRACT.md:59` binds "Rulings
   1–12": **RUN**, `grep -rniE "ruling 1[12]\b" *.md` returns zero hits, and the same holds for `L-GRADE`,
   cited once with no rubric anywhere. The plan states the rubric it uses and marks it a reconstruction.
8. **The scoreboard is 2 of 44 and the leap call is still owed.** Both points are the it.4 freeze; the theory
   table's $+6$ was refused twice and carried unclaimed; the single leap call was **not spent**; it.36 to
   it.45 are unspent.

**The bibliographic traps.** The audit merged eight sweep bibliographies into **408** canonical entries (508 parsed, 51 same-key
duplicates dropped, 49 same-identifier entries folded through `bib_aliases.md`) and re-ran the identifier check directly against
arXiv and Crossref, the audit service having returned "not subscribed": 278 of 278 arXiv identifiers resolved, 91 of 92 DOIs
resolved, and all seven title-agreement misses were subtitle truncations of the same works; the one unresolved entry,
`singh-2007-mapper`, is a Eurographics identifier absent from Crossref and is carried `[U]`. Four traps were caught by hand: DOI
`10.1007/978-1-4684-9455-6` is *Denumerable Markov Chains*, not *Finite Markov Chains*, and must not be attached to the Kemeny–Snell
entry; arXiv:2501.00663 is Titans, not test-time regression (2501.12352); arXiv:2302.11294 is not Sander et al. (2302.01425);
Contreras Arredondo et al. carry the journal title *Learning the committor without collective variables*. One further trap was found
**RUN** this session by scanning the merged file for duplicate titles: **three** groups survive the identifier-level fold —
`altman-1999-cmdp` with `altman-1999-constrained`, `basseville-1993-abrupt-changes` with `basseville-1993-detection`, and
`kemeny-1960-finitemarkov` with `kemeny-1976-finitemarkov` (`kemeny-1976-finite` being a third key for the Springer printing under
its appendix subtitle). All are books carrying neither an arXiv identifier nor a Crossref DOI — precisely the 38-entry class the
audit says it could not check — so the fold is *known* incomplete there and the assembler picks one key per work before typesetting.
None of this establishes that a source supports the sentence it is cited for: that is the `[V-eq]` mark, carried per citation, and
the record's rule stands that theorem and equation numbers read through a rendered page are re-checked against the compiled PDF
before publication.

## 4.4 Where the campaign stands

**The scoreboard.** After thirty-five iterations of a forty-five-iteration round the carried score is **2 of 44** — both points the
it.4 wing freeze, nothing else banked. The denominator is 44 and not the contract's 48 because the *winner's cost at most one-tenth
of the incumbent's* bonus ($+4$) was shown unreachable by either surviving wing: on both timing bases W3 costs **more** than
softmax, $1.078\times$ and $1.059\times$. The theory table's $+6$ was frozen at it.14 and ruled NOT FIT for the leap, refused at
it.10 and again at it.11, carried unclaimed; the $+12$ is gated on a single unmade author ruling; the annex's $+4$ is forfeit. The
single leap call was **not spent**, for the reason on the record: *calling it now would consume the round's single shot on a table
whose header miscounts its own body, whose published grading rule contradicts its applied one, and three of whose cells the gate
cannot read.* Phase E has not begun.

**The frozen wings, and the struck one.** `FROZEN-N = 2`; `FREEZE-SHA256 =
fbf17e07e6cab495bd1fdbb4392ab88e92f5b0610e40d2b8a283d8bcb9c542ff` over eight normalised `wing|clause|path:line|anchor` rows — the
digest covers which wings, lines and anchor strings, and deliberately **not** the content of the cited files, which is why the
anchor check is a separate node re-run at every HEAD. W1 is `arm_smprime`, the path product $G_{ij}=\prod_k m_k e^{i\theta_k}$ with
$\beta$, QK and $g$ switches (`ceq/arm_smprime.py:144`); W3 is `arm_pl`, a real prefix scan, key-side only, over a value-zero BOS
sink (`ceq/arm_pl.py:88`). Wing identifiers are held at their it.1 values so the strike of **W2** is a visible gap rather than
hidden by renumbering. The discriminator that struck it, stated so it can be run: *a wing is FOUND if and only if `results/` holds
at least one journalled record whose `kind` field is the wing's arm name; otherwise it is NAMED.* The census by `kind` over every
`results/**/*.jsonl`: `arm_pl` **191**, `softmax` **182**, `arm_smprime` **182**, `arm_phase` **0** — a counter calibrated on both
sides, since it must read $0$ for `arm_phase` and above $0$ for `softmax`, so the zero is a measurement and not a broken search
(**V-7**, `:117`). **W2 is STRUCK and RETIRED to UNPRODUCED, not refuted**: the route back is one `make_arm` branch, an optimizer
path and a field emit, at $5.114$ s per cell on the `arm_pl` basis, roughly $41$ to $130$ GPU-s for eight cells. *The W2 gate is
under half a minute of wall clock on this box. It was never a cost decision.* Every arm the plan proposes enters under this
discriminator (card **S-28**).

**The eight gate cells, both offices, no coordination.** The theory table grades two wings against Q1–Q6: twelve cells, one F0, five
F1, one F2, two F3, three F4. The gate's population is **eight**, not the nine its own heading claims — the ninth row is Q1/W1 at
F0, *not a failure* — and one office rules Q4's two F3s a single harness fact, so the gate graded **seven distinct failures on eight
cells**.

| cell | grade | JUPITER | MARS | field, and the killer where one is named |
|---|---|---|---|---|
| Q1/W3 | F1 | LEAPABLE | LEAPABLE | realization theory (Hankel/Kronecker); formal verification of recurrence invariants by induction on a telescoping product |
| Q2/W3 | F1 + const | TERMINAL | TERMINAL | the bound $\mathrm{err}_i\ge\mathrm{dist}(t_i,\ \mathrm{hull})$ — row-stochastic implies a convex combination for every $(g,s,q,k)$, a nonexistence at any size; killer, one $(g,s,q,k)$ with $Z_i\ne1$ |
| Q3/W1 | F2 | LEAPABLE | LEAPABLE, conditional on an instrument swap | transfer-operator / Koopman spectral theory, attached to `lambda_hat_live` at `:384` and not `lambda_hat` at `:383` — one bit |
| **Q3/W3** | F1 + const | **LEAPABLE** | **TERMINAL** | bifurcation theory / gradient-flow convergence against "causal direction unidentified from $n=8$"; killer, a declaration predicting the sign of $\hat\lambda$ from initialisation alone |
| Q4/W1 | F3 | TERMINAL | TERMINAL | the exponent in $S$ is unidentified at $n=1$, and no theorem supplies a slope from one point |
| Q4/W3 | F3 | TERMINAL | TERMINAL | the same harness fact; `brute_force_path_sums` at `ceq/arm_pl.py:304` is $O(2^{S})$ and unguarded — an implementation guard, not a theorem |
| Q5/W1 | F1 + const | LEAPABLE, **field contested** | LEAPABLE, **field corrected** | approximation theory / Kolmogorov $n$-width — **not** the table's rate–distortion, on both readings |
| **Q5/W3** | F1 + const | **LEAPABLE** | **TERMINAL** | finite-mixture inference against "a mean over a bimodal population is not a statement about either mode"; killer, a simultaneous component-wise coverage theorem from pooled bootstrap draws |

Tallies: 5 LEAPABLE and 3 TERMINAL against 4 (one conditional) and 4; agreement on 6 of 8 tokens, both offices against the table on
Q5/W1's field. **The two divergences are a missing amendment, not a judgement call.** Q3/W3 and Q5/W3 are exactly the cells whose
verdict column carried a hybrid — *LEAPABLE, but about 6 GPU-s buys it outright* and *LEAPABLE, and it is a SCORING RULE, not a
theorem* — placed in the gate-class and field columns, where the ruling that a hybrid verdict is not a verdict does not reach;
beneath them, the ledger publishes one grading rule (LEAPABLE when the gap is a missing statement **or a missing measurement**)
while a ruling resolved a cell to TERMINAL against it — *the round has published one grading rule and applied another, twenty-one
iterations, it.14 to it.35.* Three further cells the gate cannot see at all (Q2/W1, Q6/W1, Q6/W3), one filing a **refutation**
inside a token meaning *unattempted*; that ruling is in its seventeenth iteration, unruled. Mechanisms **V-14a** (`:770`, the scope
test that condemns every refusal guard — the role must be declared) and **M-20** (`:1723`, a pre-registration predicting both
outcomes in two sections that never met).

**The laws in force, one line each.** **D-1**: work is a DAG, parallel dispatch only on nodes with no shared repository state.
**D-2**: planet names label responsibilities inside documents, never concurrent processes. **D-3**: no loop mounts until the
previous loop's cause of death is one sentence, and the iteration count comes from the DAG's critical path. **D-4**: contracts
behind an unreached round are staged, not started. **L-DOM**: every theorem that gates a run ships a domain census, and no overlap
means decoration (pays for V-25). **L-SIGN**: a counter-prediction of equal specificity beside every prediction, with a calibration
column across rounds (pays for D-7). **L-DIAG**: a contract prescribes what a diagnostic must distinguish, never which statistic
does it (M-18). **L-FLOOR**: every capability number ships beside its information floor, so "how good" reads as distance-to-floor —
and $\mathrm{floor}_1$ is not one. **L-CERT**: every mask ships its certificate, F0 exact or F1 with $\delta$ printed; a mask
without one is refused. **L-EQ**: `[V]` is inadmissible for a load-bearing statement, and `[V-eq]` requires the theorem with
hypotheses plus one numeric instance run. **L-LEAN**: the arm is not trained before its identity theorems are green. **L-G2**:
journals never move and are never deleted, superseded cells staying with a marker. **FOUND-not-NAMED**: a wing named rather than
found is struck. **R-SKY**: the native skyline is read beside every bed with a $\Delta_{\rm sky}$ column, and "beats softmax" is not
licensed where softmax is a fellow approximator. **Ruling 1**: CUDA determinism with `warn_only=True`, bitwise for replay and every
deciding forward cell. **Ruling 3**: matched parameters, a $0.032\%$ residual counting as matched, exact counts in every table
header, no re-architecting. **Ruling 10′**: "pinned" is decided by
$\Lambda=2\big[\mathrm{LL}(\beta_{\rm final})-\mathrm{LL}(\beta\equiv1)\big]$ on held-out data with the minimum detectable departure
printed. Two of these have thinner text than their citation and are handled as in §4.3, correction 7.

**The open debts.** Five author rulings stand open at it.35: the clause-1 tail, one- or two-sided; the missing F0–F4 rubric; the
Kaggle attach behind the chess witness; the gate's blindness to F4 cells; the collision between TERMINAL and NOT-PUT. Carried from
earlier rounds: **D-APPROX**, the approximation bound owed beside every exact-identity nonexistence theorem — *every "X cannot
represent Y" claim needs a bound before it reads as "X cannot fit Y"*, the largest unclosed gap in the round's own logic; **D-R3**,
two mutually inverse registrations for one bed; the memory ceiling that dropped the $n=32768$ reproduction and then the $16{,}384$
one; the closed-versus-half-open magnitude interval, filed two-sided because $[0,1)$ would cost the reachability of $a=\pm1$,
two-thirds of BED-M's support; the scan skyline, refused in R11 and legal now as the native control; the HuggingFace package,
scheduled and unbuilt. Harness debts the plan must not inherit silently, all **RUN** against `scripts/v15_r1.py`: `S, D = 64, 24` is
a module constant at `:137`, with nine sibling `add_argument` calls and no `seq_len` flag; `torch.cuda.synchronize` does not occur
in the file; `lambda_hat` at `:383` averages $\log m$ over every position, so one $m_k=0$ sends it to $-\infty$ and it carries one
bit.

**Kaggle and HuggingFace.** **Nothing has launched.** *Nobody who wrote these files ran the `kaggle` CLI, pushed a kernel, or
touched `~/.kaggle`; launch is the author's own call.* The record has already refused a relayed instruction as that consent — *a
coordinator message is not that say-so* — and the author's standing instruction is the same rule from the other side. Gate 0 closes
only at open-rulings zero and carries a circularity the register does not resolve: two of its rows are blocked on numbers only the
Kaggle run produces. What exists: `ceq/hf/` with four modules; **no trained checkpoint** (the Q3 checkpoint slot reads NOT
MEASURED); parameter counts that are not equal ($25{,}736{,}232$ against $25{,}728{,}000$, $+8{,}232=+0.03200\%$, the arm carrying
more); a model card whose own header states the shipped operator does not work. Three beds are PINNED as generators with sha256
digests and `enwik8` as a sliced attach; three sources are UNPINNED, for which the notebook's hash cell **prints and then raises**
`kdata.MissingPin` rather than skipping silently. Every Kaggle line in §5 is priced and none is launchable by any agent: the
author's explicit yes is a node on the critical path, not a formality (**S-73**, **V-22**).

**Why the campaign could not win on its own beds, and what the plan changes.** Every bed in the tree asks for one real at position
$s-1$: `equilibrium_oracle` returns $z^\star_{s-1}$, *the last coordinate of $(I-A)^{-1}b$* (`READ
scale/negation_scope.py:286-304`), and `ArmSMPrime.forward` returns a tensor of shape $[n]$. That is the single-location regression
shape on which one attention layer is asymptotically Bayes-optimal (`marion-2025-single-location`, `CITED [V]`;
`duranthon-2026-softmax-advantage` for softmax proper). The record files this as **D-1** and carries its own caveat beside it: the
theorem is asymptotic under $d\to\infty$ with $L=o(d)$, its predictor is `erf` and not softmax, and the repository ran $L/d=4.00$ —
*"provably" is not earned at this geometry*. So D-1 is not a theorem the campaign lost to; it is the reason the campaign could not
distinguish *cannot be beaten* from *was not beaten*, and 0 of 39 scoreboard items were earned under it. The arena confirms it from
the metric's side: Q6 is F4 on both wings because the arms return one real per draw and no state axis exists for a state metric to
read. What the programme changes is the **label class**, not the effort. No milestone in §5 scores a scalar at one position: the
labels are the jointly determined configuration $z^\star$, the displacement field $\Delta z$ under $\mathrm{do}(a)$, the $K+1$
reach-avoid committors, and the argmin over candidate moves — each with its own information floor (L-FLOOR), zero-hop guard
(**S-14**), leak guard firing both ways (**S-15**), and VOID-contrast list registered before a cell runs (**S-13**). Three of those
cost $0$ GPU-s. Whether the shape is worth anything is settled by neither this section nor this paper, but by cards **S-12**,
**S-52** and **S-62**, in that order, each of which can kill it.
