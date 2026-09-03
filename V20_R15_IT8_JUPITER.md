# V20 R15 — it.8 — JUPITER (MYCROFT) — Q4 COST LAW AND Q5 INFORMATION FLOOR, BOTH WINGS

**Branch `v17k-gate0`, HEAD `207e7b9`.** Wings frozen at `V20_R15_WING_MANIFEST.md:20-21`:
W1 `arm_smprime`, W3 `arm_pl`. Four cells, not six. Q6 stays for it.9.

**`[RUN]` `python -m pytest tests/jupiter/test_v20_r15_it8_q4_q5.py -q` → 12 passed in 1.97s.**
Two REDs recorded verbatim in §6 before green; six planted negatives, all firing.

**A run started at `2026-09-02 02:56:13` is in flight and is not mine**
(`results/v20_r15_it8_armpl_seeds8_15.jsonl`, header + identity row only, `arm_pl`
seeds `[0,8..15]`). **Nothing below rests on it.** Its header journals `s: 64` like every
other, which is itself §1's evidence.

---

## 0. CELL SHEET

| | W1 `arm_smprime` | W3 `arm_pl` |
|---|---|---|
| Q1 EXACT CLASS | F0 | F1 |
| Q2 OUTSIDE | F1 + const | F1 + const |
| Q3 LEARNABILITY | F2 | F1 + const |
| **Q4 COST LAW** | **F3, §2** | **F3, §3** |
| **Q5 INFORMATION FLOOR** | **F1 + const, §5.3** | **F1 + const, §5.4** |
| Q6 | it.9 | it.9 |

**Ten of twelve.** Both Q4 cells are F3 and §1 is why: the question asks for a law in
sequence length and **the instrument cannot vary sequence length.**

---

## 1. THE Q4 QUESTION CANNOT BE ANSWERED BY THIS CAMPAIGN, AND THE REASON IS ONE LINE OF SOURCE

Q4 asks for GPU-seconds **versus sequence length**. Across **34 banked cell records** in
the two journals — 32 distinct cells plus seeds 0 and 1 measured twice —

```
s = 64        on 34 of 34          t_star = 2     on 34 of 34
steps = 150   on 34 of 34          n_train = 2048, n_eval = 4096 on 34 of 34
```

`[RUN] ::test_q4_sequence_length_never_varies_so_the_cost_exponent_is_unidentified`.
**The independent variable has n = 1.** No regression exists, and none can be fitted.

**This is not an omission, it is the shipped instrument.** `[READ] scripts/v15_r1.py:137`:

```
S, D = 64, 24
```

a module constant, with `--steps`, `--n-train`, `--n-eval`, `--seeds`, `--threads`,
`--arms`, `--device` all exposed as flags at `:547-557` and **`S` exposed as none of
them.** `[RUN] ::test_q4_the_timer_is_unsynchronised_and_the_length_is_a_module_constant`.
**The length exponent is not unmeasured; it is unmeasurable without editing source.**

### 1.1 The timer is also not measuring what a cost law needs

`[READ] scripts/v15_r1.py:249,:267` — `t0 = time.time()` … `secs = time.time() - t0`,
and the bracket contains **only the `steps` training loop** (`:250-266`). Eval, bootstrap,
gate columns and the manifest are outside it. **There is no `torch.cuda.synchronize()`
anywhere in the file**, so every `secs` on `device: cuda` is un-synchronised host wall
clock over an asynchronous queue. `[RUN]` same node.

**That is the mechanism behind the spread the brief refuses to let this office quote as a
point**, and the spread is worse than the brief's figure:

| arm | n | min | max | mean | sd | **spread** |
|---|---|---|---|---|---|---|
| `arm_smprime` | 18 | 14.852 | 22.069 | 17.097 | 1.964 | **1.4859×** |
| `arm_pl` | 8 | 1.726 | 1.916 | 1.780 | 0.075 | 1.1101× |
| `softmax` | 8 | 1.660 | 1.749 | 1.681 | 0.029 | 1.0536× |

The brief's `1.387×` is the it.6 process alone (`22.069 / 15.907`); **over the full record
it is `1.4859×`.** `[RUN] ::test_q4_cost_is_a_RANGE_not_a_point_on_every_arm`.

### 1.2 The repeat control — and it says the spread is NOT measurement noise

Seeds 0 and 1 were run in **both** journals. Their `eval_nrmse`, `frac_gate_annihilated`
and `a_hat_max` are **bitwise identical** across the two processes; their `secs` are not.

| seed | retake `secs` | it.6 `secs` | relative difference |
|---|---|---|---|
| 0 | 16.164 | 16.104 | **0.373 %** |
| 1 | 16.032 | 15.957 | **0.470 %** |

`[RUN] ::test_q4_repeat_control_bounds_measurement_noise_far_below_the_spread`.

**Same computation, same box, two processes: 0.47 %. Different seeds, one process:
48.6 %.** The repeat noise is **77× smaller** than the spread. So the spread is a real
per-cell effect and the timer is not the whole story — **but the un-synchronised timer
means this office cannot say which effect.** Filed as the calibration, not as an excuse.

---

## 2. Q4 / W1 `arm_smprime` — WHAT CAN BE STATED, DERIVED FROM SOURCE

### 2.1 The cost form, `[DERIVED]` from the operator

`[READ] ceq/arm_smprime.py:4-7` — `W_ij = G_ij · exp(qk·q_i·k_j) / Z_i^beta`,
`G_ij = Π_{k=j+1}^{i} m_k e^{iθ_k}`. There is **no Python loop over `S`**; every
`S`-dependence is materialised:

| site | object | order |
|---|---|---|
| `ceq/arm_smprime.py:163-172` | `path_product` builds `[..., S, S]` rows, then `cumprod` | **O(n·S²)**, complex128 = 16 B/entry |
| `ceq/arm_smprime.py:223` | `q @ kᵀ` | O(n·S²·d) |
| `ceq/arm_smprime.py:227` | `exp` over the full `[n,S,S]` | O(n·S²) |
| `ceq/arm_smprime.py:206` | **two** path products, `(G, R)` | 2 × O(n·S²) |
| `ceq/arm_smprime.py:264,:576` | complex `a @ v` | O(n·S²·d_model) |

**Statement: `T_W1(n, S) = Θ(n·S²·d_model)` in complex arithmetic, with three separate
`[n,S,S]` materialisations.** The source declares the leading term itself
(`ceq/arm_smprime.py:159-161`, a `ponytail:` comment naming the O(S²) materialisation).

### 2.2 The constant, and it is a factor of 1.49 wide

`arm_smprime / softmax` at `s = 64`: **mean `10.171×`**, extreme-to-extreme
**`8.492× – 13.295×`**. `arm_smprime / arm_pl`: **`9.605×`**.
`[RUN] ::test_q4_neither_arm_converts_its_sparsity_into_cost`.

**The cost is not in the parameter count.** `n_params` is `4806 / 4803 / 4769` for
`arm_smprime / arm_pl / softmax` — **+0.776 % parameters against ×10.17 seconds.**

**Cost versus gate statistics, the second half of the question** `[DERIVED]`, Spearman over
the 16 distinct W1 cells:

```
secs vs frac_gate_annihilated   rho = +0.5197   p = 0.0391
secs vs eval_nrmse              rho = +0.3794   p = 0.1472
secs vs seed                    rho = +0.7029   p = 0.0024
```

**The strongest correlate of run time is the seed index**, which is run order. On an
un-synchronised timer inside one process that is a drift signature, and it is stronger
than the gate correlation it would have to be separated from. **`secs` versus gate
statistics is confounded with run order and this office will not report the `+0.52` as a
gate effect.**

### 2.3 The EXACT sparsity the primitive induces — F0, and no M9 number is cited

**L-CERT is binding and the certificate produced here is exact, so M9's F1/F1′ machinery
is not needed and is not invoked.** WILSON's census stands: `cantelli`, `azuma`,
`bonferroni`, `union_bound` have **0 producing `.py` in the tree** — the only hits are
string literals in `tests/saturn/test_v20_r15_wing_rubric.py:261,:282`. **`d = 65`,
`δ = 1 %`, `d = 20` and `1.3e-3` are not quoted anywhere below.** SATURN's K6 stays RED
and this cell does not lean on it.

**The certificate, stated and proved:**

> Let `Z ⊆ {0..S−1}` be the positions with `m_k = 0`. Since `G_ij = Π_{k=j+1}^{i} a_k`,
> **`G_ij = 0` ⟺ `Z ∩ (j, i] ≠ ∅`**. Therefore for each row `i`, with
> `z(i) = max{k ≤ i : k ∈ Z}` (`0` if none), the live entries of row `i` are exactly the
> **contiguous suffix `[z(i), i]`**, of size `i − z(i) + 1`.

The `⟺` is `pathProd_eq_zero_iff`, `[READ] lean/CEQ/V16Domain.lean:129`, re-verified under
the green `lake build` this office ran at it.7. **No probability appears. There is no `ε`,
no `δ`, no tail bound and no union.** This is M9's own F0 branch — zero-gate segmentation —
and it is the branch that has a producer.

**`[RUN] ::test_q4_zero_gate_certificate_is_EXACT_mask_fidelity_and_causality`**, five
zero-sets (`[]`, `[5]`, `[3,9]`, `[1,2,3]`, all-but-0) against the shipped
`arm_smprime.path_product`. Contracts exercised, named from `anthropic-skills:tda-tdd` and
not hand-rolled: **MASK FIDELITY** (predicted zero set equals the operator's zero set,
bitwise), **CAUSALITY** (`triu(1)` empty), **ALL-MASKED-ROW guard** (the diagonal is live
in every row, so no row is fully annihilated and no NaN row can arise), **DETERMINISM**
(two calls, `torch.equal`). Planted negative `cert_off_by_one` shifts `z(i)` by one and
takes the node RED.

### 2.4 THE SPARSITY IS CERTIFIED, DENSE, AND MEASURED ON THE WRONG OBJECT

**Three defects, each `[READ]`, and together they are the Q4/W1 grade.**

1. **The mask is produced by paying for it.** `path_product` materialises the full
   `[..., S, S]` block and *then* `cumprod`s the zeros into it
   (`ceq/arm_smprime.py:163-172`). `zb = mod.sum(-1)**beta` (`:249-251`) sums the dense
   row. **A gate that annihilates 99.46 % of the live band costs no less than one that
   annihilates nothing** — `[RUN]` the flat band's cheapest cell is `15.907 s` against
   seed 2's dense `14.852 s`. **The certified sparsity buys zero seconds.**
2. **The operator-level mask exists and is never called.** `ArmSMPrime.zero_hop_mask`
   (`ceq/arm_smprime.py:559-567`) returns the `[n,S,S]` bool. **`zero_hop_mask` does not
   appear anywhere in `scripts/v15_r1.py`.**
3. **`frac_gate_annihilated` is not the operator's sparsity.**
   `[READ] scripts/v15_r1.py:386` — `float((~fin).double().mean())` over
   `lg = torch.log(m)` on the **live band only**, `live = range(head+1, S)` at `:697-698`
   with `head = S − 1 − T_STAR`, i.e. **2 positions × 4096 rows = 8192 entries**, not the
   `64 × 64` causal block. **The number the whole round has been reading as this arm's
   sparsity is a statistic on 2 of 64 positions.** VENUS's lattice
   (`k ∈ {0, 4069, 4123, 8002, 8148}`, `4069 + 4123 = 8192`) is exact and is a lattice on
   *that* denominator — her arithmetic is untouched; what changes is what the denominator
   names.

`[RUN] ::test_q4_the_certified_sparsity_is_computed_but_never_exploited`, which also
asserts **no `topk`, no `top_k`, no `torch.sparse`, no `nonzero(` anywhere in the arm.**

### 2.5 GRADE Q4/W1: **F3.** HOW-BAD gap

**F3 and not F2:** the form is derived from source and the certificate is F0-exact, but
**the law's exponent is unidentified (n = 1 in `S`), its constant is known only to a factor
of 1.49, and the timer producing that constant is un-synchronised CUDA wall clock.** Three
independent gaps, none closable by reading.

**L-DOM census for Q4, two columns, and it is empty on both wings:**

| cost model in this repo | domain it accepts | W1 in domain? | W3 in domain? |
|---|---|---|---|
| `ceq/sizing.py:145-172` `flops_per_token` | `{"softmax","signed","sgate"}`, raises `ValueError` otherwise at `:169` | **NO** | **NO** |
| `scale/m3_flops.py:207` `CELLS` | `("softmax","glance","settled","twin","argmax")` | **NO** | **NO** |
| `ceq/mz_kernel.py:170-180` `attention_flops` | Triton tile plans | **NO** | **NO** |

**Zero of two frozen wings are in the domain of any of the three cost models this
repository already ships.** `[RUN]` same node. This is the fourth domain-empty finding of
the round and the first on the cost side.

**Replacement route, priced.** One argparse line (`--seq-len`, default 64) turns `S` into a
flag; the exponent then needs `S ∈ {32, 64, 128}` × 3 seeds × 3 arms. At the measured
`s = 64` costs and an `S²` scaling that is `≈ 3 × (0.25 + 1 + 4) × (16.16 + 1.78 + 1.68)`
**≈ 275 GPU-s** — one and a half it.6 runs for the cell's actual question. **Add
`torch.cuda.synchronize()` around `:249,:267` first, or the 275 s buys the same
un-synchronised number three times.**

---

## 3. Q4 / W3 `arm_pl` — THE ARM THAT COSTS NOTHING AND SAVES NOTHING

### 3.1 Cost

`[READ] ceq/arm_pl.py:113-119` — `w = q@kᵀ/√d`, `w += (s − cumsum(g))_j`, `tril` mask,
causal `softmax`. **Real dtype, one `[n,S,S]` materialisation, `Θ(n·S²·d)`** — the *same*
order and the *same* materialisation count as the `softmax` incumbent, with one extra
`[S]`-shaped `cumsum` broadcast into the score.

Measured: **`arm_pl / softmax = 1.0589×` mean, range `0.987× – 1.154×`.** The two
distributions **overlap** (`arm_pl` `1.726–1.916`, `softmax` `1.660–1.749`). **W3 is free.**

### 3.2 Sparsity: exactly zero, on 8 of 8

`frac_gate_annihilated == 0.0` on all eight `arm_pl` cells `[RUN]`, and the arm has no
zero-producing construction at all: its gate enters as an **additive log-domain bias**
(`ceq/arm_pl.py:93-96`, `key_bias(g,s) = s − cumsum(g)`) inside a softmax, so every entry
of the causal block is strictly positive. **`arm_pl` induces no sparsity, therefore it
needs no certificate, therefore L-CERT is satisfied vacuously and buys nothing.** The
planted negative `pl_dense` — one `arm_pl` cell given `frac = 0.4` — takes the node RED, so
the `0.0` is asserted, not assumed.

### 3.3 GRADE Q4/W3: **F3.** HOW-BAD gap

The constant is tight and the sparsity statement is exact-and-empty; **the cell still fails
at the same place W1 does — there is no law in `S`, because `S` never varied.** An arm that
costs `1.06×` softmax at one length is a *ratio*, not a cost law. **The same `--seq-len`
flag closes both cells; W3's share of the 275 GPU-s is `≈ 21 s`.**

**Second gap, W3-specific:** `ceq/arm_pl.py:316-327` ships `brute_force_path_sums` with a
`for i × for j × for size × itertools.combinations` nest — **O(2^S)**. It is a DAG
cross-check and is not on the forward path, but **it is in the module a cost model would
be read from**, and no test forbids a caller from reaching it. Filed, not struck.

---

## 4. THE β AMENDMENT — TAKEN, AND BOUND

The coordinator's Q3 amendment is used below and is therefore bound here.

`[READ] ceq/arm_smprime.py:12-14,:51` — `beta = 0` is the exact corner, **`beta = 1` is
softmax**. The twelve flat-band failures span `beta ∈ [0.58758, 1.00081]` and land at
`eval_nrmse 0.852–0.927`; `softmax`'s own eight cells land at **`0.9388–0.9734`** `[RUN]`.

> **The flat band is not descending to the arm's corner. It is converging to the
> incumbent, and it arrives in the incumbent's neighbourhood because it has become the
> incumbent.**

This office's it.7 §2.2 struck M1's *transfer* on the grounds that `β` does not order the
outcome (`ρ = −0.032`). **The amendment is stronger and this office adopts it:** `β` does
not order the outcome **because eleven of sixteen cells have collapsed onto one
neighbourhood of it**, and a coordinate with no spread cannot order anything. `ρ = −0.032`
is the signature of the collapse, not evidence against a mechanism. **Filed as a correction
to this office's own reading — the second self-correction this round.**

**Consequence carried into Q5:** a cell that has become softmax cannot cross a floor
softmax does not cross. §5.3 is where that becomes a number.

---

## 5. Q5 — WHICH FLOOR ACTUALLY BOUNDS THESE WINGS

**L-FLOOR is binding. The census comes before the number.**

### 5.1 L-DOM CENSUS OF THE FOUR CANDIDATE FLOORS — THREE ARE PROSE

| candidate | what it needs to bind | **beds satisfying it** | **producing `.py`** | verdict |
|---|---|---|---|---|
| **M11 Fano** `P(err) ≥ 1 − (I(X;Y)+ln2)/ln m` (`CEQ_V20_R15_CONTRACT.md:226-229`) | an estimate of `I(X;Y)` and a finite alphabet `m` | **0 of 3.** No bed journals a mutual information or an alphabet size | **0 hits for `fano` or `mutual_information` in any `.py`** | **PROSE. Not graded on.** |
| **M12 rate–distortion** `D(R) = σ²2^{−2R}` (`:230-234`) | a rate `R` and a Gaussian source `σ²` | **0 of 3.** No bed journals a rate | **0 hits for `rate_distortion` in any `.py`** | **PROSE. Not graded on.** |
| **M16 Hankel `R²_k`, `1/d` anchor** (`:247-249`) | a **delay bed** with a `d` | **1 of 3 — BED-K(a) only** (`CEQ_V20_R15_CONTRACT.md:116-117`) | `ceq/hankel.py` exists; **no `hankel_bound`, no AAK, no Glover identifier in any `.py`**; the `1/d` curve is computed **only inside this office's own it.6 test**, `tests/jupiter/test_v20_r15_it6_q2_outside_bound.py:75-105` | **DOES NOT BIND EITHER WING.** Both W1 and W3 are **BED-M** arms. Its citation stays refused (it.6: AAK/Glover state an operator-norm `σ_{k+1}` result, M16 computes a Frobenius sum-of-squares; the correct source is Eckart–Young–Mirsky) |
| **BED-M `floor₁ = 0.7071067811865476`** | a `t*` | **1 of 3 — BED-M, and both frozen wings are on it** | **`scripts/v15_r1.py:586` `floor1 = math.sqrt((T_STAR-1)/T_STAR)`**; second producer `scale/it11_verdict.py:451-458 hop_floor`; journalled on **34 of 34** cells | **THE ONLY ONE THAT BINDS. Graded on.** |

**Of the four floors the brief put on the table, exactly one reaches the frozen wings, and
it is the one the annex does not call an information floor.**

### 5.2 AND `floor₁` IS NOT AN INFORMATION FLOOR EITHER — IT IS A ONE-HOP THRESHOLD

`[READ] scripts/v15_r1.py:17-19`, the source's own statement:

```
NRMSE < floor_1 AND h_hat > 1 ARE ONE EVENT.  h_hat = t*(1 - NRMSE^2)
and floor_1 = sqrt((t*-1)/t*), so h_hat = 1 exactly at NRMSE = floor_1.
```

Producers at `:810-811` (`eval_h_hat`, `dist_to_floor`) and `:907-908` (aggregate).
`[RUN] ::test_q5_floor_1_is_derived_not_cited_and_holds_to_machine_precision` verifies
`floor_1 == math.sqrt((t*−1)/t*)` **exactly** on 34 of 34, and
`::test_q5_floor_1_is_a_ONE_HOP_THRESHOLD_not_an_information_floor` verifies the identity
`eval_h_hat = t*(1 − nrmse²)` to `1e-12` and the equivalence
`(nrmse < floor₁) ⟺ (h_hat > 1)` on 34 of 34.

**A lower bound is never violated. This one is, by six cells** — five `arm_pl`
(`dist_to_floor` `−0.0450 … −0.0734`) and `arm_smprime` seed 2 (`−0.503187`). `[RUN]`.
**So `floor₁` is a capability threshold at `h_hat = 1`, not a bound on achievable error,
and calling it a floor is the thing that has to stop.**

**The real information floor is journalled and calibrated** `[RUN]`, from the `t="bar"`
record of `results/v17k_r4_retake.jsonl`, `ok: true`, `"BAR CALIBRATED"`:

```
oracle                = 0.0                     <- the information floor
trained_two_feature   = 0.013981630466969725
payload_only          = 1.2272241529888959
predict_the_mean      = 1.0000000843170462      <- the trivial ceiling
flipper_dependence    = 1.4012437605053387
```

**The task is deterministic given the input; the irreducible error is `0.0`.** Everything
this round has called a floor sits **above** it.
`[RUN] ::test_q5_the_real_calibrated_floor_and_ceiling_are_journalled`.

**And the upper reference is missing everywhere:** `dist_to_skyline` is `None` on **34 of
34**, with `dist_to_skyline_why` reading `"no v15/v16 scan-skyline module exists (R-SKY)"`
on all 34. `[RUN] ::test_q5_dist_to_skyline_is_null_on_every_cell_with_the_reason_recorded`.

### 5.3 Q5 / W1 — HOW FAR ABOVE, AND THE THEORY'S PREDICTION

| W1 regime | n | `eval_nrmse` | `dist_to_floor` | **`h_hat` = hops resolved of 2** |
|---|---|---|---|---|
| **crossing** (seed 2) | 1 | 0.203920 | **−0.503187** | **+1.916833** |
| **flat band** | 12 | 0.852–0.927 | +0.145 … +0.220 | **+0.281 … +0.548** |
| **intermediate** (seed 3) | 1 | 0.916258 | +0.209151 | +0.320923 |
| **over-decayed** (11, 13) | 2 | 1.121 / 1.203 | +0.413 / +0.496 | **−0.512 / −0.896** |

**The theory's prediction, `[DERIVED]` from §4 and stated before the number:** a cell that
has converged to `β ≈ 1` **is** softmax, and `softmax` lands at `0.9388–0.9734`,
`dist_to_floor` `+0.2317 … +0.2663`. **Predicted: the flat band lands in softmax's
neighbourhood, `+0.20 … +0.27` above `floor₁`. Observed: `+0.145 … +0.220`.** The band
overlaps softmax's interval and its best cells sit slightly *below* it — the arm buys
`0.087` of NRMSE over the incumbent for `10.17×` the seconds.

**GRADE Q5/W1: F1 with the constant.** Constant: `floor₁ = 0.7071067811865476`, exact and
derived; predicted band `+0.20 … +0.27`, observed `+0.145 … +0.220`.

**HOW-BAD gap:** the floor graded on **is not an information floor**, and the three the
annex names as information floors are all prose with an empty domain. The honest distance
to the *real* floor (`oracle = 0.0`) is **`0.203920` at the arm's single best cell and
`0.852` at its modal cell** — and no theory in the annex predicts either number.
**Replacement route:** an information floor for BED-M needs one bed journalling `I(X;Y)`
against the exact oracle, which BED-M already computes. **That is an instrument line, not a
theorem, and it costs 0 GPU-s.**

### 5.4 Q5 / W3 — THE ROUND'S FIRST CROSSING IS BLOCKED BY VARIANCE, NOT BY THE FLOOR

**Five of eight `arm_pl` cells sit below `floor₁` at the point level, with `h_hat > 1`:**

| seed | `lambda_hat` | `eval_nrmse` | `dist_to_floor` | `h_hat` |
|---|---|---|---|---|
| 4 | −1.38449 | 0.633739 | **−0.073368** | **+1.196749** |
| 5 | −1.45285 | 0.641999 | **−0.065108** | **+1.175676** |
| 1 | −1.43660 | 0.644517 | **−0.062589** | **+1.169195** |
| 0 | −1.43248 | 0.644673 | **−0.062434** | **+1.168794** |
| 6 | −1.47142 | 0.662128 | **−0.044979** | **+1.123172** |
| 3 | +0.18896 | 1.113339 | +0.406232 | −0.479048 |
| 7 | +1.21686 | 1.148927 | +0.441820 | −0.640065 |
| 2 | +0.75843 | 1.152280 | +0.445173 | −0.655496 |

The registered verdict is `crosses: false` for all three arms `[RUN]` (`t="agg"` records).
**It is false for a reason that has nothing to do with the floor:**

```
all eight:   mean 0.8302001907   sd 0.2554136624   ci_hi 1.0437313561
the five:    mean 0.6454112029   sd 0.0103523668   ci_hi 0.6582633033   <- clears by 0.0488
```

`[RUN] ::test_q5_no_arm_crosses_and_W3_is_blocked_by_variance_not_by_the_floor`.
**The eight-cell mean alone is above `floor₁`; no reduction in variance could save it. The
three `λ̂ > 0` divergent cells are the entire obstruction**, and the five-cell mean
`h_hat = 1.166889` says the surviving sub-population resolves **1.17 of 2 hops.**

**GRADE Q5/W3: F1 with the constant.** Constant: the five-cell margin below `floor₁`,
**`0.0488` on `ci_hi` at `n = 5`**, `t = 2.776`.

**HOW-BAD gap, and it is the honest one:** **the five-cell CI is post-hoc.** The subset was
selected on `λ̂ < 0`, a coordinate this office established at it.7 *from these same eight
cells*. **A selected CI is not a crossing and this office does not claim one.** What it
claims is that the crossing is one experiment away and the experiment is already priced.

**Replacement route, and its value has tripled:** the **~6 GPU-s** capped run at
`arm_pl` seeds 2, 3, 7 (ledger row **L-6**, `1.884 s`/cell
`[READ] results/v17k_r4_retake.jsonl`) now settles **three** cells at once — Q3/W3's causal
direction, Q5/W3's crossing, and whether `λ̂ < 0` is a *pre*-registerable selector rather
than a post-hoc one. **Six GPU-seconds against the round's first floor crossing is the
best-priced experiment on the board, and it has now been proposed twice with no taker.**

---

## 6. TEST-BOUND — TWO REDs VERBATIM, SIX PLANTED NEGATIVES

`tests/jupiter/test_v20_r15_it8_q4_q5.py`, twelve nodes.

**RED #1 — this office's cell count was wrong, and wrong in the direction that would have
shrunk the evidence base** `[RUN]`:

```
>       assert len(cells) == 26
E       AssertionError: assert 34 == 26
```

The draft counted `24 + 10`, having read `results/v17k_r4_retake.jsonl` as holding 8 cells
per arm and forgotten that its `arm_smprime` block carries the `_0step` columns as a
**separate 16-record signature**. **34 is the right number and it is VENUS's 34** — her L1
was measured on exactly this set. Correcting up rather than down is why it matters: every
census in §1 and §5 is now on 34, not 26.

**RED #2 — a transcription slip in the load-bearing Q5/W3 constant** `[RUN]`:

```
>       assert abs(statistics.mean(good) - 0.6454312) < 1e-6
E       assert 1.9997130265458907e-05 < 1e-06
E        +  where 0.6454112028697345 = mean([0.6446726..., 0.6445174..., ...])
```

The draft had `0.6454312`; the value is **`0.6454112028697345`**. **A `2.0e-5` error in the
number §5.4's whole margin is computed from.** The tolerance is now `1e-12` and the digits
are the machine's. This is the second iteration running in which RED-first caught a defect
in this office's own draft rather than in someone else's work.

**PLANTED NEGATIVES — six, named and seeded** via `JUP_IT8_MUTATE`, applied to **loaded
cells and the certificate function only, never to the journals on disk** `[RUN]`:

| mutation | what it breaks | nodes RED |
|---|---|---|
| `seq_len` | one cell's `s → 128` | 1 |
| `floor_formula` | one cell's `floor_1 → 0.75` | 2 |
| `h_hat_identity` | one cell's `eval_h_hat → 0.0` | 1 |
| `pl_dense` | one `arm_pl` cell's `frac → 0.4` | 1 |
| `cost_point` | every `arm_smprime` `secs → 16.161` (the point estimate the brief refuses) | 2 |
| `cert_off_by_one` | the certificate's `z(i) → z(i)−1` | 1 |

**Every mutation reaches at least the node it targets; none is a no-op.**
**GREEN `[RUN]`: 12 passed in 1.97s.**

**No journal was written. No git command was run. Kaggle was not touched. No moon ran
code; two read the tree.**

---

## 7. F-GRADES AND HOW-BAD GAPS

| cell | grade | HOW-BAD gap | replacement route, priced |
|---|---|---|---|
| **Q4 / W1** | **F3** | exponent in `S` **unidentified** (`s = 64` on 34/34, and `S` is a module constant at `scripts/v15_r1.py:137`, not a flag); constant known to **±1.49×**; timer is **un-synchronised CUDA wall clock** (`:249,:267`, no `cuda.synchronize` in the file); the strongest correlate of `secs` is **run order** (`ρ = +0.70`), confounding the gate reading | `--seq-len` flag + `torch.cuda.synchronize()`; `S ∈ {32,64,128}` × 3 seeds × 3 arms ≈ **275 GPU-s** |
| **Q4 / W1 sparsity** | **F0 certificate, zero cost saving** | the exact mask is produced by materialising the dense block first (`ceq/arm_smprime.py:163-172`); `zero_hop_mask` (`:559-567`) is never called from the runner; `frac_gate_annihilated` measures **2 of 64 positions**, not the operator | call `zero_hop_mask` and journal the operator-level density — **0 GPU-s**, one line |
| **Q4 / W3** | **F3** | same missing exponent; `1.0589×` is a ratio at one length, not a law; `brute_force_path_sums` is **O(2^S)** in the module (`ceq/arm_pl.py:316-327`) and unguarded | W3's share of the 275 GPU-s ≈ **21 s** |
| **Q5 / W1** | **F1 + const** (`floor₁ = 0.7071067811865476`; predicted `+0.20…+0.27`, observed `+0.145…+0.220`) | the floor graded on is a **one-hop threshold, not an information floor** — violated by 6 of 34 cells; distance to the real floor (`oracle = 0.0`) is `0.204` at best and `0.852` modally, and no annex theorem predicts either | journal `I(X;Y)` against BED-M's exact oracle — an instrument line, **0 GPU-s** |
| **Q5 / W3** | **F1 + const** (five-cell `ci_hi = 0.6582633033`, clears by `0.0488`) | the five-cell CI is **post-hoc**, selected on `λ̂ < 0` from the same eight cells | the **~6 GPU-s** capped run at seeds 2, 3, 7 — settles Q3/W3, Q5/W3 and the selector at once |
| **M11 Fano** | **F4** | **0 of 3** beds; **0 `.py`** | retire, or one bed journalling `I(X;Y)` and an alphabet |
| **M12 rate–distortion** | **F4** | **0 of 3** beds; **0 `.py`** | retire, or one bed journalling a rate |
| **M16 Hankel** | **F4 for these wings** | BED-K only; **both frozen wings are BED-M**; citation still refused; `1/d` computed only inside a test | do not cite it on W1/W3 at all |

**Standing gap, named once:** everything above is read off two banked journals, 32 distinct
cells, **one box**, on an **un-synchronised** timer. No cross-box reproduction exists for
any cost number in this report, and §1.2's `0.47 %` repeat bound is a two-process bound on
one machine, not a reproduction.

---

## 8. SCOREBOARD

No new points. Phase B is **10 of 12** cells filled — Q1 F0/F1, Q2 F1/F1, Q3 F2/F1,
**Q4 F3/F3, Q5 F1/F1**. The `+6` needs Q6 on both wings at it.9. Carried: **2 of 44.**

**Four new leap rows appended to `V20_R15_LEAP_LEDGER.md` as they were produced**
(L-9 … L-12), per the author's instruction.
