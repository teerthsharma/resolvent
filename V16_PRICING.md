# V16 MERCURY (it. 6) — RULE 4, THE DECIDING CELLS PRICED

Node `MERCURY`, iteration 6 of CEQ v16. Discharges `CEQ_V16_CONTRACT.md` PART VI
it.6: *"MERCURY prices R1′/R2/R-SKY (RULE 4)."* Every wall-clock figure cites
`V16_DEVICE_CERT.md`'s law; no timing is inherited (L-TIME).

**Nothing was trained.** No cell, no arm, no gradient step. This file evaluates a
fitted law and measured constants in closed form. The only compute it spent is
arithmetic. Files written: this one. No git command that writes was run.

**Provenance tag on every number.** `[MEASURED]` = read off this box by a named
node in a named run. `[MODULE]` = computed by `ceq/sizing.py`. `[FITTED]` =
projected from `V16_DEVICE_CERT.md` §4.1's law, with its R². `[INHERITED]` =
carried from an earlier file, cited. `[PROJECTED]` = this node's arithmetic on
the tagged inputs above. **NOT MEASURED** where it could not be obtained.

---

## 0. LEAD — THREE THINGS THAT MOVE THE PRICE, BEFORE ANY TABLE

**1. The arm this round will run is not the arm the certificate priced.**
`V16_DEVICE_CERT.md` measures `ceq/arm_phase.py` — `258,609 B/example`,
`×2.085` on the wall-clock law. it.5 did not ship that module. It shipped
`ceq/arm_smprime.py` (`V16_ARM_SMPRIME.md`), a **different construction**: a
masked cumulative **product**, not a `cumsum` of `log m`. That file **refuses to
quote a memory figure and says why** — §9: *"Any cell priced on this arm must
measure rather than model."* It is right to refuse, and this node inherits the
refusal rather than papering over it. **Every complex-arm hour and byte below is
therefore priced on `arm_phase` and is a FLOOR, not an estimate**, and §1 gives
the structural reason the true figure is above it.

**2. The certificate's own precondition 4 is discharged, and precondition 1 with
it.** `V16_DEVICE_CERT.md` §6.3 signs conditional on the phase arm's
`log(clamp(u,0,1))` being made finite — *"or no hour above is spendable on it at
all."* `V16_ARM_SMPRIME.md` (g) records **8 of 8** configurations
(`{cpu,cuda} × {fp32,fp64} × {as-constructed, identity_heads()}`) completing 40
gradient steps with no non-finite gradient `[MEASURED]`, at `55.58%` of positions
on the closed lower endpoint — *more* than the `50.08%` that kills `arm_phase` at
step 0. And §9 measures `arm_smprime.operator` and `path_product` running **OK**
under `use_deterministic_algorithms(True)` on cuda where `arm_phase` **RAISES**,
because `cumprod` has a deterministic CUDA kernel and `cumsum` does not. **The
two conditions that made the schedule unspendable are gone for the round's own
arm.** What replaces them is §1's unmeasured cost.

**3. A third precondition, half discharged while this file was being written,
and the state is read rather than asserted.** `V16_DEVICE_CERT.md` LIMITS
records that **no shipped runner can instantiate a complex arm** — *"every
`arm_phase` hour in §6.2 is priced for an arm no shipped runner can currently
instantiate."* At repo HEAD `deee6c4` that is still true:
`git show HEAD:scripts/v15_r1.py`'s `make_arm` knows `"arm_pl"` and
`Arm(kind, s)` `[MEASURED by read]`. **In the working tree it is no longer
true.** `scripts/v15_r1.py` is `M` in `git status`, owned by another node this
iteration, and its `make_arm` (line 118-123) now builds
`arm_smprime.ArmSMPrime(s, d_model=D_MODEL)`; `--arms` takes free strings and
`--device` accepts `cuda` with the v15 abort gone `[MEASURED by read of the
working tree]`. **`scale/r10_capacity_sweep.py` is clean and still constructs
`Arm(arm, s)` at line 102**, so the sweep path reaches neither complex arm.
**R1′'s eight complex cells are now reachable; the sweep-driven nodes are
not.** §7.2 carries this as a per-file state rather than a verdict, because a
certificate written against a moving tree is a certificate about a moving
object — the certificate's own words, and this file inherits the problem rather
than escaping it.

---

## 0.1 THE PRICE IN SEVEN LINES

1. **R1′** (it.7, `n = 2048`, 16 cells): **0.699 GPU-h** point, **0.731 h**
   pessimistic at the 9600 ladder; **0.011 / 0.011 h** at the 150-step floor.
   Peak working set **0.693 GiB**, 10.1% of free VRAM. Fits with room no
   plausible arm consumes.
2. **R2 as registered** (`n = 32768`, 16 cells): **10.382 / 10.865 GPU-h** at
   9600 — and it **DOES NOT RUN**. Complex working set **7.892 GiB** against
   **6.855 GiB free** and **7.996 GiB total**. Not a squeeze; an impossibility.
3. **R2 at `n = 16384`**: **5.288 / 5.533 GPU-h**, working set **3.946 GiB**,
   **41.4% headroom**. **This node recommends it**, and §3.4 states what the
   round loses.
4. **R-SKY** (it.10, 16 cells): **3.588 / 3.755 GPU-h** as registered. **If R2
   moves, R-SKY@R2 must move with it** — `Δ_sky` is a same-cell column — which
   takes the pair from **13.744 h to 7.000 h**.
5. **The 40-iteration plan**: **68.354 GPU-h** point / **71.533 h** pessimistic
   at the 9600 ladder; **1.068 / 1.118 h** at the floor. Against **815.877 /
   2,077.461** and **12.748 / 32.460 CPU-h**. **11.94× at the point estimate,
   29.0× at the pessimistic end.**
6. **The complex column is 43.0% of the total and is priced on the wrong arm.**
   At `1.5×` `arm_phase`'s cost the plan is **83.0 h**; at `2.0×`, **97.7 h**.
   The correction is one probe run, not a re-plan.
7. **Any CPU figure carries a factor of two that averaging cannot remove.** Six
   measurements of ONE configuration spanned **2.0×–4.0×**. The certified CUDA
   lane's spread under the same sweep is **1.05×**, so the certified lane
   removes it — for the tensor work, and not for the host work beside it. §5.

---

## 1. THE ARM BEING PRICED, AND WHY ITS FIGURE IS A FLOOR

`V16_DEVICE_CERT.md` §3.4 closes with an `[ASSUMED]` and a must-fire:

> *"that it.5's phase-gate arm is `ceq/arm_phase.py` as it stands. If it.5
> rebuilds it, **every `arm_phase` number in this certificate must be
> re-measured with this probe before it.7**."*

**it.5 rebuilt it.** The must-fire has fired and it is not this node's to
discharge — pricing is not measuring, and the brief forbids running cells. What
this node can do is bound the direction, from source rather than from taste.

**The direction is up, and it is structural.** Reading `ceq/arm_smprime.py`
(`path_product`, `numerator`, `operator`), one forward materialises, at `[n,s,s]`:

| tensor | dtype | `arm_phase` | `arm_smprime` |
|---|---|---|---|
| content term `exp(qk·q·kᵀ)` masked | real | yes | yes |
| the complex hop | complex | yes (`exp(C_i−C_j)`, one scan) | yes (`path_product(gate)`, a masked `cumprod` with a `where`, two `flip`s, a `cat` and a `masked_fill`) |
| **its real modulus row `R_ij = Π m_k`** | **real** | **no** | **yes — a SECOND full `path_product`** |
| numerator `G·e` and modulus `R·e` | complex + real | one | two |
| the quotient, real and imag parts separately | 2 × real → complex | one | three |

`V16_ARM_SMPRIME.md` §1 states the second path product is deliberate: the
normaliser sums `path_product(m)` — *"clause 1's own right-hand side"* — rather
than taking `abs` of a complex zero, which is the repair that removed a `nan`
gradient at step 11 (§7). **The repair is correct and it costs bytes.**

**So `258,609 B/example` and `×2.085` are lower bounds on the arm that will run,
not predictions of it.** `ceq/sizing.py` cannot supply the true figure either:
the certificate measured the module **under-predicting the complex arm at
`m/p = 1.842`** `[MEASURED, V16_DEVICE_CERT.md §3.1]`, `C_OPERATOR` at **7.50 at
8 B/element**, and **`C_RESIDUAL` NOT IDENTIFIED** at `s = 64` because the
operator term is 84% of the total there. Under-prediction is the one direction a
sizing gate must never have, so the brief's instruction not to use the module for
the complex arm is honoured throughout: **no complex figure below descends from
`ceq/sizing.py`.**

**What this costs the pricing, stated as a number rather than a worry.** §3.5 and
§6.4 carry the sensitivity: the complex column is 43.0% of the plan, so a `k×`
error in the complex arm's cost is a `0.43(k−1)` error in the total, and a `k×`
error in its bytes moves the affordable `n` by `1/k`.

**NOT MEASURED, and it is one probe run:** `arm_smprime`'s bytes-per-example at
`s = 64`, and its s/step at `n = 2048` on cuda. `scripts/v16_device_probe.py`
sections C1 and D3 already compute both for a named arm. Until they are run, §3's
cap and §6's total are floors with a stated direction.

---

## 2. THE LAW, CITED NOT RE-DERIVED

Every wall-clock figure in this file is this law evaluated. It is not re-fitted
here, and this node does not believe it is wrong.

> **`s/step = exp(−11.9670) · n^0.9734`**, **R² 0.999384**, log-log OLS over
> five points `n = 2048…32768`, softmax arm, `s = 64`, CUDA.
> `[FITTED, V16_DEVICE_CERT.md §4.1]`

| input | value | provenance |
|---|---|---|
| CUDA law intercept / exponent | `−11.966997693137031` / `0.9734063827079997` | `[FITTED, V16_DEVICE_CERT.md §4.1, R² 0.999384]` |
| CPU law intercept / exponent | `−11.023567676348247` / `1.1227998768908996` | `[FITTED, ibid., R² 0.999704]` |
| arm multiplier, softmax | `×0.9986` | `[MEASURED, ibid. §4.2, n = 2048, CUDA]` |
| arm multiplier, complex | **`×2.0850`** | `[MEASURED, ibid., on `arm_phase` — a FLOOR, §1]` |
| pessimistic factor, CUDA | **`×1.0465`** | `[MEASURED, ibid. §2.3, host-pressure sweep spread]` |
| pessimistic factor, CPU | **`×2.5463`** | `[MEASURED, ibid.]` |
| bytes/example, softmax-class | `135,321` (activations + `x_train`) | `[MEASURED, ibid. §3.4 / probe C1, pessimistic end of a 0.09% spread]` |
| bytes/example, complex | **`258,609`** | `[MEASURED, ibid., on `arm_phase` — a FLOOR, §1]` |
| free VRAM at the cap run | `6.855 GiB` (total `7.996`) | `[MEASURED, ibid. probe C3]` |
| eval-set reserve | `0.200 GiB` | `[ASSUMED, ibid. §5, carried unchanged]` |

**Consistency check, so the law is being applied the same way.** Re-evaluating
the certificate's own 232-cell plan from these inputs reproduces
**`68.354` / `1.068` GPU-h** and **`815.877` / `12.748` CPU-h**, and the
pessimistic **`71.533` / `1.118`** and **`2,077.461` / `32.460`** — every figure
to the digit the certificate prints `[PROJECTED, this node]`. The caps
reproduce to within one example (`22,105` against the certificate's `22,106`;
`27,631` against `27,633`; `42,244` against `42,247`), the residue being that
§3.4's caps rest on the `6.855 GiB` snapshot rather than the `6.939 GiB` of §1.
**The certificate's numbers are used, not this node's, wherever they differ** —
they are the more conservative of the two.

**The pessimistic column is a measured spread, not a discount.**
`V16_CALIBRATION.md` **D-CALIB-3** authorises *"sign, never size"*; no numeric
shrink factor is applied anywhere in this file.

---

## 3. THE THREE CELLS, PRICED

`N = 8` is contract-fixed (PART IV). `P` = the complex §S-M′ arm; `S` = a
softmax-class real-valued arm. Arms-per-node follows `V16_DEVICE_CERT.md` §6.1
and its reasons are restated where they bind.

### 3.1 R1′ — it.7, BED-M, `t* = 2`, `n = 2048`

| | |
|---|---|
| cells | **16** = 8 seeds × {`P`, `S`} |
| why two arms | **the softmax control cannot be inherited from R1.** R1 is CPU-taken; `V16_DEVICE_CERT.md` §5.5.2 forbids the cross-device cell comparison, so the control is re-taken on the certified device |
| seeds | 8, contract-fixed |
| steps | 150 (the floor R1 actually bought) and 9600 (the full ladder) |
| `floor₁` | `√((t*−1)/t*) = √(1/2) = 0.7071067811865476` `[MEASURED from `scale/r10_it10_frontier.py:19`]` — a closed form in `t*`, no device and no `n` in it |

**Wall clock** `[PROJECTED from §2's law]`:

| ladder | arm | per cell, point | per cell, pessimistic | node total, point | node total, **pessimistic** |
|---|---|---|---|---|---|
| **150** | `P` (8 cells) | 3.3 s | 3.5 s | 0.0074 h | 0.0077 h |
| **150** | `S` (8 cells) | 1.6 s | 1.7 s | 0.0035 h | 0.0037 h |
| **150** | **both** | | | **0.0109 h** | **0.0114 h** |
| **9600** | `P` (8 cells) | 212.5 s | 222.4 s | 0.4723 h | 0.4943 h |
| **9600** | `S` (8 cells) | 101.8 s | 106.5 s | 0.2262 h | 0.2367 h |
| **9600** | **both** | | | **0.6985 h** | **0.7310 h** |

**Peak memory** `[PROJECTED from measured B/example]`:

| arm | activations + `x_train` | + eval reserve | share of `6.855 GiB` free |
|---|---|---|---|
| `P` | **0.493 GiB** (floor, §1) | 0.693 GiB | **10.1%** |
| `S` | **0.258 GiB** | 0.458 GiB | 6.7% |

**R1′'s memory is not at risk under any plausible correction to §1.** At
`n = 2048` the 20%-margin budget allows **`2,791,309 B/example` = 10.79×
`arm_phase`'s measured figure** before the cell stops fitting `[PROJECTED]`. The
S-M′ arm materialises roughly twice `arm_phase`'s `[n,s,s]` tensors, not eleven
times. **R1′ is priced with confidence; R2 is not, and §3.3 is why.**

### 3.2 R-SKY — it.10, the scan skyline read beside R1′ and R2

| | |
|---|---|
| cells | **16** = 8 @ R1′'s `n` + 8 @ R2's `n` |
| arm class | **`S`** — it.9 specifies `m = 93`, `4,769` parameters, which is exactly softmax's count `[INHERITED, V16_DEVICE_CERT.md §6.1]`, so it is param-matched and priced as softmax-class |
| why both scales | PART IV: *"read beside R1′/R2"*. `Δ_sky` is a **same-cell** column — a skyline at a different `n` from the arm it is differenced against is not a distance, it is two measurements |

| ladder | rung | point | **pessimistic** | working set |
|---|---|---|---|---|
| **150** | @ R1′, `n = 2048` | 0.0035 h | 0.0037 h | 0.258 GiB |
| **150** | @ R2, `n = 32768` | 0.0525 h | 0.0550 h | **4.130 GiB** |
| **150** | @ R2, `n = 16384` *(if R2 moves)* | 0.0268 h | 0.0280 h | 2.065 GiB |
| **150** | **total, as registered** | **0.0561 h** | **0.0587 h** | |
| **9600** | @ R1′, `n = 2048` | 0.2262 h | 0.2367 h | 0.258 GiB |
| **9600** | @ R2, `n = 32768` | 3.3621 h | 3.5185 h | **4.130 GiB** |
| **9600** | @ R2, `n = 16384` *(if R2 moves)* | 1.7123 h | 1.7920 h | 2.065 GiB |
| **9600** | **total, as registered** | **3.5884 h** | **3.7552 h** | |

**R-SKY fits at every candidate `n` and is the only one of the three cells with
no memory question in it** — it is softmax-class, and the certificate measured
softmax at `n = 32768` at **`4.000 GiB` against a `4.005 GiB` prediction,
`m/p = 0.999`** `[MEASURED, §3.3]`. **The `0.950` model holds here.**

**One open risk, named because it is not in the certificate.** R-SKY is *"the
gated **scan**"*. If it.9 builds it on `torch.cumsum`, it inherits the whole of
`V16_DEVICE_CERT.md` §5.3 — **BLOCKED under `use_deterministic_algorithms(True)`
on cuda** — which the round's own arm has just escaped by using `cumprod`
(`V16_ARM_SMPRIME.md` §9). The skyline does not exist yet, so this is a
precondition on it.9 rather than a finding, and it costs nothing to honour if it
is honoured before the module is written. **NOT MEASURED**: the skyline's dtype,
its scan primitive, and therefore its determinism status.

### 3.3 R2 — it.10, BED-M, `t* = 8`, `n = 32768` as registered

| | |
|---|---|
| cells | **16** = 8 seeds × {`P`, `S`} |
| `floor₁` | **`√(7/8) = 0.9354143466934853`** `[MEASURED from `scale/r10_it10_frontier.py:19`, formula at `scripts/v15_r1.py:18`]` |
| `ĥ` | `ĥ = t*(1 − NRMSE²)`, so **`ĥ = 1` exactly at `NRMSE = floor₁`** — one event, which is the double-payment PART IV repeals |

**Priced at all three candidate `n`, both ladders** `[PROJECTED]`:

| ladder | `n` | CUDA point | CUDA **pessimistic** | complex half | CPU point | complex working set | headroom | verdict |
|---|---|---|---|---|---|---|---|---|
| 150 | **32,768** | 0.162 h | 0.170 h | 0.110 h | 1.97 h | **7.892 GiB** | **−17.1%** | **DOES NOT RUN** |
| 150 | 22,106 | 0.111 h | 0.116 h | 0.075 h | 1.27 h | 5.324 GiB | 21.0% | fits `arm_phase` exactly |
| 150 | **16,384** | **0.083 h** | **0.086 h** | 0.056 h | 0.90 h | **3.946 GiB** | **41.4%** | **fits** |
| 9600 | **32,768** | 10.382 h | 10.865 h | 7.020 h | 126.06 h | **7.892 GiB** | **−17.1%** | **DOES NOT RUN** |
| 9600 | 22,106 | 7.078 h | 7.407 h | 4.786 h | 81.03 h | 5.324 GiB | 21.0% | fits `arm_phase` exactly |
| 9600 | **16,384** | **5.288 h** | **5.533 h** | 3.575 h | 57.89 h | **3.946 GiB** | **41.4%** | **fits** |

*Headroom is against `(6.855 − 0.200) GiB`. The softmax half fits at every row:
`4.130 / 2.786 / 2.065 GiB` respectively.*

**`n = 32768` is not a squeeze, it is an impossibility.** `7.892 GiB` of
activations exceeds **`6.855 GiB` free** *and* exceeds **`7.996 GiB` total**
`[MEASURED, V16_DEVICE_CERT.md §3.3]`. The certificate did not attempt it and was
right not to. **It does not fit on CPU either** — `7.893 GiB` against a host that
showed `8.0–10.0 GiB` available, *before* the interpreter, torch's image and both
corpora, which `V15_R1.md` §9 measured at **`1.0023 GiB` of process working set
for a cell 16× smaller** `[MEASURED]`. **Moving to CPU is not a route around
this**, and the 11.94× is not what is at risk here — a memory wall has no device
in it.

### 3.4 THE R2 DECISION, MADE EXPLICITLY

**RECOMMENDATION: `n = 16,384`.** Four reasons, in order of how much they bind.

**(a) It is the only candidate with headroom for the arm that will actually
run.** This is the argument the certificate could not make, because it did not
know it.5 would ship a different module. The affordable bytes-per-example at each
`n`, as a multiple of `arm_phase`'s **measured** figure `[PROJECTED]`:

| `n` | allowance @ 20% margin | as × `arm_phase` | allowance @ 0% | as × `arm_phase` |
|---|---|---|---|---|
| 32,768 | 174,457 B | **0.675×** | 218,071 B | **0.843×** |
| 22,106 | 258,600 B | **1.000×** | 323,249 B | 1.250× |
| **16,384** | **348,914 B** | **1.349×** | 436,142 B | **1.686×** |
| 8,192 | 697,827 B | 2.698× | 872,284 B | 3.373× |

**`n = 22,106` is affordable only if the S-M′ arm costs not one byte more than
`arm_phase`.** That is what a 20% cap computed *from `arm_phase`'s own bytes*
means, and §1 shows the S-M′ arm carries a whole second `path_product` that
`arm_phase` does not. **`n = 22,106` is therefore a cap for a module that is not
running, quoted to three significant figures, and adopting it would be exactly
the transfer-of-a-constant-across-a-change defect the certificate names three
times** (V-22, the `C_OPERATOR` transfer of §3.2, the determinism measurement of
§5.3). `n = 16,384` leaves **1.349×** of room at the same margin, which is the
first candidate that could absorb the second path product without a
re-measurement deciding the round.

**(b) `floor₁` does not move, and the brief is right that it does not.**
`floor₁ = √((t*−1)/t*)` is a closed form in `t*` alone: **`0.9354143466934853` at
`t* = 8`, with no `n` and no device in it** `[MEASURED from source]`. `ĥ = t*(1 −
NRMSE²)` likewise. **So the verdict's threshold at `n = 16,384` is the same
number it is at `n = 32,768`**, and R2 remains a legitimate floor-crossing
reading — the same argument `V16_DEVICE_CERT.md` §5.5.2 makes for reading a CUDA
R1′ against `floor₁ = 0.7071`.

**(c) It is a power of two, and the campaign's ladder is powers of two.** R1′ at
`2,048`, R7's rungs at `512 / 1,024 / 2,048`, R2 at `32,768`. `22,106` sits on no
ladder, is reproducible only by quoting a cap that depends on a VRAM snapshot
(`6.855 GiB`, which the certificate's own LIMITS call *"a property of today"* —
`6.939 GiB` free gives `22,385`), and would put a **derived** number into a
cell's identity. **A registered `n` that changes when a display driver allocates
differently is not a registration.**

**(d) It is cheaper, at both ladders, in the same direction as the fit.**
**5.288 h against 10.382 h** at 9600, **0.083 h against 0.162 h** at the floor.
With R-SKY@R2 following (§3.2), the it.10 block goes **13.744 h → 7.000 h**
`[PROJECTED]`. The schedule improves under the option that also fits, which is
worth stating plainly because it is the rare case where the two arguments point
the same way.

---

**WHAT THE ROUND LOSES, stated as specifically as the recommendation.**

**1. Half the data budget, and the resolution that budget buys.** `floor₁` does
not move — but **`Δ_res = t_{.975,N−1}·sd/√N` does**, because the seed `sd` is a
function of `n` (`V15_NEPTUNE_SYSTEMS.md` §2.5's own argument). **A narrower data
budget widens the seed spread, and a wider spread is more likely to STRADDLE the
floor — which is precisely how R1 failed.** R1 read `5/8` seeds across `floor₁`
with a CI that straddled and an achieved power of **`0.232077`**
`[INHERITED, V15_R1.md §9, LIMITS]`. **This is the honest cost and it lands
directly on the two scoreboard events R2 exists to pay** (`R2 ĥ>1`, **+4**; and
the `+12` R1′ banking that R2 is read beside). It is not priced in hours because
it is not an hours question, and this node does not model it: **NOT MEASURED —
the seed `sd` at `n = 16,384` is a capability quantity, and L-LEAN forbids
running the cell to find it.**

**2. R2 is no longer the contract's registered cell.** PART IV names
`n = 32768`. A future R2 at `32,768` — on a card with more VRAM, or with a
memory-cheaper operator — **may not be pooled with this one**, by exactly the
logic `V16_DEVICE_CERT.md` §5.5.1 applies to device: the change is larger than
M-10's measured floor of `2.345e-3` on `eval_nrmse`. The `n` must be journalled
in the cell's identity, and it already is (`cell=f"{kind}:t{T_STAR}:n{a.n_train}:seed{seed}"`,
`scripts/v15_r1.py:432` `[MEASURED by read]`).

**3. R4's matched-`n` sentence, unless R3 moves too.** R3(a)/R3(b)/R5 sit at
`n = 32768` `[ASSUMED, V16_DEVICE_CERT.md §6.1]` **specifically so R4 compares
two beds at one resolution**. If R2 moves and R3 does not, R4's table compares
BED-M at `16,384` against BED-K at `32,768`, and its sentence means less. §6.3
prices the consistent alternative — **every `n = 32768` node at `16,384`** — and
it is *cheaper*, not more expensive: **36.428 h against 68.354 h**.

**4. What is NOT lost.** The `t* = 8` bed, the `floor₁` threshold, the `ĥ`
identity, the two-arm structure, the eight seeds, and the one-event scoring. **No
scoreboard event is forfeited by the `n` change itself.**

---

**The two options this node does not recommend, priced so the choice is a
choice.**

| option | cost at 9600 | what it buys | why not |
|---|---|---|---|
| **`n = 22,106`** | 7.078 / 7.407 h | 33% more data than 16,384 | zero headroom for the arm that runs (**1.000×** `arm_phase`); not a power of two; cap depends on a VRAM snapshot |
| **R2 does not run** | 0 h | nothing | forfeits **+4** and the deciding cell; leaves `ĥ > 1` unattempted for a fourth round; and the round's `floor₁` at `t* = 8` is the only place `ĥ > 1` is reachable |
| *(struck)* R2 with real arms only at `n = 32,768` | 3.362 h | the registered `n` | **the arm the round exists to measure is absent from its deciding cell** — this is the certificate's own option 2 and it is a worse trade than a smaller `n` |
| *(struck)* a memory-cheaper complex operator | — | — | **a new construction; the round's first law strikes it** |

---

## 4. EVERY PROJECTION AS A PAIR — THE PESSIMISTIC END IS NOT OPTIONAL

`V16_CALIBRATION.md` records R11's errors as **`7 of 8` signed rows optimistic,
one-sided sign test `p = 0.0352`** `[MEASURED, V16_CALIBRATION.md §
calibration column]` — *"direction established at α = 0.05."* **D-CALIB-3**
authorises sign and forbids size, which is why the pessimistic column below is a
**measured spread** and not a discount.

| cell | ladder | CUDA point | **CUDA pessimistic** | CPU point | **CPU pessimistic** |
|---|---|---|---|---|---|
| **R1′** | 150 | 0.0109 h | **0.0114 h** | 0.088 h | **0.223 h** |
| **R1′** | 9600 | 0.6985 h | **0.7310 h** | 5.605 h | **14.273 h** |
| **R2 @ 32768** | 150 | 0.1622 h | **0.1698 h** | 1.97 h | **5.02 h** |
| **R2 @ 32768** | 9600 | 10.3819 h | **10.8648 h** | 126.06 h | **321.00 h** |
| **R2 @ 16384** | 150 | 0.0826 h | **0.0865 h** | 0.90 h | **2.30 h** |
| **R2 @ 16384** | 9600 | 5.2875 h | **5.5334 h** | 57.89 h | **147.42 h** |
| **R-SKY** | 150 | 0.0561 h | **0.0587 h** | 0.666 h | **1.696 h** |
| **R-SKY** | 9600 | 3.5884 h | **3.7552 h** | 42.640 h | **108.573 h** |
| **PLAN, 232 cells** | 150 | 1.068 h | **1.118 h** | 12.748 h | **32.460 h** |
| **PLAN, 232 cells** | 9600 | **68.354 h** | **71.533 h** | **815.877 h** | **2,077.461 h** |

**The decision is 11.94× at the point estimate and 29.0× at the pessimistic
end.** `[MEASURED spreads, V16_DEVICE_CERT.md §2.3; PROJECTED totals]` **The
contract's cited 13.6× is the optimistic reading of the decision**, and the
asymmetry is not symmetric noise: the pessimistic end is worse for CPU by a
factor of 2.43, because the contention and placement spread falls almost entirely
on the host lane (`2.55×` CPU against `1.05×` CUDA under the identical sweep).

**A schedule quoted only at its point estimate is the exact failure the
calibration column exists to prevent, and there is a worked example in this
round's own files.** `V15_MERCURY_DEVICE.md` filed the determinism cost at
`0.921×` from a single measurement on the one arm strict mode does not block; the
certificate re-measured and found two of three arms **BLOCKED**
`[MEASURED, V16_DEVICE_CERT.md §5.3]`. **A single reading in the favourable
direction is how a round acquires an optimistic schedule, and both halves of that
sentence are on this round's record.**

**A pessimistic factor this file does NOT apply, and says so.** The complex arm's
`×2.085` and `258,609 B/example` are floors (§1), and the honest pessimistic end
of those is unbounded until measured. §6.4 gives the sensitivity rather than
inventing a factor — **a made-up multiplier on an unmeasured quantity would be
size, which D-CALIB-3 forbids.**

---

## 5. THE P-CORE / E-CORE HAZARD AS A SCHEDULE RISK

**The finding, restated in its own terms.** `V16_DEVICE_CERT.md` §2.3 measured
the same cell — same arm, same shape, same `threads = 8`, same corpus, same box —
at **`0.0912 s/step` on P-cores and `0.2701 s/step` on E-cores under contention**,
and that single E-cores × contention configuration read **`2.96×`** in the run of
record and **`2.00×, 2.18×, 2.75×, 2.93×`** in four earlier runs and **`4.03×`**
in an independent standalone measurement `[MEASURED]`. **Six measurements of ONE
configuration spanning 2.0× to 4.0×.**

**Why averaging does not remove it, and this is the whole point.** The spread is
**between configurations, not within one**. `torch.set_num_threads(k)` names a
**count**, never a **placement**; the Windows scheduler decides which core class
eight threads land on, and it demotes work it reads as background. Averaging
`n` samples of a quantity reduces the standard error of the *mean* by `√n` — but
here the samples are drawn from different machines wearing the same name.
**More samples estimate the mixture more precisely; they do not tell a scheduler
what the next cell will cost, and they do not make two cells comparable.**

**Three consequences, in order of what they cost.**

**1. A CPU schedule is a range, not a number, and the range is multiplicative on
every CPU hour.** R1′ on CPU at the 9600 ladder: **5.605 h** point, **11.211 h**
at the band's lower edge (`2.0×`), **22.421 h** at its upper (`4.0×`), and
**32.511 h** if R1's own cell is admitted as a sample (`5.8×`) `[PROJECTED]`.
**A node that must plan around a 5.6-hour job and a 32.5-hour job is not
planning.** The 40-iteration CPU plan at the floor is **12.748 h** point and
**32.460 h** pessimistic; at the ladder, **815.877 h** (34.0 days) and
**2,077.461 h** (86.6 days). **Neither ladder figure is a schedule at any point
in that band.**

**2. It is a comparability hazard before it is a schedule hazard, and that is the
more expensive half.** M-10 measured **thread count alone moving `eval_nrmse` by
`2.345e-3`** — `0.464` of `Δ_eq = 0.005051` — which is why `threads` is part of a
cell's identity. **Under M-10's own logic core class is part of a cell's identity
too, and nothing in the repository records it** (`V16_DEVICE_CERT.md` §2.6). So
two CPU cells journalled at the same `threads` may have run on different core
classes and are **not known to be poolable**. The fix is one line — pin affinity,
not only the count, and journal it beside `threads` — and it is a `MISTAKES.md`
candidate outside this node's write scope, as it was outside the certificate's.

**3. The certified CUDA lane removes it for the tensor work, and only for the
tensor work.** Under the identical host-pressure sweep the CUDA cell's spread is
**`1.05×`** against the CPU cell's **`2.55×`** `[MEASURED]`, and the CUDA
**exponent** reproduced across eight runs at a spread of `0.005` against the CPU
exponent's `0.050` — **ten times the more stable object**. That is the
schedulability argument, it is independent of the speed argument, and this node
endorses it: **on CUDA the wall clock is a property of the work; on CPU it is a
property of the box's mood.**

**What the CUDA lane does NOT remove, named rather than waved past:**
- **The host work beside each cell.** Corpus construction, `calibrate_bar`'s
  seeded **CPU** generator (`negation_scope.py:96, :426–427`), journaling, and
  the eval pass all touch the host. They are small — `calibrate_bar` costs
  **`0.447 s` on cuda** `[MEASURED, V16_DEVICE_CERT.md §5.2]` — and small × 4.0
  is still small. **Not a schedule risk; named so the claim is bounded.**
- **Exit 3, if anyone takes it.** `V16_DEVICE_CERT.md` §5.3.1 exit 3 moves R1′ to
  CPU at `5.61 CPU-h`. **The determinism conflict that motivated exit 3 is gone**
  for the round's arm (§0.2), so exit 3 has lost its reason — and if it is taken
  anyway, **R1′'s wall clock re-acquires the 2.0×–4.0× band**, i.e. the round's
  cheapest node becomes its least predictable one. **Recommendation: do not take
  exit 3.**
- **The 1.05× is a property of today.** `scale/hyperbolic.py:239` selects cuda
  automatically when available; a concurrent node running it shares the device
  and the argument weakens by however much that takes
  `[INHERITED, V16_DEVICE_CERT.md LIMITS]`. **Under the campaign's own
  parallel-planet dispatch that is not a hypothetical** — §2.4 dates a competing
  `lake build` finishing *inside* R1's own measurement window and reads a **13.1%
  step down** across it.

---

## 6. THE 40-ITERATION PLAN ON THE CERTIFIED DEVICE

### 6.1 As registered — 232 cells

`[PROJECTED from §2's law; reproduces V16_DEVICE_CERT.md §6.2 to the digit]`

| it | node | `n` | `P` cells | `S` cells | CUDA point | **CUDA pess.** | CUDA point | **CUDA pess.** |
|---|---|---|---|---|---|---|---|---|
| | | | | | *150 steps* | | *9600 steps* | |
| 7 | R1′ | 2,048 | 8 | 8 | 0.0109 | 0.0114 | 0.6985 | 0.7310 |
| 10 | **R2** | 32,768 | 8 | 8 | 0.1622 | 0.1698 | **10.3819** | **10.8648** |
| 10 | R-SKY @ R1′ | 2,048 | 0 | 8 | 0.0035 | 0.0037 | 0.2262 | 0.2367 |
| 10 | R-SKY @ R2 | 32,768 | 0 | 8 | 0.0525 | 0.0550 | 3.3621 | 3.5185 |
| 12 | MARS param-match | 2,048 | 0 | 8 | 0.0035 | 0.0037 | 0.2262 | 0.2367 |
| 17 | **R3(a)** delay | 32,768 | 8 | 24 | 0.2673 | 0.2797 | **17.1062** | **17.9018** |
| 18 | **R3(b)** power-law | 32,768 | 8 | 24 | 0.2673 | 0.2797 | **17.1062** | **17.9018** |
| 20 | R4 skylines | 32,768 | 0 | 16 | 0.1051 | 0.1100 | 6.7243 | 7.0370 |
| 25 | **R5** BED-1 | 32,768 | 8 | 8 | 0.1622 | 0.1698 | **10.3819** | **10.8648** |
| 28 | R6 hidden-cause | 2,048 | 0 | 32 | 0.0141 | 0.0148 | 0.9049 | 0.9469 |
| 29 | R7 `ζ` rung | 512 | 8 | 8 | 0.0028 | 0.0030 | 0.1812 | 0.1896 |
| 29 | R7 `ζ` rung | 1,024 | 8 | 8 | 0.0056 | 0.0058 | 0.3558 | 0.3723 |
| 29 | R7 `ζ` rung | 2,048 | 8 | 8 | 0.0109 | 0.0114 | 0.6985 | 0.7310 |
| | **TOTAL** | | **64** | **168** | **1.068** | **1.118** | **68.354** | **71.533** |

**Against CPU: `12.748` / `32.460` at the floor and `815.877` / `2,077.461` at
the ladder** — **11.94×** point, **29.0×** pessimistic.

**68.354 GPU-hours is under three days of AFK loop; 71.533 is under three days
too.** `815.877` CPU-hours is 34.0 days and `2,077.461` is 86.6 days, and
**neither is a schedule.** At the 150-step floor the CPU plan is `12.748 h` and
**is** affordable — which matters, because R1 ran at 150 steps, so **the floor is
the budget the round has actually been buying.**

### 6.2 The four misfit nodes moved for the complex arm only

`V16_DEVICE_CERT.md` §6.2's adjustment, reproduced: **54.576 h** at 9600
`[PROJECTED; certificate prints 54.58]`, **0.853 h** at the floor.

**This node does not recommend it.** It moves R2's complex arm to `16,384` while
leaving **R-SKY @ R2 and R4 at `32,768`**, so `Δ_sky` differences a cell at one
`n` against a skyline at another — which is not a distance (§3.2).

### 6.3 Every `n = 32768` node at `n = 16384` — the consistent alternative

| ladder | CUDA point | **CUDA pessimistic** | CPU point | **CPU pessimistic** |
|---|---|---|---|---|
| **150** | **0.569 h** | **0.596 h** | 6.072 h | 15.462 h |
| **9600** | **36.428 h** | **38.122 h** | 388.636 h | 989.581 h |

**The whole plan at one resolution costs 36.428 GPU-h against 68.354** — a **47%
saving** — and it is the only variant in which R4's two-bed table, `Δ_sky`, and
R2's own arm-vs-control contrast all compare cells at a single `n`. **Every
memory misfit clears**: the complex working set at `16,384` is `3.946 GiB`
against a `5.324 GiB` budget at a 20% margin, with `1.349×` of room for §1's
correction. **On CPU it also brings the ladder from 34.0 days to 16.2 days** —
still not a schedule, and the point estimate at that.

### 6.4 The sensitivity that matters more than either total

The complex column is **`29.387` of `68.354 h` = 43.0%** as registered, and
**`15.608` of `36.428 h` = 42.8%** at one resolution `[PROJECTED]`. §1 says
`×2.085` is a floor. So:

| S-M′ arm's true cost | as registered, 9600 | at one resolution, 9600 |
|---|---|---|
| `1.0×` `arm_phase` *(the floor, priced above)* | **68.354 h** | **36.428 h** |
| `1.5×` | 83.047 h | 44.232 h |
| `2.0×` | 97.741 h | 52.036 h |
| `3.0×` | 127.128 h | 67.645 h |

**The whole band is affordable in GPU-hours** — even `3.0×` at one resolution is
`67.6 h`, under the plan's own headline figure. **The uncertainty that matters is
not the hours, it is the bytes**, because bytes decide whether a cell runs at all
and hours only decide when it finishes. **`1.349×` is the number to beat at
`n = 16,384`, and it is one probe run away.**

---

## 7. EVERY CELL THAT DOES NOT FIT OR DOES NOT RUN

### 7.1 Does not FIT — memory, at `n = 32768` with a complex arm

`[MEASURED, V16_DEVICE_CERT.md §3.4]`. Caps: **`22,106` at a 20% margin,
`27,633` at zero, last power of two `16,384`.** Memory is not a function of the
step count, so these hold at both ladders.

| it | node | cells at risk | the arithmetic |
|---|---|---|---|
| **10** | **R2** BED-M `t*=8` | 8 of 16 | `7.892 GiB` vs `6.855` free, `7.996` **total** — *the deciding cell of the round* |
| 17 | R3(a) delay bed | 8 of 32 | the composed arm **if complex** — **NOT MEASURED** |
| 18 | R3(b) power-law bed | 8 of 32 | same |
| **25** | **R5** BED-1 | 8 of 16 | `7.892 GiB`, identical to R2 |

**And the caps themselves are for `arm_phase`, not for the arm that runs
(§1/§3.4).** The true caps are **at or below** those three numbers. **`22,106`
in particular has exactly zero headroom for any arm heavier than `arm_phase`,
and the shipped arm is heavier.**

### 7.2 Does not RUN — independent of memory, independent of `n`

| # | what | scope | status |
|---|---|---|---|
| **1a** | **Runner reachability, `scripts/v15_r1.py`** — `make_arm` at HEAD `deee6c4` knows `"arm_pl"` and `Arm(kind, s)`; **in the working tree (`M`, another node's file) it also builds `arm_smprime.ArmSMPrime`** at line 118-123 | R1′'s 8 complex cells, and R2's if it is driven from this script | **DISCHARGED in the working tree, OPEN at HEAD** `[MEASURED by read of both]` |
| **1b** | **Runner reachability, `scale/r10_capacity_sweep.py:102`** — constructs `Arm(arm, s)`, which resolves neither complex module. File is **clean**, not `M` | every complex cell driven from the sweep | **OPEN** `[MEASURED by read]` |
| **2** | The complex arm's bytes and s/step | every complex cell | **NOT MEASURED.** §1. Priced at a floor |
| **3** | R-SKY's scan primitive and its determinism status | 16 cells | **NOT MEASURED** — the skyline is built at it.9. If it uses `cumsum` it **BLOCKS** under the strict flag on cuda (§3.2) |
| **4** | R3's composed arm's dtype | 16 of 64 cells | **NOT MEASURED.** If X₃₈'s coupling is real-valued, R3(a)/(b) each fall ~3.4 GPU-h and both **fit at `n = 32768`** `[INHERITED, V16_DEVICE_CERT.md LIMITS]` |
| **5** | **D-R3's two inverse registrations** | R3(a), R3(b) — 64 cells | **A CONTRACT PRECONDITION, not a task** (PART III, PART V: *"RECONCILED BY THE AUTHOR before any cell runs"*). **No R3 hour is spendable until it.13** |
| **6** | BED-K does not exist | R3, R4 — 80 cells | built at it.13. `n` and `s` are `[ASSUMED]`; **if BED-K needs `s = 256`, matched-`n` caps BOTH beds near `n ≈ 5,273`** `[INHERITED, V15_NEPTUNE_SYSTEMS.md §2.5]` |

### 7.3 Two the certificate flagged that are now DISCHARGED

Reported because a schedule that still carries them is wrong in the expensive
direction — it under-books work it believes is blocked.

| certificate condition | status | evidence |
|---|---|---|
| **4** — the arm cannot complete 40 gradient steps on either device; *"no hour above is spendable"* | **DISCHARGED for the shipped arm** | `V16_ARM_SMPRIME.md` (g), §6.1: **8/8** configurations, none non-finite in 40 steps, at `55.58%` of positions on the closed endpoint `[MEASURED]` |
| **1** — the determinism conflict; exits 1/2/3 priced | **DISCHARGED for the shipped arm** | `V16_ARM_SMPRIME.md` §9: `arm_smprime.operator` and `path_product` **OK** under `use_deterministic_algorithms(True)` on cuda; `cumprod` has a deterministic kernel, `cumsum` does not `[MEASURED]` |

**Consequence for the schedule: exit 2's `warn_only` is no longer needed for the
round's own arm, and exit 3 has lost its motive.** The bar's certification regime
(`V16_BAR_RECERT.md`, strict) and the run's regime can now be **the same flag at
the same setting with no exemption at all** — which is strictly better than the
exit 2 the certificate had to recommend. **This is the cheapest schedule
improvement available to the round and it was bought by a theorem, not by a
budget.** It is contingent on the skyline (§7.2 item 3) and on the softmax
control, which strict mode does not block (`0.01172 s/step`, `[MEASURED,
V16_DEVICE_CERT.md §5.3]`).

**Certificate condition 3 — R2's `n` reconciled with §3.4 — is what §3.4 of this
file answers.** Condition 2 (the bar read in the run's regime) is untouched by
this node.

---

## 8. LIMITS

Everything this node could not establish, collected once.

- **The complex arm's cost is NOT MEASURED and every complex figure here is a
  floor.** `258,609 B/example` and `×2.085` belong to `ceq/arm_phase.py`; the
  round runs `ceq/arm_smprime.py`, which materialises a second full
  `path_product` and three quotient tensors `arm_phase` does not (§1). The
  direction is established from source; **the magnitude is not, and this node
  does not claim it is.** The correction is `scripts/v16_device_probe.py`
  sections C1 and D3 pointed at the new module — pricing is not measuring, and
  the brief forbids running cells.
- **`1.349×` is the number the `n = 16,384` recommendation rests on**, and it is
  a computed allowance, not a measured margin. If the S-M′ arm exceeds it, the
  next power of two down is `8,192` (allowance `2.698×`), and R2's data budget
  halves again.
- **The seed `sd` at `n = 16,384` is NOT MEASURED**, so the resolution cost of
  §3.4's recommendation is stated as a mechanism and not as a width. It is the
  quantity that decides whether R2's CI clears `floor₁`, it is a capability
  question, and L-LEAN forbids buying it here.
- **`ceq/sizing.py` is not used for any complex figure**, by instruction and by
  measurement: `m/p = 1.842` under-predicting, `C_RESIDUAL` **NOT IDENTIFIED**
  at `s = 64`. Every complex byte here descends from measured bytes-per-example.
  **If BED-K forces a longer sequence the decomposition must be re-identified at
  the new `s`** before anything is priced from it.
- **The caps rest on a VRAM snapshot.** `6.855 GiB` free at the certificate's C4
  run against `6.939 GiB` at its §1 — an `84 MiB` difference that moves the
  complex cap from `22,106` to `22,385`. **A cap that moves with the display's
  allocation is a reason to prefer a power of two, not a number to register.**
- **Arms-per-node is `[ASSUMED]` at every row**, inherited from
  `V16_DEVICE_CERT.md` §6.1. The contract specifies its deciding measurements by
  their readings, not by their arm lists. A different census moves §6
  proportionally.
- **it.19's bandwidth sweep is priced nowhere in §6.** If **D-CALIB-1** is
  honoured — the author's counter is the point estimate, and R3's counter is
  *"composed loses on (a)"* — the sweep is the **base case**, and it adds
  **14.30–28.08 GPU-h** `[INHERITED, V16_DEVICE_CERT.md §6.1]`. **At 28.08 h it
  is 41% of the plan**, and it is excluded only because the contract does not
  list it as a cell.
- **No timing figure is inherited** (L-TIME). `V15_R1.md`'s `91.63 s` and
  `71.32 s` are quoted twice, both times as the subject of a diagnosis rather
  than as an input: they **reproduce at `0.15×` and `0.17×`** on a quiet lane
  `[MEASURED, V16_DEVICE_CERT.md §2.2]`, and **R1's timings remain void for
  pricing**, as R1 itself ruled. Nothing in §3 or §6 descends from them.
- **The certificate's law is cited, not re-derived, and this node does not
  believe it is wrong.** Its CUDA exponent reproduced to four decimals across
  eight runs at a spread of `0.005`, R² `0.999384`. **The only quantity this
  node re-computed from it was to check the arithmetic was being applied
  identically — it reproduces the certificate's totals to the printed digit.**
- **This file was written against a moving tree, and that is stated rather than
  hidden.** Repo HEAD at read time `deee6c4`; `scripts/v15_r1.py` and
  `house-events.jsonl` were `M` in `git status` throughout, and
  `V16_VENUS_R1PRIME.md` appeared untracked while this ran. **The runner claim
  in §0.3 / §7.2 reverses between HEAD and the working tree**, which is why both
  states are printed rather than one asserted. `ceq/` and
  `scale/r10_capacity_sweep.py` were clean and were read, never written. **If
  `scripts/v15_r1.py` lands differently from the state read here, §7.2 item 1a
  must be re-read before it.7 commits** — the same call `V16_DEVICE_CERT.md`
  §5.0 made about its own §5, one file along.
- **Nothing here is trained, fitted, or scored.** No cell, no arm, no gradient
  step, no NRMSE, no seed, no verdict. **No git command that writes was run**,
  and the only file written is this one.

---

**Distance to the north star.** This node moves no measurement toward it and is
not supposed to: RULE 4 prices, and what it prices is whether the round's two
deciding readings can be bought at all. **They can.** R1′ costs `0.70` GPU-h
point and `0.73` pessimistic and fits in 10% of the card; R2 costs `5.29 / 5.53`
at the `n = 16,384` this file recommends, where the certificate's `n = 32,768`
costs `10.38` and **does not run at any price**; R-SKY costs `3.59 / 3.76` and
follows R2's `n`. **The 40 iterations cost `68.354` GPU-h as registered and
`36.428` at one resolution, against `815.877` CPU-hours — an 11.94× decision at
the point estimate and 29.0× at the pessimistic end, and the contract's own
13.6× is the optimistic reading of it.** Two of the certificate's four signed
conditions were discharged not by a budget but by it.5 replacing a `cumsum` of
`log m` with a cumulative product, which is the round's first law paying for
itself. **The claim sentence this pricing permits is: the deciding cells are
affordable on the certified device at both ladders, the deciding cell of the
round does not run at its registered `n` on any device, and the one number that
could still move the answer — what the shipped complex arm actually costs — is a
single probe run that nobody has made.**
