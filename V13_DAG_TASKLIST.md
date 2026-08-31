# V13 — the work list, aligned to CEQ v-main.8

Under CONTRACT.md §0. Every row carries its D-1 class, its blocker, and its
status. Planet names are accountability labels, not dispatch instructions (D-2).
Architecture-and-state document is `workdonenew.md`; `ARCH.md` does not exist.

## A. PARALLEL-SAFE — no shared repository state

| # | Node | Output | Status |
|---|------|--------|--------|
| A1 | Tier-6 prior art: MSM, EBM, committor learning, saddle search, options/HRL | `V13_TIER6_PRIOR_ART.md` | **DONE** |
| A2 | D-3 forensics: deactivation, MISTAKES, standing failures | `V13_D3_LOOP_FORENSICS.md` | **DONE** |
| A3 | Tier-1/2 derivations: discrete Kramers, fan-out null | `V13_TIER12_DERIVATIONS.md`, `scripts/v13_derivation_check.py` | **DONE** — every assert passes, exit 0, 13.6 s |
| A4 | X₂₅ G1: NMN, recursive nets, program graphs | `V13_X25_G1_PRIOR_ART.md` | **DONE** |
| A5 | X₂₆: CUSUM/ARL calibration on loss trajectories | `V13_X26_TRAJECTORY_MONITOR.md`, `scripts/v13_cusum_arl_calib.py` | **STRUCK** — lead-time CI `[−13.88, +1.50]` includes zero; kill clause fired |
| A6 | X₂₇ G1: Grebogi–Ott–Yorke, Tél transient chaos, uncertainty exponent in ML | `V13_X27_G1_PRIOR_ART.md` | **DONE** — X₂₇b STRUCK (theorem absent and false as written), X₂₇a instrument not novel, X₂₇c/d collapsed |
| A7 | Ralph-loop cap-parsing defect (V-20), fix and verify by execution | `V13_RALPH_LOOP_FIX.md` | **DONE** — cause was a prose-form cap, not a parser bug; fix verified by 8 mount + 6 hook cases |
| A8 | Chain-corpus sweep could not select an arm; add `--arm` | `scale/r10_capacity_sweep.py` | **DONE** — default `softmax` reproduces published cells at `delta = 0.000e+00` |
| A9 | Table generator could not render THE READING; add arm select, derived `n`, `ĥ` column, floor verdict | `scale/r10_it8_table.py` | **DONE** — runs; first output below |
| A10 | 0-step gate fired on its own null; add measured tolerance | `scale/r10_capacity_sweep.py`, `tests/test_zero_step_gate.py` | **DONE** — `GATE_TOL = 1e-3` from a 16-seed null; 4/4 tests pass |
| A11b | Gain sweep identity check | `scripts/v13_hop2_gain.py` | **PASSES** — `gain=0.00` reads `0.951602` sd `0.015381`, `+0.000000` against the softmax control, so `z = x + a@x` is softmax's forward exactly |
| A11 | X₂₈b tangent kit: JVP, Lyapunov via Benettin, adjoint gradcheck, both must-fires | `V13_X28B_TANGENT_KIT.md`, `scripts/v13_tangent_kit.py` | **PARTIAL, report claim false** - positive must-fire passes at `6.114e-10` from `ln 2`, but the script exits on an AssertionError while the report says "exit 0, every assertion passed" with its numbers left as placeholders |
| A12 | X₂₈c prior art: Kantz–Grassberger `κ = λ(1−d)`, and whether the triangle is circular | `V13_X28C_PRIOR_ART.md` | **DONE** — 720 lines; `d` is `D₁` not `D₀`, so X₂₇a's box counting cannot feed it; triangle not circular but not a physics test |
| A13 | Independent audit of this session's numeric claims | `V13_CLAIM_AUDIT.md` | **DONE** — 44 CONFIRMED / 9 DISCREPANT / 1 UNVERIFIABLE; all nine corrected, B9 reopened |
| A17 | Mechanical adjudicator for the filed hop-2 prediction | `scripts/v13_adjudicate_hop2.py` | **DONE** — refuses below N=8; currently exits 2 at 4/8 |
| A14 | X₂₉a differential hops: deflated operator, CMRR, both must-fires | `scripts/v13_deflated_hop2.py`, `V13_X29A_DEFLATION.md` | **DONE** - prediction holds for the renorm projector (`0.951400`, 3/3 seeds below `0.960945`, p=0.0399); the hop stops doing damage and still does not work. Superseded note: **PARTIAL** — must-fires, CMRR and causality guard done; §8's trained comparison is still `SWEEP_TABLE_PLACEHOLDER`, so the pre-registered recovery number does not exist |
| A15 | X₂₉b Wiener equalizer: per-mode gains, water-filling, DC-gain must fall out | `scripts/v13_wiener_hop.py`, `V13_X29B_WIENER.md` | **DONE** — both must-fires pass; all 64 modes inside the null band trained and untrained; Wiener K=64 `0.958816`, not resolved against softmax |
| A16 | X₂₉ prior art: graph-SP Wiener, VSA/HRR, Walsh/CDMA, comms-in-ML | `V13_X29_PRIOR_ART.md` | RUNNING |

## B. CHAIN — strictly ordered, one lane

| # | Node | Blocked by | Status |
|---|------|-----------|--------|
| B0 | R10 wave B: `n=32768` at `t*` = 2, 8, 32 | — | **DONE, and done before this round began** |
| B1 | Round 11 it.1–22: registration + the reading | A8 | **NOT STARTED**, preconditions below (D-4 blocker for B2 onward) |
| B2 | BED-1 multi-basin bed + exact oracle, dual route ≤ 1e-10 | B1 | blocked |
| B3 | Admissibility C-A..C-F with δ(\|B\|), CK clause, Morse census | B2 | blocked |
| B4 | Fan-out detector + α-instrument, planted must-fires both directions | B3, A3, A6 | blocked |
| B5 | Basin head and barrier head, manifest-bound | B4 | blocked |
| B6 | S5′ reading with ceiling columns, CK gate, N=8 | B5, prediction filed first | blocked, **rescope owed** |
| B7 | S5′ verdict against the kills | B6 | blocked, **rescope owed** |
| B8 | TOST machinery, Δ_eq per bar, difference-test refusal must-fire | — | **DONE** — appended to `scale/it11_verdict.py`; self-check passes, incl. both must-fires |
| B8b | C-CAP floor verdict, `hop_floor` / `cap_verdict` | — | **DONE** — decidable at N=8; proves a planted crossing, refuses a short N, no hop count above the bar |
| B9 | 4060 throughput probe, 500 steps, tokens/sec and peak bytes | — | **REOPENED** — memory figures stand; the throughput figures came from a 3-step run, not 500, and `results/r10_v13_b9_4060_probe.txt` is 0 bytes so no artifact survives. Rerun queued for a quiet host |
| B10 | Local training run 1: winning arm + softmax control | B9, B7 | blocked |
| B11 | Local training run 2: AdamW / O1 / O2 ladder | B10 | blocked; **X₂₆ monitor removed, A5 STRUCK** |
| B12 | Parity table: TOST columns, floor columns, S5′ distance-to-ceiling | B10, B11, B8 | blocked |
| B13 | Claim-ladder verdict, C-PAR ∧ (C-CAP ∨ C-TS) | B12 | blocked |
| B14 | Package with local weights, limits-first card | B13 | blocked |

### B0 correction

Wave B was recorded in this list as needing a run. It did not.
`results/r10_it8_waveB.log` lines 16–26 hold the completed wave —
`t*=2 n=32768 eval NRMSE=0.874834 [LEARNS] boot[0.8620,0.8903] 624.5s`,
`t*=8 … 0.972372 [LEARNS] boot[0.9641,0.9814] 607.8s`,
`t*=32 … 1.006066 [NO READING] boot[1.0033,1.0094] 680.0s`, `WAVE B COMPLETE` —
and the first `bc: command not found` appears at line 27, after it. The journals
carry `n_train=32768` cells at seeds 0–7 for both `t*=8` and `t*=32`, a complete
N=8; `t*=2` has seed 0 only. Filed as V-19 (rewritten) and V-21.

## B1 preconditions established this iteration

**Runner.** `scale/r10_capacity_sweep.py` bound `Arm("softmax", s)` at line 83,
the journal name at 141 and the header arm field at 148, so the chain-corpus
runner could produce one arm only. THE READING needs four. An `--arm` flag now
threads through, restricted to the kinds `scale/m3_capability.py:131-153`
actually defines: `softmax`, `pivot_unsigned`, `windowed_signed`,
`pivot_signed`. The default is `softmax`, which reproduces the previous journal
path exactly and was verified to return published cells bit-for-bit.

**The fourth arm does not exist.** `khop-as-skyline` is not an `Arm`. The only
`khop` in the tree is `scale/r10_it17_battery.py:249`, an untrained function
`u_k = Σ_{m<k} P_II^m P_IB g` used as a reference inside the admissibility
battery. It is a skyline, not a trainable arm, and belongs in THE READING as a
computed reference column beside each cell rather than as a fourth training
run. Building it as an `Arm` would be inventing an object the contract only
ever described as a ceiling.

**Threads must be pinned before any cell is taken.** Filed as M-10: thread
count changes the result by `0.002345`, which is `0.464` of the equivalence
margin `Δ_eq = 0.005051` derived from the N=8 seed sd of `0.010101` at
`t*=2, n=2048`. The sweep's `--threads` default is 8; the published N=8 wave
was taken at 6. THE READING must fix one value, record it per cell, and refuse
to pool across values.

## B1 pricing, measured (supersedes any earlier estimate)

Cost is **superlinear in n**. On the idle host at `threads=12, steps=150,
t*=8`, measured medians are `n=2048 → 18.19 s`, `n=8192 → 94.74 s`,
`n=16384 → 293.94 s`, a log-log slope of `secs ~ n^1.338` between endpoints, or `n^1.317` by
three-point OLS, where linear would be `1.000`. The cause is visible in the runner: `train_with_checkpoints` is
full-batch — `model(x_train)` over all `n` at once — so the `[n, s, s]`
operator is materialised for every sample simultaneously. At `S=64` that tensor
is `0.268 GB` at `n=16384` and `1.611 GB` at `n=98304`, and `C_OPERATOR = 3.9`
of them are retained.

THE READING wants `n = 4096 / 32768 / 98304` at `t* = 2 / 8 / 32`. Extrapolating
the fitted slope gives, per arm at N=8, `0.10 h`, `1.65 h` and `7.18 h`, so
`8.93 h` per arm and about `26.8 h` for the three trainable arms. The last
figure extrapolates 6× beyond the largest measured point and is a floor rather
than an estimate: if page-file pressure begins above a ~2 GB working set the
true slope is worse. One anchor cell at `t*=32, n=98304` is running to replace
the extrapolation with a measurement.

Feasibility was checked by allocation, not by arithmetic. `ullAvailPhys`
reports `5.78 GB` free of `16.87 GB` at 65% load, which suggested the largest
point would not fit; a direct allocation of `x_train` plus four `[n, s, s]`
tensors at `n=98304` succeeded at `6.85 GB`, because Windows commits beyond
available-physical. The point is feasible.

**Lane discipline (M-10).** Threads change the number, and `paired_arm.py`
records that GPU rows are tolerance-checked against CPU and never bitwise. THE
READING must therefore run every cell in one lane — one device, one thread
count — and cells from different lanes must not be pooled into a seed spread.

## B1 first instrument output — the floor has never had a candidate

`scale/r10_it8_table.py` glob-ed `r10_%s_capacity_softmax_t*.jsonl`, pinned
`N_TRAIN = (2048, 8192, 32768)`, and had no `ĥ` anywhere in the tree, so it
could render neither the arms A8 enabled nor the `n` values THE READING asks
for. It now selects an arm, derives its `n` columns from the journal, and
prints the hop floor `√((t*−h)/t*)` at `h=1` beside the mechanism column
`ĥ = t*(1 − NRMSE²)`.

Its first output over the existing N=8 softmax cells:

| `t*` | `n` | mean | sd | CI95 | floor_1 | `ĥ` | verdict |
|---|---|---|---|---|---|---|---|
| 2 | 2048 | 0.952349 | 0.010101 | [0.946161, 0.959054] | 0.707107 | 0.186 | consistent with 1 hop |
| 8 | 16384 | 0.983154 | 0.004732 | [0.979991, 0.986083] | 0.935414 | 0.267 | consistent with 1 hop |
| 8 | 32768 | 0.975371 | 0.002360 | [0.973865, 0.976869] | 0.935414 | 0.389 | consistent with 1 hop |
| 32 | 32768 | 1.003371 | 0.003759 | [1.000851, 1.005679] | 0.984251 | −0.216 | NO READING |
| 32 | 49152 | 0.998841 | 0.001873 | [0.997535, 0.999930] | 0.984251 | 0.074 | consistent with 1 hop |

No cell's CI lies below its floor. `ĥ` never exceeds `0.389`, so the softmax arm
is not merely capped at one hop — it is not reaching one. The theorem-grade
verdict "a cell below floor_1 with CI is PROVEN multi-hop" has never had a
candidate to adjudicate, and the CEQ arms have never been run here at all.

**Instrument limit.** `ĥ` is defined by inverting the floor and goes negative
wherever `NRMSE > 1` (`−2.108` at `t*=8, n=2048`; `−10.594` at `t*=32,
n=2048`). A NO READING cell has no meaningful hop count, and the column must be
read as undefined there rather than as a negative hop budget.

## B8 result — the parity test has no passing branch at N=8

TOST was appended to `scale/it11_verdict.py`, which already carried the two
refusals this round needed (`by_seed` refuses below N=8, and refuses to pool
thread counts). A third refusal joins them: `refuse_difference_test_parity`
raises on any attempt to read a failed difference test as equality, and
`delta_eq` refuses a margin below twice the measured reduction-order floor
`2.345e-3`, since such a margin would certify equivalence over a window that
changing `--threads` moves a single cell across.

The machinery's first act was to fail its own demo, correctly. At
`Δ_eq = 0.5σ` the 90% CI half-width is `t₍.₉₅,2N−2₎·√(2/N)` in units of `σ`:

| N | half-width / σ | CI fits margin | TOST power at true diff 0 |
|---|---|---|---|
| 8 | 0.8807 | no | 0.000 |
| 16 | 0.6001 | no | 0.000 |
| 23 | 0.495473 | **first fit** | 0.0668 |
| 24 | 0.4846 | yes | 0.0834 |
| 40 | 0.3722 | yes | 0.431 |
| 70 | — | yes | **first ≥ 0.80** |

Two bit-identical arms return NO VERDICT at N=8. Parity needs **N=70 per arm**,
`8.8×` the registered count; priced on the measured cost curve that turns THE
READING's three points from about `8.9 h` per arm into about `78 h` per arm.

## The first arm contrast on the chain corpus

`pivot_unsigned`, `t*=2, n=2048, steps=150, threads=6`, N=8 distinct seeds, all
eight LEARNS — `0.960719, 0.950924, 0.971501, 0.957839, 0.957380, 0.929644,`
`0.970670, 0.953524`. Against the thread-matched softmax cell:

| arm | N | mean | sd |
|---|---|---|---|
| softmax | 8 | 0.952349 | 0.010101 |
| pivot_unsigned | 8 | 0.956525 | 0.013133 |

Contrast `+0.004176`, 90% CI `[−0.006189, +0.014542]`, `Δ_eq = 0.005051`,
power `0.000` → **NO VERDICT**. The arms are neither shown equal nor shown
different, and by the table above no N=8 design could have shown either. This is
the first CEQ arm ever run on this corpus; the result is that the registered
design cannot read it.

## The ladder's two halves are not equally reachable

C-PAR and C-CAP were budgeted as if they cost the same. They do not, and the
difference is structural rather than a matter of effort.

C-PAR asks whether two arms agree. That needs a two-sample equivalence margin,
and M-13 shows the registered margin has no passing branch until N=23 and no
adequate power until N=70.

C-CAP asks whether one arm's reading lies below `√((t*−h)/t*)`, a constant
closed-form from the task. One sample, no margin, no second arm. It is decidable
at the N this round actually runs, and it is the half of the ladder that
distinguishes the architecture rather than matching it. `cap_verdict` in
`scale/it11_verdict.py` implements it, refuses below N=8 like its neighbours,
reports `proven_hops` as the largest `h` whose floor the whole interval clears,
and returns **no** hop count above the bar rather than the negative one that
inverting the floor would produce.

The improvement over softmax that a crossing requires, from the thread-matched
N=8 cells:

| `t*` | `n` | softmax mean | sd | floor_1 | needed mean | gap |
|---|---|---|---|---|---|---|
| 2 | 2048 | 0.952349 | 0.010101 | 0.707107 | 0.700107 | **26.5%** |
| 8 | 32768 | 0.975371 | 0.002360 | 0.935414 | 0.933779 | **4.3%** |
| 32 | 49152 | 0.998841 | 0.001873 | 0.984251 | 0.982953 | **1.6%** |

The `t*=8, n=32768` cell is the most tractable crossing and is the one being run.

**What the corpus says so far.** `ĥ` at `t*=2, n=2048` reads `0.186` for softmax
and `0.170` for `pivot_unsigned` — the CEQ arm shows no multi-hop signature
there, and neither arm is reaching even one hop.

## C-CAP census — complete, and empty

`cap_verdict` over every cell in `results/` carrying N=8 distinct seeds at a
fixed thread count. Eight such cells exist.

| arm | `t*` | `n` | thr | mean | CI | floor_1 | `ĥ` | verdict |
|---|---|---|---|---|---|---|---|---|
| pivot_unsigned | 2 | 2048 | 6 | 0.956525 | [0.947501, 0.964422] | 0.707107 | 0.170 | consistent with 1 hop |
| softmax | 2 | 2048 | 6 | 0.952349 | [0.946161, 0.959054] | 0.707107 | 0.186 | consistent with 1 hop |
| softmax | 8 | 2048 | 6 | 1.124057 | [1.116245, 1.131436] | 0.935414 | n/a | NO READING |
| softmax | 8 | 16384 | 12 | 0.983154 | [0.979991, 0.986083] | 0.935414 | 0.267 | consistent with 1 hop |
| softmax | 8 | 32768 | 12 | 0.975371 | [0.973865, 0.976869] | 0.935414 | 0.389 | consistent with 1 hop |
| softmax | 32 | 2048 | 6 | 1.153720 | [1.130700, 1.183720] | 0.984251 | n/a | NO READING |
| softmax | 32 | 32768 | 12 | 1.003371 | [1.000851, 1.005679] | 0.984251 | n/a | NO READING |
| softmax | 32 | 49152 | 12 | 0.998841 | [0.997535, 0.999930] | 0.984251 | 0.074 | consistent with 1 hop |

Zero crossings. `ĥ` peaks at `0.389`. The `t*=8, n=32768` `pivot_unsigned` cell
now running is the ninth, and the first with a plausible gap to close (4.3%).

## The hop-2 term is inert — structural read plus a filed prediction

`scale/m3_capability.py:119-165`, read rather than inferred. `Arm._operator`
returns `bench._softmax_operator(q, k)` for **both** `softmax` and
`pivot_unsigned`; they share the operator exactly. `Arm.forward` computes
`z = x + a @ x` for every arm, then for every arm except `softmax` adds
`hop2 = batched_pivot_hop2(a, batched_select_pivots(k, self.k_pivots))` and
`z = z + hop2 @ x`. So `pivot_unsigned` is `softmax` plus a pivot-routed second
hop at `K_PIVOTS = 8` against `s = 64`.

That makes its achievable floor `floor_2`, not `floor_1`:

| `t*` | floor_1 | floor_2 |
|---|---|---|
| 2 | 0.707107 | **0.000000** |
| 8 | 0.935414 | 0.866025 |
| 32 | 0.984251 | 0.968246 |

At `t*=2` a working 2-hop arm can read the label **exactly**. The measured N=8
cell reads `0.956525` — `0.956525` above its own floor, and `0.004176` *worse*
than the 1-hop softmax it is built from. An added term that leaves the reading
unchanged to within a seed sd, in the one place where a working version drives
the error to zero, is inert rather than underpowered.

**Direct probe, and a correction to the word "inert".** Instantiating both arms
on `e3_t8` at `n=256` and measuring the terms without training:
`||a @ x||` is `39.7479` for **both** arms, identical, confirming the shared
operator by value rather than by reading. `||hop2 @ x||` is `7.0393`, a ratio of
`0.1771` to hop 1, and `38.54%` of the hop-2 matrix entries are non-zero. The
term is therefore active and substantial, not a dead path.

That makes the failure sharper, not weaker: a second hop carrying roughly a
fifth of hop 1's magnitude, with live pivot routing, still moves the reading by
less than one seed sd. The routed hop carries signal that does not help the
label. "Inert" was the wrong word and is withdrawn; the accurate statement is
*active and ineffective*. Limit: the probe is at initialization, and since the
operator holds no parameters of its own, training can influence hop 2 only
through `wq`/`wk`.

`V13_PREDICTION_HOP2.md` was filed while the `t*=8, n=32768` cell was running
and its journal held zero cell rows. It is left **unedited** — amending a
pre-registration after seeing more evidence would void it, so this probe is
recorded here instead: the reading will land in `[0.960, 0.990]`,
will not clear `floor_1`, will sit at least `0.09` above `floor_2 = 0.866025`,
and `ĥ` will stay below `1.0`. If those hold, the empty C-CAP census is a
mechanism failure and no increase in `n` fixes it.

## Arm inventory, measured

At the harness geometry, untrained, `n=32`:

| arm | operator source | `‖a‖` | min entry | negative frac | hops |
|---|---|---|---|---|---|
| softmax | `_softmax_operator` | 12.3028 | +0.000000 | 0.0000 | 1 |
| pivot_unsigned | `_softmax_operator` | 12.3028 | +0.000000 | 0.0000 | 2 |
| pivot_signed | `_causal_sgate_operator(w=0)` | 15.1003 | +0.000000 | 0.0000 | 2 |
| windowed_signed | `_causal_sgate_operator(w=8)` | 21.5133 | +0.000000 | 0.0000 | 2 banded |

`max|softmax − pivot_unsigned| = 0.000e+00`. The pairwise distances among the
other combinations are `2.273e-01` and `1.832e-01`, so `pivot_signed` is a
genuinely different operator from `pivot_unsigned` — the round's withdrawn G4
finding is about *signedness* being absent, not about the two being numerically
the same, and this measurement is consistent with it rather than correcting it.

Two consequences for THE READING. Four arms run over **three** distinct
operators. And no arm has a single negative entry, so the round has **zero
signed operators** despite two arms carrying "signed" in their names — the
capability under test between `softmax` and `pivot_unsigned` is not the operator
at all, it is the hop-2 term alone.

## Why hop 2 does not help — measured, not inferred

`scale/pivot_probe.py:94-100`. `pivot_hop2(a, P) = a[:, P] @ a[P, :]`, with the
docstring stating "Rank ≤ |P| by construction, which IS the mechanism".
Measured on `e3_t8` at `s=64`, `K_PIVOTS=8`, untrained, `n=16`:

| quantity | median / mean |
|---|---|
| `rank(a)` | 63 |
| `rank(a @ a)` — the full second hop | **62** |
| `rank(pivot_hop2)` — the routed second hop | **8** |
| `cos(hop2, a@a)` | 0.7398 (min 0.5481, max 0.8385) |
| `‖hop2‖ / ‖a@a‖` | **0.1932** mean-of-ratios (median `0.1451`, aggregate Frobenius `0.2350`) |

The routed hop is a rank-8 approximation to a rank-62 object. It keeps 74% of
the direction and about 19% of the magnitude, discarding 54 of 62 dimensions.
The `0.1932` is a mean of per-example ratios reported beside a column of
medians; the median ratio is `0.1451` and the aggregate Frobenius ratio is
`0.2350`. The three agree on the order of magnitude and the conclusion, but a
mean and a median in one table should carry their labels. Since
`pivot_unsigned` shares softmax's operator bit-identically, this term is the
whole architectural difference between the two arms.

**The scope this licenses.** `scale/pivot_probe.py:83-86` says the selector —
top-k by key-norm — is "deliberately crude", chosen so that "a cleverer selector
is a confound at this stage", with selection deferred to S2/C4. So a negative
C-CAP result here supports *"routing at K=8 through a key-norm selector does not
deliver the second hop on this task"*. It does **not** support "multi-hop
routing does not work", and the round must not let the first sentence be written
as the second.

## The K sweep — isolating the bottleneck from the hypothesis

The rank measurement says the routed hop is rank `|P|` where the full hop is
rank 62. That yields a controlled test the deciding cell cannot give: if the
rank bottleneck is *why* the arm shows no second hop, raising `K` must recover
it. `Arm.__init__` already accepts `k_pivots`, so no patch is needed.

`scripts/v13_kpivot_sweep.py` trains `pivot_unsigned` at `K ∈ {8, 16, 32, 64}`
against a `softmax` control, at `t*=2, n=2048, 150 steps`, seeds 0-2. `t*=2` is
chosen because `floor_2 = 0.000000` there — a working second hop can read the
label exactly, so any recovery has the widest possible room to show.

**Harness control — WITHDRAWN, it was invalid.** This section previously
reported the standalone script's softmax control as `0.951348` (sd `0.014927`)
against the published `0.952349` (sd `0.010101`), a gap of `0.001001`, and used
that to license reading the K comparison. Three things are wrong with it.

The numbers are stale. They came from the pre-crash run at `threads=4`; the
restarted run overwrote `results/r10_v13_kpivot_t2.txt` at `threads=8`, where the
control reads **`0.950252`** sd **`0.019256`**. The figures quoted were carried
forward from a superseded run and no longer existed on disk.

The control crossed thread lanes. Its journal header records `threads=8` while
the published N=8 softmax cell it was compared against is `threads=6` — the exact
pooling `it11_verdict.by_seed` refuses and that M-10 was filed to prevent. Its
seed-0 value `0.971432` is the *threads=8* reading, which is why it looked close.

And it could not have worked even if run correctly. The true gap is `0.002097`,
which is `0.894` of the reduction-order floor `0.002345`. A control whose signal
is smaller than the thread floor cannot separate "harness differs" from "thread
count differs", so it never had the resolution to license anything.

The K-sweep result itself is unaffected — every one of its points was taken in a
single lane at `threads=8`, and internal comparison across `K` is what the sweep
claims. What is withdrawn is the cross-check against the round's own journals.

The two outcomes separate cleanly:

- **Reading improves with `K`.** The rank-8 bottleneck is the cause, the routing
  hypothesis survives, and the fix is a larger `K` or a better selector. The
  round's negative C-CAP result would then be a statement about `K=8`, not about
  routing.
- **Reading is flat in `K`.** The bottleneck is not the cause. At `K=64` the
  routed term is full-rank and `pivot_hop2` reduces to the full `a@a`, so a flat
  curve means the second hop does not help *even when nothing is discarded* —
  which moves the failure to the term's placement or to the task not rewarding a
  second hop at all.

Either way the result is attributable, which the deciding cell alone would not
have been.

**Interim reading, 3 seeds per point, threads=8.** `softmax` control
`0.950252` (sd `0.019256`); `K=8` `0.960945` (sd `0.010289`), `+0.010692`;
`K=16` `0.968518` (sd `0.008712`), `+0.018266`. The curve runs the wrong way for
the bottleneck hypothesis: more pivots is monotonically worse so far, where a
rank-limited term should recover as `K` grows toward the full rank 62.

`K=32` reads `0.965750` (sd `0.011859`), `+0.015497` — better than `K=16`, so
the curve is **not** monotonic and the previous sentence describing it as such
was wrong. At 3 seeds with sd near `0.010`, the spread across `K` sits inside
noise.

**Complete, and the answer inverts the hypothesis.** Four points, 3 seeds each,
`threads=8`, `t*=2, n=2048, steps=150`:

| arm | mean | sd | vs softmax | verdict |
|---|---|---|---|---|
| softmax (1 hop) | 0.950252 | 0.019256 | — | LEARNS |
| `K=8` | 0.960945 | 0.010289 | +0.010692 | LEARNS |
| `K=16` | 0.968518 | 0.008712 | +0.018266 | LEARNS |
| `K=32` | 0.965750 | 0.011859 | +0.015497 | LEARNS |
| `K=64` | **1.000329** | 0.027601 | **+0.050076** | **NO READING** |

At `K=64` the pivot set is every position, so `a[:, P] @ a[P, :]` **is** `a @ a`
— the full second hop with nothing discarded. The arm then fails to clear
predict-the-mean.

The rank-8 routing was therefore never a bottleneck starving a working second
hop. It was limiting the damage that second hop does: restricting it to 8 of 64
columns keeps the arm learnable, and removing the restriction destroys the
reading. `pivot_probe.py:98` calls rank `≤ |P|` "the mechanism"; measured, it
behaves as damage control.

This kills two readings written earlier in this file. The rank-8 bottleneck does
not explain the missing hop, and "flat in K" understates it — the curve is flat
across `K = 8, 16, 32` within a seed sd of about `0.010` and then falls off a
cliff at full rank. The surviving explanation is that `a @ a` on this task
carries a component the readout cannot use and whose magnitude swamps hop 1:
`‖hop2 @ x‖ / ‖a@x‖` was measured at `0.1771` for `K=8`, and the full term is
about five times that.

**What it licenses.** A negative capability result at `K=8` is not "multi-hop
routing does not work". It is "on this task, the second hop as constructed is
harmful, and the routing that was proposed as the mechanism is what keeps the arm
usable at all". Repairing it means changing what hop 2 computes — its scale, its
placement relative to the readout, or the task's reward for depth — not widening
`K`.

## X₂₈ placement (v-main.8b)

X₂₈b's derivations and must-fires are parallel-safe and dispatched. X₂₈a
(stock-flow gate, `ṗ = pQ` against counted occupancies) and X₂₈c (the
consistency triangle) ride BED-1's build and stay blocked under D-4 — BED-1
itself is blocked on Round 11.

**A circularity flag raised before the triangle is built.** X₂₈c proposes
measuring `κ`, `λ` and `d` independently and checking `κ = λ(1 − d)`.
`V13_X27_G1_PRIOR_ART.md` already records Tél's `D₁ = 1 − κ/λ`, which is the
same relation rearranged. If two of the three quantities can only be obtained
through that relation, the triangle is an algebraic identity checked against
itself and certifies nothing. Both dispatched agents were told to answer this
before any number ships from it. The contract's worked figure
`0.69 × (1 − 0.8) = 0.138` follows from the formula alone, so it tests the
arithmetic and not the physics.

## The gain sweep — magnitude or content

`K=64` collapsing to NO READING leaves exactly two live explanations, and a
scalar gain on the hop-2 term separates them. `scripts/v13_hop2_gain.py` sweeps
`γ ∈ {0, 0.05, 0.1, 0.25, 0.5, 1.0}` on `z = x + a@x + γ·(hop2 @ x)` at `K=64`,
`t*=2, n=2048`, seeds 0-2.

Readings fixed before the numbers land:

- **MAGNITUDE.** The term is directionally useful but too large, swamping hop 1.
  The curve then has an interior optimum — some `γ` beats `γ = 0` — and the fix
  is a scale, which is one constant.
- **CONTENT.** The term carries a component the readout cannot use at any scale.
  The curve is then monotone in `γ` with its best point at `γ = 0`, and no
  scaling repairs it; the construction has to change.

**Independent magnitude measurement, filed before the sweep reported.** At
`t*=2`, untrained, `n=16`: `‖x‖ = 35.2102`, `‖a@x‖ = 9.9380`, and
`‖a@a@x‖ = 10.1613`, so the full second hop is `1.0225` times hop 1 — the same
size, and both small against `x`. The ratio scales with `K` as
`0.2482 / 0.4161 / 0.6134 / 1.0225` for `K = 8 / 16 / 32 / 64`, with median rank
`8 / 16 / 30 / 62`. The quantity the readout actually absorbs is `‖z‖`, and
adding hop 2 moves it by `+1.30%` at `K=8` and `+9.45%` at `K=64`.

A `9.45%` perturbation of `z` is what separates LEARNS at `0.950252` from NO
READING at `1.000329`. That is far too small a scale change to be a swamping
effect, so the MAGNITUDE branch is already unlikely on evidence independent of
the gain sweep, and the CONTENT branch is the standing prediction. The gain
sweep remains the test; this is the prior it will be read against, recorded
before it reported so it cannot be fitted afterwards.

`γ = 0` is also an identity check rather than just a control. With
`Arm._operator` returning `bench._softmax_operator` for both kinds,
`z = x + a@x` at `γ = 0` **is** softmax's forward, so `γ = 0` must reproduce the
softmax arm. If it does not, `GainArm` differs from the shipped arm somewhere
other than the gain and no other row in the sweep can be trusted.

## A mis-specified probe, recorded as such

A linear-readout probe was run to ask which hop carries label information, by
ridge-fitting `y` on each feature and reading held-out NRMSE. Every feature came
back above the bar: raw `x` `1.130948`, `a@x` `1.113370`, `x + a@x` `1.132502`,
`a@a@x` `1.034558`, `hop2_K8@x` `1.030686`, `x + a@x + a@a@x` `1.132398`.

The probe is worthless and the reason is a design error, not a property of the
arms. The operator `a` is built from `wq` and `wk`, which are **learned**; at
initialisation it is a random causal attention carrying no task structure, so
the probe measured a random matrix. The information plainly exists — the trained
arm reads `0.950252` at this cell — and training is exactly the step the probe
omitted.

A correct version trains each arm first and probes its converged features. That
is a strictly larger experiment than the gain sweep already running, which
answers the same question by intervention rather than by observation, so it is
not worth running now. Recorded so the numbers above are never mistaken for a
finding about the hops.

## X₂₉ — the interference layer, and why the diagnosis was accepted

v-main.8c reads the `K=64` collapse as common-mode swamping: `a @ a` decomposes
into a rank-1 carrier `𝟙πᵀ` at unit gain plus a differential remainder at gain
`λ₂²`, and `K=8` was an accidental common-mode filter. That was checked before
being built on.

Measured at `t*=2`, untrained, `n=16`, decomposing each operator into a
common mode (every row equal to the column-mean) plus a remainder:

| operator | common-mode energy fraction |
|---|---|
| `a @ a` (full second hop) | **0.7550** |
| `pivot_hop2` at `K=64` | 0.7550 — it *is* `a@a` |
| `pivot_hop2` at `K=8` | **0.4899** |

`a` is strictly lower triangular and sub-stochastic — row sums run `0.000000` to
`1.000000` with row 0 identically zero — so `a@a` is Perron-dominated by the
rank-1 mode, and the routing that keeps 8 of 64 columns keeps roughly half the
common-mode energy the full product carries. "Damage control" now has a
mechanism and a number behind it.

**Why hop 1 survives and hop 2 does not.** Common-mode dominance compounds with
depth. Measured on the same operator, `a^h` for `h = 1..6`:

| `h` | `‖a^h‖` | common-mode fraction |
|---|---|---|
| 1 | 8.6992 | **0.4006** |
| 2 | 9.0210 | **0.7550** |
| 3 | 9.4650 | **0.8944** |
| 4 | 8.4114 | 0.8866 |
| 5 | 6.0958 | 0.8190 |
| 6 | 3.6117 | 0.7363 |

Hop 1 is the least common-mode-dominated term the architecture has, and hop 2 is
nearly twice as dominated. The fall after `h=3` is the strictly-causal operator
running toward nilpotency — `a` is strictly lower triangular so `a^s = 0` — not a
recovery.

**The structure is task-invariant.** Measured across all three chain tasks at
the same geometry, untrained:

| task | `cm(a)` | `cm(a@a)` | `cm(K=8)` | `‖hop2‖/‖a·x‖` |
|---|---|---|---|---|
| e3_t2 | 0.4006 | 0.7550 | 0.4899 | 0.2482 |
| e3_t8 | 0.4006 | 0.7549 | 0.4837 | 0.2436 |
| e3_t32 | 0.4003 | 0.7550 | 0.4675 | 0.1565 |

`cm(a)` and `cm(a@a)` agree to four figures across tasks. The common-mode
structure is a property of causal softmax at `s=64` — the mask and the
normalisation — and not of what the data encodes. Two things follow: the
deciding cell at `t*=8` sits in the same regime the diagnosis was measured in,
so the finding was not extrapolated across tasks; and a deflation is one fix for
every cell of the reading rather than a per-task adjustment. The routed term's
relative size does fall with `t*` (`0.2482 → 0.2436 → 0.1565`), so hop 2
contributes least where the label needs it most.

This is the architectural consequence, and it is larger than one arm: **the
construction cannot go deeper than one hop without a common-mode fix, and the
contamination worsens with each hop.** Every multi-hop claim in the round, and
therefore the whole capability half of the north-star sentence, depends on a
deflation of this kind working. X₂₉a is not an optimisation; it is a
precondition.

**Placement.** X₂₉a rides the current cell's follow-up and is dispatched with
its prediction fixed in advance: deflated `K=64` recovers to at most `0.960945`,
`K=8`'s reading. X₂₉b and X₂₉c enter BED-1's window behind their fetches, and
BED-1 is blocked on Round 11 under D-4.

**Three cautions carried into the dispatches that the delta does not state.**
The deflation projector must be checked for causality — a naive row-mean
subtracted from a strictly-causal operator can introduce entries at or above the
diagonal, and a deflation that leaks future information invalidates everything
downstream. The Wiener treatment assumes an eigenbasis, but `a` is defective and
non-normal, so the mode decomposition has to be justified rather than assumed.
And "Walsh crosstalk is exactly 0" holds for synchronous, equal-power,
multipath-free channels; whether a residual stream is such a channel is a
question about this architecture, not about CDMA, and the fetch was told to
answer it.

## Naive deflation breaks causality — measured, and the fix

The deflation X₂₉a proposes subtracts `𝟙πᵀ` from `a`. Since `a` is strictly
lower triangular, row 0 is identically zero, so row 0 of `a − 𝟙πᵀ` is `−π`,
which is nonzero on and above the diagonal. Measured at `t*=8`, `n=8`, summing
absolute mass on and above the diagonal:

| operator | mass on/above diag | max entry there | common-mode fraction |
|---|---|---|---|
| `a` (shipped) | 0.000000e+00 | 0.000000e+00 | 0.4006 |
| `𝟙πᵀ` | 1.299356e+02 | 7.407572e-02 | — |
| `a − 𝟙πᵀ` | **1.299356e+02** | **7.407572e-02** | 0.0000 |
| `tril(a − 𝟙πᵀ, −1)` | **0.000000e+00** | 0.000000e+00 | 0.0296 |

The naive deflated operator attends to the present token and to future ones. Any
reading taken through it is invalid regardless of how good the number looks,
because the arm would be permitted to see the answer.

Re-masking after deflation restores strict causality and keeps almost all of the
benefit: the common-mode fraction falls from `0.4006` to `0.0296` at hop 1, and
from `0.7550` to `0.0075` for the second hop built from the re-masked operator.
Perfect removal (`0.0000`) is available only in the non-causal form, so the
causal price of deflation is a residual common-mode fraction of about `3%` at
hop 1 and under `1%` at hop 2.

**Correction, from X₂₉a's must-fire.** The `tril` form preserves causality but
does **not** remove the common mode: its measured DC gain is `0.358921`, i.e.
about `9 dB` of rejection, and it **fails** the planted common-mode must-fire.
The `cm = 0.0296` reported above is a Frobenius *energy fraction*, which is not
the same quantity as the *gain* a DC input sees through the operator, and the
must-fire measures the second. A renormalised projector
`P = diag(c)·𝟙π̂ᵀ` passes both must-fires — common-mode gain `7.944525e-08`,
differential gain `1.000000` — where `c = a𝟙` and `π̂` is the normalised mean
row. Measured CMRR for the hop-2 term: `157.2 dB` renorm against `12.4 dB` tril
at `K=64`, the renorm figure being a float32 noise floor rather than a physical
limit.

**Correction, on "Perron".** `a` is strictly lower triangular, hence nilpotent:
`σ(a) = {0}`, `a^s = 0`. There is no eigenvalue 1, no stationary left vector and
no spectral projection, so `𝟙πᵀ` is not a Perron projection and the identity
`A_d² = A² − 𝟙πᵀ` does not hold here. The phrase "Perron-dominated" used earlier
in this file, and "the Perron projection" in the X₂₉a dispatch, are both wrong.
What survives is the action on the constant direction — `a𝟙 = c` with `c₀ = 0`
and `c_i = 1` otherwise — so the object to remove is the least-squares best
constant-row approximation to `a`, an oblique common-mode projector, not a
spectral one.

The standing causality guard was the right ask and landed: `assert_causal` runs
inside `DeflatedArm.forward` on `a`, on `a_d` and on the hop-2 term, on every
forward of every step, rather than as a one-time inspection.

## X₂₈c fetch — three consequences the contract does not carry

`V13_X28C_PRIOR_ART.md`, 720 lines, `[V]` 6 · `[V-t]` 6 · `[U]` 6.

**1. The wrong dimension, with a cheap fix.** The relation is stated everywhere
with the information dimension `D₁`, and `D₁ ≤ D₀` always, so box counting
under-predicts `κ` one-sidedly — `0.49%` at slope ratio 1.5, `4.83%` at 3,
`9.56%` at 5, `24.5%` at 500. X₂₇a measures box counting. The repair does not
need a new instrument: weight X₂₇a's existing boxes by visitation frequency and
take the entropy slope rather than the count.

**2. The worked figure is blind to the finding.** The contract pre-registers
`λ = 0.69, d = 0.8 ⇒ κ = 0.138`. But `λ ≈ ln 2` is the constant-slope case, and
constant `|f′|` is exactly where `D₀ = D₁` and the box-counting error vanishes.
The pre-registered configuration is the single one in which the defect above
cannot show. A must-fire that cannot fail on the known failure mode is not a
must-fire, and this one needs a second plant at non-uniform slope.

**3. A hyperbolicity gate is mandatory.** In non-hyperbolic systems `κ → 0` and
`d → 1`, and `κ = λ(1 − d)` is satisfied vacuously. A GREEN triangle on a
non-hyperbolic bed certifies nothing, so hyperbolicity has to be established
before the triangle is read, not after.

**Provenance limit.** Kantz & Grassberger 1985 is `[V-t]` only — closed at
OpenAlex, Semantic Scholar and ScienceDirect. The relation is closed at equation
level by three concordant restatements plus an independent rigorous
reconstruction, but not at primary-source level, and the file says so rather
than implying the original was read.

## Does deflation survive depth — measured

The raw operator's common-mode contamination compounds with hop depth. The
re-masked deflation `ad = tril(a − 𝟙πᵀ, −1)` was composed to depth 6 on `e3_t8`
and checked for leakage at every power:

| `h` | `cm(a^h)` | `cm(ad^h)` | `‖a^h‖` | `‖ad^h‖` | mass on/above diag |
|---|---|---|---|---|---|
| 1 | 0.4006 | **0.0296** | 8.6996 | 6.4182 | 0.00e+00 |
| 2 | 0.7549 | **0.0075** | 9.0207 | 4.3053 | 0.00e+00 |
| 3 | 0.8944 | **0.0049** | 9.4646 | 2.5667 | 0.00e+00 |
| 4 | 0.8865 | 0.0185 | 8.4112 | 1.2116 | 0.00e+00 |
| 5 | 0.8189 | 0.0834 | 6.0961 | 0.4525 | 0.00e+00 |
| 6 | 0.7363 | 0.2716 | 3.6122 | 0.1408 | 0.00e+00 |

Two readings, and the second matters more than the first.

**The fix composes.** Causality is exact at every depth — the re-masked
projector does not reintroduce leakage under repeated multiplication, which was
not obvious and had to be checked rather than argued. Contamination stays under
`2%` through hop 4 against `89%` raw.

**But deflation does not create depth, it reveals its absence.** `‖ad^h‖` decays
`6.42 → 4.31 → 2.57 → 1.21 → 0.45 → 0.14` while `‖a^h‖` holds near 9 through
`h=3`. The raw operator only appears to carry signal at depth because the
rank-1 carrier sustains its norm; the differential content decays from the start.

**Correction: the decay is not `λ₂^h`, and there is no `λ₂`.** `a` is strictly
lower triangular, therefore nilpotent — measured `max|λ| = 0.000e+00`, spectrum
exactly `{0}`, and `a^s = 0` at `s = 64`. No bound of the form `ρ^h` applies.
The spectral norm does not help either: `s₁(ad) = 1.2459 > 1`, so `s₁^h` predicts
**growth** while the operator decays:

| `h` | `‖ad^h‖_F` measured | `s₁^h` bound | ratio |
|---|---|---|---|
| 1 | 1.6044 | 1.2459 | 1.2877 |
| 2 | 1.0755 | 1.5524 | 0.6928 |
| 3 | 0.6405 | 1.9341 | 0.3312 |
| 4 | 0.3020 | 2.4098 | 0.1253 |
| 5 | 0.1127 | 3.0025 | 0.0375 |
| 6 | 0.0351 | 3.7409 | 0.0094 |

The bound is not loose, it points the wrong direction. What is being observed is
non-normal transient decay, which no single scalar rate summarises.

**Do not conflate this with the round's dose theorem.** `workdonenew.md` records
`err(t) ≤ λ₂^t` as false as written and true in the degree-weighted 2-norm — but
that theorem is about the **corpus graph operator**, a genuine stochastic matrix
with a real `λ₂`. The attention operator measured here is a different object:
nilpotent, non-normal, spectrum `{0}`. Carrying the `λ₂` language across from one
to the other would attach a proved bound to an operator it does not govern. By hop 6 the deflated operator retains about `4%` of the
raw operator's magnitude.

So the rising `cm(ad^h)` at `h = 5, 6` is not the deflation failing — it is the
common mode of a signal that has nearly vanished. The honest architectural
statement is that after the carrier is removed, hop depth is limited by
**signal strength**, not by contamination, and any multi-hop claim has to say
which of the two it is defeating. X₂₉a will show whether removing the carrier is
enough to make hop 2 useful; this table already says it will not make hop 5
useful.

## B9 throughput — withdrawn and rerunning

The audit marked the throughput figures UNVERIFIABLE and it is right. The
numbers reported (`54,837 / 20,194 / 15,090` tok/s at `L=16`) came from the
`--steps 3` validation run, not from the 500-step run B9 specifies. The
500-step run was launched into the background and killed when the machine was
serialised for the anchor cell, so `results/r10_v13_b9_4060_probe.txt` exists at
**0 bytes** and no artifact of either run survives. A re-run on a loaded host
returned `32,433 / 13,401 / 9,981`, which is what a throughput number measured
under different load looks like and is why B9 asks for a controlled 500 steps.

Two things stand and one does not. The **memory** figures were computed from
`ceq/sizing.py`'s calibrated constants and from allocator peaks, and the audit
confirmed them. The **ratios** — signed against softmax — are internally
consistent within either run (`2.72×` at 3 steps, `2.42×` at the re-run). The
**absolute** tok/s figures are withdrawn until a 500-step run completes on a
quiet host with its output actually on disk.

A separate inconsistency in the same script, also from the audit:
`scripts/v13_b9_4060_probe.py` prints `sizing.activation_bytes` predictions in
`bf16_autocast` while running the model in fp32, so a reader reproducing M-12
from its output sees `153.7`/`641.6` where M-12 records `204.8`/`1160.0`. M-12
is correct at fp32; the script's prediction column is the part that disagrees
with its own measurement.

## The hop-2 prediction will be adjudicated by script, not by reading

`scripts/v13_adjudicate_hop2.py` reads the four clauses of
`V13_PREDICTION_HOP2.md` off the journal and prints HOLDS or FAILS for each. It
was written while the cell stood at 4 of 8 seeds, so its logic was fixed before
the outcome existed.

It refuses below N=8 rather than reporting on the seeds available — the same
refusal `verdict` and `cap_verdict` carry, for the same reason: the prediction is
about an eight-seed cell, and a verdict over four answers a question nobody
registered. Run now it exits 2 and prints the seed count.

The point is not convenience. A prediction filed by the same agent that later
reports on it invites reading the result generously, and the failure mode is
silent — nobody writes down the reading they almost gave. Fixing the arithmetic
in code beforehand removes the discretion. Its closing line says what to do if
the prediction is wrong: record which clause failed, and do not restate the
prediction to match the outcome.

## REVISION — common mode does not explain the K dependence

X₂₉b made an observation none of the earlier measurements did. `Arm.forward`
ends `readout(mlp(z)).squeeze(-1)[:, s-1]` — it returns **only the final
sequence position**, and the MLP is position-wise, so only row `s-1` of every
operator reaches the loss. All the common-mode fractions recorded above are
full-matrix Frobenius quantities averaged over 63 rows that are computed and
discarded.

Restricting to the row that reaches the readout reverses the conclusion:

| operator | full Frobenius | row `s-1` only |
|---|---|---|
| `a` (hop 1) | 0.4006 | 0.5183 |
| `a@a` (`K=64`) | 0.7549 | **0.8343** |
| `pivot_hop2` at `K=8` | 0.4837 | **0.8646** |

On the full matrix `K=8` looks like a common-mode filter. On the operative row it
is not — it is marginally **worse** than `K=64`. So the statement "the pivot
routing was an accidental common-mode filter" is **withdrawn**: it is true of the
whole matrix and false of the part the architecture uses, and the K-dependence it
was invoked to explain is not explained by it.

**What does scale with `K` is magnitude at that row.** Measured over 64 examples,
`‖x[s-1]‖ = 8.4855` and `‖hop1[s-1]‖ = 1.4044`:

| `K` | `‖hop2[s-1]‖` | ratio to hop 1 | ratio to `‖z‖` |
|---|---|---|---|
| 8 | 0.1869 | 0.1331 | **0.0215** |
| 16 | 0.3791 | 0.2699 | 0.0437 |
| 32 | 0.6988 | 0.4976 | 0.0805 |
| 64 | 1.4768 | 1.0516 | **0.1701** |

The surviving mechanism is the simplest one available and it needs no spectral
story: **the second hop carries no usable signal at the readout row, and the
damage scales with how much of it is added.** `K` is one dial on that quantity
and the gain `γ` is another, which is why the gain sweep already reads
`+0.008261` at `γ = 0.05`. X₂₉b reached the same place independently from the
spectrum: its Wiener gains fall below the zero-correlation null band for **all
64 modes**, not for the common mode alone, so the MMSE-optimal treatment of this
term is to delete it.

**Two claims were conflated, and only one of them is withdrawn.** Re-measuring
the depth table at the readout row separates them:

| `h` | `cm_full(a^h)` | `cm_row(a^h)` | `cm_row(ad^h)` | `‖a^h[s-1]x‖` |
|---|---|---|---|---|
| 1 | 0.4006 | 0.5194 | **0.0002** | 0.8826 |
| 2 | 0.7550 | **0.8345** | **0.0002** | 1.2016 |
| 3 | 0.8945 | 0.9408 | 0.2133 | 1.5441 |
| 4 | 0.8865 | 0.9765 | 0.7777 | 1.6575 |
| 5 | 0.8189 | 0.9893 | 0.9131 | 1.4228 |
| 6 | 0.7363 | **0.9945** | 0.9238 | 0.9752 |

*Stands:* `a@a`'s readout row is common-mode dominated at `0.8345`, and the
deflation removes it essentially completely there, to `0.0002`. The full-matrix
figure understated this rather than overstating it.

*Withdrawn:* that `K=8`'s advantage over `K=64` comes from common-mode
filtering. On the readout row `K=8` reads `0.8646` against `K=64`'s `0.8343`, so
whatever separates them, it is not this.

**So the previous paragraph's conclusion was too hasty and is corrected.** X₂₉a's
prediction — deflated `K=64` recovers to at least `K=8`'s reading — does not rest
on the withdrawn claim. It rests on the surviving one, and the surviving one is
strong at hop 2: `0.8345 → 0.0002` at the row the readout sees. The prediction is
live, and its outcome is genuinely unknown rather than expected to fail.

**A new limit, visible only at the readout row.** Raw common-mode dominance rises
*monotonically* with depth there — `0.5194, 0.8345, 0.9408, 0.9765, 0.9893,
0.9945` — with no peak-and-fall, so by hop 6 the operative row is 99.45% carrier.
And the deflation stops holding past hop 2: `cm_row(ad^h)` runs `0.0002, 0.0002,
0.2133, 0.7777, 0.9131, 0.9238`, because powers of the re-masked operator develop
their own constant-row component. Deflation is a hop-2 fix at this geometry, not
a depth fix, and the earlier full-matrix table hid both facts.

## THE COMMON-MODE NUMBERS WERE THE NULL

The X₂₉ prior-art fetch pointed out that `frac(a@a)` is a monotone function of
attention sharpness alone and has no null in this file. Computing it settles the
matter:

| statistic | measured | sharpness-matched null, 20 seeds | z |
|---|---|---|---|
| `cm_full(a@a)` | 0.7549 | **0.7550 ± 0.0004** | **−0.30** |
| `cm_row(a@a)` | 0.8343 | 0.8328 ± 0.0006 | +2.38 |

The null is random Gaussian logits at the measured logit sd (`0.0375`), causal
masked, softmaxed, squared. **The headline `0.7550` is the null to four
decimals.** It says nothing about this operator, this task or this architecture:
it is what causal-masked softmax at `s=64` with near-uniform logits produces by
construction.

This retires the task-invariance result as well. That `cm(a)` and `cm(a@a)`
agreed to four figures across `e3_t2`, `e3_t8` and `e3_t32` was recorded as a
finding — "a property of causal softmax at s=64, not of what the data encodes".
The observation was right and the conclusion stopped one step short: a quantity
identical across every condition **and** equal to its own null is not a property
worth reporting, it is the baseline.

**The deeper error, stated plainly.** Every common-mode measurement in this file
was taken on an **untrained** operator, at logit sd `0.0375`, and was used to
explain performance differences between **trained** arms. Attention sharpness
changes under training, `frac(a@a)` is monotone in sharpness, so the untrained
figure does not describe the operator whose readings were being explained. The
K-sweep and gain-sweep results are unaffected — those are trained readings — but
the mechanism offered for them was measured on the wrong object.

**What survives.** The trained readings stand: `K` sweep `0.950252 / 0.960945 /
0.968518 / 0.965750 / 1.000329`, the gain sweep's identity check at
`+0.000000`, and X₂₉b's independent spectral finding that Wiener gains fall
below the zero-correlation null for all 64 modes. The second hop still carries
no usable signal. What is withdrawn is the common-mode *explanation* for it, in
full — not merely the `K=8`-as-filter half withdrawn earlier.

## Gain sweep, four of six points — nothing separates from zero yet

`t*=2, n=2048, steps=150`, seeds 0-2, `K=64`, `threads=6`:

| `γ` | mean | sd | vs softmax |
|---|---|---|---|
| 0.00 | 0.951602 | 0.015381 | **+0.000000** (identity check) |
| 0.05 | 0.959863 | 0.015795 | +0.008261 |
| 0.10 | 0.958562 | 0.009422 | +0.006960 |
| 0.25 | 0.953986 | 0.006788 | +0.002384 |

Every contrast is inside about one control sd at three seeds, so the correct
statement is that **no gain in `[0, 0.25]` is distinguishable from `γ = 0`**, not
that the curve is rising or falling. The apparent shape — up then back down — is
what three-seed noise looks like at this spread, and reading it as a trend would
repeat the mistake M-15 records.

`γ = 1.0` is the sweep's own consistency check rather than a new datum: it is the
full second hop at `K=64`, which the K sweep measured at `+0.050076`. If the gain
arm's `γ=1.0` does not land near that, the two harnesses disagree and neither
sweep can be read until the disagreement is resolved.

## The untrained measurements do not transfer — measured

Every common-mode figure in this file was taken at initialisation. Training one
arm 150 steps at `t*=8, n=2048` and re-measuring:

| | logit sd | `cm_full(a@a)` | `cm_row(a@a)` |
|---|---|---|---|
| untrained | 0.0367 | 0.7551 | 0.8328 |
| trained | **2.7645** | **0.4245** | **0.3736** |

Logit sharpness rises `75.4×` and the common-mode fraction more than halves. At
the readout row the trained operator carries `0.3736` where the untrained one
carries `0.8328`.

**Consequently the following, all recorded earlier in this file, describe an
object the arms do not use after training and are void as explanations:** the
depth-compounding table, the task-invariance table, the `K=8` versus `K=64`
common-mode comparison at both full-matrix and readout-row resolution, and the
deflation-at-depth table. They remain correct as statements about an untrained
causal-softmax operator at `s=64`, which is a fact about the construction and
not about this architecture.

**What is untouched.** Every trained reading: the K sweep, the gain sweep, the
deciding cell, the arm inventory's operator identity (`max|softmax −
pivot_unsigned| = 0.000e+00`, a structural fact independent of weights), the
0-step null, and X₂₉b's spectral result. The measurements were never the
problem; the mechanism assembled on top of them was, and it was assembled from
the wrong object.

## The one measurement worth redoing trained

Of the voided tables, only one asked a question the round actually needs
answered: whether the trained `K=8` arm differs from the trained `K=64` arm in
what its hop-2 term carries at the readout row. Everything else was descriptive.
That comparison is being rerun on trained operators, at `t*=2, n=2048`, 150
steps, reporting logit sd, `cm_row(hop2)`, the magnitude ratio to hop 1, and the
eval reading together, with the untrained figures printed beside them.

It has three possible outcomes and all three are informative. If the trained
arms differ in `cm_row` in the direction the K sweep's readings differ, the
common-mode mechanism survives — measured correctly this time — and the earlier
error was one of object, not of substance. If they do not differ, the mechanism
is dead on trained operators as well as untrained, and the surviving explanation
is the magnitude ratio alone. If the trained `cm_row` values are both small, the
question dissolves: there is no common mode left to filter after training, and
the entire X₂₉ interference framing applies to an operator that only exists at
initialisation.

The third outcome is the one the survival check makes likely, and it would mean
X₂₉a's deflation is repairing a condition the trained arm does not have. That is
worth knowing before its `SWEEP_TABLE_PLACEHOLDER` is filled, not after.

**Result — the third outcome, and worse than pre-registered.** Trained 150 steps
at `t*=2, n=2048`, measured at the readout row with the untrained figures beside:

| `K` | logit sd | `cm_row(hop2)` | `‖hop2‖/‖h1‖` | eval | verdict |
|---|---|---|---|---|---|
| 8 | 1.1329 | **0.1707** | **0.5121** | 0.960719 | LEARNS |
| 64 | 1.3666 | **0.2419** | **0.6097** | 1.014347 | NO READING |
| | *untrained 0.0256* | *0.8727 / 0.8329* | *0.1438 / 1.1400* | | |

Two explanations die here rather than one.

*Common mode.* After training both arms carry `0.17`–`0.24` at the readout row,
down from `0.87`–`0.83`. There is almost nothing left to filter, so X₂₉a's
deflation and X₂₉b's DC attenuation both address a condition the trained arms do
not have. X₂₉b reached the same place from the spectrum, finding zero modes above
the null band on the trained operator.

*Magnitude.* Untrained, the hop-2-to-hop-1 ratio differs `8×` between the arms
(`0.1438` against `1.1400`) and that was offered as the surviving mechanism.
Trained, the ratios converge to `0.5121` and `0.6097` — a `19%` difference — while
the readings diverge from LEARNS to NO READING. A `19%` difference in a term's
size cannot produce that.

So neither surviving explanation for the K dependence holds on the arms that were
actually measured. The `K=8` eval of `0.960719` reproduces the pilot journal's
seed-0 value exactly, so the harness is consistent; it is the explanations that
are not. What remains is X₂₉b's bilinear diagnosis — the label's two-hop term is
`a·a·b`, which no linear functional of `x` can produce — and that is a claim about
the *shape* of the term rather than its scale or its spectrum, so it is untouched
by every measurement above.

## X₂₉b — the strongest negative result of the round, and a bilinear diagnosis

Both must-fires pass without special-casing. `SNR 10 → g = 0.909090909`, off the
closed form by `4.441e-16`. The DC gain **falls out of the formula** rather than
being asserted: `wiener_gains(c, y)` takes a coordinate matrix and a target, with
no basis vector, no index and no DC test anywhere, and an exactly-DC direction
reads `g = 2.564e-04` against a null mean of `1/(n−2) = 4.888e-04`.

The mode basis is SVD and the justification is structural rather than numerical:
`a@a` is nilpotent, `max|eig| = 0.000e+00`, all 64 eigenvalues zero with
eigenvectors that do not span. An eigendecomposition here carries *no*
information, not merely ill-conditioned information.

**The result: every one of the 64 modes falls inside the zero-correlation null
band** — max `g = 0.004041` against the 99.9% quantile `0.005287` — at `K=64`
and `K=8`, against the exact oracle, against the raw label, with all 16 channels
credited. `‖P_wiener‖_F = 0.008173` against `‖I‖_F = 8`. The MMSE-optimal
treatment of this term is to attenuate it about `979×`, and re-measuring from the
**trained** operator after 150 steps leaves zero modes above the band. This is
the one mechanism claim in the round that survives the survival check.

**The arms.** softmax `0.950252`, plain `K=64` `1.000329` — both reproducing the
commissioned figures to six decimals, which is an independent replication of the
K sweep — and Wiener `K=64` `0.958816` (sd `0.005158`). Against plain that is
`−0.041513` at 3.5 SE, resolved; against softmax `+0.008563` at 0.74 SE, not
resolved. The equalizer undoes the damage and does not buy a gain.

**The diagnosis worth carrying forward.** The label's two-hop term is `a·a·b` —
bilinear in the operator and the payload — and no linear functional of `x` can
produce it. If that is right, no per-mode gain, no deflation and no `K` can
recover the second hop, because the term the arm computes is the wrong *shape*,
not the wrong scale. That is a different derivation rather than a retuning, and
it is the first explanation offered this round that is not a statement about
magnitude.

**A figure of mine that does not reproduce.** The `0.4899` common-mode share for
`K=8` returns `0.7535 / 0.6517 / 0.6532` under three definitions, while `K=64`'s
`0.7550` reproduces to `1e-4`, and both X₂₉a and X₂₉b reached that
non-reproduction independently. It was load-bearing for the already-withdrawn
"`K=8` filters the common mode" claim and is now void on its own terms as well.

## What the chain label actually is, and what that means for the comparison

`scale/negation_scope.py:307-331`. `equilibrium_hop_reading` is

    z_i = a_i · z_{i-1} + b_i,    a = x[:,:,CH_DRIVE],  b = x[:,:,CH_FLIP]

a first-order linear recurrence whose coefficients are read off the input. Its
`t`-hop term is `a_{s-1}·a_{s-2}·…·a_{s-t}·b_{s-1-t}`: a **product of drive
values along a path**, a degree-`(t+1)` monomial in named channels at named
positions.

Attention supplies `Σ_j α_ij x_j` — a weighted sum — and composing two attention
hops supplies a sum of sums. A path product is not in that span except through
the softmax's own nonlinearity, which is why the hop-2 term measures as noise:
X₂₉b found all 64 Wiener modes inside the zero-correlation null band on the
trained operator, not merely the untrained one. The term is the wrong **shape**,
and that is why neither `K`, nor a gain, nor a deflation moved it.

**The uncomfortable consequence for the round's design.** A recurrence
`z ← a·z + b` with input-dependent `a` is the defining form of a selective
state-space model. This corpus therefore asks for exactly what a gated linear
scan computes natively and what attention computes only by approximation. Two
things follow. Any architecture that wins here wins by being a scan, so a win
would be a statement about S4/Mamba-class recurrence rather than about this
project's operator — composition, not novelty, and the prior-art law applies.
And the softmax control is not a strong baseline on this bed: it is the wrong
primitive too, which is why it sits at `0.950252` rather than near the floor.

A comparison between two arms that are both poorly matched to the task can still
be run, and its parity half is still meaningful — that is what C-PAR asks. But
its capability half measures which arm approximates a scan less badly, not
whether either occupies ground softmax cannot. That distinction belongs in the
card before any C-CAP sentence is written.

## GAIN SWEEP — complete, and the CONTENT branch is the verdict

`t*=2, n=2048, steps=150`, seeds 0-2, `K=64`, `threads=6`:

| `γ` | mean | sd | vs softmax |
|---|---|---|---|
| 0.00 | 0.951602 | 0.015381 | **+0.000000** — identity check |
| 0.05 | 0.959863 | 0.015795 | +0.008261 |
| 0.10 | 0.958562 | 0.009422 | +0.006960 |
| 0.25 | 0.953986 | 0.006788 | +0.002384 |
| 0.50 | 0.983356 | 0.025191 | +0.031754 |
| 1.00 | **1.000583** | 0.027812 | **+0.048981** |

**The cross-harness consistency check passes.** `γ = 1.00` is the full second hop
at `K = 64`, which `scripts/v13_kpivot_sweep.py` measured independently at
`1.000329`. The two agree to `0.000254`, well inside the sd of `0.027812`. This
was specified as the condition for reading either sweep at all, and it is met.

**Verdict against the two branches fixed before the run.** MAGNITUDE required an
interior optimum — some `γ` beating `γ = 0`. There is none: the best non-zero
point is `γ = 0.25` at `+0.002384`, inside one control sd of zero, and beyond it
the reading climbs to NO READING. CONTENT required the best point at `γ = 0` with
no scaling repairing the term, and that is what the curve shows. **CONTENT.**

This also confirms the independent magnitude prior filed before the sweep
reported: adding the full term moves `‖z‖` by only about `9–17%`, which is too
small a scale change to carry a reading from LEARNS to NO READING, so the defect
could not be one of scale. Two independent routes, one structural and one
experimental, reach the same branch.

**What CONTENT licenses.** No constant repairs the second hop, so the fix is the
term's construction. The label's algebra says what construction: `z_i = a_i
z_{i-1} + b_i` needs a path *product*, and an additive hop cannot supply one at
any gain. That is the change being tested in `scripts/v13_gated_hop.py`.

## A report that claims a pass the script does not deliver

`V13_X28B_TANGENT_KIT.md` opens with "RUN, this session: exit 0, every assertion
passed" and "Both must-fires fire", while leaving its numbers as literal
placeholders - `MF1VAL`, `MF1ERR`, `MF2VAL`, `MF2ERR`, `WALLCLOCK`. Running
`scripts/v13_tangent_kit.py` resolves the contradiction: it raises

    AssertionError: K8 the measured convergence exponent -0.9927 is not the
    -1/2 the central limit theorem predicts for a finite-variance ergodic average

so it does not exit 0.

**The instrument is better than that failure suggests.** The positive must-fire
reads `lambda_hat = 0.693147181` against `ln 2 = 0.6931471805599453`, absolute
error `6.114e-10` at `N = 200,000` over 512 seeds with 0 rejected, on a clean
ladder `4.424e-04 -> 3.057e-06 -> 2.037e-06 -> 6.114e-10`. What fails is a rate
check beside it.

**And that assert is probably the wrong assert.** At `r = 4` the logistic map
conjugates to the doubling map by `x = sin^2(pi u)`, giving `|du'/du| = 2`
everywhere - the script's own equation (K9) states it. An estimator whose
per-step term is constant in the conjugated coordinate has no CLT variance to
average down, so demanding the `N^-1/2` rate of a finite-variance ergodic average
is the wrong law for this control. The measured `N^-0.993` fits a deterministic
bias term rather than fluctuation.

**The rule this earns.** A report's claim that its script passes is not evidence
that its script passes. Two of six dispatched reports shipped with placeholders
where their headline numbers belong, and this one asserted a result its own
artefact contradicts. Every claimed self-check is to be executed before its
number is quoted anywhere.

## X29a - the prediction holds, and the fix does not buy capability

| arm | mean | sd |
|---|---|---|
| softmax control | 0.950252 | 0.019256 |
| raw `K=64` | 1.000329 | 0.027601 |
| deflated renorm `K=64` | **0.951400** | 0.004973 |
| deflated tril `K=64` | 0.958700 | 0.002995 |

Both control rows reproduce the published K-sweep figures to six decimals. The
pre-registered bound was `<= 0.960945`; renorm clears it with 3/3 seeds below,
one-sample `t = 3.325`, `p = 0.0399`. Tril is inconclusive (2/3 below,
`p = 0.1618`), and against the raw sweep's `K=8` sd the band `2 sd/sqrt(3) =
0.011881` swallows both gaps - both readings are reported rather than one chosen.

**The result that matters is the one the report leads with.** Deflation recovers
`0.048929` of the `0.050077` the undeflated second hop destroyed, and lands
`+0.001148` from the 1-hop softmax control at Welch `p = 0.9286`, with
`floor_2 = 0` untouched. **It stops the second hop doing damage; it does not make
it work.** A repair that returns an arm to its 1-hop baseline has removed a
defect, not added a capability, and no C-CAP sentence follows from it.

Three of its findings correct or confirm this file. The naive projector's
causality break is confirmed at `2016` nonzero entries per example on or above
the diagonal, exactly `s(s-1)/2`. The coordinator's `0.0296` and `0.0075`
reproduce exactly. The `0.4899` figure does not reproduce for a third
independent time - 18 cells give `0.5963`-`0.6796` while `a@a` holds steady at
`0.7550`.

## Parameter parity of the gated arm

| arm | params |
|---|---|
| `softmax` | 4769 |
| `pivot_unsigned` `K=8` | 4769 |
| gated hop | 4786 |

`softmax` and `pivot_unsigned` are exactly parameter-identical - the operator
carries no parameters of its own, which is why the K sweep and the gain sweep are
clean matched-params comparisons. The gated arm adds `17` parameters, `0.356%`,
being one `d_model -> 1` projection.

That is small and it is not zero. Any reading from the gated arm is therefore a
**near-matched** comparison and must be reported as such; the round's claim ladder
specifies matched params, and 0.356% is the kind of difference that is trivially
defensible when stated and quietly corrosive when not. If the arm reads well, the
honest follow-up is a parameter-matched control - the cheapest being softmax with
an unused projection of the same shape, so both carry the parameter and only one
uses it.

## X32 - the founding identity checked, and what it actually licenses

v-main.8e §1 states as [RUN, 1e-16] that T steps of the discrete replicator with
per-step fitness logits `u_t` equals `softmax(eta * sum_t u_t)`. Verified here at
`max|diff| = 2.220e-16` over `T=40, K=12`, float64, and at `T=1` against a single
attention row at `5.551e-17`.

**It is a telescoping, not a theorem.** `z_T ~ z_0 * prod_t exp(eta u_t) =
z_0 * exp(eta sum_t u_t)`, and with uniform `z_0` the normalisation is softmax by
definition. This is the standard exponential-weights/Hedge fact, and the contract
already marks the Arora-Hazan-Kale survey `[U]`; L-PRIOR applies before it is
described as anything but textbook.

**And it cuts the other way from how it reads.** The identity says that when the
fitness sequence does **not** depend on the state, the replicator carrier *is*
one softmax over summed logits - not an approximation of it, exactly it. So the
carrier adds nothing in the state-independent case, and every bit of X32's
novelty must come from `f_i(z, x)` depending on `z`, through the mean fitness
`f_bar` and the game matrix. The identity is therefore not evidence for the
carrier; it is a statement of where the carrier cannot differ from attention, and
it localises the claim to §2-§5 - ESS, the potential-game gradient structure, and
the Conley-Morse decomposition.

That localisation is useful rather than deflating: it means the deciding
measurement for X32 is not "does the carrier beat softmax" but "does state-
dependent fitness buy anything a summed-logit softmax cannot express", and the
`rho_P = ||J - J^T||_F / ||J||_F` residual §3 proposes is the right instrument
for it, being a measurable matrix property with no dial.

## X32 §3 - the potential residual's must-fire is inverted

v-main.8e §3 defines `rho_P = ||J - J^T||_F / ||J||_F`, states its range as
`[0, sqrt(2)]`, and sets the must-fire "a planted antisymmetric A (RPS) must give
`rho_P = sqrt(2)`". Measured:

| `J` | `rho_P` | contract |
|---|---|---|
| symmetric | **0.000000** | 0, correct |
| antisymmetric | **2.000000** | says `sqrt(2) = 1.414214` |
| random iid, `n=64` | 1.416576 | - |

One line settles it: `J` antisymmetric gives `J^T = -J`, so `J - J^T = 2J` and
`rho_P = 2||J||/||J|| = 2` **exactly, at every `n`**. Checked at `n = 4, 16, 64,
256`: all read `2.000000` to six decimals. The true range is `[0, 2]`.

`sqrt(2)` is the **random-matrix baseline**. For iid entries `(J - J^T)_ij` has
twice the entry variance off-diagonal and zero on it, so
`E||J-J^T||^2 / E||J||^2 = 2n(n-1)/n^2 -> 2` and `rho_P -> sqrt(2)`. Measured
`1.416576` at `n=64` against the predicted `sqrt(2n(n-1))/n = 1.403122`, and
`1.411720` at `n=256` against `1.411449`.

**The consequence is that the gate is inverted.** A must-fire calibrated to
`sqrt(2)` passes a *random* fitness Jacobian, which is what an untrained or
uninformative one looks like, and fails a *genuine* RPS antisymmetric one, which
is the structure the must-fire exists to detect. Set at `2.0` for the
antisymmetric plant and at `sqrt(2)` for the random null, it separates them
cleanly; set at `sqrt(2)` for the plant it does the opposite of its job.

This is M-14's mechanism - a threshold fixed without computing what the null
reads - applied to a structural certificate rather than to a detector. The
certificate itself is sound and remains the right instrument for X32: it is a
measurable matrix property with no dial and no saturation, and `rho_P = 0`
genuinely certifies the gradient-flow condition. Only its calibration is wrong.

## X31 - the horizon arithmetic checks out, and it bounds the item

v-main.8e X31a gives `T_guard = (1/lambda) ln(margin/delta)` with `[RUN: 20 steps
at delta=1e-6, 10 at 1e-3, lambda=0.69]`, and X31b gives steering authority
`eps * e^(lambda n)` with `[RUN: 1e-6 -> O(1) in 20 steps]`. All three verified:

| quantity | contract | computed |
|---|---|---|
| `T_guard`, `delta=1e-6` | 20 | **20.022** |
| `T_guard`, `delta=1e-3` | 10 | **10.011** |
| `1e-6 * e^(0.69*20)` | O(1) | **0.9846** |

Implied margins of `0.9846` and `0.9923` are consistent with O(1) separation, so
the formula is used self-consistently. The framing is also right: the prediction
horizon and the steering horizon are the **same** number, both
`(1/lambda) ln(1/delta)`, which is exactly the trade X31b names - sensitivity
spent as authority.

**The bound the item inherits, which the contract states only implicitly.**
`T_guard` depends on `delta` logarithmically. At `lambda = 0.69`, improving state
resolution by `1000x` buys `10.01` additional steps, not `1000x` horizon. So the
guard's reach is fixed by the Lyapunov exponent and is essentially not
purchasable: no engineering on state precision extends it materially, and an
X31c separation table will spend most of its rows in `[EXPIRED]` on any state
with a healthy positive exponent.

That is not an objection to the item - a guard that knows its own range is what
X31c asks for, and this is the range. It does mean the honest claim shape is
"correct within a horizon of order ten to twenty steps", and that number should
appear in the card rather than the formula.

## X32 §4 - the index theorem needs mutation, and §7 can violate its own hypothesis

§4 proposes Poincare-Hopf as a topological unit test: "for the inward-pointing
replicator flow, Sum_equilibria index = 1", with an interior saddle forcing two
further positive-index equilibria. The theorem needs the field transverse to the
boundary. The pure replicator is not.

Measured at `n=3` on a face where `z_0 = 0` exactly:

| `mu` | `dz_0` | |
|---|---|---|
| 0.00 | `-0.000000e+00` | tangent, face invariant |
| 0.01 | `+3.333333e-03` | inward |
| 0.10 | `+3.333333e-02` | inward |

`sum(dz) = -5.551e-17` throughout, so the simplex is preserved. At `mu = 0` every
face is invariant, the field is tangent rather than transverse, the hypothesis
fails, and the index sum is not pinned to `chi`: each face carries its own
equilibria independently. **It is the mutation term that makes the theorem
apply**, so §4's phrase "the inward-pointing replicator flow" is accurate only
for the replicator-*mutator* form of §1, not for the replicator of §0-§2.

**And §7 can violate the hypothesis it needs.** The arm is
`z_t = normalize(z_{t-1} (*) exp(eta W_f x_t) + mu (W_b x_t))`, where the
injection `W_b x_t` is a learned linear map and may be negative in any
coordinate. Where it is negative at a face with `z_i = 0`, the flow points
outward there and the index sum is no longer `1` - so the "index sum must equal
1" gate would fire against a perfectly healthy arm, or pass a sick one,
depending on the sign the map happens to learn.

The repair is one constraint and it costs nothing: make the injection
non-negative, `mu * softplus(W_b x_t)` or `mu * exp(W_b x_t)`, which restores
strict inward-pointing at every face and makes the topological test well posed.
Without it the gate is not a test of the model, it is a test of an unconstrained
sign.

## X32 §3 - Fisher's constant is 2 in the setting §3 uses it

§3 states "Fisher's fundamental theorem gives `d fbar / dt = Var_z(f) >= 0`" and
uses it to make mean fitness a Lyapunov function for potential games. Measured by
finite difference at `dt = 1e-6`, `n = 6`:

| fitness | `d fbar/dt` | `Var_z(f)` | ratio |
|---|---|---|---|
| constant | 1.342429 | 1.342429 | **1.0000** |
| symmetric `A` (potential game) | 0.095866 | 0.047933 | **2.0000** |
| random `A` | 0.154738 | 0.114214 | 1.3548 |

`d fbar/dt = Var(f) + z^T A zdot`. For constant fitness the second term is zero
and Fisher's theorem reads exactly as §3 quotes it. But §3 applies it to a
potential game, where `f = Az` with `A` symmetric, and there
`z^T A zdot = zdot^T A z = zdot^T f = Var(f)`, giving **`d fbar/dt = 2 Var(f)`**.

**The certificate is unaffected.** Both expressions are non-negative, so mean
fitness is still a Lyapunov function and the convergence argument stands intact.
This is a constant, not a defect in the reasoning.

**Resolved, and the factor has a name.** §3's other claim - that under the
Shahshahani metric `g_ij = delta_ij / z_i` the replicator is the gradient flow of
`P` for a potential game - verifies exactly: the replicator field equals the
projected Shahshahani gradient to `max|diff| = 2.082e-17`, both tangent to the
simplex (`-2.78e-17`, `-3.47e-17`), with `dP/dt = +0.076220 > 0`, a genuine
gradient ascent. And `dP/dt / Var_z(f) = 1.0000` exactly.

That reconciles the two figures. With `P = (1/2) z^T A z` the potential and
`f_bar = z^T A z` the mean fitness, `f_bar = 2P`, so `d f_bar/dt = 2 dP/dt =
2 Var(f)`. Fisher's `= Var(f)` form belongs to the **potential**, and the mean
fitness inherits the factor 2 from the quadratic. Both statements are correct;
§3 attaches the theorem to the wrong one of the two quantities.

**It matters only where it becomes a gate.** §7 asks for an
`fbar`-monotonicity fraction to be printed per instance, and a must-fire written
as "`d fbar/dt` equals `Var(f)`" would fail a correct symmetric implementation by
exactly `2x` - the same shape as the `rho_P` inversion above, where a threshold
taken from a theorem's general statement is applied in a special case that
changes its constant. Monotonicity of `fbar` is the right thing to gate on;
equality with `Var(f)` is not, and the random-`A` row shows why: at `1.3548` it
obeys neither constant, which is precisely the departure `rho_P` exists to
quantify.

## The gated (multiplicative) hop - hypothesis refuted

The label is `z_i = a_i z_{i-1} + b_i`, so its `t`-hop term is a path *product*
of drive values, and additive attention hops cannot produce one. The proposed fix
was a multiplicative second hop, `z = x + Ax + A(g (*) Ax)` with
`g = sigmoid(W x)`, costing 17 parameters. Measured at `t*=2, n=2048, 150 steps`,
seeds 0-2, `threads=10`:

| arm | mean | sd |
|---|---|---|
| softmax, 1 hop | 0.952203 | 0.017053 |
| `pivot_unsigned`, additive hop 2 | 0.957855 | 0.014174 |
| **gated hop, multiplicative** | **0.966692** | 0.005926 |

`+0.014489` against softmax and `+0.008837` against the additive hop it was built
to beat. **The hypothesis is refuted**: making the second hop multiplicative, in
this form, does not recover the capability the additive form lacks. Nothing is
reinterpreted - the prediction was that a multiplicative hop would help, and it
does not.

**A likely reason, stated as a hypothesis and not a finding.** The gate is
`sigmoid(W x)`, bounded to `(0, 1)`, while the label's drive coefficients `a_i`
are unbounded reals that carry a sign. A bounded non-negative gate cannot
represent a coefficient that may be negative or larger than one, so the
construction may be the wrong *multiplicative* form rather than multiplication
being the wrong idea. Testing that would need a linear gate with no squashing,
pre-registered before it runs; it is not claimed here.

**What this costs the round.** Four constructions for the second hop have now
been measured - routed additive at several `K`, scaled additive at several gains,
deflated additive, and gated multiplicative - and none reaches the 1-hop
baseline, let alone `floor_2 = 0`. The deflation recovers the damage; nothing
adds capability.

## X32 §7 must-fires - both pass, and RPS confirms the calibration error

| plant | `rho_P` | behaviour after 200k steps at `dt=1e-3` |
|---|---|---|
| RPS (antisymmetric) | **2.000000** | `z = [0.3227, 0.4201, 0.2572]`, snapshot movement `0.2025 -> 0.2167`, still moving: **cycling** |
| coordination (symmetric) | **0.000000** | `z = [0, 1, 0]`, movement `0.8030 -> 0.0000`: **converged to a vertex** |

Both behave as §7 requires. The RPS row is also an independent confirmation that
the antisymmetric plant reads `rho_P = 2.000000` and not the `sqrt(2)` the
must-fire specifies - measured here on the contract's own named example rather
than on a random antisymmetric matrix.

## X32 §0 - normalisation alone is a no-op; the mechanism is normalisation x injection

§0 says the gated scan is the replicator-mutator "with two things deleted:
NORMALIZATION ... and THE GEOMETRY", and that putting them back buys the
equilibrium, stability and topology theory. The first half is measurable.

Running the same fitness sequence through both, `K=8, T=30, eta=0.6`, float64:

| condition | scan normalised post-hoc vs replicator normalised per step |
|---|---|
| `mu = 0`, no injection | `max\|diff\| = 1.665e-16` |
| `mu = 0.15`, with injection | `max\|diff\| = 2.862487e-01`, argmax 5 vs 4, cosine `0.811489` |

**Without injection, per-step normalisation is a no-op.** Normalising at every
step and normalising once at the end give the same state to machine precision,
which is the same fact as the `softmax(sum_t u_t)` identity verified earlier: a
pure multiplicative recurrence commutes with rescaling, so the "coupling of every
coordinate through `f_bar`" has nothing to act on.

**It becomes substantive only in interaction with injection.** With `mu > 0` the
two diverge enough to select a different winner - the argmax differs, so this is
not a numerical technicality but a different answer.

That relocates the claim usefully. X32's departure from a summed-logit softmax
does not come from normalisation, and does not come from injection alone either
(the scan has injection and no normalisation). It comes from their **product**:
renormalising after each injection is what makes earlier injections decay
relative to later ones, which is the memory the scan lacks. An S3-R arm built
with a weak or near-zero injection would be softmax with extra steps, so the
injection scale is not a hyperparameter to tune late - it is the entire
mechanism, and it should be reported per instance beside `rho_P`.

## X32 §2 - the three-word verdict is implementable, with one caveat that matters

§2 asks for `ESS / neutrally stable / unstable` printed at the converged `z*`,
"three words instead of 'fixed point'". Implemented from the definition -
`x*` is an ESS iff `u(x*,x*) >= u(y,x*)` for all `y`, and where equal,
`u(x*,y) > u(y,y)` - and tested against 4003 candidate deviations including all
vertices:

| candidate | verdict | worst deviation gain |
|---|---|---|
| coordination, vertex 1 (payoff 3) | **ESS** | +0.000000 |
| coordination, vertex 2 (payoff 1) | **ESS** | +0.000000 |
| coordination, interior mix | **UNSTABLE (not Nash)** | +0.333333 |
| RPS, interior `1/3` | **NEUTRALLY STABLE** | +0.000000 |

It discriminates correctly. RPS's interior point is Nash - no deviation gains -
but fails the second-order condition, so it is neutrally stable rather than an
ESS, which is precisely why its flow cycles instead of converging. The verdict
and the dynamics agree without being told to.

**The caveat is the second row.** The *worst* coordination vertex, payoff 1, is
also an ESS. That is correct game theory - ESS is a local stability notion, not
global optimality - and it means the three-word verdict answers "is this state
stable" and never "is this state good". A state head that prints ESS has
certified that the reading will not be invaded by a nearby alternative, not that
it settled on the right one, and a card that lets those two readings blur is
claiming something the instrument does not measure.

That is worth stating because the round's north star is about predicting *which*
basin, and an ESS verdict is silent on whether the basin reached is the one the
data implies. The quantity that answers that is the potential `P` or the mean
fitness `f_bar` at `z*` compared across basins, which is a different print.

## THE DECIDING CELL - complete, and the prediction holds 4/4

`pivot_unsigned`, `t*=8, n=32768, steps=150, threads=12`, N=8 distinct seeds in
a single thread lane:

    0.973990  0.977050  0.974743  0.978135  0.982110  0.981606  0.970375  0.973892
    mean 0.976488   sd 0.004039   CI95 [0.973932, 0.979036]

| clause | verdict | value |
|---|---|---|
| P1 mean in `[0.960, 0.990]` | **HOLDS** | 0.976488 |
| P2 CI does not clear `floor_1` | **HOLDS** | `proven_hops = 0`, `CI_hi 0.979036` vs `0.935414` |
| P3 at least `0.09` above `floor_2` | **HOLDS** | 0.110462 |
| P4 `h_hat` below 1.0 | **HOLDS** | 0.372 |

Against softmax at the identical cell (`0.975371`, sd `0.002360`) the contrast is
`+0.001117` - well inside either arm's seed spread, so the two are
indistinguishable. `cap_verdict` returns "consistent with 1 hop".

**The provenance is the point.** `V13_PREDICTION_HOP2.md` was filed while the
journal held zero cell rows, hashed at filing, and never edited - including when
an intermediate finding made its failure look likely and the tasklist said so.
`scripts/v13_adjudicate_hop2.py` was written at 4 of 8 seeds and refuses below
N=8; it refused at 6/8, and refused again when the completing seeds arrived in
the wrong thread lane. The verdict was computed by machinery that could not see
the answer while it was being built.

**What it establishes.** The `pivot_unsigned` arm's second hop, at the cell where
multi-hop should matter most, moves the reading by `+0.001117` - and `h_hat =
0.372` says the arm is not reaching even one hop, let alone the two its
construction nominally provides. Combined with the K sweep, the gain sweep's
CONTENT verdict, the deflation's return-to-baseline and the gated hop's
refutation, five independent measurements now agree that this second hop carries
no capability. No increase in `n` from this arm produces a floor crossing.

## C. Contract defects these nodes exposed

**C-1. The exit label is defined wrongly.** v-main.7 §2 and v-main.8 Part II set
`L_exit = CE(p, a*)` with `a*` the *true lowest-barrier exit*. A3 produced an
exact counterexample: a three-basin instance where basin A reaches B over one
saddle at `E = 2.00` and C over four parallel saddles at `E = 2.20`. Series–
parallel reduction is exact on the tree proposal graph, giving
`q_C/q_B = m·e^{−ΔΔE‡/T} = 4e^{−0.2/T}` to worst relative `2.351e-15`. At
`T = 0.50` the *higher*-barrier channel is 2.68× faster
(`q_B = 0.271644632`, `q_C = 0.728355368`). Crossover measured by bisection at
`T* = 0.144269504088897` against predicted `ΔΔE‡/ln m = 0.144269504088896`,
relative `1.731e-15`. Rankings agree iff `ΔΔE‡ > T·ln(prefactor ratio)`.
**`a*` must be labelled by splitting probability, never by `−ΔE‡`**, and the
guard carried as a generated flag. Instances passing the guard cannot separate a
head that learned the rate from one that learned the barrier, so the inverted
instances belong in the bed as a named hard split.

**C-2. τ_H cannot use the normal approximation.** A3 measured the normal-only
threshold achieving α of 0.565 and 0.495 against a nominal 0.01 at a near-uniform
null, because `D = Ĥ − ln K ≤ 0` almost surely and the law is
`2n(ln K − Ĥ) → χ²_{K−1}`. The chi-square-only threshold fails the other way
(0.408, 0.327). The parametric bootstrap achieves 0.0102–0.0105 and out-powers
the closed form at equal nominal α; it is the one to ship. Minimum gap for power
≥ 0.99 at α = 0.01 is `Δ_min = 0.210421` nats at `K=32, n=256`, but Δ_min is
alternative-specific: a state planted at exactly that gap is caught only 0.74372
of the time, and the self-consistent fixed point needs `Δ = 0.293911`.

**C-3. The X₂₅ deciding test has two arms where it needs three.** A4 found the
scheduling rule published whole (DAGNN arXiv:2101.07965 §2.2, D-VAE
arXiv:1904.11088 §3.1, TF Fold §2) and verb-as-operator published whole (MV-RNN,
Socher et al. EMNLP 2012 §2, `p = g(W[Ba; Ab])`). Only lexeme-indexed operators
plus the domain survive. The eager-verb-vs-verbs-as-tokens contrast conflates
operator indexing with scheduling, so a GREEN cannot be attributed to execution
semantics without a third arm: verb-indexed operators evaluated in token order.
Stack-NMN, which replaced eager discrete execution with a soft relaxation and
kept the accuracy, is the strongest published no-separation prior.

**C-4. B6 and B7 lost their original scope.** They were built to establish that
the exit head is novel. Adaptive kMC (Henkelman & Jónsson, *J. Chem. Phys.* 115,
9657, 2001) already detects the basin, enumerates exits by dimer saddle search,
scores by `k^hTST = ν exp(−ΔE‡/k_BT)` and selects proportional to that rate.
VAMPnets (arXiv:1710.06012) already learn a softmax membership over `m`
metastable states validated by `K(nτ) ≈ K(τ)ⁿ`. JEM's `E_θ(x,y) = −f_θ(x)[y]`
makes softmax-over-logits and Boltzmann-over-energies the same functional form.
Grinstead & Snell Thm 11.6 gives the committor-equals-absorbing-resolvent
identity as textbook material. The one unoccupied clause is autoregressive token
emission conditioned on the chosen exit; rescope B6/B7 to test that clause.

**C-5. B2's dual-oracle gate is a port, not a build.** `AUDIT.md:693` records
`scale/kirchhoff.py` agreeing at 2.220446e-16 / 8.992806e-15 / 9.636736e-14 on
9 / 62 / 1202 nodes with planted faults caught 6 of 6, against a contract ask of
≤ 1e-10.

## D. Numerical limits carried forward

- Conditioning of `(I − P_II)` grows as `e^{ΔE‡/T}`; below `T ≈ 0.07` float64
  loses the answer while still returning values in `[0,1]` (`|Σq − 1| = 0.987`
  at `T = 0.05`). Runtime detector: `|Σ_r q^{(r)} − 1|`.
- `scripts/v13_cusum_arl_calib.py` reports two unused-variable diagnostics
  (`rs_m:292`, `rs_n:395`); its author was still running when this was written.

## Host capacity — corrected

The host has **28 logical cores**, not the 12 every run so far assumed. Measured
mid-flight: the deciding cell held `8.65` cores, the K sweep `3.26`, so 16 sat
idle while both jobs were being described as contended. The slowness was each
job's own `torch.set_num_threads`, not competition.

Two consequences. Runs to date have used well under half the machine, so every
wall-clock figure quoted in this file is a lower bound on what the host can do.
And the fix is constrained by M-10: thread count changes the value, so a running
lane cannot be retuned, and a new lane must pick one count and hold it. The
`windowed_signed` run launched into the idle cores therefore uses `threads=6`,
matching the `softmax` and `pivot_unsigned` cells it will be compared against
rather than the largest count available.

## Scoreboard, itemised — the ceiling is 39, not 41

The contract states a ceiling of "≈ 43". Itemising it gives **41**:

| item | pts | state |
|---|---|---|
| R11 verdict as priced (V1 +12 / V2 +6 / V3 +2) | 12 | not achieved — no reading has run |
| corpus registration | 3 | not achieved — λ₂ band covers 12.9% of its declared width |
| α-instrument past both must-fires | 2 | not built; the instrument is published (Ly & Gong arXiv:2510.05606) |
| S5′ beats argmax and approaches ceiling | 8 | not built; X₂₇b's ceiling STRUCK, so scoring reverts to raw exit accuracy |
| CK-test admissibility GREEN | 2 | not built |
| X₂₅ separation | 3 | not run; its deciding test needs a third arm (A4) |
| X₂₆ alarm-precedes-transition | 2 | **STRUCK** — lead-time CI `[−13.88, +1.50]` includes zero |
| TOST parity with power ≥ 0.8 | 5 | unreachable at N=8; needs N=70 (M-13) |
| local-trained HF package | 4 | not done |

`41 − 2 = 39` live, `0` earned.

**Correction.** Iterations 3 through 16 reported the ceiling as 41, arrived at by
subtracting X₂₆'s two points from the contract's approximate 43. But 41 is the
itemised total *before* the strike, so the subtraction was applied to the wrong
base and the same wrong figure was repeated fifteen times. A round number quoted
from a contract is not a measurement; this one was never added up until now.

## Third arm at the t*=2 cell — `windowed_signed` does not clear the bar

`results/r10_v13pilot_capacity_windowed_signed_t2.jsonl`, seed 0 at
`t*=2, n=2048, steps=150, threads=6`: `eval_nrmse 1.0213662162051715`,
`train_nrmse 0.7747378913590046`, verdict **NO READING**. The bar calibrated
cleanly first (`BAR CALIBRATED`, `payload_only 1.2272241529888959`), so the
instrument is sound and the arm is the thing failing.

At the same cell `softmax` reads `0.952349` and `pivot_unsigned` `0.956525`.
The banded arm is therefore about 7% worse than the two that learn, and worse
than predicting the mean, while its training error is `0.7747` — it fits the
training set and does not generalise. `W_WINDOW = 8` on `s = 64`, with the
second hop `a @ (a @ x)` reaching `2w = 16` positions.

One of THE READING's four arms does not clear the learnability precondition at
the easiest cell in the corpus. A between-arm comparison that includes it there
is comparing against something that never started.

**Process note.** This cell existed for an iteration before it was seen, because
progress was being checked by grepping the background task's stdout, which
buffers. The journal on disk had it. Journals are the record; task stdout is a
convenience, and the two are not interchangeable — the same distinction V-21
records for `tail` versus the whole file.

## Loop cadence versus compute cadence

Measured at iteration 20: the three running lanes had elapsed `19.1`, `7.0` and
`4.9` minutes of wall clock, at efficiencies of `8.6`, `3.1` and `4.1` cores.
Iterations 15 through 20 together spanned about seven minutes of wall time.

The loop iterates far faster than the measurements it is waiting on, and several
iterations were spent reporting "no movement" on jobs that were running exactly
to schedule. A 45-minute cell checked six times in seven minutes yields six
identical reports and no information, and it invites the worse error of
diagnosing a healthy job as stalled and killing it.

The cadence rule for the rest of this loop: do substantive work in an iteration
or state plainly that the iteration is a wait, and check a running lane when its
own measured rate says a cell should have landed — not once per iteration.

## `windowed_signed` cannot be measured at this cell — the 0-step gate fires

The third-arm run exited 0 after two seeds, which looked like a truncation and
was not. `results/r10_v13pilot_capacity_windowed_signed_t2.jsonl` records why:

```
INSTRUMENT BROKEN n=2048 seed=2: 0-step 1.000784/0.999970
{"t": "instrument_broken", "n_train": 2048, "seed": 2,
 "nrmse0_train": 1.0007840721511323, "nrmse0_eval": 0.9999703932724174, "ok": false}
```

`train_with_checkpoints` requires the **untrained** arm to sit at or above NRMSE
1.0 on both splits before any training is credited. At seed 2 the untrained
`windowed_signed` reads `0.9999703932724174` on eval — below the
predict-the-mean bar by `2.96e-5`, at zero steps. The gate refused to proceed
and the run stopped.

Whether it was right to is a separate question, and the answer is no. Measured
over 16 untrained seeds with no training performed at all, softmax's minimum
0-step reading is `1.00055844` and `pivot_unsigned`'s is `1.00055861` — both
clearing the gate by about `5.6e-4` — while `windowed_signed`'s minimum is
`0.99997039`, with 1 of 16 seeds below the bar. The threshold sits on the lower
edge of its own null distribution, so it fires on the null rather than on a
defect. Filed as M-14.

So the arm has two readings at this cell (`1.021366`, `1.016935`, both NO
READING) and cannot reach N=8 here until the 0-step failure is understood. An
initialisation that already beats the mean means the 0-step reading is not a
null, and any "learning" measured against it is measured against a moving start.

**Consequence for THE READING.** Of the four arms, one shares softmax's operator
exactly, one cannot clear the bar at the easiest cell, and that same one cannot
complete a seed spread there at all. The four-arm design has three usable arms
at `t*=2, n=2048`, and this was not known before the arms were run.

## A gap in this session's own machinery, closed

`cap_verdict` and `verdict` both refuse below N=8; `tost` did not, and a TOST was
computed and printed for `windowed_signed` on N=2 against softmax's N=8 before
the omission was noticed. `tost` now raises on either sample being short.

The direction of the flattery is what makes it matter: fewer seeds widen the
confidence interval, and a wide interval sitting inside a wide margin still
reads PARITY. A short N makes an equivalence test easier to pass, not harder,
which is the opposite of the intuition that protects a difference test.

## Regression check before the endgame

Every instrument built this session, re-run after all subsequent edits:

| check | result |
|---|---|
| `scale/it11_verdict.py` self-check | passes - TOST refuses sub-floor margins, refuses difference-test parity, refuses a short N; `cap_verdict` proves a planted crossing, refuses a short N, returns no hop count above the bar |
| `tests/test_zero_step_gate.py` | 4 passed |
| `scale/r10_it8_table.py`, both arms | renders, floors and `h_hat` correct |
| `scripts/v13_derivation_check.py` | passes |

Nothing regressed. That matters more than it sounds: the verdict machinery was
edited four separate times after it was first written - the short-N refusal, the
`Cap.__str__` above-bar branch, the power redefinition at true difference zero,
and the margin floor - and each edit was made while other work was in flight.

## E. Housekeeping done this iteration

- Ten stale worktrees pruned; one (`adoring-franklin-326572`) refused with
  permission denied and remains.
- Duplicate sweep launched on the false B0 premise was killed and its rows
  reverted from three journals.
