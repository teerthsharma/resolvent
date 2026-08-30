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
| A11 | X₂₈b tangent kit: JVP, Lyapunov via Benettin, adjoint gradcheck, both must-fires | `V13_X28B_TANGENT_KIT.md`, `scripts/v13_tangent_kit.py` | RUNNING |
| A12 | X₂₈c prior art: Kantz–Grassberger `κ = λ(1−d)`, and whether the triangle is circular | `V13_X28C_PRIOR_ART.md` | RUNNING |
| A13 | Independent audit of this session's numeric claims | `V13_CLAIM_AUDIT.md` | RUNNING |

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
| B9 | 4060 throughput probe, 500 steps, tokens/sec and peak bytes | — | **DONE** — `scripts/v13_b9_4060_probe.py`, card is an RTX 4060 **Laptop** (8188 MiB, sm_89, 24 SMs); signed 2.72× slower and 2.49× heavier than softmax at L=16 |
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
`n=16384 → 293.94 s`, a log-log slope of `secs ~ n^1.338` where linear would be
`1.000`. The cause is visible in the runner: `train_with_checkpoints` is
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
| 23 | 0.4948 | **first fit** | ≈0.03 |
| 24 | 0.4846 | yes | 0.042 |
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
| `‖hop2‖ / ‖a@a‖` | **0.1932** |

The routed hop is a rank-8 approximation to a rank-62 object. It keeps 74% of
the direction and 19% of the magnitude, discarding 54 of 62 dimensions. Since
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

**Harness control.** Before the K comparison is read, the standalone script's
own softmax control was checked against the published sweep at the same cell:
`0.951348` (sd `0.014927`, seeds 0-2) against `0.952349` (sd `0.010101`, N=8).
The `0.001001` gap sits well inside both spreads, so a difference across `K` will
be attributable to `K` rather than to the harness. Without this the sweep would
be comparing its own training loop against the round's, not one `K` against
another.

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

If `K=32` and `K=64` continue the trend, the rank explanation is dead and the
reading becomes that the routed term injects signal uncorrelated with the label
— more pivots admitting more of it — so the failure is in what hop 2 computes,
not in how much of it survives. That is a different repair: the term's
construction, not its width. Held as provisional until all four points land.

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

## E. Housekeeping done this iteration

- Ten stale worktrees pruned; one (`adoring-franklin-326572`) refused with
  permission denied and remains.
- Duplicate sweep launched on the false B0 premise was killed and its rows
  reverted from three journals.
