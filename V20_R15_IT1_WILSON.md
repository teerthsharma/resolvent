# V20 R15 it.1 — WILSON: GROUND TRUTH

Repository `C:\Users\seal\Desktop\New folder (32)`, branch `v17k-gate0`, HEAD `207e7b9`.
All commands run 2026-09-02 from the repository root. Nothing in the tree was
modified except this file and the two `house-events.jsonl` log lines.

Working tree at the time of reading (`git status --porcelain`):

```
 M house-events.jsonl
?? CEQ_V20_R15_CONTRACT.md
?? V20_R15_JOURNAL.md
```

`CEQ_V20_R15_CONTRACT.md` and `V20_R15_JOURNAL.md` are **untracked**; the
contract's mtime is `Sep  2 00:47`, inside this session. Section 7 reads it as
it stands on disk.

---

## 1. THE ARMS THAT EXIST

`find ceq -name "*.py" -not -path "*__pycache__*"` returns 43 files. The
listing below covers every module named in the question plus every other
top-level `ceq/*.py`.

### 1.1 The named modules

| module | entry point, at line | operator it computes |
|---|---|---|
| `ceq/arm_phase.py` | `class ArmPhase(nn.Module)` at `ceq/arm_phase.py:456`; `operator()` at `:158` | `A_ij = softmax_j(q_ij − Re C_j + s_j) · exp(i(Im C_i − Im C_j))` with `m_j = clamp(u_j, 0, 1)`, `C_j = Σ_{k≤j} log m_k + i Σ_{k≤j} θ_k` — a complex prefix-phase carrier with a hard-capped magnitude (docstring `ceq/arm_phase.py:1-8`) |
| `ceq/arm_pl.py` | `class ArmPL(nn.Module)` at `ceq/arm_pl.py:358`; `operator()` at `:99` | `O_i = Σ_{j≤i} softmax_j(q_ij − C_j + s_j) · V_j`, `C = scan(g)` — one causal softmax head carrying both the parity bind and the path-product label bind, key-side only (`ceq/arm_pl.py:1-5`) |
| `ceq/arm_smprime.py` | `class ArmSMPrime(nn.Module)` at `ceq/arm_smprime.py:497`; `operator()` at `:234`; `path_product()` at `:144` | `W_ij = G_ij · exp(qk·q_i·k_j) / Z_i^β` with `G_ij = Π_{k=j+1}^{i} m_k e^{iθ_k}` — the path product, not a prefix scan (`ceq/arm_smprime.py:1-7`) |
| `ceq/arms.py` | `class Arm(nn.Module)` at `ceq/arms.py:42`; `ALL_KINDS` at `:138` | Three-tier ladder over one operator: `attention` = `X + PX`; `appnp` = `Σ_k (ρP)^k X`; `signed` = `Σ_k A^k X`; `nash`. Docstring: "Three arms, one ladder. They differ ONLY in the operator." (`ceq/arms.py:1`) |
| `ceq/attention.py` | `ceq_operator()` at `ceq/attention.py:151`; `ceq_attention()` at `:237`; `register()` at `:266` | `out = v + Av + A²v + … + A^K v`, `A` strictly lower triangular and **signed** (`ceq/attention.py:1-3`) |
| `ceq/multizoom.py` | `class ZoomPlan` at `ceq/multizoom.py:97`; `multizoom_reference()` at `:237`; `coarsening_bound()` at `:369` | Exact attention nearby, mean-pooled far field with a `log n` logit correction, plus a stated coarsening error bound (`ceq/multizoom.py:1`) |
| `ceq/mz_kernel.py` | `_mz_kernel()` at `ceq/mz_kernel.py:46`; `multizoom_attention()` at `:182` | The Triton fused kernel for the above: pooled far field, `log n` mass correction and causal test evaluated inside one online-softmax loop (`ceq/mz_kernel.py:1-8`) |
| `ceq/hybrid.py` | `ceq_hybrid_attention()` at `ceq/hybrid.py:121`; `stock_attention()` at `:76`; `register()` at `:191` | `out = stock_attention(q,k,v) + Σ_{h=1..K} (αA)^h v`; at `α = 0` bitwise stock attention (`ceq/hybrid.py:1-5`) |
| `ceq/nonnormal.py` | `causal_operator()` at `ceq/nonnormal.py:105`; `settle_exact()` at `:121`; `non_normality()` at `:142` | A strictly causal, nilpotent, maximally non-normal coupling operator; `A^N = 0` so `(I−A)^{-1}` terminates exactly (`ceq/nonnormal.py:1`, `:23-32`) |
| `ceq/hankel.py` | `rank_real()` at `ceq/hankel.py:131`; `rank_plus_lower()` at `:279`; `myhill_nerode_classes()` at `:152`; `print_table()` at `:416` | The Hankel instrument: exact real rank by SVD, a rectangle-cover lower bound on nonnegative rank, and the Myhill-Nerode state count, kept in three separate columns (`ceq/hankel.py:1-9`) |
| `ceq/rips.py` | `make_case()` at `ceq/rips.py:207`; `corpus()` at `:261`; `rerouted_corpus()` at `:265`; `rips_edges()` at `:87` | Geodesic Vietoris-Rips graphs on `S²` spanning a connectivity transition; a Python port of a MuJoCo benchmark generator, arithmetic unchanged (`ceq/rips.py:1-10`) |
| `ceq/nash.py` | `nash_operator()` at `ceq/nash.py:128`; `qre_stance()` at `:67`; `safe_tau()` at `:58` | `A_ij = ρ·P_ij·stance_j` where `stance = 2s − 1` and `s = sigmoid((Ms+b)/τ)` — quantal-response equilibrium as the source of a token's stance (`ceq/nash.py:1`, `:14-16`) |
| `ceq/eviction.py` | `evicted_operator()` at `ceq/eviction.py:95`; `gated_operator()` at `:81`; `settle_evicted()` at `:118` | Removes a token from the key set **before** the softmax so survivors renormalise over what remains, against a post-softmax gate that leaves the token in the denominator (`ceq/eviction.py:1-17`) |
| `ceq/hopcache.py` | `class HopCache` at `ceq/hopcache.py:92`; `step()` at `:230`; `prefill_correction()` at `:75` | A scoped hop cache: one vector per hop per position, making incremental decode byte-identical to prefill at `K` attention rows per token (`ceq/hopcache.py:1-12`) |
| `ceq/lm.py` | `class TinyLM` at `ceq/lm.py:221`; `class Attention` at `:61`; `operator()` at `:71`; `compare()` at `:287` | A byte-level from-scratch LM with a pluggable attention operator; two arms — `softmax` via SDPA, `signed` via the path sum — at identical parameter counts (`ceq/lm.py:1-12`) |
| `ceq/capability.py` | `main()` at `ceq/capability.py:134`; `run_split()` at `:81`; `verdict()` at `:72` | The capability run: exact-match scores with a matched control, writing `results/capability.json` (`ceq/capability.py:1-2`, `:13`) |

### 1.2 Everything else in `ceq/`

Not named in the question but present: `ceq/autopilot.py`, `ceq/bench.py`,
`ceq/compat.py`, `ceq/corpus.py`, `ceq/diagnose.py`, `ceq/harness.py`,
`ceq/kdata.py`, `ceq/sizing.py`, `ceq/__init__.py`, plus the packages
`ceq/beds/` (`bed_1.py`, `bed_k.py`), `ceq/certs/` (`topological.py`),
`ceq/hf/` (`configuration_ceq.py`, `modeling_ceq.py`, `smoke.py`, `train.py`),
`ceq/hf_artifact/` (`configuration_ceq.py`, `modeling_ceq.py`), `ceq/x35/`
(`residual.py`), `ceq/x35p/` (`crb.py`, `kk.py`, `source.py`), and
`ceq/vendor/`.

### 1.3 Wired versus orphaned

The grep that establishes each answer, run per module `M` as

```
grep -rn "import M\b\|from \.M\b\|from ceq\.M\b\|ceq\.M\b" --include=*.py --include=*.ipynb \
  "C:/Users/seal/Desktop/New folder (32)" --exclude-dir=__pycache__ --exclude-dir=.git
```

**No module in the list is an orphan in the strict sense — every one has at
least one importer.** The distinction that does separate them is whether the
importers are live or retired. `pytest.ini:8` sets
`norecursedirs = attic *.egg .* _darcs build CVS dist node_modules venv {arch}`,
so anything imported only from `attic/` is not collected by a bare `pytest`.

| module | live (non-`attic`, non-snapshot) importers | status |
|---|---|---|
| `arm_phase` | `ceq/arm_smprime.py:74`, `scripts/v15_r1.py:118`, `scripts/v16_device_probe.py:67,878,1018`, `tests/arm_phase/test_arm_phase.py:46`, `tests/gate0/test_g11_label_cell.py:43` | wired |
| `arm_pl` | `scripts/v16_device_probe.py:66,877,1147`, `tests/arm_pl/test_arm_pl.py:47`, `tests/arm_smprime/test_runner_wiring.py:42`, `tests/mars_v15/test_neumann_leak_reading_tensors.py:178,180` | wired |
| `arm_smprime` | `ceq/compat.py:462`, `ceq/hf/modeling_ceq.py:478`, `scripts/k_cert.py:82`, `tests/arm_smprime/*`, `tests/gate0/test_g01_identity.py:46,416`, `tests/gate0/test_g11_label_cell.py:42`, `tests/gate0/test_g13_beta_learnable.py:45,235` | wired |
| `arms` | `ceq/diagnose.py:44`, `tests/deimos/test_deimos_r9_iteration1.py:199`, `tests/w4/test_w4_intervention.py:43`, `tests/w7/test_w7_nash.py:52` | wired |
| `attention` | `ceq/arms.py:30`, `ceq/hybrid.py:43`, `ceq/hopcache.py:82`, `ceq/hf/modeling_ceq.py:42`, `ceq/hf_artifact/modeling_ceq.py:31` | wired |
| `hybrid` | `ceq/hf/smoke.py:193`, `scripts/v15_norm_sweep_probe.py:14`, `tests/chase/test_hub_package_hardening.py:245,258,271,383,405,640`, `tests/w8/test_w8_real_model.py:45` | wired |
| `nonnormal` | `ceq/eviction.py:39`, `tests/w2/test_w2_nonnormal.py:36`, `tests/w3/test_w3_eviction.py:38` | wired |
| `hankel` | `scale/negation_scope.py:175`, `scale/planted.py:11`, `tests/cameron/test_hankel_worked_example.py:29`, `tests/cameron/test_m3_counter_squared.py:63`, `tests/loop/test_attic_never_removes_the_last_must_fire.py:65` | wired |
| `rips` | `ceq/certs/topological.py:76`, `scale/e4_harmonic.py:43`, `scale/e4_harmonic_reroute.py:40`, `scale/e4prime_spread.py:40`, `scale/foreman_lambda2.py:97`, `scale/impact.py:252`, `scale/negation_scope.py:894`, `scale/rips_gate.py:55` | wired |
| `nash` | `ceq/arms.py:31`, `scale/negation_scope.py:241`, `tests/cameron/test_m3_etasks.py:71`, `tests/deimos/test_deimos_r9_iteration1.py:304,332` | wired |
| `eviction` | `tests/chase/test_checklist_kills_are_evaluable.py:24`, `tests/chase/test_structural_zero_guard.py:25`, `tests/loop/test_m4_eviction_is_calibrated.py:38`, `tests/mars/test_mars_green_attacks_it4.py:21`, `tests/w3/test_w3_eviction.py:37` | wired |
| `hopcache` | `tests/chase/test_lean_refcount_binding.py:342`, `tests/w9/test_w9_hopcache.py:44` | wired |
| `lm` | `ceq/harness.py:34`, `ceq/sizing.py:327`, `ceq/bench.py:175`, `ceq/hf/smoke.py:100`, `scripts/k_cert.py:81`, `scripts/v13_b9_4060_probe.py:26`, `colab/train_ceq.ipynb:258` | wired |
| `capability` | `tests/cameron/test_capability_result.py:36`, `tests/foreman/test_capability_and_theorem_share_no_object.py:32,62,133` | wired |
| **`multizoom`** | `ceq/mz_kernel.py:37` only. Every test importer is under `attic/`: `attic/tests/chase/test_multizoom_cost.py:30,234`, `attic/tests/chase/test_multizoom_kernel.py:27`, `attic/tests/chase/test_multizoom_r5.py:37,287` | **imported only by `mz_kernel` and by retired tests** |
| **`mz_kernel`** | none outside `attic/`. All 11 importers are `attic/tests/chase/test_multizoom_cost.py:128,146,190,235`, `attic/tests/chase/test_multizoom_kernel.py:109,199,288,325,359`, `attic/tests/chase/test_multizoom_r5.py:366` | **no live importer** |

`ceq/__init__.py` declares
`__all__ = ["nonnormal", "eviction", "attention", "arms", "corpus", "nash", "hybrid", "hopcache", "lm", "sizing"]`
— `multizoom` and `mz_kernel` are absent from it, as are the three arm modules.

---

## 2. THE LEAN LEDGER, AS IT ACTUALLY COMPILES

### 2.1 Source files

`find lean -type f -not -path "*/.lake/*" | sort`:

```
lean/CEQ.lean
lean/CEQ/Contraction.lean
lean/CEQ/Nilpotent.lean
lean/CEQ/Occupancy.lean
lean/CEQ/OracleSeparation.lean
lean/CEQ/OrbitBound.lean
lean/CEQ/Refcount.lean
lean/CEQ/V15.lean
lean/CEQ/V15Fork.lean
lean/CEQ/V15Kernel.lean
lean/CEQ/V15Phase.lean
lean/CEQ/V15Source.lean
lean/CEQ/V16Domain.lean
lean/lake-manifest.json
lean/lakefile.lean
lean/lean-toolchain
```

Thirteen `.lean` source files, 3,278 lines total (`wc -l`): `CEQ.lean` 33,
`Contraction` 161, `Nilpotent` 114, `Occupancy` 89, `OracleSeparation` 205,
`OrbitBound` 84, `Refcount` 183, `V15` 328, `V15Fork` 251, `V15Kernel` 640,
`V15Phase` 395, `V15Source` 201, `V16Domain` 594.

### 2.2 The build

Toolchain present. `lean/lean-toolchain` reads `leanprover/lean4:v4.7.0`.
Inside `lean/`, `lean --version` reports
`Lean (version 4.7.0, x86_64-w64-windows-gnu, commit 6fce8f7d5cd1, Release)` and
`lake --version` reports `Lake version 5.0.0-6fce8f7 (Lean version 4.7.0)`.
(The shell default outside that directory is 4.33.1; elan resolves the pin.)

`lake build` in `lean/` **exits 0 with no output** — the project is fully built
from cache. `find .lake/build -name "*.olean" | wc -l` reads **13**, one per
source file. `lakefile.lean` requires mathlib4 at `v4.7.0`.

`grep -rn "sorry" CEQ.lean CEQ/*.lean` returns **seven hits, all inside prose
comments** asserting the absence of `sorry` (`CEQ.lean:15`, `V15.lean:41`,
`V15Fork.lean:44`, `V15Kernel.lean:72`, `V15Phase.lean:72`, `V15Source.lean:60`,
`V16Domain.lean:75`). **No `sorry` occurs in any tactic block.**

### 2.3 The cited numbers against the source

The numbered ledger is `CEQ_V16_CONTRACT.md:137-157` (17 rows, `#1`–`#17`, with
`#5` split into `#5a`/`#5b`); the earlier form is `CEQ_V15_CONTRACT.md:148-161`
(14 rows). Each contract row names a Lean identifier. Checked by
`grep -rn "\b<name>\b" CEQ.lean CEQ/*.lean`:

| # | name the contract gives | source declaration | verdict |
|---|---|---|---|
| 1 | `chain_path_product` | `lean/CEQ/V15.lean:74` | **EXISTS** |
| **2** | `prefix_logit_mask` | `lean/CEQ/V15.lean:128` | **EXISTS** |
| 2 (re-stated) | `prefix_logit_mask_restated` | `lean/CEQ/V16Domain.lean:221` | **EXISTS** |
| 3 | `parity_sign` | `lean/CEQ/V15.lean:190` | **EXISTS** |
| 4 | `resolvent_eq_path_sum` | — | **NO SOURCE DECLARATION** |
| **5a** | `three_corners_containment` | `lean/CEQ/V16Domain.lean:433` | **EXISTS** |
| 5b | `gate_zero_beta_zero_is_linear_attention` | `lean/CEQ/V16Domain.lean:483` | **EXISTS** |
| 6 | `bounded_gates_stable` | `lean/CEQ/V15.lean:266` | **EXISTS** |
| 7 | `scan_assoc` | `lean/CEQ/V15.lean:314` | **EXISTS** |
| 8 | `replicator_eq_cumsoftmax` | — | **NO SOURCE DECLARATION** |
| 9 | `walsh_two_point` + `multiplicative_epistasis_nonzero` | — | **NO SOURCE DECLARATION** (neither name) |
| **10** | `hop_floor` | — | **NO SOURCE DECLARATION** |
| **11** | `lyapunov_gate_stationarity` | — | **NO SOURCE DECLARATION** |
| 12 | `first_order_cannot_delay` | `lean/CEQ/V15Kernel.lean:140` | **EXISTS** |
| 13 | `GL_weights_powerlaw` | name occurs only in the header prose at `lean/CEQ/V15Kernel.lean:5`. The file declares `GL_weights_ratio_recurrence` (`:405`), `GL_weights_alpha_zero` (`:418`), `GL_weights_alpha_one` (`:426`), `GL_weights_pos` (`:435`), `GL_weights_strictAnti` (`:447`) | **NO DECLARATION UNDER THE CITED NAME** |
| **14** | `committor_eq_harmonic` | — | **NO SOURCE DECLARATION** |
| 15 | `resolvent_inverse_is_difference` | name occurs only in the header prose at `lean/CEQ/V15Source.lean:2`. The file declares `source_is_first_order_difference` (`:108`), `two_sources_recovered` (`:128`), `no_fill_in` (`:143`), `source_entry` (`:188`) | **NO DECLARATION UNDER THE CITED NAME** |
| 16 | `unit_phase_product` | `lean/CEQ/V15Phase.lean:92` | **EXISTS** |
| **17** | `approximation_bound` | — | **NO SOURCE DECLARATION** |

`#17'`, `#18`, `#19`, `#20`, `#21`, `#22`, `#23` **do not appear in
`CEQ_V16_CONTRACT.md`'s ledger at all.** Their only occurrence anywhere in the
tree's markdown is the untracked `CEQ_V20_R15_CONTRACT.md`, at lines 179, 185,
192, 202, 213, 219 and in the summary line at 250-251:

> `LEAN LEDGER this round: #18 (M1), #19 (M2), #22 (M9-F0) [M];`
> `#17′ (M3), #20 (M4), #21 (M6), #23 (M9-F1) [S]; #14 committor,`
> `#10/#11 floors [D].`
> — `CEQ_V20_R15_CONTRACT.md:250-252`

No Lean identifier is given for any of `#17'`, `#18`–`#23` in that file, and
none of them has a source declaration.

The `#` references that occur inside the Lean sources themselves are `#1, #2,
#3, #5, #5a, #5b, #6, #7, #12, #13, #16` (e.g. `lean/CEQ/V15.lean:5-6`,
`lean/CEQ/V16Domain.lean:586-594`). The train-gate axiom check at
`lean/CEQ/V16Domain.lean:586-594` prints `#print axioms` for exactly nine
declarations: `#1`, `#2` original, `#2` re-stated, `#3`, `#5a`, `#5b`, `#6`,
`#7`, `#16`.

**Summary of theorem numbers cited by the question with NO corresponding source
declaration: `#10`, `#11`, `#14`, `#17`, `#17'`, `#18`, `#19`, `#20`, `#21`,
`#22`, `#23`.** Of the question's list, `#5a` and `#2` exist.

---

## 3. THE MEASURED CELLS IN `results/`

`results/` holds **158 regular files** plus 4 subdirectories
(`kaggle_v17k_output/`, `m3_quintuple_v2_cuda_weights/`,
`m3_quintuple_v2_weights/`, `paired/`). Sizes and line counts from
`stat -c %s` and `wc -l` per file.

### 3.1 Zero-length files

**`results/r10_v13_b9_4060_probe.txt` — 0 bytes, 0 lines.** It is the only
zero-length regular file in `results/`. Confirmed by the per-file loop:
`results/r10_v13_b9_4060_probe.txt bytes=0 lines=0`.

`results/m3_quintuple_v2_cuda.lock` is 5 bytes, 0 lines (no trailing newline) —
a lock file, not a measurement.

### 3.2 The largest cells, with the quantity each actually holds

Keys read from the files, not inferred from names:

| file | bytes | lines | what it holds |
|---|---|---|---|
| `phaseD_weights_s64_d24_n8192_st150_seed{0,1,2}.pt` | 2,108,341 each | binary | torch checkpoints |
| `wilson_arms.jsonl` | 240,389 | 36 | per-arm records |
| `v17k_r4_retake.jsonl` | 173,525 | 424 | the R4 re-take cells |
| `hilbert.jsonl` | 109,165 | 142 | — |
| `cameron_aggregators.jsonl` | 881,838 | 8 | eight records, ~110 KB each |
| `arm_a_k1.jsonl` | 438,827 | 48 | keys `key`, `meta.seconds`, `value.{flip,gamma,tau,theta,tv}` — 100-element arrays per record |
| `arm_a.jsonl` | 1,092 | 3 | keys `s, k, draws, theta_c, theta_f, ci_c, ci_f, flip, d_theta, d_tv, tau, gamma, secs` |
| `m3_capability.txt` | 64,815 | 1,561 | — |
| `k_cert_local.json` | 35,843 | 1,272 | the local device certificate |
| `k_data_manifest.json` | 8,571 | 169 | dataset manifest |
| `p1prime_rows.json` | 35,712 | 2,189 | — |
| `v15_r1.jsonl` | 34,933 | 26 | R1's deciding cells; `secs` fields cited by `V16_DEVICE_CERT.md` §2.1 |
| `v17k_r4_floor.jsonl` | 48,408 | 118 | — |
| `G2_BASELINE_it2.sha256` / `_it4.sha256` | 36,129 / 36,849 | 262 / 270 | file hashes |
| `capability.json` | 4,265 | 142 | written by `ceq/capability.py` (`ceq/capability.py:13`) |

The full 158-row inventory was produced by
`for f in results/*; do printf "%s\tbytes=%s\tlines=%s\n" "$f" "$(stat -c %s "$f")" "$(wc -l < "$f")"; done`
and is reproducible verbatim by that command.

### 3.3 Referenced versus orphaned

Test, per basename `b`:
`grep -rl -F "$b" --include=*.md . | grep -v "kaggle/snapshot" | wc -l`.

**Referenced by at least one `.md`: 116. Referenced by none: 42.**

The 42 with no markdown mention:

```
capability_table_v1.md            r10_it17_battery.md
chase_k1_replay_it0.txt           r10_it18_scope_census.md
chase_k2_salience.json            r10_it19_priced_1c.jsonl
chase_m3_synth.jsonl              r10_it19_priced_1c.txt
chase_slope_ci.json               r10_it20_coherence.md
e4_harmonic_reroute.txt           r10_it20_verify_t8.log
eprocess_calibration.txt          r10_it3_result.md
foreman_consequence_d256_s258.txt r10_it8_sweep_t2.log
foreman_j3_repeat.txt             r10_it8_sweep_t32.log
m3_quintuple_v2_cuda.lock         r10_it8_sweep_t8.log
m3_synthetic_settled.txt          r10_it8_waveAC_t2.log
r10_it10_frontier.md              r10_it8_waveAC_t32.log
r10_it10_n49152.log               r10_it8_waveAC_t8.log
r10_it12_e4prime_spread.md        r10_it8_waveB2.log
r10_it13_saturn_binding.md        r10_it9_seedbind.log
r10_it15_dual_oracle.md           r10_it9_seeds47.log
                                  r10_rho_axis_run.log
                                  r10_v13_gatedhop_t2.txt
                                  r10_v13_hop2gain_t2.txt
                                  r2_bucket1.txt
                                  r2_trap.txt
                                  sprt_k1_plan.txt
                                  trained_projections_phaseD.txt
                                  trained_projections_seed1.txt
                                  trained_projections_seed2.txt
```

`results/r10_v13_b9_4060_probe.txt` — the zero-length file — **is** referenced
by a `.md`, so it is not in this list.

---

## 4. THE DEVICE FACTS

### 4.1 The machine

Command as specified, output verbatim:

```
$ python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
2.5.1+cu121 True NVIDIA GeForce RTX 4060 Laptop GPU
```

Extended probe, verbatim:

```
3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
2.5.1+cu121
12.1
True
1
NVIDIA GeForce RTX 4060 Laptop GPU
(8, 9)
```

`nvidia-smi`, verbatim (first lines):

```
Wed Sep  2 00:47:02 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.79                 Driver Version: 595.79         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   52C    P0             13W /  106W |       0MiB /   8188MiB |      0%      Default |
|  No running processes found                                                             |
```

### 4.2 What the repository claims

`V16_DEVICE_CERT.md` §1 "THE BOX":

> `| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, sm_89, 24 MPs | torch.cuda.get_device_properties(0) [MEASURED] |`
> `| VRAM total | 8,585,216,000 B = 7.996 GiB | torch.cuda.mem_get_info()[1] [MEASURED] |`
> `| torch | 2.5.1+cu121, Windows 11 | [MEASURED] |`

and §0.1 line 1: *"**The device is CUDA** — NVIDIA RTX 4060 Laptop, sm_89,
7.996 GiB — signed in §6.3."*

`COSTS.md:5-6`: *"the local RTX 4060 Laptop, which **decides** every research
number, and a Kaggle card, which **reproduces**"*. `COSTS.md:57-58`:
`| box | NVIDIA GeForce RTX 4060 Laptop GPU, sm_8.9, 7.996 GiB | [MEASURED] |`,
`| torch / CUDA | 2.5.1+cu121 | [MEASURED] |`. `COSTS.md:158`: the Kaggle
certificate slot is **"EMPTY, TO BE FILLED BY L2"**.

`V17_GPU_QUEUE.md` names the same card as "the certified 4060" and rules
`n=16,384` out against it: *"`n=16,384` does not fit the certified 4060
(10.578 GiB vs 7.996)"*.

### 4.3 Agreement

**The claim and the machine agree.** Model (`NVIDIA GeForce RTX 4060 Laptop
GPU`), compute capability (`(8, 9)` = sm_89), and torch build (`2.5.1+cu121`)
match the certificate exactly. `nvidia-smi` reports `8188 MiB` total against the
certificate's `7.996 GiB` (= 8188 MiB), also matching. The driver and its bundled
CUDA runtime version (`595.79` / `13.2`) are not stated in the certificate, so
no comparison is available for those two fields.

---

## 5. THE TEST SUITE, AS IT RUNS RIGHT NOW

`pytest.ini` supplies no `addopts`; it sets one marker and `norecursedirs`.
`pytest-timeout` is not configured, so the invocation used was
`python -m pytest -q` from the repository root.

**Verbatim summary line:**

```
!!!!!!!!!!!!!!!!!! Interrupted: 200 errors during collection !!!!!!!!!!!!!!!!!!
2 skipped, 15 warnings, 200 errors in 16.84s
```

**Pass count 0. Fail count 0. Error count 200. Skip count 2. Wall time 16.84 s.**

The run **did complete** — it was interrupted by pytest's own collection-error
ceiling, not by a timeout. **No test function executed.** Collection aborted
before any test ran, so there are no failing tests to name: every one of the 200
entries is a collection ERROR, not a test failure.

### 5.1 The cause, verbatim from the first error

```
_________________ ERROR collecting scale/b1_collapse_test.py __________________
import file mismatch:
imported module 'b1_collapse_test' has this __file__ attribute:
  C:\Users\seal\Desktop\New folder (32)\kaggle\snapshot\repo\scale\b1_collapse_test.py
which is not the same as the test file we want to collect:
  C:\Users\seal\Desktop\New folder (32)\scale\b1_collapse_test.py
HINT: remove __pycache__ / .pyc files and/or use a unique basename for your test file modules
```

Every one of the 200 errors has this identical shape, differing only in the file
named. The shadowing tree is `kaggle/snapshot/repo/`, which holds a full copy of
the repository including **208 `test_*.py` files**
(`find kaggle/snapshot -name "test_*.py" | wc -l` = 208).

That directory is gitignored — `.gitignore:46-49`:

```
# the code snapshot is REGENERATED by `git archive` at each push -- committing
# it would nest every snapshot inside the next one and double the tree each time
kaggle/snapshot/repo/
results/kaggle_v17k_output/
```

`git ls-files kaggle/snapshot` returns exactly one tracked path,
`kaggle/snapshot/dataset-metadata.json`. `pytest.ini`'s `norecursedirs` excludes
`attic` but **not** `kaggle`, so the untracked regenerated snapshot is collected
alongside the real tree and every basename collides.

### 5.2 Names of the erroring entries

All 200 are collection errors. The tail of the run lists, verbatim, among
others: `ERROR tests/loop/test_merkle_journal.py`,
`ERROR tests/loop/test_monge_oracle.py`,
`ERROR tests/loop/test_no_cross_device_pooling.py`,
`ERROR tests/loop/test_no_module_writes_a_file_at_import.py`,
`ERROR tests/loop/test_no_struck_constant_ships.py`,
`ERROR tests/loop/test_phase1a_modules_are_bound.py`,
`ERROR tests/loop/test_pivot_arms_distinct.py`, `ERROR tests/loop/test_settle.py`,
`ERROR tests/loop/test_the_activation_clause_prices_the_peak_tensor.py`,
`ERROR tests/loop/test_the_bar_control_is_scored_out_of_sample.py`,
`ERROR tests/loop/test_the_variance_law_is_stated_once.py`,
`ERROR tests/loop/test_two_dof_lemma.py`, `ERROR tests/mars/test_mars_argmax_ste.py`,
`ERROR tests/mars/test_mars_control_entry_point.py`,
`ERROR tests/mars/test_mars_green_attacks.py`,
`ERROR tests/mars/test_mars_green_attacks_it4.py`,
`ERROR tests/mars/test_mars_r9_iteration1.py`,
`ERROR tests/mars/test_p1_presumption_overfires.py`,
`ERROR tests/mars/test_spotcheck_draw_is_powered_against_clustered_error.py`,
`ERROR tests/mars/test_spotcheck_floor_covers_every_rule.py`,
`ERROR tests/mars_v15/test_guard_itinerary_timestamp_order.py`,
`ERROR tests/mars_v15/test_neumann_leak_reading_tensors.py`,
`ERROR tests/mars_v15/test_resolution_statement_achieved_power.py`,
`ERROR tests/mars_v15/test_skyline_gate_containment.py`,
`ERROR tests/mercury/test_r10_it19_pricing.py`,
`ERROR tests/mercury/test_r9_eprocess_overflow.py`,
`ERROR tests/mercury/test_r9_estimator_named.py`,
`ERROR tests/mercury/test_r9_exact_interval.py`,
`ERROR tests/mercury/test_r9_limits_e.py`,
`ERROR tests/mercury/test_r9_mechanical_fixes.py`,
`ERROR tests/mercury/test_r9_seed_agreement.py`,
`ERROR tests/mercury/test_r9_table_cut.py`,
`ERROR tests/mercury/test_r9_verdict_names.py`,
`ERROR tests/neptune/test_capability_table_truth.py`,
`ERROR tests/neptune/test_per_row_arm.py`,
`ERROR tests/saturn/test_census_does_not_attic_refutation_instruments.py`,
`ERROR tests/saturn/test_journal_path_is_discoverable.py`,
`ERROR tests/saturn/test_r10_it2_spotcheck_reds.py`,
`ERROR tests/saturn/test_r10_it3_doc_extractor_is_the_censuss.py`,
`ERROR tests/test_zero_step_gate.py`, `ERROR tests/w10/test_w10_from_scratch.py`,
`ERROR tests/w11/test_w11_claims_resolve.py`,
`ERROR tests/w13/test_w13_h1_selectivity.py`,
`ERROR tests/w14/test_w14_identity_double_count.py`,
`ERROR tests/w15/test_w15_rho_is_the_self_other_ratio.py`,
`ERROR tests/w2/test_w2_nonnormal.py`, `ERROR tests/w3/test_w3_eviction.py`,
`ERROR tests/w3b/test_w3b_lean_nilpotent.py`,
`ERROR tests/w4/test_w4_intervention.py`, `ERROR tests/w7/test_w7_nash.py`,
`ERROR tests/w8/test_w8_real_model.py`, `ERROR tests/w9/test_w9_hopcache.py`,
`ERROR tests/watson/test_vgpe_binds.py`, `ERROR tests/wilson/test_hop2_vec.py`,
`ERROR tests/x35/test_residual_onset.py`, `ERROR tests/x35p/test_kk_crb.py`,
`ERROR tests/x35p/test_source.py`.

The first 141 entries in the alphabetical listing are **unverified by name** —
pytest's tail printed only the last 59 of the 200 ERROR lines and the run was
not repeated with a wider capture.

**Nothing was fixed.** The suite is reported as it stands.

**Comparison point, for the record only:** commit `d8ad268`'s body states
`tests/loop at 15 failed and 515 passed` at the R11 loop's ceiling. That figure
is from a different tree state and is not reproduced here.

---

## 6. THE LOOP STATE

### 6.1 `.claude/ralph-loop.local.md`

**Does not exist.**

```
$ test -f ".claude/ralph-loop.local.md" && echo "EXISTS" || echo "DOES NOT EXIST"
DOES NOT EXIST
```

`ls -la .claude/` shows only `ralph-loop.R10.stopped.md`, `settings.local.json`,
and `worktrees/`. `git log --all --name-only -- ".claude/*ralph*"` shows
`.claude/ralph-loop.local.md` in ten historical commits and
`ralph-loop.R10.stopped.md` in **zero** — the R10 state file was never committed.

### 6.2 `.claude/ralph-loop.R10.stopped.md`, verbatim and complete

```
---
active: false
iteration: 600
session_id: 8df7537d-5a0f-4947-bdcd-5eeebfe72ef7
max_iterations: 0
completion_promise: "CEQ R10 COMPLETE: v-main.3M it.40 and v-main.4 Phase 2 it.40 both done"
started_at: "2026-08-30T11:33:41Z"
---

CEQ R10 — v-main.3M iterations 2-40, then v-main.4 Phase 2 iterations 1-40. Each iteration dispatches the named planets in parallel per PHASE2_CONTRACT_V_MAIN_4.md and the v-main.3M script, Inspector audits before each prognosis, one round record per iteration.
```

497 bytes, mtime `Aug 31 00:42`.

### 6.3 How the most recent autonomous loop ended

The most recent autonomous loop is the **R11 / v15 loop**. Its termination is
stated in `V15_LEDGER.md`, section "TERMINATION RECORD — for the next round's
D-3 gate", added by commit `d8ad268` (`Record the first CEQ loop to stop at its
own ceiling`, 2026-08-31 17:35:09 +0530), which touched that one file and added
36 lines.

The sentence the record itself nominates, **`V15_LEDGER.md:568-572`**:

> **The R11 loop reached its declared ceiling of 30 iterations and stopped
> there.** The cap was passed as the literal `--max-iterations 30` flag and read
> back out of the state file at mount rather than trusted from the request; prose
> forms of the cap fall through the launcher's catch-all and leave the sentinel
> `0`, which is what killed R10 and the loop before it.

And `V15_LEDGER.md:559-561`:

> **This loop terminated by reaching its ceiling.** `.claude/ralph-loop.local.md`
> read `iteration: 30`, `max_iterations: 30` at the stop. No hand-rename, no
> `rm`, no orphaned process.

The state file it refers to has since been deleted; only `ralph-loop.R10.stopped.md`
survives in `.claude/`.

### 6.4 The v16 / v17-K rounds

Rounds v16 (Round 12) and v17-K are the commits **after** `d8ad268`
(`9bae267` through `207e7b9`, 27 commits). Searches run:

- `git log d8ad268..HEAD --format="%h|%s" --grep="loop" -i` — 9 commits whose
  bodies mention "loop", none recording a loop mount or a loop end.
- `grep -rn -i "loop ended\|loop died\|loop stopped\|loop terminated\|loop halted\|stopped the loop\|killed the loop\|loop exited\|iteration budget\|max-iterations\|ralph-loop" --include=*.md .`
  — every hit outside `kaggle/snapshot/` is about R10, the R11 ceiling, or the
  launcher defect (`CONTRACT.md:40-58`, `V13_D3_LOOP_FORENSICS.md`,
  `V13_RALPH_LOOP_FIX.md`, `DONE.md:13822`, `done5.md:5`,
  `DONE_ARCHIVE_ROUND1.md:3774`).
- `ls .claude/ | grep -i "ralph\|loop"` — one file, the R10 one.

**No statement in the tree records a v16 or v17-K autonomous loop ending, and no
loop state file for either round exists — "unverified — no such statement
located" for those two rounds specifically.** The most recent loop for which a
termination statement exists is R11, quoted at 6.3.

For completeness, the R10 cause of death, `V13_D3_LOOP_FORENSICS.md:26-27`:

> Short form for a mount request: **the loop ran out of affordable work, not of
> iterations, and was killed by hand because it had no ceiling of its own.**

---

## 7. THE NUMBERS THE ANNEX CLAIMS AS `[RUN]`

The annex is `CEQ_V20_R15_CONTRACT.md`, "THE MATHEMATICS ANNEX (Jupiter owns)",
lines 170-254. It is **untracked** (`git status` shows `?? CEQ_V20_R15_CONTRACT.md`),
16,255 bytes, mtime `Sep  2 00:47`. Its header states: *"instances marked [RUN]
were executed this session"* (`CEQ_V20_R15_CONTRACT.md:172-173`). Sixteen items
M1-M16; `grep -n "\[RUN" ` returns 16 lines in the annex block.

Search performed per value: full-tree literal grep excluding `.git`, `.lake`,
`__pycache__`, `.pytest_cache`, `.benchmarks`, `kaggle/snapshot/`, and the annex
and journal themselves — then classified by whether the hit is in executable
code (`.py`, `.ipynb`), in a `results/` journal, or only in prose.

| item | value | verdict |
|---|---|---|
| M1 | `+20.87` | **NOT FOUND IN TREE.** Zero hits outside the annex. Bare `20.87` matches once, inside a `tau` array element `20.872072219848633` in `results/arm_a_k1.jsonl:10` — a different quantity. |
| M2 | `1.3e-3` | **NOT FOUND IN TREE.** Zero hits outside the annex; `1.3e-03` zero hits anywhere. |
| M4 | `1e-9` (identifiability) | **FOUND as a literal, not as this quantity.** `1e-9` occurs at `ceq/certs/topological.py:112` (`INTEGRALITY_TOL = 1e-9`), `:630`, `:633`; and as a prior round's cited figure at `CEQ_V15_CONTRACT.md:190` (`[RUN: recovered to 1e-9 from one bump]`) and `CEQ_V16_CONTRACT.md:180` (`gates are Jacobians [RUN 1e-9]`). No producer computing `a_i = Δz_i/δ` was located. |
| M5 | `0.050` (PPI delay anchor) | **FOUND as a literal, not as this quantity.** `ceq/bench.py:461` (`0.050781`), `CHECKLIST.md:1167` and `:1208` (paired sd `0.050146`/`0.050147`), `D1.md:410`, `DONE.md:339`. All are NRMSE spreads, not a Hankel R². |
| M5 | `0.865` (PPI power-law anchor) | **FOUND as a literal, not as this quantity.** `CHECKLIST.md:772`, `DONE.md:2960,3751,5569`, `done6.md:63,165` — all `0.865614`, a `kappa_emp_max` at β=0.9. |
| M5 | `1.000` (PPI AR(1) anchor) | **FOUND as a literal, not as this quantity.** Many hits, e.g. `.superpowers/.../mercury-report-it3.md:408` (`1.000933` NRMSE). No PPI producer. |
| M5 | `0.0375` (epistasis) | **FOUND, in a prior contract and one probe.** `CEQ_V15_CONTRACT.md:194` — `have nonzero Walsh degree-2 by Lean #9 [RUN: 0.0375 vs 0.0000 additive]`; echoed by `scripts/v15_x8_mobius_probe.py:314`, which **prints the contract's cited figure as a string** (`print("  Contract's cited figure for this cell is \`[RUN: 0.0375 vs 0.0000")`) rather than computing it. Lean `#9` has no source declaration (§2.3). |
| M8 | `1.000` (τ\*) | **FOUND as a literal, not as this quantity.** `tau*` as a symbol occurs at `ceq/nash.py:36-37` (`tau* = ||M||_2 / 4`) and `tests/deimos/test_deimos_r9_iteration1.py:295,314,321,340`; `safe_tau` is the producer of a τ\*. Nothing emits `τ* = 1.000` on a toy well. |
| M9-F0 | `8.9e-16` | **FOUND, with a live producer path.** `ceq/x35p/source.py:7` (`(a) h_hat = (I - A) r  [RUN 8.9e-16, two sources]`) and `:44`; asserted by `tests/x35p/test_source.py:6,32,161`; cited by `CEQ_V15_2_DELTA.md:18,23`. This is the X35-prime source solve's float64 rounding figure, **not** a zero-gate segmentation figure. |
| M9-F0 | `595` | **NOT FOUND as this quantity.** `595` matches `ceq/kdata.py:616` (`"size": "595 MB"`, a dataset size) and digit-substring hits such as `0.015595124842990749`. No `595×` speedup producer. |
| M9-F1 | `d=65` | **NOT FOUND IN TREE.** Zero hits outside the annex. |
| M9-F1 | `32x` | **FOUND as a literal, not as this quantity.** `DONE_ARCHIVE_ROUND1.md:1213` (`5.6x, 8.5x, 9.3x, 18x, 32x at s = 8..128`), `house-events-round1.jsonl:185` (`32x128` block shape). No Cantelli/Boole cutoff producer. |
| M9-F1′ | `d=20` | **NOT FOUND as this quantity.** Hits are seeds (`seed=20260825`, `seed=200 + i`) and the annex's own bed line `CEQ_V20_R15_CONTRACT.md:116`. No Azuma producer. |
| M10 | `4096` | **FOUND as a literal, not as this quantity.** Widely present as a draw count or dimension: `attic/scale/run_m2.py:19` (`return 16384 if s >= 1024 else 4096`), `ceq/sizing.py:76`, `ceq/compat.py:592`. No β₀ persistence-barcode producer. |
| M11 | `0.19` | **FOUND as a literal, not as this quantity.** `ceq/harness.py:130` (`arXiv:2310.19956`), `attic/workdonenew.pre-v13.md:193` (`0.1932` magnitude retained). No Fano-floor producer. |
| M11 | `0.22` | **FOUND as a literal, not as this quantity.** `ceq/hf/modeling_ceq.py:194` (`0.225 at 8x`), `ceq/hf_artifact/README.md:39` (`+0.226893`). No Fano-floor producer. |
| M12 | `0.25` | **FOUND as a literal, not as this quantity.** `attic/scale/dfloor_probe2.py:29` (`KEEP_FRAC = 0.25`), `ceq/certs/topological.py:530` (`T: float = 0.25`), `ceq/kdata.py:481`, `ceq/eviction.py:24`. No rate-distortion producer. |
| M12 | `0.004` | **FOUND as a literal, not as this quantity.** `.superpowers/.../mercury-report-it3.md:94` (`-0.004092`), `attic/workdonenew.pre-v13.md:189` (`+0.004176`). No `D(R) = σ²2^{−2R}` producer. |
| M13 | `0.492` (W1) | **FOUND as a literal, and it is a DIFFERENT quantity.** Every hit is `0.492188` — the fraction of drawn third-token interventions on which a trained non-negative arm puts a negative sign (`CHECKLIST.md:1179`, `DONE.md:2268,2393`, `HOUSE_BRIEF.md:115`, `house-events.jsonl:3319`, and producers `scale/foreman_consequence.py:12`, `scale/foreman_signfloor.py:382`). Not a Wasserstein distance. |
| M13 | `0.519` (KL) | **FOUND as a literal, and it is a DIFFERENT quantity.** `CEQ_V15_CONTRACT.md:126` (`GELU passed 0.519`, a dissipation budget), `BOARD.md:262` (`0.5193`), `DONE_ARCHIVE_ROUND1.md:5630` (`R²=0.519`), and many `house-events.jsonl` test durations. Not a KL divergence. |
| M13 | — | Quantity keyword search over `.py` in `ceq/ scale/ scripts/ tests/`: `wasserstein` **NO PY FILE**, `kantorovich` **NO PY FILE**. |
| M14 | (φ, Cheeger) | The annex itself records this one as failed: `CEQ_V20_R15_CONTRACT.md:241` — `[RUN: my crude φ FAILED the sanity check — grade F3 pending`. Keyword search finds `cheeger` in `scale/e4_harmonic.py`, `scale/e4_harmonic_reroute.py`, `scale/e_ladder.py` and `conductance` in `scale/e4_harmonic.py`, `scale/foreman_lambda2.py`, `scale/kirchhoff.py` — those modules exist; whether they produced the failed φ is **unverified**. |
| M15 | `0.49` / `3.97` / `5.99` | **FOUND as literals, none as χ²/LR-test null quantiles.** `0.49`: `attic/tests/mars/MARS_REPORT_IT4.md:77`, `CHECKLIST.md:591`. `3.97`: `CHECKLIST.md:510`, `COSTS.md:231`, `D1.md:95`, `DONE.md:6432,7410`, `done5.md:67` — all a `sqrt(TV)` percentage margin. `5.99`: no direct hit; nearest are `15.99` at `scale/e4_harmonic.py:82` and `scale/e4_harmonic_reroute.py:7`, and `25.991893` at `DONE.md:2876`. The χ²₁ 95% critical value `3.841` appears in the annex text only. |
| M15 | `0.055` | **FOUND as a literal, not as a false rate.** `attic/scale/r4b_units.py:5` (`max \|slope\| = 0.055`), `scale/m3_flops.py:106` (`0.055674`), `DONE.md:11593,11599,11601`, and many `house-events.jsonl` test durations. |
| M16 | `1/d for delays` | No numeric value to search. Keyword `hankel_bound` **NO PY FILE**; `ceq/hankel.py` exists (§1.1) but nothing in it is named for an AAK/Glover bound. |

### 7.1 Aggregate for section 7

Of the values enumerated in the question:

- **NOT FOUND IN TREE (zero hits outside the annex):** `+20.87`, `1.3e-3`,
  `d=65`.
- **FOUND with a producer that emits this exact quantity:** `8.9e-16`
  (`ceq/x35p/source.py:7,44`, `tests/x35p/test_source.py:161`) — though for the
  X35-prime source solve, not for M9's zero-gate segmentation. `0.0375` appears
  in `scripts/v15_x8_mobius_probe.py:314`, but that line **prints the contract's
  own string** rather than computing the figure.
- **FOUND as a literal string somewhere in the tree, attached to a different
  quantity:** `1e-9`, `0.050`, `0.865`, `1.000`, `595`, `32x`, `d=20`, `4096`,
  `0.19`, `0.22`, `0.25`, `0.004`, `0.492`, `0.519`, `3.97`, `0.055`.
- **`5.99`:** no exact hit; only `15.99` and `25.991893` as substrings.

Quantity-level keyword search over `.py` under `ceq/ scale/ scripts/ tests/`:
`wasserstein`, `kantorovich`, `fano`, `rate_distortion`, `union_bound`,
`cantelli`, `azuma`, `bonferroni`, `hankel_bound`, `proxy_index` all return
**NO PY FILE**. `PPI` case-sensitive returns 14 hits, all inside unrelated
uppercase words (`STOPPING`, `MAPPING`) — **no `PPI` identifier exists**.

This is a search result. No judgement is offered on what it means.

---

## UNVERIFIED

1. **§5 — the names of 141 of the 200 collection errors.** The captured tail
   printed only the last 59 `ERROR` lines and the head only the first ~9 error
   blocks. The run was not repeated with a wider capture, so 141 entries are
   named only by count.
2. **§6.4 — how the v16 (Round 12) and v17-K rounds ended.** No loop state file,
   no commit body, and no markdown statement records a loop mount or termination
   for either round. **Unverified — no such statement located.** The four
   searches run are listed in §6.4. Whether those rounds ran under an autonomous
   loop at all is also unverified.
3. **§2.2 — whether the `.olean` files correspond to the current sources.**
   `lake build` exits 0 with no output, which is consistent both with a fully
   current build and with a build lake considers up to date. A forced clean
   rebuild was not performed, so "the source as it stands compiles" is verified
   only to the extent that lake's own staleness check accepts it.
4. **§7 M14 — whether the named Cheeger/conductance modules produced the failed
   φ.** `scale/e4_harmonic.py`, `scale/e4_harmonic_reroute.py`,
   `scale/e_ladder.py`, `scale/foreman_lambda2.py` and `scale/kirchhoff.py`
   contain the keywords, but no run log tying them to the annex's F3 grade was
   located.
5. **§4.3 — driver and bundled CUDA runtime version against the certificate.**
   `nvidia-smi` reports driver `595.79` and CUDA `13.2`; neither
   `V16_DEVICE_CERT.md` nor `COSTS.md` states a driver version, so no comparison
   exists for those two fields.
6. **§3.2 — the semantic content of several `results/` files.** For binary
   `.pt` checkpoints and for several `.txt` logs, the first 500 bytes were read
   but no key schema exists to report. Those rows give size and line count only.
7. **§1.1 — `ceq/vendor/`.** The directory exists but contains no `.py` file in
   the `find` output; its contents are unverified.
