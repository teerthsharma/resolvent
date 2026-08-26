# THE ENGINEERING BOARD — nurse findings, for Wilson

**What this is.** The named agents (Foreman, Cameron, Chase) do theory: is the
mathematics right, is the claim novel, what kills it. **The nurses are
engineers.** Engineering in computer science is a different discipline from the
theoretical kind, and it decides whether any of the theory ever runs. This board
is where their findings collect.

**Who consumes it.** WILSON. He is the deterministic one — no stance, no angle —
and his job is to state what is verifiably true. The board is raw material for
that: every row must be checkable, or it does not belong here.

**The rule for every row:** a claim, a NUMBER, and a source (file:line, command,
or fetched URL). A row without a number is a suggestion, not a finding, and goes
in OPEN at the bottom.

**Evidence classes:** RUN (command + output) / READ (file:line) / CITED (URL or
arXiv id + quote) / DERIVED (steps shown from RUN or READ) / GUESS.

---

## 1. WINS — measured, ranked by payoff

| # | finding | number | class | source |
|---|---|---|---|---|
| **1** | **The M2 probe never needed the matrix it spends all its time building.** `grad[j].sum() == (delta_ij + A[i,j] + hop2[i,j]) * wo.sum()` exactly, so hop2 at one entry is an O(s) dot product, not an O(s^3) matmul. | dense s=2048 **735.7 -> 68.1 ms/draw**; single entry measured **2,643x** cheaper than `a@a`; entries agree to **1.788e-07** absolute; identity worst rel err **1.8e-04** (float32 accumulation) over 64 draws at s=32/128/512 | RUN | verified independently this session |
| **2** | **A streaming form EXISTS — the FlashAttention worry was backwards.** Online softmax's running max exists ONLY to compute a row-global denominator. This operator has none, so tiling needs no rescaling at all. | s=2048: dense `[s,s]` **16.00 MiB -> 1.00 MiB** peak, **16x**; max rel err **3.46e-07** | RUN | Chase |
| **3** | **The O(ks) decode argument is real and unbuilt.** At decode only one query row is live, so `A[i,P] @ A[P,:]` is O(ks) against dense O(s^2). Bytes touched scale exactly s/k. | **3.1x / 60.8x / 397x / 4946x** at s=512/1024/2048/4096. Fitted slope pivot **-0.16** (flat), dense **+3.31** (super-quadratic — falls out of cache) | RUN | Chase. **Biggest unmade claim the project has evidence for.** Nothing in the repo combines pivot selection with incremental caching; `hopcache.py` attends the whole prefix. |
| **4** | **Training checkpoint/resume works, bit-exact.** This was the stated blocker for every free-tier GPU plan, since all of them cap session length. | loss sequences identical to **0.0** across kill/resume; `1 passed in 77.10s` | RUN | Chase. Also hardened `torch.load` `weights_only=False -> True` (untrusted pickle). |

## 2. NEGATIVE RESULTS — measured, and they save time

| finding | number | class |
|---|---|---|
| **Do NOT write a custom kernel. The win is algebraic.** `a@a` is compute-bound (arithmetic intensity 85–341 vs ridge 14–26). The PIVOT form is intensity **3.9 FLOP/B** — dominated by an `[s,s]` write nothing reads; launch-bound, not occupancy-bound (0.635 ms vs 0.078 ms launch, 3 kernels). | materialization tax **121.8 MiB/draw at s=2048 = 4.35x L2, 3.69x L3** — thrashing | RUN |
| **TF32 is SLOWER here.** | **0.81x** at head_dim 64 | RUN |
| **`.contiguous()` in the hot path is a regression.** | **2.4–3.7x slower** | RUN |
| **Mask caching + bool-multiply**: bitwise identical, 23% faster at s=1024, **a wash at s=2048** (bandwidth-bound). | 23% / ~0% | RUN |
| **The SDPA baseline is FAIR** (`ceq/lm.py:176`), so the `COSTS` ratios stand — the comparison is against a tuned baseline, which C6 demands. | — | READ |

## 3. PRIOR-ART HITS FROM THE SCAN — these cost claims, not time

| finding | why it matters | class |
|---|---|---|
| **THE TERMINATING RESOLVENT IS ALREADY SHIPPED**, three weeks before this project. `sgl-project/sglang`, `python/sglang/kernels/ops/attention/linear/kda_nvidia_prefill/Akk_inverse_lower_triangle_bf16.py:12,224-229`, **NVIDIA-authored, vendored**: `(I+L)^-1 = (I-L)(I+L^2)(I+L^4)(I+L^8)`, `[L strictly lower triangular, L^16 = 0]`. | **M5's novelty claim needs revising BEFORE the HuggingFace release.** | READ |
| The NARROWER claim survives: `fla-org/flash-linear-attention` genuinely has **0 hits** for `nilpotent` / `Neumann`; its `solve_tril.py` uses forward substitution + Schur merge, **not a series**. | So "nobody proves the finite-termination property" is defensible; "nobody uses the terminating resolvent" is not. | READ |

## 4. DEVICE CATALOGUE — what production stacks do to self-attention

**The question this answers:** nobody ships `softmax(QK^T/sqrt(d))V`. That is the
textbook object, and this operator is currently at exactly that stage. Which
production devices can it INHERIT, which need ADAPTING, and which are
STRUCTURALLY IMPOSSIBLE for a signed multi-hop path sum?

**Nurse batch 1 — 7 devices, all with file:line from live fetches** of
`vllm-project/vllm`, `sgl-project/sglang`, `fla-org/flash-linear-attention`,
`huggingface/transformers`:

| verdict | count | devices |
|---|---|---|
| **INHERIT** | 2 | sliding window + sinks (plain form); RoPE (plain) |
| **ADAPT** | 5 | KV quantization; sink SEMANTICS; YaRN; speculative-decode verify; batch-invariant/deterministic kernels |
| **STRUCTURALLY IMPOSSIBLE** | 1 | chunked linear-attention streaming, **in fla's specific chunk-recurrent form** |

**Note the tension with WIN #2** and resolve it before publishing either: Chase
measured a 16x streaming reduction for OUR operator, while the nurse found fla's
chunk-recurrent streaming impossible for it. Those are different constructions
and may both be true — **but nobody has said so in one place with both numbers.**
Wilson should not accept either until they are reconciled.

**Already in our operator by accident, and worth documenting as convergent design
rather than claiming as novelty:** `tanh(qhat . khat / tau)` L2-normalizes q and
k — that is **QK-norm**. And `tanh` is a **logit softcap**, which is what Gemma
added. The field arrived at both independently.

**Nurse batch 2 — 6 more devices**, file:line from `vllm-project/vllm`
(`triton_attention_helpers.py`, `paged_attn.py`, `block_pool.py`,
`chunked_prefill_paged_decode.py`, `cuda_graph.py`, `dcp.py`, `cp_common.py`)
against this project's `ceq/hopcache.py`, `lean/CEQ/Refcount.lean`,
`scale/pivot_probe.py`.

**IT CORRECTED MY OWN PREMISE, independently of Chase, and it matters:**
`ceq/bench.py:257-313` (`_causal_tgate_operator`) is the denominator-free
operator M2 measures. **`ceq/attention.py:151-189` (`ceq_operator`) — the one
actually wired into `register()` — is L1-NORMALIZED.** Different objects. Two
agents reached this from different directions, so it is not a reading error:
**the M-programme measures an operator that is not the one the module registers.**

**Its sharpest device finding — chunked prefill vs pivot selection:**
`select_pivots` (`scale/pivot_probe.py:80-91`) takes a **global, non-causal
top-k** over key norms. Chunked prefill does not have the whole sequence when a
chunk is processed, so the selector as coded cannot run incrementally. The nurse
marked it STRUCTURALLY IMPOSSIBLE and noted **no fix exists in either fetched
reference repo.**

**I am DOWNGRADING that verdict to ADAPT, and the distinction is the point:**
*not found in the reference repos* is not the same claim as *impossible*. A
prefix-causal top-k — select pivots from the prefix only, maintained as a running
heap across chunks — is a design change with a cost, not a contradiction. It also
COSTS something real and that must be priced: prefix-only selection sees less
content, so pivot quality at position i is strictly worse than the global
selector M2 measured. **That means M2's numbers are an UPPER BOUND on what a
chunked-prefill deployment would get**, and nobody has measured the gap.
Whoever picks this up owns that measurement.

**Its second finding confirms a divergence already on record:** this project's
own tests document BY NAME that the `foliation` / `CEQ.Refcount` proof is not
wired to any shipped refcount-tracking code
(`tests/chase/test_lean_refcount_binding.py:318-362`). Lean's `IsFreeFace` is
`refcount f p = 1`; `HopCache.evict` admits on SCOPE DEPTH; **no refcount ships
anywhere in the Python.** The theorems certify the criterion, not the
implementation.

**STILL OUT:** the nurse on the 847-test suite that has never run to completion.

### 4b. RECONCILIATION — why both streaming findings hold

Both findings stand; they are statements about two different operators, and the
tension above was an ambiguity in the words "our operator."

**The operator WIN #2 streamed is denominator-free.** Chase's 16x peak-memory
reduction at s=2048 (16.00 MiB -> 1.00 MiB, RUN, §1 row 2) is a measurement of
M2's operator `_causal_tgate_operator`, whose every entry is
`g_i * tanh((qhat_i . khat_j)/tau)` for j < i — a pointwise function of (i, j)
with no row sum anywhere (READ, `ceq/bench.py:257-313`; its own docstring:
"UNNORMALIZED ... WHY NO DENOMINATOR", `ceq/bench.py:260,278-283`). Online
softmax's running max exists only to feed a row-global denominator; this
operator has none to feed, so a tile boundary changes no entry and tiling needs
no rescaling pass (DERIVED from the READ source: pointwise-in-(i,j) implies
tile-decomposable).

**The operator the module registers is L1-normalized — a different object.**
`ceq_operator` divides each row by its L1 norm (`ceq/attention.py:188-189`) and
is what `register()` wires in via `ceq_attention` (call site
`ceq/attention.py:249`; registration `ceq/attention.py:266-271`). Every output
entry there depends on a sum over the whole prefix, so a tiled form must carry
running row mass across tiles — exactly the dependency online softmax's running
max exists to serve (DERIVED from READ, `ceq/attention.py:151-189`).

**fla's impossibility targets carry-state recurrence, which neither tiling
needs.** The nurse's STRUCTURALLY IMPOSSIBLE verdict addresses fla's
chunk-recurrent linear-attention streaming, where each chunk consumes a carry
matrix produced by the previous chunk (READ; live fetches recorded in §4
above). A signed multi-hop path sum has no carry to recur through chunks: hop-h
mass routes through intermediate tokens explicitly, so chunk-recurrent state
would have to be shown to compose under signed paths — that is the structural
obstruction the nurse named. WIN #2's tiling forms no carry at all; it needs
only tile-local entry computation, which `_causal_tgate_operator` satisfies and
which a chunk-recurrent form never asks about. Different constructions, no
contradiction (DERIVED).

**One sentence:** streaming is blocked where per-entry output requires
whole-prefix state (L1-normalized rows, recurrent carries) and free where
entries are pointwise in (i, j) — M2's operator is the second kind; the
registered operator and fla's chunks are the first.

**Reconciliation is not closure. Still unmeasured:** (i) the prefix-causal
pivot-selection gap — `select_pivots` is a global, non-causal top-k
(`scale/pivot_probe.py:80-91`, READ), so every M2 number remains an upper bound
for a chunked deployment and no number exists for the gap (OPEN, §5 below);
(ii) no streaming measurement exists for the REGISTERED `ceq_operator` itself —
the 16x belongs to `_causal_tgate_operator`, and quoting it for the module
would repeat exactly the operator swap this section untangles (DERIVED from
READ, `ceq/bench.py:257-313` vs `ceq/attention.py:151-189`).

## 5. OPEN — no number yet, so not a finding

- `requirements.txt` written but not validated on a clean environment.
- CI path-gating proposed, not measured.
- The 847-test suite: **nine attempts across three agents, never completed.**
  `--collect-only` = 847 in 49.56 s is the only whole-suite number that exists.
  **No total pass/fail count exists for this repo.**
- **BOTH scan nurses' relays bounced** (`SendMessage` could not resolve
  `general-purpose` — that is an agent TYPE, not a callable address, and the
  nurses correctly refused to guess a ref rather than fabricate one). Their
  findings were routed here by hand. **Any nurse result that lives only in a
  transcript is invisible to Wilson.** Process fix, now standing: nurses write
  findings into `BOARD.md` DIRECTLY and do not depend on the relay.
- **The chunked-prefill gap is unmeasured.** Prefix-causal pivot selection is
  strictly weaker than the global selector M2 uses, so every M2 number is an
  upper bound for a chunked deployment. No number exists for the gap.
- **Capability table v0 stale, v1 fresh.** `results/capability_table_v0.{json,md}`
  are frozen evidence stamped `journal_commit 9629616` with three clauses of
  their own Limits string now false (STATE.md item 39) — cite them as historical
  only. Current build on disk: `results/capability_table_v1.{json,md}`,
  regenerated from the journals at HEAD by the pure post-processing generator
  (`python -m scale.capability_table`; it trains nothing and uploads nothing,
  `scale/capability_table.py:6-8`) with outputs repointed to v1, because
  OUT_MD/OUT_JSON hardcode the v0 paths (`scale/capability_table.py:94-95`) and
  the plain command would clobber committed evidence. v1 provenance stamp:
  `journal_commit b8a9ace` = `head_commit`. Class RUN.
- **The table generator would clobber the weight-manifest sync (commit 0162bdd)
  on the next cut.** `render()` still prints *"This package carries NO trained
  weights."* into every generated card (`scale/capability_table.py:338-342`) and
  `write_artifact` writes that render straight over `ceq/hf_artifact/README.md`
  (`scale/capability_table.py:443`), which since 0162bdd carries a weights
  paragraph plus a "Weights shipped with this package" section describing the five
  verified safetensors — regenerating with `--artifact ceq/hf_artifact` would
  restore a false absence claim over a true manifest-backed one. The generator's
  template needs updating BEFORE the next table cut. Numbers: template absence
  sentence at :338-342; README write at :443; 5 safetensors shipped, MANIFEST
  `passed 5 / failed 0 / excluded []`; 0162bdd diff = +44 README lines, +118
  MANIFEST lines. Class READ (sources: `scale/capability_table.py:338-342,:94-95,:443`;
  `ceq/hf_artifact/weights/MANIFEST.json`; `git show 0162bdd --stat`).
- **The GPU-lane tolerance policy has no standing home.** The lane's rules — CUDA
  rows tolerance-checked never bitwise, atol=2e-4 / rtol=1e-4, separate journal
  `m3_quintuple_v2_cuda.jsonl` never appended to from the CPU lane, CUDA
  reproducibility via `cudnn.deterministic` while thread pinning stays CPU-only —
  live ONLY in the `tests/gpu/test_cuda_parity.py` docstring (:3-15) and its
  ATOL/RTOL constants (:39). A search for cuda/tolerance/atol/rtol/bitwise across
  `METHODS.md` returns **0 hits**, as does `inspector.py`. Before the CUDA lane
  produces quotable numbers, the policy needs a standing home in `METHODS.md` or
  the inspector docs. Numbers: 2 hit locations in one test file vs 0 everywhere
  else; bar atol 2e-4 / rtol 1e-4; parity battery 15 passed in 3.10 s this session.
  Class RUN + READ (`python -m pytest tests/gpu/test_cuda_parity.py -q`;
  Select-String over METHODS.md/inspector.py).
- **Single-seed per-cell table crash in `scale/m3_quintuple.py`, data safely
  banked.** The per-cell summary divides by `len(ev) - 1`
  (`scale/m3_quintuple.py:737`), so requesting ONE seed per cell raises
  ZeroDivisionError — but only AFTER `run_bucket` has journalled every unit and
  `require_complete` has read them back (:715-725), so the loss is cosmetic
  reporting, never data. No RED test exists yet:
  `tests/chase/test_m3_quintuple_single_seed_table.py` is NOT in the tree as of
  this writing, so NO "fix in flight" is claimed — defect stands open exactly as
  described. Number: division by `len(ev) - 1 == 0` at :737; journal-first order
  at :715-725. Class READ (glob for the test filename: 0 hits).

## 6. PHASE R8 ITERATION 0 — CAMERON READING START (X₁₇ / E4′ Rips corpus, discrete Dirichlet)

**Scope:** iteration 0 of 8 (RULE 2 clock started 2026-08-26). Every row of the ladder must print measured λ₂ (exact eig, n≤1024, stratified [0.90,0.95]) and δ̂ (Gromov four-point, factor-1 product form), with Cheeger as design intuition only. Re-draw not re-label if λ₂ misses. Gates bind with planted must-fires seen firing on identical instances/features/split; every zero carries non-degeneracy (sd>0, 0<frac<1, both classes nonempty, discard=0).

### 6a. Measured λ₂ and δ̂ per instance (exact, per-row covariates)

Generated `results/covariate_table_phaseR8_iter0.json` (RUN, `scale/hyperbolic.py:112 hop_distances`, `scale/hyperbolic.py:205 gromov_delta`, `scale/foreman_lambda2.py:278 lambda_2` via `np.linalg.eigvals` on Q, `scale/foreman_lambda2.py:317 absorption_probabilities` via `solve`, `ceq/rips.py:87 rips_edges`, `ceq/rips.py:175 _add_critical_bridge(join="largest")`, `scale/foreman_lambda2.py:139 merged_component`, `scale/foreman_lambda2.py:214 diameter`).

| spec (REROUTED_CASES) | n | merged | bridge | λ₂ natural (eig) | λ₂ tuned α·ρ (eig re-measured) | t_rel=1/(1-λ₂) | in [0.90,0.95] | δ̂ | diam | source |
|---|---|---|---|---|---|---|---|---|---|
| LargestJoin_S2Rips_64 | 64 | 18 | (21,55) 8×10 | **0.9789623187** | **0.9250000000** (α=0.9448780432) | **13.33** | **YES (tuned) / NO (natural)** | **0.500000** | 8 | RUN `covariate.py` + `scale/foreman_lambda2.py:278` + `scale/hyperbolic.py:205` |
| LargestJoin_S2Rips_1024 | 1024 | 62 | (343,354) 30×32 | **0.9984623637** | **0.9250000000** (α=0.9264245040) | **13.33** | **YES (tuned) / NO (natural)** | **1.000000** | 29 | RUN `covariate.py` + same |

- **λ₂ is exact per instance via eig (n≤1024, Q=16×16 and 60×60).** Construction and re-measured values agree to 1.3e-15 (1024) and 0.0 (64) (`scale/foreman_lambda2.py:645 engineer` asserts |lam-construction|<1e-12). Every natural λ₂ misses the band (0.9789 and 0.9984 >0.95); tuned via α=target/ρ lands exactly 0.9250000000 with t_rel 13.33 inside 10–20. **Cheeger intuition:** conductance of single-bridge cut Φ=1/vol(S) gives t_rel ≥ vol/2 = 47.53 (64) and 650.35 (1024) at natural, explaining why band unreachable without tuning (READ `scale/e4_harmonic.py:246 cheeger_t_rel_floor`, `scale/foreman_lambda2.py:300 bridge_conductance`).
- **δ̂ is Gromov four-point product form (*) `max[min((x|z)_w,(z|y)_w)-(x|y)_w]` with x≠y mask (`scale/hyperbolic.py:16-18,51`).** Measured 0.500000 (18 nodes, diam 8) and 1.000000 (62 nodes, diam 29). Both satisfy 0 ≤ δ̂ ≤ diam and are >0 and ≠ diam, so not degenerate.

**Controls (must-fires SEEN FIRING, non-degenerate, file:line):**

- **Tree δ̂=0 (`scale/hyperbolic.py:30-34`, `tests/foreman/test_gromov_delta.py:143 test_random_trees_read_delta_exactly_zero`):** random recursive trees n=8,16,33,65 all read **0.000000 exactly** (`==` not tolerance); paired with same-size cycle positive (n=32 tree 0.000000 vs cycle 8.000000) so zero is discriminating not degenerate (`tests/foreman/test_gromov_delta.py:156 test_tree_zero_is_discriminating_not_degenerate`). RUN this iter: n=8 tree 0.0, n=16 tree 0.0, n=32 tree 0.0.
- **Cycle δ̂≈girth/4 (`scale/hyperbolic.py:59-66`, `tests/foreman/test_gromov_delta.py:170 test_cycles_read_girth_over_four`):** C_n 4|n reads n/4 exactly (multiples of 1/2, float32 lossless ≤2^23 halves). RUN this iter: n=8 →2.000000 (expect 2.0), n=16 →4.000000 (expect 4.0), n=32 →8.000000 (expect 8.0), |err|≤1e-9. Odd C_13 sanity 0<δ̂≤6.5, complete graph K_n reads 0.000000 exactly. **If either fails, K-R8c fires and all δ̂ numbers VOID** (`tests/foreman/test_gromov_delta.py:23`). Both fired, so K-R8c does **NOT** fire.
- **λ₂/δ̂ on same vertex set:** both covariates use `merged_component` (`scale/foreman_lambda2.py:139`) and disconnected check refuses inf (`scale/hyperbolic.py:143`), so `results/covariate_table_phaseR8_iter0.json:delta_hyperbolicity` and `lambda_2` describe same induced subgraph (READ `scale/hyperbolic.py:88`).

### 6b. Gates — both must bind with planted must-fires SEEN FIRING (E4′ reroute, LargestJoin)

**Sources:** `ceq/rips.py:175 _add_critical_bridge(join="largest")`, `scale/rips_gate.py:60 PASS_BAR=0.5 FAIL_BAR=0.9`, `scale/rips_gate.py:255 truncation_ladder`, `scale/rips_gate.py:142 local_features` (symmetric min/max, no component count), `scale/rips_gate.py:162 fit_eval` (first-half train, second-half held-out NRMSE, std=0.5 balanced), `scale/rips_gate.py:177 planted_degree_labels`, `scale/negation_scope.py:689-783 make_e4prime_batch/e4prime_oracle/e4prime_hop_reading` (executable oracle rebuilt from CH_COORD+CH_BRIDGE+CH_QA/CH_QB, no answer key), `tests/cameron/test_e4prime_registration.py:148 test_gate_a_truncation_fires` etc., `results/e4prime_gates.txt:RUN` (reproduced this iter), `results/covariate_table_phaseR8_iter0.json`.

**Truncation gate (existing, file:line `scale/rips_gate.py:255`, `scale/e4_harmonic.py:329-347`, `scale/negation_scope.py:825 e4prime_hop_reading`):**
- E4′ 1024 (target n): k=0 **1.4142** (=√2, balanced label), k=1 **1.4128**, k=2 **1.4087**, k=4 **1.3947**, k=8 **1.2694**, k=16 **0.7943**, k=32 **0.0000** (exact at full budget), monotone decreasing (each step ≥ next −1e-9), k=1 >1.0 (bounded away), so no part legible at 1 hop — not a static task. Same at n=64: 1.4142→1.4031→1.3571→1.0411→0.0000. Verdict: **PASS, seen firing.** Re-drawn do()-paired instances (30×32 merge, bridge (343,354)) the same truncation holds via `e4prime_hop_reading` k-hop reachability: k=0 ≥1.0 through k=8, >0.75 at k=16, 0.0 at full budget (READ `tests/cameron/test_e4prime_registration.py:215`).
- **K-1 per-rung (PASS_BAR):** rung 1 label has sd>0 and 0<nonzero<1 (non-degeneracy) but trades at decoder 0.000002 — struck as standalone task, kept as NULL rung / PASS-half witness (READ `scale/e4_harmonic.py:365`).

**Decoder gate (planted LOCAL probe must FAIL):** static local-degree / one-hop predictor vs harmonic label, held-out half, identical instances/features/split.

| spec | decoder r0 (degree-only) | r1 | r2 | r3 | r5 | planted sum (same rows) | planted median (balanced) | gap | non-degeneracy | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| LargestJoin_S2Rips_64 (leak demo, NOT admissible) | **1.0019** (≥0.9 FAIL) | 0.9130 | 0.5474 | **0.2185** (<0.5 PASS) | **0.0055** (<0.5 PASS) | **2.9e-08** (<1e-6) | **0.4826** (<0.70) | 0.5193 | y01 frac 0<frac<1, median sd>0, both classes nonempty | **FAIL at r3/r5 — leaks, reproducing 0.0055 at n=64 (results/e4_gate.txt:44, results/e4prime_gates.txt). Builder refuses s<1024 (ValueError, `scale/negation_scope.py:745`).** |
| **LargestJoin_S2Rips_1024 (target n, admissible)** | **1.0001** (≥0.9 FAIL) | 1.0018 (≥0.9 FAIL) | 1.0035 (≥0.9 FAIL) | **0.9951** (≥0.9 FAIL) | **0.8220** (>0.75, >FAIL at r3) | **2.04e-08** (<1e-6) | **0.5530** (<0.70) | **0.4471** (>0.30) | y01 mean 0.5 exactly (balanced do-paired, 1024 each), y std=0.5, planted median sd>0, 0<median mean<1, connectivity frac 0<1 | **PASS: local probe collapses to mean predictor out to radius 3 (and >0.75 at r5); planted controls fire on IDENTICAL instances/features/split, gap >0.30, so absence is not a broken decoder. Reproduces prior E4 lesson 0.0055→0.9951 (n=64 PASS vs n=1024 FAIL). Source: `scale/rips_gate.py:195 planted_degree_labels`, `tests/cameron/test_e4prime_registration.py:171`, `results/e4prime_gates.txt:47-53`, RUN `covariate.py` table above.** |

- **Leak check at target n (n=1024):** degree-only **1.0001** and +3hop **0.9951** both ≥0.9, so leak **closed**. At n=64 +5hop **0.0055** (<0.9) leak **open**. Previous E4 lesson reproduced **0.0055 vs 0.9951**.

**δ̂ controls for gate section:** tree 0.000000 and cycle 8.000000 (n=32) both seen firing this iter, with brute-force cross-reference to literal definition within 1e-12 on n=9-11 (`tests/foreman/test_gromov_delta.py:129`), so δ̂ numbers not VOID per K-R8c. If either failed, all δ̂ VOID.

**Every zero carries CP interval + discard count discipline:** the fourteenth vacuous control (SupercriticalDense single component, constant label sd=0 NRMSE nan) is struck and every current zero is paired with non-degeneracy sd>0, 0<frac<1, discard 0 (balanced draw, no discard). Decoder NRMSE=1.0 is predict-the-mean by definition `nrmse(y.mean(),y)` (`scale/negation_scope.py:1060`, `scale/rips_gate.py:118`), so ≥0.9 is "doing nothing" calibrated.

**N3 discipline:** ω_v = row v of (I−P_II)⁻¹P_IB is exact Poisson kernel (`scale/foreman_lambda2.py:317 solve`, `scale/e4_harmonic.py:190 fixed_point` via `np.linalg.solve` on I−Q, `scale/negation_scope.py:783 e4prime_oracle` rebuilds graph from tensor); displacement probe cross-checked via rank correlation (reported separately), bar pre-registered `PASS_BAR=0.5 FAIL_BAR=0.9` inherited unchanged.

**RULE N1 curvature ledger:** Dirichlet averaging/settling solves linear system on ℝ (Hilbert/CAT0) with uniqueness via contraction / invertible I−Q; +curvature sphere averaging (Karcher mean on S², 48.3% non-unique at k=128, `scale/foreman_lambda2.py` header refs) retired and not used here (this corpus is killed random walk / harmonic, not spherical).

### 6c. X₁₇ build status (E4′ Rips discrete Dirichlet corpus)

- **Oracle:** discrete Dirichlet u(v)= (1/deg v) Σ_{w~v} u(w) interior, u|_B=g, solved as u_I=(I−P_II)⁻¹P_IB g (`scale/foreman_lambda2.py:317`, `scale/e4_harmonic.py:190`, `scale/negation_scope.py:689` header). Provenance: `ceq/rips.py:87` sampler (SplitMix64 exact), `ceq/rips.py:175` largest-join bridge, Frobenius not needed. n≤1024 exact eig per instance (Q 60×60 at n=1024, 16×16 at n=64) before any training.
- **Ladder extension:** every row now prints measured λ₂ and δ̂ via covariate table `results/covariate_table_phaseR8_iter0.json` → ready to render as λ₂×δ̂×capability columns per instance (scaffold). `scale/e4_harmonic.py:282 measure` + `scale/hyperbolic.py:205` to be wired into per-row print (next iter: extend `report()` to emit table line).
- **M3_TASKS registration:** `scale/negation_scope.py:1028 M3_TASKS["e4prime"]` (4-tuple batch/oracle/features/flipper_dependence 0.0, `tests/cameron/test_e4prime_registration.py:101` GREEN) + double-registration scaffold U1 RAG `rag_multihop_t{1,2,8,32}` (`scale/negation_scope.py:1053`, `tests/cameron/test_u1_rag_registration.py`, bitwise deterministic `torch.equal` on x and y, document-graph naming via CH_DOC, measurement-only until R10 per §U) — scaffolding present, gates not yet claimed as capability.
- **Covariate table scaffold:** `results/covariate_table_phaseR8_iter0.json` with rows above (λ₂ natural/tuned, α, t_rel, δ̂, diam, decoder r0/r3/r5, planted controls, truncation ks). Next: render `results/covariate_table.md` with capability columns once har­monic label trained.

### 6d. X₂₂ wave arm heterogeneous plant (amendment v10.4)

- **Status:** plant **NOT YET SUPPLIED** this iter (one-action cap). Required: heterogeneous graph instance where Dirichlet-energy trace (X₂₁ instrument) will be measured, needed for P3 collapse-resistance. Due iter ≤8. Next action: supply graph with mixed degree / weighted conductance heterogeneity (e.g., two-block planted partition or degree-heterogeneous Rips) and show energy trace instrument pass on it.

### 6e. U2/U3a gate batteries preparation (scaffolding only)

- **U2 mujoco contact-graph DSU n≥1024** and **U3 Tonnetz lattice graphs** — scaffolding code exists as stubs (no M3_TASKS admission). **K-U3 not triggered:** no U3 measurement claim made before reading completes (per §U, scaffolding in R8 allowed, admission only after both gates + planted controls pass). Will build batteries next iters but remain measurement-only until R10.

### 6f. Kills flagged and replacement routes (RULE 5)

| kill | fires? | evidence | replacement route |
|---|---|---|---|
| **K-R8a λ₂ stratification (re-draw not re-label)** | **FIRES (conditional)** | Natural λ₂ 0.9789623187 (64) and 0.9984623637 (1024) both **outside [0.90,0.95]**; band hit only via α-scaling `rho(αP_TT)=α·rho(P_TT)` exact (`scale/foreman_lambda2.py:68-76,631 killing_rate`). Under strict "re-draw not re-label" this is re-labelling (tuning R without new graph) — spec violation if counted as natural stratification. | **REROUTE (measured):** generate fresh Rips draws varying seed / target_degree toward percolation threshold (degree is corpus dial, not search) and **re-draw until measured λ₂∈[0.90,0.95] via eig** without α. Cheeger only design intuition (Φ≤1/vol). Keep α path as **REPRICE** reference (cost: label sd/mean degeneracy vs band) for comparison column. Next iter: implement draw loop with discard count + CP interval on zero-λ₂ cases. |
| K-R8b truncation gate | NO | k=1 >1.0, monotone, exact at k=32 on both specs, so ladder requires iteration. |
| **K-R8c δ̂ controls** | **NO — controls FIRED** | Tree 0.000000 and cycle 8.000000 (n=32) both hit, brute-force 1e-12, so δ̂ numbers valid. If had failed, all δ̂ VOID. |
| K-R8d decoder leak at target n | NO | 1024 holds at 1.0001/0.9951; leak correctly only at 64 (0.0055). |
| K-U3 (U3 before reading) | NO | No U3 admission; scaffolding only. Flag would fire if Tonnetz admitted to M3_TASKS before gates+controls. |
| RULE N1 (+curvature retired) | NO | Using ≤0 curvature Hilbert linear solve; no sphere averaging. |

### 6g. Limits, collected once (this iter)

- n≤1024 exact eig via `np.linalg.eigvals` on Q (60×60); n=4096 case (not in covariate table) would need iterative eig or power method — not used here.
- λ₂ band hit via α-tuning exact to 0.9250000000; natural stratification via re-draw not yet demonstrated — covariate table shows both natural and tuned columns so replacement is measurable.
- δ̂ exact mode O(n⁴) cost; 62 nodes <1s on CPU; sampled mode monotone lower bound not needed here.
- Decoder bars PASS=0.5 FAIL=0.9 inherited unchanged (`scale/rips_gate.py:60`); planted median control not against PASS_BAR (step-function linear floor 0.5530) but against 0.70 + gap >0.30, so absence not vacuous.
- Heterogeneous X₂₂ plant and U2/U3 batteries not yet supplied — due by iter 8 per scope; no claim made about them.

**Scoreboard pending:** X₁₇ ladder extension (covariate columns) scaffolded, gates measured, δ̂ controls firing; full X₁₇ +3 requires training capability column and re-draw stratification demo — not claimed this iter.
