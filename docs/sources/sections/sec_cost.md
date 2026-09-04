# 4.x The cost law and the kernel path

Evidence classes on every load-bearing claim: `RUN` (executed this session on the certified box, 2026-09-03, torch 2.5.1+cu121, RTX 4060 Laptop, HEAD `207e7b9`), `READ path:line` (quoted at HEAD `207e7b9`), `CITED [V]/[U]`, `DERIVED` (steps shown). Cost tags follow `COSTS.md:16-20`: `[MEASURED]`, `[FITTED]` with its R², `[MODULE]`, `[INHERITED]`, `[ASSUMED]`. The FLOP convention is the house one: one multiply-add (MAC) = 2 FLOPs, a matmul `[m,p]@[p,q]` costs `2mpq` FLOPs (READ `scale/m3_flops.py:41`); every count below is stated in MACs first and FLOPs second so that no OPS/HOUSE mix of the kind `V12_PRICING.md:31-39` catalogues can enter.

The object priced is the read `O = P (I − γP)^{-1} V` with `P` causal (lower-triangular including the diagonal), row-stochastic, absorbing rows equal to identity rows, `γ ∈ [0,1)`, per head, one sequence of length `s`, head width `d`. The design constraint that governs every cost below: the shape at `γ = 0` is `PV` bitwise (identity I1, RUN by the coordinator), so **every cost is stated as an increment over the softmax head it contains**, never as a stand-alone figure. That is the M-8 rule (READ `MISTAKES.md:575-578`: price each arm at its own measured rate; never carry a cross-arm ratio across a task, a geometry or an `n`).

## 4.x.1 The serial law: forward substitution against QKᵀ and PV (DERIVED)

`M = I − γP` is lower-triangular with diagonal `1 − γP_ii ≥ 1 − γ > 0`, so `z = M^{-1}V` is a forward substitution, row by row:

```
z_i = ( v_i + γ Σ_{j<i} P_ij z_j ) / (1 − γ P_ii),        i = 0 … s−1          (C1)
```

Per head, per sequence, the MAC count of (C1) is `d · Σ_{i} i = d·s(s−1)/2` for the inner sums plus `s·d` reciprocal-multiplies for the diagonal:

| term | MACs per head | FLOPs (house) | depth (sequential rounds) |
|---|---|---|---|
| `QKᵀ`, causal lower triangle only (a fused kernel skips the upper triangle, READ `ceq/sizing.py:153-157`) | `d·s(s+1)/2 ≈ s²d/2` | `≈ s²d` | 1 |
| `QKᵀ`, dense as `FlopCounterMode` counts it (READ `ceq/sizing.py:149-157`) | `s²d` | `2s²d` | 1 |
| `PV` (softmax read), causal | `≈ s²d/2` | `≈ s²d` | 1 |
| **forward substitution (C1)**, causal | `d·s(s−1)/2 + s·d ≈ s²d/2` | `≈ s²d` | **`s`** |
| `Pz` (the shape's read, replaces `PV`) | `≈ s²d/2` | `≈ s²d` | 1 |

So the shape's forward costs `≈ 3·s²d/2` MACs per head against softmax's `≈ s²d` MACs: **+50 % of the causal MACs, +25 % if `QKᵀ` is counted dense** (the brief's I5 figure `s²d` vs `2s²d` is the dense count of the same arithmetic). The softmax itself (exp, sum, divide, `3·s²` elements, READ `scale/m3_flops.py:375-376`) is paid identically by both and is not in the ratio. The one thing the table's last column says that the MAC column hides: the substitution is **depth `s`**, and every other row is depth 1. On a GPU the substitution is therefore not "half a matmul"; it is a latency chain of `s` dependent steps, each a `[1,i]·[i,d]` product, which is exactly what §4.x.2 removes.

Backward (DERIVED from the adjoint of a triangular solve): with `ḡ = ∂L/∂z`, the adjoints are `V̄ = M^{-T} ḡ` (one more triangular solve, upper, `≈ s²d/2` MACs, depth `s`) and `M̄ = −M^{-T} ḡ zᵀ = −V̄ zᵀ` (one outer product, `s²d` MACs over the causal triangle `≈ s²d/2`, depth 1), then `P̄ = −γ M̄` and `γ̄ = −⟨M̄, P⟩`. The backward is therefore `≈ 2×` the forward's solve MACs, the usual ratio, and carries one more depth-`s` chain. Per training step (forward + backward) the solve adds `≈ 3·s²d/2` MACs per head to the softmax head's `≈ 3·s²d` (forward `s²d` and backward `2s²d`): the +50 % is preserved through the step.

The record's own corner is the floor of this law. When `P` is the corner-3 chain — the only nonzero band is the sub-diagonal, `A[i,i−1] = x[i, CH_DRIVE]` — (C1) collapses to the first-order scan `z = a_i z + b_i` (READ `scale/negation_scope.py:288-303`, `equilibrium_oracle`), cost `O(s·d)`, and the path product is a `cumprod` (READ `ceq/arm_smprime.py:144-172`, `path_product`; its own note at `:165-168` names a segmented associative scan as the move past `s = 64`). The general row-stochastic `P` has no scan: the resolvent of a dense lower-triangular matrix is the triangular solve, and (C1) is its cost.

## 4.x.2 The chunked block-triangular solve (DERIVED; the occupied pattern CITED)

Partition the `s` positions into `s/C` chunks of `C` rows. Write `M` in blocks `M_kl`, `l ≤ k`. Then

```
M_kk z_k = v_k − γ Σ_{l<k} P_kl z_l ,        k = 0 … s/C − 1                    (C2)
```

| piece | how | MACs per head | depth |
|---|---|---|---|
| diagonal blocks `M_kk^{-1}` | dense `[C,C]` lower-triangular inverse, all `s/C` blocks in parallel | `(s/C)·C³/3 = s·C²/3` (inverse) or `(s/C)·C²d/2 = s·C·d/2` (apply) | `C` once (inverse) then 1 per chunk |
| off-diagonal propagation | one GEMM `[C, kC]·[kC, d]` per chunk `k` | `Σ_k C·kC·d = C²d·(s/C)(s/C−1)/2 ≈ s²d/2` | 1 per chunk |
| **total** | | `≈ s²d/2 + s·C·d/2 + s·C²/3` | **`s/C`** |

The MAC total is the same `≈ s²d/2` as (C1) — chunking moves no arithmetic, it re-shapes it: the off-diagonal work becomes `s/C` dense GEMMs that tensor cores execute at depth 1 each, and the depth falls from `s` to `s/C`. At `s = 64, C = 64` the whole thing is one dense `[64,64]` triangular solve, which is what `torch.linalg.solve_triangular` executes (one cuBLAS `trsm` per batch element) and what §4.x.6 timed; at `s = 4096, C = 64` the depth is 64 instead of 4096, and the memory of the retained diagonal blocks is `(s/C)·C² = s·C` elements instead of `s²/2`.

**The occupied kernel pattern.** DeltaNet's chunkwise training algorithm — Yang, Wang, Zhang, Shen, Kim, *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, arXiv:2406.06484 (CITED [V], abs page fetched 2026-09-03, title matched) — computes inside each chunk of size `C` a lower-triangular `[C,C]` inversion (its "UT transform"), forms `W = T K` and `U = T V`, and passes a `d×d` state between chunks; the chunk size interpolates between the fully-parallel (`C = L`) and recurrent (`C = 1`) forms, and the per-chunk complexity is stated as `O(LCd + Ld²)` (CITED [V], HTML v3 fetched 2026-09-03). The intra-chunk triangular inversion of (C2)'s diagonal blocks is that pattern and is owned by it. What differs, stated narrowly:

1. **The inter-chunk term has no `d×d` state.** DeltaNet's transition is the outer-product form `KVᵀ` (rank `≤ d`), so everything before chunk `k` compresses into a `[d,d]` matrix and the inter-chunk cost is `O(L d²)`. The shape's `P = softmax(QKᵀ)` (β = 1 corner) has no low-rank inter-chunk state; the term `Σ_{l<k} P_kl z_l` in (C2) is a genuine `[C, kC]·[kC, d]` GEMM, and the inter-chunk cost is `≈ s²d/2` MACs, quadratic. The shape is quadratic **because** it keeps softmax's `P` bitwise at `γ = 0` (I1); DeltaNet is linear because it does not.
2. **DeltaNet is linear attention, no softmax** (CITED [V]); the shape at `γ = 0` is softmax bitwise. The Lean `gate_zero_beta_zero_is_linear_attention` corner (READ brief §2) is the record's own linear-attention setting, and it is the *other* corner of the family from the one priced here.
3. **`γ` is one scalar with a certificate** (`γ^{K+1}/(1−γ)`, §4.x.3); DeltaNet's per-token `β_t` is a learned write strength with no truncation certificate in the cited abstract. NOT FOUND in this session's search: a chunkwise solve of `(I − γ·softmax(QKᵀ))^{-1}` — the search covered arXiv:2406.06484 only and is not a literature census; the prior-art planet owns that.
4. **Absorbing rows** (identity rows on the constraint sets) are a boundary condition the diagonal blocks carry for free — an identity row makes `M_kk`'s corresponding row `(1−γ)·e_i` — and DeltaNet has no such rows.

Design against the taxonomy: the chunked law is stated as MACs *and* depth so that a wall-clock reading is never scaled off the MAC ratio alone — `scale/m3_flops.py:101-121` (READ) records the FLOP model being optimistic by `2.0×` at the pilot geometry and `4.4×–6.6×` at the shipped one because dispatch is not in the arithmetic (M-3, pilot spread taken as the realised spread). No chunked kernel exists in the tree at HEAD `207e7b9`; its clock is `NOT MEASURED — needs a chunked kernel`.

## 4.x.3 The Neumann path at K hops and its crossover (DERIVED + RUN)

Truncate `(I − γP)^{-1} = Σ_{k≥0} (γP)^k` at `K`:

```
z_K = Σ_{k=0}^{K} (γP)^k V   — K matmuls [s,s]·[s,d], each depth 1                (C3)
‖(I − γP)^{-1} − Σ_{k≤K}(γP)^k‖_∞ ≤ γ^{K+1}/(1−γ)     (P row-stochastic, γ < 1)     (C4)
```

(C4) is identity I4 of the brief (RUN by the coordinator at K = 1, 2, 4, 8, 16, error equal to the bound to `1e-15`; planted non-stochastic negative reads `119.37` against a bound of `1.143`). MACs: `K · s²d/2` causal per head, depth `K`. Against the solve's `s²d/2` at depth `s` (or `s/C`):

| | MACs | depth | exact? |
|---|---|---|---|
| solve (C1)/(C2) | `s²d/2` | `s` or `s/C` | yes, to rounding |
| Neumann (C3) | `K·s²d/2` | `K` | no; δ = `γ^{K+1}/(1−γ)` printed |

**MAC crossover: none.** `K·s²d/2 < s²d/2` requires `K < 1`. Truncation never beats the solve in arithmetic; it beats it only in depth, and only when the solve's serial chain is the bottleneck. Hops needed for a target operator-norm δ: `K ≥ log(δ(1−γ))/log γ − 1` — at `γ = 0.5, δ = 10^{-6}`: `K ≥ 19.9`, i.e. 20 hops, `20×` the solve's MACs; at `γ = 0.9`: 129 hops. (The record's own settle count agrees in kind: `scale/arm_s.py:326-334` `neumann_for(β)` routes β through the Δ↔κ inverse and returned `n_neumann = 21` at β = 0.5, READ `V12_PRICING.md:42`.)

**Measured on the certified box** (RUN this session; `n` sequences batched, `s = 64`, `d = 16`, `γ = 0.5`, float32, `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before process start, `use_deterministic_algorithms` off for this timing, median of 14 timed forward+backward calls after 2 warm-ups, `torch.cuda.synchronize` bracketed; no Adam step):

| `n` | `PV` fwd+bwd | solve + `Pz` fwd+bwd | Neumann K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|---|---|
| 2048 | 1.473 ms | **2.514 ms** | 3.001 | 4.524 | 6.511 | 13.182 | 25.409 |
| 4096 | 2.347 ms | **4.659 ms** | 5.684 | 8.652 | 14.571 | 28.064 | 53.470 |
| 8192 | 5.374 ms | **9.916 ms** | 12.222 | 18.812 | 31.835 | 58.147 | 115.050 |

At this shape the exact solve is cheaper than **one** Neumann hop (2.514 vs 3.001 ms at `n = 2048`): the solve is one `trsm` launch per batch and each hop is a `bmm` launch plus an add, so the comparison is launch-bound at `s = 64`, and the solve's depth-64 chain is invisible. The crossover in wall-clock is therefore `K < 1` on this box at `s = 64` — the "cheap path" is not cheap here. Where the Neumann path earns its place is (i) `s ≫ 64` with no chunked kernel, where depth `s` becomes visible — `NOT MEASURED` past `s = 64`; and (ii) the sparse case of §4.x.5, where `(γP)^k` on a block-sparse `P` stays block-sparse (reachability within `k` hops) while the exact inverse fills in.

**The certificate's units** (DERIVED; designed against V-17, a threshold imported out of its units, READ `MISTAKES.md:855`). (C4) is an operator ∞-norm. The error on the *vector* `z` is `≤ δ·‖V‖_∞`. The RUN above printed `max|z_solve − z_{K=16}| = 6.13e-05` (n = 2048) against a bare `δ = γ^{17}/(1−γ) = 1.526e-05`; the bare bound is not violated, because `V` was standard Gaussian over `2.1e6` entries, whose maximum is `≈ 5` [ASSUMED from the Gaussian extreme-value order `√(2 ln N) ≈ 5.4`; not printed by the run], giving a vector bound `≈ 7.6e-05 ≥ 6.13e-05`. The L-CERT rule for every Neumann mask in the arena is therefore: **print `δ·‖V‖_∞`, not `δ`**, or the printed certificate is in the wrong units and a compliant kernel will read as a violation. A float32 solve-vs-Neumann diff below `δ·‖V‖_∞` is the must-fire planted check; a planted non-stochastic `P` (row sums 1.5, as in I4) is the rejection region (V-24, empty rejection region).

## 4.x.4 The segmentation dividend (DERIVED from the F0 certificate; the number READ)

`pathProd_eq_zero_iff` (READ `lean/CEQ/V16Domain.lean:129`; summarised at `:41`: the product is `0` iff a zero magnitude is on the path) is the F0 certificate: a zero gate at position `k` sends every entry whose window contains `k` to exactly `0`. In resolvent language: if the strictly-causal part of `P` has zero blocks `P_BA = 0` for every pair of segments `A < B`, then `M = I − γP` is block-diagonal by segment and so is `M^{-1}` — no approximation, no δ. The solve cost falls from `s²d/2` to

```
Σ_m L_m² d / 2 ,   L_m = length of segment m,   Σ_m L_m = s ;   dividend  D = s² / Σ_m L_m²    (C5)
```

For `s/L̄` equal segments, `D = s/L̄`. Two things follow. First, the dividend is a property of the **corpus's** segment-length distribution, not of the arm, and it must be quoted with the corpus named — a dividend carried from one bed to another is V-22 (READ `MISTAKES.md:1140`, a pre-registered constant carried in from another system). Second, it applies to the exact solve, the Neumann path and the CSR path alike, because a block-diagonal `M` makes all three block-diagonal.

**BED-M's number.** On BED-M `e3_t2` at `n = 128, s = 64`, `96.78 %` of causal pairs `j ≤ i` annihilate — positives `257,664 / 266,240` (READ `V16_ARM_SMPRIME.md:518-522`; the same `96.78 %` at `workdonenewseal.md:92` and `README.md:147`). The nonzero fraction of the causal resolvent is `1 − 0.9678 = 0.0322`, so

```
D_BED-M = 1 / 0.0322 = 31.06×     (DERIVED from the READ fraction; denominator-independent)
```

The same line reads `133,120` causal pairs while `266,240 = 128 · 64·65/2` is the count of pairs `j ≤ i` and matches the printed positives (`257,664/266,240 = 0.9678`); the `133,120` is half of that and is flagged here as a provenance inconsistency in the source (P-class), which does not move the fraction. What the fraction *means* is the domain-census point (V-25, READ `MISTAKES.md:1954`): `0.0322 · 2080 ≈ 67` live pairs per 64-row block; with `s/L̄` segments of length `L̄` the live pairs number `(s/L̄)·L̄(L̄+1)/2 = 32(L̄+1)`, so `L̄ ≈ 1.1`. BED-M's real support at `t* = 2` is one to two positions per segment — the 31× dividend on BED-M is a statement that BED-M is almost entirely dead gates, not that natural text segments are short. The paper carries the dividend as the law (C5) with BED-M's `31.06×` as one measured instance and the segment-length distribution of BED-S, BED-K and any text corpus as `NOT MEASURED — needs the corpus's zero-gate census`.

For a softmax `P` an exact zero needs a `−∞` logit, i.e. a mask; that is the F0 case. When the mask drops entries that are small but nonzero, the certificate is F1 — Cantelli per pair with a Boole/Bonferroni union over the mask (`d = 65` at `δ = 1 %`, `32×`), or Azuma for token-dependent gates (`d = 20` at the same δ) — READ `CEQ_V20_R15_CONTRACT.md:212-219`, annex M9. Every arena mask carries one of F0/F1 or is refused (L-CERT, READ `:64-67`).

## 4.x.5 The CSR two-stage path, and what a Mapper cover buys (READ + DERIVED)

The executor exists and is merged: `triton-lang/kernels#22`, forward-only, consumes a causal CSR block schedule built from sink blocks, local-window blocks and a 0D-persistence salience over key-block centroids; tested for dense-CSR parity, block selection, schedule validation, dtype, 2-D and batched/headed correctness (READ `THEORY.md:112-119`; CITED [V] GitHub PR page fetched 2026-09-03: "Add topology-derived sparse attention kernel", merged 2026-07-28). Its two failure modes that escaped validation — an empty schedule row returning `acc/0 = NaN` silently, and a negative block index passing the `k_pos < seq` mask into an out-of-bounds load that poisons the CUDA context — are made structural in the tree's own kernel (READ `ceq/mz_kernel.py:10-21`, `:80-83`, `:116-121`). The useful-FLOP count the tree already uses is

```
attention_flops = 4 · units · B² · d · BH        (READ ceq/mz_kernel.py:170-179; 2·QKᵀ + 2·PV per visited tile)
```

**Two stages.** Stage 1 (topology selects candidates): the schedule is data, not code — a CSR list of `[B,B]` tiles per query block (READ `THEORY.md:122-131`). Stage 2 (dense resolvent inside): per query block-row, the selected tiles are the nonzeros of `P`'s block-row, and (C2) runs over them. Cost per visited tile of the solve's off-diagonal propagation: `B²d/2` MACs = `B²d` FLOPs, so the resolvent adds `1 · units · B² · d · BH` FLOPs to the kernel's `4·units·B²·d·BH` — **+25 % over the visited tiles**, plus the diagonal tiles' intra-tile inverse (`(s/B)·B³/3`).

**The fill-in caveat, which decides which of §4.x.1–4.x.3 the CSR path can host.** The inverse of a block-sparse lower-triangular matrix is dense along reachability: if tile `(k,l)` is dropped but `(k,m)` and `(m,l)` are kept, `M^{-1}` has a nonzero `(k,l)` block. So the *exact* solve on a CSR schedule is exact only when the dropped tiles disconnect the graph (§4.x.4, block-diagonal); otherwise the exact solve either fills in (cost returns toward `s²d/2`) or silently computes the resolvent of a *different* `P` — the masked one — with no certificate for the difference. The path that keeps the schedule's sparsity honestly is Neumann on the sparse `P`: `(γP)^k` has the sparsity of `k`-hop reachability, each hop is a block-sparse matmul over the schedule, and the total certificate is the union of the mask's F1 δ (dropped mass) and the truncation's `γ^{K+1}/(1−γ)`. The paper states the two-stage path as: **F0 segmentation ⇒ exact block-diagonal solve; F1 mask ⇒ Neumann on the sparse `P` with a union certificate; exact solve on an F1 mask ⇒ refused** (a mask without a certificate is a heuristic, L-CERT).

**What a Mapper / nerve cover buys at long context.** `topoml` supplies `metric_cover / nerve_graph / mapper_graph` (READ `THEORY.md:23`). A cover `{U_a}` of the key positions gives candidate tiles = nerve edges; the resolvent's cost becomes `Σ_a |U_a|²d/2` plus the overlap terms, against `s²d/2` — the same form as (C5) with overlaps, so a cover buys exactly what segmentation buys, minus the overlaps, and its dividend is again a corpus property. Two measured warnings apply. (i) A candidate set realised by `index_select` pays the gather: `2.58 ms` of `index_select` against `0.82 ms` of attention over the same slice at 65,536 positions on this same card (READ `ceq/multizoom.py:12-16`); the CSR tile form of kernels#22 reads contiguous `[B,B]` tiles and avoids it, so a Mapper cover must be quantised to tiles before it is a schedule. (ii) The far-field the cover drops needs a printed bound: the tree's only computable one is the mean-pool coarsening bound `‖A − Ã‖_∞ ≤ (D_∞/2)(e^{δ_max} − 1) + max_G[min(2, e^{R_G} − 1)·r_G]` from key/value geometry alone (READ `ceq/multizoom.py:38-49`); a Mapper cover with a comparable printed δ is `NOT FOUND` in the tree, so a Mapper schedule enters the arena only through the F1 union bound of annex M9. Backward through kernels#22 does not exist (READ `THEORY.md:223-224`, risk 6); the CSR path is inference-side until it does.

## 4.x.6 Determinism (RUN; READ of the installed torch; CITED)

Method: the installed `torch.use_deterministic_algorithms` docstring (READ, `torch/__init__.py` of torch 2.5.1+cu121 on this box, printed via `inspect.getsource`) was read in full for each operator's class, and each operator was then executed under `use_deterministic_algorithms(True)` on the certified card with `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before the process started (`V17_R4_RETAKE_PRICE.md:178-181` READ: set in-process it does not take). Three outcomes are distinguished — executable-and-bitwise, executable-and-drifting, not executable — because collapsing "could not tell" into "pass" is V-16 (READ `MISTAKES.md:828`).

| operator | docstring class (READ, torch 2.5.1) | RUN, flag ON, CUDA, this box | class of the verdict |
|---|---|---|---|
| `torch.cumsum` | in the **throws `RuntimeError`** list: "when called on a CUDA tensor when dtype is floating point or complex" | forward **raises**, backward **raises** (`cumsum_cuda_kernel does not have a deterministic implementation`) | RUN + READ |
| `torch.cumprod` | **absent** from both lists | forward **OK**; backward **raises** the same `cumsum` error — autograd differentiates a cumulative product through `cumsum` | RUN; matches `COSTS.md:147-154` (READ, 8 repeats, hop bitwise flag OFF and ON, gradient NOT EXECUTABLE flag ON) |
| `torch.linalg.solve_triangular` | **absent** from both lists | forward **OK**, backward **OK**; over 8 repeats, forward `max|Δ| = 0.0`, backward `max|Δ| = 0.0` on `[2048,64,64]·[2048,64,16]` float32 | RUN (repeatability, one box, one process); the docstring's silence means no deterministic *guarantee* is documented by torch |
| `torch.mm / bmm` | deterministic **only if** `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8` is set, else raises | forward OK, backward OK | RUN + READ |

The torch 2.5 reproducibility note (CITED [V], `docs.pytorch.org/docs/2.5/notes/randomness.html`, fetched 2026-09-03) names none of `cumsum`, `cumprod`, `solve_triangular` and defers CUDA matmul reproducibility to the cuBLAS workspace setting. cuBLAS's own reproducibility statement (CITED [V], `docs.nvidia.com/cuda/cublas` §results-reproducibility, fetched 2026-09-03) guarantees bitwise-identical results per toolkit version on GPUs of the same architecture and SM count, withdraws the guarantee under multiple active streams unless one workspace or handle per stream (or the `CUBLAS_WORKSPACE_CONFIG` values above) is used, and does not name `trsm`. That `solve_triangular` dispatches to cuBLAS `trsm` on CUDA is [U] (not verified from source this session; the ATen binary is compiled). The class of the determinism claim for the shape is therefore: **`solve_triangular` — RUN-bitwise on the certified card under the single-stream, workspace-pinned regime, with no torch-documented guarantee; `cumsum` — documented non-deterministic, raises; `cumprod` — undocumented, forward bitwise, backward blocked by `cumsum`.**

Consequence for the arms. The shape's solve path is executable under strict mode forward *and* backward, which `arm_smprime`'s `cumprod` path is not (backward raises, `COSTS.md:151-154` READ; reproduced at `V17_R4_RETAKE_PRICE.md:161-164` READ). So a shape built on the triangular solve is the first arm in the record whose *training* step can run under `use_deterministic_algorithms(True)`; the record's "hazard follows reduction length, not launch size" finding (`workdonenewseal.md:328-334` READ: bitwise at reduction 64 and 4,096, drifting only at `1e6`) is the measured floor for the flag-OFF regime. Design against P-1 (a number with no live producer): the RUN's exact command is the one-liner recorded in this session's transcript — `torch.use_deterministic_algorithms(True)` then `solve_triangular` on `tril(rand(64,64)) + I` against `randn(64,16)`, 8 repeats forward and backward — and it must be added to `scripts/k_cert.py::determinism_at_64` (READ `:579-640`) as a fourth quantity beside hop / forward / gradient before any arena cell quotes it; that edit is code and is owed, not done.

## 4.x.7 Memory: the `[S,S]` materialisation (READ + DERIVED)

The one number that decides everything in `ceq/sizing.py` (READ `:9-16`): SDPA never forms the `[S,S]` matrix and its activations are `O(S)`; an operator that materialises `P` and lets autograd retain it grows as `O(S²)`, measured **`3.9` tensors of `[B,H,S,S]` per layer** (`C_OPERATOR`, READ `:48`), re-solved against the CUDA allocator at **`3.823`, R² `0.996373`** (READ `COSTS.md:90`), with `C_RESIDUAL = 17.874` (R² `0.996373`, `:89`). The measured ratio signed/softmax activation bytes at `B=4 d=256 L=4 H=4` fp32 runs `1.44×` at `seq 128` to **`8.06×` at `seq 2048`** (READ `ceq/sizing.py:17-22`) — a function of `S`, not a constant.

Bytes per element on the certified device: fp32 `4.0` B; under bf16 autocast the operator term measures **`3.341` B/elem (R² `0.999830`)**, not 2 (READ `COSTS.md:91`, `ceq/sizing.py:54-65`), and the residual term `2.383` B/elem is the one constant the module gets wrong, optimistic by `+8.3 %` (READ `COSTS.md:92-99`). The shape's solve consumes `P` explicitly — (C1) reads row `i` of `P` — so it inherits the `O(S²)` class unless the chunked path of §4.x.2 recomputes off-diagonal tiles from `Q, K` in the backward (flash-style), at the price of two more `QKᵀ` tile products per tile; that kernel is `NOT MEASURED — needs a chunked kernel`. The retained-tensor increment of the solve itself (DERIVED): the forward retains `M` (or `P`, 1 tensor of `[S,S]`) and `z` (`[S,d]`); the backward forms `M̄ = −V̄ zᵀ` (one `[S,S]` transient). Estimate `C_OPERATOR_shape ≈ C_OPERATOR + 1…2` [ASSUMED; the record's `3.823` was fitted on the signed arm, not on this operator, and carrying it is M-8 unless re-solved]. Furthermore `torch.linalg.solve_triangular` has no bf16/fp16 CUDA path in torch 2.5.1 [U, from the linalg dtype policy; the RUN used float32], so the solve's `[S,S]` operand stays at `4.0` B/elem under autocast where the softmax's stays at `3.341`.

Measured at the arm's shape (RUN, same run as §4.x.3, `torch.cuda.max_memory_allocated` above a reset baseline): peak extra bytes for solve+`Pz` fwd+bwd equal `PV`'s at `n = 4096` (`48.1` vs `48.0` MiB) and `n = 8192` (`96.1` vs `96.0` MiB); at `n = 2048` the two readings (`24.0` vs `88.0` MiB) are caching-allocator ordering, not a saving, and are not quoted. `z` at `n = 2048` is `2048·64·16·4 B = 8 MiB`. Residency on the card is unchanged from `COSTS.md:111-116` (READ): the workhorse arm is resident to `n = 8192` and pages over PCIe at `n = 16384` under a training loop (reserved `10.578 GiB` against `7.996`); the shape on the softmax corner inherits the softmax rows (resident to `n = 32768`, `4.908 GiB` reserved) plus `8·(n/2048)` MiB.

## 4.x.8 The price of one arena cell on the certified RTX 4060 ([FITTED] + [RUN], every extrapolation labelled)

The arena cell (READ `CEQ_V20_R15_CONTRACT.md:119-127`): 4,769 parameters matched, **150 steps**, **N = 8** seeds, one BH family, every cell reporting crossing rate, conditional NRMSE, distance-to-floor, distance-to-skyline, GPU-seconds-to-floor, peak bytes, and the certificate grade of any mask; criterion (3) is *lowest GPU-seconds-to-floor*. The R1′ shape is `n_train = 2048, s = 64, d_model = 16` (READ `V17_R4_RETAKE_PRICE.md:45-47`).

Laws (`[FITTED]`, READ `COSTS.md:73-74`; R² re-read from `results/k_cert_local.json` `throughput.laws`): `softmax` `s/step = exp(−12.1852)·n^{0.9963}`, R² `0.9999975`, 5 points `n = 2048…32768`; `arm_smprime` `s/step = exp(−10.0187)·n^{1.0026}`, R² `0.9999999`, 3 points `n = 2048, 4096, 8192` — the arm's law does not extend past 8192 because those points are bus measurements, not arm measurements (READ `COSTS.md:76-79`). Both laws are `s = 64, d_model = 16`, forward + backward + `Adam.step`, median over 3 children of the median of ≥ 12 steps — the R1 regime (READ `V17_R4_RETAKE_PRICE.md:220-224`). The measured control on the law: the arm cell read `15.970 s / 150 steps` (mean of 2) against the law's `13.960 s`, `+14.4 %`; softmax `1.497` vs `1.524 s`, `−1.8 %` (READ `:228-229`).

The shape's increment is **the solve, measured at the shape's own geometry** (RUN §4.x.3, fwd+bwd, no Adam): `Δ = 2.514 − 1.473 = 1.041 ms/step` at `n = 2048`, `2.312 ms` at `4096`, `4.542 ms` at `8192`. Adam on 4,769 parameters is taken as unchanged between the two arms [ASSUMED; the parameter count is matched by the contract]. The increment is a per-op microbenchmark and therefore a **floor** on the end-to-end cell increment, not a prediction of it — the record measured the FLOP-model-to-wall-clock gap at `2.0×–6.6×` when Python dispatch entered (READ `scale/m3_flops.py:101-121`; P-8, an upper bound stated as a price, READ `MISTAKES.md:387`); the honest statement is that the increment lies in `[1.041 ms, ~2–7 ms]` per step at `n = 2048` until an end-to-end cell is timed.

| base corner of `P` | law s/step at `n = 2048` `[FITTED]` | + solve `[RUN]` | 150-step cell | × 8 seeds | + fixed 4.0 s (READ `V17_R4:198-201, 328-331`) |
|---|---|---|---|---|---|
| softmax (β = 1), the shape's parity corner | `0.010162` | **`0.011203`** (+10.2 %) | **`1.680 s`** | `13.4 s` | **`17.4 s`** |
| `arm_smprime` (corner 3, complex64 path product) | `0.093066` | **`0.094107`** (+1.1 %) | **`14.12 s`** | `112.9 s` | **`116.9 s`** |
| softmax control (no solve), the cell's fellow | `0.010162` | — | `1.524 s` | `12.2 s` | `16.2 s` |

So **one arena bed-cell pair (shape on the softmax corner + its softmax control, 8 seeds each) prices at `≈ 34 s` of GPU time `[FITTED + RUN]`**, and on the corner-3 base at `≈ 133 s`; peak memory `0.9224 GiB` measured for the arm cell (READ `V17_R4:139`) plus `8 MiB` for `z`, under the `7.996 GiB` card by `8.6×`. At the full ladder (9600 steps, `×64`, READ `V17_R4:243-254`) the pair is `≈ 36 min` on the softmax corner. Extrapolations: to `n = 8192` the softmax law gives `0.040436 + 0.004542 = 0.044978 s/step` `[FITTED R² 0.9999975 + RUN]`, inside the law's fitted range; to `n = 16384` and `32768` the softmax law extends (`0.080663`, `0.160907`, READ `COSTS.md:237-238`) but the solve increment is `[ASSUMED linear in n from the three RUN points: 1.041 → 2.312 → 4.542 ms, ratio 2.22 and 1.96 per doubling]` and is `NOT MEASURED` there; the corner-3 base has no law past 8192 at all. No Kaggle figure is derived: the Kaggle certificate slot is empty (READ `COSTS.md:158-168`) and a threshold carried across a device boundary is V-22 (READ `COSTS.md:190-193`).

Design against the taxonomy, in one place: M-8 (each corner priced at its own law; the solve's increment measured, not ratioed); M-3 (increment measured at `s = 64`, the shipped geometry, at all three `n`, never scaled from a smaller `s`); P-8 (the microbenchmark stated as a floor with the measured dispatch gap beside it); V-22 (no cross-device carry); P-1 (every number names its producer: `scripts/k_cert.py`, `results/k_cert_local.json`, `V17_R4_RETAKE_PRICE.md`, and this session's RUN command); L-COST (the GPU-seconds-to-floor column is the arena's, and this section supplies the per-step term it multiplies).

## 4.x.9 The cost law in one table, and the kernel path's must-fire checks

Per head, per sequence, MACs (multiply `× 2` for house FLOPs), increments stated over the softmax head the shape contains at `γ = 0`:

| path | forward MACs | backward MACs | depth | memory class | certificate | status at HEAD `207e7b9` |
|---|---|---|---|---|---|---|
| softmax head (fused causal SDPA) | `s²d` | `2s²d` | 1 | `O(s)` | exact | shipped (control) |
| shape, serial substitution (C1) | `+ s²d/2` | `+ s²d` | `s` | `O(s²)` (`P` explicit) | exact | `torch.linalg.solve_triangular`, RUN §4.x.3 |
| shape, chunked (C2), chunk `C` | `+ s²d/2 + sCd/2 + sC²/3` | `≈ 2×` | `s/C` | `O(sC)` retained + recompute | exact | `NOT MEASURED — needs a chunked kernel`; pattern occupied by arXiv:2406.06484 [V] |
| shape, Neumann `K` hops (C3) | `+ K·s²d/2` | `+ K·s²d` | `K` | `O(s²)` or block-sparse | `δ = γ^{K+1}/(1−γ)`, print `δ·‖V‖_∞` | `bmm` loop, RUN §4.x.3; never wins in MACs, `K < 1` |
| shape, F0-segmented (C5) | `+ Σ_m L_m² d/2` | `≈ 2×` | `max L_m` | `O(Σ L_m²)` | exact (`pathProd_eq_zero_iff`) | dividend `31.06×` on BED-M (READ fraction), corpus-dependent |
| shape, CSR two-stage (§4.x.5) | `+ units·B²d/2` | none | `s/B` | `O(units·B²)` | F1 union bound (annex M9) + Neumann δ; exact solve refused on an F1 mask | kernels#22 forward-only [V]; resolvent stage `NOT MEASURED` |
| record's corner 3 (chain `P`) | `+ sd` (scan) | `+ 2sd` | `s` (`cumprod`) | `O(s²)` as shipped (`path_product`) | exact | `ceq/arm_smprime.py:144` READ; backward blocked under strict mode (`cumsum`) |

Each row is designed against a named mechanism: the increment column against M-8 (no cross-arm ratio), the depth column against M-3 (a MAC ratio never stands in for a clock), the certificate column against L-CERT and V-17 (units), the status column against P-1 and P-4 (no claimed scaffolding that does not exist — every `NOT MEASURED` names the missing kernel).

**Must-fire checks the kernel path ships with**, each a planted negative so that the parity test is not V-2 (a hand-built example where right and wrong coincide) and the bind has a rejection region (V-24):

1. **Parity at `γ = 0`:** `torch.equal(P (I − 0·P)^{-1} V, P V)` bitwise (I1, RUN by the coordinator); rejection region `γ = 0.5`, max abs `2.3003` (RUN, brief §1).
2. **Solve against dense inverse:** `solve_triangular` vs `torch.linalg.inv` to `1.8e-15` in float64 (I5); planted negative: a non-triangular `M` passed with `upper=False` must disagree.
3. **Neumann certificate in vector units:** `max|z_solve − z_K| ≤ γ^{K+1}/(1−γ) · ‖V‖_∞` at K ∈ {1, 2, 4, 8, 16}; planted negative: row sums `1.5` (I4's non-stochastic plant, err `119.37` vs bound `1.143`).
4. **Segmentation exactness:** with a `−∞` logit column at position `k`, `M^{-1}` has exactly-zero blocks across `k` (`torch.equal` to zero, not `allclose`); planted negative: replace `−∞` by `−30` and the block must be nonzero — the F0/F1 boundary made visible.
5. **CSR fill-in guard:** on a schedule that drops tile `(k,l)` but keeps `(k,m)` and `(m,l)`, the exact solve's `(k,l)` block must be reported nonzero and the mask refused unless a Neumann δ is printed; the empty-row `NaN` and negative-index loads of `ceq/mz_kernel.py:13-21` (READ) remain the two structural guards.
6. **Determinism, three outcomes:** `solve_triangular` forward and backward bitwise over 8 repeats under strict mode (RUN, `0.0`); `cumsum` must raise (RUN); a run that reports "pass" on an operator that raised is V-16 and is refused.
7. **Cell price control:** the shape's 150-step cell on the softmax corner must read within the record's own law-vs-measured band (`−1.8 %` to `+14.4 %`, READ `V17_R4:228-229`) of `1.680 s`; a reading above `2.2×` that (the dispatch gap's lower end, READ `scale/m3_flops.py:106-108`) flags the microbenchmark increment as the P-8 floor it is declared to be.

## Limits

Every RUN number here is one box, one process, one session, float32, `use_deterministic_algorithms` off for timings and on for the executability table, with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; the laptop GPU's clock is not stationary (`V17_R4_RETAKE_PRICE.md:143-150` READ: the same cell read `14.390` and `17.839 s`), so the timing table is quoted to the medians it printed and no per-op figure is offered to better than the `±12 %` spread the record measured on cells. The solve increment is a per-op microbenchmark without Adam and without Python dispatch, and is a floor. The `n = 2048` peak-memory readings are allocator-ordering artefacts and are struck. The chunked solve (§4.x.2), the flash-style recompute path (§4.x.7), the Neumann path past `s = 64`, the CSR-hosted resolvent, and any Mapper schedule have no kernel in the tree and are priced in MACs and depth only; their clocks are `NOT MEASURED`. `solve_triangular`'s determinism is repeatability on one card with no torch-documented guarantee, and its dispatch to cuBLAS `trsm` is [U]. `C_OPERATOR` for the shape is not re-solved; the `3.823` carried is the signed arm's and is marked [ASSUMED] where used. The DeltaNet comparison rests on its abs and HTML pages fetched this session and on no other paper; "NOT FOUND" for a chunkwise resolvent of a softmax `P` reflects a one-source search, not a census. The BED-M dividend uses a fraction whose source line carries an internally inconsistent pair count; the fraction itself is consistent with the printed positives and is the only quantity used. The Kaggle card has no certificate; nothing here transfers to it.
