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
| **The SDPA baseline is FAIR** (`ceq/lm.py:175`), so the `COSTS` ratios stand — the comparison is against a tuned baseline, which C6 demands. | — | READ |

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
