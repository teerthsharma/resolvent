# Cutting edge — scan of github.com/teerthsharma, 2026-08-24

Scanned 122 owned repositories (50 public, 72 private). About 30 of those are
upstream forks kept for PR work (pytorch, vllm, triton, xla, mujoco, tinygrad,
TensorRT-LLM, candle, highway, zstd, cugraph, xformers, DirectStorage, …). They are
working copies, not results, and are excluded below.

What remains is one research programme in twenty-odd repositories, and it is more
unified than the repository names suggest.

---

## 1. The spine: one theorem, five names

The scan's main finding is not a repository. It is that the same statement is
implemented independently in five places under five different vocabularies, and in
at least two of them it does not appear to be connected up.

The statement: **a map that merges distinct inputs destroys something no later stage
recovers, and the amount destroyed is countable without ground truth.**

| repo | name it goes by | the quantity |
|---|---|---|
| `caustic` | Theorem 1, Orbit Error Bound | `err(f) ≥ n − m`, orbits of the answer partition |
| `branchcut` | Result 1, application-stripped | same bound, aimed at quantisers / hashes / caches |
| `monodromy` | injectivity without the Jacobian | can the map be run backwards; two-point collision |
| `Epsilon-Hollow` / `foliation` | leaf space = quotient by block-aligned-prefix | `refcount` = cardinality of the fibre |
| `sigmoid` | sheaf consistency gate | detects when a trajectory left the calibrated manifold |

`caustic` and `branchcut` know they are the same — branchcut's abstract says so
outright. `monodromy` is the same question asked of a general map instead of an
answer map. **`foliation` is the one that is not connected up, and it is the
interesting one**, because it is the only place where the partition is computed by a
running system at 4 KiB granularity rather than by an analysis script.

### The second, independent coincidence

`Epsilon-Hollow`'s `stratum` and `topological-ml-toolkit` compute the *same pipeline*
in two languages for two purposes:

```
stratum  (no_std Rust, in-kernel):  v_t → Takens delay embed (tau=1, d=3) → VR 1-skeleton → cycle rank
topoml   (Python/Rust, userspace):  samples → time_delay_embedding(dim, tau) → persistent_homology → betti_at
```

`stratum` is `topoml.time_delay_embedding` composed with `topoml.persistent_homology`
composed with `betti_at(...).beta1`, written a second time under a 4,792-byte
fixed-memory budget with no allocator. That is not duplication to delete — the
constraints are genuinely different — but it does mean **`topoml` is the reference
implementation `stratum` should be differentially tested against**, and nothing in the
scan shows it being used that way. `stratum`'s only oracle today is seven hand-written
regime fixtures plus a naive gap baseline that must misfire on the healthy-but-noisy
control.

---

## 2. The junction the idea is asking about

`epsilon-hollow ∩ caustic ∩ topological-ml` has one concrete intersection and one
seductive false one. The false one first, because it is the one that looks best on a
slide.

### The false junction: "certify KV cache error with Theorem 1"

The kernel already holds both terms of the bound. `foliation` knows `n` (live
sequences) and `m` (distinct leaves). Theorem 1 appears to hand over a
ground-truth-free error floor for free.

It does not, and the reason is the theorem's own precondition. Theorem 1 needs the
ground relation to be **injective** — distinct entities must have distinct correct
answers. In the foliation the ground relation is *sequence → its true KV state*, and
two sequences sharing a block-aligned prefix genuinely **do** have the same correct
answer. Collapse there is the feature. Prefix sharing is not error; it is the entire
point of the data structure. `n − m` on the foliation counts successful deduplication,
and certifying it as error is exactly backwards.

Worth writing down before building, because the arithmetic works and only the
semantics refuse.

### The real junction: `fold_key` is an unverified injectivity assumption

`foliation` seals a block with:

```
key = fold_key(prev_key, tokens)
```

and then `descend(current_leaf, key)`. Two sequences that land on the same key land on
**the same physical frame**. So the correctness of the entire cache rests on: distinct
token blocks produce distinct keys. That is an injectivity precondition on a hash, it
is currently assumed, and it is precisely the failure caustic already found, measured,
and shipped a checker for.

Caustic's own tokenizer collisions are the template:

- `Asmara` / `Asuncion` both first-token 1634
- `Lusaka` / `Ljubljana` both 444
- and the total case: every numeric answer collapsing to token 220 under Qwen- and
  Llama-family tokenizers, where Theorem 1 would certify 19 wrong answers on a model
  that answered all twenty correctly

Caustic's response was `verify_injective`, described in its own README as "the only
thing between a caller and that false certification". **The transplant is
`verify_injective` pointed at `fold_key`** — and `branchcut/APPLICATIONS.md` already
names hashes and caches among its ten screened targets, so the aim was half-taken.

Consequence if it collides: two different prompts silently share a KV frame. That is
not a slow cache, it is a wrong-output cache, and it is invisible to every existing
boot gate. `collapse_violations` and `referenced_evictions` both check structure;
neither checks that the keys were distinct to begin with.

### The third piece: the floor as an objective, not a diagnostic

`caustic` §10.2 and `branchcut`'s `select_by_floor` describe a governor: candidate
settings run in competition, scored by a bound that consults no answer key, with the
do-nothing candidate always entered so declining is a first-class outcome.

`Epsilon-Hollow`'s `[KVPOLICY]` gate already runs four policies — foliation, LRU,
random, Belady — on a provably identical candidate set with an identical descent
sequence. **That is the competition, already built, with no scoring function on it.**
The foliation policy's victim choice is currently a hand-ranked tuple:

```
(entrants, −depth, last_use)
```

Three hand-chosen terms in a hand-chosen order. Everything else in that subsystem is
derived or measured — `LOOP_SCALE_MARGIN = 1.5` gets a floor (`√5/√3 ≈ 1.291`), a
ceiling (`2.0`) and a midpoint argument — and this tuple gets none. It is the softest
thing in an otherwise hard module, and `select_by_floor` is the shape of its
replacement.

---

## 3. Load-bearing versus gated

Worth keeping straight, because the programme's public claims are unusually careful
and these notes should not be less careful than the repos are.

**Measured and standing:**

- caustic: AUROC 0.995 — *on collapse-type errors only*. Disperse-type falls to
  0.67–0.71 with bootstrap intervals spanning 0.5, and caustic says so first, in a
  Scope section placed above its own abstract
- caustic: 163 closed-form tests, 11 theorems, 1 no-go, 0 violations over 2,048,574
  exhaustive configurations for Theorem 8
- monodromy: 116 tests, 15/15 mutants caught, 0/24 false positives, 8/8 against the
  determinant test's 6/8
- Epsilon-Hollow: `stratum` and `foliation` both real, both boot-gated, both with
  negative controls that must fail
- sigmoid: 880× lower cost per step; exact MuJoCo island partition; bit-identical
  Triton kernel salience at 13×

**Explicitly not established, by the repos' own admission:**

- caustic `element_symbol` accuracy 1.000 — graded on first letters; 4 of 16 golds are
  multi-token
- caustic "pooled, full context" — one relation, not a pool
- sigmoid sparse attention path — reported as *not* faster on available hardware
- Epsilon-Hollow benchmarks — gated, and one of its own headings reads "Nothing In The
  Kernel Has Ever Been Tested"
- topoml backends — C++/ASM/Triton/PyTorch/TF are contracts, not implementations, and
  the README says so rather than pretending gated acceleration is done

The habit of leading with the scope limits is the most transferable thing in the whole
programme, and it should survive into whatever this idea becomes.

---

## 4. Build order

Smallest first; each stands alone if the next is never done.

1. **`verify_injective` on `fold_key`.** Port caustic's checker into
   `ml_engine/foliation.rs` as a boot-gate assertion over the embedded 210-descent
   trace: distinct token blocks, distinct keys, fail loudly on collision. This is the
   one with an actual correctness bug behind it.
2. **Differential-test `stratum` against `topoml`.** Same delay embedding, same
   filtration, same cycle rank, on the seven regime fixtures plus generated folds.
   Turns seven fixtures into a property test with an independent oracle.
3. **Score the eviction competition by a floor.** Replace `(entrants, −depth, last_use)`
   with `select_by_floor` over the existing identical candidate set, do-nothing
   entered. Belady stays as the ceiling the online policy must not beat.

Do not build the Theorem-1-certifies-KV-error version. It is arithmetically fine and
semantically inverted.

---

## 5. Identifiers

- caustic — DOI 10.5281/zenodo.21997746
- topological-ml-toolkit — DOI 10.5281/zenodo.21997818
- Epsilon-Hollow / Seal OS v0.4.7.5 — DOI 10.5281/zenodo.20264206
- monodromy — DOI 10.5281/zenodo.22064739, arXiv:2608.00222
- profile — arXiv:2604.19792

Private repos on this spine not opened in this scan: `charlie`, `lambda-topo`,
`topobridge-q`, `pytorch-topology`, `topo-asm`, `hollow-asm`, `EPSILON-PHASE`,
`laamba-silence`, `hamliton`, `aether-link`, `topoflow`, `phi-mem`.
