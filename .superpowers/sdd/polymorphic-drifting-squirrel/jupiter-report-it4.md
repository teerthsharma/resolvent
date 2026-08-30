# JUPITER / MYCROFT — iteration 4, the maths survey

Worktree `agent-ade92ee4d5e164d8f`, fast-forwarded `486ae41` → `f02410d`.
Three commits: `b953205`, `e682dec`, `098dd8c`.

**No wall-clock measurement was taken.** The `4.560600 s/step` rate is an input,
read from `scale/m3_flops.py:107`, not a clock taken here. Everything else is
combinatorial, closed-form, or read from a journal already on disk.

**On the crash.** The repo working tree was lost; the scratchpad was not. All
five survey parts, the `MATHEMATICS.md` append and all three verification
scripts survived intact in the session scratchpad and were recovered. Every
journal-derived number was **re-verified at the new HEAD** before anything was
committed, and all reproduce exactly. First commit landed before any further
work, per instruction.

---

## THE FRAMING WAS WRONG, AND FIXING IT IS THE SURVEY'S MAIN RESULT

Brief says: *"discrete selection being trained despite the selection being
non-differentiable."* There are **two** discrete stages here, not one.

| | discrete object | site | trained |
|---|---|---|---|
| **Stage A** | which `k` rows are pivots | `m3_quintuple.py:169` → `pivot_probe.py:80`, `topk(key.norm)` | **never, in any arm** |
| **Stage B** | mixture `α` over the chosen `k` | `m3_quintuple.py:440` | yes |

`argmaxste` estimates **stage B**. Forward bitwise `argmax`'s — `soft −
soft.detach()` is exactly `+0.0`, checked over 200 drawn gates in float64.
**Stage A is byte-identical in `argmax`, `softmax`, `argmaxste`, `settled` and
`twin`.** Nothing measured this round bears on stage A at all.

That split sorts the whole field, so it is stated first.

---

## RESULT 1 — THE GRADIENT IS THE EFFECT; THE MECHANISM IS NOT

Recomputed at HEAD from `results/m3_quintuple_v2.jsonl`, `ntr8192`, 5 seeds, via
the repo's own `contrast`:

```
    argmax    -> argmaxste  +0.225760  CI [+0.212433, +0.245886]  5/5
    argmaxste -> twin       +0.004092  CI [-0.023107, +0.029187]  3/5
    settled   -> twin       +0.002959  CI [-0.031557, +0.048587]  2/5
    argmaxste -> settled    +0.001133  CI [-0.069616, +0.057312]  3/5
```

Three structurally different stage-B mechanisms lie within `0.004092`, every CI
covering zero. The one contrast excluding zero at 5/5 is *having a gradient*.

## RESULT 2 — THE PRICING RULE, AND IT KILLS A WHOLE LITERATURE

MDE at 5 seeds, paired, `α=0.05`, power `0.80`. Two paths: normal approximation,
and an exact noncentral-`t` solve (which needs a `nan` guard — `scipy.stats.nct`
returns `nan` at large noncentrality and an unguarded bisection converges to a
non-monotone wrong answer; I caught that and fixed it).

**At 5 seeds this instrument resolves `0.057946` or larger.** Stage-B mechanism
differences sit `14×`, `20×` and `51×` below it. Seeds needed: `559`, `2257`,
`41266` (normal `556.3 / 2254.1 / 41263.5`).

> Any candidate whose contribution is a better **stage-B** relaxation is
> competing for an effect this instrument cannot see at any plausible seed
> budget. **Unfalsifiable here**, therefore inadmissible under this project's own
> evidence rule — not on merit, on measurement.

## RESULT 3 — THE STAGE-A EXPERIMENT IS WRITTEN AND UNRUN

K4 (`DONE_ARCHIVE_ROUND1.md:4707`) ablated stage A with content-blind pivots,
read `+0.081`, and concluded content selection is *"not load-bearing for M2"*.
**That null was forced, not measured.** For any content-blind schedule of size
`k`, `(k/s)·(1/k) = 1/s` exactly — `k` cancels. M2 draws `c` *from* `P`, so it
conditions away the only factor where selection could show. Verified two paths:
symbolic, and 200 000-draw Monte Carlo over 12 `(s,k)` pairs.

`scale/recall_probe.py:3-7` states that identity and `:9-11` names the quantity
that is not an artefact. **It is imported by zero Python files, has no results
artifact, and `DONE_ARCHIVE_ROUND1.md:5833` calls it "already sitting unrun".**
Stage A's space is `C(62,8) = 3 381 098 545` sets; `topk` explores one.

> **The highest-value mathematics in this survey is not missing. It is written,
> correct, and unrun.**

## RESULT 4 — A PROVABLE GREEDY BOUND EXISTS, AND THE SIGNED ARM VOIDS IT

Derived, not cited. `B_P = Σ_{p∈P} u_p v_pᵀ` — rank-1 per pivot, independent of
the rest — so captured mass is `1_PᵀM1_P` with constant second difference
`2M_ab`, `M_pq = (u_p·u_q)(v_p·v_q)`.

- **Captured mass is SUPERMODULAR** on a nonnegative operator. No `1−1/e`. The
  intuition that pivot selection is obviously diminishing-returns is **wrong on
  the obvious objective**.
- **Reconstruction `h(P)` is monotone submodular**, `h(∅)=0`, marginal gain
  `≥ ‖u_a v_aᵀ‖²` — tight to `−3.55e−15` over **21 936 `(S,a)` pairs**. Greedy
  attains **`1 − 1/e = 0.632121`**.
- **Signed operators lose it**: `M` negative in `297/300` draws at `n∈{4,5,6}`,
  `300/300` at `n=6`.

**A prediction of mine that was wrong, recorded**: I expected `h` non-monotone.
`0` of 21 936 pairs decreased it. Corrected derivation shipped.

## RESULT 5 — THE PROJECT'S CENTRAL CITATION IS OVERSTATED THREE WAYS

`FINDINGS.md` C2 cites `arXiv:2410.01537` as *"softmax provably Bayes-optimal"*
against *"linear attention"*. Equations fetched:

| repo says | paper says |
|---|---|
| softmax | `T_λ^{k,v}(𝕏) = erf(λ𝕏k)ᵀ𝕏v` — **erf** |
| provably | Corollary 2 is **asymptotic**, `d→∞`, `L=o(d)` |
| linear **attention** | Prop 3 refutes **linear regression**, `ℛ(β⋆)→ε²+γ²` |

And the regime: repo runs `L=64`, `D_MODEL=16` (`m3_capability.py:79`; the `d24`
in journal keys is `make_batch`'s flipper **offset**). `L/d = 4.00` — the
opposite of `L=o(d)`. **Label shape does match**, so the worry is sound, but
"provably" is not earned at this geometry. For whoever holds `FINDINGS.md`.

## RESULT 6 — THE `nash.py` SHARED-`tau` DEFECT IS REAL AND TOO SMALL

`ceq/nash.py:64` returns one `float` for the batch from the worst instance,
applied at `:146`. Attenuation `τ_i/τ_batch`: median `0.789701` at `n=2048`, min
`0.690213`. Two paths (small-signal ratio; exact sigmoid amplitude).

**A `1.27×` attenuation cannot produce OOD NRMSE `2.6151`–`5.8198`.** Before the
fix and rerun, one line settles it: `matrix_norm(game).max()/median` on a real
batch. Near `1.27` ⇒ the bug cannot be the cause and the ordinal idea failed on
its merits. Surrogate draw, so this is a lower bound on attenuation.

## RESULT 7 — THE VECTOR READOUT IS ALREADY BUILT

`FINDINGS.md` C reads as a proposal. It is implemented:
`m3_quintuple.py:112 ROW_CELLS`, `:368 vector_readout`, `:483 return out if
self.vector_readout else out[:, s-1]`. Every vector-valued candidate has its
plumbing already, at zero parameter cost.

---

## CANDIDATES: 16 SURVEYED, AND WHAT ACTUALLY SURVIVED THE EQUATION CHECK

**Be careful with this count — the classes differ.**

| class | n | which |
|---|---|---|
| **Equations/code fetched and verified BY ME** | **3** | Berthet `2002.08676` (Prop 2.1/2.2 + top-k set `𝒞`); Marion et al. `2410.01537` (predictor, Cor 2, Prop 3); k-DPP via DPPy reference docs (`det(L_S)/e_k(L)`, `O(N³)`, `O(Nk²)`) |
| **Derived by me from the repo's own objects, two paths each** | **2** | the content-blind identity; the submodularity structure |
| **Resolved and title-matched, equations NOT fetched** | **1** | `2603.02289` "Topological Causal Effects", Kim & Lee, 2026-03-02 |
| **Moon-reported with equations claimed, NOT re-verified by me** | **10** | SIMPLE, Sander, Cuturi, SOFT/OT top-k, NeuralSort, SoftSort, LapSum, DFTopK, cellular sheaves, hyperbolic cones, Shapley/IG, influence functions, SCM surgery |

So: **16 surveyed; 5 carry a verdict I stand behind personally; 1 is
citation-verified but not equation-verified; 10 remain moon-class and are marked
as such in the survey.** I re-verified exactly one moon verdict by fetch
(Berthet — its moon first reported `Ω` must be *learned*, which would have failed
the parameter gate, then reversed itself; I fetched and confirmed `Ω` is the
Fenchel dual of `F₁`, analytic, never fitted) and two moon code claims by grep
(`vector_readout`, `safe_tau` — both confirmed exactly).

**Expected to fail, plainly:** every stage-B relaxation (unfalsifiable, §2);
NeuralSort/SoftSort (`O(s²)` for a permutation whose only consumer is a gathered
`[n,k]` set); influence functions (wrong axis, `O(p³)` at `p=4769`); cellular
sheaves as published (restriction maps `σ(V[x_v‖x_u])` need a new `V`); LapSum
with learned `α` (literally `+1 nn.Parameter`); k-DPP as a *trainable* selector
(its gradient wants an observed subset; there isn't one); submodularity on the
captured-mass objective (supermodular).

**Ranked by value per unit cost to find out:** (1) run `recall_probe.py`;
(2) `matrix_norm(game).max()/median`, one line; (3) `ACE_i` vs existing `B`
columns, one correlation; (4) restate `FINDINGS.md` C2/C3; (5) IG as a
diagnostic; (6) SIMPLE/Sander — **gated behind (1)**; (7) greedy under `h` —
gated; (8) persistent-homology label.

---

## CLAIM LEDGER

| # | claim | class | check |
|---|---|---|---|
| L1 | Stage A never trained; `argmaxste` is stage B | READ+RUN | `m3_quintuple.py:169,440`; STE forward bitwise `argmax` over 200 drawn gates, float64 |
| L2 | gradient effect `+0.225760`, 5/5 | RUN | `contrast` on journal, re-verified at `f02410d` |
| L3 | three stage-B mechanisms within `0.004092`, all CIs cover zero | RUN | same, three pairwise contrasts |
| L4 | MDE at n=5 is `0.057946` | RUN | two paths: normal `0.043164`, exact nct `0.057946`, monotone in `n` after the nan guard |
| L5 | 559 / 2257 / 41266 seeds needed | RUN | normal `556.3/2254.1/41263.5`, exact-t `559/2257/41266` |
| L6 | `(k/s)(1/k)=1/s`, `k` cancels | DERIVED+RUN | symbolic + 200k-draw MC, 12 `(s,k)` pairs; `scale/pivot_selection_theory.py` asserts both |
| L7 | `recall_probe.py` unrun | READ | zero importers, no results artifact, `DONE_ARCHIVE_ROUND1.md:5833` |
| L8 | captured mass supermodular; reconstruction monotone submodular; `1−1/e` | DERIVED+RUN | closed form `±2M_ab` vs direct evaluation on every checked subset; monotonicity tight to `−3.55e−15` over 21 936 pairs |
| L9 | signed operator voids it | RUN | `297/300` at `n∈{4,5,6}`, `300/300` at `n=6` |
| L10 | `2410.01537` says erf / asymptotic / linear regression | CITED (single source) | ar5iv full text fetched; ICLR PDF returned compressed streams |
| L11 | repo sits at `L/d = 4.00` | READ | `m3_capability.py:79 D_MODEL=16`; `negation_scope.py:84-86` |
| L12 | nash `tau` attenuation median `0.789701` | RUN | two paths; games built as `nash.py:138-144` |
| L13 | `vector_readout` already built | READ | `m3_quintuple.py:112,368,483` |
| L14 | Berthet `Ω` is analytic, not learned | CITED | `ar5iv/2002.08676` Prop 2.1, fetched by me |
| L15 | k-DPP cost `O(N³)` + `O(Nk²)` | CITED (single source) | DPPy reference docs; monograph PDF unreadable |
| L16 | `2603.02289` resolves and title-matches | CITED | fetched; Kim & Lee, 2026-03-02 |

---

## ADVERSARIAL PASS

- **Did I nearly recommend a re-run of something already done?** Yes. K4 had
  ablated stage A. Grepping first turned "never tried" into "tried on a bed that
  could not answer it, and the reason is written down in the repo."
- **Was my own arithmetic wrong anywhere?** Twice. The nct bisection returned a
  non-monotone MDE (`nan` at high noncentrality) — caught because MDE must fall
  with `n`. And I predicted `h` non-monotone; it is monotone, provably.
- **Did a moon ship something wrong?** Its first Berthet fetch said `Ω` is
  learned, which would have failed the parameter gate. I re-fetched; it is
  analytic. Ten other moon verdicts remain un-re-verified and are labelled.
- **Is the pricing rule self-serving?** It rules out most of the brief's own
  suggested territory. It is derived from the repo's own paired SDs and survives
  two independent power calculations.
- **Is the `1−1/e` useful?** Honestly, probably not yet: it bounds greedy against
  the best pivot set under an objective nobody has shown matters.

---

## WHAT COULD NOT BE VALIDATED

Ten of the sixteen candidates rest on moon fetches I did not personally re-check,
and the base prompt is explicit that moon output is unverified until re-checked —
so those ten carry a lower evidence class than the five I stand behind, and the
survey labels them rather than quietly averaging them in. Two of my own citations
are single-source: `2410.01537`'s `L=o(d)` regime rests on one ar5iv fetch
because the ICLR PDF returned compressed streams, and the k-DPP complexity rests
on the DPPy reference documentation because the Kulesza–Taskar PDF did the same;
both are load-bearing for a verdict and neither has a second path. The
`1−1/e` result is proved for a nonnegative operator and checked on drawn
matrices up to `n=6`, never on a real attention matrix from this corpus, and NWF
1978 itself was cited from standard knowledge and not fetched. The nash `tau`
attenuation uses Gaussian `q,k` rather than the trained distribution, so it is a
surrogate lower bound and the real spectral spread is exactly what the one-line
diagnostic I recommend would return. Nothing here establishes that stage A is
load-bearing — that is the whole point of ranking `recall_probe.py` first, and
until it runs every stage-A candidate in this survey, including the ones I rank
highest, is a bet on an unmeasured axis. I trained nothing, took no clock, ran no
full suite, and touched none of the held files.
