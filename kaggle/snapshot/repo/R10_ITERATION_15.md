# R10 — iteration 15: the dual oracle, and where it stops being dual

**Script task.** *"SATURN: generator + dual oracle; per instance assert
|u_absorbing − u_kirchhoff|_∞ ≤ 1e-10 (matrix-tree cofactor route) — disagreement
halts the corpus, not the reading."*

## The result that matters

Both routes agree to **1.554e-15** over 29 instances, and each rejects a planted
`deg+1` defect on its own side:

| plant | rejected | gaps |
|---|---|---|
| absorbing-side only | **29/29** | 2.917e-01 – 6.302e-01, median 0.4831 |
| Kirchhoff-side only | **17/17** | 2.917e-01 – 5.954e-01 |
| **applied to BOTH** | **0/17** | max gap **3.331e-16** |

`I − Q = D⁻¹L_T` is a **diagonal** relation, so the two routes share adjacency and
degrees. A defect in the graph data is invisible to both.

**So the agreement licenses less than it appears to.** It certifies
normalisation, indexing, block structure and transposition conventions. It does
**not** certify the shared graph. A corruption of the adjacency itself passes
1.554e-15 and reads as a clean dual check.

This is the answer to the question the brief insisted on — *name what is
structurally different, or the two routes may be the same computation twice*:

- **absorbing:** solves nonsymmetric `I − P_II` on the interior alone, reads `u` directly
- **Kirchhoff:** solves symmetric unnormalised `L` on `interior ∪ {b}` against `e_b`, recovers each value as the **ratio** `z_v / z_b`

tied to literal spanning-forest counts (8/13, 7/13, 4/13) by brute-force
enumeration at **0.000000e+00**. The routes are genuinely different in solver and
recovery. They are identical in their input.

## Counts: four categories, not three

31 instances (28 rungs + 3 probes):

| category | n |
|---|---|
| passed | 29 |
| failed | 0 |
| errored | 0 |
| **inapplicable** | 2 |
| **never generated** | **1** |

The fourth category surfaced during the run: `widen-n128-d4`, seed `0x3a150000`,
where bisection stepped from above 0.905 to 0.8943072442 with no integer `|B|` in
band. It is emitted under its own `kind` — and a dict-merge that had been filing
it back under `kind: "rung"` was found and fixed at
`scale/r10_dual_oracle.py:578`. Under the merge it would have read as a rung that
simply passed.

**Inapplicable is reported as UNCHECKED, never as a pass:** `|B|=1` (the cofactor
ratio is 0/0) and a two-component graph with one boundary node per component
(every grounding singular). Both carry `dual_checked=false`. A third probe on the
same ring with `|B|=2` passes, so the refusals are the precondition biting, not a
broken route. **Zero of the 28 drawn rungs are inapplicable — which is precisely
why the probes exist.**

**Weight positivity is UNTESTED, not satisfied.** `kirchhoff.laplacian` builds
unit conductances only.

## The halt is literal

| corpus | emitted |
|---|---|
| 3 instances, clean | 3 rows |
| same 3, `deg+1` planted on instance 1 | **0 rows**, `CorpusHalted`, gap 3.981e-01 |

The two instances that passed **their own** check are not emitted. That is what
*"disagreement halts the corpus, not the reading"* has to mean if it means
anything: a corpus with one unconfirmable instance is not a corpus minus one
instance.

## The tolerance cannot be re-picked

`1e-10` is imported as `kirchhoff.AGREEMENT_TOL` from `scale/kirchhoff.py:106`,
and `demo()` asserts **object identity** — so a local re-pick fails the
self-check rather than silently flattering the agreement. A single-route check
runs alongside: `Σ_b ω^(b) = 1`, max deviation 2.220e-15.

## The band clustering, confirmed and priced

JUPITER's it.14 limit reproduces **exactly**: 12/12 rungs in
[0.9434514476, 0.9499393658], all in the top `[0.94, 0.95)` bin, 121 eigensolves,
**λ₂ delta 0.0** against `results/r10_it14_corpus_spec.jsonl`, identical boundary
sizes.

**The corpus does not cover the band it claims. It covers the top fifth.**

Widening is one argument (`hi = target` against a pre-declared ladder) and gives
16 rungs spanning [0.9033, 0.9450], 3–4 per bin. Priced: 1 band miss in 17
attempts, eigensolves per rung unchanged (9.7 vs 10.1), but **64% more
second-oracle work per rung** (44.2 vs 27.0 Kirchhoff solves) because lower
targets need larger `|B|`.

## Correction to this record's author

Three static-analysis leads were forwarded to SATURN mid-run. **All three were
false positives.** The "unknown import symbol" on `scale/r10_corpus_spec.py` was
the analyzer failing on an *untracked* file — the module resolves and exports
every name used, and `reproduce_check` proves the generator was **imported, not
reconstructed**, at delta 0.0. The two arity hits were a starred-unpack the
analyzer cannot count through.

This is the second time this round linter output was forwarded as candidate
defects and did not hold; JUPITER's "redeclared `isofit`" was a guarded
`except ImportError` fallback. Both times the leads were labelled as leads, and
both times they cost an agent a verification pass. A static analyzer's complaint
is evidence that a human-readable invariant *might* be violated, not that one is —
and on an untracked or dynamically-assembled module the false-positive rate is
high enough that forwarding without first checking reachability is not worth the
recipient's time.

## Open

- The dual oracle cannot certify the graph data. Any future claim resting on
  "checked two ways" must say **which** two ways, because on this pair the answer
  excludes the input.
- The corpus covers `[0.943, 0.950]`, not `[0.90, 0.95]`. The widening is priced
  and not taken.
- Weight positivity untested; only unit conductances have been exercised.
