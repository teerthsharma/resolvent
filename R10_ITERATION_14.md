# R10 — iteration 14: the corpus spec, and a theorem that needed a norm

**Script task.** *"JUPITER: fetch F1–F8; corpus spec: labels u solve
u(v) = (1/deg v)Σ_{w∼v} u(w) interior, u|_B = g; oracle u_I = (I−P_II)⁻¹P_IB g
(the proved resolvent); dose theorem err(t) ≤ λ₂ᵗ with λ₂ computed exactly per
instance (one eig, n ≤ 1024), rungs stratified to λ₂ ∈ [0.90, 0.95]."*

## F1–F8: seven cited, one absent and declared

| | result | status |
|---|---|---|
| F1 | `λ₂ := ρ(P_II)`, **not** the literal 2nd eigenvalue of `P` | cited `foreman_lambda2.py:24-25`, `MATHEMATICS.md:426` |
| F2 | resolvent oracle `N = (I−Q)⁻¹`, `B = NR` | cited `MATHEMATICS.md:423` |
| F3 | exact truncation identity `u − u_t = Qᵗu` | cited `MATHEMATICS.md:431` |
| F4 | the dose theorem in the weighted sup norm | cited, **machine-checked** at `lean/CEQ/Contraction.lean:72`, `:99` |
| F5 | Kirchhoff / matrix-tree second oracle | cited `scale/kirchhoff.py:1-90` |
| F6 | tolerance `1e-10` and its window | cited `kirchhoff.py:98-106` — **imported, not re-picked** |
| F7 | band `[0.90, 0.95]` and the refuted conductance dial | cited `foreman_lambda2.py:110` |
| F8 | `ρ(P_II) < 1` iff boundary reachable | **NOT IN REPO — ASSUMED** |

F8 is recorded under its standard name (substochastic Perron–Frobenius) with **no
line number invented**. Three further absences are named rather than cited:
reversibility of the interior block (the repo's conjugation at
`foreman_lambda2.py:289` covers only the *ergodic* walk), Perron monotonicity
under principal-submatrix deletion, and Gelfand's formula.

That is the behaviour the brief demanded, and it is the opposite of what a
fabricated citation looks like. F6 matters for the same reason: the tolerance was
imported from where it already lives rather than chosen fresh to fit the result.

## The oracle, checked three ways

| comparison | max disagreement |
|---|---|
| resolvent solve vs averaging map iterated from adjacency lists | **9.992007e-16** |
| Kirchhoff matrix-tree route, `\|B\|=2` path family | 2.220446e-16 |
| against the closed form `u_x = 1 − x/(L+1)` | 2.220446e-16 |

Tolerance `1e-10`, imported. The second route forms **no matrix**, so the two
oracles do not share an implementation path.

**Must-fire:** a planted `deg+1` defect is **rejected 12/12**, gaps 4.07e-01 to
6.30e-01. Without it, agreement at 1e-16 would only show the two routes are the
same code twice.

## The dose theorem is false as the spec states it

`err(t) ≤ λ₂ᵗ` depends on a norm the spec never names.

| norm | result |
|---|---|
| `l_∞` | **held 6/12, VIOLATED 6/12**, worst ratio **1.290631** at t=64 |
| degree-weighted `l₂` | **12/12 and tight**, worst rate slack 7.00e-04 (0.074%) |
| unweighted `l₂` | 12/12, but flagged as luck — available violation factor `sqrt(d_max/d_min) = 4.36` |
| Perron-weighted sup (the repo's formalised norm) | 7/7 where defined, **undefined on 5/12** (reducible `P_II`), reported `null`, never a pass |

**On `g ≡ 1` instances it fails by construction**: `u ≡ 1` exactly, so
`err(1) = 1.0000000000 > λ₂`, because any interior vertex without a boundary
neighbour survives one step with probability 1 — and **every in-band instance has
one**. This is not a numerical edge case; it is the band selecting for the
geometry that breaks the sup-norm claim.

**Amendment, stated and not applied silently:** `err(t) ≤ λ₂ᵗ` needs *"in the
degree-weighted 2-norm"* to be true. The unweighted `l₂` pass is explicitly not
offered as support, since the degree spread on these instances leaves a 4.36×
violation available — passing there is luck that has not yet run out.

**Second amendment:** `(I − P_II)` is singular unless every interior vertex
reaches `B`, with the guard shown firing on planted graphs.

**Dose must-fire:** a fixed −0.02 plant fires 12/12 in the degree-weighted `l₂`
and 6/7 in the Perron sup. The one non-firing rung has measured slack 0.026786 —
that bound is genuinely too loose to see a 2%-wrong rate in 64 steps. Reported as
**the vacuity class, not a broken instrument**, and the adaptive plant fires 7/7.

## Stratification: what the corpus costs

| quantity | value |
|---|---|
| naive prior, 200 draws, seed `0x3a141000` | median λ₂ 0.8305, range 0.5625–0.9944 |
| **yield into `[0.90, 0.95]`** | **16/200 = 8.0%** |
| stratified: bisection on `\|B\|`, monotone therefore exact | **121 eigensolves → 12 rungs**, 10.1/rung, 0 misses, 0 errors |

**Limit stated rather than left for a reader to find:** every rung lands in
`[0.9435, 0.9499]` — always the **upper half** of the band. The corpus is not
uniform across `[0.90, 0.95]`, and any reading that assumes it is will be reading
a narrower regime than it thinks.

## Why this iteration matters beyond its own deliverable

λ₂ is the quantity that decides how many hops a label needs. The capacity sweep
has been measuring softmax against a **1-hop** budget, and the dose bound is the
formal statement of what one hop can reach. So a dose theorem that is false in
the sup norm and true in the degree-weighted 2-norm is not a technicality — it
changes which norm every downstream truncation claim in this round has to be
stated in.

## Open

- The corpus is built only in the upper half of the λ₂ band.
- F8 is assumed, not proved in-repo.
- it.15's per-instance `|u_absorbing − u_kirchhoff|_∞ ≤ 1e-10` assertion is
  demonstrated on the `|B|=2` family here; SATURN owes it across the corpus, with
  the standing rule that **disagreement halts the corpus, not the reading**.
