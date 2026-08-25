# Theory 2 — DEAD. All three branches, killed by the room.

The plan below is what replaced it. The autopsy is at the bottom because the
replacement matters more than the corpse.

---

## The plan

Three items, in order. None of them is new mathematics. All three are things this
project already half-built and then walked past.

### P0 — Bind the bracket. The only nonzero commutator on the board.

Foreman restated House's result one abstraction higher and it is the correct statement: the
obstruction is the **commutant**, not polynomials. For a diagonalizable operator every
spectral projector IS a polynomial in it (Sylvester interpolation), so a spectrum escapes
nothing. And Hodge is the maximal instance of the flatness rather than an escape, because
`d1 d2 = 0` forces `Delta_up Delta_down = Delta_down Delta_up = 0` identically.

He demanded the number. Measured, `sgate` at rho=1.5 lam=0.10, seed 0:

| pair | `\|[X,Y]\|_F` | relative |
|---|---|---|
| **head 0 vs head 1** (House's `A1, A2`) | **6.682667e-01** | **1.100419e-01** |
| `A1` with itself | 0.000000e+00 | 0.000000e+00 |
| `A1` with `p(A1)` — the commutant | 7.391322e-07 | 4.314323e-08 (machine precision) |
| `Delta_up` vs `Delta_down` — Hodge | **0.000000e+00** | 0.000000e+00 |

**Two different heads are the only pair on the board that does not commute.** Dr House
handed over a non-commuting object and Theory 2 replaced it with three commuting ones. That
was the error, and Foreman's criterion is what exposes it.

    out = v + (A1 + A2) v + beta [A1, A2] v      beta = 1/2 per BCH

`A1, A2` from splitting an existing head, so parameter count stays exactly matched. The
bracket is identically zero for any single-generator scheme — SimA, DeltaNet, linear
attention, APPNP — and identically zero at hop 1, so multi-hop becomes NECESSARY rather than
optional. That last point is the only account anyone has given of why hops=2 loses to hops=1
(1.0334 vs 1.0258): `A^2` of one generator is flat by construction.

**Falsifier (Dr House's, unchanged):** three arms at matched parameters — (i) `A1+A2` flat
control, (ii) `A1+A2+A2A1` raw product, (iii) `A1+A2+beta[A1,A2]` bracket only. **Dead if
(iii) does not beat (i), or if (ii) matches (iii)** — the second means the antisymmetric part
carries nothing the symmetric part did not.

### P1 — Re-measure budget scaling on `sgate`. First, before anything.

W10 measured the gap **widening** with budget on the L1 operator: 1.2191x at 250 steps
became 1.337x at 600. **That has never been re-measured on `sgate`**, the operator that
now carries the 1.0334 parity claim. `STATE.md` says it is the first thing to check before
any 0.5B run, and Theory 2 scheduled it nowhere.

If the gap widens on `sgate` too, the parity result is a small-budget artifact and every
downstream proposal was a mechanism *of nothing*.

**Measure:** ratio at 300 / 600 / 1200 / 2400 steps, 3 seeds, `rho=1.5 lam=0.10 hops=2
lr=1e-3`, against the softmax control at each budget.
**Kill:** monotone increase in ratio across budgets.

### P2 — COGS and SCAN. The benchmarks are already on disk.

`data/cogs_{train,dev,test,gen}.tsv` and six SCAN splits exist. `ceq/harness.py` is
mid-build with published references keyed by paper. Its tests went RED and stayed RED.

Theory 2 contained the strings "COGS" and "SCAN" **zero times** while proposing three new
operators. That is the whole indictment.

**The published row that can move:** Edge Transformer **0.874 +/- 0.004** against a
Universal Transformer control at **0.784** on COGS — someone already changed the attention
and moved that number. ARC-AGI has no equivalent at any parameter scale.

**Measure:** softmax control against `sgate` at matched parameters on COGS-gen and SCAN
addprim_jump, published reference in the third column.
**Kill:** `sgate` does not beat its own matched control. Then the operator wins nothing on
the only benchmark chosen because it could see the property.

### P3 — Sweep depth and parameter count. The axis the surviving claim lives on.

The only claim that survived three rounds of novelty review is **depth/parameter
efficiency**: one GELU hands content-conditional sign back to softmax at 0.0547, so
`sgate`'s 0.1641 is a rate and not a capability.

Theory 2 proposed three operators and **none of them varied depth or parameter count.**

**Measure:** hold the property fixed, sweep depth 1/2/4 and width, and find whether `sgate`
reaches a given sign-flip rate at fewer parameters or fewer layers than softmax+GELU.
**Kill:** softmax+GELU matches it at equal depth and equal parameters. Then there is no
efficiency claim either, and the module's last distinguishing property is a tie.

### The 300M gate, restated because it was a termination clause

Cameron: *"a gate whose only outcome is OOM is a termination clause wearing a gate's
clothes."* Correct. 36.31 GiB against Colab's 33.53 GiB for a 0.5B signed model at batch 1.

**Substitute, stated in advance so it cannot be moved later:** P1's ratio must be
**non-increasing** from 300 to 2400 steps, and P3's efficiency margin must **not shrink**
from depth 1 to depth 4. Those are scale proxies measurable on hardware that exists here.
If both hold, the 300M run is worth buying. If either fails, it is not.

---

## What the room broke

- **Chase:** T1's falsifier fires on a theorem, not a measurement. `pow_card_eq_zero` makes
  the shipped operator nilpotent, so its spectrum is the single point 0 -- **measured
  |lambda_1| = |lambda_2| = 0.000e+00, gap 0.000e+00, 0 of 256 eigenvalues above 1e-12**.
  "No spectral gap ⇒ the equilibrium story fails" is true for every input at every setting
  and carries zero bits about content. → **T1 DELETED.**
- **Chase:** `occupancy_is_exact_inverse` proves a two-sided inverse, which is unique. There
  is exactly **one** fixed point, by theorem. ESS selects among multiple equilibria. →
  **T3 DELETED by my own Lean proof.**
- **Cameron:** "the harmonic component has no local witness" is **false**. R6 scored AUROC,
  which is detection, not attribution, and an LS residual is a local readout that sees curl
  and harmonic together. Worse: on a bare graph there are no 2-cells, so curl is undefined
  and *every* independent cycle is harmonic -- the harmonic dimension is a hand-picked choice
  of which cycles get filled. → **T2 DELETED.**
- **Chase:** T2 was called "cheapest" and "a probe on quantities already computed." The graph
  Hodge 1-Laplacian is `[E,E]` with E = S(S-1)/2 -- **2,096,128 edges at seq 2048, a
  4.39e12-entry operator**, plus 1.43e9 triangles for curl. Neither boundary operator exists
  anywhere in this repo. → the cost claim was wrong by twelve orders of magnitude.
- **Cameron:** the only surviving claim is depth/parameter efficiency and **none of T1, T2,
  T3 varies depth or parameter count.** → became P3.
- **Cameron:** COGS and SCAN are on disk, their harness is RED, and Theory 2 mentioned
  neither. → became P2.
- **Chase:** the re-measurement `STATE.md` calls the first thing to do was scheduled
  nowhere. → became P1, and it gates everything.
- **Foreman:** the organizing claim was stated one abstraction too low. It is the
  **commutant**, not polynomials — spectral projectors are polynomials by Sylvester, and
  Hodge's `Delta_up`/`Delta_down` commute **exactly** (measured 0.000000e+00). All three
  picks were commuting families. → **became P0, and it is the only novel operator left.**
- **Foreman:** T2's central sentence contradicts this repo's own R6 docstring — "module and
  baseline read the SAME subspace, the cycle space". On a 1-complex `ker Delta_1` IS the
  cycle space and LS rms IS harmonic energy, already on the board at 0.7741-0.8671 and
  already beaten by LS max|r| at 0.9860. Harmonic contradiction energy is **column 1 of a
  table I deleted**.
- **Foreman:** T1's falsifier could not come back red — a finite-rank numerical estimate on a
  learned matrix generically has distinct eigenvalue moduli, so "look for a gap" always finds
  one. An unfalsifiable falsifier is not one.

## Still open

- **Foreman and Wilson have not reported.** Wilson is verifying all nine attributions and
  Foreman is attacking the same theory. Their findings arrive after this revision and may
  change P1-P3.
- **T1's one surviving reading is unexamined.** A Ruelle operator on the *model's dynamics*
  is a different object from the attention matrix, and Chase's nilpotency kill does not
  reach it. It is not rescued here, only not-refuted; nobody has stated what it would act on.
- **Cameron's last question has no answer.** *"When all three land and the module still loses
  COGS-gen to a vanilla transformer -- who reads 'harmonic contradiction energy' second,
  after the person who computed it?"* Nobody. That is why P2 exists and why it outranks
  every new operator.
