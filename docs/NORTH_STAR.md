# The forward S-matrix reconciliation

<p style="color:var(--rs-dim);font-size:.85rem">A reconciliation, not a result. It gives this project a vocabulary and a lineage that predate it by decades. It adds no measurement — every number that follows is either a physics identity checked on a synthetic instance, or a repeat of a verdict measured and reported elsewhere on this site.</p>

## The sentence

> **The S-matrix is the unitary matrix connecting sets of asymptotically free particle states — the in-states and the out-states — with no account of the path between them.**
>
> Wheeler introduced the object in 1937 (*"On the Mathematical Description of Light Nuclei by the Method of Resonating Group Structure,"* Phys. Rev. 52, 1107–1122); Heisenberg made it the organizing object of particle theory through the 1940s (*"Die beobachtbaren Größen in der Theorie der Elementarteilchen,"* Z. Phys. 120 (1943), 513–538, with a third installment in 1944 on its analytic structure).

That is the whole claim, in one line: **where it ends up, from where it started, with no account of the path.** It was written down as a physics object nine decades before this project asked the same question of an attention matrix.

## The mathematics is the same object, not a metaphor

The Lippmann–Schwinger equation defines the scattering operator `T` from a free Hamiltonian `H0`, a perturbation `V`, and an energy `E` approached from the causal side:

```
T = V + V G0 T                      Lippmann–Schwinger
G0 = (E − H0 ± iε)⁻¹                the free resolvent
```

Solved in closed form, `T = (I − V G0)⁻¹ V`. Expanded in powers of `V G0` instead of inverted, the same equation is the **Born series**:

```
T = V + V G0 V + V G0 V G0 V + …  =  Σ_{k=0}^{∞} (V G0)^k V
```

**The Born series is the Neumann series of the resolvent** — the identical object this project calls the hop expansion, `(I − gP)⁻¹ = I + gP + (gP)² + …`. Both are geometric series of an operator, and both carry the same convergence law:

```
Born series converges   iff   ρ(G0 V) < 1        (spectral radius of the free resolvent times the perturbation)
hop expansion converges iff   γ·ρ(P) < 1          (this project's own condition, ceqjepa/operator.py)
```

Two names for one condition: a bounded operator composed with itself, geometrically, converges exactly when its spectral radius sits under 1. Unitarity of the full `S`-matrix is Born-series mass conservation — probability is neither created nor destroyed by the expansion, term by term. The `± iε` prescription is what selects the causal (retarded) resolvent over the anti-causal one; it is the **limiting absorption principle**, and it is the same margin this project already carries in its own `E − H0 + iε` construction of `G0` and in the resolvent read's own boundary convention.

### Worked instance

Verified on this box rather than asserted. `n = 12`, complex128, `H0` diagonal with entries drawn uniform on `[1, 3]`, `V` a symmetric real matrix with entries `N(0, 0.1²)`, `E = 4 + 0.05i` (the `+iε` margin folded into `E`'s own imaginary part). Script: `born_series_check.py`, `python born_series_check.py`, seed 0.

| quantity | value |
|---|---|
| `ρ(G0 V)` | **0.2596** |
| Born-series error, `V + (VG0)V + … ` truncated at 5 applications of `VG0`, ‖·‖<sub>F</sub> | **1.8×10⁻⁴** |
| truncated at 10 applications | **2.1×10⁻⁷** |
| truncated at 20 applications | **2.8×10⁻¹³** |

`ρ(G0 V) < 1` holds, and the error against the exact `T = (I − VG0)⁻¹V` falls geometrically, roughly six orders of magnitude every ten hops — the same rate the spectral radius predicts.

### The honesty that comes with it

The first instance run for this reconciliation put `E` **inside** the band spanned by `H0` (`E = 2 + 0.05i` against `H0 ∈ [1, 3]`, same seed). `ρ(G0 V) = 1.6688`. The series diverges — the truncation error *grows*, not shrinks, with more hops: `3.5`, `25`, `4.1×10³` at 5, 10, 20 applications, same seed 0, same script. That run is reported here **corrected and unscored**, not quietly replaced with a convergent `E`. It is the same standard the rest of this project holds itself to: a bar whose branch never fires is worth nothing, and a divergent instance that gets moved off the page without a note is the same failure in a physics costume.

## What this is, and is not

This reconciliation gives the project a **vocabulary and a lineage** — the hop expansion is not a new numerical trick, it is the Born series of a resolvent physics has used since 1937, and its convergence condition is not a heuristic, it is a spectral-radius theorem. That is the entire content of the claim.

**It adds no measurement.** Nothing here changes any number reported elsewhere on this site — not the operator-arm result, not the chess-prediction result, not the open question of why the gate closes. A reader who comes away believing physics *validated* this project's results, or that the project *derived* something new about physics, has been misled, and that costs more credibility than the framing buys.

Sentences that are not permitted on this page or any other page of this site, because none of them is true of what was built:

- Any sentence describing the read in terms of **waves** — there is no wave equation here, only a resolvent and a spectral condition.
- Any sentence about **inverse scattering** — recovering `V` from `S` is a different, harder problem this project does not touch. The read runs forward only: parameters and structure in, an outcome distribution out.
- Any sentence claiming the operator **is** an S-matrix, rather than that reading it this way **has that shape**.
- Any sentence suggesting physics **validates** this project's results. It does not, and cannot; the two are checked by entirely different evidence.

The one sentence the page is allowed to make, and no larger:

> **The read is a forward S-matrix — in-state to out-state through the resolvent; unitarity is the Born mass; the `+iε` is the causality margin the project already carries.**

## The verdicts this reconciliation does not change

These are measured elsewhere on this site (`docs/PHASE_H.md`, `docs/PHASE_I.md`, `docs/PHASE_I1.md`); the physics framing above does not add to or subtract from any of them.

- **The operator represents order.** Handed S5's 120-state automaton as bare integer symbol ids — no permutation matrices, no transition table, verified by the source not reaching any arm — the operator arm scores `0.8620 ± 0.0556` against a commuting-diagonal control at `0.2860 ± 0.0150`, matched at 404 parameters by `numel()`. An independent re-run put the operator ahead on 5 of 5 seeds pairwise with no overlap, and the control is saturated within `0.031` of its own multiset ceiling of `0.3110`. *Scope: one parameter budget, one word length, no length-generalization check.*
- **The operator does not predict.** On the repaired chess bed, recalibrated resolution is `0.001469` against an oracle ceiling of `0.10117` — 1.45% of what the bed offers — and it is tied on that resolution by an eight-bin histogram of total piece count computed from the operator's own 769-float input, losing on Brier score, `0.6334` against `0.5020`.
- **The gate closes and nobody knows why.** Two proposed mechanisms — gradient starvation and step-0 zero density — are refuted by the same table built to demonstrate them; a third account (exactness as an eval-only property) and a fourth (a live gradient unfreezing a closed gate) were each pre-registered and killed in the three most recent commits to this repository. The cause remains open.
- **Twelve checks that could not fail have been catalogued** (`docs/PHASE_I1.md`), and each could not fail for a structural reason — a null space, a vacuous control, a grep with no true-positive path — rather than because training had settled anything. The standing practice this catalogue set: audit a table's fixed structure before trusting its trainable part.

---

## Diagrams

Each diagram below is meant to stand alone: read its caption first, then the picture.

<figure>
<svg viewBox="0 0 640 420" role="img" aria-label="A cube with axes beta, g and qk. Softmax attention sits at one corner, the unnormalized kernel at another, and the exact path product runs along one edge.">
  <defs>
    <style>
      .cube-edge{stroke:var(--rs-line2);stroke-width:1.4;fill:none}
      .cube-edge-back{stroke:var(--rs-line);stroke-width:1;fill:none;stroke-dasharray:3 3}
      .cube-lbl{fill:var(--rs-dim);font:12px var(--rs-mono,monospace)}
      .cube-pt{fill:var(--rs-ink)}
      .cube-cap{fill:var(--rs-dim);font:12.5px var(--rs-serif,serif)}
    </style>
  </defs>
  <text x="320" y="26" text-anchor="middle" class="cube-cap" style="fill:var(--rs-ink);font-weight:600">The operator family as a cube</text>

  <path class="cube-edge-back" d="M180 320 L180 160 M180 320 L380 320 M180 160 L380 160"/>
  <path class="cube-edge" d="M100 260 L300 260 L300 100 L100 100 Z"/>
  <path class="cube-edge" d="M300 260 L380 320 L380 160 L300 100"/>
  <path class="cube-edge" d="M100 100 L180 160 L380 160"/>
  <line x1="100" y1="260" x2="180" y2="320" stroke="var(--rs-unmet)" stroke-width="4" stroke-linecap="round"/>

  <circle class="cube-pt" cx="300" cy="100" r="6" fill="var(--rs-proved)"/>
  <text x="312" y="90" class="cube-lbl" fill="var(--rs-proved)">softmax attention</text>
  <text x="312" y="104" class="cube-lbl">β=1 · g off · qk on</text>

  <circle class="cube-pt" cx="100" cy="100" r="6" fill="var(--rs-link)"/>
  <text x="96" y="72" class="cube-lbl" fill="var(--rs-link)" text-anchor="middle">unnormalized kernel</text>
  <text x="96" y="86" class="cube-lbl" text-anchor="middle">β=0 · g off · qk on</text>

  <text x="140" y="345" class="cube-lbl" fill="var(--rs-unmet)">exact path product — β=0 · qk off · any g</text>

  <text x="240" y="392" class="cube-lbl" text-anchor="middle">β: 0 → 1 (total → mean)</text>
  <text x="70" y="200" class="cube-lbl" text-anchor="middle" transform="rotate(-90 70 200)">qk: off → on</text>
  <text x="392" y="230" class="cube-lbl">g: off → on</text>
</svg>
<figcaption><b>Shows:</b> the three switches (β, g, qk) as the axes of a cube; softmax attention and the unnormalized kernel are two corners (green, blue), the exact path product is the marked edge (amber). <b>Does not show:</b> which corner performs better at anything — this is a statement about what one family contains, proved in Lean (<code>three_corners_containment</code>), not a ranking.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 680 240" role="img" aria-label="An in-state enters a resolvent box on the left and an out-state exits on the right; no path between them is drawn, only the two states and the operator connecting them.">
  <defs>
    <style>
      .sm-box{fill:var(--rs-panel2);stroke:var(--rs-line2);stroke-width:1.4}
      .sm-lbl{fill:var(--rs-ink);font:13px var(--rs-mono,monospace)}
      .sm-dim{fill:var(--rs-dim);font:12px var(--rs-serif,serif)}
      .sm-arrow{stroke:var(--rs-link);stroke-width:2.2;fill:none;marker-end:url(#arrow)}
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="var(--rs-link)"/>
    </marker>
  </defs>
  <text x="340" y="26" text-anchor="middle" class="sm-lbl" style="fill:var(--rs-ink);font-weight:600">Forward only: in-state to out-state through the resolvent</text>

  <circle cx="90" cy="130" r="34" fill="none" stroke="var(--rs-ink)" stroke-width="1.6"/>
  <text x="90" y="135" text-anchor="middle" class="sm-lbl">in⟩</text>
  <text x="90" y="185" text-anchor="middle" class="sm-dim">asymptotically free,</text>
  <text x="90" y="200" text-anchor="middle" class="sm-dim">before</text>

  <rect class="sm-box" x="230" y="80" width="220" height="100" rx="8"/>
  <text x="340" y="122" text-anchor="middle" class="sm-lbl">T = (I − V G0)⁻¹ V</text>
  <text x="340" y="146" text-anchor="middle" class="sm-dim">the resolvent, causal branch</text>
  <text x="340" y="164" text-anchor="middle" class="sm-dim">(the +iε margin)</text>

  <circle cx="590" cy="130" r="34" fill="none" stroke="var(--rs-ink)" stroke-width="1.6"/>
  <text x="590" y="135" text-anchor="middle" class="sm-lbl">out⟩</text>
  <text x="590" y="185" text-anchor="middle" class="sm-dim">asymptotically free,</text>
  <text x="590" y="200" text-anchor="middle" class="sm-dim">after</text>

  <path class="sm-arrow" d="M124 130 L230 130"/>
  <path class="sm-arrow" d="M450 130 L556 130"/>
  <text x="620" y="45" class="sm-dim" text-anchor="end">no arrow runs right to left —</text>
  <text x="620" y="60" class="sm-dim" text-anchor="end">this is not an inverse-scattering read</text>
</svg>
<figcaption><b>Shows:</b> the S-matrix picture this project's resolvent read has the shape of — an in-state mapped to an out-state by the operator T, with no path drawn between them. <b>Does not show:</b> a wave, a scattering trajectory, or a reverse (out→in) inference — the read this project ships is forward-only, and inverse scattering is not attempted.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 680 260" role="img" aria-label="The hop expansion as a Neumann series: a sum of increasing powers of V G0 applied to V, with the spectral radius convergence condition stated on the figure.">
  <defs><style>
    .ns-lbl{fill:var(--rs-ink);font:14px var(--rs-mono,monospace)}
    .ns-dim{fill:var(--rs-dim);font:12px var(--rs-serif,serif)}
    .ns-term{fill:var(--rs-panel2);stroke:var(--rs-line2)}
  </style></defs>
  <text x="340" y="28" text-anchor="middle" class="ns-lbl" style="fill:var(--rs-ink);font-weight:600">Born series = Neumann series of the resolvent = the hop expansion</text>

  <text x="20" y="90" class="ns-lbl">T ≈</text>
  <rect class="ns-term" x="70" y="65" width="46" height="34" rx="5"/><text x="93" y="88" text-anchor="middle" class="ns-lbl" style="font-size:13px">V</text>
  <text x="122" y="88" class="ns-lbl">+</text>
  <rect class="ns-term" x="140" y="65" width="86" height="34" rx="5"/><text x="183" y="88" text-anchor="middle" class="ns-lbl" style="font-size:12px">VG0·V</text>
  <text x="232" y="88" class="ns-lbl">+</text>
  <rect class="ns-term" x="250" y="65" width="126" height="34" rx="5"/><text x="313" y="88" text-anchor="middle" class="ns-lbl" style="font-size:11.5px">VG0·VG0·V</text>
  <text x="382" y="88" class="ns-lbl">+ …</text>
  <text x="470" y="88" class="ns-dim">(k hops = k factors of VG0)</text>

  <text x="20" y="140" class="ns-dim">converges iff</text>
  <text x="145" y="140" class="ns-lbl">ρ(G0 V) &lt; 1</text>
  <text x="270" y="140" class="ns-dim">— same law as</text>
  <text x="405" y="140" class="ns-lbl">γ·ρ(P) &lt; 1</text>

  <g transform="translate(20,175)">
    <text class="ns-dim" x="0" y="0">measured, seed 0, n=12: ρ(G0V)=0.2596 (converges)</text>
    <line x1="0" y1="14" x2="400" y2="14" stroke="var(--rs-line)"/>
    <rect x="0" y="20" width="103.8" height="10" fill="var(--rs-proved)"/>
    <rect x="103.8" y="20" width="296.2" height="10" fill="var(--rs-line)"/>
    <text x="103.8" y="46" text-anchor="middle" class="ns-dim">1.0 (divergence boundary)</text>
    <text x="0" y="70" class="ns-dim">error at 5 / 10 / 20 hops: 1.8e-4 / 2.1e-7 / 2.8e-13</text>
  </g>
</svg>
<figcaption><b>Shows:</b> the hop expansion written as a finite sum of increasing powers of V·G0 applied to V, with the convergence condition ρ(G0V) &lt; 1 stated on the figure and satisfied by the worked instance (bar at 0.2596, well under the 1.0 boundary). <b>Does not show:</b> the divergent instance (ρ = 1.6688) — that failure mode is reported separately, corrected and unscored, in the text above.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 680 230" role="img" aria-label="A row of gates between six tokens; the path product multiplies every gate along a span, and one gate at exactly zero severs every path that crosses it.">
  <defs><style>
    .gp-tok{fill:var(--rs-panel2);stroke:var(--rs-line2);stroke-width:1.4}
    .gp-lbl{fill:var(--rs-ink);font:13px var(--rs-mono,monospace)}
    .gp-dim{fill:var(--rs-dim);font:11.5px var(--rs-serif,serif)}
    .gp-live{stroke:var(--rs-link);stroke-width:2}
    .gp-dead{stroke:var(--rs-unmet);stroke-width:2;stroke-dasharray:4 3}
  </style></defs>
  <text x="340" y="24" text-anchor="middle" class="gp-lbl" style="fill:var(--rs-ink);font-weight:600">G_ij = ∏ m_k along the path — one zero severs every span that crosses it</text>

  <g>
    <circle class="gp-tok" cx="60" cy="130" r="18"/><text x="60" y="135" text-anchor="middle" class="gp-lbl" style="font-size:12px">x₀</text>
    <circle class="gp-tok" cx="180" cy="130" r="18"/><text x="180" y="135" text-anchor="middle" class="gp-lbl" style="font-size:12px">x₁</text>
    <circle class="gp-tok" cx="300" cy="130" r="18"/><text x="300" y="135" text-anchor="middle" class="gp-lbl" style="font-size:12px">x₂</text>
    <circle class="gp-tok" cx="420" cy="130" r="18"/><text x="420" y="135" text-anchor="middle" class="gp-lbl" style="font-size:12px">x₃</text>
    <circle class="gp-tok" cx="540" cy="130" r="18"/><text x="540" y="135" text-anchor="middle" class="gp-lbl" style="font-size:12px">x₄</text>
    <circle class="gp-tok" cx="660" cy="130" r="18"/><text x="660" y="135" text-anchor="middle" class="gp-lbl" style="font-size:12px">x₅</text>
  </g>

  <line class="gp-live" x1="78" y1="130" x2="162" y2="130"/><text x="120" y="115" text-anchor="middle" class="gp-dim">m₁=0.9</text>
  <line class="gp-dead" x1="198" y1="130" x2="282" y2="130"/><text x="240" y="115" text-anchor="middle" class="gp-dim" fill="var(--rs-unmet)">m₂=0</text>
  <line class="gp-live" x1="318" y1="130" x2="402" y2="130"/><text x="360" y="115" text-anchor="middle" class="gp-dim">m₃=0.8</text>
  <line class="gp-live" x1="438" y1="130" x2="522" y2="130"/><text x="480" y="115" text-anchor="middle" class="gp-dim">m₄=0.95</text>
  <line class="gp-live" x1="558" y1="130" x2="642" y2="130"/><text x="600" y="115" text-anchor="middle" class="gp-dim">m₅=0.7</text>

  <path class="gp-dead" d="M60 170 Q300 230 660 170" fill="none"/>
  <text x="360" y="205" text-anchor="middle" class="gp-dim" fill="var(--rs-unmet)">G₅,₀ = m₁·m₂·m₃·m₄·m₅ = 0 exactly — every path crossing gate 2 is severed, not merely small</text>
</svg>
<figcaption><b>Shows:</b> the gate path product across six tokens; one factor at exactly zero (gate 2) sends every span crossing it — including the full x₀→x₅ path — to exactly zero, not asymptotically small. <b>Does not show:</b> a logit-space decay gate (the additive kind used elsewhere in the literature); Lean's <code>no_prefix_scan_represents_a_zero_gate</code> proves no such gate can do this, because <code>exp(Cᵢ−Cⱼ)</code> is never zero.</figcaption>
</figure>

<figure>
<svg viewBox="0 0 680 300" role="img" aria-label="A scoreboard of four rows: measured and holding, measured and refuted, tied or lost, and open, each with its number.">
  <defs><style>
    .sb-lbl{fill:var(--rs-ink);font:13px var(--rs-serif,serif)}
    .sb-dim{fill:var(--rs-dim);font:11.5px var(--rs-mono,monospace)}
    .sb-row{stroke:var(--rs-line);stroke-width:1}
  </style></defs>
  <text x="340" y="24" text-anchor="middle" class="sb-lbl" style="fill:var(--rs-ink);font-weight:600;font-family:var(--rs-mono,monospace)">The scoreboard: measured, refuted, open</text>

  <line class="sb-row" x1="20" y1="46" x2="660" y2="46"/>
  <circle cx="34" cy="70" r="7" fill="var(--rs-proved)"/>
  <text x="52" y="66" class="sb-lbl">Represents order — S5, bare integers, matched params</text>
  <text x="52" y="82" class="sb-dim">0.8620 vs control 0.2860 · control saturated at 0.031/0.3110 · scope: 1 budget, no length-gen</text>

  <line class="sb-row" x1="20" y1="102" x2="660" y2="102"/>
  <circle cx="34" cy="126" r="7" fill="var(--rs-unmet)"/>
  <text x="52" y="122" class="sb-lbl">Does not predict — repaired chess bed</text>
  <text x="52" y="138" class="sb-dim">resolution 0.001469 / ceiling 0.10117 (1.45%) · tied by an 8-bin histogram · Brier 0.6334 vs 0.5020 loss</text>

  <line class="sb-row" x1="20" y1="158" x2="660" y2="158"/>
  <circle cx="34" cy="182" r="7" fill="var(--rs-open)"/>
  <text x="52" y="178" class="sb-lbl">Gate closes, cause unknown — two accounts refuted by their own table</text>
  <text x="52" y="194" class="sb-dim">gradient starvation ✗ · step-0 zero density ✗ · eval-only exactness ✗ · live-gradient unfreeze ✗</text>

  <line class="sb-row" x1="20" y1="214" x2="660" y2="214"/>
  <circle cx="34" cy="238" r="7" fill="var(--rs-kill)"/>
  <text x="52" y="234" class="sb-lbl">Twelve checks catalogued that could not fail, each for a structural reason</text>
  <text x="52" y="250" class="sb-dim">null space · vacuous control · grep with no true-positive path · standing practice: audit structure first</text>

  <text x="20" y="280" class="sb-dim">This reconciliation (Wheeler 1937 / Lippmann–Schwinger) changes none of the four rows above.</text>
</svg>
<figcaption><b>Shows:</b> the four load-bearing verdicts of the project as they stand today — one measured and holding within its stated scope, one measured and lost, one open with its refuted explanations named, one a catalogue of checks that could not fail. <b>Does not show:</b> a trend or a forecast — each row is a single measurement with the command and file that produced it named in the linked pages, not an average or a projection.</figcaption>
</figure>

---

## Sources

- Wheeler, J. A. "On the Mathematical Description of Light Nuclei by the Method of Resonating Group Structure." *Physical Review* 52, no. 11 (1937): 1107–1122. DOI: [10.1103/PhysRev.52.1107](https://doi.org/10.1103/PhysRev.52.1107).
- Heisenberg, W. "Die beobachtbaren Größen in der Theorie der Elementarteilchen." *Zeitschrift für Physik* 120 (1943): 513–538; continued 120 (1943): 673–702; analytic-structure installment, *Zeitschrift für Physik* 123 (1944): 93–112.
- Lippmann, B. A., and J. Schwinger. "Variational Principles for Scattering Processes. I." *Physical Review* 79, no. 3 (1950): 469–480.
- This project's own resolvent read: `ceqjepa/operator.py`, convergence condition `γ·ρ(P) < 1`.
- Worked-instance script: `born_series_check.py` (this session's scratchpad; not part of the shipped package — a verification script, not a claim of a new numerical method).
