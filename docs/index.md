---
title: resolvent
rs: true
hide:
  - navigation
  - toc
---

<div class="rs story">

<header class="hero">
  <p class="over">A research record by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></p>
  <h1 class="title">resolvent</h1>
  <p class="thesis">Attention reads the next token. <br>This asks where a process <em>ends up</em>.</p>
  <p class="sub">One causal head that is softmax attention at one setting and the exact path product of a Markov chain at another, proved in Lean 4. Every result that failed stays on the record.</p>
  <a class="cta" href="#ch-hop">Scroll the story ↓</a><a class="cta ghost" href="experiment/">Read it in full</a>
</header>

<div class="scrolly">
  <div class="stage" aria-hidden="true"><svg viewBox="0 0 640 440" data-stage></svg><p class="stage-cap" data-stagecap></p></div>
  <div class="steps">

  <section class="step" id="ch-hop" data-step="hop">
    <p class="kick">01 · attention</p>
    <h2>Attention takes one hop.</h2>
    <p>The last token reads every earlier token directly, and splits its attention among them. The weights are proportions of one budget.</p>
  </section>

  <section class="step" data-step="chain">
    <p class="kick">02 · a chain</p>
    <h2>A chain takes every hop.</h2>
    <p>A Markov chain relays along a path, and the weight of a route is the <em>product</em> of its steps. The farther back, the more steps multiply.</p>
  </section>

  <section class="step" data-step="resolvent">
    <p class="kick">03 · the resolvent</p>
    <h2>One inverse sums every route.</h2>
    <p><span class="m">(I − gP)⁻¹ = I + gP + (gP)² + …</span> Causal masking makes it exact: one forward substitution, no iteration, no truncation.</p>
  </section>

  <section class="step" data-step="gate">
    <p class="kick">04 · the gate</p>
    <h2>Close one gate. Every path across it is exactly zero.</h2>
    <p>Not small. Zero. A decay in the logit can never do this: <span class="m">e<sup>Cᵢ−Cⱼ</sup></span> is never zero. Lean: <code>no_prefix_scan_represents_a_zero_gate</code>.</p>
  </section>

  <section class="step" data-step="cube">
    <p class="kick">05 · one head</p>
    <h2>Three switches. Three known operators.</h2>
    <p>β, g and qk span softmax attention, the unnormalized kernel and the exact path product. Proved containment: <code>three_corners_containment</code>.</p>
  </section>

  <section class="step" data-step="held">
    <p class="kick">06 · what held</p>
    <h2>It represents order.</h2>
    <p>Given S5 as bare integers, the operator scores <b>0.8620</b>. A control that cannot see order stops at <b>0.2860</b>, within 0.031 of its own ceiling. 404 parameters each, 5 of 5 seeds.</p>
  </section>

  <section class="step" data-step="died">
    <p class="kick">07 · what died</p>
    <h2>The language-model win belongs to position.</h2>
    <p>A zero-parameter ALiBi twin recovers <b>108.8%</b> of the win. A published forget gate recovers <b>100.2%</b>. And on chess, prediction reached <b>1.45%</b> of what the bed offers, tied by a histogram.</p>
  </section>

  <section class="step" data-step="proofs">
    <p class="kick">08 · the proofs</p>
    <h2>207 theorems hold it up.</h2>
    <p>166 theorems and 41 lemmas in Lean 4, 523 dependencies between them. <a href="atlas/">Walk the proof atlas →</a></p>
  </section>

  <section class="step" data-step="now">
    <p class="kick">09 · now</p>
    <h2>Deep chains, where no shortcut reaches.</h2>
    <p>Two test beds died to hand-built shortcuts. What survives is scoring only the far band, past every window a softmax stack can build. The resolvent learns the near chain; the far band is the open question. <a href="see/status/">Where it stands →</a></p>
  </section>

  </div>
</div>

<footer class="end">
  <h2>Read the record.</h2>
  <p>The dense version lives in the paper: every number, every control, every retraction.</p>
  <a class="cta" href="experiment/">The experiment, in full</a><a class="cta ghost" href="CEQ_SHAPE/">The shape paper</a><a class="cta ghost" href="https://github.com/teerthsharma/resolvent">GitHub</a>
</footer>

</div>
