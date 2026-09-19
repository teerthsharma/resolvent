---
title: resolvent
hide:
  - navigation
  - toc
---

<div class="rs">

<div class="hero">
    <p class="over">Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a> · research record</p>
    <h1 class="title">resolvent</h1>
    <p class="thesis">Softmax attention and Markov path composition are the same operator.</p>
    <p class="sub">One causal head, three switches. Softmax attention, unnormalized-kernel attention and the exact path product of a Markov chain are settings of it — proved in Lean&nbsp;4, matched bitwise in code, and every result that failed is kept on the record beside the ones that held.</p>
    <p class="pills">
      <span class="pill">Lean 4 <b>v4.7.0</b></span>
      <span class="pill"><b>134</b> theorems + <b>41</b> lemmas</span>
      <span class="pill">sorry <b>0</b></span>
      <span class="pill">corner tests <b>69/69</b></span>
      <span class="pill">Apache-2.0</span>
    </p>
    <a class="cta" href="#experiment">Read the experiment</a><a class="cta ghost" href="https://github.com/teerthsharma/resolvent">github.com/teerthsharma/resolvent</a>

    <figure style="margin-top:2.4rem">
      <svg viewBox="0 0 760 232" role="img" aria-label="Nine tokens on a line. The last token attends to every earlier one. A gate between tokens four and five is closed, so every attention path that crosses it is exactly zero; the three paths that do not cross it stay live.">
        <text class="t-dim" x="20" y="24">query t₈ reads every earlier token through the gates between them</text>
        <!-- dead arcs: from t8 (x=700) to t0..t4, crossing the closed gate m5 -->
        <path class="dead" d="M700 150 Q380 -62 60 150"/>
        <path class="dead" d="M700 150 Q420 -40 140 150"/>
        <path class="dead" d="M700 150 Q460 -18 220 150"/>
        <path class="dead" d="M700 150 Q500 4 300 150"/>
        <path class="dead" d="M700 150 Q540 26 380 150"/>
        <!-- live arcs: t8 to t5..t7 -->
        <path class="live draw" d="M700 150 Q580 48 460 150"/>
        <path class="live draw d2" d="M700 150 Q620 70 540 150"/>
        <path class="live draw d3" d="M700 150 Q660 92 620 150"/>
        <!-- the closed gate m5 between t4 (380) and t5 (460) -->
        <line class="gate" x1="420" y1="128" x2="420" y2="172"/>
        <g>
          <circle class="tok" cx="60" cy="150" r="13"/><circle class="tok" cx="140" cy="150" r="13"/>
          <circle class="tok" cx="220" cy="150" r="13"/><circle class="tok" cx="300" cy="150" r="13"/>
          <circle class="tok" cx="380" cy="150" r="13"/><circle class="tok" cx="460" cy="150" r="13"/>
          <circle class="tok" cx="540" cy="150" r="13"/><circle class="tok" cx="620" cy="150" r="13"/>
          <circle class="tok q" cx="700" cy="150" r="15"/>
        </g>
        <g class="t-serif" text-anchor="middle">
          <text x="60" y="190">t₀</text><text x="140" y="190">t₁</text><text x="220" y="190">t₂</text>
          <text x="300" y="190">t₃</text><text x="380" y="190">t₄</text><text x="460" y="190">t₅</text>
          <text x="540" y="190">t₆</text><text x="620" y="190">t₇</text><text x="700" y="190">t₈</text>
        </g>
        <text class="t-mono" x="420" y="218" text-anchor="middle" style="fill:var(--rs-kill)">gate m₅ = 0</text>
        <g class="t-dim">
          <line class="live" x1="520" y1="206" x2="546" y2="206" style="animation:none;stroke-dasharray:none"/><text x="552" y="210">live path</text>
          <line class="dead" x1="626" y1="206" x2="652" y2="206"/><text x="658" y="210">exactly 0</text>
        </g>
      </svg>
      <figcaption>The gate is a path product, <var>G<sub>ij</sub></var> = ∏<sub><var>k</var>=<var>j</var>+1..<var>i</var></sub> <var>m<sub>k</sub></var> <var>e</var><sup><var>i</var>θ<sub><var>k</var></sub></sup>, so one closed gate zeroes every path across it — exactly, not approximately.<br>A prefix scan in the logit cannot: <var>e</var><sup><var>C<sub>i</sub></var>−<var>C<sub>j</sub></var></sup> is never zero. Lean theorem <code>no_prefix_scan_represents_a_zero_gate</code>.</figcaption>
    </figure>

<pre class="result" aria-label="Measured results">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 <span class="g">resolvent</span> · measured 2026-09-11 · Windows 11 · Python 3.11.9 · torch 2.14.0 (CPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Lean 4 proofs          13 files · 134 theorems + 41 lemmas · <span class="g">0 sorry</span> · build exit 0
 Corners in code        <span class="g">69 / 69</span> tests pass            <span class="r">tests/arm_smprime, tests/arm_pl</span>
 Softmax corner         max |Δ| vs causal softmax     <span class="g">0.000e+00</span>
 Committor closed form  max |Δ|                       <span class="g">4.441e-16</span>
 Curl vs node models    share left unrepresented      mixed 0.6720 · pure curl 1.0000
 Planted negative       pure-gradient target          6.5e-16
 Refusal                sensitivity · specificity     <span class="g">100.00% · 100.00%</span>  (138 + 262)
 Prediction             beats a trivial baseline      <span class="a">not yet — open</span>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</pre>
</div>

<nav class="contents" aria-label="Sections">
  <a href="#experiment"><span class="n">1</span>The experiment</a>
  <a href="#operator"><span class="n">2</span>The operator</a>
  <a href="#reads"><span class="n">3</span>The reads</a>
  <a href="#progress"><span class="n">4</span>How it is going</a>
  <a href="#died"><span class="n">5</span>What died</a>
  <a href="#next"><span class="n">6</span>Where next</a>
  <a href="#documents"><span class="n">7</span>Documents</a>
  <a href="#reproduce"><span class="n">8</span>Reproduce</a>
</nav>


<div class="abstract">
  <p class="label">Abstract</p>
  <p>This is a research programme with one target: an attention mechanism that understands <em>consequences</em> — where a process ends up after an action, not only which token comes next — while standing as an equal to self-attention. Its object is a single causal head whose three switches span softmax attention, unnormalized-kernel attention and the exact path product of a Markov chain; the containments are proved in Lean&nbsp;4 and matched bitwise in float64. On that head sit two reads: a <em>resolvent read</em> that answers where a process ends up with one triangular solve, and refuses when a counterfactual has no defined answer, and a <em>graded read</em> that sees edge flows with curl, which no node-level model can represent at any size.</p>
  <p>The exact machinery holds. The learned side has not yet earned a win: over its first 387 commits (2026-08-25 to 2026-09-19), every matched comparison on prediction ended in a tie or a loss, and each is published beside the number that decided it. In the repository's own words, the family is <em>strictly more reachable, not yet more accurate</em>.</p>
  <p class="kw"><b>Keywords:</b> causal attention · softmax attention · path products · resolvent · committor functions · Hodge decomposition · selective prediction · pre-registration · formal verification · Lean 4</p>
</div>

<section id="experiment">
  <h2 class="sec"><span class="num">1</span>The experiment</h2>
  <p class="lede">Transformer attention mixes information across positions. Recurrences and Markov chains pass values along paths. The experiment asks whether one attention head can do both — and whether the path-following half gives a model something softmax attention lacks: reasoning about the consequences of an intervention.</p>

  <p><strong>The object.</strong> One causal head, <var>W<sub>ij</sub></var> = <var>G<sub>ij</sub></var> · exp(<var>qk</var> · <var>q<sub>i</sub></var>·<var>k<sub>j</sub></var>) ⁄ <var>Z<sub>i</sub></var><sup><var>β</var></sup>, with three switches: <code>β</code> (normalizer on or off), <code>g</code> (a multiplicative gate whose path product can close exactly) and <code>qk</code> (content comparison on or off). Three settings of those switches are three known operators, and that is a theorem, not an analogy (<a href="#operator">§2</a>).</p>

  <p><strong>The reads.</strong> Treat the causal attention matrix as a Markov chain, make declared boundary rows absorbing, and solve one triangular system: the answer is the committor, the probability that each position ends in each boundary set. When a counterfactual has no defined answer, the read refuses instead of returning a number. A second read covers edge flows with curl (<a href="#reads">§3</a>).</p>

  <p><strong>What is owned by prior work.</strong> The resolvent read over attention is ChaCAL's (Fagnou et al., EMNLP 2024). The delta this repository claims is the absorbing boundary rows for <var>K</var> constraint sets, the committor read, the interventional re-solve, the certificate and the Lean proofs — stated in <a href="CEQ_SHAPE/">the shape paper</a> and <a href="canon/CHARTER/">the charter</a> rather than left for a reviewer to find.</p>

  <p><strong>The method is adversarial bookkeeping.</strong> Predictions are filed with counter-predictions before any run, kill thresholds are frozen, every check carries a planted negative that must fail, and when a kill fires the result is published. The error record holds 74 failure mechanisms, 12 withdrawn constants and a retraction log (<a href="#died">§5</a>). The two applications the author names are chess and prediction-market trades.</p>

  <h3>The north star, and where it stands</h3>
  <p>The goal as the author states it has three conditions, all required: an attention mechanism that <strong>understands causality and consequences</strong>, <strong>predicts better than anything before it</strong> on a named task against a named opponent, and <strong>carries the weight of self-attention or JEPA</strong>. The repository's sentence of record says the same thing in engineering terms:</p>
  <div class="note"><p><em>"Attention EQUAL to self-attention on its own ground, built FROM softmax and AdamW, capable on ground they cannot occupy — predicting the NEXT STATE toward equilibrium, not the next token."</em><br><span style="color:var(--rs-dim);font-size:.72rem">CEQ_V20_R15_CONTRACT.md:52-55</span></p></div>

  <div class="grid">
    <div class="card star part">
      <p class="k">Condition 1</p>
      <h4>Understands causality and consequences <span class="chip part">partial</span></h4>
      <p><strong style="color:var(--rs-ink)">Holds when the operator is given.</strong> The committor is exact to 4.441e-16; refusal scores 100% / 100% on 138 + 262 cases; on 13,479 human K+Q-vs-K positions the exact committor separates consequence swaps from meaning-preserving ones at a ratio of 2.501651.</p>
      <p><strong style="color:var(--rs-ink)">Not yet learned.</strong> The trained read's ratio is 0.999745, below the planted negative's 1.073511. A trained operator head emitted κ 1.34 against a true 93.51, and the OOD-κ claim died on 5/5 seeds.</p>
    </div>
    <div class="card star unmet">
      <p class="k">Condition 2</p>
      <h4>Predicts better than anything before it <span class="chip unmet">unmet</span></h4>
      <p><strong style="color:var(--rs-ink)">Every matched race is a tie or a loss.</strong> On a Kaggle T4 over 115,628 Lichess games, a zero-parameter heuristic matched human moves at 0.3401 [0.3321, 0.3482]; the trained model reached 0.2882 [0.2806, 0.2959].</p>
      <p>Against a matched softmax head the gate ties, at 1.91–2.84× the cost. The one narrow gain: a free β lowers held-out log-loss by 0.046–0.063 nats with no change in exact accuracy.</p>
    </div>
    <div class="card star part">
      <p class="k">Condition 3</p>
      <h4>Carries the weight of self-attention or JEPA <span class="chip part">partial</span></h4>
      <p><strong style="color:var(--rs-ink)">Parity by containment is earned.</strong> The β = 1 corner is softmax to 0.000e+00, and containment survives training (Δ exact +0.00000 at 3 seeds).</p>
      <p><strong style="color:var(--rs-ink)">Standing is not.</strong> The largest run is one T4, no fused kernel exists, the family runs at 1.17–2.84× softmax wall clock, and the JEPA target moved with its encoder by 4.1101× on the synthetic bed and 136.7950× on Kaggle.</p>
    </div>
  </div>
</section>

<section id="operator">
  <h2 class="sec"><span class="num">2</span>The operator: one head, three switches</h2>
  <p class="lede">Attention mixes <em>weights across positions</em>. Recurrences and Markov chains compose <em>values along paths</em>. Architectures usually pick one side. <code>ceq/arm_smprime.py</code> implements a single causal head in which both sides are settings of the same three switches.</p>

  <div class="eq" role="math" aria-label="The family">
    <div class="row"><span class="lhs"><var>W<sub>ij</sub></var> =</span><span><var>G<sub>ij</sub></var> · exp(<var>qk</var> · <var>q<sub>i</sub></var>·<var>k<sub>j</sub></var>) ⁄ <var>Z<sub>i</sub></var><sup><var>β</var></sup></span><span class="note">j ≤ i, and 0 above the diagonal</span></div>
    <div class="row"><span class="lhs"><var>G<sub>ij</sub></var> =</span><span>∏<sub><var>k</var> = <var>j</var>+1 … <var>i</var></sub> <var>m<sub>k</sub></var> · <var>e</var><sup><var>i</var>θ<sub><var>k</var></sub></sup></span><span class="note">the gate's path product, m<sub>k</sub> ∈ [0, 1]</span></div>
    <div class="row"><span class="lhs"><var>Z<sub>i</sub></var> =</span><span>Σ<sub><var>j</var> ≤ <var>i</var></sub> |<var>G<sub>ij</sub></var>| · exp(<var>qk</var> · <var>q<sub>i</sub></var>·<var>k<sub>j</sub></var>)</span><span class="note">the row normalizer</span></div>
  </div>

  <figure>
    <svg viewBox="-40 0 680 318" role="img" aria-label="A cube whose three axes are the switches beta, g and qk. Softmax attention is the corner beta one, g off, qk on. Unnormalized-kernel attention is the corner beta zero, g off, qk on. The exact path product is the whole edge where beta is zero and qk is off, for any gate.">
      <!-- back edges -->
      <path class="edge back" d="M220 210 L420 210 M220 210 L220 60 M220 210 L140 270"/>
      <!-- front + remaining edges -->
      <path class="edge" d="M140 270 L340 270 L340 120 L140 120 Z M340 270 L420 210 L420 60 L340 120 M140 120 L220 60 L420 60"/>
      <!-- exact path product: the edge beta=0, qk off, any g -->
      <line x1="140" y1="270" x2="220" y2="210" stroke="var(--rs-kill)" stroke-width="5" stroke-linecap="round"/>
      <!-- the two attention corners -->
      <circle cx="340" cy="120" r="8" fill="var(--rs-proved)"/>
      <circle cx="140" cy="120" r="8" fill="var(--rs-link)"/>
      <g class="t-mono">
        <text x="352" y="112">softmax attention</text>
        <text x="128" y="112" text-anchor="end">unnormalized kernel</text>
        <text x="236" y="246" style="fill:var(--rs-kill)">exact path product</text>
      </g>
      <g class="t-dim">
        <text x="352" y="130">β = 1 · g off · qk on</text>
        <text x="128" y="130" text-anchor="end">β = 0 · g off · qk on</text>
        <text x="236" y="263">β = 0 · qk off · any g</text>
        <text x="240" y="300" text-anchor="middle">β : 0 → 1   (a total → a mean)</text>
        <text x="96" y="200" text-anchor="middle" transform="rotate(-90 96 200)">qk : off → on</text>
        <text x="442" y="252">g : off → on</text>
        <text x="442" y="268">(the path-product gate)</text>
      </g>
    </svg>
    <figcaption>The three switches as the axes of a cube. The named operators are two corners and one edge of it. Lean: <code>three_corners_containment</code>, <code>corners_are_distinct</code>.</figcaption>
  </figure>

  <div class="tw"><table class="t">
    <thead><tr><th>switch</th><th>off</th><th>on</th><th>what it decides</th></tr></thead>
    <tbody>
      <tr><td><code>qk</code></td><td><var>W</var> ignores content</td><td>dot-product logits</td><td><strong>Whether positions are compared at all.</strong> With <code>qk</code> off the head is pure structure.</td></tr>
      <tr><td><code>g</code></td><td><var>G</var> ≡ 1</td><td>path product ∏ <var>m<sub>k</sub></var> <var>e</var><sup><var>i</var>θ<sub><var>k</var></sub></sup></td><td><strong>Whether values compose along a path.</strong> A gate at zero closes the path exactly.</td></tr>
      <tr><td><code>β</code></td><td>β = 0, no normalizer</td><td>β = 1, rows sum to 1</td><td><strong>Whether the read is a mean or a total.</strong> The read carries <var>N</var><sup>1−β</sup> in the token count: β = 1 is intensive, β = 0 extensive.</td></tr>
    </tbody>
  </table></div>

  <div class="note warn"><p><strong>The β = 0 corner is not O(n) "linear attention".</strong> It is the unnormalized exponential kernel <code>exp(q·k)</code>, which has no finite feature map: at fixed <code>dk = 8</code> the numerical rank of the score matrix reads 8, 16, 32, 63, 126, 252 for <code>n = 8 … 256</code>. Setting β = 0 removes the division by <var>Z</var> and nothing else, so the corner costs what softmax costs. An earlier README called it linear attention; the Lean definition never did.</p></div>

  <h3>Every corner is a theorem, and every theorem has a test</h3>
  <div class="tw"><table class="t">
    <thead><tr><th>Lean theorem</th><th>what it proves</th></tr></thead>
    <tbody>
      <tr><td><code>three_corners_containment</code></td><td>one family (β, g, qk) contains all three operators; the softmax corner matches causal softmax to <code>0.000e+00</code></td></tr>
      <tr><td><code>corners_are_distinct</code></td><td>the corners are different operators: at (i, j) = (1, 0) with gate and qk off, β = 1 reads 1/2 and β = 0 reads 1</td></tr>
      <tr><td><code>gate_zero_beta_zero_is_linear_attention</code></td><td>with the gate off, β alone decides softmax-class membership</td></tr>
      <tr><td><code>no_prefix_scan_represents_a_zero_gate</code></td><td><var>e</var><sup><var>C<sub>i</sub></var>−<var>C<sub>j</sub></var></sup> is never zero, so no prefix scan represents a closed gate; the path product does</td></tr>
      <tr><td><code>Asink_computes_chain</code></td><td>a causal softmax head reproduces a chain's label exactly, for every gate with <var>a<sub>k</sub></var> ≠ 1</td></tr>
    </tbody>
  </table></div>
  <p>The theorems live in <a href="https://github.com/teerthsharma/resolvent/tree/master/lean/CEQ"><code>lean/CEQ/</code></a> (13 files, zero <code>sorry</code>); the float64 counterparts are <code>ceq/arm_smprime.py</code> and <code>ceq/arm_pl.py</code>, checked against each statement bitwise at the corners.</p>
</section>

<section id="reads">
  <h2 class="sec"><span class="num">3</span>The reads: where a process ends up, and what nodes cannot see</h2>
  <p class="lede">The head is an operator. The two reads turn it into answers to questions attention does not usually ask.</p>

  <h3>The resolvent read — one triangular solve</h3>
  <p><code>ceqjepa/operator.py</code> builds a causal row-stochastic softmax matrix <var>P</var>, makes declared boundary rows absorbing, and reads with triangular solves. With <code>teleport = 0</code> it matches causal softmax to <code>0.000e+00</code>.</p>
  <div class="eq" role="math" aria-label="The resolvent read">
    <div class="row"><span class="lhs">state read</span><span><var>z</var> = (<var>I</var> − <var>g</var><var>P</var>)<sup>−1</sup> <var>Ṽ</var></span></div>
    <div class="row"><span class="lhs">committor</span><span><var>q</var> = (<var>I</var> − <var>Q</var>)<sup>−1</sup> <var>R</var></span><span class="note">probability of ending in each boundary state</span></div>
  </div>
  <p>Refusal is built into the solve rather than bolted on. A singular transient block raises <code>SingularTransientBlockError</code> instead of returning a number; a non-finite logit raises <code>ValueError</code> instead of spreading NaN into <var>q</var>. Every counterfactual gets one of three verdicts — <strong>UNDEFINED</strong> is refused with its reason, <strong>NULL</strong> is answered with an exact zero, <strong>DEFINED</strong> is answered — scored against an independent reachability oracle:</p>
  <div class="tw"><table class="t">
    <thead><tr><th>rule</th><th>sensitivity</th><th>specificity</th></tr></thead>
    <tbody>
      <tr><td><strong>resolvent (reachability)</strong></td><td class="n yes">100.00%</td><td class="n yes">100.00%</td></tr>
      <tr><td>refuse everything</td><td class="n">100.00%</td><td class="n no">0.00%</td></tr>
      <tr><td>answer everything</td><td class="n no">0.00%</td><td class="n">100.00%</td></tr>
    </tbody>
  </table></div>
  <p style="color:var(--rs-dim);font-size:14.5px">138 undefined and 262 defined cases, <code>python -m ceqjepa.dr1</code>. Both trivial rules are printed beside it, because a refusal score means nothing without them.</p>

  <h3>The graded read — edge flows with curl</h3>
  <p>Data is graded: nodes, edges (relations) and triangles (interactions). The edge space splits three ways by Hodge decomposition, <var>R</var><sup><var>E</var></sup> = im(<var>d</var><sub>0</sub>) ⊕ im(<var>d</var><sub>1</sub><sup>⊤</sup>) ⊕ ker(<var>L</var><sub>1</sub>) — gradient, curl, harmonic. Any model that reads an edge as a difference of node values lives in im(<var>d</var><sub>0</sub>), so the curl part of a target is outside its range at any width, depth or budget. <code>ceqjepa/dr1.py</code> measures that boundary as the least-squares optimum over the whole node-level class, on a 9-node, 16-edge, 8-triangle complex:</p>
  <figure>
    <div class="bars" role="img" aria-label="Share of the target a node-level model cannot represent: pure gradient 6.5e-16, mixed 0.6720, pure curl 1.0000">
      <span class="lab">pure gradient <small>(planted negative)</small></span><span class="track"><span class="fill" style="width:0"></span></span><span class="val">6.5e-16</span>
      <span class="lab">mixed</span><span class="track"><span class="fill" style="width:67.2%"></span></span><span class="val">0.6720</span>
      <span class="lab">pure curl</span><span class="track"><span class="fill" style="width:100%"></span></span><span class="val">1.0000</span>
    </div>
    <figcaption>Share of each target that no node-level model can represent — the optimum of the whole class, not one trained network. The planted negative must read zero, and does.</figcaption>
  </figure>
</section>

<section id="progress">
  <h2 class="sec"><span class="num">4</span>How it is going</h2>
  <p class="lede">Nine phases, grouped from the commit log. Each ends on the number that decided it. A hollow amber marker is a claim that died; a green one is something that was built and still stands.</p>

  <ol class="tl">
    <li class="dead">
      <span class="when">2026-08-25 → 08-26 · 66 commits · dfc1591 → 33cb36a</span>
      <h4>A signed operator that did not work</h4>
      <p>A signed, strictly causal, denominator-free multi-hop path sum, with hops routed through content-selected pivots. The pre-registered kill fired on the shipped operator. The headlines that had said otherwise — a flat slope, a 61× separation — measured an operator the shipped path never called.</p>
      <span class="num">sign-flip slope −1.298 against a bar of −0.3</span>
    </li>
    <li class="dead">
      <span class="when">2026-08-26 · 72 commits · 8ba52e1 → 74e5590</span>
      <h4>Certificates, and the finding that nothing had been decided</h4>
      <p>Hilbert-metric contraction certificates died three ways, the measured diameter running 11–33× its prediction. The larger finding: the deciding measurement had been taken zero times, because every oracle was a closed-form function of its input, and every task asked for one scalar at one position — where one attention layer is provably optimal. The project reversed course and chose to build on softmax, not against it.</p>
      <span class="num">37% of the way to the deliverable: engineering near 80%, the central claim near 5%</span>
    </li>
    <li class="dead">
      <span class="when">2026-08-30 → 08-31 · 143 commits · 977f715 → ecbedf7</span>
      <h4>The planet rounds hit a data-budget wall</h4>
      <p>On the C1 vector corpus the first creditable contrast went against the family: at t* = 2 softmax read 0.996745 against 0.999082 and 1.000150 for the two CEQ arms. At t* = 32 every arm sat at or above predict-the-mean — and round 10 showed the wall was data, not architecture, by predicting where softmax becomes learnable before measuring it.</p>
      <span class="num">predicted n* = 48,823, landed within 0.7%</span>
    </li>
    <li class="built">
      <span class="when">2026-08-31 → 09-01 · 61 commits · 9082e99 → 207e7b9</span>
      <h4>Lean enters: parity refuted, the corners proved</h4>
      <p>The contract said a zero gate gives bitwise standard attention. Lean refuted it — <code>gate_zero_not_stochastic</code>, row <var>i</var> sums to <var>i</var>+1 — and the repair became <code>three_corners_containment</code>, with β as the switch that decides softmax membership. The R1 deciding cell did not cross its floor.</p>
      <span class="num">R1 0.829151, CI [0.617075, 1.041227] against a floor of 0.7071</span>
    </li>
    <li class="built">
      <span class="when">2026-09-04 → 09-06 · 4 commits · aa38e11 → 527ffca</span>
      <h4>The shape paper, the declutter, the canon</h4>
      <p>Round 15 earned 2 of a possible 44 scoreboard points. The shape paper conceded that the resolvent read is ChaCAL's and that no CEQ arm had yet been trained. Fifteen rounds of process documents left the tree, and the canon was written: eight books of verses, each carrying its own kill and a pre-derived replacement.</p>
      <span class="num">350 files and 159,507 lines removed · 188 verses</span>
    </li>
    <li class="built">
      <span class="when">2026-09-11 → 09-13 · 5 commits · e238db6 → c9a9434</span>
      <h4>Pre-registration on real data, and the public release</h4>
      <p>REMOVAL-ECHO was registered before any data was downloaded. On Perturb-seq, 118 of 131 gene pairs were non-additive and the prediction held; on NBA injuries the counter-prediction held instead. The README was rewritten around the resolvent family and the repository went public on 2026-09-13.</p>
      <span class="num">D1 0.9008 [0.8376, 0.9411] held · D2 0.979 counter held (commit 6aef750)</span>
    </li>
    <li class="dead">
      <span class="when">2026-09-13 → 09-14 · 11 commits · e3d56cb → b5190d4</span>
      <h4>A GPU run, and a heuristic that won</h4>
      <p>A rule assigned each latent coordinate its own attention corner, β = 1 − α, and a π-JEPA encoder was trained through it on a Kaggle T4 over 115,628 Lichess games. A zero-parameter box-shrinking heuristic beat it. The rule was retired on three grounds, and the component ledger was born: every future run carries a frozen-random arm and a trivial baseline.</p>
      <span class="num">heuristic 0.3401 against model 0.2882, disjoint intervals</span>
    </li>
    <li class="dead">
      <span class="when">2026-09-14 · 23 commits · 468bc85 → 88a7388</span>
      <h4>Component rounds: every lead met its counter</h4>
      <p>The gate ties softmax at 1.91–2.84× the cost. An encoder change closed 91.7161% of the committor gap. The consequence-swap headline's zero was an identity of the definition. The depth ladder turned out to measure optimiser steps, not depth. On BED-H a GRU abstaining on its own entropy matched the read's refusal. One lead survived: an estimated operator becomes solvable once whitened, 0.030243 against 0.348321.</p>
      <span class="num">2-layer stack 0.0555 → 0.3005 from optimiser steps alone · GRU recall 0.8959</span>
    </li>
    <li class="dead">
      <span class="when">2026-09-19 · 2 commits · 60bdcc9 → 2e09d51</span>
      <h4>Issue closeouts</h4>
      <p>The logit-Nash stance was retired over five seeds, including with both of its known defects repaired; one of those defects was a learned parameter no code ever read. The SPRT became torch-free.</p>
      <span class="num">nash 5.2888 ± 0.6041 against signed 2.7333 ± 1.0235, 0/5 seeds</span>
    </li>
    <li class="open">
      <span class="when">now</span>
      <h4>The exact side holds; the learned side is the open question</h4>
      <p>The exact identities still hold to float precision. What keeps failing is the learned part — a trained operator head (κ 1.34 against a true 93.51), a trained read (swap ratio 0.999745), a trained encoder — and every claim that the exact read's advantage carries over to a trained model. That is what decides the next move (<a href="#next">§6</a>).</p>
    </li>
  </ol>
</section>

<section id="died">
  <h2 class="sec"><span class="num">5</span>What died, and what killed it</h2>
  <p class="lede">A result nobody can check is worth nothing, so the failures are published with the same precision as the successes. Each row below was a claim this repository made, and each was withdrawn by its own instruments.</p>

  <div class="tw"><table class="t">
    <thead><tr><th>claim</th><th>what killed it</th><th>the number</th></tr></thead>
    <tbody>
      <tr><td>A signed, pivot-routed path sum reduces sign flips at scale</td><td>pre-registered slope kill on the shipped operator</td><td class="n">−1.298 vs bar −0.3</td></tr>
      <tr><td>A zero gate gives bitwise standard attention</td><td>Lean: <code>gate_zero_not_stochastic</code></td><td class="n">row <var>i</var> sums to <var>i</var>+1</td></tr>
      <tr><td>Theorem 4: a resolvent read is depth-separated from a fixed-depth stack</td><td>fitted coefficients; Cayley–Hamilton caps any separation at <var>n</var></td><td class="n">fit 1.448e-02% vs 112.6157%</td></tr>
      <tr><td>Under intervention a resolvent read's error is κ-invariant while a direct read's grows</td><td>pre-registered kill, 5/5 seeds, wrong sign</td><td class="n">OOD MAE: solve 0.0741, direct 0.0718</td></tr>
      <tr><td>κ = 164.25 on the demo's 14×14 grid</td><td>a hard-coded string; nothing ever computed it</td><td class="n">fabricated</td></tr>
      <tr><td>Spearman ρ = 0.743864 on U1/N3</td><td>no producer has ever existed</td><td class="n">struck</td></tr>
      <tr><td>Assign each coordinate its corner by β = 1 − α</td><td>no fixed point on the value axis; ranked 21st of 35 on its own null</td><td class="n">slope −1.0027</td></tr>
      <tr><td>A trained π-JEPA predicts human chess moves</td><td>a zero-parameter heuristic won</td><td class="n">0.3401 vs 0.2882</td></tr>
      <tr><td>The consequence swap moves the read and the null does not</td><td>the null's zero was an identity of the definition</td><td class="n">learned 0.999745 vs negative 1.073511</td></tr>
      <tr><td>The β = 0 corner is linear attention</td><td>no finite feature map: rank tracks <var>n</var></td><td class="n">rank 8 … 252</td></tr>
      <tr><td>Depth buys reach (stacks of 1/2/4/8 layers)</td><td>the ladder measured optimiser steps</td><td class="n">2-layer 0.0555 → 0.3005</td></tr>
      <tr><td>The read adds an architectural refusal channel</td><td>pre-registered counter held: a GRU's own entropy does it</td><td class="n">recall 0.8959 @ 0.8903</td></tr>
      <tr><td>A logit-Nash stance composes two sign flips</td><td>both kills, 5/5 seeds, every configuration</td><td class="n">5.2888 vs 2.7333</td></tr>
      <tr><td>Smoothing Sherman–Morrison with a teleport is harmless</td><td>it attenuates the interventional signal</td><td class="n">−43.1%</td></tr>
    </tbody>
  </table></div>

  <p>Most of what stands today was built by one of these deaths. The refuted parity clause became the β switch. The prefix-scan limit became the path-product gate and a theorem about it. The smoothed denominator became refusal as a decision. A raw grep that miscounted the proofs became a counting script that must first count a planted file correctly.</p>

  <div class="grid">
    <a class="card" href="ledgers/mistakes/" style="display:block;color:inherit;text-decoration:none">
      <p class="k">MISTAKES.md</p><h4>74 failure mechanisms</h4>
      <p>V 30 vacuous controls · P 16 provenance · M 21 measurement · D 7 design. Each with the instance, the rule, and the check that stops it recurring.</p>
    </a>
    <a class="card" href="ledgers/struck/" style="display:block;color:inherit;text-decoration:none">
      <p class="k">STRUCK.md</p><h4>12 withdrawn constants</h4>
      <p>Rendered from a registry the test suite enforces, so a struck number cannot quietly ship again.</p>
    </a>
    <a class="card" href="FAILS/" style="display:block;color:inherit;text-decoration:none">
      <p class="k">docs/FAILS.md</p><h4>Retracted, broken, open</h4>
      <p>Retractions, commands that do not run clean, measured negatives, known defects, and claims that were never verified.</p>
    </a>
  </div>
</section>

<section id="next">
  <h2 class="sec"><span class="num">6</span>Where it goes next</h2>
  <p class="lede">The open fronts, each with the bar it has to clear, and the one move chosen from them.</p>

  <div class="tw"><table class="t">
    <thead><tr><th>front</th><th>the standing bar</th><th>state</th></tr></thead>
    <tbody>
      <tr><td>Prediction leg of the north star</td><td>beat a trivial baseline at the family's own prediction task</td><td><span class="chip unmet">open</span></td></tr>
      <tr><td>BED-H in-context identification</td><td>the prior-mean-P baseline, 0.0852; no baseline built so far reaches it</td><td><span class="chip open">open</span></td></tr>
      <tr><td>The one-solve claim's loss condition</td><td>a stack of depth ≤ 2 reaching 0.90; the best so far reads 0.4858</td><td><span class="chip open">not fired</span></td></tr>
      <tr><td>Depth race, 160-goal point</td><td>extrapolated 0.4810, asymptote ≈ 0.4923 against the solve's 1.0000</td><td><span class="chip open">not run</span></td></tr>
      <tr><td>Committor read, component P3</td><td>+0.8851; the encoder's widest layer reads +0.8424</td><td><span class="chip open">open</span></td></tr>
      <tr><td>Joint-training gate</td><td>P1 pass, P2 and P3 answered, P4 fails at 0.9997, so joint training stays blocked</td><td><span class="chip part">blocked</span></td></tr>
      <tr><td>Canon books 00 and 07</td><td>listed in the charter as the north-star decision tree and the consolidated attacks</td><td><span class="chip open">unwritten</span></td></tr>
      <tr><td>A pinned environment</td><td>the Hugging Face tests fail on a torch drift, and the full suite has never completed on one CPU</td><td><span class="chip open">open</span></td></tr>
    </tbody>
  </table></div>

  <div class="note open">
    <p style="font:600 .62rem var(--rs-mono);letter-spacing:.14em;text-transform:uppercase;color:var(--rs-open);margin:0 0 .5rem">The next move</p>
    <p><strong>Race an estimated operator, not the true one.</strong> Every exact-read win on this page was handed the true operator. That makes the solve an oracle rather than a model, which the error ledger files as D-2, so none of those wins can tick the prediction row. The next run estimates the operator from the same transition stream the opponent trains on — the whitened recursive-least-squares estimator already measured at its sampling floor, 0.030243 against Hebbian 0.348321 on a 10,000-transition stream — and races one exact solve on that estimate against a masked attention stack at matched parameters, on goals neither has seen.</p>
    <p><strong>Registered before it runs:</strong> the bar is the best stack published so far on that bed, 0.4858, at matched transitions across five seeds; the kill is the estimated solve at or below it; the planted negative is a shuffled stream, which must drive the solve to the floor; the ceiling is the true-operator solve at 1.0000.</p>
    <p><strong>Why this one.</strong> It is the only open comparison that could put a tick in the prediction row honestly — a named opponent, a named task, no oracle — and it tests the step where the recent kills have landed: from a given operator to a learned one. It is also cheap: an estimator, a triangular solve and a stack that already has a pinned test set all run on a CPU. If it wins, the family has its first prediction result. If it dies, the record learns that the read's exactness does not survive estimation, and the claim settles where the README already puts it: strictly more reachable, not more accurate.</p>
  </div>
</section>

<section id="documents">
  <h2 class="sec"><span class="num">7</span>The documents</h2>
  <p class="lede">Every document in the repository that carries a measurement, a limitation or a decision is rendered on this site. The tabs above hold the same set.</p>

  <h3>The record</h3>
  <div class="docs">
    <a class="doc" href="FAILS/"><span class="p">docs/FAILS.md</span><span class="d">Everything retracted, broken or unresolved, with commands, exit codes and causes.</span></a>
    <a class="doc" href="COMPONENT_LEDGER/"><span class="p">docs/COMPONENT_LEDGER.md</span><span class="d">P1–P4 component bars, their status, and the gate that blocks joint training until each clears alone.</span></a>
    <a class="doc" href="PLAN/"><span class="p">docs/PLAN.md</span><span class="d">The programme: phases 0–6, a fourteen-evening critical path, bets A–M and the standing attacks.</span></a>
    <a class="doc" href="CORNER_RULE_RETIREMENT/"><span class="p">docs/CORNER_RULE_RETIREMENT.md</span><span class="d">Why β = 1 − α was retired, on three independent grounds.</span></a>
    <a class="doc" href="PI_JEPA_KAGGLE_CARD/"><span class="p">docs/PI_JEPA_KAGGLE_CARD.md</span><span class="d">The T4 run over 115,628 Lichess games, opening on the heuristic that won.</span></a>
    <a class="doc" href="DR2_CARD/"><span class="p">docs/DR2_CARD.md</span><span class="d">Four claims scored: assignment rule, length, refusal channel, and the consequence swap with its correction.</span></a>
  </div>

  <h3>Pre-registrations</h3>
  <div class="docs">
    <a class="doc" href="BED_H_PREREGISTRATION/"><span class="p">docs/BED_H_PREREGISTRATION.md</span><span class="d">BED-H registered before it was built, two correction passes, and the counter that held.</span></a>
    <a class="doc" href="prereg/REMOVAL_ECHO_v1/"><span class="p">docs/prereg/REMOVAL_ECHO_v1.md</span><span class="d">Möbius non-additivity after dated removals: predictions, counters, kills and must-fires, hashed before any download.</span></a>
  </div>

  <h3>Ledgers</h3>
  <div class="docs">
    <a class="doc" href="ledgers/mistakes/"><span class="p">MISTAKES.md</span><span class="d">74 failure mechanisms by class, with instance, rule and check, and the laws they paid for.</span></a>
    <a class="doc" href="ledgers/struck/"><span class="p">STRUCK.md</span><span class="d">12 withdrawn constants, rendered from an enforced registry.</span></a>
    <a class="doc" href="ledgers/mathematics/"><span class="p">MATHEMATICS.md</span><span class="d">The theory of record: thesis, equilibrium task family, depth law, contraction certificate, Kirchhoff oracle.</span></a>
    <a class="doc" href="ledgers/model-card/"><span class="p">MODEL_CARD.md</span><span class="d">The signed operator's negative result and the unfilled v17-K template, slot by slot.</span></a>
    <a class="doc" href="ledgers/costs/"><span class="p">COSTS.md</span><span class="d">Device certificates, dataset hashes and the cost model for the v17-K run.</span></a>
    <a class="doc" href="ledgers/v17k-rulings/"><span class="p">V17K_RULINGS.md</span><span class="d">The Gate-0 ruling ledger, with its open items and a stated circularity.</span></a>
    <a class="doc" href="ledgers/v16-calibration/"><span class="p">V16_CALIBRATION.md</span><span class="d">The calibration column: 7 of 8 signed predictions optimistic, p = 0.0352.</span></a>
  </div>

  <h3>The canon</h3>
  <p style="color:var(--rs-dim);font-size:.8rem">Written once and never edited: every verse states a claim, its evidence, its kill and a replacement derived in advance. Only the corrections log changes. Books 00 and 07 are listed in the charter and not yet written.</p>
  <div class="docs">
    <a class="doc" href="canon/CHARTER/"><span class="p">canon/CHARTER.md</span><span class="d">The constitution: the only licensed meanings of "more accurate" and "faster to train", the verse format, a 27-row defect census.</span></a>
    <a class="doc" href="canon/01_THEORY_ACCURACY/"><span class="p">01 Theory of accuracy</span><span class="d">22 verses: the exact class of the read, approximation bounds, obstructions, the ChaCAL delta.</span></a>
    <a class="doc" href="canon/02_THEORY_TRAINING/"><span class="p">02 Theory of training</span><span class="d">13 verses: when AdamW finds the resolvent before a depth-L stack learns L hops.</span></a>
    <a class="doc" href="canon/03_KERNEL/"><span class="p">03 Kernel</span><span class="d">19 verses: the fused and chunked solve, its cost law, determinism.</span></a>
    <a class="doc" href="canon/04_BEDS_AND_INSTRUMENTS/"><span class="p">04 Beds and instruments</span><span class="d">21 verses: the bed ladder, floors, the admission census, protocol.</span></a>
    <a class="doc" href="canon/05_REPAIRS/"><span class="p">05 Repairs</span><span class="d">15 verses: each known break priced as a plan.</span></a>
    <a class="doc" href="canon/06_PREDICTIONS/"><span class="p">06 Predictions</span><span class="d">36 verses: a prediction and its counter for every deciding cell.</span></a>
    <a class="doc" href="canon/08_ARCHITECTURE/"><span class="p">08 Architecture</span><span class="d">22 verses: the module as an engineer builds it.</span></a>
    <a class="doc" href="canon/09_CHESS_AND_MARKETS/"><span class="p">09 Chess and markets</span><span class="d">40 verses: chess and prediction markets as beds and as products.</span></a>
    <a class="doc" href="canon/CORRECTIONS/"><span class="p">canon/CORRECTIONS.md</span><span class="d">The only door through which the canon changes.</span></a>
  </div>

  <h3>The paper</h3>
  <div class="docs">
    <a class="doc" href="CEQ_SHAPE/"><span class="p">docs/CEQ_SHAPE.md</span><span class="d">The shape paper: the shape, its obstructions, the record, the programme, the apparatus, prior art and limits.</span></a>
    <a class="doc" href="CEQ_SHAPE.pdf"><span class="p">docs/CEQ_SHAPE.pdf</span><span class="d">The same paper, typeset.</span></a>
    <a class="doc" href="sources/"><span class="p">docs/sources/</span><span class="d">The paper's working sources: prior-art sweeps, designs and their refutations, judged sections, planet roadmaps.</span></a>
    <a class="doc" href="BIB_AUDIT/"><span class="p">docs/BIB_AUDIT.md</span><span class="d">The bibliography merge: 508 entries parsed, 408 canonical.</span></a>
  </div>
</section>

<section id="reproduce">
  <h2 class="sec"><span class="num">8</span>Reproduce and cite</h2>
  <p class="lede">Every number on this page comes with the command that reproduces it.</p>
<pre><code>git clone https://github.com/teerthsharma/resolvent &amp;&amp; cd resolvent
pip install -r requirements.txt

python -m pytest tests/arm_smprime tests/arm_pl -q   # 69 passed: the Lean corners, in code
python -m ceqjepa.operator                           # ALL SELF-CHECKS PASSED
python -m ceqjepa.dr1                                # ALL SELF-CHECKS PASSED
python scripts/lean_count.py                         # 13 files: 134 theorems + 41 lemmas = 175
(cd lean &amp;&amp; lake build CEQ)                          # exit 0</code></pre>
  <p>Lean 4 v4.7.0; mathlib is fetched by <code>lake build</code> at the revision pinned in <code>lean/lake-manifest.json</code>, and the first build downloads about 4.2&nbsp;GB. House rule for contributions: every new number ships with the command that produces it, and every new check ships with a planted negative that fails without the change.</p>
<pre><code>@software{sharma2026resolvent,
  author = {Sharma, Teerth},
  title  = {resolvent: one causal attention family spanning softmax attention,
            unnormalized-kernel attention and exact path products},
  year   = {2026},
  url    = {https://github.com/teerthsharma/resolvent}
}</code></pre>
</section>

</div>
