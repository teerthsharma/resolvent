# 3. Why softmax cannot occupy this ground — exactly as the theorems license it

JUPITER (MYCROFT), 2026-09-03, HEAD `207e7b9`. Evidence classes as in `judge/sec_shape.md`.
Every obstruction sentence below carries a theorem number, the model class it is stated for, its
hypotheses as a predicate on the geometry the record ran ($s=64$, $d=16$, $h=1$, float32 so
$p=32$), and the fraction of that geometry it admits. A "cannot" sentence outside the skyline
table of §3.7 is refused. Mechanism throughout: P-10 (a source's intro cited as its theorem),
V-25 (a theorem whose hypothesis no draw satisfies), P-3 / P-11 (a conditional bound stated as
settled), P-8 / V-17 ("depth" in two units).

## 3.1 Obstruction 1 — per-row independence (definitional)

One softmax step is $O_i=f(P_{i\cdot},V)$: the read at $i$ never depends on the read at $j$. The
shape's $z$ is the unique fixed point of the $\gamma$-contraction $z=V+\gamma Pz$ in
$\|\cdot\|_\infty$ (`Contraction.rowStochastic_perron` `:118`, `weighted_contraction` `:72`), so the
reads are mutually constrained. No citation licenses more than this; the nearest published fixed
points are per-query (`ramsauer-2021-hopfield`), and the sharpness failure of the row mixture is
`velickovic-2025-softmaxnotenough`. What makes it measurable is the harmonic residual
$r(\hat z)$ of Proposition 7 with its two unit factors printed — a $10\times$ ratio at equal
marginal error is possible only if one arm's error lies in the slow mode, which is a statement
about what the control mislearns, not "joint consistency" (`refute_falsify_occvac` §3.12). The
claim "joint determination in one read" is UNTESTED (`READ MATHEMATICS.md:106-127`) and stays a
prediction with the counter "$r_{\rm softmax}/r_{\rm shape}\le2$" (SPLIT in $(2,10)$).
Mechanism: V-26, V-17, D-7.

## 3.2 Obstruction 2 — hops need depth (skyline unconditional, lower bounds conditional)

*Skyline (unconditional).* `sanford-2024-logdepth` Thm 4.2: a causally masked transformer computes
$\mathrm{hop}_k$ at depth $\lfloor\log_2k\rfloor+2$, $m=O(1)$, $H=1$ — $3/5/7$ at $t^\star=2/8/32$.
`merrill-2025-littledepth` Thm 2: connectivity at $\lceil\log_2n\rceil$ unrolls of a uniform block.
`fagnou-2024-chacal` Thm 1: $\lceil\log_2(\mathrm{depth}(G)+1)\rceil$ layers for entity tracking —
the same law with different constants, cited together. *Lower bounds.* `sanford-2024-logdepth`
Cor. 4.3, $\Omega(\log k)$, **conditional** on the one-vs-two-cycle conjecture, $k=\Theta(N^\xi)$,
$mH=O(k^{1-\varepsilon})$; `sanford-2024-graph-algorithms` Thm 3/19, same conjecture;
`chen-2024-multilayer` Thm 1.1, **unconditional** for decoder-only softmax at depth $L$ against
$L$-sequential composition, hypothesis $H\,d\,p\le n^{2^{-4L}}$: at $L=1$, $n=64$,
$64^{1/16}=1.2968$ against $H\,d\,p=512$ — vacuous at every geometry the record ran;
`sanford-2024-inductionheads` Thm 1, unconditional at $t^\star=2$, hypothesis $h\,m\,p=\Omega(n)$:
$512\ge64$, not violated, silent here. *Width.* `yehudai-2025-depthwidth`: depth is not necessary
at linear width; the matched control fixes width as well as depth, and at the record's geometry
$d_{\rm model}=16<s=64$ the wide skyline is not instantiated at matched parameters — said, not
left blank. *Circuit class.* `merrill-2023-parallelism` Thm 2: log-precision constant depth
$\subseteq$ uniform $TC^0$; connectivity L-complete; consequences conditional on class separations.
The clause "linear systems P-complete" is dropped: rational linear systems are in
$DET\subseteq NC^2$ (`cook-1985-taxonomy`); P-completeness is linear *inequalities*
(`refute_theory_math` item 13). *What the four license together:* an asymptotic, lineage argument
that one softmax parameter set cannot compose $t^\star\ge2$ content-dependent hops — and nothing at
the record's geometry. *The missing reduction.* $\mathrm{hop}_k$ is pointer chasing in a
token-defined graph; the shape's reachability is in the graph $P$ *the layer computes*; "every
$\mathrm{hop}_k$ instance is a committor instance of some $P$" is plausible and NOT FOUND
(`sweep_expressivity.md` §3.5). Until written, obstruction 2 is a lineage argument, and the
skyline depth $\lfloor\log_2t^\star\rfloor+2$ for BED-S is a choice **by analogy** with Thm 4.2,
which does not cover BED-S's task. Mechanism: P-10, V-25, P-3, P-4.

## 3.3 The DET-class relocation — what "one operator" may claim

The exact committor solves $(I-Q)q=R\mathbb 1$; the exact $z$ is $(I-\gamma P)^{-1}V$. Inversion and
iterated product are DET-complete, $NL\subseteq DET\subseteq NC^2$ (`cook-1985-taxonomy`), and a
triangular inverse is DET-hard (iterated product embeds in $(I-A)^{-1}$ for block sub-diagonal
$A$). So the resolvent as a circuit sits at or above the class the conditional bounds place
reachability in: **the shape relocates the log-depth from the parameter stack into the linear
solve** — $O(s)$ sequential rounds by substitution, or recursive block inversion at
$O(\log^2 s)$ circuit depth / $O(\log s)$ matmul rounds with $O(s^3)$-class work, exact in regime N
and a certified Neumann truncation at $K=2^k-1$ in regime S (`refute_theory_math` item 14,
`refute_instrument_math` row 24; `NOT MEASURED — needs a parallel-prefix kernel timing`).
"Depth-1" in this paper means **one attention parameter set** ($P$, $\gamma$); the circuit-depth
reading is disclaimed in this paragraph. Mechanism: P-8, V-17.

## 3.4 Obstruction 3 — composition

`peng-2024-transformer-limitations` Thm 1 (identifier confirmed; unconditional, communication
complexity): a single $H$-head softmax layer at $p$ bits errs on $f(g(x))$ with probability
$\ge R/(3n\log n)$, $R=n\log n-H(d+1)p>0$. At $n=64$: $64\cdot6=384$ against $1\cdot17\cdot32=544$
— not satisfied, vacuous at $s=64$, $d=16$, float32; it bites at $n\gtrsim100$ or $p=16$.
`kozachinskiy-2025-strassen` Thm 3.4 removes precision (infinite precision, asymptotic $n^{\Omega(1)}$).
"A consequence *is* a composition" ($z'=R(P')V'$ after $P'=S(P,a)$) is a reduction sketch, not
a theorem about the committor. Strassen attention is a NEAR-MISS: two hops per layer by arity,
not all hops by inversion. Mechanism: V-25, P-10.

## 3.5 Obstruction 4 — non-negativity, WITHDRAWN as written

For row-stochastic $P$ the mixing matrix $\Pi_\gamma$ is non-negative and stays non-negative under
absorbing rows (Proposition 5b), so $\partial O_i/\partial V_j\ge0$ survives the resolvent and
`ceq/attention.py`'s min-entry test reads exactly $0$ on the shape — by the identity rows and the
mask, a theorem instance, not a reading (`RUN[F]`: $151$ exact zeros from the identity rows,
lower-triangle min $2.6\times10^{-4}$ without them). Absorption **redirects** mass: a walk that
hits $a\in\mathcal A_k$ stays at $a$, so the weight on every earlier token the walk would have
reached through $a$ goes to $a$ instead; on a dense causal softmax the *support* of $\Pi_\gamma$
does not change (Proposition 5c), only the weights on rows downstream of $a$. The paper writes
"a boundary row captures the mass that would have reached earlier positions", never "veto",
"subtract" or "negative influence"; the signed programme is not revived (`RESEARCH.md`,
`PROGNOSIS.md`); DoFormer's single fixed row is the NEAR-MISS named beside "boundary rows".
Mechanism: P-7, V-23, V-3.

## 3.6 D-1, stated with the right predictor and the right census

`marion-2025-single-location` Cor. 2 (erf predictor, $d\to\infty$, $L=o(d)$) and
`duranthon-2026-softmax-advantage` Prop. 4.2 (softmax proper): one layer is Bayes-optimal on
single-location regression $Y=X_{J_0}^\top v^\star+\xi$. The record ran $L/d=4.00$
(`READ MATHEMATICS.md:1044-1057`), the wrong direction; and the theorems' label model admits
**0 of 3** record beds — BED-M's label is a product $a_{s-1}\cdots a_{s-t^\star}\,b_{s-1-t^\star}$ of
$t^\star+1$ features at $t^\star+1$ positions, not single-location regression for any $t^\star\ge1$
(`refute_theory_occvac` §1.2). D-1 is therefore a *mechanism*: no bed in the record leaves the
scalar-readout class ($\mathrm{out}[:,s-1]$, `READ MISTAKES.md:677-708`), and the SLR theorems are
the nearest owned regime, cited as such. No bed in this paper scores a scalar at one position:
the labels are $z$, $\Delta z$, the reach-avoid tensor, the argmin. Single-effect estimation is
also softmax's ground (`zhang-2023-cina`), which is why the sign column of Proposition 7 is kept
and not claimed. Mechanism: D-1, P-10, V-25.

## 3.7 The skyline table — the only place a "cannot" sentence may stand

Every bed carries every row; a claim is at matched depth and parameters ($4{,}769$, recounted per
arm because a $[m,K+1]$ head or a vector readout changes it — Ruling 3) against the fellow
approximators, and the skylines at unmatched depth/width/decode length are read beside it
(R-SKY, `READ CEQ_V16_CONTRACT.md:209`).

| arm | class | setting to fix before the arena | what it decides |
|---|---|---|---|
| depth-1 softmax, same head | matched | $4{,}769$ params, same readout | the per-row control; D-1 |
| ChaCAL-diag (the shape with $\mathcal A=\emptyset$, diagonal kept) | matched | same $\hat\gamma$ | the boundary-row mechanism; identity half of B-E1 |
| ChaCAL-published (diagonal removed in the inverse, sub-stochastic read) | matched | same $\hat\gamma$ | the published object; a planted negative for bitwise parity at $\gamma>0$ |
| ChaCAL-diag + sink token | matched | same $\hat\gamma$ | the C2 kill: boundary rows vs a column sink |
| InfSA-style Neumann read, no boundaries | matched | $K=16$; cell $\approx5.1$ s | exactness vs truncation |
| cached-mixture $O=\hat P_{\rm base}(I-\gamma\hat P_{\rm base})^{-1}V_{\rm int}$ | matched | $\hat P$ frozen from the un-intervened context | whether the re-solve is needed (`momennejad-2017-sr`) |
| $\lfloor\log_2t^\star\rfloor+2$ softmax stack | skyline | depth $3/5/7$; $\approx7.6$ s per cell [ASSUMED] | Thm 4.2 by analogy |
| wide constant-depth stack | skyline | width $=n$ (`yehudai-2025-depthwidth`); NOT MEASURED | depth is not necessary |
| chain-of-thought decoder | skyline | step count fixed (`merrill-2024-cot`); NOT MEASURED | the autoregressive route |
| MuZero-style value head | skyline (consequence beds) | return, not state | a planner need not predict the state |

`wang-2024-incontext-td` and `xie-2026-softmax-rl` [V] (abstract level; theorem numbers not read,
[U]) show linear-then-softmax transformers implement TD policy evaluation layer by layer, so the
deeper stack computes the *same* resolvent by iteration: the looped block $z^{(t+1)}=V+\gamma Pz^{(t)}$
run $K$ times is the truncated Neumann sum with matrix-residual $\infty$-norm exactly
$\gamma^{K+1}/(1-\gamma)$ and vector error at most $\gamma^{K+1}\|V\|_\infty/(1-\gamma)$
(Proposition 4; `yang-2024-looped`, `gasteiger-2019-appnp` power iteration). A nonlinear loop is a
DEQ (`bai-2019-deq`) and out of scope. "Beats softmax" and "beats native" (`yang-2024-deltanet`,
`dao-2024-ssd` on the linear corner) are banned sentences; the shape's separate claims are
exactness with a printed $\delta\|V\|_\infty$, one-read consistency as a **cost** statement (one
solve, depth $s$), and the boundary-row mechanism, each stated as a property with its
counter-prediction. Mechanism: D-1, R-SKY, D-7, V-17, P-4.

## 3.8 The honest sentence

One *parameterised* operator (one $P$, one $\gamma$), not one parallel round; its hops are exact
and certified (Propositions 2, 4) rather than learned layer by layer; on a bed whose environment
chain is inside the arm's class the contrast against softmax is a reproduction-vs-non-reproduction
contrast and is pre-registered VOID as a capability number (Proposition 8); what is creditable is
the boundary mechanism against ChaCAL-diag-with-sink, exactness against the Neumann read, and the
identification $\|\hat P-P_{\rm env}\|_\infty$. Under the record's calibration (7 of 8 optimistic,
`READ V16_CALIBRATION.md:96-100`) the counter is the point estimate for every row above.

## 3.9 Limits carried to §8

Every obstruction clause is asymptotic, and every unconditional one is vacuous at $s=64$, $d=16$,
float32; the inequalities are printed so a reader sees it. The $\mathrm{hop}_k\to$ committor
reduction is unwritten. The TD-policy-evaluation skyline claim rests on two abstracts. The skyline
depths are by analogy, the skyline prices are assumed or unmeasured, and the two non-depth
skylines have no cell. No sentence here is a result.
