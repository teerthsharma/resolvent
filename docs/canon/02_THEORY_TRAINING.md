# 02 — THEORY OF TRAINING: when AdamW finds the resolvent before a depth-$L$ stack learns $L$ hops

JUPITER-II (MYCROFT, derivations), 2026-09-05, HEAD `99777ab`, under `docs/canon/CHARTER.md`.
Every number carries an evidence class; `READ path:line @ 99777ab` is the pin; struck constants
of `STRUCK.md` appear nowhere; no code file was written and no git write was made.

## Preface — what this book decides, and the verse that decides it first

1. This book decides the arena's measurable half of **T2** (CHARTER §1) as theorem-shaped verses, and
   supplies the theory input to **T1**; it claims neither A1 nor A2, which are book 01's. At the arena
   the training set is fixed and re-presented every step, so the quantity the arena writes is
   **epochs-to-floor**, $T2^{\rm ep}$, and CHARTER §1's draws-to-floor is `NOT MEASURED` there (02.1).
2. It is decided first by **02.7**, in the form that survives: under interventional pairs
   $(\text{context},\mathrm{do}(a),\text{consequence})$ the softmax corner $\gamma=0$ is **never a local
   minimum of the loss along the dial ray**, at every $\hat P$ with a positive transient downstream
   entry — because the corner predicts zero downstream consequence exactly. That is corner algebra. It
   is *not* a statement about the trajectory: under the registered $\gamma=\sigma(\theta)$, $\theta_0=0$
   the run starts at $\gamma=\tfrac12$ and never visits $\gamma=0$, and the training-side question is
   whether the dial leaves $\tfrac12$ — which is what R-21 measures. The training-side theorem is
   computed **at $\theta=0$**: $\partial_\theta L|_0=\tfrac14\partial_\gamma L|_{1/2}$, and under the
   cone condition on $F'$ its sign is $\operatorname{sign}(\tfrac12-\gamma_{\rm env})$ — the dial
   descends toward the bed's registered horizon, which forces the frozen registration
   $|\gamma_{\rm env}-\tfrac12|\ge0.2$ (at $\gamma_{\rm env}=\tfrac12$ the initialisation is the
   optimum and PINNED is evidence about the registration, not about the arm).
3. Under next-state labels (02.4) the corner is stationary in expectation on a one-hop Bayes-fit bed
   with $\mathbb E[\xi\mid V]=0$ (D-1 in landscape form) and is a strict descent direction only inside
   an $\eta^\star$-ball of $P_{\rm env}$, derived in 02.4; under committor labels (02.6) the dial is not
   in the loss at all.
4. The record's own numbers show training did **not** find it: four of five non-zero gains worse than
   $0$ at $t^\star=2$ by more than the indefensibility margin and the fifth unreadable; 3 of 8 R1 seeds
   diverging, to $\hat a_{\max}=285.07$ on the CPU lane and $116.006$ on the CUDA lane; $\beta$ leaving
   the corner on 8 of 8 seeds, every one of them carrying a $\gamma$ ceiling below $1$ (02.2, 02.3,
   02.8).
5. Every pre-written fix is derived, with its own kill: $\beta\equiv1$ as a bind (02.3), the sigmoid
   dial with the Adam step bound (02.4), the annealed dial and the fixed $\gamma_{\rm env}$ (02.4a,
   02.4b), committor supervision (02.6), interventional pairs and the clamp-magnitude reroute (02.7,
   02.7a), the gate modulus with the $(\alpha,T)=(8.69\times10^{-4},150)$ regime (02.8) — the bound
   depends on $\alpha T$ alone, so the step count cannot buy the displacement back and the budget
   shrinks $23.0\times$ with the lr; the replacement is the constrained modulus killed on the
   $\hat a_{\min}$ column (02.8a).
6. Identification (02.5): K-D2 as registered has an undecided reachability at $d_{\rm model}=16<s=64$,
   the copy reading becomes a distance to a representational floor, and the margin constant under the
   matrix $\infty$-norm is $c\ge10.31$, not $9.62$.
7. Sample complexity (02.9): **the parameter-count form of the ratio is withdrawn outright**, at every
   geometry — $p_1=4{,}097$ and $4{,}769$ are two different objects at two different widths;
   $4{,}769$ is a whole arm and decomposes as $p_{\rm layer}=4{,}752$ plus $p_{\rm shared}=17$, so
   $L\cdot4{,}769$ over-counts by $(L-1)\cdot17=68$ and the honest depth-5 figure is $23{,}777$
   DERIVED-by-multiplication; no skyline count is registered anywhere and no depth-5 arm has ever been
   constructed; and a uniform-convergence bound on distinct samples does not bound a step count on a
   fixed training set. What is asserted is an optimisation statement — steps to a target under Adam at
   $\alpha=0.02$, read off 04.15's registered `eval_trace`. The B25 T2 counter is owned by
   `06_PREDICTIONS.md` 06.31 and cited here by verse id; this book files no second counter. The tied
   loop shares every parameter and is the matched contrast; the converse names what the solve cannot
   learn (non-stationary hops, non-linear composition).
8. The AdamW regime (02.10) is a manifest field with two derived corrections (decay off the dial,
   the lr bound), never a claim; the noise floor (02.11) reads $0.0$ on 6 of 6 identical-seed cells,
   so T1/T2 are read against $\mathrm{MDE}_N$, not against $\delta$.
9. The Terminal (02.12) is the sentence of `docs/PLAN.md` §5.10: $\gamma$ is a bed-side dial and the
   shape is softmax wearing a name on this bed — with the committor head, regime N and the
   identities still licensed. Its kill names the only two registered beds that could fire it — **BED-J**
   (`04_BEDS_AND_INSTRUMENTS.md` 04.6) and **BED-C-N**, the chess witness (04.7, protocol
   `09_CHESS_AND_MARKETS.md` 09.11) — each with its seed rule, its $N$ and its realised-sd clause
   printed, and freezes MOVED at $\Lambda>\ln4096=8.318$. It sends the chain to **02.13**, a new verse
   and never back to 02.4: that bed's dial is decided by the solve's ablation gap $A$, not by R-21, and
   02.13 carries no replacement because none exists below the ablation.
10. Census rows closed here: **B12** (02.2–02.8) and the theory half of **B25** (02.1, 02.9–02.11).

---

### 02.1 — $T2^{\rm ep}$, epochs-to-floor, defined at the arena with its floor, its orientation and its instrument

**Statement.** For an arm $A$, seed $\sigma$, and a held-out distance-to-floor $D_t(A,\sigma)$ read at
gradient step $t$ (L-FLOOR: the exact restricted-view floor of the bed),
$$N_\varepsilon(A,\sigma)=\min\{t\in G:\ D_t(A,\sigma)\le\varepsilon\},\qquad
G=\{0,1,2,\dots,150\},\quad|G|=151 .$$
$G$ is the grid the **registered instrument** writes and no other: `04_BEDS_AND_INSTRUMENTS.md` 04.15
declares `eval_trace` as **per-step** eval NRMSE, one entry per gradient step, and calls a missing
declared field a refusal, so the grid the kill reads is every step of the $150$-step budget together
with the step-$0$ read.
**The decimated grid is struck, and the reason is another book's decidability, not taste.** The
record's own harness writes a tenth-step trace — the in-loop hook of `scripts/v15_r1.py` fires at
$t\bmod10=0$ for $t<\text{steps}$, giving $\{0,10,\dots,140\}$, with one unconditional post-loop read
in `model.eval()` at $t=150$ — and the first writing of this verse read $N_\varepsilon$ on exactly that
sixteen-point set. `06_PREDICTIONS.md` 06.29 reconciles the sampling rate **in favour of book 04's
field**: at every step the crossing-step resolution is $1$ step, so $k=66$ and $k=67$ are separable; at
every tenth step the resolution is $10$ steps, which separates neither 06.31's $k=30$ boundary from
$k=39$ nor 06.29's $k=66$ from $k=75$, and a journal carrying one `eval_trace` entry per ten steps is
**refused by the adjudicator**, not read at resolution $10$. This book reads the same field at the same
rate: $|G|=151$, and every price below is the $151$-point price.

**The quantity is epochs, not draws.** The arena presents one fixed training set of
$n_{\rm train}=2048$ at every step, so the count of *distinct* draws is $2048$ at every $t$ and does not
vary with $t$; $2048t$ is a count of presentations. CHARTER §1's T2 — "the number of training draws" —
is therefore **unmeasurable at the arena as registered**. What the arena writes is
$$T2^{\rm ep}(A;\varepsilon)=\text{epochs-to-floor}=N_\varepsilon(A,\sigma),$$
and the draws form is `NOT MEASURED — needs a fresh-draw or minibatch regime registered for BED-S,
with $N_\varepsilon$ re-derived in samples`.

**Orientation, fixed here for the canon: shape over control.** Both ratios are
$$T2^{\rm ep}_{d1}=\frac{\operatorname{median}_\sigma N_\varepsilon(\text{shape},\sigma)}
{\operatorname{median}_\sigma N_\varepsilon(\text{softmax}_{d1},\sigma)},\qquad
T2^{\rm ep}_{\rm sky}=\frac{\operatorname{median}_\sigma N_\varepsilon(\text{shape},\sigma)}
{\operatorname{median}_\sigma N_\varepsilon(\text{sky}_5,\sigma)},$$
the licensed direction being $<1$ (fewer epochs for the shape). This is the orientation of
`03_KERNEL.md` 03.8 ($k^\star(a)/k^\star(b)$, $a$ the shape) and of `06_PREDICTIONS.md` 06.31
($T2_{\rm sky}\le0.20$ predicted); no book may invert it, and the symbol $R_{T2}$ — which named the
reciprocal, so that $R_{T2}=5$ and $T2_{\rm sky}=0.20$ were one cell written two ways (V-17) — is
**retired**: no verse of this canon reads a ratio under that name, here or in 02.9, and the two
sentences that print the symbol are the two that retire it.

**Licensing.** CHARTER §1 grants "faster to train" only on **two** measured numbers, T1 **and** T2,
each read against base self-attention **and** against the depth skyline, at matched parameters
(`CHARTER.md` §1, the "Faster to train" paragraph). A row in which only $T2^{\rm ep}$ holds licenses the
sentence "T2 alone, in the epochs form, at this bed and this $\varepsilon$" and **never** "faster to
train". Both denominators are printed in every row: depth-1 softmax and the depth-$\lfloor\log_2t^\star
\rfloor+2$ skyline at unmatched parameters (R-SKY).

**Matched parameters (Ruling 3).** Every $T2^{\rm ep}$ row prints each arm's own `n_params` and its
residual against the depth-1 control's $4{,}769$. The shape's own count is `NOT MEASURED` (no shape cell
exists), so the word "matched" is **withheld** from every row of this book. The record's nearest
non-softmax arms miss the count by more than twenty times Ruling 3's tolerated residual: `arm_pl`
$4{,}803$, residual $34/4{,}769=0.713\,\%$; `arm_smprime` $4{,}806$, residual $37/4{,}769=0.776\,\%$;
Ruling 3's matched residual is $0.032\,\%$ (`CHARTER.md` §1, the "Base self-attention" paragraph).
$N_\varepsilon=\infty$ on a seed that never reaches $\varepsilon$ within $G$, and an infinite seed enters
the median as $+\infty$.

**Hypotheses.** Arena geometry $s=64$, $d=24$ (the bed's value width), $d_{\rm model}=16$,
$n_{\rm train}=2048$ full batch, $n_{\rm eval}=4096$, $150$ steps, $N=8$, $t^\star=8$, $K=2$, $m=8$.
No non-vacuity clause is asserted here: **the kill decides vacuity**, and a bed on which the shape
reads $N_\varepsilon=\infty$ on the majority of seeds is killed by the ratio itself, not excused by a
hypothesis. $\varepsilon$ must satisfy
$$\varepsilon\ \ge\ \text{floor}+\max\big(\mathrm{MDE}_N,\ 4.7\times10^{-3}\big),$$
**the inequality non-strict, so that the single value both kills of this book read —
$\varepsilon=\text{floor}+\max(\mathrm{MDE}_8,4.7\times10^{-3})$ at $N=8$ — is admissible under the
Hypotheses that license them.** The first writing wrote it strict, which excluded from the Hypotheses
the one $\varepsilon$ the kills name (V-11, the shape of F1 relocated into the boundary of an
inequality); the strict form is **struck**. The second term is the canon's own indefensibility bound — twice the reduction-order floor, above
which alone a margin can be defended — not the thread-count floor $2.345\times10^{-3}$, which is half
of it.

**Evidence.** Full-batch loop, the in-loop hook and the post-loop read: `READ scripts/v15_r1.py:212-227,
:250-266 @ 99777ab` (the hook journals train loss at `trace_every=10`; the post-loop read is
unconditional and in `eval()`). Steps, $n$, $N$: `READ docs/PLAN.md:627 @ 99777ab` (S-62).
$\mathrm{MDE}_8$ rows: $0.039827$ at sd $0.034451$ (placeholder), $0.126238$ at the realised
$0.109199$: `READ docs/CEQ_SHAPE.md:2668-2676 @ 99777ab`. The indefensibility bound $4.7\times10^{-3}$:
`READ docs/CEQ_SHAPE.md:2675 @ 99777ab` (A.11 item 4) and `READ MISTAKES.md:1096 @ 99777ab`. Parameter
counts: `RUN python -c "import json;[print(r['kind'],r['seed'],r['n_params']) for r in map(json.loads,
open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell']"`, 2026-09-06 — $4{,}769$ / $4{,}803$ /
$4{,}806$. The held-out per-step distance is a **registered journal field**, not a missing instrument:
`04_BEDS_AND_INSTRUMENTS.md` **04.15** requires `eval_trace` (per-step eval NRMSE, for T2) on every
cell and calls a missing declared field a refusal; `04_BEDS_AND_INSTRUMENTS.md` **04.10, link 1** reads
$T2$ from it; `06_PREDICTIONS.md` **06.29** fixes its rate at every step and refuses a tenth-step
journal. **These three are cited by verse id and carry no line and no sha**: the canon is untracked at
`99777ab` (`RUN git ls-tree 99777ab docs/canon`, 2026-09-06, returns nothing), so a `@ 99777ab` pin on
a `docs/canon/` path names no object, and a line number without a sha is not a pin at all (CHARTER §3).
The first writing's `04_BEDS_AND_INSTRUMENTS.md:215` and `:152` were both wrong at filing and are both
stale now — the same two objects read `:245` and `:176` at this session's tree — which is the reason
the citation is by verse id and not by line. No cell carrying `eval_trace` exists yet: `NOT MEASURED — needs the first BED-S
cell (04.21)`. The number showing training did not reach a floor: R1, the nearest arm the record
trained, $N=8$ CI $[0.617075,1.041227]$ straddling the one-hop *threshold* $0.7071067812$, 3 of 8 seeds
at NRMSE $\ge1.113403$ — so $N_\varepsilon=\infty$ on 3 seeds at every $\varepsilon<0.4$
(`RUN results/v15_r1.txt` §4; the threshold is not a floor, C15).

**Mechanism.** P-7 (the word had no referent: `CHARTER.md` §5 row B25), V-17 (NRMSE is
anchored: the mean predictor reads $1.0$ on every bed; and a named ratio may not be inverted between
books), V-11 (a kill inside its own hypotheses), M-3, M-9, C17, Ruling 3.

**Kill.** Frozen, **one number and no second conjunct**: $T2^{\rm ep}_{d1}\ge1.00$ at $N=8$ at
$\varepsilon=\text{floor}+\max(\mathrm{MDE}_8,4.7\times10^{-3})$, the median-over-seeds ratio of the
Statement, with $+\infty$ entering the median as $+\infty$.

**The interval conjunct is struck, and the arithmetic that strikes it is printed.** The first repair
added "with the paired per-seed ratio's $95\,\%$ interval **not** excluding $1$", which prevents the
kill firing in exactly the two cases where the shape is most decisively beaten. With $\ge5$ of $8$
seeds at $N_\varepsilon(\text{shape})=+\infty$ the paired per-seed ratios are $+\infty$ on those seeds,
so the interval lies wholly above $1$ and **excludes** $1$: the conjunct fails and the kill does not
fire on the very reading the same paragraph says fires it on its own. The same holds for any decisive
defeat — per-seed ratios $\{3.0,\dots,8.0\}$ give an interval $[3,8]$ that excludes $1$ while
$T2^{\rm ep}_{d1}\ge1.00$ holds — so the conjunct left a kill that fires only in the band where the
shape is worse but not significantly so, which is F1's V-11 moved from the Hypotheses into the Kill.
The conjunct is deleted outright rather than made one-sided: the median-with-$+\infty$ rule already
makes the number decidable at $N=8$, and a second statistic on the same column buys no decidability
and costs reachability. **Rejection region, printed so no reading is left undecided:** the kill fires
on $T2^{\rm ep}_{d1}\in[1.00,+\infty]$, the endpoint $+\infty$ included and reached whenever $\ge5$ of
$8$ shape seeds are infinite; it does not fire on $T2^{\rm ep}_{d1}\in[0,1.00)$; there is no third
region. Every ingredient lies inside the Hypotheses, whose $\varepsilon$ inequality is non-strict for
exactly this reason — vacuity and defeat are decided by the same number, and neither is excused by a
precondition.

Instrument: 04.15's registered per-step `eval_trace` field read at the $151$ points of $G$. Price
DERIVED at the held-out width $n_{\rm eval}=4096$, not at $n_{\rm train}=2048$, and at the registered
per-step rate, not at the struck tenth-step rate: $151$ held-out forwards $\times\,8$ arms
$\times\,8$ seeds $=9{,}664$ forwards; the measured softmax forward is $1.473$ ms and the shape's
$2.514$ ms at $n=2048$, and the measured solve increment at $n=4096$ is $2.312$ ms
(`READ docs/CEQ_SHAPE.md:583-600 @ 99777ab`), so one held-out forward at $4096$ costs $2.946$ ms
(softmax base, linear in $n$) to $2.946+2.312=5.258$ ms (shape), and $9{,}664$ forwards cost
$\mathbf{28.5}$ to $\mathbf{50.8}$ GPU-s ($28.470$ / $50.813$ GPU-s exactly). The first writing's
$1{,}024$ forwards and $3.017$–$5.384$ GPU-s were the sixteen-point price and are **struck**: they
under-priced the kill by $9.4\times$ while reading a field book 04 registers per step. Planted controls: at $\varepsilon$ equal to the softmax control's own final
distance the instrument must read $N_\varepsilon\le150$ on 8 of 8; at $\varepsilon=0$ it must read
$\infty$ on 8 of 8 (V-16: a floor that is reached and one that is not); the depth-1 control against
itself must read $T2^{\rm ep}_{d1}=1.00$ exactly.

**If killed.** $T2^{\rm ep}$ is re-based on the $9{,}600$-step ladder (M-5.3, spent only on a survivor,
`READ docs/PLAN.md:775 @ 99777ab`), $N_\varepsilon$ read at checkpoints every $600$ steps.
*Hypotheses:* the survivor rule of §5.10 has admitted the shape. *Evidence:* price $\approx36$ min per
pair `FITTED×64` (`READ docs/CEQ_SHAPE.md:2697 @ 99777ab`). *Kill:* $T2^{\rm ep}_{d1}\ge1.00$ at
$9{,}600$ steps — a journal read of the ladder's checkpoints, $0$ GPU-s, cheaper than the arena's trace
— and $T2^{\rm ep}$ is VOID for the shape on BED-S.

**Terminal.** $T2^{\rm ep}$ is a definition with a registered instrument; no ratio is licensed until one
cell prints $N_\varepsilon<\infty$ on both arms with both `n_params` in the row. The draws form of
CHARTER §1's T2 is unmeasurable at the arena and stays `NOT MEASURED`. The canon licenses only the
per-step MAC statement (book 03, DERIVED) and withdraws every "faster to train" sentence on the bed.

---

### 02.2 — The corner-descent problem, stated with the record's numbers (B12)

**Statement.** On every bed the record trained, gradient descent from the softmax corner either did
not leave it or left it without paying: (i) **a pilot, $N=3$ on `threads=6`, licensing no seed-rule
verdict** — the gain sweep $\gamma\in\{0,0.05,0.10,0.25,0.50,1.0\}$ at $t^\star=2$ reads
$+0.000000/+0.008261/+0.006960/+0.002384/+0.031754/+0.048981$ NRMSE against softmax, so **four of the
five non-zero gains are worse by more than the indefensibility margin $4.7\times10^{-3}$, and
$\gamma=0.25$ is unreadable at $+0.002384$ against that cell's own sd $0.006788$ at $N=3$**
($0.002384/0.006788=0.351$ of one sd, and below $4.7\times10^{-3}$): the sign at $\gamma=0.25$ is not
readable and no verdict is taken there. The pilot's lane is `threads=6` and the arena's is `threads=8`
(M-10: the thread count is a lane property), and A.11 item 1 sets $N=8$ as the minimum deduplicated by
seed, so (i) is evidence of a direction and never of a seed-rule verdict; (ii) R1 at $t^\star=2$,
$N=8$: 5 of 8 seeds cross the
one-hop threshold and 3 of 8 diverge with $\hat a_{\max}=20.31/49.66/285.07$ **on the CPU lane**
($12.77/49.66/116.01$ on the CUDA lane the kill runs in — the two columns and the lane rule are printed
in 02.8), the paired contrast
$-0.122616$ with sd $0.257560$ excluding nothing beyond $0.215326$; (iii) $\beta$ leaves the corner on
8 of 8 `arm_smprime` seeds ($0.5876$–$1.5093$) into rows whose $\beta=0$ reading over the eight quoted
rows spans $[0.724290,10.293107]$, with no certificate;
(iv) at $t^\star=8$ the arm trains to $0.860972$ on its two-hop ceiling $0.866025$ and evaluates at
$1.112208$. Bet C's counter — $\hat\gamma$ PINNED ($\Lambda\le2.7055$) on at least 6 of 8 seeds — is
the point estimate under D-CALIB. The problem has two faces: a dial whose gradient at the corner
vanishes in expectation on one-hop labels (02.4), and a switch whose motion off the corner leaves the
certified class (02.3).

**Hypotheses.** BED-M, $s=64$, $d=24$, $d_{\rm model}=16$, $n=2048$, $150$ steps, Adam at lr $0.02$
(not AdamW), full batch; arms `pivot_unsigned` (bitwise softmax's operator with one hop-2 term),
`arm_pl`, `arm_smprime`; every label a scalar at $s-1$ (D-1), so (i), (ii), (iv) are readings inside
the regime where one softmax layer is asymptotically Bayes-optimal and the corner *should* win.

**Evidence.** (i) `RUN results/r10_v13_hop2gain_t2.txt`, whose header reads verbatim
`t*=2 n=2048 steps=150 seeds=[0, 1, 2] K=64 threads=6` — $N=3$, `threads=6`, printed here because the
row is a pilot; the per-cell sds are $0.015381/0.015795/0.009422/0.006788/0.025191/0.027812$; the
$N=8$ minimum and the lane rule: `READ docs/CEQ_SHAPE.md:2669 @ 99777ab` (A.11 item 1),
`READ MISTAKES.md:1072-1108 @ 99777ab` (M-10), `READ docs/PLAN.md:627 @ 99777ab` (the arena lane,
`threads = 8`); the indefensibility margin `READ docs/CEQ_SHAPE.md:2675 @ 99777ab`;
(ii) `RUN results/v15_r1.txt` §3–§4,
`READ V15_R1.md:177-188 @ 99777ab`; (iii) `READ V20_R15_IT1_MARS.md:355-358 @ 99777ab` reading
`results/v17k_r4_retake.jsonl` `manifest.smp_values`, row sums `READ V16_ARM_SMPRIME.md:28-32 @
99777ab`; (iv) `READ MATHEMATICS.md:302-306 @ 99777ab`; Bet C and its counter `READ
docs/PLAN.md:905-928 @ 99777ab`; D-CALIB 7 of 8 optimistic, $p=0.0352$: `READ V16_CALIBRATION.md:96-100
@ 99777ab`; the optimiser: `READ scripts/v15_r1.py:235 @ 99777ab`, `scale/m3_capability.py:88,178`.

**Mechanism.** D-1 (`READ MISTAKES.md:677-708`), D-7, M-20, V-9, V-22.

**Kill.** This verse is a census; it is refuted by one record cell with `kind` $\ne$ `softmax`, a
non-zero dial or gain, and eval NRMSE below the softmax control **by more than $\mathrm{MDE}_8$ at the
realised paired sd** on at least 6 of 8 seeds — the threshold CHARTER §1 licenses
(`CHARTER.md` §1, the "More accurate" paragraph), not "one seed sd", which at the record's row-2 sd $0.034451$
reads $0.034451$ against $\mathrm{MDE}_8=0.039827$ and is $0.86\times$ as strict as the licensed
threshold. Until S-62 fills the realised paired sd, the frozen number is the **placeholder**
$\mathrm{MDE}_8=0.039827$ at sd $0.034451$, named a placeholder here and replaced by the realised
figure on the first BED-S cell (04.21); at the realised $0.109199$ it would read $0.126238$.
Instrument: `grep` over `results/**/*.jsonl` for a `gain`/`gamma` field beside
`eval_nrmse` — the fields occur in `results/arm_a.jsonl` (3), `arm_a_k1.jsonl` (48),
`cameron_aggregators.jsonl` (8), `wilson_arms.jsonl` (36) (`RUN grep -c '"gamma"'`, 2026-09-05), all
on the dead signed programme, none a resolvent dial. Price $0$ GPU-s. Planted positive: the grep must
return `results/r10_v13_hop2gain_t2.txt` for `gain=` (it does, with `results/v15_r1.txt`).

**If killed.** The census row is rewritten naming that cell, and B12 closes on it. *Hypotheses:* the
cell's seed rule and thread lane are journalled. *Evidence:* the cell's row. *Kill:* the cell's
$\Lambda\le2.7055$ under the boundary null (02.4), $0$ GPU-s.

**Terminal.** B12 stands as filed: no cell in the record shows a paid departure from the corner, and
every fix below is a plan with its own kill, not a result.

---

### 02.3 — $\beta$: the switch must not descend; $\beta\equiv1$ is a bind in the shape lane

**Statement.** In Definition 1's base $W_\beta=\mathrm{num}/Z^\beta$, the row sum is $Z_i^{1-\beta}$, so
$\|W_\beta\|_\infty=\max_iZ_i^{1-\beta}$ is unbounded above $1$ for $\beta<1$. **The record's quoted
rows are a $\beta=0$ reading, and there are eight of the sixty-four**: at $\beta=0$ they read
$[1.312192,0.724290,2.563817,2.264559,10.293107,2.721943,3.096841,1.337183]$, so the range over the
quoted rows is $[0.724290,10.293107]$ — the minimum is **below one**, and every number below is a lower
bound on the true $\max_i$ over $64$ rows.

No convergence claim is made. Convergence of $\sum_k(\gamma W)^k$ needs $\gamma\rho(W)<1$;
$\gamma\|W\|_\infty<1$ is **sufficient only**, with $\rho(W)\le\|W\|_\infty$ and the inequality strict
generically, and $\rho(W)$ is `NOT MEASURED`. What fails is the *bound*, not the series: Proposition 4
states that for $\|P\|_\infty>1$ its bound **can** fail. So the statement in force is: at
$\|W\|_\infty=10.293107$ Proposition 4's equality clause no longer applies for every
$\gamma\ge1/10.293107=0.0971537$ (DERIVED), Proposition 5(b)'s row-stochastic mixture property is lost,
and the certificate field is `refused` (S-42).

**The trained-$\beta$ reading, per seed (Ruling 2a: no sentence transfers across corners without a
bind at that corner).** Since $Z_i>0$, $\max_iZ_i^{1-\beta}$ is $(\max_iZ_i)^{1-\beta}$ for $\beta<1$
and $(\min_iZ_i)^{1-\beta}$ for $\beta>1$. Over the eight quoted rows, at the eight measured $\hat\beta$:

| seed | $\hat\beta$ | $\|W_{\hat\beta}\|_\infty$ (8 rows) | $\gamma$ ceiling $1/\|W\|_\infty$ |
|---|---|---|---|
| 0 | $0.732560$ | $1.865499$ | $0.536050$ |
| 1 | $0.896975$ | $1.271503$ | $0.786471$ |
| 2 | $1.343933$ | $1.117328$ | $0.894992$ |
| 3 | $1.509306$ | $1.178548$ | $0.848501$ |
| 4 | $0.587580$ | $2.615740$ | $0.382301$ |
| 5 | $0.782138$ | $1.661865$ | $0.601734$ |
| 6 | $0.834885$ | $1.469556$ | $0.680478$ |
| 7 | $0.901199$ | $1.259044$ | $0.794254$ |

Two seeds train to $\hat\beta>1$, where $Z_i^{1-\beta}<1$ on the rows with $Z_i>1$ — but the quoted
minimum row is $Z=0.724290<1$, so those two seeds' maximum is taken there and still exceeds $1$.
**All eight seeds carry a $\gamma$ ceiling strictly inside $[0,1)$**, from $0.382301$ to $0.894992$;
$1/10.293107$ is kept only as the $\beta=0$ corner-3 number, with that label.

Therefore the shape lane fixes $\beta\equiv1$ **by construction** — a bind on the row-stochastic class,
carried by `three_corners_containment` — and the learnable $\beta$ of Ruling 2 belongs to the
`arm_smprime`/HF lane, where no resolvent is applied. The corner-descent problem for $\beta$ is
thereby removed, not solved: the shape has no $\beta$ to descend. **Gradient descent moved $\beta$ off
$1$ on 8 of 8 seeds in 150 steps** — a movement statement, with no speed word in it: no form (A1, A2,
T1, T2), bed, floor, $N$ or matched count is claimed for it, and none is licensed.

**Hypotheses.** Regime S ($\beta=1$, $P_{ii}>0$, $\gamma\in[0,1)$); the certificate or the mixture
property is claimed on the cell; the lane separation is journalled in the manifest (S-01). The
inequality is non-vacuous on the record's own $\beta=0$ rows ($\max$ over the eight quoted
$=10.293107$) and on all eight trained $\hat\beta$ (ceiling table above). The eight rows are quoted of
sixty-four; every $\|W\|_\infty$ here is a lower bound.

**Evidence.** Row sums, all eight, at $\beta=0$, `READ V16_ARM_SMPRIME.md:29 @ 99777ab` row (e);
Propositions 4, 5 `READ docs/CEQ_SHAPE.md:391-436 @ 99777ab` (Proposition 4's "for $\|P\|_\infty>1$ the
bound *can* fail", `:394-395`); trained $\hat\beta$ per seed
`RUN python -c "import json;[print(r['seed'],r['manifest']['smp_values']['beta']) for r in
map(json.loads,open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell' and r['kind']=='arm_smprime']"`,
2026-09-06 — $0.7325605,0.8969751,1.3439332,1.5093061,0.5875798,0.7821376,0.8348854,0.9011988$; the
ceiling table DERIVED from those two readings by $\max_iZ_i^{1-\beta}$; Ruling 2
`READ V17K_RULINGS.md:47-54 @ 99777ab`; S-42 `READ docs/PLAN.md:705-710 @ 99777ab`; the switches as
`nn.Parameter` at init $1$: `READ ceq/arm_smprime.py:527 @ 99777ab`; $\rho(W)$: `NOT MEASURED — needs a
spectral-radius read on the trained $W$ (no journal field carries it)`.

**Mechanism.** V-25 (a theorem whose hypothesis the trained rows violate), V-3, V-22 (Ruling 2's
constant carried across lanes), Ruling 2/2a, P-10 (a sufficient condition read as necessary), P-2,
V-10 (a kill no cell can satisfy), M-2 (a threshold with no number).

**$\delta_\beta$: the imported constant is struck, and what replaces it is measured in $\hat\beta$.**
The first repair froze $\delta_\beta:=2.345\times10^{-3}$ "in $\hat\beta$ units", citing
`READ docs/CEQ_SHAPE.md:2675 @ 99777ab` (A.11 item 4). That source states the opposite of the unit
claimed: A.11 item 2 reads, verbatim, "The MDE is in NRMSE units; the cosine and the `φ`-NRMSE carry
their own realised sd" (`READ docs/CEQ_SHAPE.md:2673 @ 99777ab`), and item 4's $2.345\times10^{-3}$ is
a **thread-count margin on that same NRMSE scale**. A margin cannot be carried from NRMSE into a
parameter's units by assertion (V-17), and $2.345\times10^{-3}$ therefore appears **nowhere in this
book as a $\hat\beta$ threshold**; it survives only where it was measured, on `eval_nrmse` (02.11).

**The $\hat\beta$ identical-seed pair, RUN.** The reading the repair required exists and costs
$0$ GPU-s:

`RUN python -c "import json;f={(r['kind'],r['seed']):((r.get('manifest') or {}).get('smp_values') or
{}).get('beta') for r in map(json.loads,open('results/v17k_r4_floor.jsonl')) if r.get('t')=='cell'};
g={(r['kind'],r['seed']):((r.get('manifest') or {}).get('smp_values') or {}).get('beta') for r in
map(json.loads,open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell'};[print(k,f[k],g[k],
abs(f[k]-g[k])) for k in sorted(f) if k in g and f[k] is not None and g[k] is not None]"`, 2026-09-06,

which prints `('arm_smprime', 0) 0.7325604557991028 0.7325604557991028 0.0` and
`('arm_smprime', 1) 0.8969751000404358 0.8969751000404358 0.0`. Two corrections to the finding's own
count follow from the same read and are printed rather than absorbed: the identical-seed block holds
**six** cells but only **two** carry a $\beta$ at all — `arm_pl` and `softmax` journal
`manifest.smp_values` as `null`, so the $\hat\beta$ identical-seed evidence is $2$ of $2$, not $6$ of
$6$ — and the measured value is
$$\delta_\beta^{\rm rep}\ :=\ \max_\sigma\big|\hat\beta_\sigma-\hat\beta'_\sigma\big|\ =\ \mathbf{0.0}
\quad(\text{RUN, }2\text{ of }2\text{ cells, }0\text{ GPU-s}),$$
the lane's $\hat\beta$ reproducibility floor: on this lane the trained $\hat\beta$ is **bitwise** the
same across two journals of the same seed.

**What a zero floor may and may not decide, stated once for the whole book.**
$\delta_\beta^{\rm rep}=0.0$ is a legitimate threshold for an **identical-seed** comparison, where the
question is whether an intervention changed a bitwise-reproducible number: those are 02.10's two kills,
which read it as "the pair is bitwise equal" and whose planted control predicts a difference of
$0.029557\,|\hat\beta|$, four orders above the floor. It is **not** a legitimate half-width for an
interval around $1$: at $\delta_\beta=0$ the interval $[1-\delta_\beta,1+\delta_\beta]$ collapses to
$\{1\}$, and a learnable $\beta$ under Adam leaves $\{1\}$ at step $1$ whenever its gradient is
non-zero, so an interval kill at that width fires by construction (V-10). **02.3's kills therefore
carry no $\delta_\beta$ at all**: they are re-based below on $\|W_{\hat\beta}\|_\infty$, whose
threshold is the exact $1$ that Propositions 4 and 5(b) themselves turn on, and on the interior null's
own critical value $3.841$. The minimum detectable **departure of $\hat\beta$ from $1$** — a different
object from the reproducibility floor — is
`NOT MEASURED — needs a $\hat\beta$ null block (two arms differing only in seed, on the shape lane),
which no verse of book 04 registers`, and no kill of this book conditions on it.

**Kill.** The verse's two conjuncts were mutually exclusive — an arm pinned inside
$[1-\delta_\beta,1+\delta_\beta]$ on every seed *is* the $\beta\equiv1$ arm to $\delta_\beta$, and a
paired NRMSE gap above $\mathrm{MDE}_8$ is not attainable from a parameter difference of at most
$\delta_\beta$ — so the kill is split in two, each decidable alone.

**Kill (a) — the bind is load-bearing, read on the norm and not on an interval.** Frozen: a shape-lane
cell with $\beta$ learnable whose $\|W_{\hat\beta}\|_\infty>1$ on at least 6 of 8 seeds **while** the
certificate field reads `refused` on those seeds. Then the bind is doing work and the verse stands;
its failure — $\|W_{\hat\beta}\|_\infty\le1$ on $\ge6/8$ — says the class never left row-stochasticity
and the bind is decoration (V-9). **The threshold is the exact $1$ of Proposition 4's own hypothesis
and carries no resolution constant**: the first writing read this kill on
$[1-\delta_\beta,1+\delta_\beta]$ with $\delta_\beta$ imported from an NRMSE margin, which is V-17 in
the threshold and, at the measured $\delta_\beta^{\rm rep}=0.0$, V-10 in the interval; both are struck
and the reading moves to the quantity the bind is actually about. $\|W_{\hat\beta}\|_\infty$ is
$\max_iZ_i^{1-\hat\beta}$, DERIVED at $0$ GPU-s from the journalled $\hat\beta$ and the row sums the
cell already writes, so no instrument is added with the change. Instrument: S-62 with one added arm,
$8\times1.680\approx13.4$ s. Planted negative: the `arm_smprime` lane must read
$\|W_{\hat\beta}\|_\infty>1$ on 8 of 8 — it does, $[1.117328,2.615740]$ in the ceiling table above, on
every one of the eight seeds — so the instrument condemns where the record already shows departure.
Planted positive that must **not** fire: the $\beta\equiv1$ arm, whose $\|W_1\|_\infty=\max_iZ_i^0=1.0$
exactly on 8 of 8 by construction.

**Kill (b) — the bind cost accuracy.** Frozen: a $\beta$-learnable cell whose $z$-channel NRMSE is
below the $\beta\equiv1$ cell's by more than $\mathrm{MDE}_8$ at the realised paired sd on at least
6 of 8 seeds, **whatever $\hat\beta$ reads** — then the bind is lifted. Instrument and price the same
$\approx13.4$ s (one journal, both kills). Planted negative: the $\beta\equiv1$ cell against itself must
read a gap of exactly $0.0$ on 8 of 8, below $\mathrm{MDE}_8$, so the kill does not fire on an identity.

**If killed (a or b).** $\beta=\sigma(\theta)\in(0,1)$ with the read projected,
$W\leftarrow W/\max(1,\|W\|_\infty)$ per row. **The projection recovers only the certificate half.** It
yields $\|W\|_\infty\le1$, i.e. sub-stochastic rows, which is exactly Proposition 4's second clause
("for $\|P\|_\infty\le1$ the bound holds as $\le$") — so the certificate survives. Proposition 5(b)'s
mixture property needs row sums equal to $1$, which $\le1$ does not give, so it is **not** recovered
and the mixture read stays lost. *Hypotheses:* the projection is in the forward and journalled.
*Evidence:* Proposition 4 clause 2 `READ docs/CEQ_SHAPE.md:391-398 @ 99777ab`; Proposition 5(b)
`:391-436`. *Kill:* the projected arm's $\hat\beta$ fails to reject the interior null at $\beta=1$ —
$\Lambda_\beta\le3.841$ on at least 6 of 8, the $\chi^2_1$ critical value at $95\,\%$, **which is the
whole threshold: the interval $[1-\delta_\beta,1+\delta_\beta]$ the first writing wrote beside it is
struck, its half-width having been an NRMSE margin (V-17) and, at the measured
$\delta_\beta^{\rm rep}=0.0$, a degenerate point (V-10)** — the projection is decoration (V-9),
$0$ GPU-s from the S-62 journal.

**Second replacement, for the mixture half — row renormalisation to exactly $1$.** If the mixture
property is claimed on any cell, the projection is replaced by
$W\leftarrow\operatorname{diag}(W\mathbb 1)^{-1}W$, which sets every row sum to exactly $1$ and restores
$\beta\equiv1$ in effect (the map is the identity at $\beta=1$ and is exactly the $Z^{-1}$
renormalisation elsewhere), so Proposition 5(b) applies and both halves are back. *Hypotheses:*
$W\mathbb 1>0$ rowwise, which holds because every $\mathrm{num}_{ij}>0$ at finite logits.
*Evidence:* DERIVED, one line. *Kill:* the renormalised arm's $z$-channel NRMSE within
$\mathrm{MDE}_8$ of the $\beta\equiv1$ arm's on at least 6 of 8 — then $\beta$ was never a free
parameter and the learnable switch is deleted from the shape lane; $0$ GPU-s from the same journal,
more decisive than the projection's kill because it decides the parameter, not the read.

**Terminal.** The shape is a $\beta=1$ object; every $\beta<1$ reading belongs to the corner-3 lane,
carries no certificate, and licenses no sentence about the resolvent.

---

### 02.4 — $\gamma$: the dial does not leave its initialisation — the landscape at $\theta_0=0$, and the corner as algebra

**Statement.** Let $M=(I-\gamma P)^{-1}$, $O(\gamma)=(1-\gamma)PMV$, $L=\tfrac12\|O-y\|^2$. Since
$\partial_\gamma M=MPM$,
$$\partial_\gamma O=-PMV+(1-\gamma)PMPMV,\qquad \partial_\gamma O\big|_{\gamma=0}=P(P-I)V,\qquad
\partial_\gamma L\big|_{\gamma=0}=\langle PV-y,\;P(P-I)V\rangle .$$

**What the corner is, and what it is not.** The registered parametrisation is $\gamma=\sigma(\theta)$
with $\theta_0=0$, so $\gamma_0=\tfrac12$ and **$\gamma=0$ is attained only at $\theta=-\infty$**.
The gradient in the parameter that actually moves is
$$\partial_\theta L=\sigma'(\theta)\,\partial_\gamma L=\gamma(1-\gamma)\,\partial_\gamma L
\ \xrightarrow[\gamma\to0]{}\ 0 ,$$
so the corner is **absorbing in $\theta$ whatever $\partial_\gamma L|_0$ reads**: no sign at
$\gamma=0$ is a statement about a trajectory. Clauses (a), (b), (c) below are therefore filed as
**corner algebra on the $\gamma\to0^+$ ray**, never as training-side theorems; the training-side
problem is restated as **"the dial does not leave its initialisation"**, which is exactly what R-21's
PINNED verdict measures at $\gamma_0=\tfrac12$.

(a) *(corner algebra)* On a bed whose label is one-hop Bayes-fit, $y=PV+\xi$ with
$\mathbb E[\xi\mid V]=0$: $\mathbb E\,\partial_\gamma L|_0=\mathbb E\langle-\xi,P(P-I)V\rangle=0$ — the
corner is a stationary point of the dial in expectation (D-1 in landscape form). **Independence of
$\xi$ and $V$ is not enough**: it leaves $\mathbb E[\xi]\ne0$ admissible and the inner product then
does not vanish. The clause is asserted only on beds whose label is one-hop Bayes-fit with conditional
mean zero. It is **not** asserted on the record's beds: their label is a product of three features
$a_{s-1}\!\cdot\!a_{s-2}\!\cdot\!b_{s-3}$ (`READ MATHEMATICS.md:302-306 @ 99777ab`), for which
$y=\hat PV+\xi$ with $\xi\perp V$ does not hold at all, and the residual's dependence on $V$ there is
`NOT MEASURED — needs the residual-on-$V$ regression at the arena`.

(b) *(corner algebra, with a derived radius)* On BED-S's $z$-channel with label
$y=O_{\rm env}(\gamma_e)=P_{\rm env}V+\gamma_eG+O(\gamma_e^2)$, $G:=P_{\rm env}(P_{\rm env}-I)V$: at
$\hat P=P_{\rm env}$, $\partial_\gamma L|_{0^+}=-\gamma_e\|G\|_F^2+O(\gamma_e^2)<0$. That exact point is
**empty at the arena's geometry** — 02.5(ii) proves $F_{\rm rep}=\min_{\hat P\in\mathcal C}
\|\hat P-P_{\rm env}\|_\infty>0$ at $d_{\rm model}=16<s=64$ — so the clause is replaced by a
perturbation statement with an explicit radius. Write $\hat P=P_{\rm env}+\Delta$,
$\epsilon:=\|\Delta\|_\infty\le1$, $\nu:=\max_i\|V_i\|_2$. Then
$$\partial_\gamma L\big|_{0^+}=-\gamma_e\|G\|_F^2+\langle\Delta V,G\rangle+\langle\Delta V,R\rangle
-\gamma_e\langle G,R\rangle,\qquad R:=\big[\Delta(P_{\rm env}-I)+P_{\rm env}\Delta+\Delta^2\big]V,$$
and with $\|\Delta V\|_F\le\sqrt s\,\epsilon\nu$, $\|P_{\rm env}\|_\infty=1$,
$\|P_{\rm env}-I\|_\infty\le2$, $\|R\|_F\le\epsilon(3+\epsilon)\sqrt s\,\nu\le4\epsilon\sqrt s\,\nu$,
$$\partial_\gamma L\big|_{0^+}\ \le\ -\gamma_e\|G\|_F^2+\epsilon\sqrt s\,\nu\|G\|_F(1+4\gamma_e)
+4\epsilon^2s\nu^2 .$$
Requiring each positive term below $\tfrac12\gamma_e\|G\|_F^2$ gives the radius (DERIVED)
$$\eta^\star=\min\left\{\frac{\gamma_e\|G\|_F}{2\sqrt s\,\nu\,(1+4\gamma_e)},\ \
\frac{\|G\|_F}{2\nu}\sqrt{\frac{\gamma_e}{2s}}\right\},$$
and **$\partial_\gamma L|_{0^+}<0$ for every $\hat P$ with $\|\hat P-P_{\rm env}\|_\infty\le\eta^\star$**.
The clause is non-vacuous iff $F_{\rm rep}\le\eta^\star$. Both $\|G\|_F$ and $\nu$ are bed constants
computable from the generator at $0$ GPU-s; their values and $F_{\rm rep}$'s at the arena are
`NOT MEASURED — needs the first BED-S cell (04.21), whose $\|\hat P-P_{\rm env}\|_\infty$ per seed
upper-bounds $F_{\rm rep}$`. Until that row exists the sign in (b) is asserted **only** on the
$\eta^\star$-ball and nowhere else.

(c) At an arbitrary $\hat P$ the sign of $\langle\hat PV-y,\hat P(\hat P-I)V\rangle$ is undetermined,
and the one-hop gradient on $\hat P$ at $\gamma=0$ fits $y$ with a one-hop surrogate; the surrogate
$\Pi_{\gamma_e}(P_{\rm env})=(1-\gamma_e)P_{\rm env}M_{\rm env}$ is row-stochastic and causal
(Proposition 5b), so it is inside the softmax class whenever its logits are representable from the
tokens — which they are not exactly at $d_{\rm model}<s$ from adjacency tokens (02.5). The dial's
signal is therefore **conditional on $\hat P$**: a corner-stationary line, not a minimum, inside the
$\eta^\star$-ball.

**At the trajectory's actual start.** $\partial_\theta L|_{\theta=0}=\tfrac14\,
\partial_\gamma L|_{\gamma=1/2}$, non-zero iff $\partial_\gamma L|_{1/2}\ne0$ — a codimension-one
condition on $(\hat P,P_{\rm env},V,\gamma_e)$, so the dial is **generically non-stationary at
$\gamma_0=\tfrac12$ with a sign that no clause of this verse determines**. The training-side question
is therefore not "does descent leave the corner" but "does $\hat\gamma$ leave $\tfrac12$", and the
number that answers it is R-21's $\Lambda$. A clamp $\gamma=\min(\max(\theta,0),1-\epsilon)$ has zero
gradient at the boundary and makes the corner absorbing in the clamp too (V-10 by construction).

**The Adam step bound, with its bias corrections.** At $(\beta_1,\beta_2)=(0.9,0.999)$ the
bias-corrected step is $\Delta\theta_t=-\alpha\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ with
$\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$. DERIVED, keeping both factors:
$|m_t|\le(1-\beta_1)\sum_{k\le t}\beta_1^{t-k}|g_k|$ and
$v_t\ge(1-\beta_2)\beta_2^{t-k}g_k^2$ for each $k$; Cauchy–Schwarz on
$\sum_k\beta_1^{t-k}|g_k|$ against $\big(\sum_k\beta_1^{2(t-k)}/\beta_2^{t-k}\big)^{1/2}
\big(\sum_kg_k^2\beta_2^{t-k}\big)^{1/2}$ and $\beta_1^2<\beta_2$ give
$$|\Delta\theta_t|\ \le\ \alpha\,\frac{1-\beta_1}{1-\beta_1^t}\cdot
\frac{\sqrt{1-\beta_2^t}}{\sqrt{1-\beta_2}}\cdot\frac{1}{\sqrt{1-\beta_1^2/\beta_2}}
\ \le\ \frac{\alpha(1-\beta_1)}{\sqrt{1-\beta_2}}\cdot\frac{1}{\sqrt{1-\beta_1^2/\beta_2}}
\ =\ 3.16228\,\alpha\cdot2.29906\ =\ \mathbf{7.2704\,\alpha} ,$$
the last inequality because $(1-\beta_1)/(1-\beta_1^t)\le1$ and $\sqrt{1-\beta_2^t}\le1$ for $t\ge1$.

**The Cauchy–Schwarz factor, evaluated, and the two constants the first repair got wrong.** At
$(\beta_1,\beta_2)=(0.9,0.999)$: $\beta_1^2/\beta_2=0.81/0.999=0.8108108$, $1-{}=0.1891892$,
$\sqrt{\ }=0.4349589$, so
$$\frac{1}{\sqrt{1-\beta_1^2/\beta_2}}\ =\ 2.29906\qquad(\text{DERIVED, five places}),$$
and $(1-\beta_1)/\sqrt{1-\beta_2}=0.1/0.0316228=3.16228$, whose product is $7.2704$. The first repair
printed that factor as $1.0483$ and the chain as $3.3151\,\alpha$: $1.0483=1/\sqrt{1-0.09}$, i.e.
$\beta_1^2$ evaluated as $0.09$ instead of $0.81$, an error of $2.19\times$ in the factor and of
$2.193\times$ in the chain. It is **struck**. Worse, the same paragraph then wrote "the canon uses
$3.1623\,\alpha$ … and is a **lower** bound on the displacement the true step permits" while five
load-bearing places bounded **above** with it — a lower bound used as an upper bound is not a slip of
adjective but an inverted inequality, and every one of those five numbers is re-derived below at
$7.2704\,\alpha$. **The canon's Adam displacement constant is $7.2704\,\alpha$ per step, an upper
bound, and $3.1623\,\alpha$ names only the Kingma–Ba prefactor $\alpha(1-\beta_1)/\sqrt{1-\beta_2}$
with the Cauchy–Schwarz term omitted — a quantity this book bounds nothing with.**

**Numeric instance at $t=1$:** $\hat m_1=g_1$, $\hat v_1=g_1^2$, so $|\Delta\theta_1|=\alpha$ exactly,
which is inside $7.2704\,\alpha$ — the bound is checked at the one step where the step is exact.

**The five downstream numbers, re-derived at $7.2704\,\alpha$** (each replacing a figure quoted at
$3.1623\,\alpha$, and each **larger**, because the true bound is looser than the one the book used):
reaching $\gamma\le0.05$ ($\theta\le\operatorname{logit}0.05=-2.9444$) from $\theta_0=0$ needs at least
$\lceil2.9444/(7.2704\alpha)\rceil$ steps, i.e. **$21$** steps at $\alpha=0.02$ ($20.249$ before
rounding, against the struck $47$) and **$1{,}350$** at $\alpha=3\times10^{-4}$ ($1{,}349.95$, against
the struck $3{,}104$); at $\alpha=3\times10^{-4}$ and $150$ steps,
$|\theta|\le7.2704\cdot3\times10^{-4}\cdot150=\mathbf{0.327168}$ and
$\hat\gamma\in[\mathbf{0.418932},\mathbf{0.581068}]$ whatever the data (against the struck
$0.1423$ and $[0.4645,0.5355]$) — a PINNED verdict there is still pinned-without-signal (Ruling 2a's
census word), and the widened interval makes the sentence weaker, not stronger, which is why it is
printed rather than left at the smaller figure. The remaining two are 02.7's per-step displacement
and 02.8's $\hat a_{\max}$ regime, re-derived in those verses.

**Hypotheses.** Regime S, $\beta=1$; loss on the $O$-channel; Adam defaults, $\epsilon$ negligible;
(a) needs $\mathbb E[\xi\mid V]=0$ **and** a one-hop Bayes-fit label, and is not asserted on any bed
whose label is a three-way feature product; (b) needs $\gamma_e>0$ and
$\|\hat P-P_{\rm env}\|_\infty\le\eta^\star$, and is vacuous unless $F_{\rm rep}\le\eta^\star$; every
$\gamma=0$ statement is a limit along the ray and no cell is ever initialised there; the arena's
lr $0.02$ makes the $\theta$-range reachable ($|\Delta\theta|\le9.487$ in 150 steps at
$3.1623\,\alpha$), the LM lr does not.

**Evidence.** DERIVED (the four displayed steps). Adam constants: `RUN python -c "import
torch,inspect;print(inspect.signature(torch.optim.Adam.__init__))"` on the installed `torch
2.14.0+cpu` (B1: the certified `2.5.1+cu121` carries the same defaults `[ASSUMED]`). Arena lr:
`READ scale/m3_capability.py:88 @ 99777ab`, `scripts/v15_r1.py:235`; LM lr: `READ ceq/lm.py:255 @
99777ab`. Boundary null $2.7055$, $\ln4096=8.318$: `READ docs/PLAN.md:327-334 @ 99777ab` (S-61),
`docs/sources/design/refute_instrument_math.md:190-200`. The record's number: every $\gamma>0$ worse at
$t^\star=2$ (02.2 (i)) — case (a), the label is a product of three features (D-1 regime).

**Mechanism.** D-1, V-9, V-10, M-20, Ruling 2a, Ruling 10′.

**Kill.** Frozen, one number, no second conjunct: on BED-S at the design point, $\Lambda\le2.7055$ on
at least 6 of 8 seeds (R-21). The verse as first written conditioned this kill on
$\|\hat P-P_{\rm env}\|_\infty$ reading within $0.1$ of $F_{\rm rep}$; $F_{\rm rep}$ is `NOT MEASURED`
and no book registers an instrument for it, so that conjunct made the kill undecidable while the kills
table priced it at $0$ GPU-s (V-11, P-4). It is **dropped**: the dial's failure to move is decided by
$\Lambda$ alone, which S-61/R-21 already writes on every cell.
Instrument: S-61/R-21, $0$ GPU-s plus the ablation $(I-\hat\gamma\hat P)^{-1}\to I$ at $\le17.4$ s.
Controls: S-61's plants — a generator at $\gamma_{\rm env}=0$ reads PINNED 8 of 8, at
$\gamma_{\rm env}=0.6$ MOVED 8 of 8. Reachability: decidable at lr $0.02$; a cell trained at
$3\times10^{-4}$ for 150 steps is VOID for R-21 (the dial cannot move), and its verdict is refused.

**If killed → 02.4a, the annealed dial.** Hold $\gamma=\gamma_A=0.5$ for $T_0=50$ steps, then release
$\theta$. At fixed $\gamma>0$ the $\hat P$-gradient, DERIVED with both terms carried explicitly: from
$O=(1-\gamma)PMV$ and $\delta M=\gamma MEM$ for a unit perturbation $E$ of $P$,
$$\delta O=(1-\gamma)\big[EMV+\gamma\,PMEMV\big],\qquad
\partial_EL=\langle r,\delta O\rangle\text{-form}=(1-\gamma)\Big[r\,(MV)^\top+\gamma\,(PM)^\top r\,(MV)^\top\Big],
\quad r=O-y .$$
The first term is $r\,(MV)^\top$ and **not** $M^\top r\,(MV)^\top$: the first writing carried a
spurious leading $M^\top$, which the identity $\langle r,EMV\rangle=\langle E,r(MV)^\top\rangle$
refutes (P-2, an algebraic slip in a DERIVED block). The conclusion is unchanged, because it rests on
the **second** term alone: $\gamma\,(PM)^\top r\,(MV)^\top$ vanishes at $\gamma=0$ and is the only
hop-$\ge2$ channel, so holding $\gamma=\gamma_A>0$ fits $\hat P$ to hops $\ge2$ before the dial is
asked; release then meets case (b). *Hypotheses:* $T_0<150$; the hold is a
**manifest field of that arm**, journalled per cell by S-01 (`READ docs/PLAN.md:627 @ 99777ab`), not a
new instrument — the first writing deferred it to "the schedule flag (book 04)", and no verse of
`04_BEDS_AND_INSTRUMENTS.md` registers a schedule flag, so that deferral is struck (P-4). *Evidence:*
the anneal is one added arm on S-62 carrying $(\gamma_A,T_0)$ in its manifest;
`NOT MEASURED — needs that arm, priced $8\times1.680\approx13.4$ s DERIVED from S-62's per-cell price`.
*Kill:* the released $\hat\gamma$ returns PINNED ($\Lambda\le2.7055$) on at least 6 of 8 — read from
the $\Lambda$ column S-62 already writes, **$\approx13.4$ s** for the added arm, cheaper than 02.4's
$\le17.4$ s ablation. The kills table carries $\approx13.4$ s, not $0$ GPU-s: a run that must happen is
not free.

**If 02.4a is killed → 02.4b, the fixed registered dial.** $\gamma\equiv\gamma_{\rm env}$, the bed constant
of Definition 7 (`READ docs/CEQ_SHAPE.md:330-343 @ 99777ab`); no dial, no $\Lambda$. *Hypotheses:*
$\gamma_{\rm env}$ is registered before any cell (M-2), with $|\gamma_{\rm env}-\tfrac12|\ge0.2$ (02.7).
*Evidence:* the bed spec S-11; one added arm on S-62, `NOT MEASURED — needs that arm, priced
$8\times1.680\approx13.4$ s`. *Kill:* the fixed-$\gamma$ arm's $z$-channel NRMSE within
$\mathrm{MDE}_8$ of depth-1 softmax on at least 6 of 8 — the terminal `eval_nrmse` column S-62 writes,
$\approx13.4$ s for the added arm, and strictly more decisive than 02.4a's: it kills the channel, not
the dial.

**Terminal.** $\gamma$ is a bed-side dial and the shape is softmax wearing a name on this bed (02.12).
Still licensed: parity at $\gamma=0$, the committor head at $\gamma=1$ (which never carried $\gamma$),
regime N at $\gamma=1$.

---

### 02.5 — Identifiability of $P$ from next-state labels, and the representational floor under K-D2

**Statement.** (i) For a fixed known $\gamma_e\in(0,1)$, $z(\hat P;V)=z(P;V)$ for a set of value columns
spanning $\mathbb R^s$ gives $(I-\gamma_e\hat P)^{-1}=(I-\gamma_e P)^{-1}$, hence $\hat P=P$ (one line);
at $\gamma_e=0$ the same holds from $PV$. (ii) On BED-S no single draw supplies a spanning set: the
bed's **value width is $d=24$** — the record's own `S, D = 64, 24` (`READ scripts/v15_r1.py:137 @
99777ab`), journalled as the cell field `d: 24`, and **not** $d_{\rm model}=16$, which is the model
width and was the symbol the first writing wrongly spent here (V-22, one letter carrying two
registered quantities). So one draw supplies $24$ value columns against a causal row-stochastic $P$'s
**$2{,}016$ free entries**: $s(s+1)/2=64\cdot65/2=2{,}080$ is the count of lower-triangular entries,
and row stochasticity imposes $s=64$ equality constraints, leaving $2{,}080-64=2{,}016$ free
(DERIVED). Either count exceeds $24$ by more than $80\times$, so the conclusion is untouched by the
correction: $24<2{,}016$. $P_{\rm env}$ changes per draw, so what is
identified is the map $x\mapsto P$; with $x_i=(a_i,e_i)$ (multi-hot out-adjacency, one-hot id) and
logits $x_i^\top W_Q^\top W_Kx_j$, exact $\hat P=P_{\rm env}$ needs the adjacency-by-id block of
$W_Q^\top W_K$ to equal $c\,I_s$ — rank $s=64$ — impossible at $\operatorname{rank}\le d_{\rm model}=16$;
so $F_{\rm rep}:=\min_{\hat P\in\mathcal C}\|\hat P-P_{\rm env}\|_\infty>0$ at the arena's geometry,
with its magnitude `NOT MEASURED`.

(iii) **The norm convention, stated here because K-D2 states none.** $\|\cdot\|_\infty$ on a matrix is
the maximum absolute row sum, $\|A\|_\infty=\max_i\sum_j|A_{ij}|$, and every $\|\hat P-P_{\rm env}\|_\infty$
in this book and in K-D2 (`READ docs/PLAN.md:375 @ 99777ab`, which prints no convention) is read that
way. With finite logit margin $c$ between edges and non-edges, row $i$ leaks
$$L_i(c)=\frac{(i+1-\deg_i)e^{-c}}{\deg_i+(i+1-\deg_i)e^{-c}}$$
onto its $i+1-\deg_i$ non-edges — and the **same** mass $L_i$ is lost from its $\deg_i$ true edges,
because both rows sum to $1$. The row's absolute deviation is therefore $2L_i$, not $L_i$: the
one-sided reading was the defect (V-17, a threshold imported out of its units). K-D2's $10^{-3}$ at
$i=63$, $\deg_i=4$ ($i+1-\deg_i=60$) needs $L_i<5\times10^{-4}$, i.e.
$60e^{-c}/(4+60e^{-c})=5\times10^{-4}$, $e^{-c}=3.335\times10^{-5}$, and
$$c\ \ge\ \ln 29{,}985.0\ =\ 10.3085\ \Rightarrow\ c\ge10.31\quad(\text{DERIVED}),$$
against $c\ge9.6148$ under an elementwise max $\max_{ij}|A_{ij}|$ — a convention K-D2 does **not** use
and which is named here only to say which number belongs to which norm. The canon's constant is
$c\ge10.31$. Consequence: K-D2 as registered
($\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on at least 6 of 8) has an **undecided reachability** at
$d_{\rm model}=16$ (the reachability rule, CHARTER §2), and the identification reading is printed as
the bare measured column at its own width, with the copy sentence licensed only from the two-width
excess $E$ of the replacement below — never as $\|\hat P-P_{\rm env}\|_\infty-F_{\rm rep}$, which
subtracts a quantity no instrument in this canon computes.

**Hypotheses.** Single head; bilinear logits; features exactly the adjacency row and the id (the S-11
spec: "nothing derived from any solve"); $s=64$; **BED-S's value width $d=24$** and the model width
$d_{\rm model}=16$, printed separately and never substituted for each other — the spanning-set count of
(ii) uses $d=24$, the rank bound of (ii) uses $d_{\rm model}=16$; $\deg_i=4$ `[ASSUMED]` for the
numeric instance only. The rank bound says nothing about $F_{\rm rep}$'s size: a rank-16 form can
carry a sign pattern with a margin, so (ii) proves $F_{\rm rep}>0$, not $F_{\rm rep}\ge10^{-3}$.

**Evidence.** DERIVED. K-D2: `READ docs/PLAN.md:371-377 @ 99777ab` (S-13), `:626-633` (S-62); the
S-11 feature spec `READ docs/PLAN.md:345-358 @ 99777ab`; "$\hat P=P_{\rm env}$ representable at
$d_{\rm model}\ge n_{\rm nodes}$" `READ docs/CEQ_SHAPE.md:517-520 @ 99777ab`; arena $d_{\rm model}=16$
`READ docs/PLAN.md:627`; the bed's value width $d=24$ `READ scripts/v15_r1.py:137 @ 99777ab`
(`S, D = 64, 24`), confirmed as a journalled cell field by
`RUN python -c "import json;[print(r['kind'],r['d'],r['d_model']) for r in map(json.loads,
open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell']"`, 2026-09-06 — `d: 24, d_model: 16`;
$2{,}016=2{,}080-64$ DERIVED. The record's one identification number, on another bed: attribution
recovery needs $m\ge C\,s\ln(n/s)$ with $C=1.44270$ measured, `READ MATHEMATICS.md:753-807 @
99777ab` — a floor, never a claim that training reached it. The logit scale reachable in 150 steps
at lr $3\times10^{-4}$ ($|\Delta w|\le0.1423$ per weight): `NOT MEASURED — needs the per-step max-logit
trace`.

**Mechanism.** V-11 (a kill that cannot fire is not a kill), V-10, D-2, M-7, P-4 (a kill carrying RUN
and NOT MEASURED at once, on an instrument no book registers), V-17 (a constant derived under an
unstated norm convention), Ruling 3 (a replacement escaping by re-architecting the width).

**Kill, re-based on a column the arena already writes.** The verse as first written killed itself on
`RUN $F_{\rm rep}$ at construction on 512 draws` **and** in the same sentence declared
`NOT MEASURED — needs the representational-floor instrument (book 04)`: two evidence classes on one
number, and no verse of `04_BEDS_AND_INSTRUMENTS.md` registers an $F_{\rm rep}$ instrument (P-4,
claimed scaffolding that does not exist). $F_{\rm rep}$ is a minimum over the class and no registered
instrument computes it, so no kill of this book may condition on it.

Frozen instead on the arena's own registered column: **$\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on at
least 6 of 8 seeds at $d_{\rm model}=16$**, read under the matrix $\infty$-norm of (iii). A realised
$\hat P$ is a member of $\mathcal C$, so that reading is an **upper bound** on $F_{\rm rep}$: it fires
$F_{\rm rep}<10^{-3}$ as a theorem, K-D2 stands exactly as registered, and this verse's
undecided-reachability consequence is withdrawn. Instrument: S-62's `*measure*` line
$\|\hat P-P_{\rm env}\|_\infty$ per seed (`READ docs/PLAN.md:626-633 @ 99777ab`), journalled in
04.15's `census` block; price $0$ GPU-s, a read of the arena journal, no second run. Planted negative
that would make it fire: the BED-S64 cell of the replacement below, at $d_{\rm model}=64\ge s$, where
the block $W_Q^\top W_K=c\,I_s$ is representable and the column must read $<10^{-3}$ on 8 of 8.
Planted positive that must **not** fire: the step-0 cell of the same arm, whose column must read
$\ge0.5$ on 8 of 8 (an untrained bilinear logit is near-uniform on the causal prefix).

**If killed** ($\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on $\ge6/8$ at width 16): K-D2 as filed, and
(ii)'s rank argument bounds nothing the bed can see.

**If not killed (the expected case) → K-D2′, a different number on a different object, measured by a
different instrument.** No replacement of this verse may name $F_{\rm rep}$, so the copy reading is
re-derived as a **difference between two widths**: register **BED-S64** — the same generator, seed
rule and hash (Ruling 7) at $d_{\rm model}=64$, with **its own** depth-1 control, its own exact
restricted-view floor and its own $\mathrm{MDE}_N$ from its own realised paired sd. BED-S64 is filed
as a **reroute to a second bed**, never as a re-match of the $d_{\rm model}=16$ row: Ruling 3 forbids
re-architecting to close a count (`READ docs/canon/CHARTER.md:79-80 @ 99777ab`), and the verse as
first written cited Ruling 3 as the licence for the very move it forbids — raising the arm's width
$16\to64$ is a $4\times$ re-architecture, and no count of the width-16 row may be re-derived at width
64. Then "copied" is licensed on the **excess**
$$E:=\|\hat P-P_{\rm env}\|_\infty^{(16)}-\|\hat P-P_{\rm env}\|_\infty^{(64)}\ \ge\ 0\ \text{expected},$$
a measured quantity with no unmeasured term in it. *Hypotheses:* both cells exist under Ruling 7 with
their own controls, floors and $\mathrm{MDE}_N$; the two rows are never differenced as a capability
contrast (04.1's VOID list), only as identification. *Evidence:* `NOT MEASURED — needs the BED-S64
cell, priced as a second arena at $\approx98$ s DERIVED from S-62`. *Kill of K-D2′:* the width-64
column itself reads $\ge10^{-3}$ on at least 6 of 8 — then width is not the binding constraint, the
gap is optimisation and not representation, $E$ measures nothing, and the identification reading is
VOID on both beds. Decided at $0$ GPU-s from the BED-S64 journal; **strictly more decisive** than the
verse's own kill, which cannot separate "the class cannot" from "training did not".

**Terminal.** Identification is a **measured** quantity or it is nothing: the row prints
$\|\hat P-P_{\rm env}\|_\infty$ under the matrix $\infty$-norm at its own width, and the sentence "the
arm copied the environment" is licensed only from that column at $d_{\rm model}\ge s$ or from the
two-width excess $E$. $F_{\rm rep}$ names a real obstruction and is `NOT MEASURED` with no instrument
in any book; no kill, no replacement and no row of this canon conditions on it, and (ii) survives as
the sign statement $F_{\rm rep}>0$ and nothing more. If BED-S64 is never run, $E$ is never read, and
the canon withdraws every "copied" sentence at $d_{\rm model}=16$ while keeping K-D2's threshold as
the one decidable reading the arena writes.

---

### 02.6 — Committor labels: what they identify, and committor supervision as the auxiliary loss

**Statement.** (i) The committor head $\hat q=(I-\hat Q)^{-1}\hat R_k\mathbb 1$ carries no $\gamma$
(Definition 7's ruling); for a perturbation $\delta Q$, $\delta\hat q=(I-\hat Q)^{-1}\delta Q\,\hat q$
and $\delta\hat q=(I-\hat Q)^{-1}\delta R\,\mathbb 1$ (DERIVED, one line each from
$(I-\hat Q)\hat q=\hat R\mathbb 1$), so the gradient reaches $\hat P$ through the exact solve
independently of the dial. (ii) A row clamp at $v_a$ moves $\hat q$ along one column
$(I-\hat Q)^{-1}e_{v_a}$ up to a scalar (Proposition 7(iii)); $m$ moves times $K+2$ channels reveal at
most $m$ columns and one right-hand side each; $m=8<|T|\le62$: $\hat Q$ is **not identified per draw**
from committor labels, and identification is through the token map across draws (02.5).

(iii) **The weight is a swept manifest field, not an assumption.** The auxiliary loss is
$L=L_z+\lambda L_q$, and the clause "gives $\hat P$ a hop-$\ge2$ gradient at every $\gamma$" is a
statement about the **relative** magnitude of $\lambda\,\partial_PL_q$ and $\partial_PL_z$: at
$\lambda$ small enough it is false, and at $\lambda=0$ it is false identically. $\lambda=1$
`[ASSUMED]` therefore made the whole clause rest on an unpriced scale (P-1), and book 08 cannot build
the module from an assumption. The floor is DERIVED as a ratio of two gradient norms at
initialisation,
$$\lambda\ \ge\ \lambda_{\min}\ :=\ \frac{\big\|\partial_PL_z\big\|_F\big|_{t=0}}
{\big\|\partial_PL_q\big\|_F\big|_{t=0}},$$
the weight at which the committor channel's contribution to $\partial_P L$ is at least the
$z$-channel's, so the hop-$\ge2$ component is not dominated by the one-hop fit. Both norms are bed
constants at initialisation, read from one forward–backward on the first cell:
`NOT MEASURED — needs the first BED-S cell (04.21), whose step-0 backward supplies both norms at
$0$ GPU-s`. Until that reading exists $\lambda$ is **registered as a swept manifest field of the arm
(S-01) with its grid frozen here (M-2)**:
$$\lambda\in\Lambda_{\rm grid}:=\{0,\ \tfrac14,\ 1,\ 4,\ 16\},\qquad|\Lambda_{\rm grid}|=5,$$
five arms on S-62, one decade and a half either side of the first writing's $1$, with $\lambda=0$ the
$L_z$-only baseline and simultaneously this verse's planted negative. No sentence of this book is
asserted at a $\lambda$ outside that grid, and no number of book 08 may read $\lambda=1$ as derived.

With $\lambda\ge\lambda_{\min}$ the auxiliary loss gives $\hat P$ a hop-$\ge2$ gradient at
every $\gamma$, a pinned one included, and **drives $\hat P$ toward the $F_{\rm rep}$-ball**,
$\{\hat P\in\mathcal C:\|\hat P-P_{\rm env}\|_\infty=F_{\rm rep}\}$ — **not** toward $P_{\rm env}$.
The limit $\hat P\to P_{\rm env}$ is empty at $d_{\rm model}=16<s=64$ (02.5(ii)), and a loss cannot
make an empty limit live; the clause as first written asserted exactly that (V-25). What the ball
buys is stated with its radius: 02.4(b) derives
$\partial_\gamma L|_{0^+}<0$ for every $\hat P$ with $\|\hat P-P_{\rm env}\|_\infty\le\eta^\star$,
$$\eta^\star=\min\left\{\frac{\gamma_e\|G\|_F}{2\sqrt s\,\nu\,(1+4\gamma_e)},\ \
\frac{\|G\|_F}{2\nu}\sqrt{\frac{\gamma_e}{2s}}\right\},$$
so committor supervision carries a **dial consequence iff $F_{\rm rep}\le\eta^\star$**, and inside the
ball but outside the $\eta^\star$-radius 02.4(c) leaves the dial's gradient sign undetermined. The
decidable form, on the column S-62 writes: the sign is live on a seed whose journalled
$\|\hat P-P_{\rm env}\|_\infty\le\eta^\star$, that reading being an upper bound on $F_{\rm rep}$;
$\|G\|_F$ and $\nu$ are bed constants at $0$ GPU-s and their values are
`NOT MEASURED — needs the first BED-S cell (04.21)`. Until a seed prints a column entry at or below
$\eta^\star$, **committor supervision is a fix for the head and not for the dial**: it carries no
corner consequence at all. What it does carry without any condition is the head itself — the
safest-move capability (H-q) is trained through the exact $\gamma=1$ solve whether or not $\gamma$
pins, because (i) puts no $\gamma$ in that gradient; $\gamma$-pinning kills the horizon-dial sentence
and the $z/\Delta z$ channel only.

**Hypotheses.** BED-S admitted; $0\in\mathcal A_{\rm sink}$ so $\rho(\hat Q)=\max_{i\in T}\hat P_{ii}<1$
(Proposition 10); $|T|\ge m$; the label is the $\gamma=1$ Dirichlet solve on $P_{\rm env}$, never on
$\hat P$ (D-2); $\lambda\in\Lambda_{\rm grid}$ and, for the gradient clause, $\lambda\ge\lambda_{\min}$,
which is `NOT MEASURED` — so the gradient clause is asserted at no realised $\lambda$ in this book and
is a conditional statement until the step-0 backward prints both norms. Clause (iii)'s **head** half is
asserted unconditionally; its **dial** half is
non-vacuous only where $F_{\rm rep}\le\eta^\star$ at the bed's $(\gamma_e,\|G\|_F,\nu,s)$, and that
inequality is `NOT MEASURED` at the arena, so the dial half is asserted nowhere in this book.

**Evidence.** DERIVED. Definition 7 `READ docs/CEQ_SHAPE.md:330-343 @ 99777ab`; Proposition 7(iii)
`:466-471`; Proposition 10 `:539-548`; FATAL-2 (why the head carries no $\gamma$: $q$ to $1\%$ at
$\tau=8$ needs $\hat\gamma\ge0.99857$) `READ docs/sources/design/refute_instrument_occvac.md:213-229 @
99777ab`. The record has no trained committor cell; BED-1's `bed["q"]` matched the resolvent read at
$0.0$ only as an identity on the bed's own $P$ (`READ docs/CEQ_SHAPE.md:369-389 @ 99777ab`):
`NOT MEASURED — needs S-62`.

**Mechanism.** D-2, V-9, M-20, V-12, V-25 (a fix asserted at a limit the geometry proves empty).

**Kill.** V-9, frozen, read at the grid's $\lambda=1$ arm: its $\|\hat P-P_{\rm env}\|_\infty$ within
one seed sd of the $\lambda=0$ arm's on at least 6 of 8 seeds **and** the R-21 verdict unchanged on at
least 6 of 8 — the auxiliary loss changed nothing. Instrument: S-62 with the four added arms of
$\Lambda_{\rm grid}$ beyond the $\lambda=0$ baseline, $4\times8\times1.680\approx53.8$ s.

**Planted negative that would make the kill fire (V-15).** $\lambda=0$ against itself: the grid's own
baseline arm, re-run at a second seed block, must fire the kill on **8 of 8** — the auxiliary loss is
literally absent there, so an instrument that does not condemn $\lambda=0$ cannot condemn anything, and
the verdict "the auxiliary loss changed nothing" is proved reachable on a known-bad plant before it is
read on $\lambda=1$. The first writing carried only a planted **positive** where CHARTER §2 requires a
planted negative, so the condemning rule had no plant it must fire on (V-15); that plant is supplied
here and costs nothing beyond the grid, $\lambda=0$ being an arm the sweep already runs.
Planted positive that must **not** fire: $\lambda=16$, the grid's top (the first writing wrote
$\lambda\to\infty$, which is not an arm any manifest can carry, M-2), must move $\hat P$ by more than
one seed sd from the $\lambda=0$ arm on 8 of 8 — the loss is live.

**If killed.** Direct $\hat P$ supervision, row-wise cross-entropy $L_P=-\sum_{ij}P_{{\rm env},ij}
\log\hat P_{ij}$ on the edge tokens, legal only as a **pretraining stage** and filed as identification
(it is the copy task by construction, D-2). *Hypotheses:* the stage ends before any scored cell and
its cells carry `void_contrasts`. *Evidence:* E1's registration comment as the model, `READ
scale/negation_scope.py:1033-1041 @ 99777ab`. *Kill:* after the stage the $z$-channel is still within
$\mathrm{MDE}_8$ of depth-1 softmax on at least 6 of 8 — $0$ GPU-s journal read, decides the channel.

**Terminal.** The committor head is exact at $\gamma=1$ on whatever $\hat P$ the tokens produce and
needs no dial; if $\hat P$ cannot be learned from tokens, no supervision on any head repairs it, and
BED-S reads identification only.

---

### 02.7 — Interventional pairs at the registered initialisation: the corner as algebra, the dial's sign at $\gamma_0=\tfrac12$

**Statement.** Two statements, kept apart, because the first is an identity about a point no cell
occupies and only the second is a training-side theorem.

**(A) Corner algebra — an identity, and withdrawn as a training-side claim.** Under a row clamp
$P'=\hat P+e_iu^\top$, $u^\top\mathbb 1=0$ (Definition 6(i)), the first-order-in-$u$ displacement of
$O(\gamma)=(1-\gamma)\hat P\hat MV$, $\hat M=(I-\gamma\hat P)^{-1}$, follows from
$\delta[\hat P\hat M]=\Delta P\,\hat M+\gamma\,\hat P\hat M\,\Delta P\,\hat M$ at $\Delta P=e_iu^\top$:
$$\Delta O=(1-\gamma)\Big[e_i\big(u^\top\hat MV\big)+\gamma\,\big(\hat P\hat Me_i\big)\big(u^\top\hat MV\big)\Big],
\qquad\text{so for }j\ne i:\quad \Delta O_j=(1-\gamma)\,\gamma\,(\hat P\hat M)_{ji}\,\big(u^\top\hat MV\big)$$
(DERIVED, exact in $\gamma$ at first order in $u$; its $O(\gamma)$ truncation is the
$\Delta O_j=\gamma\hat P_{ji}(u^\top V)+O(\gamma^2)$ of the first writing). Hence
$\Delta O_j|_{\gamma=0}=0$: the softmax corner predicts **zero downstream consequence exactly**. Along
the $\gamma\to0^+$ ray, with $L=\tfrac12\sum_{j>i}\|\Delta O_j-\Delta z^{\rm lab}_j\|^2$ and the bed's
label $\Delta z^{\rm lab}_j=\gamma_eP_{{\rm env},ji}(u^\top V)+O(\gamma_e^2)$,
$$\partial_\gamma L\big|_{\gamma=0}=-\sum_{j>i,\ j\in T}\hat P_{ji}\,\big\langle u^\top V,\ \Delta z^{\rm lab}_j\big\rangle
=-\gamma_e\,\big\|u^\top V\big\|_2^2\sum_{j>i,\ j\in T}\hat P_{ji}P_{{\rm env},ji}<0 ,$$
**the sum running over the transient rows $j\in T$ only**, whenever $u^\top V\ne0$ and $P_{\rm env}$ has
at least one **transient** row $j>i$ with $P_{{\rm env},ji}>0$. **The square is a squared norm, not a
scalar square.** $V\in\mathbb R^{s\times d}$ with the bed's value width $d=24$ at the arena (the
journalled cell field `d: 24`, 02.5), so $u^\top V\in\mathbb R^{d}$ is a $d$-vector, $\Delta z^{\rm lab}_j$
is a $d$-vector, and their pairing is the inner product written above; the scalar writing
$-\gamma_e(u^\top V)^2\sum\dots$ of the first writing is **struck** as a type error (P-2, a $d$-vector
squared as a scalar), and $\|u^\top V\|_2^2\ge0$ carries the sign conclusion unchanged — the
inequality, its hypothesis $u^\top V\ne0$ (read as: the $d$-vector is not the zero vector) and every
kill below are untouched by the correction. The universal form — "$\hat P_{ji}>0$
for every $j>i$, because softmax with finite logits is everywhere positive" — is **false on this
architecture** and is struck (V-25, a positivity asserted where the shape builds exact zeros):
Definition 3 makes the sink, the goal set $\mathcal A_0$ and the $K$ constraint sets
$\mathcal A_1..\mathcal A_K$ **identity rows** (`READ docs/CEQ_SHAPE.md:295-296 @ 99777ab`), and S-10
places "the sink, $\mathcal A_0$ and $\mathcal A_1..\mathcal A_K$ all before the query"
(`READ docs/PLAN.md:352 @ 99777ab`), so BED-S puts absorbing rows at exactly the positions $j>i$ the
sum ranges over. For an absorbing $j$, $\hat P_{ji}=0$ **exactly** — not small, zero by construction —
and its term vanishes; with $K=2$ the arena carries $K+2=4$ such rows. The inequality therefore holds
on the transient sub-sum and only there, and the kill's census fraction below measures exactly that
sub-sum's support. **That inequality is corner algebra and nothing more.** The
registered parametrisation is $\gamma=\sigma(\theta)$, $\theta_0=0$ (02.4), so $\gamma=0$ is attained
only at $\theta=-\infty$, $\partial_\theta L=\gamma(1-\gamma)\partial_\gamma L\to0$ as $\gamma\to0$, and
no cell is ever initialised there. The sentence the verse first carried — "the corner is a strict
descent direction of the dial at every initialisation" — quantified over $\hat P$ and **never over
$\gamma$**, and is **withdrawn** (V-10: the corner is absorbing in the parameter that actually moves,
so the gate was satisfied by construction).

**(B) The training-side theorem, computed at the initialisation the bed registers.** At $\theta_0=0$,
$\partial_\theta L|_{\theta=0}=\tfrac14\,\partial_\gamma L|_{\gamma=1/2}$. Write the displacement of (A)
as
$$F(\gamma):=\phi(\gamma)\,C(\gamma)\,w(\gamma),\qquad \phi(\gamma):=(1-\gamma)\gamma,\quad
C(\gamma)_j:=(\hat P\hat M)_{ji}\ (j>i),\quad w(\gamma):=u^\top\hat MV,$$
so $\Delta O_{>i}=F(\gamma)$ and the label is $F_{\rm env}(\gamma_e)$, the same map on $P_{\rm env}$.
At $\hat P=P_{\rm env}$ the loss is $L(\gamma)=\tfrac12\|F(\gamma)-F(\gamma_e)\|_F^2$, zero at
$\gamma=\gamma_e$, and
$$\partial_\gamma L(\gamma)=\Big\langle\int_{\gamma_e}^{\gamma}F'(t)\,dt,\ F'(\gamma)\Big\rangle
\qquad(\text{DERIVED, one line from }F(\gamma)-F(\gamma_e)=\textstyle\int_{\gamma_e}^{\gamma}F').$$
**Cone condition.** If $\langle F'(t),F'(\gamma)\rangle>0$ for every $t$ between $\gamma_e$ and $\gamma$
— the direction of $F'$ turns by less than $90^\circ$ across the interval — then
$\operatorname{sign}\partial_\gamma L(\gamma)=\operatorname{sign}(\gamma-\gamma_e)$. At $\gamma=\tfrac12$,
**at the registered initialisation and not at the corner**,
$$\operatorname{sign}\,\partial_\theta L\big|_{\theta=0}\ =\ \operatorname{sign}\big(\tfrac12-\gamma_{\rm env}\big),$$
so $\hat\gamma$ descends **toward $\gamma_{\rm env}$**: up from $\tfrac12$ when $\gamma_{\rm env}>\tfrac12$,
down when $\gamma_{\rm env}<\tfrac12$. Note $\phi'(\tfrac12)=1-2\gamma=0$, so
$F'(\tfrac12)=\tfrac14\,(Cw)'(\tfrac12)$: the whole first-order signal at the initialisation comes from
the resolvent's own $\gamma$-dependence and none of it from the $(1-\gamma)\gamma$ prefactor.

**The registration this forces, frozen here (M-2).** At $\gamma_{\rm env}=\tfrac12$ the initialisation
**is** the optimum, $\partial_\theta L|_0=0$, and R-21 must read PINNED with the dial working perfectly
— a PINNED verdict there is evidence about the registration and not about the arm. BED-S is therefore
registered with $|\gamma_{\rm env}-\tfrac12|\ge0.2$, frozen before any cell, and a cell run at
$\gamma_{\rm env}\in(0.3,0.7)$ is **VOID** for this verse's kill.

**Where (B) is asserted.** At $\hat P=P_{\rm env}$ exactly, which 02.5(ii) proves **empty** at
$d_{\rm model}=16<s=64$. The map $\hat P\mapsto\partial_\gamma L|_{1/2}$ is a rational function of
$\hat P$, continuous on the row-stochastic set (where $\rho(\tfrac12\hat P)\le\tfrac12<1$, so $\hat M$
exists and is analytic), hence a radius exists on which the sign survives; its size is
`NOT MEASURED — needs $\|F'\|$ and its Lipschitz constant in $\hat P$ at the bed, both computable from
the generator at $0$ GPU-s once 04.21 prints the first cell`. Until that radius is read, (B)'s sign is
asserted at $\hat P=P_{\rm env}$ and on a neighbourhood of unmeasured size, and nowhere else. What
survives with **no** condition on $\hat P$ is (A) alone: the corner predicts zero downstream
consequence exactly, which is why the consequence channel is the channel the corner cannot occupy —
an exactness statement, never a speed or accuracy statement.

**Hypotheses.** Regime S, $\beta=1$; clamp position $i$ before the query with at least one downstream
true edge; **at least one transient row $j>i$ with $P_{{\rm env},ji}>0$** — the absorbing rows of
Definition 3 (sink, $\mathcal A_0$, $\mathcal A_1..\mathcal A_K$) are identity rows with
$\hat P_{ji}=0$ exactly and contribute nothing, and S-10 places all $K+2$ of them before the query, so
this hypothesis is not free at the arena's geometry; $u^\top V\ne0$ ($V\equiv\mathbb 1$ gives
$\Delta z\equiv0$, Proposition 7(iv), the planted
negative); the loss restricted to transient rows $j>i$ — row $i$ carries the zero-order term $e_iu^\top\hat MV$,
visible at the corner, and must be excluded or the corner is not the zero point; the label at
$\gamma_e>0$; the bed's clamps are query-side (S-11). (A) needs nothing further and is an identity.
(B) needs $\hat P=P_{\rm env}$ (empty at $d_{\rm model}=16$, so it is asserted on a neighbourhood of
unmeasured radius), the cone condition on $[\min(\gamma_e,\tfrac12),\max(\gamma_e,\tfrac12)]$, and the
frozen registration $|\gamma_{\rm env}-\tfrac12|\ge0.2$; it is asserted at no other initialisation and
determines no sign at $\gamma=0$.

**Evidence.** DERIVED (the displayed steps of (A) and (B)). Proposition 7
`READ docs/CEQ_SHAPE.md:460-483 @ 99777ab`; the B-G1 plants $0.1096/0.363/1.127$ and
$V\equiv\mathbb 1\Rightarrow1.1\times10^{-16}$ `READ docs/CEQ_SHAPE.md:2570-2572 @ 99777ab`. The
parametrisation and $\partial_\theta L=\gamma(1-\gamma)\partial_\gamma L$: 02.4. **The instrument is
registered and the cell is not.** S-62's `*measure*` list already writes, per arm and seed, the field
cosine and magnitude ratio of the displacement read, $\hat\gamma$, $\Lambda$ and
$\|\hat P-P_{\rm env}\|_\infty$ (`READ docs/PLAN.md:627 @ 99777ab`); no verse of this book defers this
kill to an unregistered instrument. What does not exist is the cell: no interventional pair has ever
been trained, `NOT MEASURED — needs one added arm on S-62 (04.21) carrying interventional pairs in the
loss, $8\times1.680\approx13.4$ s DERIVED from S-62's per-cell price`. The record's nearest number is a
zero declared without a movement test, `impact_flipper_dependence = 0.0` (`READ MISTAKES.md:755-769 @
99777ab`, D-5) — a training signal declared absent by construction.

**Mechanism.** D-5, V-24, D-1 (the label class leaves the scalar-at-$s-1$ regime), V-8, V-11, V-10 (the
withdrawn corner reading), P-4 (an instrument deferred to a book that registers none), P-2 (a
$d$-vector squared as a scalar at $d=24$).

**Kill, split in two because the Statement and the training column decide different things
(L-VERSE).** The first writing froze one kill — R-21 PINNED, or $\hat\gamma$ moving against
$\operatorname{sign}(\gamma_{\rm env}-\tfrac12)$ — and named it as deciding "the derivative's sign is
wrong **or** its magnitude is below the Adam floor". A PINNED verdict cannot separate those two
disjuncts: by 02.4's bound the per-step displacement is at most $3.1623\,\alpha$, so a correct negative
$\partial_\gamma L|_{0^+}$ of small magnitude produces PINNED with the sign entirely right, and the
Statement's DERIVED algebra was unfalsifiable by the instrument named. The two are separated:

**Kill (a) — the arithmetic check, deciding the sign the Statement asserts.** Frozen: evaluate
$$S:=\sum_{j>i,\ j\in T}\hat P_{ji}\,P_{{\rm env},ji}\quad\text{and}\quad
\partial_\gamma L\big|_{0^+}=-\gamma_e\,\big\|u^\top V\big\|_2^2\,S$$
numerically on **one admitted draw** of the S-12 census, at the bed's own $(P_{\rm env},u,V,\gamma_e)$
and at $\hat P$ read from the same draw's step-0 forward. The kill fires iff
$\partial_\gamma L|_{0^+}\ge0$ on that draw — then (A)'s inequality is false at the bed's own numbers
and the Statement dies as algebra. Instrument: the S-12 admitted-draw census on the generator, which
already computes $\|\Delta z^{\rm lab}_{>i}\|_\infty$ per draw from the same four objects; price
**$0$ GPU-s**, no cell, no training. Planted negative that would make it fire: $u^\top V=0$
($V\equiv\mathbb 1$), where $\partial_\gamma L|_{0^+}=0\ge0$ and the check must fire on 1 of 1; and a
draw whose only $j>i$ rows are the $K+2$ absorbing rows of Definition 3, where $S=0$ exactly and it
must fire again — the two plants that the V-25 repair above makes reachable. Planted positive that
must **not** fire: any draw with a transient $j>i$ edge and $u^\top V\ne0$.

**Kill (b) — the training check, deciding reachability and nothing else.** Frozen, on the columns
S-62 writes, with two disjuncts and no undecidable conjunct: on the
added interventional-pair arm, **either** $\Lambda\le2.7055$ on at least 6 of 8 seeds (R-21: the dial
does not move at all), **or** $\operatorname{sign}(\hat\gamma-\tfrac12)\ne\operatorname{sign}(\gamma_{\rm env}-\tfrac12)$
on at least 6 of 8 seeds (the dial moves the wrong way, contradicting (B)'s sign). Both read off
$\hat\gamma$ and $\Lambda$, which S-62 already measures. **What (b) decides is reachability under the
Adam step bound — whether $3.1623\,\alpha T$ of displacement budget moves the dial off $\tfrac12$ far
enough for R-21 to see it — and never the sign of $\partial_\gamma L|_{0^+}$, which is (a)'s and
(a)'s alone.** A PINNED verdict with (a) not fired licenses "the sign is right and the budget is too
small", not "the derivative is wrong". Instrument: S-61/R-21 on that arm, price
$\approx13.4$ s for the added arm plus the ablation $\le17.4$ s. Reachability (V-11): the census prints
the fraction of admitted draws with $\|\Delta z^{\rm lab}_{>i}\|_\infty>0.05\,\mathrm{sd}$ **restricted
to transient rows $j>i$, the absorbing rows of Definition 3 excluded from the norm** — so the fraction
measures the support of the transient sub-sum $S$ and nothing else, and a bed whose downstream rows are
all absorbing reads $0$ there rather than passing on rows that contribute exactly zero; below $0.5$
the bed cannot decide the kill and the cell is VOID; and a cell registered at
$\gamma_{\rm env}\in(0.3,0.7)$ is VOID by (B)'s degenerate case. Planted negative that would make it
fire: $V\equiv\mathbb 1$, where $u^\top\hat MV=0$ and $\Delta z\equiv0$, must read PINNED on 8 of 8.
Planted positive that must **not** fire: a bed registered at $\gamma_{\rm env}=\tfrac12$ must read
PINNED on 8 of 8 by (B) with the dial working — showing the verdict tracks the registration and not
the arm, which is exactly why $\gamma_{\rm env}=\tfrac12$ is refused above. Lr $0.02$ (02.4).

**If killed → 02.7a, the clamp magnitude, not the horizon.** The first writing's replacement raised
$\gamma_{\rm env}$ to $0.9$ "because the label is $\propto\gamma_e$ to first order". (A)'s exact
prefactor refutes that: the label carries $\phi(\gamma_e)=(1-\gamma_e)\gamma_e$, which is **maximised
at $\gamma_e=\tfrac12$** and reads $0.09$ at $0.9$ against $0.25$ at $0.5$ — raising the horizon
**shrinks** the first-order label by $2.78\times$, and only the resolvent's $1/(1-\gamma_e)$ growth
pulls the other way, with no determined net sign. That replacement is therefore struck as algebra, not
softened.

The replacement in force is a **different knob, a different number and a different instrument**: the
displacement is exactly linear in the clamp, $\Delta O_{>i}\propto u$ with no $\gamma$ in the
proportionality, so scaling $u\to\kappa u$ scales every downstream label by $\kappa$ exactly. Freeze
$$\kappa^\star:=\frac{0.05\,\mathrm{sd}}{q_{50}},\qquad
q_{50}:=\operatorname{median}_{\rm draws}\big\|\Delta z^{\rm lab}_{>i}\big\|_\infty\ \text{at}\ \kappa=1,$$
a one-line rescaling of a census the bed already computes on the generator. *Hypotheses:* the
rescaling is registered before any cell (M-2); the clamped row must stay a probability row, and
$u^\top\mathbb 1=0$ gives the row sum for free, so the binding constraint is
$\kappa\le\kappa_{\max}:=\min_{j:\,u_j<0}\hat P_{ij}/|u_j|$. *Evidence:* DERIVED (linearity in $u$);
the census is $0$ GPU-s (S-12). *Kill of 02.7a — a different object from the verse's:*
$\kappa^\star>\kappa_{\max}$ on more than half the admitted draws — the rescaling that would lift the
label above the census bar takes the clamp **outside the simplex**, so no admissible intervention on
this bed produces a readable downstream label at all; Bet A is VOID on BED-S with the reason named.
Decided at $0$ GPU-s on the generator before any training, **strictly cheaper** than the verse's kill
(no cell is run) and **strictly more decisive** (it condemns the bed, not the arm).

**Terminal.** The consequence channel on BED-S is VOID; the displacement identity survives as
algebra (Proposition 7) and the channel as a cost statement (`docs/PLAN.md` §5.10, S-33's death).

---

### 02.8 — The gate modulus: $|a|\le1$ by construction, its cost, and the lr regime derived from R1

**Statement.** R1's three failures are gate divergence, not the hop algebra. **Two lanes measured the
same three seeds and disagree, so both columns are printed and the lane the kill runs in is named**
(Ruling 2a: no sentence transfers across corners — or lanes — without a bind at that corner):

| seed | eval NRMSE (CPU) | $\hat a_{\max}$, CPU lane | $\hat a_{\max}$, CUDA lane | eval NRMSE (CUDA) |
|---|---|---|---|---|
| 0 | $0.645614$ | $1.4107$ | $1.4104527235031128$ | $0.6446726192039927$ |
| 1 | $0.644454$ | $1.2868$ | $1.2868505716323853$ | $0.6445174549187496$ |
| **2** | $1.152430$ | $20.3090$ | $12.767516136169434$ | $1.1522795055459243$ |
| **3** | $1.113403$ | $49.6613$ | $49.6605224609375$ | $1.1133392329955414$ |
| 4 | $0.634002$ | $1.4390$ | $1.4536346197128296$ | $0.6337391039935976$ |
| 5 | $0.641881$ | $1.5051$ | $1.5051767826080322$ | $0.6419986310848815$ |
| 6 | $0.662021$ | $1.1029$ | $1.1029453277587890$ | $0.6621282051474511$ |
| **7** | $1.139404$ | $285.0719$ | $116.00607299804688$ | $1.1489267727154717$ |

The CPU lane is `results/v15_r1.txt`, header `torch 2.5.1+cu121 torch.get_num_threads()=8 device=cpu`;
the CUDA lane is `results/v17k_r4_retake.jsonl`, header `device: cuda, threads: 8, torch 2.5.1+cu121`,
same task `e3_t2`, same eight seeds, same $150$ steps, same lr $0.02$. The lanes agree to five
significant figures on the five crossing seeds and **disagree by $1.59\times$ at seed 2 and by
$2.457\times$ at seed 7** ($285.0719/116.00607=2.4574$) — on the very quantity the verse's mechanism
rests on. **The kill runs on the CUDA lane**: S-52 registers the capped run "with the same thread lane
and flag regime as the retake journal" (`READ docs/PLAN.md:523 @ 99777ab`), and this verse's Kill
control quotes the retake journal's $0.6446726192039927$ / $0.6445174549187496$ bitwise. **The lane's
maximum is therefore $116.006$, and $285.07$ does not occur where the kill fires**; $285.07$ is kept in
the table as the CPU lane's reading with that label and is used in no bound.

**The bound, and what it cannot buy.** By 02.4's Adam bound, from $g_0\approx0$,
$$\hat a_{\max}\le\exp(3.1623\,\alpha\,T),$$
**a function of the displacement budget $\alpha T$ alone.** At the arena's $(\alpha,T)=(0.02,150)$,
$\alpha T=3.0$ and the bound reads $\exp(9.487)=1.32\times10^4$, which contains the CUDA lane's
$116.006$ (loose by $114\times$) and the CPU lane's $285.072$ (loose by $46\times$). Read against the
lane the kill runs in, the $\alpha$ at which the bound reproduces that lane's own worst case is
$$\alpha=\frac{\ln 116.00607}{3.1623\cdot150}=1.0021\times10^{-2}\quad(\text{DERIVED}),$$
i.e. the arena's $0.02$ is $2.0\times$ the largest lr whose bound is as tight as the realised
divergence. Keeping $\hat a_{\max}\le1.51$ by the bound alone needs
$\alpha T\le\ln1.51/3.1623=0.130317$; **at $T=150$ that is $\alpha\le8.688\times10^{-4}$, and the step
count cannot compensate**, because the bound sees only the product: at
$(\alpha,T)=(8.69\times10^{-4},3{,}453)$ the product is $3.0$ again and the bound returns to
$1.32\times10^4$. The first writing claimed both — $\alpha=8.69\times10^{-4}$ *and* $T=3{,}453$ *and*
"the bound itself keeps $\hat a_{\max}\le1.51$" — and the three are mutually inconsistent at
$\exp(9.489)=1.32\times10^4$ (P-2). **The regime in force is $(\alpha,T)=(8.69\times10^{-4},150)$ and
the displacement budget shrinks $23.0\times$ with it**: $\alpha T$ falls from $3.0$ to $0.130317$, and
the whole-run displacement bound falls from $|\Delta\theta|\le9.487$ to $|\Delta\theta|\le0.4121$ on
every parameter — which by 02.4's own arithmetic is the pinned-without-signal regime, so a persisting
failure there says nothing about the hop algebra and the regime is filed as a *diagnosis of the gate*,
never as a training route.

The constraint by construction — $a=\tanh(g)$, or $a=\exp(-\mathrm{softplus}(g))\in(0,1)$ for the (L)
arm — removes the divergence without touching the budget, and costs the reachability of $a=\pm1$:
two-thirds of BED-M's support (B23), reached only as $|g|\to\infty$ with gradient $1-a^2\to0$; the (L)
construction needs $a\ne1$ and the (P) construction $a\equiv1$. S-52 (the cap at $1.0$ on seeds 2, 3,
7, $\approx9.65$ s) is the same object on the three seeds and decides which mechanism.

**Hypotheses.** Adam defaults; $g$ the key-logit gate of `arm_pl` (`-C_j+s_j`, a 64-term prefix sum);
BED-M at $t^\star=2$, $n=2048$, 150 steps, lr $0.02$; the bound is on $|\Delta g|$ per step, loose by
the factor of consistent-sign gradients, and depends on $(\alpha,T)$ only through $\alpha T$; every
$\hat a_{\max}$ read against it is read in the lane it was measured in, and the two lanes are never
differenced.

**Evidence.** The CPU column and its header: `RUN results/v15_r1.txt` §3 (`device=cpu`,
`torch.get_num_threads()=8`) and `READ V15_R1.md:177-188 @ 99777ab` (the `â_max` column
$1.4107/1.2868/20.3090/49.6613/1.4390/1.5051/1.1029/285.0719$). The CUDA column:
`RUN python -c "import json;[print(r['kind'],r['seed'],r['n_params'],r['a_hat_max'],r['eval_nrmse'])
for r in map(json.loads,open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell' and
r['kind']=='arm_pl']"`, 2026-09-06 — $1.4104527235031128$, $1.2868505716323853$, $12.767516136169434$,
$49.6605224609375$, $1.4536346197128296$, $1.5051767826080322$, $1.102945327758789$,
$116.00607299804688$. S-52's lane clause `READ docs/PLAN.md:523 @ 99777ab`; B23
`READ docs/canon/CHARTER.md:230`; the three bounds ($1.32\times10^4$, $1.0021\times10^{-2}$,
$8.688\times10^{-4}$) DERIVED. The number showing training did not find it: 3 of 8 seeds "never fit
even their own training set" (`READ V15_R1.md:572 @ 99777ab`; train NRMSE $0.8398/0.8829/0.8553$ at
`:180-186`).

**Mechanism.** M-6, D-6 (the capped run priced four times and never taken), V-22, M-3, P-8, P-2 (a
regime contradicting the bound it is derived from), Ruling 2a (two lanes read as one).

**Kill — one card, one frozen kill, reconciled with the registration (M-2).** The first writing froze
a two-sided reading of Bet I — "all three cross: the cap is the fix; at most one crosses: the mechanism
is not gate divergence" — which leaves **exactly two crossings undecided**, and that band is precisely
the one the register declares: Bet I's row prints "SPLIT band $2$ of $3$"
(`READ docs/PLAN.md:926 @ 99777ab`). Worse, the card S-52 registers a **different** kill again — "*KILL*
any capped seed still above $1.0$" (`READ docs/PLAN.md:527 @ 99777ab`) — and a card may carry one
frozen kill, not two.

**The kill in force is S-52's own, as registered:** *any capped seed still above $1.0$* eval NRMSE, on
the retake journal's lane. Nothing else in this book kills 02.8. The other two numbers are kept and
labelled, not silently merged:

| reading | source | what it is | what it decides here |
|---|---|---|---|
| any capped seed $>1.0$ | S-52 *KILL* (`READ docs/PLAN.md:527 @ 99777ab`) | the card's registered kill | **the kill of this verse**; fires $\Rightarrow$ 02.8a |
| crossings of $0.7071067811865476$, out of $3$ | S-52 *measure* / Bet I | the scored column | the ledger verdict, **never** a kill |
| SPLIT band $2$ of $3$ | Bet I (`READ docs/PLAN.md:926 @ 99777ab`) | the register's band | a `SPLIT` token, scored wrong with the sign of the optimistic half |

The two readings are not the same number and their gap is printed rather than closed: a capped seed
landing in $[0.7071067811865476,\,1.0]$ **does not cross** and **does not fire the kill** — it is a
non-crossing that leaves the verse standing, and the ledger scores it in the SPLIT band. Three
crossings score `HOLDS`; two score `SPLIT`; at most one, with no seed above $1.0$, scores `COUNTER`
while the kill still does not fire; any seed above $1.0$ fires the kill whatever the crossing count.
Price $\approx9.65$ s. Lane: the retake journal's, where the three seeds read $12.768$, $49.661$,
$116.006$. Control: seeds 0 and 1 re-run uncapped in the same lane must reproduce
$0.6446726192039927$ and $0.6445174549187496$ bitwise (the identical-seed floor of 02.11). Planted
negative that would make the kill fire: the **uncapped** cells of the same three seeds, whose eval
NRMSE reads $1.1522795055459243$, $1.1133392329955414$, $1.1489267727154717$ in that lane — all three
above $1.0$, so the instrument condemns where the record already shows failure.

**If killed** (any capped seed still above $1.0$, S-52's registered *KILL*) **→ 02.8a, the constrained
modulus, killed on a different column.**
The lr regime is **not** the replacement: at $T=150$ it is the same three seeds read through a
displacement budget $23.0\times$ smaller, so a persisting NRMSE $\ge1.0$ there is Ruling 2a's
pinned-without-signal and decides nothing — and raising $T$ to restore the budget restores the
divergence bound exactly, which is the contradiction struck above. The replacement is the construction
$a=\exp(-\mathrm{softplus}(g))\in(0,1)$ on the same three seeds in the retake's lane, at the arena's
own $(\alpha,T)=(0.02,150)$ so the budget is untouched. *Hypotheses:* the construction is a manifest
field of that arm (S-01), not a schedule; $\partial a/\partial g=-a\,\sigma(g)$, so the same map that
forbids $a>1$ drives $a\to0$ with a vanishing gradient. *Evidence:* DERIVED; the uncapped $\hat a_{\min}$
column already reads $0.1886/0.0318/0.0160$ on the three seeds (`READ V15_R1.md:177-188 @ 99777ab`), so
the column exists and the kill below is not satisfied at the point of departure. *Kill of 02.8a — a
different number, on a different column, of a different object:* the arm's own $\hat a_{\min}$ reading
$<10^{-3}$ on at least 2 of the 3 seeds — the modulus has collapsed rather than diverged, the gate is
off, and the construction relocated the failure instead of removing it; the constraint is then not a
fix and Q3/W3 stays TERMINAL. Price $\approx9.65$ s, the same three seeds, and **strictly more
decisive** than the verse's own kill because it separates "the hop algebra fails" from "the gate is
off", which S-52's NRMSE read cannot. Planted negative that would make it fire: an arm initialised at
$g_0=-20$ must read $\hat a_{\min}<10^{-3}$ on 3 of 3. Planted positive that must not: the (P)
construction $a\equiv1$ must read $\hat a_{\min}=1.0$ on 3 of 3.

**Terminal.** R1 keeps "3 of 8 NO READING, mechanism unresolved"; Q3/W3 stays TERMINAL as graded;
the gate's admissible range $(0,1)$ is a registration, not a property training delivers.

---

### 02.9 — One solve versus $L$ learned hops: the parameter-count form withdrawn, the step-count form asserted, and the converse

**Statement.** **The parameter-count form of the T2 ratio is withdrawn, and the reason is arithmetic.**
Three defects, each named with its number:

*(1) Two geometries.* $p_1=2d_{\rm model}d_x+1=4{,}097$ at $d_x=128$ (adjacency $64$ + id $64$) is
BED-S's token$\to P$ map at $d_{\rm model}=16$; $4{,}769$ is the **whole** `e3_t2` arena arm at the
bed's value width $d=24$. A ratio taken across the two is a ratio of two different objects, and the
$5.0$ the first writing printed was neither: $23{,}845/4{,}097=5.8201$, and $5.0$ is $L$, obtained by
dividing $L\cdot4{,}769$ by $4{,}769$ — i.e. by silently substituting $p_1=4{,}769$ for $4{,}097$.

*(2) $L\cdot4{,}769$ over-counts.* $4{,}769$ is one whole arm, not one layer. Decomposed
(`READ scale/m3_capability.py:104-110 @ 99777ab`, the five modules of `Arm.__init__`; the arm consumes
the $d_{\rm model}=16$ feature vector directly, so there is **no** input projection and that term is
zero here):
$$\underbrace{256}_{W_Q}+\underbrace{256}_{W_K}+\underbrace{2{,}176}_{\text{MLP }16\to128}
+\underbrace{2{,}064}_{\text{MLP }128\to16}+\underbrace{17}_{\text{readout}}=4{,}769,$$
so $p_{\rm layer}=4{,}752$ (the per-layer block $W_Q,W_K,\text{MLP}$) and $p_{\rm shared}=17$ (one
readout for the stack, not one per layer), giving
$$p_L=L\cdot p_{\rm layer}+p_{\rm shared}=L\cdot4{,}752+17,\qquad p_5=23{,}777\ \ (\text{DERIVED}),$$
against the $L\cdot4{,}769=23{,}845$ printed before — an over-count of exactly $(L-1)\cdot17=68$.

*(3) No skyline count is registered, and none is measured.* The words "as registered" are **struck**:
S-35 registers depth $3/5/7$ and a price ($\approx7.6$ s per cell `[ASSUMED]`) and **no count**
(`READ docs/PLAN.md:619-625 @ 99777ab`), and §3.7's skyline row reads "depth $3/5/7$;
$\approx7.6$ s per cell [ASSUMED]", class "skyline", explicitly "at unmatched depth/width/decode
length" (`READ docs/CEQ_SHAPE.md:766 @ 99777ab`). The retake journal's own
`dist_to_skyline_why` reads "no v15/v16 scan-skyline module exists (R-SKY)": **no depth-5 arm has ever
been constructed**, no `n_params` for one has ever been READ, and $23{,}777$ is DERIVED by
multiplication and by nothing else. Until book 04 registers a depth-5 count on a constructed cell,
**this book reads no ratio of parameter counts.**

*(4) Uniform convergence does not bound a step count.* The first writing read
$T2^{\rm ep}_{\rm sky}\approx p_1/p_L$ "under a uniform-convergence form $N_\varepsilon\propto p$".
02.1 defines $N_\varepsilon$ as a **gradient-step index on a grid**, and `READ
scripts/v15_r1.py:250-256 @ 99777ab` trains full batch on one fixed $n_{\rm train}=2048$ set, so the
distinct-sample count is $2048$ at every $N_\varepsilon$ and does not vary with $p$. Uniform
convergence bounds the sample count and never the step count (V-17, a bound imported out of its
units); the `[ASSUMED]` tag did not license the transfer and the form is **struck**.

**What is asserted instead — an optimisation statement, on the field 04.15 registers.** In 02.1's
orientation and no other (shape over control, licensed direction $<1$; the symbol $R_{T2}$ appears
nowhere because it named the reciprocal),
$$T2^{\rm ep}_{\rm sky}=\frac{\operatorname{median}_\sigma N_\varepsilon(\text{shape},\sigma)}
{\operatorname{median}_\sigma N_\varepsilon(\text{sky}_5,\sigma)},\qquad
N_\varepsilon(A,\sigma)=\min\{t\in G:\ D_t(A,\sigma)\le\varepsilon\},$$
both arms under Adam at the arena's fixed $\alpha=0.02$, both read off the registered per-step
`eval_trace` field at the sixteen points of $G$ (`04_BEDS_AND_INSTRUMENTS.md:215`, 04.15). It is a
count of optimisation steps to a target, which is what the instrument writes; it carries no parameter
count and no sample-complexity assumption.

**Who owns the counter (D-CALIB).** The B25 T2 counter is owned by **`06_PREDICTIONS.md` 06.31** and is
cited here by verse id, not restated: prediction $T2^{\rm ep}_{\rm sky}\le0.20$ (equivalently $k\le30$
at $n_{\rm train}=2048$), counter $T2^{\rm ep}_{\rm sky}\ge1.0$ or undefined, SPLIT $(0.20,1.0)$, and
under D-CALIB **the single point estimate for B25 is 06.31's counter**. This book files **no second
counter** for the row. The apparent contradiction of the first writing was two errors compounding: the
retired $R_{T2}$ named the reciprocal (02.1, F4), and 06.31's clause "$T2_{\rm sky}<1.0$" is the
condition under which its **counter dies**, not the counter itself (`READ
docs/canon/06_PREDICTIONS.md:445 @ 99777ab`, the Kill line). Both books now read one orientation and
one point estimate.

**What this book contributes that 06.31 does not: the mechanism of the counter, and the tied bound.**
With weight tying (the looped block $z^{(t+1)}=V+\gamma\hat Pz^{(t)}$) the stack shares $(W_Q,W_K)$ and
$\gamma$ across layers, so $p_L=p_1$ **exactly** at every $L$ and no parameter argument separates the
two arms at all: the tied stack is the Neumann iterate of the same operator (Proposition 4, §3.7).

**The tied count, derived entry by entry, and the $-1$ struck.** The loop introduces no module and
drops none. Every parameter of the depth-1 arm is read at every iterate: the per-layer block
$p_{\rm layer}=W_Q\,256+W_K\,256+\text{MLP }(2{,}176+2{,}064)=4{,}752$ is applied once to produce
$\hat P$ and reused unchanged at each of the $L$ passes; the readout $p_{\rm shared}=17$ is applied once
after the last pass; and the dial $\gamma$ — the single scalar that is the $+1$ of
$p_1=2d_{\rm model}d_x+1=4{,}097$ at $d_x=128$ — appears **inside the recursion itself**,
$z^{(t+1)}=V+\gamma\hat Pz^{(t)}$, so the loop cannot drop it without deleting its own update rule.
Hence, at every $L\ge1$,
$$p_L\ =\ p_{\rm layer}+p_{\rm shared}\ =\ 4{,}769\ =\ p_1\quad\text{(the arena arm)},\qquad
p_L\ =\ 2d_{\rm model}d_x+1\ =\ 4{,}097\ =\ p_1\quad\text{(BED-S's token}\to P\text{ map)},$$
each read at its own geometry and the two never mixed (defect (1) above). The first writing's
$p_L=p_1-1$ is **struck**: it removed one parameter and named none, and the only scalar it could have
named is $\gamma$, which the tied loop carries. Its companion figure is struck with it — $4{,}096/4{,}097
=0.99976$ printed as $1.0$ is a ratio rounded up to the value it was supposed to establish, and it was
printed under the symbol $R_{T2}$, which 02.1 retires because it named the reciprocal of this book's
orientation. **No verse of this book prints a ratio under that symbol**, and the tied contrast is read
as the terminal accuracy gap $\Delta_5$ of 02.9a, not as a parameter ratio: $p_L=p_1$ means the
parameter axis separates nothing, which is exactly why 02.9a is the one contrast in this book to which
Ruling 3's word "matched" applies without withholding (residual $0.000\,\%$).
The counter's mechanism is that each untied stack layer has a dense one-hop gradient from step $0$
while the solve's hop-$\ge2$ signal is conditional on $\hat P$ (02.4c). **Converse** (what the solve
cannot learn that depth can): (i) non-stationary hops
$P_1P_2\cdots P_L$ — the solve has one $P$; the record's instance is the delay bed, where no
first-order recurrence delays (`first_order_cannot_delay`) and the best fit reads $R^2=-0.000166$;
(ii) non-linear composition $f\circ g$ — the read is linear in $V$ (Peng's class, vacuous at $s=64$ but
a converse in kind).

**Hypotheses.** Skyline untied at unmatched parameters (R-SKY, §3.7); $\varepsilon$ of 02.1; both arms
at the arena's $\alpha=0.02$ and $150$ steps, so the ratio is a ratio of step counts at one optimiser
setting and at no other; the tied bound requires the stack's layers to share $(W_Q,W_K)$ and $\gamma$.
No parameter-count hypothesis is asserted: the parameter form is withdrawn above, and no clause of
this verse conditions on $p_1$, $p_L$ or a uniform-convergence bound.

**Evidence.** The decomposition $256+256+2{,}176+2{,}064+17=4{,}769$: DERIVED from
`READ scale/m3_capability.py:104-110 @ 99777ab` and checked against the measured count,
`RUN python -c "import json;[print(r['kind'],r['seed'],r['n_params']) for r in
map(json.loads,open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell']"`, 2026-09-06 — `softmax`
$4{,}769$ on 8 of 8 seeds. The tied count $p_L=p_1$ at every $L$: DERIVED from that decomposition
together with the recursion $z^{(t+1)}=V+\gamma\hat Pz^{(t)}$, in which $\gamma$ — the $+1$ of
$p_1=2d_{\rm model}d_x+1$ — is the update's own coefficient and cannot be dropped; the loop therefore
drops no parameter and the struck $p_L=p_1-1$ names none. Skyline registration, price and the absent count
`READ docs/CEQ_SHAPE.md:751-770 @ 99777ab` (§3.7, "at unmatched depth/width/decode length"),
`READ docs/PLAN.md:619-625 @ 99777ab` (S-35, depth $3/5/7$ and $\approx7.6$ s per cell `[ASSUMED]`,
no count); the depth-5 arm itself `NOT MEASURED — needs a constructed depth-5 cell with its n_params
READ; the retake journal's dist_to_skyline_why field reads "no v15/v16 scan-skyline module exists
(R-SKY)"`. Tying: CITED [V]
`yang-2024-looped`, `wang-2024-incontext-td`, `xie-2026-softmax-rl` (abstract level, [U] for theorem
numbers — inadmissible as load-bearing, so the tied ratio is DERIVED from Proposition 4, not cited).
Delay theorem `READ lean/CEQ/V15Kernel.lean:14-19 @ 99777ab`; $R^2=-0.000166$ `READ
MISTAKES.md:1723-1782 @ 99777ab`. Learnability under a curriculum: CITED [V]
`wang-2025-easytohard` (abstract; not load-bearing). The record's one T2-shaped reading: the signed
arm at $3{,}319{,}296$ matched parameters, 800 steps on TinyStories, val loss $2.0064$ against softmax
$1.4282$ ($1.405\times$) — behind at equal steps, on the dead programme (`READ TRAINING.md:414-416 @
99777ab`).

**Mechanism.** R-SKY, D-1, P-10, V-25 (Peng vacuous: $384<544$), D-7, L-EQ, P-2 (a ratio false at its
own printed counts), P-1 (a count over-multiplied and a registration that does not exist), V-17 (a
uniform-convergence bound imported out of its units), D-CALIB (two books filing opposite point
estimates for one census row), P-4 (a $-1$ inside a parameter count with no derivation behind it, and
a ratio $0.99976$ printed as the $1.0$ it was meant to establish).

**Kill.** Frozen, in 02.1's orientation: $T2^{\rm ep}_{\rm sky}\ge1$ at
$\varepsilon=\text{floor}+\max(\mathrm{MDE}_8,4.7\times10^{-3})$ on the paired per-seed ratio,
$N=8$; instrument 02.1's `eval_trace` read on the arena plus the depth-5 skyline cells
($\approx64.8$ s `[ASSUMED]`). Planted control: shape at $\gamma=0$ against depth-1 softmax must read
$T2^{\rm ep}_{d1}=1.00$ exactly (bitwise parity, Proposition 1), $0$ GPU-s when both are in the arena.

**If killed → 02.9a, the tied loop, on a different number and a different object.** The first writing's
replacement was $t_\gamma$, "the first step at which $\Lambda>2.7055$ on the held-out set", read from
`NOT MEASURED — needs the per-step $\Lambda$ trace (book 04)`. No verse of
`04_BEDS_AND_INSTRUMENTS.md` registers a per-step $\Lambda$ trace; S-62 writes $\Lambda$ once per
cell, not per step (`READ docs/PLAN.md:627 @ 99777ab`), so that replacement was a kill on an instrument
no book carries (P-4) and is **struck**, not softened.

The replacement in force reads a column S-62 already writes, on a different arm. Add the **tied loop**
$z^{(t+1)}=V+\gamma\hat Pz^{(t)}$ at $L=5$, sharing $(W_Q,W_K)$ and $\gamma$ with the shape, so the two
arms have **identical parameter sets by construction** — residual $0.000\,\%$, the one contrast in this
book to which Ruling 3's word "matched" applies without withholding. The statement is then not a step
ratio at all but a **terminal accuracy gap**:
$$\Delta_5:=D_{150}(\text{shape})-D_{150}(\text{loop}_5)\ \ \text{per seed, paired}.$$
*Hypotheses:* the loop arm is one added arm on S-62 with the shape's own $(W_Q,W_K,\gamma)$; the
comparison is at $\varepsilon$-free terminal distance, so `eval_trace` is not needed and the
terminal `eval_nrmse` column suffices. *Evidence:* Proposition 4 and §3.7 for the Neumann identity;
price $8\times1.680\approx13.4$ s for the added arm, DERIVED from S-62's per-cell price. *Kill of
02.9a:* $|\Delta_5|$ within $\mathrm{MDE}_8$ at the realised paired sd on at least 6 of 8 — the exact
solve and its own five-term iterate are the same object at the arena's $\gamma$, and no
sample-efficiency sentence survives on either arm. Decided from that journal, **strictly cheaper** than
the verse's own kill ($\approx13.4$ s against the arena plus $\approx64.8$ s of skyline cells) and
**strictly more decisive** because it decides the operator rather than the schedule.

**Terminal.** The shape's T2 sentence is withdrawn on the bed; the parameter-count form of it is
withdrawn everywhere, at every geometry, until a constructed depth-5 cell prints its own `n_params`;
what remains is the per-step cost statement (book 03) and whatever A1/A2 book 01 licenses.

---

### 02.10 — The AdamW regime: what the record fixes, what it leaves NOT MEASURED, and the decay on the dial

**Statement.** Arena regime (every arm identical): Adam — not AdamW — at lr $0.02$, full batch
$n_{\rm train}=2048$, 150 steps, no gradient clipping, no warmup, no decay, targets standardised by
the training mean and sd. LM regime: AdamW at lr $3\times10^{-4}$, `clip_grad_norm_` $1.0$, batch 32,
no scheduler, `weight_decay` at torch's default $0.01$ on **every** parameter including $\beta$ (init
$1$) and any dial.

**The decay term is multiplicative, so it is identically zero at a zero initialisation.** Decoupled
decay multiplies a zero-gradient parameter by $(1-\alpha\lambda)^T$, and a parameter initialised at $0$
is multiplied to $0$: at 02.4's registered $\theta_0=0$ the dial's decay term is
$0\cdot(1-\alpha\lambda)^T=0$ exactly, $\theta_\gamma$ **never moves**, $\hat\gamma$ stays at its
initialisation $\tfrac12$, and **Ruling 10′ reads that as PINNED, not MOVED**. The first writing read
the same arithmetic as "a drift of $\theta_\gamma$ toward $0$ ... which Ruling 10′ would read as
MOVED-without-signal", which is the opposite of what the arithmetic gives; it is **struck** (V-9: a
consequence vacuous at the initialisation the book itself prescribes). The consequence is therefore
restricted to parameters with a **non-zero** initialisation:

| parameter | init | $\alpha$ | $(1-\alpha\lambda)^T$ | drift with no data signal |
|---|---|---|---|---|
| $\beta$ (LM lane) | $1$ | $3\times10^{-4}$ | $0.99955$ at $T=150$ | $4.5\times10^{-4}$ |
| $\beta$ (LM lane) | $1$ | $3\times10^{-4}$ | $0.94176$ at $T=20{,}000$ | $0.05824$, a $5.82\,\%$ drift toward $0$ |
| $\beta$ (arena lane) | $1$ | $0.02$ | $0.970443$ at $T=150$ | $0.029557$, $12.6\,\delta_\beta$ |
| $\theta_\gamma$ at $\theta_0=0$ | $0$ | any | any | $0$ **exactly**, at every $T$ |
| $\theta_\gamma$ at $\theta_0=\operatorname{logit}\gamma_{\rm env}$ | $2.1972$ at $\gamma_{\rm env}=0.9$ | $0.02$ | $0.970443$ at $T=150$ | $\theta\to2.13229$, $\hat\gamma:0.9\to0.894001$, i.e. $5.999\times10^{-3}=2.56\,\delta_\beta$ |

The dial acquires a decay consequence **only** once its initialisation is non-zero, which 02.7's frozen
registration $|\gamma_{\rm env}-\tfrac12|\ge0.2$ makes the case whenever the dial is initialised at the
bed's horizon rather than at $\theta_0=0$. Prescriptions, derived: `weight_decay=0` on the
parameter group $\{\beta,\gamma,\text{gates}\}$; the shape's lr inherited from the control's budget
("identical budget for every arm"); warmup `NOT MEASURED`. Through the solve the $\hat P$-gradient
scales with $\|M\|_\infty=1/(1-\gamma)$; Adam's per-coordinate normalisation removes the scale, not the
conditioning, and the mirror kill $\hat\gamma>0.99$ prints $1/(1-\hat\gamma)>100$ beside every $\delta$.

**Hypotheses.** Torch defaults on the certified stack equal the installed stack's `[ASSUMED]`; the
decay arithmetic is the zero-gradient limit; the LM chunk length is TRAINING.md's $20{,}000$.

**Evidence.** `READ scripts/v15_r1.py:235, :250-256 @ 99777ab`; `scale/m3_capability.py:88,178`;
`ceq/lm.py:255-272`; `ceq/harness.py:328-352` ("Identical budget for every arm: same steps, lr, batch,
seed"); `ceq/hf/train.py:304,325,378,402` ("There is no scheduler"); chunk `steps=20000` `READ
TRAINING.md:331 @ 99777ab`. Default `weight_decay=0.01`: `RUN python -c "import torch,inspect;
print(inspect.signature(torch.optim.AdamW.__init__))"` on `torch 2.14.0+cpu`. Decay arithmetic
DERIVED. Ruling-1 LM floor: `READ COSTS.md:283-287 @ 99777ab` ("NOT YET MEASURED"). The number, cited
on the field the journal actually carries: the per-cell time field is **`secs`**, read on the rows whose
row-type field is `t="cell"` —

`RUN python -c "import json,statistics;c=[r for r in map(json.loads,
open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell'];m=lambda k:statistics.mean(
[r['secs'] for r in c if r['kind']==k]);print(m('arm_smprime'),m('softmax'),m('arm_smprime')/m('softmax'))"`,
2026-09-06 — mean over the eight seeds $16.161$ s (`arm_smprime`) against $1.681$ s (`softmax`) at
$150$ steps, ratio $9.614$; an unsynchronised host clock, per-op floor class (C17). The first writing's
pointer `RUN results/v17k_r4_retake.jsonl` (`t=wall`) is **struck**: no cell row carries a field named
`wall`, `t` is the row-type field whose value on a cell row is `cell`, and the one `t="wall"` row of the
journal is a whole-run summary carrying `secs_per_run_by_arm`, not a per-cell time — a number pointed
at a field that does not exist is a number with no provenance (P-2). The quoted figures are unchanged
by the correction because they check out against `secs`.

**Mechanism.** V-22 (a constant carried in from torch), M-8, C17, Ruling 2a, V-9, P-2 (a RUN pointer
naming a journal field that does not exist).

**Kill, re-based on the arena, where the pair is decidable.** The first writing's control was "at
$T=20{,}000$ the same pair must differ by at least $0.05\,|\hat\beta|$ — `NOT MEASURED — needs the LM
chunk behind the author's yes`", and its If-killed then hung on the same chunk. Both links were
unreachable on the bed as registered: the record's LM cell reads "NOT YET MEASURED"
(`READ COSTS.md:283-287 @ 99777ab`) and Bet M is registered "`NOT MEASURED`, behind the author's yes"
(`READ docs/PLAN.md:928 @ 99777ab`). That chain is **struck** (V-15: a condemning rule whose planted
negative cannot be run), and the whole chain is re-based on the arena, where the arena's own
$\alpha=0.02$ makes the drift $12.6\times$ larger than the LM's.

Frozen: the identical-seed pair with `weight_decay` $0.01$ against $0$ on the group
$\{\beta,\gamma,\text{gates}\}$ reads $|\hat\beta-\hat\beta'|\le\delta_\beta=2.345\times10^{-3}$ on 8 of
8 **at $T=150$ on the arena** — the decay prescription changes nothing where it is prescribed and is a
note, not a repair (V-9). Price $2\times8\times1.680\approx27$ s. Control, now on a run that exists and
needs no authorisation: the arithmetic's own prediction at the arena is
$1-(1-0.02\cdot0.01)^{150}=0.029557$ of $|\hat\beta|$, which is $12.6\,\delta_\beta$, so the pair
**must** differ by at least $0.0296\,|\hat\beta|$ on a zero-gradient $\beta$ — the instrument sees the
decay where the decay is. The $T=20{,}000$ figure ($0.94176$, a $5.82\,\%$ drift at the LM's
$\alpha=3\times10^{-4}$) is retained above as a **note with no kill attached**:
`NOT MEASURED — needs the LM chunk behind the author's yes`, and no kill, replacement or row of this
book conditions on it.

**If killed → a different column, a different parameter, the same journal.** If the arena pair reads
within $\delta_\beta$ despite the arithmetic predicting $0.029557$, the decay is not reaching $\beta$ —
$\beta$ carries a data gradient that cancels it — and the $\hat\beta$ column cannot decide the
prescription at all. The replacement reads the **dial** instead, at the non-zero initialisation 02.7's
registration forces: with $\theta_0=\operatorname{logit}\gamma_{\rm env}=2.1972$ at
$\gamma_{\rm env}=0.9$, decoupled decay alone carries $\theta$ to $2.13229$ and $\hat\gamma$ from
$0.9$ to $0.894001$, a drift of $5.999\times10^{-3}$ — $2.56\,\delta_\beta$, and **above the resolution
Ruling 10′ reads MOVED at**. *Hypotheses:* the dial is initialised at the bed's horizon, not at
$\theta_0=0$ (at $\theta_0=0$ the decay term is identically zero and this replacement is vacuous by
construction, which is stated rather than hidden). *Evidence:* DERIVED, the table above; the
$\hat\gamma$ column is in S-62's `*measure*` list (`READ docs/PLAN.md:627 @ 99777ab`). *Kill of the
replacement:* $|\hat\gamma-\hat\gamma'|\le\delta_\beta$ on 8 of 8 between the decay-on and decay-off
cells at $T=150$ — then decoupled decay reaches neither parameter at the arena, the prescription is
deleted from the arena manifest, and it survives only as an LM-chunk note that nothing in this book
decides. Same $\approx27$ s journal, and **strictly more decisive** than the verse's own kill because
it decides the verdict instrument — whether an $\hat\gamma$ MOVED reading can be manufactured by the
optimiser with no data signal — and not merely a parameter's value.

**Terminal.** The optimiser regime is a manifest field journalled per cell (S-01), never a claim; no
number about "faster" is licensed from it.

---

### 02.11 — The training-noise floor, and how T1/T2 are read against it

**Statement.** Ruling 1's floor is two identical-seed chunks read as $|\Delta|$ final loss. At the
arena it is measured: $\delta_{\rm nrmse}=0.0$ on 6 of 6 identical-seed cells (`arm_pl`, `arm_smprime`,
`softmax`; seeds 0, 1; CUDA; `warn_only=True`; the same `instrument_hash`) between
`results/v17k_r4_floor.jsonl` and `results/v17k_r4_retake.jsonl`. A zero floor is not a tolerance: the
resolution for reading $N_\varepsilon$ is the thread-count floor $2.345\times10^{-3}$ (M-10) and the
seed sd — **read in the lane the number is used in** (Ruling 2a): the v17k_r4 lane's own softmax seed
sd over its eight cells is $\mathbf{0.011452}$ at $N=8$, and the R1 figures $0.011824$ (softmax) and
$0.253673$ (`arm_pl`) are `results/v15_r1.txt` §4 readings on `device=cpu` at a different value width,
kept here as a labelled cross-lane note that no threshold of this verse reads. So $\varepsilon$ in 02.1 is
$\text{floor}+\mathrm{MDE}_N$, never $\text{floor}+\delta$. The shape's own floor: `NOT MEASURED` (no
shape cell; its backward runs through `solve_triangular`, bitwise over 8 repeats on one box with no
documented guarantee, N-01). The LM chunk floor: `NOT YET MEASURED` (COSTS §4). T1 is read against
synchronised timers and interleaved order (book 03); T2 against $\varepsilon$.

**Hypotheses.** One lane (threads 8, CUDA, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, torch `2.5.1+cu121`),
`instrument_hash` `5d41a63d…`, steps $=150$; the pair is identical-seed, not seed-to-seed (Ruling 9's
distinction).

**Evidence.** The literal command, as every other RUN in this book prints it — the first writing gave
an angle-bracketed description of a command instead of a command, which is a number with no
reproducible provenance (P-1) and is struck:

`RUN python -c "import json;f={(r['kind'],r['seed']):r['eval_nrmse'] for r in
map(json.loads,open('results/v17k_r4_floor.jsonl')) if r.get('t')=='cell'};g={(r['kind'],r['seed']):
r['eval_nrmse'] for r in map(json.loads,open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell'};
[print(k,abs(f[k]-g[k])) for k in sorted(f) if k in g and k[1] in (0,1)]"`, 2026-09-06, on
`torch 2.14.0+cpu` reading journals written on `2.5.1+cu121`. It prints `('arm_pl', 0) 0.0`,
`('arm_pl', 1) 0.0`, `('arm_smprime', 0) 0.0`, `('arm_smprime', 1) 0.0`, `('softmax', 0) 0.0`,
`('softmax', 1) 0.0`: the six
differences are $0.0$ exactly (`arm_pl` $0.6446726192039927$, $0.6445174549187496$; `arm_smprime`
$0.9260818361710341$, $0.8812466551692475$; `softmax` $0.9734002295313872$, $0.9387956332491751$).
Ruling 1 `READ V17K_RULINGS.md:39-45 @ 99777ab`; Ruling 9 `:354-379`; COSTS §4 `READ
COSTS.md:283-287 @ 99777ab`; M-10 `READ MISTAKES.md:1072-1108 @ 99777ab`; N-01 `READ
docs/PLAN.md:529-534 @ 99777ab`. The number on "faster": none — the floor journal's wall record puts
`arm_smprime` at $16.31$ s per run against `softmax` $1.638$ s, per-op floor class (C17).

**Mechanism.** M-10, M-16, V-16 (a $0.0$ floor must be shown to be a measurement), Ruling 1, Ruling 9,
Ruling 2a and V-22 (a control whose difference and whose threshold came from two different lanes).

**Kill.** Frozen: any identical-seed pair on the shape reading $|\Delta|>2.345\times10^{-3}$ — the
shape's training is not bitwise on this lane and Ruling 9's measured-floor clause applies with that
number. Price $2\times1.680$ s. Control (V-16), **read entirely inside one lane**: a pair with deliberately
different seeds must read $|\Delta|\ge$ that lane's own seed sd — `softmax` seed 0 against seed 1 on
`results/v17k_r4_floor.jsonl` (`device: cuda`, `d: 24`, task `e3_t2`),
$|0.9734002295-0.9387956332|=0.034605$, against the **v17k_r4 lane's own** softmax seed sd

`RUN python -c "import json,statistics;c=[r for r in map(json.loads,
open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell'];print(statistics.stdev(
[r['eval_nrmse'] for r in c if r['kind']=='softmax']))"`, 2026-09-06 — $0.011452$ at $N=8$,

so $0.034605>0.011452$ and the instrument sees a difference when there is one. The first writing
compared that CUDA difference against $0.011824$, the seed sd of `RUN results/v15_r1.txt` §4 on
`device=cpu`: a control whose difference and whose threshold came from two different beds, devices and
value widths. It is **struck** under Ruling 2a — no sentence transfers across lanes without a bind at
that lane — and the constant no longer crosses a device boundary. The control fires either way; what
changes is that it now fires on one lane's arithmetic.

**If killed.** The floor is that number; $\varepsilon:=\text{floor}+\max(\mathrm{MDE}_N,\delta)$ and every
T2 reading carries it. *Hypotheses:* $\delta$ measured once per lane. *Evidence:* the pair.
*Kill:* $\delta\ge\mathrm{MDE}_8$ — $0$ GPU-s arithmetic — and T2 is unreadable at $N=8$.

**Terminal.** No T2 number at that $N$; $N$ is repriced from the realised sd (A.11 item 2), and until
then the sentence licensed is "the training noise floor on this lane is $\delta$".

---

### 02.12 — Terminal: $\gamma$ is a bed-side dial and the shape is softmax wearing a name on this bed

**Statement.** When 02.4, 02.4a, 02.4b and 02.7 are dead on a bed, the sentence licensed is: "$\gamma$
is a bed-side dial; on this bed the trained shape is softmax wearing a name." What survives:
Proposition 1 (bitwise at $\gamma=0$); the committor head at $\gamma=1$ on $\hat Q$ (it never carried
$\gamma$, 02.6); regime N at $\gamma=1$ — corner 3, exact without a dial, admitting no absorbing row
(Definition 2: at $\gamma=1$ with an absorbing row $\det(I-A')=0.0$), so BED-M is contained, not won;
the certificate discipline; the identities. What is withdrawn: every "faster to train" and "more
accurate" sentence on that bed; Bet C; S-64 and S-65 (void, ChaCAL at $\hat\gamma=0$ is softmax);
S-73 before it is made.

**Hypotheses.** The kills fired under the seed rule (at least 6 of 8) at $N=8$ with realised sd at most
$2.18\times$ the pilot; the bed was admitted (S-12).

**Evidence.** `READ docs/PLAN.md:1069 @ 99777ab` (the R-21 PINNED row of §5.10); Definition 2 and
$\det(I-A')=0.0$ `READ docs/CEQ_SHAPE.md:286-293 @ 99777ab`; the tree-B sentence `READ
docs/PLAN.md:1052-1058 @ 99777ab`. The two candidate beds of the Kill and their seed rules:
`READ docs/canon/04_BEDS_AND_INSTRUMENTS.md:92-106 @ 99777ab` (04.6, BED-J),
`:107-121` (04.7, the chess witness), `:145-152` (04.9 clauses 1, 4 and 6),
`:236-247` (04.15, the `gamma` block as a declared field and the refusal rule),
`:310-315` (04.21 step (5), where $\sigma_d$ is filled), and
`READ docs/canon/09_CHESS_AND_MARKETS.md:118-128 @ 99777ab` (09.11, `REQUIRED_SEEDS = 8`, interleaving,
K-P). The MOVED and PINNED thresholds $\ln4096=8.318$ and $2.7055$:
`READ docs/canon/04_BEDS_AND_INSTRUMENTS.md:314 @ 99777ab`, `READ docs/PLAN.md:327-334 @ 99777ab`
(S-61); the boundary-null citation is owed (Limits). Neither bed has a cell:
`NOT MEASURED — needs the first BED-J cell (04.21) or the first BED-C-N cell (09.11)`.

**Mechanism.** P-3 (a stale claim never retracted), V-9, D-7, V-11 and P-4 (a kill on an unregistered
bed, priced as if it were decidable), M-2 (the MOVED threshold frozen here at $\Lambda>8.318$ rather
than left as the word "MOVED").

**Kill, with its beds named, because "a later bed" is not a registration.** The first writing froze
"a later bed on which R-21 reads MOVED on at least 6 of 8 under the boundary null" and named no bed, no
draw, no seed rule and no $N$ — which CHARTER §2's reachability rule refuses: "Every kill names the
draw, seed rule and $N$ at which it is decidable, and the realised-sd clause that makes its
$\mathrm{MDE}_N$ honest" (`READ docs/canon/CHARTER.md:146-148 @ 99777ab`). An unnamed bed is a kill on
an object that does not exist (V-11, P-4), and it is **struck**.

Frozen instead on **exactly two registered candidate beds, and no third**: R-21 reads **MOVED** —
$\Lambda>\ln4096=8.318$, the verdict's own MOVED threshold, against PINNED at $\Lambda\le2.7055$ under
the boundary null $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ (`READ docs/canon/04_BEDS_AND_INSTRUMENTS.md:314
@ 99777ab`; the band $(2.7055,\,8.318]$ is **neither** verdict and does **not** fire this kill) — on at
least 6 of 8 seeds on one of:

| candidate bed | registered at | seed rule | $N$ | realised-sd clause |
|---|---|---|---|---|
| **BED-J**, the joint-consistency bed (jittered `bed_1`, $K=2$, $\vert T\vert=9$) | `04_BEDS_AND_INSTRUMENTS.md` **04.6**, protocol **04.9**, order **04.21** | 04.9 clause 1: $N\ge8$ distinct deduplicated seeds, one thread lane read from the journal, byte-identical draws per seed across arms (train batch at `seed`, eval batch at `seed+12345`, init from the global generator after `torch.manual_seed(seed)`, `RNG_PLAN`), the pairing asserted as a relation and never as two marginal counts | $8$ minimum, repriced by 04.9 clause 6 before any bet is scored | $\mathrm{MDE}_N$ at the realised $\sigma_d$ of 04.9 clause 4, `NOT MEASURED — needs $\sigma_d$ from 04.21 step (5)`; $\sigma_d\ge0.109199$ (the M-3 precedent's own realised value) reprices $N$ and nothing at $N=8$ is falsifiable |
| **BED-C-N**, the chess witness on natural ground | `04_BEDS_AND_INSTRUMENTS.md` **04.7**, protocol `09_CHESS_AND_MARKETS.md` **09.11**, oracle 09.5–09.6 | 09.11: seeds $0$–$7$ deduplicated (`REQUIRED_SEEDS = 8`), byte-identical draws per seed across arms, arms interleaved, run order randomised and journalled, every timed region bracketed by two `torch.cuda.synchronize()` calls | $8$; $N=16$ at $\approx68$ s per pair when K-P fires | K-P: realised paired sd $\ge2.18\times$ the pilot's $0.050146$ $\Rightarrow$ nothing at $N=8$ is falsifiable and $N$ is repriced from the realised sd |

**The R-21 row exists on any cell of either bed by construction, so the kill is decidable and not
merely wished for**: 04.15's identity manifest declares `gamma` (init, final $\hat\gamma$, $\Lambda$,
verdict) a required field of **every** journalled cell and calls a missing declared field a refusal
(`READ docs/canon/04_BEDS_AND_INSTRUMENTS.md:238 @ 99777ab`). Neither bed's other gates block this
reading: BED-J's contrasts wait on book 01's D-APPROX bound for creditability and BED-C-N carries no
capability number until BED-C-D's digests are pinned (04.7's Kill) — but $\Lambda$ is a **parameter
verdict, not a contrast and not a capability number**, so it is readable from the cell under both
gates. What does not exist is the cell: `NOT MEASURED — needs the first BED-J cell (04.6, 04.21) or the
first BED-C-N cell (09.11)`; the kill is registered and undecided, never unreachable. Price: a journal
read, $0$ GPU-s **beyond that bed's own price**, which is that bed's book's and is not counted here.
Planted negative that would make it fire: S-61's registered plants, carried into whichever bed runs
first — a generator at $\gamma_{\rm env}=0$ must read PINNED on 8 of 8 and one at
$\gamma_{\rm env}=0.6$ MOVED on 8 of 8 (02.4). Planted positive that must **not** fire: a bed
registered at $\gamma_{\rm env}=\tfrac12$, which reads PINNED on 8 of 8 by 02.7(B) with the dial
working perfectly — which is why both beds inherit 02.7's frozen registration
$|\gamma_{\rm env}-\tfrac12|\ge0.2$ and a cell at $\gamma_{\rm env}\in(0.3,0.7)$ is VOID for this kill.

**If killed → 02.13, not 02.4.** The first writing sent this link back to 02.4 — "02.4 is re-filed for
that bed ... *Kill:* its R-21" — while 02.4's own chain terminates in 02.12. That is a **cycle**, and a
cycle has no terminal link: CHARTER §2's depth rule requires every chain to end in a Terminal within at
most three replacements, and each link's kill to be strictly cheaper or strictly more decisive than the
one before it, which a re-entry with the same R-21 at the same $0$ GPU-s cannot be (L-VERSE). The
re-entry is **struck**; the link is the new verse below, whose kill is a different number on a
different object.

**Terminal.** Every sentence about $\gamma$ in the canon is bed-scoped; no bed-free sentence about the
dial is ever licensed.

---

### 02.13 — The later bed's dial, killed by the solve and not by the dial

**Statement.** On a bed where 02.12's kill has fired — R-21 reads MOVED ($\Lambda>\ln4096=8.318$) on at
least 6 of 8 under the boundary null, on one of the two beds 02.12 names, **BED-J**
(`04_BEDS_AND_INSTRUMENTS.md` 04.6, protocol 04.9, order 04.21) or **BED-C-N** (04.7, protocol
`09_CHESS_AND_MARKETS.md` 09.11), under that bed's own seed rule and $N$ as printed in 02.12's table —
the dial's sentence is re-licensed **for that bed only**, and the question moves one
level down: a moving dial is a moving parameter, not a working operator. The statement asserted is
therefore about the solve, not about $\hat\gamma$: on that bed the exact resolvent read carries signal
the identity read does not, measured as the **ablation gap**
$$A:=D_{150}\big[(I-\hat\gamma\hat P)^{-1}\to I\big]-D_{150}\big[(I-\hat\gamma\hat P)^{-1}\big]$$
per seed, paired — the distance-to-floor lost when the solve is replaced by the identity in the same
trained cell, with every other parameter held.

**Hypotheses.** That bed's census admitted it (S-12); its own realised paired sd and $\mathrm{MDE}_N$
are printed in the row; the ablation is applied at evaluation to the trained cell, so no second
training run enters; $\gamma_{\rm env}$ registered with $|\gamma_{\rm env}-\tfrac12|\ge0.2$ (02.7).

**Evidence.** The ablation instrument and its price, $\le17.4$ s: 02.4's Kill field. That bed's
$\hat\gamma$, $\Lambda$ and terminal `eval_nrmse` columns are S-62's registered `*measure*` list
(`READ docs/PLAN.md:627 @ 99777ab`) and, on either named bed, 04.15's declared `gamma` block
(`READ docs/canon/04_BEDS_AND_INSTRUMENTS.md:238 @ 99777ab`). The realised seed sd the kill is read
against is that bed's own: 04.9 clause 4's $\sigma_d$ on BED-J, K-P's on BED-C-N (02.12's table), and
never this book's arena placeholder. The value of $A$: `NOT MEASURED — needs the later bed's own cell;
no cell on either named bed exists, and no bed on which R-21 has read MOVED exists in the record`.

**Mechanism.** L-VERSE (the cycle this verse replaces), V-9, D-7, P-3.

**Kill.** Frozen, one number, **strictly more decisive than R-21**: $A\le$ one realised seed sd of that
bed on at least 6 of 8 seeds. R-21 decides a parameter — whether $\hat\gamma$ left its initialisation;
this decides the operator — whether the solve carries anything the identity does not. A dial that moves
while $A$ reads zero is a parameter wandering inside an operator that does no work, and the dial's
re-licensed sentence dies with it on that bed. Instrument: the $(I-\hat\gamma\hat P)^{-1}\to I$
ablation on the trained cells, $\le17.4$ s, no retraining. Planted negative that would make it fire:
the depth-1 softmax control, whose $\hat P$ enters no solve, must read $A=0.0$ on 8 of 8. Planted
positive that must not: a cell with $\hat\gamma$ pinned at the bed's $\gamma_{\rm env}=0.9$ by
construction must read $A$ above one seed sd on 8 of 8, since $1/(1-0.9)=10$ and the identity discards
the whole resolvent.

**If killed.** **No replacement exists, and this book says so rather than inventing one.** Every
cheaper instrument in this book — R-21's $\Lambda$, the $\hat\gamma$ column, the $\hat\beta$ column,
S-62's `*measure*` line — reads a parameter, and a parameter reading cannot separate "the operator does
nothing" from "the parameter moved". The ablation is the cheapest instrument that decides the operator,
and nothing below it decides anything this verse needs. The Terminal below is in force from the moment
this kill fires.

**Terminal.** On that bed the solve is decoration: the shape is a depth-1 softmax layer carrying a
parameter that moves and changes nothing, every "more accurate" and "faster to train" sentence is
withdrawn there as it is on BED-S (02.12), and what the canon keeps is what 02.12 keeps — parity at
$\gamma=0$, the committor head at $\gamma=1$, regime N, the certificate discipline and the identities.
No further bed re-licenses the dial without its own ablation number printed in the same row.

---

## Census rows closed

| row | what is broken | verse(s) |
|---|---|---|
| B12 | the corner descent: $\beta$ and $\hat\gamma$ at the softmax corner; R1's 3 of 8 diverging seeds | 02.2 (the census with numbers), 02.3 ($\beta\equiv1$ bind), 02.4/02.4a/02.4b (the dial's landscape, parametrisation, anneal, fixed $\gamma_{\rm env}$), 02.5 (identification floor), 02.6 (committor supervision), 02.7/02.7a (interventional pairs at the registered initialisation, the clamp magnitude), 02.8/02.8a (gate modulus, the $\alpha T$ budget, the constrained modulus), 02.12 (Terminal), 02.13 (the later bed's ablation gap) |
| B25 (theory half) | "faster to train" never defined or measured | 02.1 ($T2^{\rm ep}$ defined with instrument and floor), 02.9 (the step-count form on 04.15's `eval_trace`, the parameter-count form withdrawn, the converse, and 06.31 cited as the owner of the B25 counter), 02.9a (the tied loop, the one matched contrast), 02.10 (the regime), 02.11 (the noise floor and the reading rule) |

## Kills, cheapest first

| verse | kill number (frozen) | price | replacement verse |
|---|---|---|---|
| 02.2 | a record cell beating softmax by $>\mathrm{MDE}_8$ at the realised paired sd on $\ge6/8$ with a non-zero dial | $0$ GPU-s (grep) | the named cell |
| 02.5 | $\lVert\hat P-P_{\rm env}\rVert_\infty<10^{-3}$ on $\ge6/8$ at $d_{\rm model}=16$, matrix $\infty$-norm | $0$ GPU-s (S-62's registered column, no added run) | K-D2 as filed / K-D2′ on BED-S64 |
| 02.7 (a) — arithmetic | $\partial_\gamma L\vert_{0^+}=-\gamma_e\lVert u^\top V\rVert_2^2\sum_{j>i,\,j\in T}\hat P_{ji}P_{{\rm env},ji}\ \ge0$ on one admitted draw — the Statement's sign is false at the bed's own numbers | $0$ GPU-s (S-12 census on the generator, no cell) | 02.7a, the clamp magnitude $\kappa^\star$ |
| 02.7a | $\kappa^\star>\kappa_{\max}$ on $>\tfrac12$ the admitted draws — the clamp leaves the simplex | $0$ GPU-s (S-12 census on the generator, before any cell) | Bet A VOID, Terminal |
| 02.11 (If killed) | $\delta\ge\mathrm{MDE}_8$ | $0$ GPU-s (arithmetic) | reprice $N$ |
| 02.12 | R-21 MOVED ($\Lambda>\ln4096=8.318$; PINNED at $\le2.7055$, the band between firing neither) on $\ge6/8$ on **BED-J** (04.6, seed rule 04.9 clause 1, $N\ge8$ repriced by clause 6, $\mathrm{MDE}_N$ at 04.9 clause 4's realised $\sigma_d$) **or BED-C-N** (04.7 / 09.11, seeds $0$–$7$ deduplicated, $N=8$, $N=16$ on K-P at realised sd $\ge2.18\times0.050146$) — no third bed | $0$ GPU-s beyond that bed's own price (journal read of 04.15's declared `gamma` block) | **02.13**, never 02.4 |
| 02.11 | shape identical-seed pair $\lvert\Delta\rvert>2.345\times10^{-3}$ | $3.4$ s | $\varepsilon:=\text{floor}+\max(\mathrm{MDE}_N,\delta)$ |
| 02.1 | $T2^{\rm ep}_{d1}\ge1.00$ at $N=8$, the paired per-seed interval not excluding $1$ | $3.017$–$5.384$ GPU-s inside the arena | the $9{,}600$-step ladder |
| 02.8 | S-52 as registered: **any capped seed still above $1.0$** eval NRMSE, CUDA lane ($\hat a_{\max}$ $12.77/49.66/116.01$); the crossing count out of three is the scored column with Bet I's SPLIT band $2$ of $3$, not a kill | $\approx9.65$ s | 02.8a, the constrained modulus $a=\exp(-\mathrm{softplus}\,g)$ |
| 02.8a | $\hat a_{\min}<10^{-3}$ on $\ge2$ of 3 seeds — the gate collapsed instead of diverging | $\approx9.65$ s | Terminal: mechanism unresolved |
| 02.3 (a) | $\hat\beta$ inside $[1-\delta_\beta,1+\delta_\beta]$ on $\ge6/8$ with the certificate `refused`, $\delta_\beta=2.345\times10^{-3}$ — the bind is decoration | $\approx13.4$ s | projected $\beta=\sigma(\theta)$ |
| 02.3 (b) | a $\beta$-learnable cell below the $\beta\equiv1$ cell by $>\mathrm{MDE}_8$ on $\ge6/8$, whatever $\hat\beta$ reads | $\approx13.4$ s (same journal) | row renormalisation to exactly $1$ |
| 02.6 | the $\lambda=1$ arm's $\lVert\hat P-P_{\rm env}\rVert_\infty$ within one seed sd of the $\lambda=0$ arm's on $\ge6/8$ and R-21 unchanged; $\lambda$ swept on the frozen grid $\{0,\tfrac14,1,4,16\}$, $\lambda=0$ the planted negative | $\approx53.8$ s (four added arms beyond the $\lambda=0$ baseline) | direct $\hat P$ pretraining stage (identification only) |
| 02.4a | released $\hat\gamma$ PINNED ($\Lambda\le2.7055$) on $\ge6/8$ | $\approx13.4$ s (one added arm on S-62) | 02.4b |
| 02.4b | fixed-$\gamma$ $z$-channel within $\mathrm{MDE}_8$ of depth-1 softmax on $\ge6/8$ | $\approx13.4$ s (one added arm on S-62) | 02.12 |
| 02.9a | $\lvert\Delta_5\rvert$ within $\mathrm{MDE}_8$ on $\ge6/8$ — the solve and its own five-term tied iterate are one object | $\approx13.4$ s (the tied-loop arm) | Terminal |
| 02.4 | $\Lambda\le2.7055$ on $\ge6/8$ (one number; the $F_{\rm rep}$ conjunct dropped) | $0$ GPU-s $+\le17.4$ s ablation | 02.4a |
| 02.7 (b) — training | $\Lambda\le2.7055$ on $\ge6/8$, **or** $\hat\gamma$ moved against $\operatorname{sign}(\gamma_{\rm env}-\tfrac12)$ on $\ge6/8$ — decides reachability under the Adam step bound, never the sign | $\approx13.4$ s (added arm) $+\le17.4$ s ablation | 02.7a, the clamp magnitude $\kappa^\star$ |
| 02.13 | ablation gap $A\le$ one realised seed sd of that bed on $\ge6/8$, on whichever of BED-J (04.6) or BED-C-N (04.7 / 09.11) fired 02.12 | $\le17.4$ s, no retraining | **none — no replacement exists**; Terminal in force |
| 02.10 | decay-on/off identical-seed pair within $\delta_\beta$ on 8/8 at $T=150$ **on the arena** | $\approx27$ s | the $\hat\gamma$ column at $\theta_0=\operatorname{logit}\gamma_{\rm env}$ |
| 02.10 (If killed) | $\lvert\hat\gamma-\hat\gamma'\rvert\le\delta_\beta$ on 8/8 at $T=150$ | $\approx27$ s (same journal) | prescription deleted from the arena manifest |
| 02.9 | $T2^{\rm ep}_{\rm sky}\ge1$ at $\varepsilon=\text{floor}+\max(\mathrm{MDE}_8,4.7\times10^{-3})$, $N=8$ | arena $+\approx64.8$ s skyline | 02.9a, the tied loop's terminal gap $\Delta_5$ |
| 02.1 (If killed) | $T2^{\rm ep}_{d1}\ge1.00$ at $9{,}600$ steps | $0$ GPU-s (ladder checkpoints); the ladder $\approx36$ min per pair, survivor only | $T2^{\rm ep}$ VOID on BED-S |

## Limits

Every DERIVED bound above is stated at the geometry printed in its Hypotheses and nowhere else; the
Adam step bound is loose by the factor of consistent-sign gradients; the torch defaults were read from
the installed `2.14.0+cpu`, not the certified `2.5.1+cu121` (B1); no CEQ arm has been trained, so
every kill on BED-S is decidable only after S-12 admits and S-62 runs; the boundary-null citation is
owed and not named; `[V]` sources for tying and learnability are abstract-level and carry no
load-bearing sentence here; the identical-seed floor of 02.11 is a reading on two journals of one
lane, one box and one torch version, and licenses nothing about the shape's own backward. Two further
limits enter with repair round 1, batch 2: the CPU and CUDA lanes of `e3_t2` disagree by $2.457\times$
on $\hat a_{\max}$ at seed 7 and by $1.59\times$ at seed 2, so every gate number in this book carries
its lane and the two columns are never differenced (02.8); and **no depth-5 arm has ever been
constructed**, so $p_5=23{,}777$ is DERIVED by multiplication from a decomposition of a depth-1 count
and is not a reading of anything — the parameter-count form of every T2 ratio is withdrawn until a
constructed depth-5 cell prints its own `n_params` in book 04. 02.13's Terminal is in force the moment
its ablation gap reads at or below one seed sd: that verse carries no replacement, and the loss is
stated rather than routed around. Three further limits enter with repair round 1, batch 3:
$\lambda_{\min}$ is a ratio of two initialisation gradient norms and no cell has printed either, so
02.6's hop-$\ge2$ gradient clause is conditional at every realised $\lambda$ and only the frozen grid
$\{0,\tfrac14,1,4,16\}$ is registered; 02.7's cheap kill (a) is decidable at $0$ GPU-s on the S-12
census and no admitted draw has been generated, so it is decidable and undecided; and 02.7's
reachability census now reads $\|\Delta z^{\rm lab}_{>i}\|_\infty$ over transient rows only, which is a
change to an instrument's field definition that this book states and `04_BEDS_AND_INSTRUMENTS.md` must
carry. Three further limits enter with repair round 1, batch 4: 02.12's kill is now frozen on **two
named beds and no third** — BED-J (`04_BEDS_AND_INSTRUMENTS.md` 04.6) and BED-C-N (04.7, protocol
`09_CHESS_AND_MARKETS.md` 09.11) — and **neither has a cell**, so the kill is registered, decidable and
undecided, 02.13's ablation gap $A$ has no value, and both beds' $\mathrm{MDE}_N$ waits on the same
$\sigma_d$ from 04.21 step (5) that eight kills of book 04 wait on; the MOVED threshold
$\Lambda>\ln4096=8.318$ joins $2.7055$ in carrying **no `references.bib` citation**, so that debt is
owed twice over; and every seed sd this book reads is now read in the lane it is used in — 02.11's
control against the v17k_r4 lane's own $0.011452$ at $N=8$ — which leaves the R1 CPU figures
$0.011824$ and $0.253673$ as labelled cross-lane notes that no threshold of this book reads.

## Attacks answered

MARS's findings, repair round 1, batch 1 of 4, each verbatim as filed, with the repair beside it. A
finding that reads more gently here than when filed is itself a defect; none is softened, none is
answered by deleting a verse, and `replacement_survives: false` is answered by a **different** number
on a **different** object measured by a **different** instrument, or by an OPEN row with its Terminal
in force.

---

**F1 · verse 02.1 · severity `strike` · mechanism V-11**

*Flaw (verbatim).* The Kill fires only in the region the verse's own Hypotheses declare VOID, so it
cannot fire on the bed as registered.

*Number (verbatim).* Hypotheses: "Non-vacuous only if the shape reads $N_\varepsilon<\infty$ on at
least 6 of 8 seeds; otherwise T2 is VOID on the bed." Kill: "$N_\varepsilon(\text{shape},\sigma)=\infty$
on at least 3 of 8 seeds". 3 of 8 infinite leaves at most 5 of 8 finite, i.e. exactly the complement of
the non-vacuity clause. Within the Hypotheses the kill has probability zero.

*`replacement_survives`: false.*

*Repair — 02.1.* The $\ge6/8$ non-vacuity clause is **deleted** from the Hypotheses, which now read
"No non-vacuity clause is asserted here: **the kill decides vacuity**". The kill is re-frozen inside
the Hypotheses as $T2^{\rm ep}_{d1}\ge1.00$ at $N=8$ with the paired per-seed interval not excluding
$1$; an infinite seed enters the median as $+\infty$, so $\ge5$ of $8$ infinite shape seeds forces the
median infinite and fires the same number. Vacuity and defeat are now decided by one number, and no
precondition excuses either. The kills table row for 02.1 is rewritten to that number.

---

**F2 · verse 02.1 · severity `strike` · mechanism P-7**

*Flaw (verbatim).* "Draws consumed" counts presentations of one fixed training set, so T2's
sample-efficiency half has no referent on the arena.

*Number (verbatim).* `READ scripts/v15_r1.py:250-256 @ 99777ab`: `for t in range(steps): loss =
mse_loss(model(x_tr), y_std)` — the same `x_tr` of $n_{\rm train}=2048$ every step. Distinct draws are
$2048$ at every $t$; the verse's "draws consumed $=n_{\rm train}\cdot t$" is $2048t$ presentations. At
$t=150$ the two differ by $150\times$.

*`replacement_survives`: false.*

*Repair — 02.1.* The quantity is renamed **epochs-to-floor**, $T2^{\rm ep}$, throughout the book, the
preface and both ratio symbols. CHARTER §1's draws-to-floor is filed `NOT MEASURED — needs a
fresh-draw or minibatch regime registered for BED-S, with $N_\varepsilon$ re-derived in samples`, and
the Terminal states it is unmeasurable at the arena as registered. No minibatch regime is invented
here: registering one is book 04's, and this book withdraws the claim rather than renaming it.

---

**F3 · verse 02.1 · severity `strike` · mechanism P-7**

*Flaw (verbatim).* The licensing sentence grants "faster to train" on T2 alone and on a skyline ratio
at unmatched parameters, both narrower than CHARTER §1.

*Number (verbatim).* `READ docs/canon/CHARTER.md:96-104 @ 99777ab`: "**Faster to train** means two
measured numbers" (T1 **and** T2), each "ratio against base self-attention **and** against the depth
skyline", "at matched parameters". 02.1 licenses on $R_{T2}>1$ alone, against "the skyline ... at
unmatched parameters", with depth-1 softmax only "printed in the same row".

*`replacement_survives`: false.*

*Repair — 02.1.* Two ratios are printed in every row, $T2^{\rm ep}_{d1}$ against depth-1 softmax and
$T2^{\rm ep}_{\rm sky}$ against the depth skyline. The licensing paragraph now reads: a row in which
only $T2^{\rm ep}$ holds licenses "T2 alone, in the epochs form, at this bed and this $\varepsilon$"
and **never** "faster to train", which needs T1 (book 03) as well. The word "matched" is **withheld
from every row of this book**: the shape's `n_params` is `NOT MEASURED`, and the record's nearest
non-softmax arms miss $4{,}769$ by $0.713\,\%$ (`arm_pl` $4{,}803$) and $0.776\,\%$ (`arm_smprime`
$4{,}806$) against Ruling 3's $0.032\,\%$.

---

**F4 · verse 02.1 · severity `strike` · mechanism V-17**

*Flaw (verbatim).* $R_{T2}$ is defined as the reciprocal of the T2 ratio used by two neighbouring
canon books, so the same name carries two directions.

*Number (verbatim).* 02.1: $R_{T2}=\operatorname{median}_\sigma N_\varepsilon(\text{skyline},\sigma)/
N_\varepsilon(\text{shape},\sigma)$, licensed iff $>1$. `03_KERNEL.md:194`: "$T1(a)/T1(b)=(k^\star(a)/
k^\star(b))\cdot(\bar t_{\rm step}(a)/\bar t_{\rm step}(b))$: the draws-to-floor ratio (T2, book 02)"
with $a=$ shape, $b=$ control. `06_PREDICTIONS.md:377-381`: "Prediction: $T2_{\rm sky}\le0.20$ — the
shape needs at most a fifth of the stack's draws". $R_{T2}=5$ and $T2_{\rm sky}=0.20$ are the same cell
written two ways.

*`replacement_survives`: false.*

*Repair — 02.1 and 02.9.* One orientation is fixed in 02.1 — **shape over control**, licensed
direction $<1$ — matching 03.8's $k^\star(a)/k^\star(b)$ and 06.31's $T2_{\rm sky}\le0.20$, and the
symbol $R_{T2}$ is **retired**: 02.9's Statement, prediction, counter, SPLIT band and Kill are all
rewritten in 02.1's orientation ($T2^{\rm ep}_{\rm sky}\approx p_1/p_L=4{,}097/23{,}845=0.1718$; tied
counter $4{,}097/4{,}096=1.0002$; prediction $[0.20,0.50]$; counter $\ge1$; SPLIT $(0.50,1)$; kill
$T2^{\rm ep}_{\rm sky}\ge1$), and the kills table with them. The two sentences that still print
$R_{T2}$ are the two that retire it. **Not repaired here:** 03.8 and 06.31 cannot be edited from this
book, so the cross-book citation-by-verse-id the finding asks for is owed to the critic; this book's
orientation is the one they must cite.

---

**F5 · verse 02.1 · severity `strike` · mechanism P-4**

*Flaw (verbatim).* The Kill's instrument is deferred to book 04, which contains no verse defining it,
and its price is derived at the wrong $n$.

*Number (verbatim).* Kill instrument: "S-62 with the held-out trace every 10 steps"; Evidence:
"`NOT MEASURED — needs the per-step held-out trace (book 04)`". `grep -nE "held-out|per-step .*trace"
04_BEDS_AND_INSTRUMENTS.md` returns nothing; the file's only hit is "per-step eval trace" in passing at
`:152`. Price is DERIVED from "$1.473$ ms per $PV$ forward at $n=2048$" while the held-out set is
$n_{\rm eval}=4096$, whose measured increment is $2.312$ ms (`READ docs/CEQ_SHAPE.md:589-590 @
99777ab`).

*`replacement_survives`: false.*

*Repair — 02.1.* The kill is re-based on a **registered journal field**, not a missing instrument:
`04_BEDS_AND_INSTRUMENTS.md:215` (04.15) declares `eval_trace` (per-step eval NRMSE, for T2) on every
cell and makes a missing declared field a **refusal**, and `:152` (04.10, link 1) reads T2 from it.
The kill reads that field at the sixteen points of $G$. The price is re-derived at the held-out width:
$16\times8\times8=1{,}024$ forwards at $n_{\rm eval}=4096$, one forward costing $2.946$ ms (softmax
base, linear in $n$) to $2.946+2.312=5.258$ ms (shape), so $3.017$ to $5.384$ GPU-s — the kills table
carries that band, not "$<5$ GPU-s". What remains `NOT MEASURED` is named honestly: no cell carrying
`eval_trace` exists yet, `needs the first BED-S cell (04.21)`.

---

**F6 · verse 02.2 · severity `strike` · mechanism M-3**

*Flaw (verbatim).* "Every $\gamma>0$ worse" rests in part on a margin below the canon's own
indefensibility bound and a third of that cell's own seed sd.

*Number (verbatim).* `RUN results/r10_v13_hop2gain_t2.txt`: at `gain=0.25` the contrast is $+0.002384$
with cell sd $0.006788$ at three seeds. $0.002384<4.7\times10^{-3}$ (`READ docs/CEQ_SHAPE.md:2675 @
99777ab`) and $0.002384/0.006788=0.351$ of one sd. The sign at $\gamma=0.25$ is unreadable.

*`replacement_survives`: true.*

*Repair — 02.2.* Clause (i) is restated exactly as required: "**four of the five non-zero gains are
worse by more than the indefensibility margin $4.7\times10^{-3}$, and $\gamma=0.25$ is unreadable at
$+0.002384$ against that cell's own sd $0.006788$ at $N=3$**", with the ratio $0.351$ printed, "no
verdict is taken there", and the per-cell sds of all six gains listed. The row is further marked a
pilot at `threads=6` against the arena's `threads=8` (M-10), licensing a direction and never a
seed-rule verdict.

---

**F7 · verse 02.3 · severity `strike` · mechanism P-10**

*Flaw (verbatim).* A sufficient condition for Neumann convergence is asserted as necessary, and the
whole $\gamma\ge0.09718$ consequence follows from the converse error.

*Number (verbatim).* Statement: "The Neumann series behind Proposition 4 converges only if
$\gamma\|W\|_\infty<1$". Convergence of $\sum(\gamma W)^k$ requires $\gamma\rho(W)<1$;
$\gamma\|W\|_\infty<1$ is sufficient only, and $\rho(W)\le\|W\|_\infty$ with strict inequality
generically. Proposition 4 itself states a bound, not a convergence criterion: "for $\|P\|_\infty>1$
the bound *can* fail" (`READ docs/CEQ_SHAPE.md:394-395 @ 99777ab`). What fails at $\gamma\ge1/10.29$ is
the bound, not the series.

*`replacement_survives`: true.*

*Repair — 02.3.* The convergence claim is withdrawn in the words the finding prescribes: "No
convergence claim is made … $\gamma\|W\|_\infty<1$ is **sufficient only** … $\rho(W)$ is `NOT
MEASURED`. What fails is the *bound*, not the series", and the consequence is re-stated as: at
$\|W\|_\infty=10.293107$ Proposition 4's equality clause no longer applies for
$\gamma\ge1/10.293107=0.0971537$, Proposition 5(b)'s mixture property is lost, and the certificate
field is `refused` (S-42). $\rho(W)$ is carried in Evidence as `NOT MEASURED — needs a spectral-radius
read on the trained $W$`.

---

**F8 · verse 02.3 · severity `strike` · mechanism V-10**

*Flaw (verbatim).* The Kill's two conjuncts are mutually exclusive, so no cell can satisfy it.

*Number (verbatim).* Kill: "$\hat\beta_\sigma\in[1-\delta_\beta,1+\delta_\beta]$ on 8 of 8 seeds
**and** whose $z$-channel NRMSE is below the $\beta\equiv1$ cell's by more than $\mathrm{MDE}_8$ on at
least 6 of 8". An arm whose $\hat\beta$ pins inside $[1-\delta_\beta,1+\delta_\beta]$ on every seed
*is* the $\beta\equiv1$ arm to within $\delta_\beta$; a paired NRMSE gap exceeding $\mathrm{MDE}_8$ is
by construction not attainable from a parameter difference of at most $\delta_\beta$.

*`replacement_survives`: false.*

*Repair — 02.3.* The kill is split into the two the finding names, each decidable alone, each with its
own planted control: **(a)** $\hat\beta$ **leaves** $[1-\delta_\beta,1+\delta_\beta]$ on $\ge6/8$ while
the certificate field reads `refused` — the bind is load-bearing; its failure says the bind is
decoration (V-9). Planted negative: the `arm_smprime` lane shows $\beta$ leaving on 8 of 8.
**(b)** a $\beta$-learnable cell beats the $\beta\equiv1$ cell by $>\mathrm{MDE}_8$ on $\ge6/8$
**whatever $\hat\beta$ reads** — the bind cost accuracy. Planted negative: the $\beta\equiv1$ cell
against itself must read $0.0$ on 8 of 8. The two replacements stay distinct — the projection recovers
the certificate half only, row renormalisation recovers the mixture half — and the kills table now
carries both rows.

---

**F9 · verse 02.3 · severity `strike` · mechanism M-2**

*Flaw (verbatim).* $\delta_\beta$ is the threshold of two kills and is never given a number anywhere in
the book.

*Number (verbatim).* $\delta_\beta$ appears in 02.3's Kill, in 02.3's If-killed kill, and in 02.10's
Kill and If-killed. `grep -n 'delta_\\beta' 02_THEORY_TRAINING.md` returns four occurrences and no
assignment. CHARTER §2 requires "the single decidable number ... with its threshold FROZEN here
(M-2)".

*`replacement_survives`: false.*

*Repair — 02.3.* A block **"$\delta_\beta$, frozen here (M-2)"** freezes
$\delta_\beta:=2.345\times10^{-3}$ in $\hat\beta$ units — the arena's identical-seed resolution
(`READ docs/CEQ_SHAPE.md:2675 @ 99777ab`, A.11 item 4; the identical-seed floor itself reads $0.0$,
02.11, so the thread-count floor is the resolution in force) — and states that this is the single
number for **every** occurrence in the book, 02.3's two kills and 02.10's two. Its realised-sd
replacement is named ($\mathrm{MDE}_8$ on the $\hat\beta$ column once S-62 prints it) and the realised
sd of $\hat\beta$ is filed `NOT MEASURED`.

---

**F10 · verse 02.4 · severity `strike` · mechanism V-25**

*Flaw (verbatim).* Clause (b), the only derived descent direction off the corner under next-state
labels, is asserted at a state 02.5 proves unreachable at the arena's geometry, and no perturbation
bound replaces it.

*Number (verbatim).* 02.4(b) Hypotheses: "needs $\hat P=P_{\rm env}$". 02.5(ii): "exact
$\hat P=P_{\rm env}$ needs the adjacency-by-id block of $W_Q^\top W_K$ to equal $c\,I_s$ — rank $s=64$
— impossible at $\operatorname{rank}\le d_{\rm model}=16$; so $F_{\rm rep}>0$". 02.4(c): "At an
arbitrary $\hat P$ the sign ... is undetermined." The set on which the descent is derived is empty at
$d_{\rm model}=16$.

*`replacement_survives`: false.*

*Repair — 02.4(b).* The perturbation bound the finding demands is **derived**, with the explicit
radius: writing $\hat P=P_{\rm env}+\Delta$, $\epsilon=\|\Delta\|_\infty$, $\nu=\max_i\|V_i\|_2$,
$$\partial_\gamma L\big|_{0^+}\le-\gamma_e\|G\|_F^2+\epsilon\sqrt s\,\nu\|G\|_F(1+4\gamma_e)
+4\epsilon^2s\nu^2,$$
and requiring each positive term below $\tfrac12\gamma_e\|G\|_F^2$ gives
$$\eta^\star=\min\left\{\frac{\gamma_e\|G\|_F}{2\sqrt s\,\nu(1+4\gamma_e)},\
\frac{\|G\|_F}{2\nu}\sqrt{\frac{\gamma_e}{2s}}\right\},$$
with $\partial_\gamma L|_{0^+}<0$ for every $\hat P$ inside the $\eta^\star$-ball. The clause is
declared non-vacuous **iff** $F_{\rm rep}\le\eta^\star$, and $\|G\|_F$, $\nu$ and the arena's
$\|\hat P-P_{\rm env}\|_\infty$ column are named as what decides it, `NOT MEASURED` until 04.21. The
sign is asserted on the ball and nowhere else.

---

**F11 · verse 02.4 · severity `strike` · mechanism V-10**

*Flaw (verbatim).* The landscape is computed at a point outside the range of the parametrisation the
same verse prescribes, and at a $\gamma$ no cell ever occupies.

*Number (verbatim).* Statement: "Parametrisation: $\gamma=\sigma(\theta)$, $\theta_0=0$
($\gamma_0=\tfrac12$)". Under $\gamma=\sigma(\theta)$, $\gamma=0$ is attained only at
$\theta=-\infty$, and $\partial_\theta L=\sigma'(\theta)\,\partial_\gamma L$ with
$\sigma'(\theta)=\gamma(1-\gamma)\to0$ as $\gamma\to0$: the corner is absorbing in $\theta$ whatever
$\partial_\gamma L|_0$ reads. Every derivative in (a), (b), (c) is evaluated at $\gamma=0$; the
registered init is $\gamma_0=0.5$.

*`replacement_survives`: false.*

*Repair — 02.4.* The verse is retitled "**the dial does not leave its initialisation**". A block
"What the corner is, and what it is not" prints $\partial_\theta L=\gamma(1-\gamma)\partial_\gamma L
\to0$ and files (a), (b), (c) as **corner algebra on the $\gamma\to0^+$ ray**, never as training-side
theorems. The trajectory's actual start is computed: $\partial_\theta L|_{\theta=0}=\tfrac14
\partial_\gamma L|_{\gamma=1/2}$, non-zero on a codimension-one condition, so the dial is generically
non-stationary at $\gamma_0=\tfrac12$ **with a sign no clause of the verse determines**; the
training-side question is restated as "does $\hat\gamma$ leave $\tfrac12$", which is what R-21's
PINNED verdict measures. The clamp alternative is noted as V-10 by construction. The preface item 2
carries the same restatement.

---

**F12 · verse 02.4 · severity `strike` · mechanism V-11**

*Flaw (verbatim).* The Kill conditions on $F_{\rm rep}$, a quantity 02.5 marks NOT MEASURED with an
instrument that exists in no book, while pricing itself at $0$ GPU-s plus an ablation.

*Number (verbatim).* Kill: "$\Lambda\le2.7055$ on at least 6 of 8 seeds **while**
$\|\hat P-P_{\rm env}\|_\infty$ reads within $0.1$ of the representational floor $F_{\rm rep}$ of
02.5". 02.5 Kill: "`NOT MEASURED — needs the representational-floor instrument (book 04)`".
`grep -n 'F_{\\rm rep}\|representational' 04_BEDS_AND_INSTRUMENTS.md` returns zero lines. The kills
table prices this at "$0$ GPU-s $+\le17.4$ s ablation".

*`replacement_survives`: true.*

*Repair — 02.4.* The $F_{\rm rep}$ conjunct is **dropped**, in the finding's own first option. The kill
is frozen on $\Lambda\le2.7055$ on $\ge6/8$ alone, which S-61/R-21 already decides at $0$ GPU-s plus
the $\le17.4$ s ablation; the verse says in the Kill field why the conjunct was struck (V-11, P-4), and
the kills table row is rewritten to the single number. No number in this book now conditions on
$F_{\rm rep}$.

---

**F13 · verse 02.5 · severity `strike` · mechanism P-4**

*Flaw (verbatim).* The Kill carries an evidence class of RUN and NOT MEASURED at once, and its
instrument exists in no book.

*Number (verbatim).* Kill: "**RUN** $F_{\rm rep}$ at construction on 512 draws (a CPU optimisation
over the class per draw, $0$ GPU-s)" followed by "`NOT MEASURED — needs the representational-floor
instrument (book 04)`". `grep -n 'F_{\\rm rep}\|representational' 04_BEDS_AND_INSTRUMENTS.md` returns
zero lines. The kills table also prices it at "$0$ GPU-s (CPU, NOT MEASURED — book 04)".

*`replacement_survives`: false.*

*Repair — 02.5.* Book 04 cannot be edited from here, so the second option is taken: the kill is
re-based on the column the arena already writes. Frozen at
$\|\hat P-P_{\rm env}\|_\infty<10^{-3}$ on $\ge6/8$ at $d_{\rm model}=16$ under the matrix
$\infty$-norm, read from S-62's `*measure*` line (`READ docs/PLAN.md:626-633 @ 99777ab`) in 04.15's
`census` block, price $0$ GPU-s, one evidence class. A realised $\hat P$ is a member of $\mathcal C$,
so the reading **upper-bounds** $F_{\rm rep}$ and firing it proves $F_{\rm rep}<10^{-3}$ as a theorem.
Planted negative that makes it fire: the BED-S64 cell at $d_{\rm model}=64\ge s$. Planted positive that
must not fire: the step-0 cell, which must read $\ge0.5$ on 8 of 8. The Terminal now states that
$F_{\rm rep}$ conditions no kill, replacement or row anywhere in this book.

---

**F14 · verse 02.5 · severity `strike` · mechanism V-17**

*Flaw (verbatim).* The margin constant $c\ge9.62$ is derived one-sided; under the matrix $\infty$-norm
the row deviation is twice the leaked mass and the constant is $10.31$.

*Number (verbatim).* Statement (iii): "K-D2's $10^{-3}$ at $i=63$, $\deg_i=4$ needs $c\ge9.62$
(DERIVED)". Solving $60e^{-c}/(4+60e^{-c})=10^{-3}$ gives $c=9.6148$ (the printed value). But
$\|\hat P-P_{\rm env}\|_\infty$ is the maximum absolute row sum, and a row that leaks mass $L$ onto
$i+1-\deg_i$ non-edges also loses $L$ from its $\deg_i$ edges, so the row's deviation is $2L$. K-D2 at
$10^{-3}$ then needs $L<5\times10^{-4}$: $60e^{-c}/(4+60e^{-c})=5\times10^{-4}$ gives $c=10.3085$. The
norm convention is stated nowhere in the verse or in K-D2 (`READ docs/PLAN.md:375 @ 99777ab`).

*`replacement_survives`: true.*

*Repair — 02.5(iii).* The convention is stated explicitly — $\|A\|_\infty=\max_i\sum_j|A_{ij}|$, the
maximum absolute row sum, for every $\|\hat P-P_{\rm env}\|_\infty$ in this book and in K-D2, which
prints none — the two-sided derivation is written out (the row loses the same $L_i$ from its
$\deg_i$ edges, so the deviation is $2L_i$), and the canon's constant is printed as
$c\ge\ln 29{,}985.0=10.3085$, i.e. $c\ge10.31$. The one-sided $c\ge9.6148$ is kept only as the
elementwise-max reading with that convention named, and the preface carries $10.31$.

---

**F15 · verse 02.5 · severity `strike` · mechanism Ruling 3**

*Flaw (verbatim).* The replacement's own kill escapes by raising $d_{\rm model}$ from 16 to 64, and
cites Ruling 3 as the licence for the very move Ruling 3 forbids.

*Number (verbatim).* Kill of K-D2′: "$F_{\rm rep}\ge0.5$ on the median draw ... then $d_{\rm model}$ is
raised to $s$ with the control re-matched at that width (Ruling 3 forbids re-architecting to close a
count, so the count is re-derived, $0$ GPU-s)". Raising $d_{\rm model}$ from $16$ to $64$ is a $4\times$
width change on the arm, i.e. re-architecting; `READ docs/canon/CHARTER.md:79-80` reads "no
re-architecting to close it".

*`replacement_survives`: false.*

*Repair — 02.5, K-D2′ rewritten.* The width change is registered as **BED-S64**, a separate bed under
Ruling 7 (same generator, seed rule and hash) with **its own** depth-1 control, its own exact
restricted-view floor and its own $\mathrm{MDE}_N$ from its own realised paired sd, filed explicitly as
a **reroute to a second bed and never as a re-match of the $d_{\rm model}=16$ row**; the verse now says
that raising the width $16\to64$ is a $4\times$ re-architecture and that no count of the width-16 row
may be re-derived at width 64. K-D2′ is re-derived on a genuinely different object — the two-width
excess $E=\|\hat P-P_{\rm env}\|_\infty^{(16)}-\|\hat P-P_{\rm env}\|_\infty^{(64)}$, which contains no
unmeasured term — priced as a second arena at $\approx98$ s, and its own kill (the width-64 column
itself $\ge10^{-3}$ on $\ge6/8$: the gap is optimisation, not representation) is decided at $0$ GPU-s
and separates what the verse's own kill cannot.

---

**F16 · verse 02.6 · severity `strike` · mechanism V-25**

*Flaw (verbatim).* Clause (iii) makes the auxiliary loss useful by driving $\hat P\to P_{\rm env}$, a
limit the next verse but one proves unreachable at the same geometry.

*Number (verbatim).* 02.6(iii): "makes 02.4(b)'s descent direction live as $\hat P\to P_{\rm env}$: the
corner fix that costs no dial". 02.5(ii): $F_{\rm rep}=\min_{\hat P\in\mathcal C}
\|\hat P-P_{\rm env}\|_\infty>0$ at $d_{\rm model}=16<s=64$. A loss cannot make an empty limit live; at
best it drives $\hat P$ to the $F_{\rm rep}$-ball, where 02.4(c) says the dial's gradient sign is
undetermined.

*`replacement_survives`: false.*

*Repair — 02.6(iii).* The clause is restated as "$L_q$ **drives $\hat P$ toward the $F_{\rm rep}$-ball**
— **not** toward $P_{\rm env}$", with the sentence "a loss cannot make an empty limit live" in the
verse. 02.4's $\eta^\star$ is imported as the radius at which the sign survives, and the dial
consequence is declared to hold **iff $F_{\rm rep}\le\eta^\star$**, which is `NOT MEASURED`; the
Hypotheses now say the dial half is asserted nowhere in this book. The decidable form is given on the
column S-62 writes (a seed whose journalled $\|\hat P-P_{\rm env}\|_\infty\le\eta^\star$). The verse
states the consequence the finding names: **committor supervision is a fix for the head and not for
the dial** until such a seed exists; what survives unconditionally is the head, because (i) puts no
$\gamma$ in that gradient.

---

**What stays owed after this batch.** F4's cross-book half — 03.8 and 06.31 citing 02.1's orientation
by verse id — cannot be done from this book and is the critic's. F5's `eval_trace`, F10's $\|G\|_F$,
$\nu$ and $F_{\rm rep}$, F13's arena column, F15's BED-S64 cell and F16's $\eta^\star$ comparison are
all `NOT MEASURED` until 04.21 prints the first BED-S cell; every kill above is decidable the day it
does, and none is decidable before. No finding of this batch is left OPEN.

---

MARS's findings, repair round 1, **batch 2 of 4**, each verbatim as filed, with the repair beside it.
The same rule holds: nothing softened, no verse deleted to escape a finding, and every
`replacement_survives: false` answered by a **different** number on a **different** object measured by
a **different** instrument, or by an OPEN row with its Terminal in force.

**What batch 2 supersedes in batch 1's own record (L-G2: the earlier row is not edited, it is marked).**
F4's repair, filed above, rewrote 02.9 in 02.1's orientation and in doing so carried the arithmetic
forward unchanged: "$T2^{\rm ep}_{\rm sky}\approx p_1/p_L=4{,}097/23{,}845=0.1718$; tied counter
$4{,}097/4{,}096=1.0002$; prediction $[0.20,0.50]$; counter $\ge1$; SPLIT $(0.50,1)$". F20 to F24 below
strike every one of those numbers: $23{,}845$ over-counts by $68$ (F21), the two counts are two
geometries (F20), no skyline count is registered at all (F22), the uniform-convergence form that
turned counts into a ratio does not reach a step count (F23), and the prediction, counter and SPLIT
band are 06.31's and are cited by verse id rather than restated here (F24). **F4's orientation stands;
F4's parameter arithmetic does not.** The batch-1 row above is left exactly as it was written, and this
paragraph is the supersede marker.

---

**F17 · verse 02.7 · severity `strike` · mechanism V-10**

*Flaw (verbatim).* The verse that "decides the book first" computes its descent direction at a $\gamma$
that lies outside the range of the parametrisation the book prescribes, where the gradient in the
actual parameter is zero.

*Number (verbatim).* Statement: "$\Delta O_j|_{\gamma=0}=0$ ... $\partial_\gamma L|_{\gamma=0}=
-\gamma_e(u^\top V)^2\sum_{j>i}\hat P_{ji}P_{{\rm env},ji}<0$ ... the corner is a strict descent
direction of the dial at every initialisation." 02.4 prescribes $\gamma=\sigma(\theta)$, $\theta_0=0$:
$\gamma=0$ is attained only at $\theta=-\infty$, and $\partial_\theta L=\gamma(1-\gamma)\partial_\gamma
L\to0$ as $\gamma\to0$. "At every initialisation" quantifies over $\hat P$, never over $\gamma$, and no
cell is ever initialised at $\gamma=0$.

*`replacement_survives`: false.*

*Repair — 02.7, split into (A) and (B).* (A) keeps the $\gamma=0$ result **as an algebraic identity
about the corner and nothing else**, with the withdrawal written into the verse: the phrase "the corner
is a strict descent direction of the dial at every initialisation" is struck, and the verse states that
it quantified over $\hat P$ and never over $\gamma$, and that the corner is absorbing in the parameter
that moves. The displacement is re-derived exactly in $\gamma$,
$\Delta O_j=(1-\gamma)\gamma(\hat P\hat M)_{ji}(u^\top\hat MV)$, from
$\delta[\hat P\hat M]=\Delta P\hat M+\gamma\hat P\hat M\Delta P\hat M$. (B) is the recomputation the
finding demands, at $\theta=0$ i.e. $\gamma=\tfrac12$: with $F(\gamma)=(1-\gamma)\gamma\,C(\gamma)w(\gamma)$
and $\partial_\gamma L(\gamma)=\langle\int_{\gamma_e}^{\gamma}F',F'(\gamma)\rangle$, under the cone
condition $\langle F'(t),F'(\gamma)\rangle>0$ the sign at the registered initialisation is
$\operatorname{sign}\partial_\theta L|_{\theta=0}=\operatorname{sign}(\tfrac12-\gamma_{\rm env})$ — the
dial descends **toward $\gamma_{\rm env}$**, and $\phi'(\tfrac12)=0$ so the whole first-order signal
comes from the resolvent. The load-bearing consequence is frozen: $|\gamma_{\rm env}-\tfrac12|\ge0.2$
is a registration requirement of the bed, because at $\gamma_{\rm env}=\tfrac12$ the initialisation is
the optimum and PINNED is evidence about the registration and not about the arm. The Kill gains that
sign as a second disjunct ($\hat\gamma$ moved against $\operatorname{sign}(\gamma_{\rm env}-\tfrac12)$
on $\ge6/8$) and a new planted positive that must not fire (a bed at $\gamma_{\rm env}=\tfrac12$ reads
PINNED on 8 of 8). Since the pre-written replacement carried the same $\gamma=0$ defect — "the label is
$\propto\gamma_e$ to first order" — it is struck and replaced by **02.7a**, a different knob: the
displacement is exactly linear in the clamp $u$ with no $\gamma$ in the proportionality, so
$\kappa^\star=0.05\,\mathrm{sd}/q_{50}$ rescales the clamp, and its kill is
$\kappa^\star>\kappa_{\max}=\min_{j:u_j<0}\hat P_{ij}/|u_j|$ on more than half the admitted draws — a
different number on a different object (the simplex, not the dial), decided on the generator at $0$
GPU-s before any cell.

---

**F18 · verse 02.8 · severity `strike` · mechanism P-2**

*Flaw (verbatim).* The pre-derived lr regime contradicts the bound it is derived from: the bound
depends on $\alpha T$ alone, and the proposed $(\alpha,T)$ leaves it unchanged at $1.32\times10^4$.

*Number (verbatim).* Statement: $\hat a_{\max}\le\exp(3.1623\,\alpha T)$; "keeping
$\hat a_{\max}\le1.51$ by the bound alone needs $\alpha\le\ln1.51/(3.1623\cdot150)=8.69\times10^{-4}$,
i.e. $23.0\times$ the steps ($3{,}453$)". If killed: "so the bound itself keeps $\hat a_{\max}\le1.51$".
At $(\alpha,T)=(8.69\times10^{-4},3453)$, $3.1623\cdot8.69\times10^{-4}\cdot3453=9.489$ and
$\exp(9.489)=1.32\times10^4$. Keeping $\hat a_{\max}\le1.51$ requires $\alpha T\le\ln1.51/3.1623=0.13032$,
i.e. $T\le150$ at that $\alpha$.

*`replacement_survives`: false.*

*Repair — 02.8.* The finding's first option is taken and the second half is written out. The verse now
states that the bound is **a function of $\alpha T$ alone**, that at
$(\alpha,T)=(8.69\times10^{-4},3{,}453)$ the product returns to $3.0$ and the bound to
$1.32\times10^4$, and that the three claims of the first writing are mutually inconsistent. **The
regime in force is $(\alpha,T)=(8.69\times10^{-4},150)$ and the displacement budget shrinks
$23.0\times$ with it**: $\alpha T$ falls from $3.0$ to $0.130317$ and the whole-run bound from
$|\Delta\theta|\le9.487$ to $|\Delta\theta|\le0.4121$ — which is 02.4's pinned-without-signal regime,
so the regime is filed as a *diagnosis of the gate* and never as a training route. Because that makes
the lr regime useless as a replacement, the If-killed is **02.8a**, the construction fix
$a=\exp(-\mathrm{softplus}(g))\in(0,1)$ at the arena's own untouched $(\alpha,T)=(0.02,150)$, whose kill
is a different column: $\hat a_{\min}<10^{-3}$ on $\ge2$ of 3 seeds — the gate collapsed rather than
diverged. The two claims the finding says cannot both stand no longer both stand.

---

**F19 · verse 02.8 · severity `strike` · mechanism Ruling 2a**

*Flaw (verbatim).* The failure magnitudes are read from one lane and the kill's control from another,
and the two lanes disagree by $2.5\times$ on the very quantity the verse's mechanism rests on.

*Number (verbatim).* Statement: "$\hat a_{\max}=\exp(\max g)\in\{20.31,49.66,285.07\}$" —
`RUN results/v15_r1.txt` §3, header `device=cpu ... threads=8`. `RUN python -c "..."
results/v17k_r4_retake.jsonl` (2026-09-06, `device: cuda`, same task `e3_t2`, same seeds): `a_hat_max`
reads $12.768$ (seed 2), $49.661$ (seed 3), $116.006$ (seed 7). S-52 registers the capped run "with the
same thread lane and flag regime as the retake journal" (`READ docs/PLAN.md:523 @ 99777ab`), and the
verse's own Kill control quotes the retake journal's $0.6446726192039927$ / $0.6445174549187496$
bitwise. The kill therefore runs where $285.07$ does not occur.

*`replacement_survives`: false.*

*Repair — 02.8.* An eight-row table prints **both lanes side by side** — CPU `â_max`
$1.4107/1.2868/20.3090/49.6613/1.4390/1.5051/1.1029/285.0719$ against CUDA
$1.4104527235031128/1.2868505716323853/12.767516136169434/49.6605224609375/1.4536346197128296/
1.5051767826080322/1.102945327758789/116.00607299804688$, with both headers quoted and both NRMSE
columns beside them — and states the disagreement as measured: $1.59\times$ at seed 2 and
$2.457\times$ at seed 7 ($285.0719/116.00607=2.4574$). **The lane the kill runs in is named CUDA**, on
S-52's own registration clause and on the verse's bitwise control, and the lane's maximum is
$116.006$; $285.07$ is kept with its CPU label and enters no bound. The lr bound is derived against
that lane's maximum: $\alpha=\ln116.00607/(3.1623\cdot150)=1.0021\times10^{-2}$, printed as the largest
lr whose bound is as tight as the realised divergence, and kept distinct from the
$8.688\times10^{-4}$ that the target $1.51$ requires at $T=150$. 02.2(ii)'s quotation of
$20.31/49.66/285.07$ is labelled "on the CPU lane" with the CUDA column beside it. The kills table row
carries the lane and the three CUDA numbers.

---

**F20 · verse 02.9 · severity `strike` · mechanism P-2**

*Flaw (verbatim).* The headline ratio is arithmetically false at the counts printed one clause earlier.

*Number (verbatim).* Statement: "$p_1=2d_{\rm model}d_x+1=4{,}097$ ... $p_L=L\cdot4{,}769=23{,}845$ at
$L=5$ ... $R_{T2}\approx p_L/p_1=5.0$". $23{,}845/4{,}097=5.8201$. The printed $5.0$ is $L$, obtained by
dividing $L\cdot4{,}769$ by $4{,}769$, i.e. by silently substituting $p_1=4{,}769$ — the e3_t2 whole-arm
count at $d=24$ — for the BED-S count $4{,}097$ at $d_x=128$.

*`replacement_survives`: false.*

*Repair — 02.9, defect (1).* The substitution is written out as the finding states it, with
$23{,}845/4{,}097=5.8201$ and "$5.0$ is $L$" printed in the verse. The finding's instruction "fix one
geometry" is met by **fixing neither and withdrawing the ratio**: the verse states that $p_1=4{,}097$
is BED-S's token$\to P$ map at $d_{\rm model}=16$ while $4{,}769$ is the whole `e3_t2` arena arm at
$d=24$, that a ratio across the two is a ratio of two different objects, and that no ratio of parameter
counts is read in this book at any geometry. The prediction band and the SPLIT band are not re-derived
from a parameter count at all — they are cited from 06.31 (F24) and rest on step counts.

---

**F21 · verse 02.9 · severity `strike` · mechanism P-1**

*Flaw (verbatim).* $p_L=L\cdot4{,}769$ counts a whole depth-1 arm $L$ times, so the numerator of the T2
ratio is an over-count with no decomposition printed and no measured instance.

*Number (verbatim).* `RUN python -c "..." results/v17k_r4_retake.jsonl`: `softmax` `n_params`
$=4{,}769$ for the **whole arm** (input projection, head, MLP, readout), not one layer. A stack of $L$
such heads carries one input projection and one readout, so $p_L<L\cdot4{,}769$ strictly. The same
journal's field `dist_to_skyline_why` reads "no v15/v16 scan-skyline module exists (R-SKY)": no depth-5
arm has ever been constructed and no count for it is READ.

*`replacement_survives`: false.*

*Repair — 02.9, defect (2), and the withdrawal.* Both halves of the finding's required repair are
done. The decomposition is printed from the source:
$256_{W_Q}+256_{W_K}+2{,}176_{\rm MLP\,16\to128}+2{,}064_{\rm MLP\,128\to16}+17_{\rm readout}=4{,}769$
(`READ scale/m3_capability.py:104-110 @ 99777ab`, checked against the measured $4{,}769$ by
`RUN python -c "import json;[print(r['kind'],r['seed'],r['n_params']) ...]"`, 2026-09-06), giving
$p_{\rm layer}=4{,}752$, $p_{\rm shared}=17$ and $p_L=L\cdot4{,}752+17$, $p_5=23{,}777$ — an over-count
of exactly $(L-1)\cdot17=68$ in the first writing. The verse also records that this arm has **no** input
projection: it consumes the $d_{\rm model}=16$ feature vector directly, so that term of the finding's
list is zero here. And the skyline count is filed `NOT MEASURED — needs a constructed depth-5 cell with
its `n_params` READ`, quoting `dist_to_skyline_why`, so **the parameter-count form of the ratio is
withdrawn** — the finding's second option, taken in addition to the first rather than instead of it.

---

**F22 · verse 02.9 · severity `strike` · mechanism P-1**

*Flaw (verbatim).* "As registered" is false: no skyline parameter count is registered anywhere in the
record.

*Number (verbatim).* Statement: "$p_L=L\cdot4{,}769=23{,}845$ at $L=5$ **as registered**".
`READ docs/PLAN.md:619-625 @ 99777ab` (S-35) registers depth $3/5/7$ and a price ($\approx7.6$ s per
cell `[ASSUMED]`) and no count. `READ docs/CEQ_SHAPE.md:766 @ 99777ab` (§3.7) registers the skyline row
as "depth $3/5/7$; $\approx7.6$ s per cell [ASSUMED]", class "skyline", explicitly "at unmatched
depth/width/decode length".

*`replacement_survives`: true.*

*Repair — 02.9, defect (3).* "As registered" is **struck** from the verse. Defect (3) states what S-35
and §3.7 register — depth $3/5/7$ and a price and no count — with both pins the finding gives, marks
$23{,}777$ **DERIVED by multiplication and by nothing else**, and adds the standing condition: *until
book 04 registers a depth-5 count on a constructed cell, this book reads no ratio of parameter
counts.* The Evidence field carries the same, and the Limits paragraph repeats it once at the end
where caveats collect.

---

**F23 · verse 02.9 · severity `strike` · mechanism V-17**

*Flaw (verbatim).* A uniform-convergence bound on distinct samples is applied to an optimisation-step
count on a fixed training set.

*Number (verbatim).* Statement: "Under a uniform-convergence form $N_\varepsilon\propto p$
`[ASSUMED]`, $R_{T2}\approx p_L/p_1$". 02.1 defines $N_\varepsilon$ as a gradient-step index on a grid,
and `READ scripts/v15_r1.py:250-256 @ 99777ab` trains full batch on one fixed $n_{\rm train}=2048$ set,
so the distinct-sample count is $2048$ at every $N_\varepsilon$ and does not vary with $p$. Uniform
convergence bounds the sample count, never the step count.

*`replacement_survives`: false.*

*Repair — 02.9, defect (4).* The uniform-convergence form is **struck**, with the finding's own
sentence in the verse: the distinct-sample count is $2048$ at every $N_\varepsilon$ and does not vary
with $p$, and the `[ASSUMED]` tag did not license the transfer. The finding's first option is taken:
the statement in force is an **optimisation statement** — the ratio of medians of
$N_\varepsilon(A,\sigma)=\min\{t\in G:D_t\le\varepsilon\}$, both arms under Adam at the arena's fixed
$\alpha=0.02$, read off the registered per-step `eval_trace` field (`04_BEDS_AND_INSTRUMENTS.md:215`,
04.15). No fresh-draw regime is invented here; registering one is book 04's, and 02.1 already files the
draws form `NOT MEASURED`.

---

**F24 · verse 02.9 · severity `strike` · mechanism D-CALIB**

*Flaw (verbatim).* The counter is the logical negation of book 06's counter for the same census row,
and both are D-CALIB point estimates.

*Number (verbatim).* 02.9: "Prediction: $R_{T2}\in[2,5]$; counter: $R_{T2}\le1$ ... under D-CALIB the
counter is the point estimate", where $R_{T2}=N_\varepsilon(\text{skyline})/N_\varepsilon(\text{shape})$,
so $R_{T2}\le1$ means the shape needs at least as many draws. `06_PREDICTIONS.md:377-382` (06.31, the
same row B25): "the counter at $k\le150$ on $\ge6/8$ with $T2_{\rm sky}<1.0$", where $T2_{\rm sky}$ is
shape over stack, so $T2_{\rm sky}<1.0$ means the shape needs fewer. Both are filed as the point
estimate.

*`replacement_survives`: false.*

*Repair — 02.9.* The orientation was reconciled in batch 1 (F4): 02.1 fixes **shape over control**,
licensed direction $<1$, and retires $R_{T2}$. On that orientation the ownership rule the finding
demands is now written: **`06_PREDICTIONS.md` 06.31 owns the B25 T2 counter, and 02.9 cites it by verse
id and files no second counter** — prediction $T2^{\rm ep}_{\rm sky}\le0.20$ (equivalently $k\le30$ at
$n_{\rm train}=2048$), counter $T2^{\rm ep}_{\rm sky}\ge1.0$ or undefined, SPLIT $(0.20,1.0)$, and
under D-CALIB the **single** point estimate for B25 is 06.31's counter. The verse also records the
reading error the finding rests on, without softening the finding: 06.31's "$T2_{\rm sky}<1.0$" sits in
its **Kill** line and is the condition under which its counter *dies*, not the counter
(`READ docs/canon/06_PREDICTIONS.md:445 @ 99777ab`). What 02.9 keeps that 06.31 does not carry is the
counter's *mechanism* (each untied layer has a dense one-hop gradient from step $0$; the solve's
hop-$\ge2$ signal is conditional on $\hat P$), the tied bound $p_L=p_1$ exactly at every $L$, and the
converse.

---

**F25 · verse 02.10 · severity `strike` · mechanism V-9**

*Flaw (verbatim).* The verse's only $\gamma$-side consequence is vacuous at the initialisation 02.4
prescribes, and the Ruling-10′ reading it names is the opposite of what the arithmetic gives.

*Number (verbatim).* Statement: "Decoupled decay multiplies a zero-gradient parameter by
$(1-\alpha\lambda)^T$: $0.99955$ at $T=150$; $0.94176$ at $T=20{,}000$ ... and of $\theta_\gamma$ toward
$0$ ($\gamma\to\tfrac12$ under the sigmoid) with no data signal, which Ruling 10′ would read as
MOVED-without-signal." 02.4 fixes $\theta_0=0$; $0\cdot0.94176=0$, so $\theta_\gamma$ never moves and
$\gamma$ stays at its initialisation $\tfrac12$. Under Ruling 10′ that reads PINNED, not MOVED.

*`replacement_survives`: true.*

*Repair — 02.10.* The verse now opens the decay clause with "**the decay term is multiplicative, so it
is identically zero at a zero initialisation**", states $0\cdot(1-\alpha\lambda)^T=0$ exactly at
02.4's $\theta_0=0$, that $\theta_\gamma$ never moves, that $\hat\gamma$ stays at $\tfrac12$, and that
**Ruling 10′ reads that as PINNED, not MOVED** — with the first writing's opposite reading struck by
name (V-9). The consequence is restricted to parameters with a non-zero initialisation and printed as a
five-row table: $\beta$ at $1$ on the LM lane ($4.5\times10^{-4}$ at $T=150$; $0.05824$, a $5.82\,\%$
drift, at $T=20{,}000$), $\beta$ at $1$ on the arena lane ($0.029557$ at $T=150$, $12.6\,\delta_\beta$),
$\theta_\gamma$ at $\theta_0=0$ ($0$ exactly at every $T$), and $\theta_\gamma$ at
$\theta_0=\operatorname{logit}0.9=2.1972$ ($\hat\gamma:0.9\to0.894001$, $5.999\times10^{-3}$).

---

**F26 · verse 02.10 · severity `strike` · mechanism V-15**

*Flaw (verbatim).* Both the Kill's control and the replacement's kill are gated on a run that does not
exist and is behind an explicit author authorisation.

*Number (verbatim).* Kill control: "at $T=20{,}000$ the same pair must differ by at least
$0.05\,|\hat\beta|$ ... `NOT MEASURED — needs the LM chunk behind the author's yes`". If killed:
"*Hypotheses:* the chunk runs. *Evidence:* `NOT MEASURED`. *Kill:* the $T=20{,}000$ pair within
$\delta_\beta$". The record's LM cell is `READ COSTS.md:283-287 @ 99777ab` ("NOT YET MEASURED"), and
Bet M is registered "`NOT MEASURED`, behind the author's yes" (`READ docs/PLAN.md:928 @ 99777ab`).
Both links of the chain are unreachable on the bed as registered.

*`replacement_survives`: false.*

*Repair — 02.10.* The whole chain is **re-based on the arena**, where the arena's own $\alpha=0.02$
makes the drift $12.6\times$ larger than the LM's: the Kill's control is now the arena's own arithmetic
prediction, $1-(1-0.02\cdot0.01)^{150}=0.029557$ of $|\hat\beta|$, i.e. $12.6\,\delta_\beta$ at
$T=150$, decidable at $\approx27$ s on a run that exists and needs no authorisation. The $T=20{,}000$
figure is retained **as a note with no kill attached**, marked `NOT MEASURED — needs the LM chunk
behind the author's yes`, and no kill, replacement or row of this book conditions on it. Because the
old replacement hung on the same chunk it is struck, and the If-killed is a different column, a
different parameter and a different predicted magnitude: the **dial** at the non-zero initialisation
02.7's registration forces, $\theta_0=\operatorname{logit}0.9=2.1972\to2.13229$, $\hat\gamma$ drifting
$0.9\to0.894001$, i.e. $5.999\times10^{-3}=2.56\,\delta_\beta$ — **above the resolution Ruling 10′
reads MOVED at** — killed by $|\hat\gamma-\hat\gamma'|\le\delta_\beta$ on 8 of 8 from the same
$\approx27$ s journal, and strictly more decisive because it decides whether a MOVED verdict can be
manufactured by the optimiser with no data signal.

---

**F27 · verse 02.11 · severity `strike` · mechanism P-1**

*Flaw (verbatim).* The RUN citation is a placeholder rather than an exact command, so the number
carries no reproducible provenance.

*Number (verbatim).* Evidence: "`RUN python -c "<read both jsonl, print |eval_nrmse_floor −
eval_nrmse_retake| per (kind, seed)>"`". CHARTER §3 requires running existing code with the exact
command; the angle-bracketed body is a description. The reading itself reproduces —
`RUN python -c "import json; ..." results/v17k_r4_{floor,retake}.jsonl` (2026-09-06) gives $0.0$ on all
six (kind, seed) pairs — so the defect is the citation, not the number.

*`replacement_survives`: true.*

*Repair — 02.11.* The literal one-liner replaces the description, and its literal output is printed
beside it: `RUN python -c "import json;f={(r['kind'],r['seed']):r['eval_nrmse'] for r in
map(json.loads,open('results/v17k_r4_floor.jsonl')) if r.get('t')=='cell'};g={(r['kind'],r['seed']):
r['eval_nrmse'] for r in map(json.loads,open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell'};
[print(k,abs(f[k]-g[k])) for k in sorted(f) if k in g and k[1] in (0,1)]"`, 2026-09-06, printing
`('arm_pl', 0) 0.0`, `('arm_pl', 1) 0.0`, `('arm_smprime', 0) 0.0`, `('arm_smprime', 1) 0.0`,
`('softmax', 0) 0.0`, `('softmax', 1) 0.0`. The dated line is corrected from 2026-09-05 to the
2026-09-06 execution. The six eval values already printed in the verse are unchanged.

---

**F28 · verse 02.12 · severity `strike` · mechanism L-VERSE**

*Flaw (verbatim).* The book's fallback chain cycles: 02.4's chain terminates in 02.12, and 02.12's
replacement is 02.4.

*Number (verbatim).* 02.4 → 02.4a → 02.4b → "**Terminal.** $\gamma$ is a bed-side dial ... (02.12)".
02.12 **If killed:** "02.4 is re-filed for that bed with its own $\Lambda$, ablation and mirror line.
... *Kill:* its R-21." CHARTER §2's depth rule: "Every verse's fallback chain ... ends in a Terminal
within at most three replacements". A cycle has no terminal link, and each link's kill is the same
R-21 at the same $0$ GPU-s, so the "strictly cheaper or strictly more decisive" clause also fails.

*`replacement_survives`: false.*

*Repair — 02.12 and the new 02.13.* The re-entry into 02.4 is **struck**, with the cycle named in the
verse (L-VERSE, CHARTER §2's depth rule quoted). 02.12's If-killed is now **02.13**, a new verse id for
the later bed, exactly as the finding requires, and its kill is the ablation the finding names:
$A:=D_{150}[(I-\hat\gamma\hat P)^{-1}\to I]-D_{150}[(I-\hat\gamma\hat P)^{-1}]$, killed at
$A\le$ **one realised seed sd of that bed** on at least 6 of 8 — strictly more decisive than R-21
because R-21 decides a parameter and $A$ decides the operator, and priced at $\le17.4$ s with no
retraining (the ablation runs at evaluation on the trained cells). Planted negative: the depth-1
softmax control, whose $\hat P$ enters no solve, must read $A=0.0$ on 8 of 8. Planted positive that
must not fire: a cell pinned at $\gamma_{\rm env}=0.9$ must read $A$ above one seed sd on 8 of 8, since
$1/(1-0.9)=10$. **02.13 carries no replacement**: every cheaper instrument in this book reads a
parameter and cannot separate "the operator does nothing" from "the parameter moved", so the verse
states that no replacement exists and its Terminal is in force from the moment the kill fires — the
loss is carried honestly rather than routed around. 02.12's own price line is corrected from "$0$
GPU-s" to "$0$ GPU-s **beyond that bed's own price**, which is not this verse's".

---

**F29 · verse 02.3 · severity `strike` · mechanism P-7**

*Flaw (verbatim).* The word "faster" appears with no form, bed, floor, $N$ or matched count.

*Number (verbatim).* `02_THEORY_TRAINING.md:149`: "Training did find something faster here — the wrong
object: 8 of 8 seeds left $\beta=1$ in 150 steps." `READ docs/canon/CHARTER.md:105-109`: "A verse that
says 'more accurate' or 'faster to train' without naming which of A1, A2, T1, T2 it means, and which
bed, floor, $N$ and matched count it is measured on, is struck at refutation (V-17 ... P-7)."

*`replacement_survives`: true.*

*Repair — 02.3, verified in force, no further edit made.* The sentence the finding names is already
gone from the book, replaced by exactly the movement statement the finding prescribes: "**Gradient
descent moved $\beta$ off $1$ on 8 of 8 seeds in 150 steps** — a movement statement, with no speed word
in it: no form (A1, A2, T1, T2), bed, floor, $N$ or matched count is claimed for it, and none is
licensed." `grep -n "faster" docs/canon/02_THEORY_TRAINING.md` returns ten lines and not one of them is
a capability claim: three are CHARTER §1's licensing rule and the withholding it forces (02.1), four
are withdrawals (02.1's Terminal, 02.10, 02.11, 02.12, 02.13), one is the B25 census row, and two are
this section quoting findings. The finding is recorded here at full strength because it was filed
against this book and stands as filed; it is answered by verification, not by softening.

---

**F30 · verse 02.1 · severity `strike` · mechanism P-4**

*Flaw (verbatim).* Five kills across the book defer their instrument to book 04, which registers none
of them, and the kills table prices five at $0$ GPU-s.

*Number (verbatim).* Deferred: "the per-step held-out trace (book 04)" (02.1), "the schedule flag (book
04)" (02.4a), "the representational-floor instrument (book 04)" (02.5), "S-62's $\Delta z$ channel
(book 04)" (02.7), "the per-step $\Lambda$ trace (book 04)" (02.9 If-killed).
`grep -nE "held-out|schedule|anneal|representational|F_\{\\rm rep\}|\\Lambda trace"
04_BEDS_AND_INSTRUMENTS.md` returns zero lines; the file's only related hit is "per-step eval trace" in
passing at `04_BEDS_AND_INSTRUMENTS.md:152`. The "Kills, cheapest first" table prices 02.4a, 02.4b,
02.5, 02.9(If killed) and 02.12 at $0$ GPU-s.

*`replacement_survives`: false.*

*Repair — 02.1, 02.4a, 02.4b, 02.5, 02.7, 02.9, 02.12, and the kills table.* Book 04 cannot be edited
from here, so the finding's second option is taken for **all five**, and each is repriced:

- **02.1** — repaired in batch 1 (F5): re-based on the registered `eval_trace` field
  (`04_BEDS_AND_INSTRUMENTS.md:215`, 04.15, a missing declared field being a refusal) and repriced at
  $3.017$–$5.384$ GPU-s at the held-out width. No change needed here.
- **02.4a** — the "schedule flag (book 04)" deferral is struck; the anneal is a **manifest field of one
  added arm**, journalled per cell by S-01 (`READ docs/PLAN.md:627 @ 99777ab`), and the price is
  corrected from $0$ GPU-s to $\approx13.4$ s ($8\times1.680$, DERIVED from S-62's per-cell price).
- **02.4b** — the fixed-$\gamma$ arm is likewise one added arm; price corrected from $0$ GPU-s to
  $\approx13.4$ s, and it stays in the chain on decisiveness, not on cost.
- **02.5** — repaired in batch 1 (F13) onto S-62's registered `*measure*` column; its $0$ GPU-s stands,
  because that kill adds no run at all, and the table row now says so.
- **02.7** — the "$\Delta z$ channel (book 04)" deferral is struck: S-62's `*measure*` list already
  writes the field cosine, the magnitude ratio, $\hat\gamma$, $\Lambda$ and
  $\|\hat P-P_{\rm env}\|_\infty$ per arm and seed (`READ docs/PLAN.md:627 @ 99777ab`). What is missing
  is the **cell**, not the instrument, and the price is corrected from $0$ GPU-s to $\approx13.4$ s for
  the added interventional-pair arm plus the $\le17.4$ s ablation.
- **02.9 (If killed)** — the "per-step $\Lambda$ trace (book 04)" does not exist and S-62 writes
  $\Lambda$ once per cell, not per step, so $t_\gamma$ is struck as a kill on an unregistered
  instrument and replaced by **02.9a**, the tied loop, whose number is a terminal accuracy gap
  $\Delta_5$ read off the `eval_nrmse` column S-62 already writes, at $\approx13.4$ s.
- **02.12** — its $0$ GPU-s is qualified in the table as "$0$ GPU-s beyond that bed's own price".

No kill of this book now names an instrument that no book registers.

---

**F31 · verse 02.1 · severity `repair` · mechanism P-2**

*Flaw (verbatim).* The step grid in the Statement does not exist on the instrument the Evidence cites;
$N_\varepsilon=150$ is unobservable.

*Number (verbatim).* Statement: $t\in\{0,10,20,\dots,150\}$, 16 points. `READ
scripts/v15_r1.py:250-256 @ 99777ab`: `for t in range(steps)` with `steps=150` gives
$t\in\{0,\dots,149\}$, and `t % trace_every == 0` at `trace_every=10` fires at $\{0,10,\dots,140\}$, 15
points. The verse's own price line agrees: "15 extra forward passes per cell".

*`replacement_survives`: true.*

*Repair — 02.1, verified in force.* The grid is already written in the form the finding requires:
$G=\{0,10,20,\dots,140\}\cup\{150\}$, $|G|=16$, with the two sources separated — "the in-loop hook
fires at $t\bmod10=0$ for $t<\text{steps}$, giving $\{0,10,\dots,140\}$, and the post-loop read is
taken once, unconditionally, in `model.eval()` at $t=\text{steps}=150$" — and the verse states which
of the two the kill reads: "**The kill reads the whole of $G$**, the fifteen in-loop points and the
sixteenth post-loop point together; a verse that reads only the in-loop grid says so and writes
$\{0,10,\dots,140\}$." The price line is re-derived at $16\times8\times8=1{,}024$ held-out forwards,
consistent with the sixteen points. Recorded verbatim; no further edit made.

---

**F32 · verse 02.1 · severity `repair` · mechanism V-17**

*Flaw (verbatim).* $\varepsilon$ is floored at the thread-count drift instead of at the canon's own
indefensibility bound, which is twice as large.

*Number (verbatim).* 02.1: "$\varepsilon$ must exceed ... the thread-count floor $2.345\times10^{-3}$".
`READ docs/CEQ_SHAPE.md:2675 @ 99777ab` (A.11 item 4): "**Thread-count floor** `2.345e-3`; a margin
below `≈ 4.7e-3` is **indefensible**", and `READ MISTAKES.md:1096 @ 99777ab`: "A margin under roughly
`4.7e-3` is smaller than twice the reduction-order floor and cannot be defended."

*`replacement_survives`: true.*

*Repair — 02.1, verified in force.* The clause is already raised to exactly the form the finding
prescribes, $\varepsilon>\text{floor}+\max(\mathrm{MDE}_N,\,4.7\times10^{-3})$, with the second term
named "the canon's own indefensibility bound — twice the reduction-order floor, above which alone a
margin can be defended — not the thread-count floor $2.345\times10^{-3}$, which is half of it", and
both citations carried in Evidence (`READ docs/CEQ_SHAPE.md:2675 @ 99777ab`, A.11 item 4, and
`READ MISTAKES.md:1096 @ 99777ab`). The same $\max(\mathrm{MDE}_8,4.7\times10^{-3})$ appears in 02.1's
Kill, in 02.9's Kill and in both kills-table rows. Recorded verbatim; no further edit made.

---

**What stays owed after batch 2.** F24's cross-book half — 06.31 and this verse each carrying the
other's id — is half done here (02.9 cites 06.31 by verse id) and half owed to the critic, because
`06_PREDICTIONS.md` cannot be edited from this book. F21 and F22's skyline count is owed to book 04:
until a depth-5 cell is constructed and its `n_params` READ, **no** parameter-count ratio is licensed
anywhere in this canon, and the number $23{,}777$ is DERIVED-by-multiplication and nothing more. F30's
repricings assume S-62's per-cell $1.680$ s, which is the retake journal's lane and is itself an
unsynchronised host clock (C17). F17's radius, F19's capped cells, F26's arena decay pair, F28's later
bed and F30's added arms are all `NOT MEASURED` until the cells are run; every kill above is decidable
the day they are, and none before. No finding of this batch is left OPEN.

---

MARS's findings, repair round 1, **batch 3 of 4**, each verbatim as filed, with the repair beside it.
The rule of the section is unchanged: nothing is softened, no verse is deleted to escape a finding, and
`replacement_survives: false` is answered by a **different** number on a **different** object measured
by a **different** instrument, or by an OPEN row with its Terminal in force. Nine of this batch's
sixteen findings were already answered by the edits of batches 1 and 2; those rows read
**verified in force** and quote the text that carries the required form, so the finding stands at full
strength beside the sentence that satisfies it.

---

**F33 · verse 02.1 · severity `repair` · mechanism Ruling 3**

*Flaw (verbatim).* "Matched $4{,}769$" is printed without the shape's own parameter count and without
the Ruling-3 residual, and the record's nearest non-softmax arms miss it by more than twenty times the
tolerated residual.

*Number (verbatim).* `RUN python -c "json.loads(...)" results/v17k_r4_retake.jsonl` (2026-09-06):
`n_params` reads $4{,}769$ for `softmax`, $4{,}803$ for `arm_pl`, $4{,}806$ for `arm_smprime` —
residuals $34/4769=0.713$ % and $37/4769=0.776$ %, against Ruling 3's matched residual of $0.032$ %
(`READ docs/canon/CHARTER.md:79-80`). `05_REPAIRS.md:749` prints the same gap:
"$4{,}806-4{,}769=37=2\cdot16+5$".

*`replacement_survives`: true.*

*Repair — 02.1, verified in force.* The verse carries the second branch of the required repair in full:
a "**Matched parameters (Ruling 3)**" paragraph states "Every $T2^{\rm ep}$ row prints each arm's own
`n_params` and its residual against the depth-1 control's $4{,}769$", declares "The shape's own count is
`NOT MEASURED` (no shape cell exists), so the word 'matched' is **withheld** from every row of this
book", and prints the two residuals with the ruling's bar beside them — "`arm_pl` $4{,}803$, residual
$34/4{,}769=0.713\,\%$; `arm_smprime` $4{,}806$, residual $37/4{,}769=0.776\,\%$; Ruling 3's matched
residual is $0.032\,\%$". The three counts carry the RUN class with the exact command and date in
Evidence. Recorded verbatim; no further edit made.

---

**F34 · verse 02.2 · severity `repair` · mechanism M-9**

*Flaw (verbatim).* The headline sweep is at $N=3$ and on a different thread lane from the arena, both
of which the canon's pricing rule forbids as evidence.

*Number (verbatim).* `RUN results/r10_v13_hop2gain_t2.txt` header: "seeds=[0, 1, 2] K=64 threads=6".
`READ docs/CEQ_SHAPE.md:2669 @ 99777ab` (A.11 item 1): "**`N = 8` minimum, deduplicated by seed**"; the
arena lane is `threads = 8` (`READ docs/PLAN.md:627 @ 99777ab`, S-62), and M-10 makes the thread count
a lane property (`READ MISTAKES.md:1072-1108 @ 99777ab`).

*`replacement_survives`: true.*

*Repair — 02.2, verified in force.* Clause (i) of the Statement now opens as "**a pilot, $N=3$ on
`threads=6`, licensing no seed-rule verdict**", and closes with "The pilot's lane is `threads=6` and the
arena's is `threads=8` (M-10: the thread count is a lane property), and A.11 item 1 sets $N=8$ as the
minimum deduplicated by seed, so (i) is evidence of a direction and never of a seed-rule verdict."
Evidence quotes the header verbatim — `t*=2 n=2048 steps=150 seeds=[0, 1, 2] K=64 threads=6` — and
carries all three citations the finding names. Recorded verbatim; no further edit made.

---

**F35 · verse 02.2 · severity `repair` · mechanism M-3**

*Flaw (verbatim).* The Kill's threshold is "one seed sd", not the $\mathrm{MDE}_N$ at the realised
paired sd that CHARTER §1 requires.

*Number (verbatim).* Kill: "eval NRMSE below the softmax control by more than one seed sd on at least 6
of 8". `READ docs/canon/CHARTER.md:86-89`: "below base self-attention's by more than $\mathrm{MDE}_N$
at the realised paired sd". At the record's row-2 sd, one seed sd $=0.034451$ against
$\mathrm{MDE}_8=0.039827$ — the kill is $0.86\times$ as strict as the licensed threshold.

*`replacement_survives`: true.*

*Repair — 02.2, verified in force.* The kill reads "by more than $\mathrm{MDE}_8$ at the realised
paired sd on at least 6 of 8 seeds — the threshold CHARTER §1 licenses
(`READ docs/canon/CHARTER.md:86-89`), not 'one seed sd', which at the record's row-2 sd $0.034451$
reads $0.034451$ against $\mathrm{MDE}_8=0.039827$ and is $0.86\times$ as strict as the licensed
threshold", and freezes "the **placeholder** $\mathrm{MDE}_8=0.039827$ at sd $0.034451$, named a
placeholder here and replaced by the realised figure on the first BED-S cell (04.21); at the realised
$0.109199$ it would read $0.126238$". The kills-table row carries the same words. Recorded verbatim;
no further edit made.

---

**F36 · verse 02.3 · severity `repair` · mechanism Ruling 2a**

*Flaw (verbatim).* The constant $1/10.29$ is computed at $\beta=0$, a value no trained cell occupies,
and two of the eight trained cells sit on the other side of $1$ where the bound holds.

*Number (verbatim).* `RUN python -c "..." results/v17k_r4_retake.jsonl` (2026-09-06),
`manifest.smp_values.beta` per seed: $0.7326, 0.8970, 1.3439, 1.5093, 0.5876, 0.7821, 0.8349, 0.9012$.
Two of eight exceed $1$, where $Z_i^{1-\beta}<1$ for $Z_i>1$ and the row is sub-stochastic. The
$1.31$–$10.29$ rows are the $\beta=0$ reading of `READ V16_ARM_SMPRIME.md:29 @ 99777ab`, not a
trained-$\beta$ reading.

*`replacement_survives`: true.*

*Repair — 02.3, verified in force.* The verse carries a per-seed table headed "**The trained-$\beta$
reading, per seed (Ruling 2a: no sentence transfers across corners without a bind at that corner)**",
printing $\hat\beta$, $\|W_{\hat\beta}\|_\infty$ over the eight quoted rows and the $\gamma$ ceiling
$1/\|W\|_\infty$ for all eight seeds — $0.536050$, $0.786471$, $0.894992$, $0.848501$, $0.382301$,
$0.601734$, $0.680478$, $0.794254$ — with the two $\hat\beta>1$ seeds handled explicitly ("the quoted
minimum row is $Z=0.724290<1$, so those two seeds' maximum is taken there and still exceeds $1$"). The
corner constant is labelled as the finding requires: "$1/10.293107$ is kept only as the $\beta=0$
corner-3 number, with that label." The eight $\hat\beta$ carry the RUN class with the exact command.
Recorded verbatim; no further edit made.

---

**F37 · verse 02.3 · severity `repair` · mechanism P-2**

*Flaw (verbatim).* The row-sum range drops the cited source's own minimum, which is below one.

*Number (verbatim).* `READ V16_ARM_SMPRIME.md:29 @ 99777ab` row (e): "at `β=0` the same rows read
`[1.312192, 0.724290, 2.563817, 2.264559, 10.293107, 2.721943, 3.096841, 1.337183]`". The minimum is
$0.724290$, not $1.31$; the verse prints "$1.31$–$10.29$" in the Statement, the Hypotheses and the
census table.

*`replacement_survives`: true.*

*Repair — 02.3, verified in force.* The Statement prints the eight rows in full and reads "so the range
over the quoted rows is $[0.724290,10.293107]$ — the minimum is **below one**, and every number below
is a lower bound on the true $\max_i$ over $64$ rows", with the count stated where the finding requires
it: "**The record's quoted rows are a $\beta=0$ reading, and there are eight of the sixty-four**". The
Hypotheses repeat "The eight rows are quoted of sixty-four; every $\|W\|_\infty$ here is a lower
bound", and 02.2's census clause (iii) carries the same interval $[0.724290,10.293107]$. Recorded
verbatim; no further edit made.

---

**F38 · verse 02.3 · severity `repair` · mechanism V-9**

*Flaw (verbatim).* The pre-derived replacement restores Proposition 4's bound but not Proposition
5(b)'s mixture property, which the Statement lists as the second thing lost.

*Number (verbatim).* Statement: "Proposition 5(b)'s mixture property is lost, and the certificate field
is `refused`". If killed: "$W\leftarrow W/\max(1,\|W\|_\infty)$ per row, so that Proposition 4's second
clause ... applies and the certificate survives". The projection yields $\|W\|_\infty\le1$, i.e.
sub-stochastic rows; Proposition 5(b)'s row-stochastic mixture read
(`READ docs/CEQ_SHAPE.md:391-436 @ 99777ab`) is not recovered by a bound of $\le1$.

*`replacement_survives`: false.*

*Repair — 02.3, verified in force.* The If-killed clause now concedes the half it recovers, in the
finding's own terms — "**The projection recovers only the certificate half.** It yields
$\|W\|_\infty\le1$, i.e. sub-stochastic rows ... Proposition 5(b)'s mixture property needs row sums
equal to $1$, which $\le1$ does not give, so it is **not** recovered and the mixture read stays lost" —
and a **second replacement** is derived for the mixture half: row renormalisation to exactly $1$,
$W\leftarrow\operatorname{diag}(W\mathbb 1)^{-1}W$, with its own Hypotheses ($W\mathbb 1>0$ rowwise,
which holds at finite logits), its own Evidence (DERIVED, one line) and its own kill — the renormalised
arm's $z$-channel NRMSE within $\mathrm{MDE}_8$ of the $\beta\equiv1$ arm's on at least 6 of 8, which
deletes the learnable switch from the shape lane. That kill is a **different object** from the
projection's: the projection is killed on $\hat\beta$ pinning inside $[1-\delta_\beta,1+\delta_\beta]$
under the interior null $\Lambda_\beta\le3.841$ (a parameter-position read), the renormalisation on the
$z$-channel accuracy column (a capability read), so the second replacement does not die to the first's
number. Recorded verbatim; no further edit made.

---

**F39 · verse 02.4 · severity `repair` · mechanism D-1**

*Flaw (verbatim).* Clause (a) needs a conditional-mean-zero noise and is asserted over "every record
bed", whose labels are three-way feature products for which the hypothesis fails.

*Number (verbatim).* Statement (a): "$y=PV+\xi$ with $\xi$ independent of $V$:
$\mathbb E\,\partial_\gamma L|_0=0$". $\partial_\gamma L|_0=\langle-\xi,P(P-I)V\rangle$, whose
expectation vanishes only if $\mathbb E[\xi\mid V]=0$; independence alone leaves $\mathbb E[\xi]\ne0$
admissible. The verse's own Evidence: "the label is a product of three features (D-1 regime)" and
`READ MATHEMATICS.md:302-306 @ 99777ab`: "the term it needs is a three-way product
`a[s-1]·a[s-2]·b[s-3]`", for which $y=\hat PV+\xi$ with $\xi\perp V$ does not hold.

*`replacement_survives`: true.*

*Repair — 02.4, verified in force.* Clause (a) reads "On a bed whose label is one-hop Bayes-fit,
$y=PV+\xi$ with $\mathbb E[\xi\mid V]=0$", states the reason in the finding's terms
("**Independence of $\xi$ and $V$ is not enough**: it leaves $\mathbb E[\xi]\ne0$ admissible and the
inner product then does not vanish"), and withdraws the assertion from the record's beds: "It is
**not** asserted on the record's beds: their label is a product of three features
$a_{s-1}\!\cdot\!a_{s-2}\!\cdot\!b_{s-3}$ (`READ MATHEMATICS.md:302-306 @ 99777ab`), for which
$y=\hat PV+\xi$ with $\xi\perp V$ does not hold at all, and the residual's dependence on $V$ there is
`NOT MEASURED — needs the residual-on-$V$ regression at the arena`." The Hypotheses carry the same
condition. Recorded verbatim; no further edit made.

---

**F40 · verse 02.4 · severity `repair` · mechanism P-2**

*Flaw (verbatim).* The annealed-dial replacement 02.4a prints a wrong first term in its DERIVED
$\hat P$-gradient.

*Number (verbatim).* 02.4a: "$\partial_PL=(1-\gamma)\,[M^\top r\,(MV)^\top+\gamma\,(PM)^\top r\,
(MV)^\top]$-class". From $O=(1-\gamma)PMV$ and $\delta M=\gamma MEM$:
$\delta O=(1-\gamma)[EMV+\gamma PMEMV]$, so $\langle r,\delta O\rangle$ gives
$\partial_EL=(1-\gamma)[r(MV)^\top+\gamma(PM)^\top r(MV)^\top]$. The leading $M^\top$ is spurious.

*`replacement_survives`: true.*

*Repair — 02.4a, edited.* The DERIVED block is rewritten with both steps displayed —
$\delta O=(1-\gamma)[EMV+\gamma PMEMV]$ and
$\partial_EL=(1-\gamma)[r(MV)^\top+\gamma(PM)^\top r(MV)^\top]$ — the first term printed as
$r\,(MV)^\top$, and the slip named rather than quietly dropped: "The first term is $r\,(MV)^\top$ and
**not** $M^\top r\,(MV)^\top$: the first writing carried a spurious leading $M^\top$, which the
identity $\langle r,EMV\rangle=\langle E,r(MV)^\top\rangle$ refutes (P-2, an algebraic slip in a
DERIVED block)." The hop-$\ge2$ conclusion is re-grounded on the second term alone, as the finding
directs: "$\gamma\,(PM)^\top r\,(MV)^\top$ vanishes at $\gamma=0$ and is the only hop-$\ge2$ channel,
so holding $\gamma=\gamma_A>0$ fits $\hat P$ to hops $\ge2$ before the dial is asked". 02.4a's
Hypotheses, Evidence, price ($\approx13.4$ s) and kill (released $\hat\gamma$ PINNED on $\ge6/8$) are
untouched.

---

**F41 · verse 02.4 · severity `repair` · mechanism L-EQ**

*Flaw (verbatim).* The Adam step bound is presented as DERIVED but its sketch omits both
bias-correction factors it names, and the constant it produces is a published result with no bib key.

*Number (verbatim).* Statement: "Adam's bias-corrected step satisfies
$|\Delta\theta_t|\le\alpha(1-\beta_1)/\sqrt{1-\beta_2}=3.1623\,\alpha$ ... (DERIVED:
$|m_t|\le(1-\beta_1)\sum_k\beta_1^{t-k}|g_k|$, $v_t\ge(1-\beta_2)\max_kg_k^2\beta_2^{t-k}$,
Cauchy–Schwarz on the ratio)". The two inequalities are on the *uncorrected* $m_t,v_t$; the
bias-corrected step divides by $1-\beta_1^t$ and multiplies by $\sqrt{1-\beta_2^t}$, neither of which
appears. `grep -inE "kingma" docs/references.bib` returns zero entries (`adamjan-1971-analytic` is the
only near hit).

*`replacement_survives`: true.*

*Repair — 02.4, verified in force.* The verse takes the finding's first branch and completes the
derivation rather than adding a source: the block "**The Adam step bound, with its bias corrections**"
writes the corrected step $\Delta\theta_t=-\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ with
$\hat m_t=m_t/(1-\beta_1^t)$ and $\hat v_t=v_t/(1-\beta_2^t)$, carries **both** factors through the
chain to
$|\Delta\theta_t|\le\alpha\frac{1-\beta_1}{1-\beta_1^t}\cdot\frac{\sqrt{1-\beta_2^t}}{\sqrt{1-\beta_2}}
\cdot\frac1{\sqrt{1-\beta_1^2/\beta_2}}\le3.3151\,\alpha$, and prints the $t=1$ instance the finding
asks for: "$\hat m_1=g_1$, $\hat v_1=g_1^2$, so $|\Delta\theta_1|=\alpha$ exactly, which is inside
$3.3151\,\alpha$ — the bound is checked at the one step where it is exact." The canon's working
constant is stated as the looser reading with the reason attached: "The canon uses $3.1623\,\alpha$
where the Cauchy–Schwarz factor is dropped and says so; every number below is quoted at
$3.1623\,\alpha$ and is a **lower** bound on the displacement the true step permits." No `[V-eq]`
citation is claimed and no bib key is invented. Recorded verbatim; no further edit made.

---

**F42 · verse 02.5 · severity `repair` · mechanism P-2**

*Flaw (verbatim).* The unknown count for a causal row-stochastic $P$ overcounts by the row-sum
constraints.

*Number (verbatim).* Statement (ii): "$d=16$ columns against $s(s+1)/2=2080$ unknowns".
$64\cdot65/2=2080$ is the count of lower-triangular entries; row stochasticity imposes $s=64$ equality
constraints, leaving $2080-64=2016$ free parameters.

*`replacement_survives`: true.*

*Repair — 02.5, edited.* Statement (ii) now prints both counts with the constraint between them: "one
draw supplies $24$ value columns against a causal row-stochastic $P$'s **$2{,}016$ free entries**:
$s(s+1)/2=64\cdot65/2=2{,}080$ is the count of lower-triangular entries, and row stochasticity imposes
$s=64$ equality constraints, leaving $2{,}080-64=2{,}016$ free (DERIVED)." The conclusion is re-checked
against the corrected number in the same sentence — "Either count exceeds $24$ by more than $80\times$,
so the conclusion is untouched by the correction: $24<2{,}016$" — and $2{,}016=2{,}080-64$ is carried
in Evidence as DERIVED.

---

**F43 · verse 02.5 · severity `repair` · mechanism V-22**

*Flaw (verbatim).* $d$ is used for the value-column count at a value the record assigns to
$d_{\rm model}$, and BED-S's own $d$ is never printed.

*Number (verbatim).* Statement (ii): "$d=16$ columns". `READ scripts/v15_r1.py:137 @ 99777ab`:
`S, D = 64, 24` — the record's $d$ is $24$ and $d_{\rm model}$ is $16$
(`READ docs/PLAN.md:627 @ 99777ab`; `results/v17k_r4_retake.jsonl` cell fields `d: 24, d_model: 16`).
BED-S's $d$ appears in no Hypotheses in this book.

*`replacement_survives`: true.*

*Repair — 02.5, edited.* Statement (ii) prints the bed's value width and names the confusion: "the
bed's **value width is $d=24$** — the record's own `S, D = 64, 24`
(`READ scripts/v15_r1.py:137 @ 99777ab`), journalled as the cell field `d: 24`, and **not**
$d_{\rm model}=16$, which is the model width and was the symbol the first writing wrongly spent here
(V-22, one letter carrying two registered quantities)". The spanning-set count is taken at $24$. The
Hypotheses now carry both widths with their duties separated: "**BED-S's value width $d=24$** and the
model width $d_{\rm model}=16$, printed separately and never substituted for each other — the
spanning-set count of (ii) uses $d=24$, the rank bound of (ii) uses $d_{\rm model}=16$". Evidence adds
the source line and the journal read, `RUN python -c "import json;[print(r['kind'],r['d'],r['d_model'])
for r in map(json.loads,open('results/v17k_r4_retake.jsonl')) if r.get('t')=='cell']"`, 2026-09-06,
which returns `24 16` on every cell. The rank argument, which is the one that uses $d_{\rm model}=16$,
is unchanged and still reads $\operatorname{rank}\le16<64$.

---

**F44 · verse 02.6 · severity `repair` · mechanism P-1**

*Flaw (verbatim).* The auxiliary loss weight is [ASSUMED] and the verse's whole claim is
scale-dependent in it, with no bound derived.

*Number (verbatim).* Statement (iii): "$L=L_z+\lambda L_q$, $\lambda=1$ `[ASSUMED]`". The claim "gives
$\hat P$ a hop-$\ge2$ gradient at every $\gamma$" is a statement about the *relative* magnitude of
$\partial_P L_q$ and $\partial_P L_z$; at $\lambda$ small enough it is false, and the Kill's own planted
positive is "$\lambda\to\infty$". No $\lambda$ range is derived and book 08 needs the number to build
the module.

*`replacement_survives`: true.*

*Repair — 02.6, edited.* Both branches of the required repair are taken. The floor is DERIVED as the
ratio of the two channels' gradient norms at initialisation,
$\lambda\ge\lambda_{\min}:=\|\partial_PL_z\|_F|_{t=0}/\|\partial_PL_q\|_F|_{t=0}$, "the weight at which
the committor channel's contribution to $\partial_P L$ is at least the $z$-channel's, so the hop-$\ge2$
component is not dominated by the one-hop fit", with both norms filed
`NOT MEASURED — needs the first BED-S cell (04.21), whose step-0 backward supplies both norms at $0$
GPU-s`. Until that reading exists $\lambda$ is **registered as a swept manifest field of the arm (S-01)
with its grid frozen here (M-2)**: $\lambda\in\Lambda_{\rm grid}=\{0,\tfrac14,1,4,16\}$, five arms, and
"No sentence of this book is asserted at a $\lambda$ outside that grid, and no number of book 08 may
read $\lambda=1$ as derived." The Hypotheses state that the gradient clause is conditional on
$\lambda\ge\lambda_{\min}$ and "is asserted at no realised $\lambda$ in this book". The kill's price
rises with the sweep, from $\approx13.4$ s to $4\times8\times1.680\approx53.8$ s, and the kills table
carries the new figure.

---

**F45 · verse 02.6 · severity `repair` · mechanism V-15**

*Flaw (verbatim).* The Kill's control is a planted positive where V-15 requires a planted negative.

*Number (verbatim).* Kill: "Planted positive: $\lambda\to\infty$ (committor-only) must move $\hat P$ by
$O(1)$ from the $L_z$-only arm on 8 of 8 — the loss is live." CHARTER §2: "the control that would make
the kill fire on a known-bad plant (V-15: no condemning rule without a planted negative)". The kill's
verdict is "the auxiliary loss changed nothing"; no plant is given for which it must fire.

*`replacement_survives`: true.*

*Repair — 02.6, edited.* $\lambda=0$ is added as the planted negative, in the finding's own terms:
"**Planted negative that would make the kill fire (V-15).** $\lambda=0$ against itself: the grid's own
baseline arm, re-run at a second seed block, must fire the kill on **8 of 8** — the auxiliary loss is
literally absent there, so an instrument that does not condemn $\lambda=0$ cannot condemn anything, and
the verdict 'the auxiliary loss changed nothing' is proved reachable on a known-bad plant before it is
read on $\lambda=1$." The defect is named and not softened: "The first writing carried only a planted
**positive** where CHARTER §2 requires a planted negative, so the condemning rule had no plant it must
fire on (V-15)." The plant costs nothing beyond the sweep, $\lambda=0$ being an arm the grid already
runs. The planted positive survives in an admissible form: $\lambda\to\infty$ is replaced by
$\lambda=16$, the grid's top, "which is not an arm any manifest can carry (M-2)".

---

**F46 · verse 02.7 · severity `repair` · mechanism V-25**

*Flaw (verbatim).* The universal positivity of $\hat P_{ji}$ over $j>i$ is false on the declared
boundary rows, which the architecture builds as identity rows and BED-S places before the query.

*Number (verbatim).* Statement: "because softmax with finite logits has $\hat P_{ji}>0$ for every
$j>i$". `READ docs/CEQ_SHAPE.md:295-296 @ 99777ab` (Definition 3) makes the sink, goal and $K$
constraint sets identity rows; `READ docs/PLAN.md:352 @ 99777ab` (S-10) places "the sink,
$\mathcal A_0$ and $\mathcal A_1..\mathcal A_K$ all before the query". For an absorbing $j>i$,
$\hat P_{ji}=0$ exactly and the term vanishes.

*`replacement_survives`: true.*

*Repair — 02.7, edited.* The sum is restricted to transient rows, $\sum_{j>i,\,j\in T}$, in both
displayed forms, and the universal sentence is struck by name: "The universal form — '$\hat P_{ji}>0$
for every $j>i$, because softmax with finite logits is everywhere positive' — is **false on this
architecture** and is struck (V-25, a positivity asserted where the shape builds exact zeros)", with
both citations the finding gives and the arena's count of such rows ($K+2=4$ at $K=2$): "For an
absorbing $j$, $\hat P_{ji}=0$ **exactly** — not small, zero by construction — and its term vanishes."
The Hypotheses now require "**at least one transient row $j>i$ with $P_{{\rm env},ji}>0$**" and state
that S-10 places all $K+2$ absorbing rows before the query, "so this hypothesis is not free at the
arena's geometry". The kill's reachability census is restricted to match: the admitted-draw fraction is
read on $\|\Delta z^{\rm lab}_{>i}\|_\infty$ "**restricted to transient rows $j>i$, the absorbing rows
of Definition 3 excluded from the norm** — so the fraction measures the support of the transient
sub-sum $S$ and nothing else".

---

**F47 · verse 02.7 · severity `repair` · mechanism L-VERSE**

*Flaw (verbatim).* The Kill decides a training outcome, not the sign of the DERIVED derivative the
Statement asserts, so the Statement is unfalsifiable by the instrument named.

*Number (verbatim).* Statement is DERIVED algebra:
$\partial_\gamma L|_0=-\gamma_e\|u^\top V\|^2\sum_{j>i}\hat P_{ji}P_{{\rm env},ji}<0$. Kill:
"$\Lambda\le2.7055$ on at least 6 of 8 seeds (R-21 applied to this channel): the derivative's sign is
wrong or its magnitude is below the Adam floor." A PINNED verdict is consistent with a correct negative
derivative whose magnitude sits under $3.1623\,\alpha$ per step; it cannot distinguish the two
disjuncts it names.

*`replacement_survives`: true.*

*Repair — 02.7, edited.* The kill is split exactly as the finding directs, with the reason stated in
the finding's own arithmetic: "A PINNED verdict cannot separate those two disjuncts: by 02.4's bound
the per-step displacement is at most $3.1623\,\alpha$, so a correct negative
$\partial_\gamma L|_{0^+}$ of small magnitude produces PINNED with the sign entirely right, and the
Statement's DERIVED algebra was unfalsifiable by the instrument named."
**Kill (a) — the arithmetic check**: evaluate
$S=\sum_{j>i,\,j\in T}\hat P_{ji}P_{{\rm env},ji}$ and
$\partial_\gamma L|_{0^+}=-\gamma_e(u^\top V)^2S$ numerically on **one admitted draw** of the S-12
census at the bed's own $(P_{\rm env},u,V,\gamma_e)$ and the draw's step-0 $\hat P$; it fires iff
$\partial_\gamma L|_{0^+}\ge0$; price **$0$ GPU-s**, no cell, no training; two planted negatives that
must make it fire ($u^\top V=0$ at $V\equiv\mathbb 1$, and a draw whose only $j>i$ rows are the $K+2$
absorbing rows, where $S=0$), and a planted positive that must not.
**Kill (b) — the training check**: R-21's PINNED-or-wrong-sign disjunction, unchanged in its numbers,
but now bounded in what it decides — "What (b) decides is reachability under the Adam step bound —
whether $3.1623\,\alpha T$ of displacement budget moves the dial off $\tfrac12$ far enough for R-21 to
see it — and never the sign of $\partial_\gamma L|_{0^+}$, which is (a)'s and (a)'s alone. A PINNED
verdict with (a) not fired licenses 'the sign is right and the budget is too small', not 'the
derivative is wrong'." The kills table carries both rows, (a) at $0$ GPU-s among the cheapest and (b)
at $\approx13.4$ s plus the ablation, and both route to 02.7a.

---

**F48 · verse 02.8 · severity `repair` · mechanism M-2**

*Flaw (verbatim).* The Kill leaves the two-of-three outcome undecided, and the registered card declares
that band.

*Number (verbatim).* Kill: "all three capped seeds cross $0.7071067811865476$: the cap is the fix; at
most one crosses: the mechanism is not gate divergence". Exactly two crossings has no verdict.
`READ docs/PLAN.md:926 @ 99777ab` (Bet I) registers "SPLIT band $2$ of $3$". S-52's own registered KILL
is different again: "any capped seed still above $1.0$" (`READ docs/PLAN.md:527 @ 99777ab`).

*`replacement_survives`: true.*

*Repair — 02.8, edited.* One card, one frozen kill, and the kill in force is **S-52's own as
registered**: *any capped seed still above $1.0$* eval NRMSE on the retake journal's lane. The other
two readings are printed in a three-row table beside it — S-52's *KILL* (the kill), the crossing count
out of three (the scored column, "**never** a kill") and Bet I's registered SPLIT band $2$ of $3$ (a
`SPLIT` token, "scored wrong with the sign of the optimistic half") — with the collision named rather
than merged: "the card S-52 registers a **different** kill again ... and a card may carry one frozen
kill, not two." The undecided band is closed by printing it: "a capped seed landing in
$[0.7071067811865476,\,1.0]$ **does not cross** and **does not fire the kill** — it is a non-crossing
that leaves the verse standing, and the ledger scores it in the SPLIT band. Three crossings score
`HOLDS`; two score `SPLIT`; at most one, with no seed above $1.0$, scores `COUNTER` while the kill
still does not fire; any seed above $1.0$ fires the kill whatever the crossing count." A planted
negative is added on the same column: the uncapped cells of the same three seeds read
$1.1522795055459243$, $1.1133392329955414$, $1.1489267727154717$ in that lane, all above $1.0$, so the
instrument condemns where the record already shows failure. The If-killed trigger is re-headed "any
capped seed still above $1.0$, S-52's registered *KILL*", and the kills-table row is rewritten to the
same number.

---

**What batch 3 changes in the book's own numbers.** Three verses gain or move a price or a count:
02.5's spanning-set line now reads $24$ value columns against $2{,}016$ free entries (was $16$ against
$2{,}080$), 02.6's kill costs $\approx53.8$ s instead of $\approx13.4$ s because $\lambda$ is swept on a
five-point grid, and 02.7 carries **two** kills where it carried one, the cheaper of them at $0$ GPU-s.
02.8's kill is a different number from the one the first writing froze — "above $1.0$" replaces "at
most one crosses" — and the crossing count is demoted from a kill to a scored column.

**What stays owed after batch 3.** $\lambda_{\min}$ is a ratio of two gradient norms that no cell has
printed: until 04.21's step-0 backward supplies them, 02.6's gradient clause is conditional and the
grid is the whole of what is registered. 02.7's kill (a) is decidable at $0$ GPU-s on the S-12 census,
but no admitted draw has been generated, so it is decidable and undecided. The transient/absorbing
split of 02.7's census changes an instrument's reading rule and is owed to book 04 as a field
definition; this book cannot edit it there. No finding of this batch is left OPEN.

---

MARS's findings, repair round 1, **batch 4 of 4**, each verbatim as filed, with the repair beside it.
Nothing is softened; no verse is deleted to escape a finding. Every finding of this batch carries
`replacement_survives: true`, so no pre-written replacement of this book is re-derived here; what is
repaired is the verse's own arithmetic, its evidence class, its lane and its reachability.

---

**F14 · verse 02.9 · severity `repair` · mechanism P-4**

*Flaw (verbatim).* The tied-loop count $p_L=p_1-1$ is asserted with no derivation of the $-1$.

*Number (verbatim).* Statement: "With weight tying (the looped block $z^{(t+1)}=V+\gamma Pz^{(t)}$)
$p_L=p_1-1$ and $R_{T2}=1.0$". $p_1=2d_{\rm model}d_x+1=4{,}097$ where the $+1$ is $\gamma$; the tied
loop still carries $\gamma$, so the $-1$ is unexplained. $4{,}096/4{,}097=0.99976$, printed as $1.0$.

*`replacement_survives`: true.*

*Repair — 02.9, both halves of the required repair taken.* The tied count is now **derived entry by
entry** and the equality is written exactly. A new paragraph, "The tied count, derived entry by entry,
and the $-1$ struck", enumerates what the loop reads at every pass: the per-layer block
$p_{\rm layer}=W_Q\,256+W_K\,256+\text{MLP}(2{,}176+2{,}064)=4{,}752$, applied once to produce $\hat P$
and reused unchanged at each of the $L$ passes; the readout $p_{\rm shared}=17$, applied once after the
last pass; and the dial $\gamma$ — the $+1$ of $p_1=2d_{\rm model}d_x+1=4{,}097$ at $d_x=128$ — which
sits **inside the recursion itself**, $z^{(t+1)}=V+\gamma\hat Pz^{(t)}$, so the loop cannot drop it
without deleting its own update rule. Hence $p_L=p_1=4{,}769$ at every $L\ge1$ on the arena arm and
$p_L=p_1=4{,}097$ on BED-S's token$\to P$ map, each read at its own geometry and the two never mixed.
$p_L=p_1-1$ is **struck** with the finding's own reason printed: it removed one parameter and named
none, and the only scalar it could have named is $\gamma$, which the tied loop carries. $R_{T2}=1.0$
is struck with it, and its arithmetic is printed rather than dropped — $4{,}096/4{,}097=0.99976$ was a
ratio rounded up to the value it was meant to establish, under a symbol 02.1 had already retired for
naming the reciprocal of this book's orientation. The Evidence gains the DERIVED line for $p_L=p_1$;
the Mechanism gains P-4 for both the underived $-1$ and the rounded $0.99976$.

---

**F15 · verse 02.10 · severity `repair` · mechanism P-2**

*Flaw (verbatim).* The RUN pointer names a journal field that does not exist.

*Number (verbatim).* Evidence: "`RUN results/v17k_r4_retake.jsonl` (`t=wall`)".
`RUN python -c "print(sorted(json.loads(line).keys()))"` on the first `cell` row (2026-09-06) returns no
`wall` field; the per-cell time field is `secs`, and `t` is the row-type field whose value is `cell`.
The quoted numbers do check out against `secs`: retake means $16.161$ s (`arm_smprime`) and $1.681$ s
(`softmax`), ratio $9.614$.

*`replacement_survives`: true.*

*Repair — 02.10, Evidence.* The pointer is replaced by a literal command on the field the journal
carries, `secs`, over the rows whose row-type field reads `t="cell"`, and the two means and the ratio
are printed to the digits the finding gives: $16.161$ s (`arm_smprime`) against $1.681$ s (`softmax`)
at $150$ steps, ratio $9.614$, per-op floor class (C17). The old pointer is **struck in place and
named**: no cell row carries a field called `wall`, `t` is the row-type field, and the one `t="wall"`
row of that journal is a whole-run summary carrying `secs_per_run_by_arm` and not a per-cell time — a
number pointed at a field that does not exist is a number with no provenance. The Mechanism gains P-2
on exactly that count. The figures are unchanged by the correction, and the verse says so.

---

**F16 · verse 02.11 · severity `repair` · mechanism Ruling 2a**

*Flaw (verbatim).* The control's difference is a v17k_r4 reading and the sd it is compared against is a
v15_r1 reading from a different bed, device and value width.

*Number (verbatim).* Control: "`softmax` seed 0 against seed 1,
$|0.9734002295-0.9387956332|=0.034605>0.011824$". The two values are `results/v17k_r4_floor.jsonl`
(`device: cuda`, `d: 24`, task `e3_t2`); $0.011824$ is the seed sd of `RUN results/v15_r1.txt` §4
(`device=cpu`). The v17k_r4 lane's own softmax seed sd over its eight retake cells is $0.011452$
(`RUN python -c "statistics.stdev(...)"`, 2026-09-06).

*`replacement_survives`: true.*

*Repair — 02.11, Statement and Kill control.* The control is re-read **entirely inside one lane**: the
difference stays $|0.9734002295-0.9387956332|=0.034605$ on `results/v17k_r4_floor.jsonl` (`device:
cuda`, `d: 24`, task `e3_t2`) and the threshold becomes the v17k_r4 lane's **own** softmax seed sd,
$0.011452$ at $N=8$, printed with its literal command over the retake journal's eight `softmax` cells.
The control still fires, $0.034605>0.011452$, and the constant no longer crosses a device boundary. The
cross-lane comparison is **struck under Ruling 2a** and named struck: a control whose difference and
whose threshold came from two different beds, devices and value widths. The Statement's resolution
clause is corrected on the same terms — the lane's own $0.011452$ is the sd in force, and the R1
figures $0.011824$ (softmax) and $0.253673$ (`arm_pl`), `device=cpu`, are demoted to a labelled
cross-lane note that no threshold of the verse reads. The Mechanism gains Ruling 2a and V-22.

---

**F17 · verse 02.12 · severity `repair` · mechanism V-11**

*Flaw (verbatim).* The Kill names no bed, no draw, no seed rule and no $N$, so it is not decidable as
registered.

*Number (verbatim).* Kill: "A later bed on which R-21 reads MOVED on at least 6 of 8 under the boundary
null". CHARTER §2's reachability rule: "Every kill names the draw, seed rule and $N$ at which it is
decidable, and the realised-sd clause that makes its $\mathrm{MDE}_N$ honest". "A later bed" is
unregistered; no such bed exists in books 04 or 09 with an R-21 row.

*`replacement_survives`: true.*

*Repair — 02.12, Kill, Evidence and Mechanism; 02.13 and the kills table follow it.* "A later bed" is
**struck** as an unregistered object, with CHARTER §2's rule quoted against it. The kill is re-frozen
on **exactly two registered candidate beds and no third**, each named by verse id with its seed rule,
its $N$ and its realised-sd clause printed in a table: **BED-J** (`04_BEDS_AND_INSTRUMENTS.md` 04.6,
protocol 04.9, order 04.21) — 04.9 clause 1's paired-seed rule ($N\ge8$ distinct deduplicated seeds,
one thread lane, byte-identical draws per seed, train at `seed`, eval at `seed+12345`, init after
`torch.manual_seed(seed)`, the pairing asserted as a relation), $N=8$ minimum repriced by clause 6,
$\mathrm{MDE}_N$ at clause 4's realised $\sigma_d$ with $\sigma_d\ge0.109199$ repricing $N$; and
**BED-C-N**, the chess witness (04.7, protocol `09_CHESS_AND_MARKETS.md` 09.11) — seeds $0$–$7$
deduplicated (`REQUIRED_SEEDS = 8`), byte-identical draws, arms interleaved, run order randomised and
journalled, $N=8$ with $N=16$ at $\approx68$ s per pair when K-P fires at realised paired sd
$\ge2.18\times0.050146$. The verdict itself is frozen numerically where the first writing left a word:
MOVED is $\Lambda>\ln4096=8.318$, PINNED is $\Lambda\le2.7055$ under the boundary null, and the band
$(2.7055,\,8.318]$ is neither and fires nothing. The finding's second half — "no such bed exists in
books 04 or 09 with an R-21 row" — is answered on the manifest and not by assertion: 04.15 declares
`gamma` (init, final $\hat\gamma$, $\Lambda$, verdict) a **required field of every journalled cell** and
calls a missing declared field a refusal, so the R-21 row exists on any cell of either bed by
construction, while $\Lambda$ is a parameter verdict and not a contrast, so neither BED-J's D-APPROX
gate nor BED-C-N's digest gate blocks the reading. What does not exist is the cell, and the verse says
so: `NOT MEASURED — needs the first BED-J cell (04.21) or the first BED-C-N cell (09.11)` — registered
and undecided, never unreachable. Both plants are printed (S-61's $\gamma_{\rm env}=0$ PINNED 8 of 8
and $\gamma_{\rm env}=0.6$ MOVED 8 of 8) and both beds inherit 02.7's frozen registration
$|\gamma_{\rm env}-\tfrac12|\ge0.2$, so a cell at $\gamma_{\rm env}\in(0.3,0.7)$ is VOID for the kill.
The Mechanism gains V-11, P-4 and M-2. **02.13 is edited to match** — its Statement now names the same
two beds and the same $\Lambda>8.318$, and its Evidence reads $A$ against that bed's own realised seed
sd (04.9 clause 4 on BED-J, K-P on BED-C-N) rather than this book's arena placeholder — and the kills
table rows for 02.12 and 02.13 are rewritten to the named beds, seed rules and $N$.

---

**F18 · verse 02.7 · severity `note` · mechanism P-2**

*Flaw (verbatim).* $(u^\top V)^2$ is written as a scalar square where $u^\top V$ is a $d$-vector.

*Number (verbatim).* $V\in\mathbb R^{s\times d}$ with $d\ge16$ at the arena
(`results/v17k_r4_retake.jsonl` cell field `d: 24`), so $u^\top V\in\mathbb R^d$ and the displayed
$-\gamma_e(u^\top V)^2\sum\dots$ must read $-\gamma_e\|u^\top V\|_2^2\sum\dots$.

*`replacement_survives`: true.*

*Repair — 02.7, applied at all three sites and not only the one the finding displays.* The
corner-algebra display now reads
$\partial_\gamma L|_{\gamma=0}=-\sum_{j>i,\,j\in T}\hat P_{ji}\langle u^\top V,\Delta z^{\rm lab}_j\rangle
=-\gamma_e\|u^\top V\|_2^2\sum_{j>i,\,j\in T}\hat P_{ji}P_{{\rm env},ji}<0$, the pairing written as the
inner product it is; kill (a)'s arithmetic reads
$\partial_\gamma L|_{0^+}=-\gamma_e\|u^\top V\|_2^2\,S$; and the kills-table row carries the same norm.
A clause is added to the Statement naming the type error rather than silently fixing it:
$V\in\mathbb R^{s\times d}$ at the bed's journalled value width `d: 24`, so $u^\top V\in\mathbb R^d$ and
$\Delta z^{\rm lab}_j\in\mathbb R^d$; the scalar writing is **struck** as P-2, $\|u^\top V\|_2^2\ge0$
carries the sign conclusion unchanged, and the hypothesis $u^\top V\ne0$ is read as "the $d$-vector is
not the zero vector", so the $V\equiv\mathbb 1$ plant of both kills is untouched. The Mechanism gains
P-2 at $d=24$.

---

**What batch 4 changes in the book's own numbers.** No kill threshold of this book moves except one,
and it moves toward its own lane: 02.11's control threshold is now the v17k_r4 lane's own softmax seed
sd $0.011452$ at $N=8$ in place of the CPU lane's $0.011824$, and the control fires either way. Two
numbers gain a derivation rather than change value — $p_L=p_1$ at every $L$ (arena $4{,}769$, BED-S map
$4{,}097$), with the struck $-1$ and the struck $0.99976\to1.0$ printed — and one gains a threshold
where it had a word: 02.12's MOVED is $\Lambda>\ln4096=8.318$. 02.10's two times keep their values,
$16.161$ s and $1.681$ s at ratio $9.614$, and change evidence class from a pointer at a non-existent
field to a literal command on `secs`. 02.7's algebra changes type, not sign: $\|u^\top V\|_2^2$ for
$(u^\top V)^2$ at $d=24$.

**What stays owed after batch 4.** Neither bed 02.12 now names has a cell: BED-J's first cell (04.6,
04.21) and BED-C-N's first cell (09.11) are both `NOT MEASURED`, so 02.12's kill is registered,
decidable and undecided, and 02.13's ablation gap $A$ has no value at all. Both beds' $\mathrm{MDE}_N$
waits on the same $\sigma_d$ from 04.21 step (5) that eight kills of book 04 wait on. The
boundary-null critical values $2.7055$ and $\ln4096=8.318$ still carry no `references.bib` citation and
that debt stands (Limits). No finding of this batch is left OPEN.
