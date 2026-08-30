# V12 — the competing prediction on VGPE, filed before any V12 cell has run

**Written 2026-08-30, by the IRENE seat (the alternative forecaster).** Verified
before writing: `scale/vgpe.py`, `THEORY_V12_VGPE.md`, `V12_MORIARTY_CONTROLS.md`
do not exist in this working tree at time of writing (only `PRIOR_ART.md` among
the round's concurrent files has landed), and no grep of the tree for
`VGPE|verb-gauge|verb_gauge` returns a hit outside this file. No cell of the
five-arm-by-two-bed design has run. Every number below is a **prediction**,
never a measurement, and is marked as such.

---

## 0. The lead finding — the two beds do NOT share one label shape

The task brief's strongest line to evaluate was: *"if the beds' labels are
still scalar point predictions at a single position, D-1 applies unchanged and
the position code cannot matter, so all five arms tie."* I read both label
builders before predicting anything else, because this determines whether the
round can be informative at all.

**IMPACT bed — the fixed-position case, confirmed.** `impact_oracle` returns
one scalar per example, `y[b] = r_vec[query].float()` (`scale/impact.py:690`,
inside the loop at `:686-691`), where `query` is computed once per corpus draw
by `_impact_query_node(graph)` (`scale/impact.py:581`) and the **same graph is
reused for every one of the `n` examples in the batch** — only the news vector
varies per example (`scale/impact.py:585-596`, `_make_impact_batch_unsafe`).
So within one task instance there is exactly one fixed readout coordinate,
identical in shape to `m3_quintuple`'s `[:, s-1]` (`scale/m3_quintuple.py:483`,
`return out if self.vector_readout else out[:, s - 1]` — this line has moved
since `MISTAKES.md:613` cited it at `:311`; the current, verified location is
`:482-483`, and the repo has since added an opt-in `vector_readout` path that
is off by default, `scale/m3_quintuple.py:390,672`). **D-1's argument
transfers to the impact bed cleanly**: a scalar read from one fixed location,
regardless of how many hops of graph propagation feed it.

**Dirichlet / E4′ bed — NOT the fixed-position case.** Two label-bearing
objects exist here and they are not the same shape:

* `scale/e4_harmonic.py`'s `fixed_point`/`hop_reading` (`:184-200`) produce a
  **vector** over every transient node of one graph, and `measure()`
  (`:282-320`) treats each node as a separate regression example for the
  decoder-gate control — not a single per-instance readout at all.
* `scale/negation_scope.py`'s `e4prime_oracle` (`:1015-1052`) is scalar per
  example, but the queried pair `(qi, qj)` is **drawn fresh per example**
  inside a shared batch (`make_e4prime_batch`'s `torch.randint` calls at
  `:1001-1002`, marked into the tensor at `:1006-1007`, and the oracle's own
  docstring: *"the queried pair is PER EXAMPLE here, which is what lets one
  shared substrate carry a varying label,"* `:1022-1023`). The
  readout location is content-addressed via the `CH_QA`/`CH_QB` marker
  channels, not fixed across the corpus the way `s-1` and IMPACT's `query`
  are.

**This is the single most important thing in this filing.** D-1's cited
theorem (`arXiv:2410.01537`, one softmax layer Bayes-optimal for a scalar
target read from a fixed location) is stated for a fixed location. It applies
to the impact bed without qualification. It does **not** mechanically cover
the e4prime shape, where the location itself is the thing that varies example
to example — that is a content-addressed retrieval task, which is the regime
attention mechanisms (softmax or gauge-based) were built to be good at, and
no theorem in this repo's citation trail says a fixed-location proof extends
to it. Whichever object `THEORY_V12_VGPE.md` binds as "the Dirichlet bed"
determines whether the round has one informative comparison or zero — I could
not resolve which of the two it will be, since neither `THEORY_V12_VGPE.md`
nor `V12_MORIARTY_CONTROLS.md` exists yet, so §2 below carries both branches
explicitly rather than picking one and hiding the fork.

---

## 1. The track record read before predicting

* **Headline null, both CI families verified.** `settled − twin` at `k=8`:
  point estimate `−0.002959`. The Monte-Carlo `B=10000` reading is CI
  `[−0.048587, +0.031557]` (`results/m3_quintuple.txt:51`). The exact
  finite-population bootstrap over 5 seeds (`5**5 = 3125` resamples, 126
  distinct values) is CI `[−0.042903167939657406, +0.031557]`
  (`ceq/hf_artifact/README.md:84`, also printed at `CHECKLIST.md:1167`, verified
  resolving; `ceq/hf_artifact/README.md:84` additionally claims `STATE.md:19`
  carries the same pair, but as read in this working tree `STATE.md:19` is a
  different row entirely — the citation has drifted with the file, and I am
  not repeating it as verified). Both are legitimate readings of the same seeds; they are not
  in conflict, they are two different bootstrap constructions, and the task
  brief's `[−0.042903, +0.031557]` is the exact-combinatorial one. Verdict on
  both: **NO DIFFERENCE**.
* **D-1, in full, verified at `MISTAKES.md:609-632`.** *"Every label here is a
  scalar point prediction at position `s-1`... One softmax layer is provably
  Bayes-optimal on exactly that shape, where linear attention provably cannot
  be"* (`:612-615`). The rule it draws: *"state the regime in which the
  baseline is optimal and check the task is not inside it"* (`:625-626`).
* **The floor, pre-registered before the k=8 run.** *"With five seeds, a real
  settled-vs-twin gap below roughly 0.05 NRMSE will read `NO DIFFERENCE`
  whether or not it is real"* (`M3_QUINTUPLE_PREREGISTERED_READING.md:88-90`),
  measured from a paired delta SD of `0.056889` against `0.057089` unpaired
  (`CHECKLIST.md:820`, `M3_QUINTUPLE_PREREGISTERED_READING.md:83-84`).
* **The counter-precedent D-1 does not resolve for me.** On the same k=8
  ladder, `twin − softmax` is `+0.111396` CI `[+0.100873, +0.121920]` and
  `settled − softmax` is `+0.108437` CI `[+0.066232, +0.147110]`
  (`results/m3_quintuple.txt:54-55`) — both comfortably above the 0.05 floor,
  both excluding zero. So on this exact task family, a **mechanism**
  difference (pivot-routed vs plain softmax) produces a real, resolvable gap,
  while a **within-pivot-family** difference (settled vs twin) does not. D-1
  explains the second fact; it is silent on the first, which is architecture,
  not position code — I read this as evidence that the *readout shape* being
  fixed-scalar does not, by itself, guarantee every architectural knob ties.
  It guarantees that comparisons **between arms racing softmax's own proven
  optimum** tie. `twin` is not softmax; VGPE's four arms are not racing that
  optimum against each other, they are testing something D-1 never
  addressed — the position code layered on top of an architecture that
  already beat softmax by a resolvable margin.

---

## 2. Filed predictions, arm × bed, as numbers

All deltas below are `NRMSE(baseline arm) − NRMSE(named arm)`, positive means
the named arm wins, matching the sign convention already in use at
`results/m3_quintuple.txt:49`. Floor is 0.05 throughout (§1).

### 2a. IMPACT bed (fixed query position, D-1 applies cleanly)

| contrast | predicted delta | predicted 95% CI | verdict |
|---|---|---|---|
| `twin+VGPE` − `twin+RoPE` | `+0.00` to `+0.02` | includes 0 | **NO DIFFERENCE** |
| `twin+VGPE-abelian` − `twin+VGPE` | `0.00` to `+0.01` | includes 0 | **NO DIFFERENCE** |
| `twin+VGPE` − `twin+shuffled-gauge` | `+0.04` to `+0.10` | excludes 0 | **VGPE WINS** (real) |
| `twin+RoPE` − `softmax+RoPE` | `+0.06` to `+0.14` | excludes 0 | **TWIN FAMILY WINS** (real) |

Reasoning: the query node is a single fixed coordinate for the whole corpus
draw, and every token already carries `CH_A_DEG`/`CH_SEED`/`CH_RHO` — graph
structure the model can read locally without any help from a position code
(`scale/impact.py:713-728`, `impact_features`; `scale/impact.py:730-748`,
`impact_planted_features`).
VGPE's non-commutative routing duplicates information the twin architecture's
own attention already has access to through these channels — so RoPE and
VGPE tie, and forcing the generators abelian removes a property (order
sensitivity) that was never load-bearing here, so it also ties. Shuffled-gauge
is different in kind from the abelian ablation: it does not *remove*
information, it *injects a wrong rotation* into every QK pair that uses a
misassigned action type, which is an active corruption of attention scores
rather than a flattening of them — I expect it to cost real NRMSE, not just
fail to help. `twin` vs `softmax+RoPE` inherits the architecture-level gap
already measured on a harder-than-this-bed task (`+0.108` to `+0.111`,
§1) — I am predicting a smaller but still-resolvable gap because IMPACT's
resolvent is a genuinely deeper propagation task than negation_scope's, which
could go either way on magnitude; the floor-crossing direction is what I
commit to, not the exact number.

### 2b. Dirichlet / E4′ bed — two branches, because I could not resolve which the round ships

**Branch (i) — if the trained label is `e4prime`'s per-example varying-query
connectivity oracle** (`scale/negation_scope.py:1015`, my read of "instance
graph" in the brief, since the Rips graph and marked pair are per-instance
rather than planted-and-fixed the way IMPACT's `B` is):

| contrast | predicted delta | predicted 95% CI | verdict |
|---|---|---|---|
| `twin+VGPE` − `twin+RoPE` | `+0.03` to `+0.08` | **excludes 0**, 35% chance | **VGPE WINS** (real) |
| `twin+VGPE` − `twin+RoPE` | `0.00` to `+0.02` | includes 0, 65% chance | **NO DIFFERENCE** |
| `twin+VGPE-abelian` − `twin+VGPE` | `0.00` to `+0.02` | includes 0 | **NO DIFFERENCE** |
| `twin+RoPE` − `softmax+RoPE` | `+0.05` to `+0.13` | excludes 0 | **TWIN FAMILY WINS** (real) |

This is the one row in the whole filing where I do not have a single-mode
prediction, stated as an explicit split rather than hidden inside a wide
interval: a path-ordered, non-commutative encoding is the most direct
architectural match for "route from a per-instance marked node to a
per-instance marked node along a per-instance graph," which is exactly the
job VGPE's authors say it exists to do, and it is the only place in this
filing I give real odds to the contract's own thesis. I still weight the tie
higher, because `twin`'s own pivot/routing mechanism has already shown (§1)
that it can out-perform plain softmax on multi-hop tasks without any
position-code help at all — the same free-information argument that ties
IMPACT could tie this too, if the twin architecture routes information
through learned attention weights rather than through relative position.

**Branch (ii) — if the trained label is `e4_harmonic`'s fixed-point /
absorbing-chain coordinate** (`scale/e4_harmonic.py:184-200`), read at one
fixed coordinate the way `STATE.md:9,19`'s PHASE R8 X17 language suggests:
identical predictions to §2a, IMPACT bed, for the same reason — one fixed
scalar readout, D-1 applies cleanly, all four twin-family arms tie, twin
family beats softmax+RoPE by a real margin.

**I cannot commit to which branch is real without reading
`THEORY_V12_VGPE.md`, which does not exist yet.** Whichever branch it turns
out to be must be checked against this file before either half is scored —
the formula and the reasoning are the commitment, not a guess at which file
gets written.

---

## 3. What would falsify each piece, stated binary

1. Any `twin`-family pairwise contrast (RoPE vs VGPE vs abelian vs shuffled)
   on the **impact** bed with 95% CI excluding zero and `|delta| ≥ 0.05` in a
   direction other than "shuffled-gauge loses" — kills §2a rows 1, 2, or the
   direction of row 3.
2. Shuffled-gauge tying VGPE on the impact bed (CI including zero,
   `|delta| < 0.03`) — kills §2a row 3's "real difference" claim and would
   mean D-1 reaches further than I think: not just within-family readout
   comparisons but even a deliberately-corrupted position code ties, which is
   the strongest possible version of "the round is uninformative."
3. `twin+RoPE` failing to beat `softmax+RoPE` by at least the 0.05 floor with
   CI excluding zero on **both** beds — kills the one claim in this filing
   that does not depend on VGPE at all (the architecture-not-position-code
   claim) and would mean the twin-vs-softmax precedent from the k8 ladder does
   not generalize to either new bed.
4. On the Dirichlet bed, `twin+VGPE` beating `twin+RoPE` by `≥0.03` with CI
   excluding zero — this **confirms** the 35%-weighted branch in §2b rather
   than falsifying anything; I am flagging it here so it is not later reported
   as a surprise I failed to predict. It is the one outcome in this filing I
   already priced in as live.
5. A demonstration that IMPACT's query node is **not** fixed across a batch —
   i.e., that `_impact_query_node` is re-derived per example rather than once
   per graph draw. §0's "impact bed matches D-1 cleanly" claim rests on
   `scale/impact.py:576-596` calling `build_impact_graph` and
   `_impact_query_node` exactly once per batch; if the shipped VGPE harness
   redraws the graph per example instead of per corpus, every impact-bed
   prediction in §2a is void and must be rederived.

---

## 4. K1–K6 — no registry exists yet, so this section is explicitly a guess

Grepped `CONTRACT.md`, `CHECKLIST.md`, `STATE.md`, `BOARD.md`, `DONE.md` for a
V12/VGPE-specific K1–K6 registry: **none exists in this working tree.** The
only `K1`–`K6` hits in the repo belong to an unrelated earlier round (the X4
displacement/dual-slope ladder, e.g. `CHECKLIST.md:381,427,457,512,582`), and
reusing that numbering here would be citing the wrong gate. Everything in this
section is therefore my own reconstruction from the task brief's prose, not a
sourced repo fact, and is labeled as a guess rather than a reading.

* **K1, "RoPE-recovery bind," prior on passing clean on first
  implementation: 30–35%.** Grounds, all sourced: (a) the repo's own currently
  shipped run declares a pivot-indexing bug as a "DECLARED DEVIATION, not a
  failure" rather than a clean pass, twice, in its most recent published
  reading (`results/m3_quintuple.txt:11,18`); (b) the nearest analogous
  numbered bind in this repo's history, `K1`'s dual-slope clause, took three
  recorded correction passes before it resolved (`CHECKLIST.md:427` "NOT
  EVALUABLE," `:457` "NOW EVALUABLE," `:512` "contaminated," `:582`
  "RESOLVED AND FAILED"); (c) `D-4` (`MISTAKES.md`, registration without
  admission) and `D-5` (declared zero without a movement test) are both
  instances of a first implementation shipping before its own gate could
  actually check it; (d) the branch's own recent commit log records exactly
  this pattern already happening in the current round's setup work — "Correct
  the E4prime bridge-edge control and journal its gate battery" (commit
  `6629196`) is a correction to a control that had already been journalled
  once. A Cayley-transform reduction to RoPE's fixed rotation is new code with
  no prior art in this repo to reuse; I bet against it clearing on the first
  pass, not because the mathematics is wrong but because this repo's own
  history says first passes on new binds usually are not.
* **Which of K1–K6 I expect to fire**, mapped onto the task brief's own named
  checks since I have no other source: the impact-bed tie (§2a rows 1–2), the
  "abelian ties VGPE" line the brief itself supplies as its worked example,
  and the twin-beats-softmax architecture gap (§2a row 4) are the three I
  expect to fire as stated. The shuffled-gauge-degrades check (§2a row 3) is
  the one I expect to fire in the "real difference" direction rather than tie.
  Whatever gate number the round's own authors attach to "all five arms tie
  everywhere" (the brief's row-Ω-equivalent outcome), I expect it **not** to
  fire on the impact bed and to be genuinely live on the Dirichlet bed per
  §2b's split — that uncertainty is the one I am not willing to collapse into
  a single number.

---

## 5. Limits, collected once

Every number in §2 is conditional on the two beds being implemented the way I
read them from the currently shipped `scale/impact.py` and
`scale/negation_scope.py`/`scale/e4_harmonic.py` — none of which are the V12
round's own files, since those do not exist yet. If `THEORY_V12_VGPE.md`
builds either bed's label differently (a different query rule for IMPACT, a
different choice between the two Dirichlet-bed shapes in §2b), the formulas
in §0 and the reasoning in §2 must be re-checked against the actual builder
before this file is scored, the same condition R9's filing placed on itself
(`R9_IRENE_PREDICTION.md:302-311`).

The magnitudes in §2a row 4 and §2b's "twin family wins" rows are
extrapolated from a different task family (`negation_scope` at `k=8`) onto two
tasks that have never been run under a `twin`/`softmax` attention contrast
before at time of writing. The **direction** (twin family clears the floor
against softmax) is the commitment; the **exact size** is not, and I have
deliberately left those two rows as wide, floor-anchored ranges rather than
point estimates for that reason.

§2b's 35/65 split is a genuine, stated uncertainty, not a hedge dressed as a
number — I would be equally unsurprised by either outcome and said so before
either branch of `THEORY_V12_VGPE.md` was written. If the round's authors
read this as refusing to predict, the correct response is to treat the tie
(65%) as my filed point prediction and the 35% branch as the specific,
named way I already expect to be wrong, which is falsifier 4 in §3.

No cell was trained to produce any number above. Everything is either `READ`
against a cited, verified line (§0, §1) or a prediction stated as such (§2–4).
The adversarial check on my own strongest claim, §0: would the "impact bed
matches D-1" argument survive `_impact_query_node` being redrawn per example
instead of per batch? No — that is falsifier 5, and it is the cheapest way
for this filing to be wrong about the one bed I am most confident in.
