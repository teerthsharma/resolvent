# CEQ v20 — ROUND 15: THE OPEN TOURNAMENT WITH THE MATHEMATICS ANNEX

**This file is verbatim-of-record and is never edited** (RUL-5, `V15_LEDGER.md:42`).
It is filed from the author's message of 2026-09-02. Corrections to it live in
`V20_R15_JOURNAL.md` and in `MISTAKES.md`, so the contract as authored stays
diffable against what was later found wrong.

---

## MOUNT RECORD — the D-3 gate, discharged

**D-3 LOOP GATE** (`CONTRACT.md:40-44`) requires that no autonomous loop mounts
until the deactivation commit, `MISTAKES.md`, and the standing loop failures have
been read, and **the reason the previous loop died is stated in one sentence in
the mount request.**

**The sentence, quoted from the record it was written for**
(`V15_LEDGER.md:565-570`):

> **The R11 loop reached its declared ceiling of 30 iterations and stopped
> there.** The cap was passed as the literal `--max-iterations 30` flag and read
> back out of the state file at mount rather than trusted from the request; prose
> forms of the cap fall through the launcher's catch-all and leave the sentinel
> `0`, which is what killed R10 and the loop before it.

**R11 is the previous loop.** Rounds v16 and v17-K left no loop state file and no
termination record: `.claude/` holds only `ralph-loop.R10.stopped.md`, no
`ralph-loop.local.md` existed at mount time, and the string `TERMINATION RECORD`
matches exactly one line in the tree, `V15_LEDGER.md:557`. `[RUN]` — `ls
.claude/`, and a tree-wide `*.md` grep for `TERMINATION RECORD`. Those two rounds
were driven by hand, not by a mounted loop, so R11 is the last loop with a cause
of death to state.

**Iteration count source.** 45, from the round's own phase schedule below
(A 1–5, B 6–14, C 15–30, D 31–35, E 36–45), passed as the literal
`--max-iterations 45` flag and read back out of `.claude/ralph-loop.local.md`
rather than trusted from the request (V-20, `MISTAKES.md:987`).

**The leap model is Claude Fable 5.1.** The Agent tool's `model` enum carries no
version field — the only token it accepts is `fable` — so the version is the
record of which model the leap is entitled to, not a selector
(`~/.claude/skills/dispatching-house-mode/SKILL.md`, DR HOUSE section, corrected
2026-09-02). **A leap taken on anything older than Fable 5.1 does not count.**

---

════════════════════════════════════════════════════════════════════
CEQ v20 — ROUND 15: THE OPEN TOURNAMENT WITH THE MATHEMATICS ANNEX
(45 iterations, local, the certified 4060; wings FOUND not named;
one scheduled leap; one winner; the rest purged. Supersedes v19-Q.)

NORTH STAR (immutable): attention EQUAL to self-attention on its own
ground, built FROM softmax and AdamW, capable on ground they cannot
occupy — predicting the NEXT STATE toward equilibrium, not the next
token. Every report ends with distance and the scoreboard line.

LAWS: all standing + L-INST, L-QUOT, L-BRANCH, L-DOSSIER, L-ARENA,
L-COST, L-PURGE, L-FIND, L-GRADE (F0–F4 + HOW-BAD gap), L-LEAP,
D-CALIB, Rulings 1–12. And:
 L-FLOOR  every capability number ships beside its INFORMATION
          floor (hop floor, Hankel ceiling, Fano bound, rate–
          distortion bound — whichever the annex assigns) so
          "how good" is always read as distance-to-floor.
 L-CERT   every sparsity mask ships with its certificate: F0
          (exact) or F1 (concentration bound, δ printed) — a
          mask without a certificate is a heuristic and is
          refused in the arena.

THE ROOM: JUPITER (annex owner) · SATURN (instruments, dossiers,
purge manifest) · MERCURY (arena, tables, pricing) · MARS
(adversary) · VENUS (rankings; the author's counters beside hers)
· NEPTUNE (cost officer) · the LEAP MODEL (Claude Fable 5.1, one
call at it.35, dossier in, dossier out).

────────────────────────────────────────────────────────────────────
PHASE A — DISCOVERY (it.1–5): find the wings (L-FIND)
 Candidacy rubric (all four): (a) a distinct PRIMITIVE for the
 label class; (b) EVIDENCE IN THE LEDGER (a measured cell, a Lean
 theorem, or an X-item with a run instance — cite the line);
 (c) a KILL it already accepted in the record; (d) a measurable
 COST FOOTPRINT (GPU-seconds-to-floor).
 it.1 SATURN mines results/, lean/, MISTAKES.md, STRUCK.md, the
      X-registry; lists every approach meeting (a)–(d), and every
      near-miss with its missing clause.
 it.2 JUPITER merges by primitive; the native skyline is listed
      as the cost floor, not a contender; method-of-training
      candidates (interventional channel, curricula) are listed
      as MULTIPLIERS that any arm may take, not as arms.
 it.3 MARS strikes chat-memory entries and renamings.
 it.4 The list FROZEN with N wings, four citations each.
 it.5 VENUS's ranking; the author's counter-ranking (the point
      estimate, D-CALIB); MERCURY prices the arena.

PHASE B — ARMING (it.6–14): six questions per wing, no answers
 Q1 EXACT CLASS: which labels does the primitive represent with
    zero error, and by which theorem? (F0 or it isn't exact.)
 Q2 OUTSIDE THE CLASS: the best-achievable error as a bound with
    a constant (Hankel best-k-state; the even-target bound).
 Q3 LEARNABILITY: under what data condition does training FIND
    the representable solution, by which landscape theorem
    (attained minima vs boundary-chasing; identifiability).
 Q4 COST LAW: GPU-seconds vs sequence length and gate statistics,
    and the EXACT sparsity the primitive induces (F0 or F1 with δ).
 Q5 INFORMATION FLOOR: which annex floor bounds this primitive's
    best score on each arena bed, and how far above it does the
    theory predict the arm will land?
 Q6 STATE METRIC: how is the primitive's predicted state
    distribution compared to the oracle's — W1 by default;
    justify anything else.
 Dossier bar: ≥3 papers [V-eq], ≥5 pages [V], Q1 attempted in Lean
 or ≤1e-6 instance, Q2–Q6 instanced; every exit F-graded with a
 gap. Q1 at F4 ⇒ withdrawn. it.14: THE THEORY TABLE (N × Q1–Q6),
 frozen — the arena ticket and the leap's primary input.

PHASE C — THE ARENA (it.15–30)
 Beds: BED-M t*=2 n=2048 (floor₁ 0.7071); BED-K(a) delay d=20
 (Hankel ceiling 1/d for one state; attention-native); the chess
 witness (legality, next-FEN, eval-Δ sign; oracles recomputed).
 4,769 params matched (exact counts printed), 150 steps, N=8, one
 BH family, every cell: crossing rate (CP), conditional NRMSE,
 distance-to-floor (L-FLOOR), distance-to-skyline, GPU-seconds-to-
 floor, peak bytes, certificate grade of any mask (L-CERT),
 W1 to the oracle where a state distribution exists, quotient
 observables only.
 CRITERION (lexicographic): (1) BED-M crossing CP-lower > 0.5;
 (2) BED-K(a) within resolution of the attention-native ceiling;
 (3) lowest GPU-seconds-to-floor; (4) witness tiebreak.
 it.15–16 rig + the GPU-seconds instrument's planted-crossing
   must-fire; it.17–24 BED-M; it.25–27 BED-K(a); it.28 witness;
 it.29 THE RANKING (eliminations by clause; forecasters scored);
 it.30 MARS on the leader: early-stop gaming; sparsity leaking
   label structure (the control bed: zero gates, label not
   factoring through them); param-count; witness cherry-pick;
   certificate δ exceeded (moment census).

PHASE D — THE LEAP GATE (it.31–35)
 it.31–34 THE LEAP DOSSIER: theory table; arena table with
   clauses; MISTAKES.md and STRUCK verbatim; this round's kills;
   the calibration column (author bias, signed); the F2/F3
   failures with reproduced counterexamples; the annex with each
   item's grade; the winner's card sentence.
 it.35 EVALUATION-FOR-LEAP: JUPITER + MARS grade every F1/F2/F3
   failure LEAPABLE (name the FIELD likely to hold the missing
   theorem, not the theorem) or TERMINAL (a bound; no theorem
   removes a bound). ONE call to the leap model with the fixed
   prompt (v19-Q, unchanged: repair graded failures with named
   theorems + runnable instances; compose ONE candidate from the
   winner's primitive plus licensed components; predictions AND
   counters; every untested claim with its cheapest killer; no
   new primitive that is not a repair). Output filed [LEAP-
   UNTESTED]; its instances RUN before anything else.

PHASE E — THE STRONGEST CANDIDATE (it.36–45)
 it.36 absorbed components to [V-eq]; theorems Lean-attempted or
   instanced; F-graded; failures drop their component.
 it.37–38 the candidate built with identity binds, manifests,
   certificates on any mask, param-matched.
 it.39–41 the full arena; BH re-run over the family incl. it.
 it.42 VERDICT by the same clauses; the leap scored on its
   instances-run and predictions-falsified.
 it.43 THE PURGE: losers → attic tarballs (code + dossier +
   commit + citing cells) → removed; orphan census; manifest
   commit BEFORE deletion; suites green on the rebuilt tree; the
   five-minute cold path re-run.
 it.44 D1's tournament chapter (theory table, arena, leap, fate);
   MISTAKES entry; calibration column (author AND leap scored).
 it.45 PROGNOSIS: one arm, its dossier, its cost, its certificate,
   its floors, its sentence; the three unknowns re-graded; whether
   Kaggle is earned. STOP.

────────────────────────────────────────────────────────────────────
THE MATHEMATICS ANNEX (Jupiter owns; every item: statement ·
route · instance · decision · grade; instances marked [RUN] were
executed this session)

M1  β-GRADIENT THEOREM. ∂L/∂β = Σ_i (∂L/∂y_i)(−log Z_i) y_i for
    W = e^ℓ/Z^β. [RUN: +20.87 on path-product data ⇒ β descends
    toward the exact corner; 0.000 on softmax-native data;
    mixture = share-weighted.] Lean #18 [M]. Decides corner usage
    per corpus BEFORE training. Grade F0.
M2  GATE-LANDSCAPE THEOREM. Phase gate a = cos θ vs Rademacher
    target: two critical points, gradient flow converges off a
    null set; open-range a = 2σ(w)−1: infimum unattained,
    |w| → ∞ (R1's divergence as a theorem). [RUN: θ lands at
    1.3e-3; w still growing at 400 steps.] Lean #19 [M]. F0.
M3  EVEN-TARGET BOUND. Monotone gate vs even band-mask: best L²
    error ≥ the target's projection residual onto monotone
    functions; one squared feature ⇒ 0. Lean #17′ [S]. F1 with
    the constant.
M4  IDENTIFIABILITY. (a,b) identified from do() pairs exactly
    (a_i = Δz_i/δ [RUN: 1e-9]); unidentified observationally
    when a,b share a feature (one-parameter family). Lean #20
    [S]. Decides L_jac necessity per bed. F0/F1.
M5  PROXY INDEX. PPI = (best-1-state Hankel R², multiplicative
    Walsh degree-2 mass) [RUN anchors: delay 0.050, power-law
    0.865, AR(1) 1.000; epistasis 0.0375 vs 0.0000]. Decides
    whether real consequence labels (chess eval-Δ, CPython
    outputs) look like the beds. THE ROUND'S CENTER. F0 as an
    instrument; the VERDICT is the cell.
M6  SCAN-COMPUTABLE CLASS. One hop over prefix-scan features
    represents exactly y_i = Σ_{j≤i} φ(C_i−C_j) b_j, φ = exp;
    error outside bounded by M3/Hankel. Lean #21 [S]. F0/F1.
M7  MEMORY KERNEL OF THE CHAIN (Mori–Zwanzig). K ≡ 0 on the chain
    (Markov in z), K ≠ 0 exactly on BED-K kernels, closed form
    from the Hankel residual. Paper. Decides forced/redundant
    composition per bed. F0 target.
M8  SHOCK RETROSPECTIVE (Miyagawa 2606.18303 [V-eq]). τ* =
    −1/inf Ū'' along the gate coordinate vs the observed
    divergence step on R1's eight trajectories. [RUN: τ* = 1.000
    on the toy well.] F-grade assigned by the cell; reported to
    the author by name either way.
M9  CERTIFIED CAUSAL SPARSITY. F0: zero-gate segmentation (Lean
    #22 [M]) [RUN: 8.9e-16, 595×]. F1: Cantelli [V] per pair +
    Boole/Bonferroni [V] union over the mask ⇒ cutoff d(ε,δ)
    [RUN: d=65, 32×, δ=1%]. F1′: AZUMA [V] for token-DEPENDENT
    gates with bounded increments [RUN: d=20 at the same δ —
    tighter AND survives dependence]; Hoeffding/Bernstein when
    bounded/variance-known. Every arena mask carries one of these
    (L-CERT). Lean #23 [S]: the union-bound mask theorem.
M10 INFLUENCE PERSISTENCE. β₀ of the influence graph across the
    threshold ε is a persistence barcode of causal structure
    [RUN: 4096→…→1]; segmentation is its ε=0 endpoint; the
    differentiable-persistence lineage (1905.12200, 1904.09378,
    2011.05804 — occupied) trains the filtration end-to-end;
    delta: the filtration IS causal influence. F0 instrument.
M11 FANO FLOOR for exit prediction. P(err) ≥ 1 − (I(X;Y)+ln 2)/
    ln m [RUN: 0.19 at m=8, I=1 nat; 0.22 at m=32, I=2]. Decides
    S5′'s accuracy is read as distance-to-Fano; I(X;Y) estimated
    on beds with the exact oracle. Lean deferred; F0 statement.
M12 RATE–DISTORTION FLOOR for the fast→slow summary (X₃₈). D(R) =
    σ²2^{−2R} (Gaussian source) [RUN: 0.25 at 1 bit, 0.004 at 4].
    The anti-alias bandwidth dial gets a floor; R3(a)'s "summary
    costs the delay bed" counter becomes a rate–distortion
    prediction. F0 statement.
M13 WASSERSTEIN CALIBRATION. W1 (Kantorovich–Rubinstein [V])
    between predicted and oracle state distributions replaces KL
    for the state head [RUN: 0.492 vs KL 0.519; KL explodes on
    disjoint support, W1 measures distance]. Q6's default. F0.
M14 CHEEGER STRATIFICATION. φ²/2 ≤ 1−λ₂ ≤ 2φ [V] — corpus
    difficulty (mixing) by a combinatorial bottleneck quantity.
    [RUN: my crude φ FAILED the sanity check — grade F3 pending
    an exact sweep-cut conductance; first task of the annex, and
    it stays F3 in the table until it passes.]
    [SUPERSEDED it.19, RULING J-17d: M14 IS F4. The F3 above is a
    CONDITIONAL pre-registration -- "pending an exact sweep-cut
    conductance" -- and its condition was never discharged: M14 was
    struck by V-25 before the sweep-cut settled anything. The
    pre-registered sentence is left standing rather than rewritten,
    because restating a pre-registration is an author amendment
    (C4's class). The one live grade and its ground:
    V20_R15_LEAP_LEDGER.md, "## RULING J-17d". The journal's F1
    reading is a scoreless SCOREBOARD line, carried to that ruling
    by CORRECTIONS INDEX C20.]
M15 LR-TEST "PINNED" (Ruling 10′). Λ = 2ΔLL on held-out; 3.841
    (χ²₁), ln n (BIC); minimum detectable |β−1| printed [RUN:
    null quantiles 0.49/3.97/5.99; false rate 0.055]. F0.
M16 HANKEL BOUND (L-BOUND). Best-k-state fit R²_k = Σ_{i≤k}σ_i²/
    Σσ_i² (AAK/Glover [U → V-eq at it.6]) [RUN: 1/d for delays].
    F1 by construction.
LEAN LEDGER this round: #18 (M1), #19 (M2), #22 (M9-F0) [M];
#17′ (M3), #20 (M4), #21 (M6), #23 (M9-F1) [S]; #14 committor,
#10/#11 floors [D]. Every theorem with its two-column domain
census (L-DOM) against BED-M's {−1,0,+1} support and BED-K's.

────────────────────────────────────────────────────────────────────
KILLS: a wing named rather than found ⇒ struck; a theory exit
without F-grade + gap ⇒ not filed; a mask without certificate ⇒
refused; a capability number without its floor ⇒ not a number;
a second leap call ⇒ breach; leap output acted on before its
instance runs ⇒ struck; an absorbed component without a licensing
theorem ⇒ dropped; deletion without the manifest commit first ⇒
breach; a published cell unreachable from attic ⇒ breach; M14
cited before it passes its own check ⇒ struck.

SCOREBOARD (moves at it.4, 14, 29, 35, 42, 43, 45):
 wing list frozen, four citations each +2 · theory table N×Q6 all
 graded +6 · annex M1–M16 with ≥12 at F0/F1 and M14 resolved +4 ·
 arena complete, one family, floors on every cell +4 · a winner
 crossing BED-M and surviving BED-K(a) +12 · the winner's cost
 ≤1/10 incumbent with a certificate +4 · leap dossier + ≥50% of
 its instances run +3 · candidate beats winner by clause +8 /
 ties +3 / loses with reason +1 · purge complete +3 · D1 chapter
 +2. Ceiling 48.

YOUR TASK: it.1 — mine the ledger under the rubric; cite the line
that finds each wing; name nothing you cannot cite. JUPITER: M14
first, because an annex that ships a failed instance is the
fourteenth class in a lab coat.
════════════════════════════════════════════════════════════════════
