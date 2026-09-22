# Phase J — Addendum L (the SU(2) leap): process record

Opened 2026-09-22 17:03 local; one hour. Contract: the author's Addendum L (in the
dispatch). Three workflows named `chase`, `cameron`, `foreman`; sonnet and haiku
only. House's schedule-generator leap (RECORD.md, last section) is scored on the
same table and not run this hour.

**Pinned above every row (Wilson):** this addendum claims a certified capability
— composing order — at RoPE-class cost. It does NOT claim lower LM loss at 725k
parameters, and no row may be read as a loss claim.

Every row appends ONE JSON line to `record_L.jsonl` here with: `row`, `seat`,
`nurse`, `model`, `machine`, `bar` (as registered, with source "contract" or
"dispatcher"), `measured`, `verdict` (PASS / FAIL / NEITHER / SKIPPED), `control`,
`red_first` (the RED line of the test before the implementation passed it),
`producer`, `output`, `started`, `finished`, `killed` (one sentence, no mechanism
words), `replacement` (reroute / reprice / retire, required on any FAIL).

Bars marked "dispatcher" were set at dispatch because the addendum gives none for
that row; they are derived from the contract's own certified numbers.

---

## House-mode round L1 (2026-09-22, 22:59–00:10 local): prognosis

Fellows on opus with nurses: Foreman (root cause), Chase (break the passes),
Cameron (the path that sacrifices nothing), Wilson (ground truth). Inspector:
pass 1 at the 30-minute mark (41 audited, 8 struck) and a final pass (27 audited,
10 struck; 18 struck in total, carried unchanged). Reports and producers:
`L1/<seat>/`; audits: `L1/inspector/PASS1.md`, `L1/inspector/FINAL.md`.
Dr House was not run: the author disabled the fable leap this round.

### Prognosis

Addendum L is **"attempted" at it.L0 by its own pre-registered rule, and alive
into it.L1.** Count 1 fails as written: the contract's generators (72° about z,
120° about (1,1,1)/√3) generate an infinite group — the product ab has rotation
angle 167.298°, 2cos θ = −1 − sin 72° has the non-monic minimal polynomial
16x⁴+64x³+76x²+24x+1, and the two axes are 54.74° apart, an octahedral angle,
not the 37.38° or 79.19° of an icosahedral 5-fold/3-fold pair. The object
survives: an icosahedral pair closes at 60 in SO(3) and 120 in SU(2) with a
10,000-letter word matching its Cayley table, and that pair is the pre-registration
for it.L1, not a reversal of the it.L0 kill. Count 2 passes once the cost is
measured on an implementation that computes the contract's G_ij: a fused
`tl.associative_scan` Mode B′ runs at 1.03–1.09× forward and 1.08–1.09×
forward+backward against the fastest SDPA backend this build has
(EFFICIENT_ATTENTION), re-run on the GPU by the Inspector; the recorded 2.15× was
a launch-bound torch scan whose prefix product runs in the opposite order to the
contract's. Count 3 is unmet: the board bed is retired against its own floor, and
the two order beds failed for a missing anchor and a memorising budget, not
because of the arm.

### Chart

- **Foreman.** F1: the A5 bed has no signal that picks out key 0 (no BOS,
  single-frequency RoPE with period 12.57), so the fold's relative product cannot
  be read — RED statuses reproduce (as-built hand-set read fails; optimised
  attention to key 0 stays uniform); kept as a status, while the specific numbers
  are single draws of an unpinned split and are struck. Anchored, 512 sequences,
  400 steps, one seed: (f_Q) 1.0 at position 64 against softmax 0.0312 and FoX
  0.0156 — bound, one seed, not bitwise reproducible. F2: the recorded 0.006 loss
  is memorisation (learned generators are not a group lift) — kept as a status.
  F3: "no SU(2) gate can reach the L1-IAUT bar" — struck (RED written after its
  result; covers 2I and 2O only) → Open. F4: `a5_bed.py` seeds its split with
  Python's salted `hash()` — kept (Chase found the same, with a crc32 fix GREEN
  under two hash seeds). Kills L1-A5 and L1-IAUT as evidence against (f_Q);
  reroutes: **L1-A5′** (BOS, 512 sequences, 400 steps, stable seeding, 3 seeds,
  bar unchanged) and **I-AUT-A5** (the integer-only construction over A5 instead
  of S5).
- **Chase.** Q2 kill confirmed without breadth-first search (kept). Q3 is a
  tautology as implemented: its commutative controls never compose two gates,
  and in a real softmax read the FoX gate distinguishes AB from BA by 0.1434
  (kept) → reprice: Q3 is order-dependence at a fixed pattern, i.e. Q2's
  commutator, not an independent row. PF-GAMMA: 1/3 holds; the unit-lattice value
  2.098 at ε = 0.03 is an optimiser stall, the true lattice minimum is 0.0299
  (ε − 2ε³ within 0.1%), and `test_pf.py` collects no tests (kept) → reprice the
  token-lattice bar to 0.3305/0.2509/0.0980/0.0299. DRIFT: the 4.71e-8 miss is
  the arccos readout; atan2 gives 2.1e-16 (kept) → keep the 1e-9 bar, retire the
  1e-7 reprice. COST-B′ record says MATH where its own JSON says
  EFFICIENT_ATTENTION (kept, agrees with Wilson). A5: the FoX arm's 0.03125 is
  above chance, so "all at or below chance" is false; restated as "none
  distinguishable from chance" (binomial p > 0.05) (kept). **Overruled on cost:**
  Chase's "cost fails" measured the author's implementation, which Cameron's
  bound RED and Wilson's [V] fact show computes a conjugated product with a
  launch-bound scan. Struck: independent reproductions of Q1, FOLD, FQ-ARM and the
  hand-set 500/500 (no RED on the board); "260 parameters at d 64" (no event).
- **Cameron.** Cost forward and forward+backward GREEN, bound, re-run (kept).
  The author implementation's fold error 3.775 against the contract's G_ij (kept,
  the RED fold clause). Board bed inadmissible (kept); reconciled with Chase's
  "the bar is satisfiable if both controls score ≤ 0.90": both are true of their
  own text, and the project's floor rule decides — a margin set against arms that
  sit under a 0.9968 input-only tracker is a bed failing at its floor, so
  per-square board accuracy is retired and exact match (tracker 0.844) is the
  only board route. Shell-game bed (S3, S4): commuting ceiling and margin checks
  pass, twin plateau (A2) still RED (kept); no bar written. Struck: torch.compile
  1.069× and the length sweep, the k22-fusion retirement, the S = 16 conjugation
  check, the A4 construction.
- **Wilson.** The record was truncated at 11:38:24.913Z by a `cat >` redirect
  from Foreman's nurse Fay, dropping 11 finished rows; they are restored verbatim
  from the journal snapshot (`record_L_recovered.jsonl`). `fetch_cameron.py`
  builds its four citation rows from a hard-coded dictionary, so "four citations
  fetched" is false. `su2.py` was stable (last write 17:07:03) before every arm
  ran. Three sentences in the cache commits are false: "the baseline ran the math
  backend" (5b7e6b8), "four citations fetched" (5b7e6b8), "all three at or below
  chance" (3932ced).
- **Chase (rivals, second dispatch).** a2 equals Qwen's headwise G1 gate
  (arXiv 2505.06708); the FoX gate law ships in Qwen3-Next's Gated DeltaNet;
  Kimi Linear's KDA and GDN ship non-commuting, content-dependent order at
  linear-attention cost with per-token spectra in [0,1]; only a unit-circle
  per-token spectrum is absent from shipped work. **All struck** for this round:
  the test events used `state` instead of `status`, so no RED is on the board,
  although the Inspector reproduced 0/10 against the stub and 10/10 against the
  implementation. → Open until re-logged.
- **Inspector.** 68 audited, 18 struck (list in `L1/inspector/FINAL.md`).

### Open

- Novelty of the SU(2) claim against KDA/GDN, DeltaProduct and
  negative-eigenvalue DeltaNet (the last two unread, [U]). No novelty sentence on
  any page until the rivals row is re-logged and those papers are fetched.
- The I-AUT bound (S5 has no faithful SU(2) image): unbound this round.
- The contract's "no custom kernel": the passing implementation uses a Triton
  scan kernel; the torch.compile route (1.069×) is struck for want of a test event.
- Contract §1 writes a two-sided fold, Π_i (conj(Π_j) v_j Π_j); the arm and its
  oracle are left-only. Which one the contract means is unresolved.
- COST-B′'s standalone SDPA (5.861 ms) versus its in-pipeline SDPA stage (4.376 ms).
- Q1, FOLD, FQ-ARM: reproduced, unbound.

### Corrections to earlier text in this record's commits

The bodies of 5b7e6b8 and 3932ced cannot be rewritten; these sentences in them
are false and are superseded here: "the baseline itself ran the math backend"
(the producer's JSON says EFFICIENT_ATTENTION); "Four citations fetched" (the
rows were built from a hard-coded dictionary); "all three sit at or below chance,
1/60" (the FoX arm scored 0.03125; none is distinguishable from chance); "breaks
at eps = 0.03" (an optimiser stall; the lattice law is monotone).
