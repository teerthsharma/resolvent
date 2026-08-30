# VENUS / IRENE — R9 iteration 1

Seat: IRENE, the alternative. Discipline: write before the data.
Branch: `worktree-agent-adec25a08af78a4c8` (see §6 — `feat/r9-causal-consequence`
is checked out in another worktree and cannot be switched to from here).

**Path note.** The dispatch asked for this file at
`.superpowers/sdd/polymorphic-drifting-squirrel/venus-report.md` outside the
worktree. Worktree isolation refused every write to that path, so it is written
at the same relative path **inside** the worktree and the parent should copy it
out.

Deliverables shipped:

* `PREREGISTRATION_HOLE_AUDIT.md` — eight holes across the five
  pre-registrations, four code drifts, and the patch row for each.
* `R9_IRENE_PREDICTION.md` — the competing hypothesis with eight predicted
  numbers, five falsifiers, and the R9 fall-through row.

---

## 1. Deliverable 1 — the holes, ranked

Ranked by how badly a fall-through would be misread. A hole that prints a
**false sentence** outranks one that prints nothing, because nothing is visibly
nothing.

**H-1. `E_LADDER` row C's universal is enforced as an existential.** Row C
requires `|delta| < 0.027260` at **every** rung. `scale/e_ladder.py:280-281`
builds `big` over `("e3_t8", "e3_t32")` only, then falls through to
`return ("C", ...)` at `:288` whose sentence asserts *"every `|delta|` is below
the 13-seed resolution"*. Every CI covering zero with `|delta| >= 0.027260` at
`t*=1` or `t*=2` alone falls through both C and D and prints C with a false
universal. **The completed ladder is one rung away from having sprung it:**
`e3_t1` reads `delta = -0.036025`, CI `[-0.118936, +0.062209]` — a CI covering
zero with `|delta|` above the floor. Only `e3_t2`'s excluding CI (and G
pre-empting) kept the branch unreached. Patch row **D'** widens D to any rung.

**H-2. `E_LADDER` row A never checks "covers zero at `t*=1`".** `verdict()`
reaches the `won("e3_t32")` block at `:250` whenever `won("e3_t1")` is false,
and `won` is `ci_lo > 0.0` — so a CI at `t*=1` that excludes zero *downward*
still enters the block and returns A with the sentence *"Compatible with zero at
t\*=1"*. The outcome (negative exclusion at `t*=1`, positive at `t*=32`) falls
through all eight rows of section 6. It is a **cleaner dose-response than row A
asks for** — cost inside the hop budget, gain outside it — so the best result
the ladder could have produced had no row and would have printed a contradicting
sentence. Patch row **I** (CROSSOVER).

**H-3. `M2_TRAINED` has no row for a control flatter than the arm.** B needs the
control *steeper*, C needs it *comparably flat*. Flatter than the arm falls
through both. This is the ordering the document's **own measured table already
shows** at `:30` — arm `-1.298`, dense control `-1.088`. Still pre-data (no
trained operator has been measured), so patch row **B'** is live.

**H-4. `M2PRIME` splits a continuous quantity into two branches.** A fires on
`~0`, B and C on "high", and no number anywhere in the file separates them. A
determined fraction of `0.45` fires nothing. Patch: cut at `<= 0.05` / `>= 0.60`
plus row **E** (NOT READABLE) for the middle, both fixed against the file's own
two-ended calibration before the run.

**H-5. `M2_TRAINED`'s rows all presuppose a surviving instrument.** `:124`
commits to running the cheap instrument test first; `:68-74` says what would
convict it. Nothing says rows A-D are void if it *is* convicted, and the natural
misreading is to keep reading them. Patch row **0**, evaluated before A-D.

**H-6. `M3_QUINTUPLE` has no row for `argmax` beating `settled`.** Rows 1 and 2
both condition on settled beating or tying argmax; rows 3 and 4 on the twin
contrast going the other way. Settled beating twin and losing to argmax fires
nothing — on the axis section 4 names the **headline caveat**. Post-data for
`k=8`, live for the `k=16`/`k=32` follow-on.

**H-7. `M2`'s three readings do not partition the control's range.** "Near
zero", "rises", "flat and nonzero" leave a partially-decayed control uncovered,
and no threshold is stated though `:42` supplies the obvious one (CP upper
`0.00292`). Post-data.

**H-8. `M2PRIME` never gives R2's second kill clause a row.** *"or constraint
destroys training"* is quoted at `:21` and never returns. Patch row **F**.

### Drift — rows whose branch does not match them

`E_LADDER` section 6 is the only table in the five enforced in code. Four rows
have drifted. Two are H-1 and H-2. The other two answer deliverable 1's fourth
question:

* **D-1 — row F fires on partial data.** Section 4 `:130-132` and the module
  docstring `:29-31` both say rows **A, C and F** quantify over every rung and
  are unavailable until every rung is in. A is guarded at `:252`, C at `:276`,
  **F at `:241-244` is not guarded at all**. F is the theory-death verdict and
  can be claimed on two rungs.
* **D-2 — row H fires on partial data.** H's *"never positive at any rung"* is a
  universal; `:267` fires it on `any(lost) and not any(won)` with no `complete`
  guard. Unverifiable on a partial ladder — in the row that exists to patch a
  hole.
* **D-3 — row G's scope changed.** The document says *"that rung is credited
  nothing [...] Printed, never read as a verdict"*; `:226-231` returns G as
  **the** whole-ladder verdict, pre-empting every other row. Defensible under
  section 7b.1, which calls a G ladder *"unreadable at this budget"* — but 7b.1
  and row G's own last sentence disagree inside the document, and the code
  silently picked one.
* **D-4 — row D is stricter in code than in the document** (safe direction).

Also: `scale/e_ladder.py:23` says *"`OUTCOMES` below is a transcription of
[section 6]"*. **There is no `OUTCOMES` symbol in the file.** The promised
machine-readable transcription does not exist; the only encoding is the
imperative chain, which is where all four drifts live.

### Mechanical evaluability

None of the four non-ladder documents has an enforcing branch anywhere in the
tree. `inspector.py:335-372` parses `M2_TRAINED_PREREGISTERED_READING.md`, but
it parses the **measured-state table** and re-derives the slope from its counts;
it does not touch rows A-D. Eight predicates need a number they do not have —
listed in the audit's section 2 table. Notably `M2PRIME`'s row **D** is
**unfalsifiable**: its antecedent *"softmax's is also high"* is always true
(`:80`, `:91` state softmax reads `1.0` by construction), so D fires whenever B
or C's antecedent holds and no precedence is stated — **row C is unreachable as
written**. The one exception across all four is `M3_QUINTUPLE` section 5,
decidable because section 4 `:104-108` defines the verdict trichotomy inside the
same document. That is the pattern the other three should have copied.

## 2. Deliverable 2 — the competing prediction

Filed at `R9_IRENE_PREDICTION.md`, dated 2026-08-30, with the explicit note that
no R9 number existed.

Candidate chosen: **the third (dilution)**, sharpened past the way the dispatch
put it. The dispatch said the early positions carry almost no information. The
builder does something more damaging than that.

Two structural reads carry it:

* `scale/m3_quintuple.py:302-306` — the pivot term is written into **row `s-1`
  and no other row**. Under a vector readout, `settled`, `twin`, `argmax` and
  `softmax` compute the same `x + a@x` at 63 of 64 positions. The class
  docstring `:257-261` says so itself; it was a correctness argument while the
  readout was scalar and becomes the refutation once it is not.
* `scale/negation_scope.py:391-394` — `a[:, :head+1] = 0.0` with
  `head = s-1-t*`, so at every `p <= head` the scan gives `z_p = b_p`
  **exactly**, and `b` is written into the input as `x[:, :, CH_FLIP]`. Verified
  `RUN`, bitwise (`torch.equal`), against the shipped builder: 63 of 64
  free-copy positions at `t*=1`, 56 of 64 at `t*=8`. This is the same zero-hop
  legibility defect that `b[:, s-1] = 0.0` was introduced at `:363-376` to
  remove — the fix zeroed one position, and a vector readout re-opens it at
  `s - t* - 1` others.

The dilution factor `w = Var(z_{s-1})/SumVar` runs **0.0156 to 0.0808**: between
1.6 % and 8.1 % of the vector label's variance sits at the only position where
the arms differ.

**PREDICTION 1 — the eight numbers.** Scaling the completed ladder's scalar
margins by `sqrt(w)`:

| rung | `settled - softmax` | `twin - softmax` |
|---|---|---|
| `e3_t1`  | -0.024632 | -0.020129 |
| `e3_t2`  | -0.010574 | -0.007578 |
| `e3_t8`  | +0.010229 | +0.011447 |
| `e3_t32` | +0.013881 | +0.010150 |

**All eight below `0.027260` in absolute value.** The contract says the margin
grows; I say every one lands inside the E ladder's own pre-registered 13-seed
resolution floor. These are upper bounds — residual left on the copy positions
shrinks them further.

**PREDICTION 2 — the trap.** Pooled vector NRMSE at `t*=1` in `[0.113, 0.333]`,
`t*=2` in `[0.222, 0.415]`, against current scalar readings of `0.79`-`1.01`.
The vector lane will print numbers two to seven times better than anything the
scalar ladder produced, on rungs the scalar ladder credited **nothing** under
row G, and three of four rungs will cross from above `1.0` to below it. **Row G
does not clear.** The bar moved because the label changed.

**PREDICTION 3 — the RED gate trips.** `scale/m3_capability.py:319` requires the
untrained arm at or above NRMSE 1.0. The mechanism that tripped it before is now
present at `s - t* - 1` positions instead of one. I predict the 0-step pooled
NRMSE reads below 1.0 at `t*=1` and the lane aborts `INSTRUMENT BROKEN`, unless
the builder is changed to `b[:, :head+2] = 0.0`. Direction predicted, magnitude
explicitly not.

Five falsifiers, each binary, in section 4 of the file.

## 3. Deliverable 3 — the R9 fall-through row

Row **Omega**, in section 5 of the prediction file, written before R9's table
exists. It fires when the two lanes are not reading the same task: pooled NRMSE
moving more than `0.10` while every margin stays inside `+/-0.027260`; or the
free-copy positions not being reported with their count; or any position with
`sd = 0`, where `scale/negation_scope.py:672-677` returns `nan` and a pooled
average is an average over non-numbers. It licenses **nothing in either
direction**, forbids carrying any scalar-lane number into the vector lane, and
creates one obligation: a matched-position control at `s-1` only, inside the
vector lane, that must reproduce the scalar ladder's margins. Omega is evaluated
**before** any row of R9's table.

---

## 4. Claim ledger

| # | claim | class | check |
|---|---|---|---|
| 1 | Row C's branch scans only `t8`/`t32` while the row says every rung | READ | `scale/e_ladder.py:280-281` vs `E_LADDER_PREREGISTERED_READING.md:176` |
| 2 | Row C's returned sentence asserts the universal it did not check | READ | `scale/e_ladder.py:288-293` |
| 3 | `e3_t1` has a CI covering zero with `\|delta\| = 0.036025 >= 0.027260` | READ | `results/e_ladder_reading.txt`, `e3_t1` row |
| 4 | Row A's branch is entered on `not won("e3_t1")`, which admits a negative exclusion | READ | `scale/e_ladder.py:250`, `won` at `:235-237` |
| 5 | Row F has no `complete` guard; A at `:252` and C at `:276` do | READ | `scale/e_ladder.py:241-244, 252, 276` vs doc `:130-132` and docstring `:29-31` |
| 6 | Row H has no `complete` guard despite a universal over rungs | READ | `scale/e_ladder.py:267-268` vs doc `:181` |
| 7 | Row G is returned as the whole-ladder verdict, not per rung | READ | `scale/e_ladder.py:226-231` vs doc `:180` |
| 8 | `OUTCOMES` does not exist in `e_ladder.py` | RUN | `grep -n OUTCOMES scale/e_ladder.py` -> one docstring hit at `:23`, no symbol |
| 9 | No code enforces the four non-ladder outcome tables | RUN | `grep -rln` over `*.py` for the four filenames -> 7 files, all docstring/report strings; `inspector.py:335-372` parses the measured-state table only |
| 10 | `M2PRIME` row D's antecedent is always true, making C unreachable | READ | `M2PRIME_PREREGISTERED_READING.md:79-82, 91` |
| 11 | `M2_TRAINED`'s own table shows the control flatter than the arm | READ | `M2_TRAINED_PREREGISTERED_READING.md:30` |
| 12 | The pivot term is written into row `s-1` only | READ | `scale/m3_quintuple.py:302-306`, docstring `:257-261` |
| 13 | `z_p == b_p` bitwise for every `p <= head` | RUN | `torch.equal` on `make_equilibrium_batch(512,64,8,seed=0,t_star in {1,8})` -> True; 63/64 and 56/64 copy positions |
| 14 | The prefix scan's last element is the shipped scalar label | RUN | same draw, `torch.equal(Z[:,-1], y)` -> True at both rungs |
| 15 | `SumVar(t*) = s - 1 + t*(t*+1)/2` | DERIVED + RUN | closed-form recursion; independently drawn at `n = 4e5` — SumVar agrees to <0.25 %, `sqrt(w)` to `4e-4` |
| 16 | The untrained-arm RED gate exists and aborts below 1.0 | READ | `scale/m3_capability.py:319, 322-324` |
| 17 | The zero-hop legibility defect previously tripped that gate | READ | `scale/negation_scope.py:363-376`, `0.993760`/`0.993600` at `t*=1` |
| 18 | `nrmse` returns `nan` when `sd == 0` | READ | `scale/negation_scope.py:672-677` |
| 19 | Predicted margins (the eight numbers) | DERIVED | scalar margins from `results/e_ladder_reading.txt` x `sqrt(w)` from claim 15; both columns shown, disagreement `<1.6e-5` |

Adversarial pass on the load-bearing claim (12): it would be false if R9's arm
wrote `_alpha` into every row. That is a one-line change and it is falsifier 5
in the prediction file — the cheapest way for the contract to beat this
prediction is to make it inapplicable rather than to out-measure it. Recorded
there, not hidden.

Adversarial pass on claim 13: the check would pass with the logic deleted only
if `b` were unused, and it is not — the same draw shows `Z[:,-1] == y`, so the
scan is the real one. The two paths behind claim 15 share the builder but not
the method: one is a recursion on variances, the other measures drawn samples
and never forms the formula.

---

## 5. What I could not validate

`verdict()` was never executed. Every drift in section 1 is a reading of a
branch condition against a row of English, traced by hand through the chain, not
a run that watched the wrong row fire on a synthetic ladder — and D-1 and D-2
are exactly where that gap matters, because "the guard is absent" is established
by reading but "no earlier branch catches this case first" was traced rather
than executed. The four unenforced documents have no branch to compare against
at all, so every judgment about them is English against English, including the
claim that a given outcome falls through: I constructed the fall-through and
checked each row by hand, and a row I mis-parsed is a hole I invented. The
`M2`/`M2PRIME`/`M2_TRAINED` patch rows are marked LIVE on each file's own
statement that its run had not happened plus the absence of a matching
`results/` artifact — neither is proof that no number exists off-journal, and if
one does, those rows are post-hoc and must not be adopted. On the prediction
side, the `[n, s]` label does not exist, so section 2b's prefix scan is my
choice of what "drop the `[:, s-1]` index" means; a different label choice
invalidates the eight numbers though not the formula that produced them, and the
file says so. The per-position NRMSE brackets in PREDICTION 2 are not derived at
all — they are ranges chosen to bracket what the scalar ladder reads at `s-1`,
and the copy-position bound of `0.30` is a judgment about how well a trained
linear readout over 24 channels solves an identity copy. It is the softest
number in anything I filed. PREDICTION 3's direction I stand behind and its
magnitude I explicitly do not: the recorded precedent is a `0.0062` dip at a
`0.5` legible share, and I am extrapolating to a `0.984` share through a
mechanism I have read but not measured. Finally, three of the four rungs whose
margins I scaled were credited **nothing** by row G, so PREDICTION 1 is a
prediction about what the instrument will print and not about a capability at
any rung except `e3_t1`.

## 6. Concerns

1. **Branch.** The dispatch names `feat/r9-causal-consequence`. `git branch -a`
   marks it `+`, i.e. already checked out in another worktree, so this worktree
   cannot switch to it. Work is committed on
   `worktree-agent-adec25a08af78a4c8` and needs merging into the R9 branch.
2. **Report path.** Worktree isolation refused every write outside the worktree,
   so this file lives at the requested path *inside* the worktree and must be
   copied out.
3. **Two unguarded universals are live defects in shipped code.** D-1 and D-2
   let rows F and H fire on a partial ladder against the file's own docstring.
   The CPU ladder is complete so nothing published is affected, but the CUDA
   lane is 22 of 60 cells (`FINDINGS.md` D2) and a partial `verdict()` call
   there can print a theory-death row.
4. **Nothing was patched into the five documents.** Three are post-data and
   patching them now would be the sin the discipline exists to prevent. The
   patch rows sit beside them, marked LIVE or POST-DATA, for their owners.
