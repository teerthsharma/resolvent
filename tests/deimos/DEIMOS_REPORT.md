# DEIMOS report — R9 iteration 1

Moon of MARS (MORIARTY / standing adversary), CEQ v11.1 Baker Street protocol.
Scope: the three surfaces Mars named unaudited — `scale/e4_harmonic*`,
`scale/eprocess_perdraw.py` (and its parent `scale/eprocess.py`), and
`ceq/nash.py`. Tests live in `tests/deimos/test_deimos_r9_iteration1.py`, run
with `python -m pytest tests/deimos/ -q` (8 passed, 8 attacks, 8 fired — see
the ledger below for what "fired" means per attack). Tree state: fast-forward
merge to `bb0c04a` before any attack was written.

## Claim ledger

| # | Claim | Class | Check |
|---|---|---|---|
| 1 | `scale/eprocess.py`'s `Eprocess.value` (`:311-313`) crashes with `OverflowError` well inside `eprocess_perdraw.py`'s own designed operating range (2048–10240 pooled draws), on a plausible winning stream, *after* the decision threshold has already been crossed | RUN | `tests/deimos/test_deimos_r9_iteration1.py::test_eprocess_value_overflows_deep_inside_its_own_designed_operating_range` |
| 2 | `calibrate()`'s `max_peak = math.exp(min(max_log, 700.0))` (`:435`) is a silent clamp of the same overflow class, dormant at every horizon shipped today, live the moment a caller raises `--horizon` past ~1750 | RUN | `test_eprocess_calibrate_max_peak_clamp_is_the_same_overflow_class_dormant` |
| 3 | `ceq/nash.py:146`'s auto-tau path (`tau=None`, the one `ceq.arms.NashArm` actually uses) computes `safe_tau` over the *whole batch*, exactly the anti-pattern `scale/negation_scope.py:462-466` names and avoids for the identical construction; measured on the real corpus, mean \|stance\| is roughly halved versus the per-example tau | RUN | `test_nash_operator_shares_one_tau_across_the_whole_batch_and_overdamps_it` |
| 4 | `qre_stance`'s `return_residual` mechanism — the module's own stated defence against reading a non-equilibrium — is never consumed anywhere in the tree outside its own definition | RUN | `test_qre_residual_is_reported_but_never_read_anywhere_in_the_repo` |
| 5 | No call site checks a caller-supplied `tau` against `safe_tau`; `tests/w7/test_w7_nash.py`'s hardcoded `tau=0.25` sits at 0.749× the true threshold for the exact game it uses | RUN | `test_nothing_enforces_tau_above_tau_star_at_explicit_call_sites` |
| 6 | Below `tau*`, a real (non-hand-picked-degenerate) symmetric potential game has genuinely multiple equilibria: two different starting points reach two different fixed points, both at residual 0.0 | RUN | `test_below_tau_star_a_real_game_has_multiple_equilibria_from_different_starts` |
| 7 | `tests/cameron/test_harmonic_attribution.py` — an entire "U1/N3" contract clause with named pilot numbers — calls nine names on `scale.negation_scope` that do not exist; all 11 of its tests fail with fresh `AttributeError` | RUN | `test_harmonic_attribution_battery_cites_a_producer_that_does_not_exist`, corroborated by `python -m pytest tests/cameron/test_harmonic_attribution.py -q` → 11 failed, live this session |
| 8 | `scale/e4_harmonic.case_graph`'s `lru_cache` returns a shared mutable `list[set[int]]`; a mutation by one caller is visible to every later caller with the same arguments (currently dormant — no live caller mutates it) | RUN | `test_case_graph_lru_cache_shares_a_mutable_adjacency_list_of_sets` |
| — | `Eprocess.update`'s `abs(d) > B` raise is reachable through both live producers (`eprocess.paired_difference`, `eprocess_perdraw.clipped_errors`) | REFUTED (checked, not filed) | both producers clip *before* differencing, so `\|d\| <= B` holds by construction; the raise is a correctly-dead defensive invariant, not a bug |
| — | A reader elsewhere quotes `THRESHOLD`/`ALPHA` against the wrong `N_DIRECTIONS` budget | REFUTED (checked, not filed) | every reader (`capability_table.py`, `tests/chase/test_eprocess*.py`) pulls `EP.THRESHOLD`/`EP.ALPHA_FAMILY` from the module; none hardcodes 20 or a stale value |
| — | `e4_harmonic.local_features`/`measure` read the decoder in-sample, leaking the label the way `impact`'s decoder probe did | REFUTED (checked, not filed) | `measure()` calls `fit_eval` imported directly from `scale/rips_gate.py`, which is the same held-out-half, ridge-1e-6 function `rips_gate.py:163-174` documents; no local reimplementation, no leak found |

## Findings, ranked by cost if true

### 1. `Eprocess.value` overflows deep inside the per-draw process's own operating range (CONFIRMED, highest cost)

`scale/eprocess.py`'s `Eprocess` class keeps the mixture in log space per
draw (`update`, `:319-333`) — that part is sound and never overflows. But
`Eprocess.value` (`:311-313`), called on *every* `update()` to refresh
`self.peak`, is `math.exp(_logsumexp(self.log_arm) - log(|grid|))`. Once the
accumulated log-evidence for the best grid arm passes roughly 709 (double's
`exp` ceiling), that call raises `OverflowError: math range error`.

`scale/eprocess_perdraw.py` exists specifically to push `t` from 5 training
seeds into the thousands (2048 draws per seed, 10240 pooled over the shipped
`--seeds 0 1 2 3 4` default), and its `run()` (`:134-160`) and
`eprocess.live()` (`:584-611`) both loop `pair.update(d)` over every draw with
no early exit on a decision — only on a `void` `ValueError`.

Measured this session: a plausible winning stream (mean 0.3 NRMSE margin on
the `[-2,2]`-clipped scale, sd 0.3, seed 0, `n = 2048*5 = 10240` — the exact
shipped pool size) crosses the decision threshold (`E_t >= 40`) at draw 67, a
legitimate decision, and then crashes `Eprocess.update` with `OverflowError`
at draw 10135 — one shy of the full pooled run. A harsher adversarial stream
(`d = +B` every draw) crashes at draw 1757. Both reproduced live,
`tests/deimos/...::test_eprocess_value_overflows_deep_inside_its_own_designed_operating_range`.

The calibration battery that is supposed to certify this construction
(`calibrate()` / `_run_block()`, `eprocess.py:383-437`) never exercises
`Eprocess`/`Pair` at all — it independently re-derives the same product with
`np.cumsum(np.log1p(...))` and compares in log space (`ls >= log_thr`), which
is exactly why it never hits this. The object actually used to read live data
is not the object the must-fire battery certifies. All 37 tests in
`tests/chase/test_eprocess.py` and `tests/chase/test_eprocess_perdraw.py`
pass today — none of them run a stream long or strong enough to reach this,
confirmed by running them this session. This is a genuinely new finding, not
a symptom of the 146-failure baseline.

**Cost if unfixed**: a per-draw run that is *winning* — the exact outcome the
round wants to report — can crash outright partway through the pooled
default, instead of reporting a clean decision.

### 2. `nash_operator`'s live path shares one `tau` across the whole batch (CONFIRMED, high cost)

`ceq/nash.py:146`: `t = tau if tau is not None else safe_tau(game.reshape(-1, s_len, s_len))`.
`game` already carries the batch dimension, so the reshape is a no-op and
`safe_tau`'s internal `.max()` (`nash.py:64`) takes the max over the *entire
batch*. `ceq.arms.NashArm` is constructed with `tau=None` by default
(`ceq/arms.py:46`) and never overridden in `run_all`/`train_one` — this *is*
the path the benchmarked "nash" arm trains and evaluates on, not a corner
case.

`scale/negation_scope.py:462-466` documents the correct pattern for the
identical construction and names this exact mistake: *"`tau` is
`ceq.nash.safe_tau` PER EXAMPLE. Calling `safe_tau` on the whole batch would
return the max over it, which makes the label depend on the batch size."*
`ceq/nash.py` does the thing its sibling module warns against.

Measured this session on the real corpus (`ceq.corpus.build`, n_train=384,
seed=0, exactly the construction `Arm.operator(kind="nash")` uses):
batch-shared tau equals the single worst-case example's own `safe_tau`
(13.54), 2.2× the *median* per-example `safe_tau` (6.09); 56.5% of examples
sit below half the batch tau, i.e. are damped far past what their own game
needs; mean `|stance|` — the entire signed signal the module's docstring says
buys composition — comes out under half of what it is with per-example tau.

Context, measured live this session: `python -m pytest tests/w7/test_w7_nash.py -q`
→ 4 failed. The Nash arm's OOD NRMSE is 5.27 — worse than predicting the mean
(>1.0) and worse than the signed regression it was built to replace
(3.68–3.77). This does not prove the batch-shared tau is the entire cause,
but it is *a* cause the codebase's own other module already knew to avoid,
and it directly damps the exact mechanism (`|A_ij| <= rho*P_ij*stance_j`) the
whole construction rests on.

### 3. `qre_stance`'s well-posedness mechanism is unread and unenforced (CONFIRMED, moderate cost)

Three parts, chained:

- `return_residual` — the only way to learn whether an iterate actually
  converged — is never passed `True` anywhere in the repository outside its
  own definition in `ceq/nash.py`. The mitigation the docstring describes
  (*"reports its own residual rather than asserting convergence"*) is dead
  code.
- Nothing compares a caller-supplied `tau` against `safe_tau`. Six calls in
  `tests/w7/test_w7_nash.py` hardcode `tau` (0.25 twice, 0.5 four times).
  Measured: `tau=0.25` sits at 0.749× the true `tau*` for the exact game that
  test file constructs (n=24, seed=0) — inside the region the module's own
  docstring says can have several equilibria.
- That risk is not theoretical: constructed a symmetric potential game (n=6,
  scaled so `tau=1.0` sits at 0.168× `tau*`) where three different starting
  points converge to real fixed points (residual 0.0 to solver precision) and
  two of the three land on a *different* equilibrium (max coordinate
  difference 0.982, near the largest possible in `[0,1]^6`).

`qre_stance` always starts at the barycentre, so any *single* call in this
codebase is deterministic — this is not presently corrupting a specific
number. What it means is that the well-posedness the docstring claims is
enforced only on the one call site that happens to pass `tau=None`
(`nash_operator`'s default). Every explicit-tau call, including the module's
own unit test suite, runs with no protection and no signal that it might be
reading one equilibrium out of several.

### 4. `tests/cameron/test_harmonic_attribution.py` cites a producer that does not exist (CONFIRMED, moderate-high cost — but scoped to a file DEIMOS may not edit)

This file (302 lines, arrived via the fast-forward merge) documents a full
"U1/N3" contract clause — a harmonic-measure kernel built on
`scale.e4_harmonic`'s real `case_graph`/`absorbing_chain`/`fixed_point`, a
masking-displacement probe, a rank cross-check, and a
`PREREGISTERED_RHO_FLOOR` frozen from named pilot numbers (*"spearman
(omega_rank, mean_displacement_rank) = 0.743864, bootstrap CI [0.656532,
0.816955]"*) — written in the same provenance style as every genuinely
measured entry elsewhere in this repo.

None of the nine names it calls on `scale.negation_scope`
(`absorbing_boundary_kernel`, `harmonic_measure`, `harmonic_label_batch`,
`train_control_arm`, `masking_displacement`, `rank_crosscheck`,
`dead_control_arm`, `u1_attribution_run`, `PREREGISTERED_RHO_FLOOR`) exist
anywhere in the tree. `python -m pytest tests/cameron/test_harmonic_attribution.py -q`
→ 11 failed, run live this session, every one a fresh `AttributeError`, not a
symptom of the known 146-failure baseline.

This is `MISTAKES.md` P-2 (*"a number with no live producer"*) with no floor:
P-2's cited instances at least have a commit message as a producer. Here
there is no producer of any kind for the pilot numbers quoted above — the
functions that would compute them were never written. `scale/e4_harmonic.py`
itself is not implicated (its real functions are called correctly, read-only,
by this file's helpers); the defect is entirely in the test file and its
non-existent integration layer. Filed here because it is the sole exerciser
of `e4_harmonic`'s absorbing-chain machinery beyond `e4_harmonic.py`'s own
`report()`, and because Mars's brief named `e4_harmonic*` as the surface.
`tests/cameron/test_harmonic_attribution.py` and `scale/negation_scope.py`
are both outside DEIMOS's edit rights — reported, not touched.

### 5. `case_graph`'s `lru_cache` shares a mutable adjacency (CONFIRMED, low cost today — dormant)

`scale/e4_harmonic.py:115` wraps `case_graph` in `functools.lru_cache(maxsize=4)`;
the function returns `adjacency: list[set[int]]`, a mutable structure that
`lru_cache` hands back by identity on every later call with the same
arguments. Confirmed live: mutating one caller's `adj[0]` is visible on the
very next `case_graph` call with the same arguments. Read through every
function in `e4_harmonic.py` that touches `adjacency` (`absorbing_chain`,
`local_features`, `cheeger_t_rel_floor`, `_bridge_side`) and every current
caller (`tests/cameron/test_e4_harmonic.py`, `tests/cameron/test_harmonic_attribution.py`,
both read-only) — none mutates it, so nothing is corrupted today. It is a
footgun for the next caller that writes a "cut this edge" probe in place,
which is the natural way to write one.

### 6. `calibrate()`'s `max_peak` clamp (CONFIRMED, low cost — dormant)

`eprocess.py:435`, `max_peak = math.exp(min(max_log, 700.0))`. Same overflow
class as finding #1, but on the diagnostic-only `max_peak` field — none of
`ok_null`/`ok_planted`/`ok_broken` reads it, they all compare logs directly.
Confirmed dormant at every horizon actually shipped (`print_price`
horizon=1500 gives true max-log ≈ 606; `main()`'s default horizon=400 is
smaller still), confirmed live (not merely clamped but silently *wrong* —
off by orders of magnitude) once a horizon around 5000 is used. The CLI's
`--horizon` has no upper bound, so this is one flag away from mattering.

## What DEIMOS did not validate

The batch-shared-tau finding (#2) is measured on one seed (0) of one corpus
construction; it was not re-measured across multiple seeds, so "56.5%
over-damped" and "mean stance roughly halved" are point estimates, not a
distribution — a second seed could in principle land less badly, though the
mechanism (`.max()` over the batch) guarantees the *direction* of the effect
on any batch with genuine norm spread. It was not isolated as the *sole*
cause of the Nash arm's 5.27 OOD NRMSE; other explanations (architecture,
training budget, the estimand itself) were not ruled out, and finding #2's
report says only that it is *a* cause the codebase's own sibling file already
flags, not *the* cause.

The multi-equilibria construction (#6 in the ledger, part of finding #3) is
one hand-found instance (seed 159 of a 200-seed scan at a specific scale
factor) demonstrating the phenomenon is real, not a distributional claim
about how often it occurs on the games this codebase actually draws in
practice — no production call site was found that currently passes an
explicit low `tau` on a live (non-test) game, so the immediate blast radius
is the unit test suite's own unenforced assumptions, not a live numeric
result.

The `Eprocess.value` overflow (#1) was demonstrated on a synthetic stream
built to match the shipped default pool size and a plausible effect size; it
was not reproduced by actually training and pooling five real Nash-arm (or
settled/twin) seeds end to end, since no journal in this tree currently has
that data at n_eval=2048 pooled. The claim is about the *function*, not about
a specific historical run having already crashed in production.

`e4_harmonic.py`'s rungs 8 and 32 sitting in the PASS/FAIL "limbo" the
module's own `report()` already discloses (`:371-374`) were read and are
accurately self-described in the source; DEIMOS did not find an additional
defect there beyond what the module already states, and did not chase
whether `e4_harmonic`'s ladder is registered anywhere as a capability claim
(a targeted grep found no such registration in `M3_TASKS`).

## Addendum — the instrument bit the hunter (post-merge, fast-forwarded to bdc65e1)

`test_qre_residual_is_reported_but_never_read_anywhere_in_the_repo` flipped
RED on merge into the primary checkout, where nine agent worktrees live
nested under `.claude/worktrees/*`, each a full copy of the tree. The test's
own walk was `pathlib.Path(__file__).resolve().parents[2].rglob("*.py")`,
excluding only `.git` and `node_modules` by directory-part name. Measured by
the coordinator on the integrated tree: **12 hits** for `return_residual` —
**10 phantom** copies of `ceq/nash.py` nested under `.claude/worktrees/*`,
**1 self-hit** (this file quotes the parameter name in its own docstring),
**1 real** (`ceq/nash.py` itself). The underlying claim was never false —
`return_residual` is still consumed nowhere outside its own definition — the
walk over-counted the corpus it searched.

### The new mistake type, for `MISTAKES.md` (Saturn owns the file; this is the write-up to fold in)

**Class: a repo-wide search whose walk includes nested checkouts, reporting
phantom hits and inverting a true absence claim into a false presence.**

This is `MISTAKES.md` category V (vacuous controls) with the sign flipped.
V-7 is a search "structurally incapable of finding anything, read as
absence" — it under-counts and reports zero when the true count is nonzero.
This is the mirror: a search structurally incapable of NOT finding things it
should not have looked at, reporting a nonzero count built mostly out of
copies of the one real hit. Both failures share a cause — the search was
never run against a fixture that could tell it apart from the ground truth —
and both destroy the same kind of claim: "X appears exactly once / X appears
nowhere" is precisely the shape a phantom hit or a missed hit corrupts.

**Instance.** `tests/deimos/test_deimos_r9_iteration1.py:246-264` (pre-fix).
`root.rglob("*.py")` from the repo root, filtered only by directory-part name
against `(".git", "node_modules")`. Any file nested under
`.claude/worktrees/<agent>/...` — a full checkout of the tracked tree, one
per concurrently-dispatched agent — passes that filter and is walked, read,
and matched exactly as if it were the one real file at that relative path.
Measured on the integrated tree: 12 hits, 10 phantom (all
`.claude/worktrees/*/ceq/nash.py`), 1 self (the test file's own docstring
quoting the searched-for name), 1 real (`ceq/nash.py`). The assertion
`hits == [Path("ceq/nash.py")]` failed not because the finding was wrong but
because the corpus the walk searched was ten times too large, in a way fully
determined by how many other agents happened to be checked out at merge
time — the test's pass/fail depended on a number (concurrent worktree count)
that has nothing to do with the claim under test.

**Why `rglob`/`os.walk`/`Path.glob` from a repo root are the wrong tool for
"does this repository contain X".** A filesystem walk answers "what is on
disk under this path", which is a superset of "what is in this repository"
the moment anything not tracked by version control — a nested git worktree,
a build directory, a venv, a cache — lives inside the walked tree. The gap
between the two grows every time a new agent worktree is created, so a
search built this way gets *more* wrong over time without any code change,
purely as a side effect of how many agents are concurrently dispatched. A
blocklist of directory names (`.git`, `node_modules`, and now `.claude`)
chases the symptom one name at a time and will always be one name behind the
next nested-checkout convention.

**Design rule.** A search over "this repository" should be scoped to the
git-tracked set (`git ls-files`), not to a filesystem subtree. This is not
merely a bigger blocklist: `git ls-files` is authoritative regardless of
what untracked directories exist beneath the search root, needs no
maintained exclusion list, and gets the answer right even for
nested-checkout conventions that do not exist yet. Where a walk must be
filesystem-based (scanning generated artefacts that are legitimately
untracked, for instance), it must be scoped to a subtree that is
structurally incapable of containing a nested checkout — `results/` rather
than the repo root — and that structural argument belongs in a comment
beside the walk, not left implicit.

**Fix applied** (`tests/deimos/test_deimos_r9_iteration1.py`, this session):
replaced the `rglob` walk with `git ls-files -- '*.py'` run at the repo root,
dropping the directory-name blocklist entirely. Re-run in this worktree:
`python -m pytest tests/deimos/ -q` → 8 passed. Not re-measured against the
actual 9-worktree integrated tree (this worktree is isolated and does not
itself contain `.claude/worktrees/*`), but `git ls-files` is structurally
immune to the defect class regardless of how many sibling worktrees exist,
since a linked worktree's files are never tracked blobs of the worktree that
spawned it.

**Sweep of the rest of the tree** for the same class
(`grep -rlnE '\.rglob\(|os\.walk\(|\.glob\(' --include=*.py`, plus a second
pass for `glob.glob`/`iterdir`), four other hits, none live:

| File | Call | Verdict |
|---|---|---|
| `tests/chase/test_eprocess.py:230` | `(root/"results").rglob("*.jsonl")` | SAFE — scoped to `results/`, which cannot contain `.claude/worktrees/*` (that directory is created at the repo root, a sibling of `results/`, never nested under it) |
| `tests/chase/test_hub_package_hardening.py:538` | `pathlib.Path(REPO).glob("DONE*.md")` | SAFE — non-recursive (`glob`, not `rglob`); the pattern requires a filename starting with `DONE`, which nothing under `.claude/worktrees/` matches at the top level even if it were recursive |
| `scale/chase_struck_coverage.py:104` | `ROOT.rglob("*.md")` / `ROOT.rglob("*.py")`, `ROOT` = repo root | SAFE — already filters `if any(part in (".git", "__pycache__", ".pytest_cache", ".claude", ".benchmarks") for part in p.parts): continue` immediately after; this author already named `.claude` in the blocklist before this incident |
| `tests/chase/test_capability_table.py:280,296,309,311` | `pathlib.Path(out).iterdir()` | SAFE — non-recursive, single directory level, `out` is a test-local output/artifact directory, not the repo root |

No other author's file needed a fix. Per hard rule 2, none was touched —
`scale/chase_struck_coverage.py`'s existing `.claude` exclusion is reported
here only as corroborating evidence that the class is real and was already
anticipated once, not as a defect to file against it.
