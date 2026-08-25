# State

**Updated: 2026-08-25 — ROUND 2, ITERATION 1 complete.**

Governing documents, in precedence order:
1. `CHECKLIST.md` — work-stopping. Outranks everything.
2. `LOOP_PROMPT.md` — the per-iteration law. 80 iterations, promise `HOLDFAST`.
3. `CONTRACT.md` — the challenge, the arsenal with hooks and kills, the doctrine.
4. `DONE.md` — append-only. Round 1 archived at `DONE_ARCHIVE_ROUND1.md` (5,922 lines).

---

## Loop position

| field | value |
|---|---|
| iteration | **15 complete, 16 next** — max **80** |
| phase | **Phase 0** (iterations 1–3: instruments before numbers) |
| item in flight | **M2'** — appended verbatim, UNTESTED |
| LOCK lines | `LOCK M2 efadc390c93f` — re-verified iteration 1 against the archived copy: **byte-identical** |
| calibration | **GREEN** [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| promise | `HOLDFAST` — retires SEPARATRIX |

## Open REDs

None. **M2 is SUPERSEDED, not RED** — clause 1 did not fire (+0.0270 against a
−0.3 bar), clause 3 did not fire, and clause 2 is unevaluable by construction.
A claim whose safeguard cannot be evaluated has not survived it, so M2 leaves
the live set with its defect report rather than passing. The work-stopping
clause does not fire.

## THE ONE NEXT ACTION (iteration 16)

**Sweep softmax's budget on M3 and find out whether the 1.0 bar is reachable at
all.** Vary steps and n-train; the question is binary and decides whether four
more arms are worth running:

- **eval NRMSE crosses 1.0** -> the bar is real, softmax's 1.725 was a budget
  artifact, and the comparison becomes meaningful at that budget;
- **never crosses** -> **M3 is a termination clause wearing a gate's clothes**,
  it fails every arm identically regardless of operator, and it must be recorded
  as such rather than run four times to produce four identical failures.

### Iteration 15 - Health Inspector: 4/4 CLEAN, 0 struck

| check | result |
|---|---|
| calibration `--self-test` | exit 0, wrong target rejected first, 4/4 bit-identical |
| LOCK vs **archived copy** | `efadc390c93f`, `archived==live: True` |
| replay, bitwise, **third journal** | `r2` -> `tgate/s8` **BITWISE MATCH** |
| audit | **0 struck** |

The LOCK check matters more this pass than last: **two items have been appended
since iteration 1** (M2' and M2''), which is precisely the situation that made
the naive slice-boundary check cry wolf. The archived-copy method held.

The replay deliberately used **`r2`, a journal never replayed before** -
re-running `m2` a fourth time exercises one code path and audits nothing new.
**Three independent journals (`m2`, `s2`, `r2`) have now each reproduced bitwise
from a fresh process.**

**One thing the pass surfaced, recorded not acted on:** `r2`'s `coh_dense` and
`coh_pivot` are **identical to 16 digits at s=8** (0.7044318334094577 both),
then diverge (s=128: 0.2675 vs 0.7556). That identity is *expected* - at s=8
nearly every token is a pivot, so routed and dense hop-2 are the same
computation. It is the same boundary condition under which the M2 claim arm and
its dense control both read **0.024658**. Two unrelated instruments agreeing
there is a cross-check neither was built to provide.

## Board in flight

Three fellows are running with nurses (inference engineers: cheat hard, declare
every cheat). Their files are appearing — `scale/route_dependency.py`,
`scale/r2_units.py`, `scale/m3_capability.py`. Do not edit those.

- **Foreman** — are R1–R4 four routes or one mechanism in four costumes; which
  kills can structurally fire.
- **Chase** — build and break R2, the cheapest route.
- **Cameron** — the fifth route nobody costed, and the cheapest path to a
  *capability* rather than another statistic.

## Cheap wins queued (from DONE.md, ranked by value per hour)

1. ~15 lines of checkpoint/resume in `ceq/hf/train.py::train()` — unblocks every
   long run including the 300M gate.
2. The free 25.7M T4 Colab run — **7.0× above the 3.65M ceiling**, fits a free
   T4, one 12-hour session.
3. M5 is nearly GREEN — 27 theorems, build exit 0, zero `sorry`; needs the
   grep-bind from theorem hypothesis to shipped tensor.
4. M1 is probably GREEN from existing data — needs the frozen-zero-gradient-cell
   check.

## Machine

Windows, Python 3.11.9, torch 2.5.1+cu121, RTX 4060 Laptop, 8.0 GiB, 24 SMs.
CPU-only unless a test requires cuda. **Never concurrent CUDA jobs** — five
already killed two measurement runs at 88–89 °C. **No Bash call over 10 minutes
completes**; everything long goes through `scale/bucket.py`.
