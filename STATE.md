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
| iteration | **43 complete, 44 next** — max **80** |
| phase | **WORK-STOPPING — M2 RED; repair / prior-art / write-up only** |
| item in flight | **M2 RED, measured on shipped operator (-1.298, 0 flips at s>=128)** |
| LOCK lines | `LOCK M2 efadc390c93f` — re-verified iteration 1 against the archived copy: **byte-identical** |
| calibration | **GREEN** [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| promise | `HOLDFAST` — retires SEPARATRIX |

## Open REDs

**M2 - RED, work-stopping IN FORCE.** On the SHIPPED operator (`sgate`) the
pivot arm reads slope **-1.298** against clause 1's **-0.3** bar. The
iteration-1 SUPERSEDED ruling was made on `tgate` numbers and is corrected.
(**-1.826 stood here until iteration 35 struck it** - Cameron's number, which my
own measurement contradicted at iteration 20. The document the loop reads first
disagreed with CHECKLIST.md for fifteen iterations.)

**M5 - RED.** `||A^hops|| = 0.880500` at s=128, hops=2 - nonzero and O(1), so
the exactness hypothesis is unmet. **The 1.471448 printed here since iteration
33 was FABRICATED and is struck** (iteration 35; not reproducible at any of
1,800 settings). Value now pinned by `EXPECTED_TAIL`.

**M2' - RED.** All four routes sign-blind by the twin test.

## THE ONE NEXT ACTION (iteration 44)

**The write-up is complete and internally consistent. The remaining work is not
scientific, it is delivery** - and the decision belongs to the user.

State of the record: M2 RED and **instrument-cleared, permanently**; M2' RED on
all four routes with **G1 firing for R3** (Tropical Attention, arXiv:2505.17190,
is R3 and published); M5 RED with its hypothesis pinned by test; nine documents
agreeing; **five value binds green**; Lean **27 theorems, exit 0, zero sorry**;
resume **bitwise-verified**. **No live theory, no instrument under suspicion.**

Iteration 44 should prepare the push and stop short of making it: verify the
tree is clean, confirm no Claude attribution anywhere in the commit messages
(the `github-master` rule, which **overrides** the global CLAUDE.md trailer),
and put the one-line summary in front of the user. **Name: `resolvent`.
Private.** The push itself is the user's call and has been since iteration 34.

### Iteration 43 - PROGNOSIS.md now carries iterations 35-42

**277 -> 383 lines**, bind still green. Added: M2's permanence with the paired
floor table (**kill fires at `floor = 0`, -1.1150**; removing the gate makes it
**steeper**); the floor discarding **6.8% -> 52.6% -> 100%** of sign changes, so
`0.00000` at s=128 overstates; the two-solid-point derivation (**-1.2977** vs
**-1.2980**, zeros worth 0.0003); the fabricated `1.471448` and the `> 1e-3`
**inequality** that let it ride through fourteen passing tests.

**And a new section that is the most transferable finding here:** sorted by what
they compared, **7 instruments compared STRUCTURE and all 7 failed; 2 compared
VALUES and neither ever has.** A structure check tests a proxy, and proxies drift
when the surrounding text is reformatted, refactored or piped.

### Correction to my own iteration-39 entry

I recorded, in a RUN voice, that the random-init scope fact was *"not recorded
anywhere"*, citing **0 hits in PROGNOSIS.md**. **False.** `PROGNOSIS.md:277`
already said *"Everything is measured on random projections, not trained
checkpoints."* My grep tried three phrasings; the document used a fourth.

The gap was real and testing it was right — **what was wrong was the claim of
novelty.** DONE.md is append-only, so iteration 39's entry stands and the
correction is appended beside it.

**Eighth structure-level defect, and the mirror of iteration 34's.** That one
searched for what should be there and declared consistency. This one searched for
what should be there and declared a recorded fact missing. **Same defect: a
string search tests a proxy for a fact.**

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
