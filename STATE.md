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
| iteration | **44 complete, 45 next** — max **80** |
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

## THE ONE NEXT ACTION (iteration 45)

**Health Inspector pass** (every 5th; 45 is due). Re-run calibration, verify the
LOCK, replay a journalled unit bitwise, run all five value binds, re-run one
published number (rotation 45 mod 4 = 1 -> the calibration table itself), and
confirm the nine documents still agree after the iteration-43 rewrite.

**Blocked on the user, and only this:** the push. Everything is staged for it -
`caf5eb7` on `master`, tree clean, `gh` authenticated as **teerthsharma**,
`repo` scope present, `teerthsharma/resolvent` free. **One word and it goes.**

### Iteration 44 - iterations 34-43 committed as `caf5eb7`

Message written under the `writing-pr-messages` skill: evidence-first, third
person, exact counts with provenance, limits collected once at the end.

**Two audits, both clean [RUN]** - the first-person/deference grep, and the
attribution grep. **The second is not the default:** global `CLAUDE.md` adds a
`Co-Authored-By: Claude Opus 5` trailer to every commit, and the `github-master`
skill **overrides** it for anything published through this token. Both commits
checked.

    0 uncommitted   228 tracked   2 commits   .git 13M
    gh: teerthsharma, scopes gist/read:org/repo/workflow
    teerthsharma/resolvent: does not exist -- name free, nothing overwritten

**The push is prepared and not made.** The authorisation exists and covers
incomplete work by its own wording. But **the repository is a different object
than it was when that instruction was given** - a candidate mechanism then, a
negative result now - and it would be created under the user's name. One
confirmation costs one iteration; the other direction cannot be undone.

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
