# ADR-001: Bucketed, resumable measurement units

**Status:** Accepted
**Date:** 2026-08-25 (loop iteration 13)
**Deciders:** Teerth Sharma

## Context

M2 was launched as one long-running process covering 36 cells (6 rows × 6 sizes,
4096–16384 draws each, ~2–3 h total). It died silently after 588 s with **one row
of six complete** and no artifact written. Three separate failures combined, and
each of them is a general hazard, not a one-off:

1. **The harness caps a Bash call at 600 000 ms (10 min).** The run was killed by
   my own `timeout` argument at ~600 s. Any measurement longer than 10 minutes
   *cannot* complete in a single call. This is a hard platform constraint, not a
   tuning parameter.
2. **`python … | tee file` reports tee's exit status, not python's.** The harness
   logged **exit code 0** for a killed process. A truncated run was presented as
   a completed one — the exact shape of the thirteen instrument failures already
   on record.
3. **All state lived in process memory.** `results/m2.json` was written only at
   the very end, so 588 s of correct computation — including a full 16 384-draw
   cell at s=1024 — was destroyed rather than banked.

Earlier in the same run, two further hazards appeared: a duplicate process
(`nohup` orphan) doubling the work undetected, and a 4-draw timing sample used to
budget a 16 384-draw cell.

The forces at play: measurements are long and getting longer (M3 trains models);
the loop makes exactly one action per iteration; and the project's governing rule
is that a result must be reproducible and bit-identical on re-run.

## Decision

Restructure every long measurement as **idempotent work units journalled to an
append-only file, executed in wall-clock-bounded buckets, one bucket per loop
iteration.**

A *unit* is the smallest independently meaningful measurement — here, one
`(arm, placement, s, n_draws, seed)` tuple. A *bucket* is however many whole
units fit inside a stated wall-clock budget comfortably below the platform cap.

## Options Considered

### Option A: One long process (status quo)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost | Total loss on any interruption |
| Scalability | **Fails hard past 10 min** |
| Team familiarity | High |

**Pros:** simplest code; one command; no journal format to design.
**Cons:** impossible under the 600 s cap; no partial credit; masks its own death
behind a pipeline exit code; a duplicate launch is undetectable.

### Option B: Bucketed units with an append-only journal (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium — a journal, a skip rule, a budget check |
| Cost | Loses at most one in-flight unit |
| Scalability | Unbounded — buckets compose across iterations |
| Team familiarity | High — this is checkpoint/resume |

**Pros:** survives interruption; progress is visible per unit; fits the loop's
one-action-per-iteration rule exactly; a completed unit is never recomputed, so
re-running is cheap and *proves* determinism; a lock file makes duplicate runs
impossible instead of merely unlikely.
**Cons:** the journal is now a correctness surface — a wrong unit key would
silently skip or duplicate work.

### Option C: Reduce the measurement until it fits

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost | **Pays in evidence** |
| Scalability | None |

**Pros:** trivial.
**Cons:** the draw counts were pre-registered. Cutting them to fit a platform
limit is a silent cap, which the loop's own rules forbid — and it is how a weak
result gets reported as a strong one.

## Trade-off Analysis

A vs B is **simplicity against recoverability**, and the constraint decides it:
under a hard 10-minute cap, Option A cannot produce the artifact at all. That is
not a trade-off, it is a disqualification.

B vs C is **engineering effort against evidential strength**. C is cheaper today
and destroys the thing being bought. The pre-registered draw counts are the
reason any M2 number will be believable; spending them to avoid writing a journal
is the worst trade on the board.

The real cost of B is that the journal key becomes load-bearing. Mitigated by
making the key the **full parameter tuple** and asserting on replay that a
recomputed unit matches its journalled value bit-for-bit — which converts the new
risk into a new *determinism check* the project did not previously have.

## Consequences

**Easier**
- Long measurements complete across iterations without heroics.
- Partial progress is durable; the s=1024 cell would have been banked.
- Re-running is a free determinism audit: recompute any unit, compare to journal.
- Duplicate execution becomes impossible (lock file), not merely unlikely.

**Harder**
- Two new correctness surfaces: the unit key and the journal writer.
- Aggregation is now a separate step and must never silently aggregate a partial
  set — it must report `done/total` and refuse to emit a verdict while incomplete.

**To revisit**
- If a *single unit* ever exceeds the wall-clock cap, units must be split further
  (e.g. by draw batch), and per-unit accumulation becomes necessary.

## Engineering tactics adopted

| Tactic | What it prevents (all observed here) |
|---|---|
| **Bucketing** by wall-clock budget under the platform cap | the 600 s kill |
| **Append-only journal**, one line per completed unit | losing 588 s of correct work |
| **Idempotent units** keyed by the full parameter tuple | recompute drift; silent skips |
| **Never pipe through `tee`**; capture the real exit code | exit 0 from a killed process |
| **Lock file** with PID | the `nohup` orphan doubling the work |
| **Replay assertion** — recomputed unit must match journal bitwise | undetected nondeterminism |
| **`done/total` accounting; refuse a verdict while partial** | reporting a truncated run as complete |
| **Budget from ≥100-draw timing**, or label the estimate a lower bound | the 4-draw sample |

## Action Items

1. [x] Write `scale/bucket.py` — journal, lock, unit keying, budget loop.
2. [ ] Re-run M2 through it; the completed s≤1024 `c ∈ P` cells re-enter as
       journal seeds and are verified by replay rather than trusted.
3. [ ] Aggregate only when `done == total`; the verdict function must refuse
       otherwise.
4. [ ] Apply the same runner to M3, which trains models and will be longer still.
