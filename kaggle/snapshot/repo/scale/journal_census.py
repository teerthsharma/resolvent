"""Which thread counts reproduce which journalled units? Bucketed, resumable.

WHY THIS EXISTS. `LOOP_PROMPT.md` carried the rule *"the journal replays bitwise
only at OMP_NUM_THREADS=2"*, written in round 2 from ONE unit and generalised to
37. Measured at round 4: `dense_signed__at_pivots/s1024/b0` matches at 1 and 4
threads and **DRIFTS at 2** — exactly backwards. Different units have different
reduction shapes.

The bitwise replay is one of three instruments here that has never given a false
reading, but it ROTATES by iteration, so it samples one unit per pass out of 37.
A unit that happens to match at the pinned count is indistinguishable from a
sound journal. This census replaces the sample with the population.

BUCKETED BECAUSE TWO ATTEMPTS AT A WHOLE RUN DIED. A whole-census attempt timed
out at 10 minutes, and earlier an unbucketed training run died leaving two 0-byte
files. ADR-001 exists exactly for this: append-only journal, wall-clock budget,
resume from what is already recorded. A death here leaves evidence.

WHAT EACH OUTCOME MEANS, fixed before running:
  * ONE count reproduces all 37 -> pin it, fix `JOURNAL_THREADS`, instrument sound
  * different units need different counts -> bitwise replay is NOT available as a
    single-setting check and must record the count per unit
  * some unit reproduces at NO count -> its code changed after journalling, and
    the journal is STALE for that unit

`rate` is reported separately from `sigma`/`term`. `rate` is an integer count and
is what every published number rests on; the drift found so far is confined to
the two float reductions. Conflating them would overstate the defect.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

CENSUS = pathlib.Path(__file__).resolve().parents[1] / "results" / "journal_census.jsonl"


def journalled() -> dict:
    src = pathlib.Path(__file__).resolve().parents[1] / "results" / "m2.jsonl"
    out = {}
    for line in src.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["key"]] = r["value"]
    return out


def done() -> set:
    if not CENSUS.exists():
        return set()
    s = set()
    for line in CENSUS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            s.add((r["key"], r["threads"]))
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=420.0,
                    help="wall-clock seconds; stop cleanly before the ceiling")
    a = ap.parse_args()

    import torch
    threads = torch.get_num_threads()
    from scale.m2_units import compute, units

    j = journalled()
    u = dict(units())
    keys = sorted(k for k in u if k in j)
    have = done()
    CENSUS.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ran = 0
    for k in keys:
        if (k, threads) in have:
            continue
        if time.time() - t0 > a.budget:
            print(f"  budget reached, stopping cleanly after {ran} units", flush=True)
            break
        got = compute(u[k])
        want = j[k]
        rec = dict(key=k, threads=threads,
                   rate_match=got.get("rate") == want.get("rate"),
                   full_match=json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True),
                   got_sigma=got.get("sigma"), want_sigma=want.get("sigma"))
        with CENSUS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        ran += 1
        print(f"  [{threads}t] {k:<40} rate={'M' if rec['rate_match'] else 'D'} "
              f"full={'M' if rec['full_match'] else 'D'}", flush=True)

    tot = len(keys)
    seen = done()
    print(f"\n  census progress at {threads} threads: "
          f"{sum(1 for kk, tt in seen if tt == threads)}/{tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
