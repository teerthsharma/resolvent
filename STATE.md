# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **20 complete, 21 next** |
| phase | **ARM A under the escalation chain; autonomous to iteration 30** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 21)

**Close the self-satisfying provenance bind. It is the last live hazard, and the
Health Inspector has been auditing that file long enough that racing him is no
longer the risk - shipping the hazard is.**

**THE HAZARD, from Wilson [r5 iter 15]:** `1.44x`, `1.0334`, `0.379x` and
`3,319,296` now appear in `DONE.md` exactly once each - **inside the text of the
finding that reported them missing.** `tests/chase/test_hub_package_hardening.py`
asserts `"3,319,296" in done`. **That assertion can now be satisfied by the report
of the absence itself**, and Wilson named the sharper form: were `1.74` added, the
remaining provenance assertions would pass **only on F4's own report text**.

**A provenance test must match a MEASUREMENT, not a mention.** Two repairs are
available and the second is the real one:
  1. Point the bind at `DONE_ARCHIVE_ROUND1.md`, where the measurements actually
     live (counts **2 / 7 / 2 / 4**, verified twice).
  2. Require the number to appear in a line that also carries **run evidence** -
     a table row or a `[RUN]` marker - so a mention inside a meta-discussion
     cannot satisfy it. **This is the one that fixes the CLASS**, not the
     instance.

**Then, and only then, the prognosis.** `D1.md` exists with the round-5 chapter;
its closing limit already states that **nothing in it has passed an independent
audit**. `TWOSPHERES: BROKEN` is what the evidence supports - K1's displacement
clause resolved at **-0.4137 [-0.4579,-0.3704]** and **-0.4654 [-0.5173,-0.4160]**,
both entirely below **-0.30** - but **the promise is emitted AFTER the audit lands
and its strikes are applied**, not before.

**Dr House stays in the box.** Every kill this round fired from a measurement with
an interval. *"Dead is dead, and no leap un-refutes a fact."*

## Open REDs

None new. Round-4 carries: F-core (every additive route dead), F-cover (arm A
redirected to an additive basis of order 2), F-journ (13/37 drift in sigma/term,
rate intact 37/37).

## Carried, and load-bearing

- **F-green is the only positive result in four rounds, and it is UNSIGNED.**
- **A sign measurement without its logit scale is not a measurement.**
- **n_train >= 8192 or the reading ranks overfitting.**
- **No multiplication inside a sign decision** (G8, new) - `lo*hi` underflows to
  exactly -0.0 in float32 while both factors are healthy.
- **A leap binds before it counts.** House's displacement frame is MOTIVATION
  until a fellow writes the RED test.

## Board in flight

G1 fetches -> ARM A torque probe (K1 dual slope, K2 filler twin, K3 geometry
earns itself) -> G-b reproducible summation in parallel -> ARM B birth gates ->
M3 at n_train=8192.
