# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **14 complete, 15 next** |
| phase | **ARM A under the escalation chain; autonomous to iteration 30** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 15)

**Iteration 15 is an Inspector pass (every 5th) - and Chase aimed a finding
straight at it that must be acted on, not just recorded.**

**CHASE F5: the Inspector's clean bill covers 107 of 1278 tests - 8.4%.** He
probed two files inside the blind spot and found **8 live failures**, including
**a struck constant (`-1.389`, `0.9938`) still PINNED by
`tests/chase/test_hub_package_hardening.py:496-499`** while the struck-registry
test asserts the opposite. *"INSPECTOR PASS - exit 0, CLEAN"* is true **and is not
a statement about the repository**, and this project has written that sentence
into `DONE.md` five times.

So iteration 15 runs `python inspector.py` **and** decides what to do about its
coverage. Options, in order of cost: widen `check_suites` beyond
`tests/loop tests/w11 tests/chase/test_resume_checkpoint.py`; or make the
Inspector **report its own coverage fraction in its output**, so a clean bill can
never again be read as a statement about the suite. **The second is cheap and
removes the misreading permanently.**

**Also owed and now cheap:** Chase's F3 says two shipped tests make **logically
opposite demands on the same dict**. That is not a coverage problem, it is a
**live contradiction in the test suite**, and the registry test passing while the
other fails is exactly how a struck constant survives.

**WHAT IS SETTLED AND NEEDS NO MORE WORK:** the `max_row_mechanism` bind passes on
all six published fields; **the contradiction I reported was mine and is
withdrawn**; Cameron's mechanism is **refuted on the correct population**; the
aggregator win is **real, large, unexplained, and shared with TV**.

**Wilson holds three messages** and has `scale/wilson_probes.py` in progress.
**Never block on him.**

**Dr House stays in the box.** Nothing has died of missing innovation.

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
