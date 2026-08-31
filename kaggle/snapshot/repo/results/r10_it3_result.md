# R10 P0 iteration 3 — spot-check RESULT

Protocol: `results/r10_it3_protocol.md`, committed at `518197b` before any drawn row was measured
(caveat 1 corrected at `34d4bb1`, no measurement rule changed). Draw:
`scale/spotcheck_draw2.py` SEED=10003 pinned at `f823b02`, reproduced byte-identical.

## Verdict

**0 KEEP failures of 30. Threshold was 4 (alpha = 0.0608). THE CENSUS STANDS.**

30 ATTIC/KEEP rows measured, 5 ATTIC rows all DEAD as stated. With 0 failures in 30, the one-sided
95% upper bound on the sheet's error rate is **p <= 0.0950** — the sheet is **>= 90.5% right at 95%
confidence**. That is the whole of what this draw licenses; it is not a claim that the sheet is
95% right.

## The 35 rows

| path | type | expected | measured | verdict |
|---|---|---|---|---|
| `LOOP_PROMPT_ROUND3_ARCHIVE.md` | D | 3/4 readings; superseded archive, provenance | 3/4 exact; self-labelled archive, 3 citers; D3 clean | PASS |
| `NOTES.md` | D | NO JOURNAL; named by 1 tracked file | named by 1 (`DONE_ARCHIVE_ROUND1.md`) exact; D3 clean | PASS |
| `README.md` | D | m3_quintuple_v2.jsonl (151 rec); named by 55 | journal 151 rec exact; named by 56; D3 clean | PASS |
| `REQUIREMENTS.md` | D | 1/2 readings; named by 16 | 1/2 exact; named by 14; D3 clean | PASS |
| `done5.md` | D | arm_a.jsonl (3 rec); provenance of its round | journal 3 rec exact; 12/16 readings reproduce; 5 citers | PASS |
| `tests/deimos/DEIMOS_REPORT.md` | D | NO JOURNAL; named by 4 | named by 3; D3 0 out-of-range, 4 untracked (amendment) | PASS |
| `scale/arm_s.py` | S | producer of arm_s.jsonl | import 0, clean, 6.0 s; 5 importers; journal 82 rec | PASS |
| `scale/chase_k3_ci.py` | S | producer of arm_a.jsonl | import 0, clean, 6.1 s; `__main__`; journal 3 rec | PASS |
| `scale/dfloor_probe.py` | S | 2/2 readings, no importer | import 0, clean, 4.2 s; `__main__`; 2/2 | PASS |
| `scale/e4_harmonic.py` | S | 4/16 readings; 6 importers | import 0, clean, 0.4 s; 6 importers | PASS |
| `scale/e_ladder.py` | S | producer of m3_quintuple_v2.jsonl | import 0, clean, 6.0 s; 4 importers; journals 151 + 23 rec | PASS |
| `scale/equilibrium_probe.py` | S | producer of equilibrium.jsonl | import 0, clean, 6.2 s; 1 importer; journal 9 rec | PASS |
| `scale/foreman_quantisation.py` | S | producer of foreman_theta_tv.jsonl | import 0, clean, 7.3 s; `__main__`; journals 3 + 27 rec | PASS |
| `scale/hyperbolic.py` | S | imported by 1 module | import 0, clean, 6.6 s; 1 importer exact | PASS |
| `scale/monge.py` | S | imported by 2 modules | import 0, clean, 7.0 s; 2 importers exact | PASS |
| `scale/s2_probe.py` | S | producer of m2.jsonl | import 0, clean, 4.8 s; 3 importers; journal 40 rec | PASS |
| `tests/cameron/test_bar_control_sees_the_arms_preprocessing.py` | T | 4 tests, production path | collects; **5 passed**, exit 0 | PASS |
| `tests/cameron/test_r4_compression.py` | T | 7 tests, production path | collects; **2 failed / 12 passed**; both reds R3 band T (`"""RED. This is the test that deletes R4.`) | PASS |
| `tests/cameron/test_r6_twodof_red.py` | T | 9 tests, production path | collects; **9 passed**, exit 0 | PASS |
| `tests/chase/k22.py` | T | helper imported by 4 modules | imports 0; 4 importers exact | PASS |
| `tests/chase/test_deq_divergence.py` | T | on the KNOWN_RED ledger 6x | collects; **3 passed, 6 xfailed**, exit 0; no XPASS | PASS |
| `tests/foreman/_ceq.py` | T | helper imported by 3 modules | imports 0; 3 importers exact | PASS |
| `tests/foreman/test_capability_and_theorem_share_no_object.py` | T | 7 tests, 1/1 readings | collects; **4 failed / 3 passed** = 7 exact; all 4 reds R3 band T | PASS |
| `tests/foreman/test_deq_redundancy.py` | T | refutation instrument, `test_claim_*` | collects; **6 failed / 4 passed**; all 6 reds R2 | PASS |
| `tests/foreman/test_l1_normalizer_obstruction.py` | T | refutation instrument, `test_claim_*` | collects; **8 failed / 4 passed**; all 8 reds R2 | PASS |
| `tests/foreman/test_r1_settling.py` | T | MEASURED RED, live rejection region | collects; **8 failed / 11 passed**; all 8 reds R3 band B, all inside the `# CLAIM AS WRITTEN -- RED` section (:56–:165), none below the `-- GREEN` banner | PASS |
| `tests/jupiter/__init__.py` | T | pytest machinery | imports 0; directory collects 49 tests | PASS |
| `tests/w7/conftest.py` | T | pytest machinery | imports 0; directory collects 18 tests; 11 importers | PASS |
| `tests/w8/test_w8_real_model.py` | T | 10 tests, production path | collects; **20 passed**, exit 0 | PASS |
| `tests/w9/test_w9_hopcache.py` | T | 4/4 readings, anchored | collects; **20 passed**, exit 0 | PASS |
| `scale/frustration_audit.py` | A-orphan | 0 importers, 0 artifacts | 0 front-door importers, 0 `results/` artifacts, 0 constructed journal names | DEAD |
| `scale/lastrow_bind.py` | A-orphan | 0 importers, 0 artifacts | same, all three legs | DEAD |
| `scale/lo_probe.py` | A-orphan | 0 importers, 0 artifacts | same, all three legs | DEAD |
| `tests/loop/test_theorem_hypotheses_hold_at_shipped_settings.py` | A-vacuous | own tensor; 0/9 readings; ran all-green | imports only `ceq.bench`; **0/9 exact under both rules**; 14 passed; 0 `test_claim_*`, 0 must-fire | DEAD |
| `tests/wilson/accum_divergence.py` | A-orphan | 0 tests, 0 importers, no journal | 0 test functions, 0 importers, no journal | DEAD |

## What route 3 decided, stated because it is large

Under iteration 2's T-b (routes 1 and 2 only) **three of these 30 rows fail**:
`test_r1_settling.py` (8 reds, no `test_claim_` prefix, foreman has no ledger),
`test_capability_and_theorem_share_no_object.py` (4 reds, no prefix),
`test_r4_compression.py` (2 reds, no prefix, cameron has no ledger). At iteration 2's threshold of
2 that is another rejection. **Route 3 is deciding this verdict**, and the case for trusting it is
the measured rejection region, not the outcome:

- structural, drawn rows held out: 1,201 test functions, **60 rescued (5.0%), 1,141 refused**, 21
  files split;
- empirical, pre-registered sweep (`tests/loop` + `mars` + `saturn` + `mercury`, no drawn row):
  22 reds, **9 rescued, 13 refused (59%)**;
- empirical, post-registration confirmation (`tests/chase`, no drawn row): 30 reds, **7 rescued,
  23 refused (77%)**;
- every rescue on every drawn row names its ground per node, and the three deciding rows were
  read by eye against their source.

## Amendments

1. **`workdone2.md` — the iteration-2 amendment WITHDRAWN.** Cell restored to 1/10. The iteration-2
   arm used a >= 6-significant-digit rule; the census uses >= 5 decimal places, deduplicated
   (54/64 exact on held-out rows against 32/64 for the runner-up, 9/64 for what iteration 2 used).
   The provenance ground is not empty. Committed `8bc3832`; REDs logged first.
2. `tests/cameron/test_bar_control_sees_the_arms_preprocessing.py` — 4 tests -> **5**.
3. `tests/cameron/test_r4_compression.py` — 7 tests -> **14**.
4. `tests/w8/test_w8_real_model.py` — 10 tests -> **20**.
5. Protocol caveat 1 — corrected to MARS's withdrawal of the "1.0000 for all 61" coverage claim.

**Not amended, deliberately.** README.md 55 -> 56, REQUIREMENTS.md 16 -> 14, DEIMOS_REPORT.md
4 -> 3 citers. My citation counter is not identified against the census's the way the reading
extractor now is, and writing an unidentified instrument's number into the sheet is exactly the
mechanism that produced amendment 1. The disagreements are reported; the cells stand.

## Open

1. **`NOTES.md` is the thinnest KEEP in the draw.** Its entire ground is one citer,
   `DONE_ARCHIVE_ROUND1.md`, which is itself a round-1 archive. It passes D2 as pre-registered
   (>= 1) and the count matches the sheet exactly. A ground of one citer that is itself superseded
   is a KEEP resting on a document nobody has re-read.
2. **The doc denominator residual is unexplained in its mechanism.** 10 of 64 rows miss the
   identified rule by +-1 to +-4 with no file touched since `f823b02`, so the census's extractor
   differs from `scale/doc_readings.py` in some boundary case not yet isolated.
3. **`tests/loop/test_theorem_hypotheses_hold_at_shipped_settings.py` carries a declaration.**
   Route 3 finds 1 of its test functions declaring RED while the file runs 14 green. The ATTIC
   stands — a declared red that is green delivers nothing — but the L-SCOPE vacuity leg
   "0 `test_claim_*`" is blind to declarations, which is MERCURY's finding in miniature on an
   ATTIC row rather than a KEEP one.
4. **A 30-row draw that returns 0 failures cannot distinguish a good sheet from a permissive
   instrument on its own.** The rejection region above is what separates them, and it is measured,
   but it is measured on reds elsewhere in the tree rather than on this draw.
