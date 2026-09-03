"""it.20 -- the citation census with `want` FROZEN, and pointers into growing files REFUSED.

it.17 shipped a landing instrument covering 30 pointers. it.18 added 3. MARS's
live RED at it.18 read `30 >= 123`: **121 of 129 citations in
`V20_R15_THEORY_TABLE.md` had never been individually opened by anything**, and
the rest were certified only by RESOLUTION -- the it.14 check, which asks whether
the file exists and whether `lineno` is inside it, and which passes on a pointer
that is off by 107 lines onto a different question.

This file opens **all 103 unique pointers (129 occurrences)** and freezes an
anchor for each: a verbatim substring of the line, chosen to carry the claim the
TABLE CELL makes at that citation, not merely to be present on the line.

TWO RULINGS ARE BOUND HERE, and both exist because the it.18 instrument, as
built, could be defeated by the office it audits.

  RULING J-20a -- THE WANT FREEZE.
  The INSPECTOR's it.18 audit: a landing check CANNOT detect a repair aimed at
  the wrong claim. `test_v20_r15_it18_citation_landing.py:59-67` tests
  `want in line_at(path, lineno)`, and `want` is supplied by the repairer in the
  SAME EDIT as the pointer. Move both together and a wrong-claim repair lands as
  cleanly as a correct one. This office's own J-18f is the same finding from the
  other side, and C98/C103 at it.17 is the instance: it.17 read a cell's claim as
  the "measured 1.4060346618513293" finding, repaired the pointer to the line
  carrying that number, and the sentence actually holding the citation was the
  `BED_SPECS` [RUN] in a third file again.
  So: `want` is FROZEN at census. `CENSUS` is the ground truth; a manifest under
  test whose `want` for a cid differs from `CENSUS[cid]` is NOT SCORED AS A
  REPAIR. It is a WITHDRAWAL plus a NEW CITATION, and the instrument names it as
  one. `WANT_SEAL` is a digest over the frozen wants, so an in-place edit of a
  `want` is a two-line edit that shows in the diff rather than a one-line edit
  that does not.
  This is TAMPER-EVIDENT, NOT TAMPER-PROOF. The seal lives in the same file as
  the thing it seals; a determined office can recompute it. What the freeze buys
  is that altering a claim can no longer be done silently inside a repair, and
  that is the whole of what it buys. Bounded again under LIMITS below.

  RULING J-20b -- A POINTER INTO A GROWING FILE IS REFUSED, NOT SCORED.
  The INSPECTOR measured the census moving DURING his it.18 audit: both suites
  green at `15:59`, RED at `16:11`, test file byte-identical, failing on
  `C17: V20_R15_JOURNAL.md:650 lacks 'THE FREEZE'`. This office's own `C20`
  append to `V20_R15_JOURNAL.md` moved the heading off by a third line. It is
  still off at this census: `V20_R15_JOURNAL.md:650` is now BLANK and
  `## it.4 -- THE FREEZE` sits at `:651`.
  `J-18a` said census TOTALS expire. That under-claims it. Every pointer into a
  file still being written expires, and the journal grows at its HEAD every
  iteration, so every journal pointer is aimed at a moving target.
  The ruling, and it is the second of the two options the round put, widened by
  what happened while it was being written:
  **the instrument records each cited file's line count at census time and
  REFUSES to score any pointer whose file has changed length -- and it refuses
  every pointer into a file declared LIVE outright, without waiting for the
  growth.** Not "marks it failing" -- REFUSES: it leaves the scored population
  and is reported as needing re-anchoring. A file the round appends to every
  iteration is not a file whose line count is worth recording; recording it only
  defers the refusal to the next append. The first option (cite prose by heading, the fix the
  CORRECTIONS INDEX already adopted for itself at `P-6`) is the correct repair
  for a withdrawn pointer and is recorded as such against C17; it is not adopted
  as the instrument's rule, because the instrument cannot check a heading
  pointer's LINE, and line is the whole subject.

MANIFEST PROVENANCE. Every anchor below was produced by opening the cited file at
the cited line at this census, in bulk, four readers. None is derived from the
table: a manifest generated from the thing it checks asserts nothing. The readers
returned the line and a proposed substring and were forbidden a verdict; the
choice of anchor is this office's.

COVERAGE REACHED, AND IT IS NOT ROUNDED:
  * 103 of 103 unique pointers opened and anchored -- 129 of 129 occurrences.
  * **116 of 129** occurrences are SCORED and LAND on their frozen anchor.
  * **13 of 129** are REFUSED under J-20b, every one into a file this round
    is still writing: `V20_R15_JOURNAL.md`, `V20_R15_LEAP_LEDGER.md`,
    `house-events.jsonl`. Their cids are listed in `WITHDRAWN` with the repair
    each needs. Each anchor was opened and verified correct at this census; they
    are refused for their FILE, not for their content.
  * 0 of the 116 scored occurrences fail. The census was not raised by
    lowering the bar -- every one was opened at its line, by hand, at this census.

J-20b WAS MEASURED FIRING TWICE DURING THIS ITERATION, WHICH IS WHY IT IS WIDER
THAN THE ROUND ASKED FOR. The first take recorded `V20_R15_JOURNAL.md` at 3890
lines and refused one pointer. Minutes later the same instrument, byte-identical,
read 3908: the journal had grown 18 lines mid-census. The second take then read
`V20_R15_LEAP_LEDGER.md` at 438 against the 406 it had recorded four minutes
earlier. That is the INSPECTOR's `15:59 green / 16:11 RED`, twice more, in two
more files, inside one 20-minute iteration -- and it is why the ruling is not
"re-anchor the pointer" but "refuse the file".

LIMITS.
  * The freeze is tamper-evident, not tamper-proof: `CENSUS` and `WANT_SEAL` are
    in one file written by the office under audit. No arrangement inside this
    repo fixes that; only a second office re-taking the census does.
  * An anchor certifies that the cited LINE carries the words the cell points at.
    It does not certify that the cell's ARGUMENT is sound, and it cannot: 60
    characters of a line is a location, not a proof.
  * `FILE_LINES_AT_CENSUS` detects length changes, not in-place edits. A file
    rewritten without changing its line count passes J-20b and is caught only by
    the anchor itself.
  * 21 of the 129 occurrences are into `scripts/v15_r1.py` and 18 into
    `lean/CEQ/V16Domain.lean`; a single rigid shift in either file moves a sixth
    of the census at once, which is the failure mode it.17 named and no
    resolvability check can see.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "V20_R15_THEORY_TABLE.md"

NEWLINE = chr(10)

CITE_RE = re.compile(r"`([^`]+?):(\d+(?:-\d+)?|\*)`")
#: it.24: the second group is a SPEC, not a lineno -- `13`, `13-14` (J-24b
#: range) or `*` (J-24a, the file, not line-landed).
EXT_RE = re.compile(r"\.(py|md|lean|jsonl|json|txt|toml|hs|ipynb|pgn)$")

#: The census population, RE-MEASURED at it.20. 123 named the pre-it.17 table;
#: it.17 took it to 128 by splitting two citations; C64's it.18 repair took it to
#: 129. It has not moved since. This is dated, not constant -- J-18a.
POPULATION_AT_IT20 = 131  # 129 at it.20; +2 at it.27, M-25b and M-25c each ADD a citation.
UNIQUE_AT_IT20 = 105  # 103 at it.20; `scripts/v15_r1.py:138` and `ceq/beds/bed_k.py:236` are new.

#: cid -> (path, lineno, want). `cid` is the citation's 1-based occurrence index
#: in the table at THIS census, which is why it agrees with MARS's C17 and C64 and
#: not with his C98/C103: the it.18 repairs moved everything after them. The index
#: is itself a pointer into a file that changes, and is dated for that reason.
#:
#: `want` IS FROZEN -- RULING J-20a. Changing a value here is a WITHDRAWAL.
CENSUS: dict[int, tuple[str, int, str]] = {
    1: ('V20_R15_WING_MANIFEST.md', 3, 'Frozen at it.4 on branch'),
    2: ('CEQ_V20_R15_CONTRACT.md', 58, 'L-GRADE (F0–F4 + HOW-BAD gap)'),
    3: ('V20_R15_IT12_INSPECTOR.md', 54, 'Ruling: the narrow claim is BOUND'),
    4: ('V20_R15_IT12_INSPECTOR.md', 247, "JUPITER's rejection IS BOUND"),
    5: ('V20_R15_IT12_INSPECTOR.md', 233, 'F3 = failed instance`, `F4 = unattempted` is NOT BOUND'),
    6: ('house-events.jsonl', 12784, '"event": "finding_contract"'),
    7: ('V20_R15_IT12_JUPITER.md', 173, 'OPERATIONAL RUBRIC THESE TWELVE GRADES'),
    8: ('CEQ_V20_R15_CONTRACT.md', 64, 'every sparsity mask ships with its certificate: F0'),
    9: ('V20_R15_LEAP_LEDGER.md', 28, 'F4** | unattempted'),
    10: ('V20_R15_LEAP_LEDGER.md', 25, 'withdrawn at it.2'),
    11: ('V20_R15_LEAP_LEDGER.md', 24, 'machine-true and domain-empty'),
    12: ('V20_R15_IT12_JUPITER.md', 189, 'F2 and F3 are not ordered'),
    15: ('V20_R15_JOURNAL.md', 645, 'measured on the corrected gate'),
    16: ('V20_R15_IT12_INSPECTOR.md', 260, 'M14 — CONFIRMED, and worse than reported'),
    17: ('V20_R15_JOURNAL.md', 650, 'THE FREEZE'),
    18: ('V20_R15_LEAP_LEDGER.md', 131, '1.421901019003236'),
    19: ('V20_R15_IT12_JUPITER.md', 229, 'and neither report noticed'),
    20: ('V20_R15_IT12_JUPITER.md', 215, 'Replacement route, and it is contract text, not a run'),
    21: ('scripts/v15_r1.py', 586, 'floor1 = math.sqrt((T_STAR - 1) / T_STAR)'),
    22: ('scripts/v15_r1.py', 17, 'h_hat = t*(1 - NRMSE^2)'),
    23: ('scripts/v15_r1.py', 137, 'S, D = 64, 24'),
    24: ('scripts/v15_r1.py', '547-558', '--seeds'),  # M-25a at it.27: POINTER repair, want untouched -- the count spans the block.
    25: ('scripts/v15_r1.py', 249, 't0 = time.time()'),
    26: ('scripts/v15_r1.py', 267, 'secs = time.time() - t0'),
    27: ('V20_R15_IT13_MERCURY.md', 146, 'RETIRE the re-take'),
    28: ('lean/CEQ/V16Domain.lean', 105, 'lemma pathProd_polar'),
    29: ('lean/CEQ/V16Domain.lean', 129, 'theorem pathProd_eq_zero_iff'),
    30: ('lean/CEQ/V16Domain.lean', 121, 'theorem pathProd_abs'),
    31: ('lean/CEQ/V16Domain.lean', 251, 'theorem bedM_gate_exact'),
    32: ('lean/CEQ/V16Domain.lean', 273, 'theorem negative_draw_is_on_the_band'),
    33: ('tests/jupiter/test_v20_r15_it6_q1_exact_class.py', '*', 'Q1 EXACT CLASS, two wings'),
    34: ('lean/CEQ/V16Domain.lean', 304, 'theorem bedM_overlap_new_two'),
    35: ('lean/CEQ/V16Domain.lean', 176, 'theorem pathProd_eq_Wp'),
    36: ('scale/negation_scope.py', 429, 'a[:, :head + 1] = 0.0'),
    37: ('lean/CEQ/V16Domain.lean', 165, 'theorem no_prefix_scan_represents_a_zero_gate'),
    38: ('lean/CEQ/V16Domain.lean', 147, 'theorem lean_log_junk_makes_the_scan_form_silently_false'),
    39: ('lean/CEQ/V16Domain.lean', 355, 'theorem sixteen_is_silent_on_the_zero_draw'),
    40: ('ceq/arm_pl.py', 88, 'def scan(g: torch.Tensor) -> torch.Tensor:'),
    41: ('ceq/arm_pl.py', 93, 'def key_bias'),
    42: ('lean/CEQ/V16Domain.lean', 302, 'theorem bedM_overlap_old_two'),
    43: ('lean/CEQ/V16Domain.lean', 370, 'noncomputable def Znorm'),
    44: ('V20_R15_IT12_JUPITER.md', 135, 'domain-empty for these wings'),
    45: ('ceq/hankel.py', 108, 'def hankel_block('),
    46: ('ceq/hankel.py', 131, 'def rank_real('),
    47: ('ceq/hankel.py', 279, 'def rank_plus_lower('),
    48: ('ceq/hankel.py', 75, 'NEG_ENTRY = '),
    49: ('tests/jupiter/test_v20_r15_it6_q2_outside_bound.py', 43, 'from ceq.hankel import NEG_ENTRY'),
    50: ('tests/jupiter/test_v20_r15_it6_q2_outside_bound.py', 75, 'def _r2_curve'),
    51: ('tests/jupiter/test_v20_r15_it12_constants.py', 12, '0.9746794345'),
    52: ('lean/CEQ/V16Domain.lean', 339, 'theorem delay_zero_is_first_order'),
    53: ('V20_R15_IT13_MERCURY.md', 63, "No cell of BED-K's shape has ever been run."),
    54: ('V20_R15_IT11_INSPECTOR.md', 502, 'THEOREM SOUND, CELL STRUCK.'),
    55: ('ceq/arm_pl.py', '*', 'ARM PL -- one causal softmax head'),
    56: ('tests/jupiter/test_v20_r15_it12_constants.py', 11, '1.0845223424'),
    58: ('scripts/v15_r1.py', 383, 'lambda_hat=float(lg.mean())'),
    59: ('scripts/v15_r1.py', 384, 'lambda_hat_live=(float(lg[fin].mean())'),
    60: ('scripts/v15_r1.py', 386, 'frac_gate_annihilated=float((~fin).double().mean())'),
    64: ('ceq/arm_smprime.py', 409, 'm["smp_values"] = values'),
    65: ('scripts/v15_r1.py', 804, 'r["sign_acc_0step"] = probe(f0t, torch.sign(a_tr),'),
    66: ('tests/jupiter/test_v20_r15_it7_q3.py', 106, 'def test_w3_lambda_hat_sign_separates_all_eight_arm_pl_cells'),
    68: ('V20_R15_IT8_JUPITER.md', 105, '`path_product` builds'),
    69: ('ceq/arm_smprime.py', 144, 'def path_product('),
    72: ('ceq/arm_smprime.py', 559, 'def zero_hop_mask'),
    73: ('ceq/sizing.py', 145, 'def flops_per_token'),
    74: ('scale/m3_flops.py', 207, 'CELLS = ("softmax", "glance", "settled", "twin", "argmax")'),
    75: ('ceq/mz_kernel.py', 170, 'def attention_flops'),
    80: ('ceq/arm_pl.py', 304, 'def brute_force_path_sums('),
    81: ('ceq/arm_pl.py', 113, 'w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])'),
    84: ('V20_R15_LEAP_LEDGER.md', 70, 'F1 + const on both wings, but on the wrong object'),
    86: ('scale/it11_verdict.py', 451, 'def hop_floor(t_star: int, hops: int = 1)'),
    88: ('tests/jupiter/test_v20_r15_it8_q4_q5.py', 127, 'def test_q5_floor_1_is_a_ONE_HOP_THRESHOLD_not_an_information_floor'),
    89: ('tests/jupiter/test_v20_r15_it12_constants.py', 122, 'def test_q5_w1_flat_band_distance_to_floor1_is_0_145_to_0_220'),
    91: ('tests/jupiter/test_v20_r15_it8_q4_q5.py', 151, 'def test_q5_no_arm_crosses_and_W3_is_blocked_by_variance_not_by_the_floor'),
    92: ('V20_R15_IT9_JUPITER.md', 265, 'exactly `{2,3,7,9}`'),
    93: ('CEQ_V20_R15_CONTRACT.md', 123, 'W1 to the oracle where a state distribution exists'),
    94: ('ceq/arm_smprime.py', 577, 'return self.readout(h).squeeze(-1)[:, seq - 1]'),
    95: ('scale/negation_scope.py', 304, 'return z'),
    96: ('scale/negation_scope.py', 286, 'equilibrium_oracle'),
    97: ('scale/foreman_consequence.py', 12, 'NON-NEGATIVE arm signs `0.492188`'),
    98: ('tests/jupiter/test_v20_r15_it9_q6.py', '*', 'Q6 STATE METRIC'),
    99: ('tests/jupiter/test_v20_r15_it11_q6_oracle.py', '*', "Q6's planted negative"),
    102: ('V20_R15_IT89_INSPECTOR.md', 47, 'STRIKE I-1 — the Q6 planted negative'),
    103: ('tests/jupiter/test_v20_r15_it9_q6.py', 155, 'KILLED at it.11'),
    104: ('V20_R15_IT13_MERCURY.md', 189, 'print(list(kdata.BED_SPECS))'),
    106: ('CEQ_V20_R15_CONTRACT.md', 125, 'CRITERION (lexicographic): (1) BED-M crossing'),
    107: ('V20_R15_IT8_JUPITER.md', 473, 'do not cite it on W1/W3 at all'),
    108: ('ceq/kdata.py', 475, '"generator": "ceq.beds.bed_k.build_delay"'),
    110: ('MISTAKES.md', 451, 'A threshold refitted to the data it judges'),
    111: ('V20_R15_IT12_INSPECTOR.md', 272, 'He omits **L-8**'),
    113: ('V20_R15_IT567_INSPECTOR.md', 482, 'Naming a field and then'),
    114: ('tests/saturn/test_v20_r15_it12_saturn.py', 319, 'def inadmissible_leapable_rows('),
    115: ('tests/saturn/test_v20_r15_it12_saturn.py', 329, 'if head.startswith("none") or not head:'),
    116: ('V20_R15_LEAP_LEDGER.md', 99, 'TERMINAL rows name no field'),
    117: ('V20_R15_LEAP_LEDGER.md', 130, 'calibration / probabilistic forecasting (CRPS, pinball loss)'),
    118: ('tests/saturn/test_v20_r15_it12_saturn.py', 343, 'KNOWN_INADMISSIBLE = '),
    119: ('tests/saturn/test_v20_r15_it12_saturn.py', 353, 'def test_the_FIELD_detector_fires_on_a_planted_violation('),
    121: ('lean/lakefile.lean', '*', 'import Lake'),
    122: ('lean/CEQ/V16Domain.lean', 75, 'No `sorry`. `#print axioms`'),
    127: ('tests/arm_smprime/test_arm_smprime.py', 559, 'arm.zero_hop_mask(x)'),
    128: ('tests/jupiter/test_v20_r15_it8_q4_q5.py', 222, 'assert "def zero_hop_mask" in src'),
    129: ('tests/jupiter/test_v20_r15_it14_theory_table.py', '*', 'the frozen theory table is its own instrument'),
    # it.24 RE-ISSUES. J-20a: C13 and C63 are WITHDRAWN, not edited. See REISSUED.
    132: ('CEQ_V20_R15_CONTRACT.md', '241-244', ('grade F3 pending', 'M14 IS F4')),
    131: ('tests/jupiter/test_v20_r15_it12_constants.py', '13-14', ('+0.717647', '-0.032353')),
    # it.27 ADDITIONS. MERCURY M-25b/M-25c: a clause GAINS a citation, it does not
    # move one. No cid is withdrawn here and no frozen `want` is edited.
    133: ('scripts/v15_r1.py', 138, 'T_STAR = 2'),
    134: ('ceq/beds/bed_k.py', 236, 'def build_delay('),
}

#: RULING J-24b/J-24c bookkeeping, the same shape as it.21's `REANCHORS`: new
#: cid -> the cid it replaces. The old cid is GONE from `CENSUS` -- a withdrawal
#: is not an edit -- and `not_a_repair` therefore names it as uncensused if any
#: manifest tries to carry it forward.
REISSUED: dict[int, int] = {130: 13, 131: 63, 132: 130}

#: Digest over the frozen wants. Recomputed by `test_the_want_seal_is_intact`.
#: J-20a: editing a `want` in place without touching this is RED.
WANT_SEAL = "abad6a77d9da0793c54d19f092a30005cb3daa40aa0dad77d8d44ce5e01212c4"  # RE-TAKEN at it.27: C133/C134 ADDED under M-25b/M-25c. No existing want was edited.

#: Cited file -> line count at census. RULING J-20b: a pointer into a file whose
#: length has changed is REFUSED, not scored. Every entry is a tripwire on a file
#: that may be appended to before the it.35 leap reads this table.
FILE_LINES_AT_CENSUS: dict[str, int] = {
    'CEQ_V20_R15_CONTRACT.md': 289,
    'MISTAKES.md': 2235,
    'V20_R15_IT11_INSPECTOR.md': 739,
    'V20_R15_IT12_INSPECTOR.md': 647,
    'V20_R15_IT12_JUPITER.md': 250,
    'V20_R15_IT13_MERCURY.md': 283,
    'V20_R15_IT567_INSPECTOR.md': 691,
    'V20_R15_IT89_INSPECTOR.md': 968,
    'V20_R15_IT8_JUPITER.md': 488,
    'V20_R15_IT9_JUPITER.md': 375,
    'V20_R15_WING_MANIFEST.md': 159,
    'ceq/arm_pl.py': 410,
    'ceq/arm_smprime.py': 586,
    'ceq/beds/bed_k.py': 397,
    'ceq/hankel.py': 448,
    'ceq/kdata.py': 737,
    'ceq/mz_kernel.py': 244,
    'ceq/sizing.py': 377,
    'lean/CEQ/V16Domain.lean': 594,
    'lean/lakefile.lean': 11,
    'scale/foreman_consequence.py': 550,
    'scale/it11_verdict.py': 608,
    'scale/m3_flops.py': 415,
    'scale/negation_scope.py': 1665,
    'scripts/v15_r1.py': 1031,
    'tests/arm_smprime/test_arm_smprime.py': 662,
    'tests/jupiter/test_v20_r15_it11_q6_oracle.py': 155,
    'tests/jupiter/test_v20_r15_it12_constants.py': 258,
    'tests/jupiter/test_v20_r15_it14_theory_table.py': 133,
    'tests/jupiter/test_v20_r15_it6_q1_exact_class.py': 176,
    'tests/jupiter/test_v20_r15_it6_q2_outside_bound.py': 223,
    'tests/jupiter/test_v20_r15_it7_q3.py': 112,
    'tests/jupiter/test_v20_r15_it8_q4_q5.py': 238,
    'tests/jupiter/test_v20_r15_it9_q6.py': 216,
    'tests/saturn/test_v20_r15_it12_saturn.py': 393,
}

#: Files the round APPENDS TO EVERY ITERATION. RULING J-20b: no pointer into one
#: of these is scored, ever, and none appears in FILE_LINES_AT_CENSUS -- recording
#: a length for a file under active write only defers the refusal by one append.
#: `V20_R15_JOURNAL.md` grew 18 lines and `V20_R15_LEAP_LEDGER.md` grew 32 between
#: takes of this census; `house-events.jsonl` is an append-only event log.
LIVE_FILES = frozenset(
    ["V20_R15_JOURNAL.md", "house-events.jsonl", "V20_R15_LEAP_LEDGER.md"]
)

#: WITHDRAWALS, by cid, with the repair each one needs. A cid here is out of the
#: scored population and stays out until it is re-issued as a NEW cid with its own
#: frozen want. All three are J-20b refusals, not typos: each anchor was verified
#: correct at census and each expires on the next append to its file.
WITHDRAWN: dict[int, str] = {
    6: ("house-events.jsonl:12784 -- REFUSED under J-20b (LIVE file: append-only event log; no line in it is stable for a citation). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
    9: ("V20_R15_LEAP_LEDGER.md:28 -- REFUSED under J-20b (LIVE file: under active write this round; grew 406 -> 438 during this census). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
    10: ("V20_R15_LEAP_LEDGER.md:25 -- REFUSED under J-20b (LIVE file: under active write this round; grew 406 -> 438 during this census). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
    11: ("V20_R15_LEAP_LEDGER.md:24 -- REFUSED under J-20b (LIVE file: under active write this round; grew 406 -> 438 during this census). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
    15: ("V20_R15_JOURNAL.md:645 -- REFUSED under J-20b (LIVE file: appended at its head every iteration; grew 3890 -> 3908 during this census). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
    17: ("V20_R15_JOURNAL.md:650 -- REFUSED under J-20b (LIVE file: appended at its head every iteration; grew 3890 -> 3908 during this census). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
    18: ("V20_R15_LEAP_LEDGER.md:131 -- REFUSED under J-20b (LIVE file: under active write this round; grew 406 -> 438 during this census). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
    84: ("V20_R15_LEAP_LEDGER.md:70 -- REFUSED under J-20b (LIVE file: under active write this round; grew 406 -> 438 during this census). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
    116: ("V20_R15_LEAP_LEDGER.md:99 -- REFUSED under J-20b (LIVE file: under active write this round; grew 406 -> 438 during this census). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
    117: ("V20_R15_LEAP_LEDGER.md:130 -- REFUSED under J-20b (LIVE file: under active write this round; grew 406 -> 438 during this census). The anchor was opened and verified correct at this census; re-anchor by heading (P-6), not by digit."),
}


def line_at(path: str, lineno: int) -> str:
    """The one source line a `path:lineno` citation points at. '' if unreachable."""
    p = ROOT / path
    if not p.is_file():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[lineno - 1] if 1 <= lineno <= len(lines) else ""


def region(path: str, spec: object) -> str:
    """The text a pointer ADDRESSES. it.24 widens `line_at` to three notations.

    `13` -> that line. `'13-14'` -> those lines, J-24b. `'*'` -> the whole file,
    J-24a. Returns '' if unreachable, so an unresolvable pointer fails landing
    rather than raising.
    """
    p = ROOT / path
    if not p.is_file():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if spec == "*":
        return NEWLINE.join(lines)
    if isinstance(spec, str) and "-" in spec:
        a, b = (int(x) for x in spec.split("-"))
        return NEWLINE.join(lines[a - 1:b])
    n = int(spec)
    return lines[n - 1] if 1 <= n <= len(lines) else ""


def table_citations() -> list[tuple[str, str]]:
    text = TABLE.read_text(encoding="utf-8")
    return [(p, n) for p, n in CITE_RE.findall(text) if EXT_RE.search(p)]


def grown() -> dict[str, tuple[int, int]]:
    """Every cited file whose length has moved since census. J-20b."""
    out = {}
    for path, n in FILE_LINES_AT_CENSUS.items():
        now = len((ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines())
        if now != n:
            out[path] = (n, now)
    return out


def refused() -> set[int]:
    """Cids the instrument will not score. J-20b: withdrawn, LIVE, or moved."""
    moved = grown()
    return set(WITHDRAWN) | {
        c for c, (p, s, _) in CENSUS.items()
        if s != "*" and (p in moved or p in LIVE_FILES)
    }


#: RULING J-27a -- `:*` IS SCORED AGAINST THE LINES THAT CARRY THE WANT.
#: MARS, it.25 STRIKE C: `region(path,'*')` is the whole file, so `want in text`
#: is a grep, and a grep cannot tell a want from its own obituary -- delete the
#: asserting line, append a line quoting the want, and presence is unchanged.
#: His proposed COUNT freeze does not close it: all six `:*` wants occur exactly
#: once, so the obituary reads 1 against a frozen 1 (measured it.26, re-measured
#: it.27). What the obituary moves is the CONTENT of the carrying lines.
#:
#: want -> sha256 over the NEWLINE-joined lines of the file that carry it, frozen
#: at the it.27 census. Keyed on the WANT and not the cid because `lands(want,
#: line)` is the signature four offices already call; widening it would restate
#: the rule in every caller, which is how it.21 shipped two copies of one rule.
#: Uniqueness of the six `:*` wants is ASSERTED in the it.27 instrument, not
#: assumed. J-24a survives: an unrelated append leaves the carrying lines alone.
STAR_DIGESTS: dict[str, str] = {
    'Q1 EXACT CLASS, two wings': '0623406a059fdd3cae91d342a93714261faa24c048d95411d13b55bb34baec72',
    'ARM PL -- one causal softmax head': '72858a546727268eb8d65855ca32bf771af5d9e41bb7f17a479b8ad7bb20b32b',
    'Q6 STATE METRIC': 'a66f4b0b931bccea6fd066728f739b15545d467dfeb36386fd6424d53e003e7b',
    "Q6's planted negative": '46d525c037ba578f4dfaeeffaa7191e1d7c71ded635f961c151f112c589a0261',
    'import Lake': '716d30769aeaea6a1331bafa0edd1e3011410c9f26f8c6351181962e2bf3cf49',
    'the frozen theory table is its own instrument': '783b341bb0d3a3c2ebefd79f40f6e63b6f2412398e9fe5c76cd99b9ad7aa2dd6',
}


def carrying_digest(want: str, text: str) -> str:
    """sha256 over the lines of `text` that carry `want`. The recipe for J-27a's
    freeze, here rather than in a filing, so it can be re-derived from the tree."""
    carrying = [ln for ln in text.split(NEWLINE) if want in ln]
    return hashlib.sha256(NEWLINE.join(carrying).encode("utf-8")).hexdigest()


def lands(want: str, line: str) -> bool:
    """THE landing predicate. it.20's rule, in one place, so it.21 can import it
    instead of restating it (INSPECTOR, it.20; MARS, it.22 STRIKE 2 tail).

    J-27a: a want with a frozen `:*` digest is scored against the LINES carrying
    it, not against presence anywhere in the file."""
    if isinstance(want, tuple):
        return all(lands(w, line) for w in want)
    if want in STAR_DIGESTS:
        return carrying_digest(want, line) == STAR_DIGESTS[want]
    return want in line


def not_landing(manifest: dict[int, tuple[str, int, str]]) -> list[str]:
    """Every SCORED manifest entry whose cited line does not carry its anchor."""
    skip = refused()
    return [
        f"C{c}: {p}:{n} does not carry {w!r}; region is {region(p, n)!r}"
        for c, (p, n, w) in manifest.items()
        if c not in skip and not lands(w, region(p, n))
    ]


def not_a_repair(manifest: dict[int, tuple[str, int, str]]) -> list[str]:
    """RULING J-20a. Every entry whose `want` differs from the frozen census.

    A pointer may be repaired. A `want` may not: a manifest carrying a different
    `want` for a cid is claiming a DIFFERENT CLAIM at that cell, which is a
    withdrawal and a new citation, and this instrument refuses to score it as a
    repair however cleanly the new pointer lands.
    """
    out = []
    for c, (_, _, want) in manifest.items():
        if c not in CENSUS:
            out.append(f"C{c}: not in the frozen census -- a NEW citation, not a repair")
        elif CENSUS[c][2] != want:
            out.append(
                f"C{c}: want moved from {CENSUS[c][2]!r} to {want!r} -- "
                f"WITHDRAWAL + NEW CITATION under J-20a, not a repair"
            )
    return out


def test_the_population_is_129_and_the_number_is_dated() -> None:
    """Nothing about the table pins this count; it is measured, at it.20."""
    cites = table_citations()
    assert len(cites) == POPULATION_AT_IT20, f"census population moved: {len(cites)}"
    assert len(set(cites)) == UNIQUE_AT_IT20, f"unique population moved: {len(set(cites))}"


def test_the_census_covers_every_citation_in_the_table() -> None:
    """COVERAGE. 103 of 103 unique, 129 of 129 occurrences. Not 30 of 123."""
    cited = {f"{p}:{n}" for p, n in table_citations()}
    have = {f"{p}:{n}" for p, n, _ in CENSUS.values()}
    assert cited - have == set(), f"uncensused citations remain: {sorted(cited - have)}"
    assert have - cited == set(), f"census holds pointers the table does not: {have - cited}"
    assert len(CENSUS) == UNIQUE_AT_IT20


def test_every_scored_citation_lands_on_its_frozen_anchor() -> None:
    """118 of 131 occurrences at it.27, and 0 of the 118 fail."""
    assert not_landing(CENSUS) == []


def test_exactly_118_of_131_are_scored_and_the_other_13_are_named() -> None:
    """State the number and do not round it."""
    skip = refused()
    assert skip == set(WITHDRAWN), f"the refusal set moved: {sorted(skip)}"
    scored = {f"{p}:{n}" for c, (p, n, _) in CENSUS.items() if c not in skip}
    occurrences = sum(1 for p, n in table_citations() if f"{p}:{n}" in scored)
    assert occurrences == 118, f"scored occurrences: {occurrences}, not 118"
    assert len(table_citations()) - occurrences == 13


def test_every_refusal_is_a_growing_file_and_not_a_typo() -> None:
    """Each anchor was verified correct at census. J-20b refuses them regardless."""
    assert {CENSUS[c][0] for c in WITHDRAWN} == LIVE_FILES
    assert len(WITHDRAWN) == 10
    assert "THE FREEZE" not in line_at("V20_R15_JOURNAL.md", 650), (
        "the pointer landing again would not un-refuse it: the file is still LIVE"
    )


def test_the_want_seal_is_intact() -> None:
    """J-20a. An in-place `want` edit must not pass silently."""
    seal = hashlib.sha256(
        "\n".join(f"{c}|{CENSUS[c][2]}" for c in sorted(CENSUS)).encode("utf-8")
    ).hexdigest()
    assert seal == WANT_SEAL, (
        "a frozen `want` was edited. That is a WITHDRAWAL plus a NEW CITATION "
        "under J-20a, not a repair; move the cid to WITHDRAWN and issue a new one."
    )


def test_the_freeze_refuses_a_repair_aimed_at_a_DIFFERENT_CLAIM() -> None:
    """PLANTED NEGATIVE for J-20a -- the INSPECTOR's fatal defect, demonstrated.

    C42 is `lean/CEQ/V16Domain.lean:302`, anchored on `theorem bedM_overlap_old_two`.
    Move the POINTER to `:304` and the `want` to `theorem bedM_overlap_new_two` in
    the same edit. That is a real line in a real file carrying that real text, so:

      * it RESOLVES -- the it.14 instrument stays green;
      * it LANDS -- the it.17/it.18 instrument stays green, because `want` is
        supplied by the repairer in the same edit as the pointer;
      * and it certifies the OPPOSITE theorem. `bedM.countP satOldTwo = 1`,
        `bedM.countP satNewTwo = 3`. The cell's claim did not move; the citation
        now points at a different fact.

    The freeze must name it. This is the assertion the file exists for.
    """
    mutated = dict(CENSUS)
    mutated[42] = ("lean/CEQ/V16Domain.lean", 304, "theorem bedM_overlap_new_two")

    assert CENSUS[42] == (
        "lean/CEQ/V16Domain.lean",
        302,
        "theorem bedM_overlap_old_two",
    ), "the control citation moved; re-take the negative"
    assert "theorem bedM_overlap_new_two" in line_at("lean/CEQ/V16Domain.lean", 304), (
        "the wrong-claim pointer must be one that genuinely lands, or this "
        "negative proves nothing"
    )
    assert not_landing(mutated) == [], (
        "the LANDING check must stay GREEN on the wrong-claim repair -- that is "
        "the defect being demonstrated, not a bug in the negative"
    )
    assert not_a_repair(mutated) == [
        "C42: want moved from 'theorem bedM_overlap_old_two' to "
        "'theorem bedM_overlap_new_two' -- WITHDRAWAL + NEW CITATION under J-20a, "
        "not a repair"
    ], "the freeze scored a wrong-claim repair as a repair"

    assert not_a_repair(CENSUS) == [], "the unmutated census must be a clean repair set"


def test_the_freeze_refuses_a_cid_that_was_never_censused() -> None:
    """A NEW pointer smuggled in as a repair is named as new."""
    smuggled = dict(CENSUS)
    smuggled[9999] = ("MISTAKES.md", 451, "A threshold refitted to the data it judges")
    assert not_a_repair(smuggled) == [
        "C9999: not in the frozen census -- a NEW citation, not a repair"
    ]


def test_a_pointer_into_a_file_that_has_GROWN_is_refused_not_scored() -> None:
    """PLANTED NEGATIVE for J-20b, and it mutates no file in the tree.

    Move one file's recorded census length. Every cid into that file must leave
    the scored population -- including cids that still land perfectly, because
    the point of J-20b is that their landing has stopped being evidence.
    """
    global FILE_LINES_AT_CENSUS
    keep = FILE_LINES_AT_CENSUS
    target = "lean/CEQ/V16Domain.lean"
    into = {c for c, (p, _, _) in CENSUS.items() if p == target}
    assert len(into) >= 14, f"the negative's target file carries {len(into)} cids"
    assert into & refused() == set(), "the target's cids are already refused"
    try:
        FILE_LINES_AT_CENSUS = dict(keep)
        FILE_LINES_AT_CENSUS[target] = keep[target] - 1
        assert target in grown()
        assert into <= refused(), "a pointer into a moved file was still scored"
        assert not_landing(CENSUS) == [], "refused pointers must not be reported failing"
    finally:
        FILE_LINES_AT_CENSUS = keep
    assert into & refused() == set(), "the negative did not restore"


def test_no_cited_file_has_moved_since_this_census() -> None:
    """The tripwire itself, at HEAD. It fires on the NEXT append to any of them.

    When this goes RED it is not a defect in the table -- it is J-20b working,
    and the named pointers must be re-anchored before the leap reads them.
    """
    assert grown() == {}, f"cited files moved since census: {grown()}"
