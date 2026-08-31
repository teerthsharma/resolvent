# V17-K G0.10: RULING 6 discharged against `ceq/autopilot.py`

## Lead caveat

Everything below runs on local dummy journal lines, on CPU, at trivial shape.
No Kaggle CLI call, no upload, no `~/.kaggle` access, no GPU, no writing git
command. This is the same decision function as
`V17_G08_G09_AUTOPILOT.md` (not edited here -- it is a historical record, not
owned by this pass); this file documents the six deltas
`V17K_RULINGS.md` RULING 6 (6a-6f) forces on top of it, and the tests that
prove each one. One ruling (6b) changes the journal schema itself (a field
rename, not just a threshold): `session_end_t` is gone, replaced by
`kaggle_session_remaining_min`. Two rulings (6c, 6d) required **no code
change** -- the pre-existing implementation already matched the ruling; this
is stated plainly below rather than staged as a fix, because a diff invented
to justify "RED before" where none existed would be exactly the kind of new
construction the house rules forbid.

## Verdict table

| ruling | before | after | code change |
|---|---|---|---|
| 6a stale journal | `TIER1_STALE_MIN=15` [ASSUMED] min elapsed since the event's own `t` | no write for 2 CONSECUTIVE POLLS, counted from `history` | yes |
| 6b session <20min | `session_end_t - t < 20` (computed) | `kaggle_session_remaining_min < 20` (read directly) | yes, incl. schema rename |
| 6c simultaneous triggers | Tier-3 guards checked before Tier-2/Tier-1 in `decide()` | unchanged -- already highest-tier-wins | **no** (verified, not assumed) |
| 6d Tier-2 lineage | `_lineage_root()` loop-strips `.t2.N` to the ultimate root already | unchanged -- already root-scoped across chains | **no** (verified, not assumed) |
| 6e OOM per shape | unconditional halving, no history check | Tier-3 HALT on a second OOM at the same `(cell, micro_batch)` | yes |
| 6f frozen deciding-cells | `frozenset`, immutable as a value but nothing stopped a caller passing a bigger one on a later poll | `AutopilotConfig.deciding_cells_launch` + `__post_init__` raise `ValueError` on any growth beyond it, at construction | yes |

## RULING 6a -- stale journal = no write for 2 CONSECUTIVE POLLS

**Before**: `_match_tier1` fired `probe-restart` when `(now_t - event.t) >=
TIER1_STALE_MIN` (15 min, `[ASSUMED]`, flagged in `V17_G08_G09_AUTOPILOT.md`
as a made-up number the table never gave). This is a single-observation,
wall-clock-elapsed reading -- exactly the reading RULING 6a rejects.

**After**: `_stale_poll_count(history, event)` counts the trailing run of
consecutive polls whose raw journal text is byte-identical to the poll
immediately before it (the journal did not grow between them). `TIER1_STALE_
POLLS = 2` [MODULE, from the ruling text verbatim]. `_match_tier1` fires
`probe-restart` when that count reaches 2 -- i.e. the third poll in a row
carrying the exact same line. `now_t` is no longer read by this rule (kept in
the signature; see the docstring added at `ceq/autopilot.py:304-311`).

**RED before the change** [MEASURED, captured against the pre-edit module]:
```
=== RED 6a: 3 identical polls (2 consecutive no-write gaps), no elapsed-time gap given ===
poll1: 0 NONE
poll2: 0 NONE
poll3 (expect tier=1 probe-restart under 6a): 0 NONE
```
Three identical polls with `now_t == event.t` every time (no wall-clock gap
at all) produced `tier=0` under the pre-ruling code, because the old rule
only looked at elapsed time and none had passed.

**Planted negative (fires)**: `test_stale_journal_probes_restart` --the same
raw line submitted on 3 consecutive polls; `tier=1, action=probe-restart` on
the third.

**Non-degeneracy (stays silent, genuinely in range)**: three separate cases,
not one:
1. `test_one_no_write_poll_does_not_yet_probe` -- exactly ONE no-write poll
   (not two consecutive) stays `tier=0`, proving the rule needs both gaps.
2. `test_fresh_journal_does_not_probe` -- a journal that genuinely grows
   every poll (a new `step` each time) never fires, any number of polls.
3. `test_large_now_t_alone_does_not_probe_restart` -- a single poll against
   a 10,000-unit `now_t` gap stays `tier=0`, proving the *old* mechanism
   (elapsed time alone) is gone, not merely supplemented.

## RULING 6b -- "session <20 min" = KAGGLE'S reported remaining time

**Before**: `PollEvent.session_end_t` (the session's absolute end timestamp)
was required; the Tier-1 rule computed `session_end_t - event.t` as
"remaining minutes." `V17_G08_G09_AUTOPILOT.md` flagged the *direction*
(elapsed vs. remaining) as ambiguous but did not flag the *source* -- RULING
6b's point is narrower and stricter: even under the "remaining" reading, a
value the module computes from two other fields is not "Kaggle's reported
remaining time."

**After**: `PollEvent.session_end_t` is replaced by
`PollEvent.kaggle_session_remaining_min` (schema rename, propagated through
`_NUM_OR_NONE`, `parse_line`, and the docstring at
`ceq/autopilot.py:112-115`). The Tier-1 rule reads this field directly:
`event.kaggle_session_remaining_min < TIER1_SESSION_REMAIN_MIN` (20.0,
`[MODULE]`, unchanged value -- only the source changed, per the ruling's own
wording). `event.t` no longer appears in this comparison at all.

**RED before the change** [MEASURED, captured against the pre-edit module]:
a line built with `kaggle_session_remaining_min` instead of `session_end_t`
against the OLD schema:
```
RED (pre-change) 6b, expect tier=1 force-ckpt-end-chunk, got: Malformed 3 HALT field:'session_end_t'
```
The old schema required `session_end_t` and had no slot for the new field;
the same line that should force a checkpoint under 6b instead failed schema
validation and HALTed as malformed.

**Planted negative (fires)**: `test_session_ending_and_no_ckpt_in_30_forces_checkpoint`
-- `kaggle_session_remaining_min=10.0` (< 20) with a stale checkpoint (`t -
last_ckpt_t >= 30`) fires `tier=1, action=force-ckpt-end-chunk`.

**Non-degeneracy (stays silent, genuinely in range)**:
`test_session_ending_but_recent_ckpt_does_not_force` -- same low remaining
time, but a recent checkpoint, does not fire (proves the AND with `no_ckpt`
still holds). A second, source-specific check,
`test_session_remaining_is_read_directly_not_computed_from_t`: two lines with
`kaggle_session_remaining_min=15.0` but wildly different `t` (`9.0` vs.
`500.0`) both fire identically -- proving the number truly comes from the new
field and is independent of `t`, not silently re-derived.

## RULING 6c -- simultaneous triggers: highest tier wins; Tier-3 pre-empts a Tier-1 fix

**Verified against the code as committed, before any edit for this round**
[MEASURED]:
```
--- 6c: Tier-3 (phase-arm ahat) co-firing with Tier-1 (oom) ---
3 HALT a_hat_max-on-phase-arm
--- 6c: Tier-3 (kcert) co-firing with Tier-2 (loss_nonfinite) ---
3 HALT kcert-delta-over-tol
```
`decide()` already checks the four Tier-3 guards (root-fails-to-extend,
phase-arm â_max, deciding-cell design change, K-CERT δ/tol) unconditionally
before `_match_tier2` and before `_match_tier1` are ever called -- so a line
matching both a Tier-3 trigger and a Tier-1 (or Tier-2) trigger was already
resolved as Tier-3 HALT, never as the lower tier's fix. **No code change was
made for 6c.** Coverage did not previously include this exact cross-tier
claim, so three tests were added as permanent regression proof, not as a
fix -- confirmed GREEN both immediately (no edit) and after all other 6a/6b/
6e/6f edits landed:
- `test_tier3_co_firing_with_tier1_oom_halts_not_halves`
- `test_tier3_co_firing_with_tier2_nan_halts_not_corrects`
- `test_tier1_alone_still_fixes_without_a_co_firing_tier3` (non-degeneracy:
  the same `oom=True` field alone, with nothing Tier-3 also matching, still
  gets the Tier-1 fix -- proves the pre-emption is conditional, not a
  blanket "OOM is now inert" regression).

## RULING 6d -- Tier-2 lineage scoped PER ROOT CELL, no laundering via chains

**Verified against the code as committed, before any edit for this round**
[MEASURED]:
```
--- 6d: 3-cells-deep chain, second tier2 anywhere in it escalates ---
first (root cellA): 2 rollback+lr-decay+grad-clip cellA.t2.1
second (grandchild cellA.t2.1.t2.1): 3 HALT two-tier2-on-cell:a_hat_max
--- 6d: unrelated roots do NOT aggregate ---
tier2 on unrelated cellB: 2 rollback+lr-decay+grad-clip loss_nonfinite
```
`_lineage_root()` is a `while` loop, not a single strip: it re-applies the
`.t2.N`-suffix removal until nothing changes, so `cellA.t2.1.t2.1` already
resolved to root `cellA` two levels down, and `_tier2_count_for_lineage`
already aggregates every Tier-2 event anywhere in that root's history before
allowing a second one. **No code change was made for 6d.** The round brief's
framing ("stricter than the lineage-root scoping already implemented")
describes the ambiguity as filed in `V17_G08_G09_AUTOPILOT.md`, not a defect
in this specific implementation -- the implementation had already resolved it
this way; that reading is simply now law instead of a guess. Two tests were
added as the chain-depth and unrelated-root coverage the round brief
explicitly asked for:
- `test_second_tier2_three_cells_deep_in_the_chain_escalates` -- a Tier-2 on
  root `cellA`, then a second Tier-2 crafted on a grandchild id
  `cellA.t2.1.t2.1` (two levels deeper than the existing
  `test_two_tier2_on_one_cell_halts_on_the_second`, which only went one
  level) -- HALTs.
- `test_two_tier2_on_unrelated_roots_do_not_aggregate` (non-degeneracy) -- a
  Tier-2 on `cellA` and a Tier-2 on unrelated `cellB` are each the first for
  their own root; neither escalates because of the other.

## RULING 6e -- OOM keyed per (cell, shape): second OOM at same shape after halving -> Tier 3

**Before**: the OOM branch of `_match_tier1` halved unconditionally on every
poll where `oom=True`, with no memory of prior OOMs.

**After**: `_oom_count_for_shape(history, cell, micro_batch)` counts prior
OOM polls at the identical `(cell, micro_batch)` pair -- "shape" is read as
`micro_batch` [ASSUMED, fails safe toward HALT: see Ambiguities below];
`grad_accum` moves in lockstep to hold the effective-batch invariant, so it
carries no independent information. `decide()` gained a new Tier-3 guard,
checked alongside the other four (before `_match_tier2`/`_match_tier1`): if
`oom` and this exact shape has already OOM'd once for this cell, HALT with
trigger `oom-repeat-same-shape` instead of halving again.

**RED before the change** [MEASURED, captured against the pre-edit module]:
```
=== RED 6e: second OOM at the identical shape after a halving ===
oom#1 shape=8: 1 halve-micro-batch
oom#2 shape=8 again (expect tier=3 HALT under 6e): 1 halve-micro-batch oom
```
The second OOM at the same reported `micro_batch=8` halved again under the
old code instead of escalating.

**Planted negative (fires)**: `test_second_oom_at_same_shape_after_halving_escalates`
-- OOM at `micro_batch=8` twice on the same cell; first halves, second
HALTs with `oom-repeat-same-shape`.

**Non-degeneracy (stays silent, genuinely in range)**: two cases:
1. `test_oom_at_a_genuinely_new_shape_still_halves` -- the second OOM
   overall for the cell, but at a *different*, already-halved shape
   (`micro_batch=4`, not `8`), is the FIRST OOM at that shape and still
   halves -- proves the counter is shape-scoped, not a one-OOM-ever ceiling
   per cell.
2. `test_oom_at_same_shape_on_a_different_cell_does_not_aggregate` -- the
   same shape hit on an unrelated cell does not count against the first
   cell's shape counter.

## RULING 6f -- deciding-cell membership frozen at launch, may shrink, never grow

**Before**: `AutopilotConfig.deciding_cells` was a `frozenset` (immutable as
a *value* -- no in-place mutation possible) inside a `frozen=True` dataclass
(no *reassignment* possible) -- but nothing stopped a caller from
constructing a brand-new, larger `AutopilotConfig` for a later poll and
handing it to `decide()`; growth across polls was unenforced.

**After**: a new optional field, `deciding_cells_launch`, and a
`__post_init__` on `AutopilotConfig` that raises `ValueError` the instant a
config is constructed whose `deciding_cells` is not a subset of
`deciding_cells_launch`. This makes growth impossible to even construct, not
just impossible to sneak past a check later -- every `AutopilotConfig` that
ever reaches `decide()` had to survive this raise at `__init__` time.
Leaving `deciding_cells_launch=None` (the default) opts a config out --
intended for the single launch-time config that *establishes* the set, which
has nothing yet to be checked against.

**RED before the change** [MEASURED, captured against the pre-edit module]:
```
RED (pre-change): TypeError AutopilotConfig.__init__() got an unexpected keyword argument 'deciding_cells_launch'
```
The field did not exist, so the intended API (`deciding_cells_launch=...`)
raised the wrong exception for the wrong reason (a missing kwarg, not a
deliberate growth guard) -- there was no growth check to fail RED against,
because there was no mechanism at all.

**Planted negative (raises)**: `test_growing_deciding_cells_beyond_launch_raises`
-- launch set `{cellD}`, attempted set `{cellD, cellE}` -- `ValueError`.

**Non-degeneracy (the other direction must work)**:
`test_shrinking_deciding_cells_within_launch_succeeds` -- launch set
`{cellD}`, new set `{}` (cellD retired) -- constructs cleanly, and is fed
into `decide()` to show the shrink is live: a `design_change_request` on the
now-retired `cellD` is no longer refused as the deciding-cell kill clause.
`test_config_at_launch_with_no_launch_set_is_unrestricted` covers the launch
config itself (`deciding_cells_launch=None` is the default, unchecked).

## Seven must-fires and MARS: unchanged, still pass

**[MEASURED]** `python -m pytest tests/gate0/test_g08_autopilot.py
tests/gate0/test_g09_mars.py -q` -> **62 passed** (was 48/48 in
`V17_G08_G09_AUTOPILOT.md`, commit `ab5b485`; +14 new: 2 for 6a, 1 for 6b, 3
for 6c, 2 for 6d, 3 for 6e, 3 for 6f), Python 3.11.9, Windows 11, CPU, 0.40s.
`test_mf1_*` through `test_mf7_*` (the seven must-fires, both fire and
clean-does-not-fire halves) and every `test_every_mars_variant_is_refused` /
`test_decision_is_byte_identical_regardless_of_note_content` /
`test_note_field_is_never_referenced_by_decision_code` body is **byte-for-
byte unedited** by this pass -- the only touch to their shared fixture was
renaming one dict key (`session_end_t` -> `kaggle_session_remaining_min`) in
`_BASE`/the clean-chunk JSONL, required by 6b's schema change and applied
identically to every line.

**Structural note-blindness, re-verified after this pass's additions**
[MEASURED]: `'note' not in fn.__code__.co_names` holds for `decide`,
`decide_line`, `_match_tier1`, `_match_tier2`, `_tier2_decision`, `_halt`,
`_lambda_three_poll`, and the two new decision-path helpers this ruling
added, `_stale_poll_count` and `_oom_count_for_shape` (both now included in
`test_note_field_is_never_referenced_by_decision_code`'s checked list, since
they are new links in the chain `decide()` calls). Neither reads `.raw`-
derived free text either -- `_stale_poll_count` only ever compares two
`.raw` strings for byte equality (never parses or branches on their
content), and `_oom_count_for_shape` reads only `.cell`, `.oom`,
`.micro_batch`.

## Ambiguities still open, and which way they were failed

- **6e's "shape"**: the ruling names `(cell, shape)` as the key but does not
  define "shape" for this schema (no seq_len/hidden_size field exists on a
  poll line). Read as `micro_batch` alone, since `grad_accum` is always the
  deterministic co-variate that preserves `micro_batch * grad_accum`
  (`_match_tier1`'s existing invariant) and carries no separate information.
  This reading fails safe toward the escalation the ruling wants: it groups
  OOMs as aggressively as the schema allows, so a real repeat is never
  missed by under-keying (the only way this reading could go wrong is by
  over-keying, i.e. treating two genuinely different shapes as the same --
  not possible here since `micro_batch` is the one dimension the schema
  actually varies). `[ASSUMED]`, narrow: reported to the loop, not silently
  resolved by opening the journal's free text.
- Nothing else required an ambiguous read this pass -- 6a, 6b, 6c, 6d, and
  6f each had one plain construction from the ruling's own wording.

## Git state

**[MEASURED]** No writing git command was run.
- Start: `HEAD = ab5b48547884e04258276e6e808d5a71ea65f917`; `git status
  --porcelain` showed `M ceq/arm_smprime.py`, `M ceq/hf/configuration_ceq.py`,
  `M ceq/hf/modeling_ceq.py`, `M ceq/hf/train.py` (other agents' in-flight
  work, untouched by this pass) plus `?? ceq/autopilot.py`, `?? tests/gate0/`,
  and a dozen other untracked `V17_*.md`/`kaggle/`/`scripts/k_cert.py` files
  from prior rounds.
- End: `HEAD` unchanged at `ab5b48547884e04258276e6e808d5a71ea65f917`.
  `git status --porcelain` additionally shows more files touched by the three
  other agents sharing this tree (`M MODEL_CARD.md`, `M house-events.jsonl`,
  `M scale/identity_manifest.py`, `M scripts/v15_r1.py`, `?? COSTS.md`,
  `?? V17_G06_G07_CERT_COST.md`, `?? V17_GPU_QUEUE.md`,
  `?? scripts/k_cost.py`) -- none of which this session made or touched.
  This session's own footprint is confined to `ceq/autopilot.py`,
  `tests/gate0/test_g08_autopilot.py`, `tests/gate0/test_g09_mars.py`,
  `tests/gate0/fixtures/autopilot_clean_chunk.jsonl`, and this file.

## GREEN / RED / BLOCKED

**GREEN.** All six sub-rulings (6a-6f) are implemented and match
`V17K_RULINGS.md` RULING 6's text. Four required an edit to
`ceq/autopilot.py` (6a, 6b, 6e, 6f), each proven RED before the edit with the
exact pre-edit output pasted above, then GREEN after, with a planted
negative and a genuinely-in-range non-degeneracy case per the round brief's
requirement. Two (6c, 6d) required no edit -- verified against the pre-round
code, not assumed, with the same terminal output pasted above -- and gained
regression tests they did not have before. The seven must-fires and G0.9
MARS resistance are unchanged in body and still pass, now 62/62 (was 48/48).
The structural `note`-blindness property (`'note' not in
fn.__code__.co_names`) holds for the full decision path including this
round's two new helpers. No journal free text was opened to satisfy any of
6a-6f -- every new check reads only typed, already-parsed `PollEvent`
fields (`.raw` compared for byte equality, never parsed) or
`AutopilotConfig`, consistent with the module's existing MARS-resistance
design.
