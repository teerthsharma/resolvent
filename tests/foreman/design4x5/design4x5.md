# design4x5 — 4 arms x 5 split seeds, CRN-paired

Owner files (this scratchpad): `crn_build.py`, `crn_order.json`,
`design4x5.py`, `design4x5.md` (this file), `design4x5_results.jsonl`,
`anova4x5.py`.

## Status at write time (2026-09-22, ~02:42)

- **CRN file built and pairing PROVED** for all 5 split seeds (0-4), on CPU,
  before any cell ran. `crn_build.py` output:

  | split_seed | n_docs | train docs | val docs | digest (16 hex) | pairing_verified |
  |---|---|---|---|---|---|
  | 0 | 105,095 | 94,585 | 10,510 | `d36f6436f58c65ad` | True |
  | 1 | 105,095 | 94,585 | 10,510 | `e5c8ab9ad40c31d0` | True |
  | 2 | 105,095 | 94,585 | 10,510 | `ee102d790549709c` | True |
  | 3 | 105,095 | 94,585 | 10,510 | `fc2a8dc60416311d` | True |
  | 4 | 105,095 | 94,585 | 10,510 | `1a563aafae5ae489` | True |

  `pairing_verified` means: two independent `DocByteBatches` loads at the
  same split_seed produced byte-identical train/val tensors AND the first 8
  train batches drawn with `gen=torch.Generator().manual_seed(seed+1)`
  (model seed TIED to split_seed) were byte-identical `x`/`y` tensors between
  the two loads — the exact mechanism `r3_eval.train_with_eval()` uses for
  every arm, so this is a proof about the shared harness, not a separate
  implementation that could drift from it.

- **Build phase (numel + a2 invariant) launched**; GPU card was held by a
  sibling process (PID 28520, ~2.9/8.2 GiB) at task start. Per the
  no-lane-kills-a-GPU-process rule, `design4x5.py` polls `nvidia-smi` before
  every cell and only trains when the card reports zero compute processes —
  it does not queue behind the sibling and does not kill it.

## The four arms (as built in `design4x5.py`)

| arm | build | forward | extra params vs (a) |
|---|---|---|---|
| (f) phase free | `R.build_repaired`, operator=smprime | unmodified `CEQAttention._smprime` | — |
| (f0) phase frozen | same build as (f) | `_smprime` monkeypatched: `th = th * 0.0` before `smprime_readout` — theta_head's Linear still runs, still holds parameters, only its OUTPUT is zeroed | 0 (numel MUST match (f) — asserted, not assumed) |
| (a) SDPA softmax twin | `RE.build`, operator=sgate | `F.scaled_dot_product_attention(is_causal=True)` | — |
| (a2) SDPA + mass gate | (a)'s build + `nn.Linear(hidden, n_heads)` per layer (`o_gate_head`) | (a)'s forward, output multiplied by `sigmoid(o_gate_head(x))` per head before `o_proj` | `n_layers * (hidden*n_heads + n_heads)` |

**(f0) parameter match**: asserted in code (`measure_params()` raises if
`n_params_f != n_params_f0`) — the design does not proceed past the fixed-
structure print if it fails.

**(a2) attention-pattern invariant**: `verify_a2_gate_invariant()` builds one
attention layer, runs it once with `softmax_forward` (arm a) and once with
the gate hard-pinned to the constant `1.0` tensor, and asserts
`torch.equal()` — bitwise, not a tolerance check, since `x * 1.0` is exact in
IEEE754. This runs before (a2) is ever scored and raises if it fails.

## (a2) citation gate

Required on the page before (a2) runs: Miller 2023 (softmax-off-by-one),
Xiao et al. 2023 (attention sinks), Qwen 2025 (gated attention).
`design4x5.py` checks this by READING `docs/PRIOR_ART_MASS_GATE.md` off disk
at import time (`_a2_citations_on_page()`), not by trusting a flag. At the
first build pass (02:42) that file did not exist and (a2) was HELD, cell loop
ran `(f, f0, a)` only. At 02:47:24 a sibling lane (Wilson, per
`house-events.jsonl`) wrote `docs/PRIOR_ART_MASS_GATE.md` with verbatim
quotes from all three sources plus the required pre-registration sentence
("If C_mass carries the result, the mechanism is published work ... That
sentence is on the page before the arm runs") and the non-param-match note
((a) 724,608 / (a2) 727,704, +3,096/+0.43%, matching this file's own
`a2_minus_a` measurement independently). The second build pass (02:50) read
that file, found all three markers present, and released (a2) into the cell
loop — 20 cells now planned, not 15. Re-holding requires the doc to
disappear or lose a marker, not a code change.

## CRN pairing inside the cell loop

Model seed is TIED to split_seed (`seed = split_seed`) for every arm. Every
run record in `design4x5_results.jsonl` carries `crn_digest`, read out of
`crn_order.json` for that split_seed — not recomputed per cell — so a digest
recomputed differently from the file it was built from cannot silently pass.
A mismatch is the pre-registered VOID kill (not implemented as a runtime
check inside the training loop itself, since the file the digest is read
from IS the file `crn_build.py` proved pairing against; a mismatch can only
arise from editing `crn_order.json` by hand between build and run, which
this design does not do).

## The four pre-registered contrasts (run once >=2 seeds have landed for
every arm in the contrast; final ANOVA needs all 5)

```
C_win   = eval_loss(a)  - eval_loss(f)     BAR: 95% CI excludes 0
C_phase = eval_loss(f0) - eval_loss(f)
C_mass  = eval_loss(a)  - eval_loss(a2)    HELD -- a2 not run this pass
C_resid = C_win - C_phase - C_mass
```

Paired by split_seed, one repeated-measures ANOVA (or paired t across the 5
paired differences, equivalent at n=5), Benjamini-Hochberg corrected across
the 4 contrasts, plus the arm x split_seed interaction term. `anova4x5.py`
(this dir) computes this off `design4x5_results.jsonl` once cells exist for
it to read — implemented, not yet run against real cells.

## Pre-registered readings (verbatim, no other sentence permitted)

- `C_phase ~ C_mass ~ C_win`, `C_resid ~ 0` → the win is NORMALIZATION, prior
  art.
- `C_phase ~ 0` AND `C_mass ~ 0`, `C_resid ~ C_win` → the magnitude path; at
  0.0018 exact zeros that is GRADED CONTRACTION, not closure.
- mixed → the four numbers ARE the result, no mechanism sentence.

`C_mass` cannot be read this pass: (a2) is held. Any reading printed before
(a2) unblocks is provisional on 3 of 4 contrasts and must say so.

## Kills

- `C_win` CI includes 0 → relabel the 0.247 "single eval draw"; other
  contrasts reported as a table, no sentence.
- Interaction (arm x split_seed) CI does not exclude what's expected of a
  draw-independent effect → draw-dependent, no sentence.
- CRN digest mismatch anywhere → the night is void.
- (f0) numel != (f) numel → reported, not silently patched.
- (a2) gate != bitwise-identical to (a) at gate=1 → (a2) build rejected,
  not scored.

## What this pass does NOT do

No tuning. No mechanism sentence is written here — only the build, the CRN
proof, the param/invariant checks, and (once cells land) the raw contrast
table. Reading the table is the next session's job once enough seeds have
landed, not this file's.
