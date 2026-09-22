# FOREMAN — R-POS, round N2

**Verdict: PASS. The leap binds.** The family's language-model win (C_win) comes from a relative-position prior, not from the operator. Adding RoPE to the grid's softmax twin, with zero new parameters, recovers 96.3%, 99.7% and 101.4% of C_win (mean 99.2%). ALiBi recovers 108.8%, 106.7% and 108.8% (mean 108.1%), so the ALiBi twin beats the family (f) outright at all three seeds.

## R-POS table

Grid losses come from `tests/foreman/design4x5/design4x5_results.jsonl`:

| split seed | (a) | (f) | C_win |
|---|---|---|---|
| 0 | 1.275684 | 1.043056 | 0.232628 |
| 1 | 1.292035 | 1.053042 | 0.238993 |
| 2 | 1.301034 | 1.039085 | 0.261949 |

| arm | seed | loss | recovery | CRN digest (recomputed = file = grid) | wall s | grid (a) wall s | ratio |
|---|---|---|---|---|---|---|---|
| a_rope | 0 | 1.051606 | 0.9632 | d36f6436f58c65ad ✓ | 110.7 | 77.6 | 1.43× |
| a_rope | 1 | 1.053742 | 0.9971 | e5c8ab9ad40c31d0 ✓ | 104.1 | 77.6 | 1.34× |
| a_rope | 2 | 1.035336 | 1.0143 | ee102d790549709c ✓ | 101.8 | 75.8 | 1.34× |
| a_alibi | 0 | 1.022485 | 1.0884 | d36f6436f58c65ad ✓ | 103.6 | 77.6 | 1.33× |
| a_alibi | 1 | 1.037006 | 1.0671 | e5c8ab9ad40c31d0 ✓ | 108.1 | 77.6 | 1.39× |
| a_alibi | 2 | 1.015971 | 1.0882 | ee102d790549709c ✓ | 109.9 | 75.8 | 1.45× |
| a (re-run) | 0 | 1.275686 | 0.0000 | d36f6436f58c65ad ✓ | 91.4 | 77.6 | 1.18× |

- **Mean recovery:** a_rope 0.9915, a_alibi 1.0813.
- **Control on record:** FoX recovers 100.2%, 100.3% and 110.9% (727,704 params, 131.9 s).
- **Parameters:** 724,608 in every R-POS arm, the same as the grid's (a).
- **Wall clock:** The (a) re-run at seed 0 took 91.4 s against 77.6 s in the grid. The card is 1.18× slower today, so against today's (a) the overheads are 1.21× (RoPE) and 1.13× (ALiBi). The family (f) takes 514–519 s per cell.
- **CRN check:** Each digest was recomputed from the corpus with `crn_build.doc_order` and `digest_order`. It matches both the digest in `crn_order.json` and the grid's (a) and (f) rows at all three seeds.
- **Machinery check:** Re-running (a) at seed 0 through the same path gives 1.2756859 against the grid's 1.2756839. The difference is 1.9e-6 nats, or 8e-6 of C_win. The result is not bitwise identical, but the pairing holds.
- **Code audit:** A sonnet nurse reviewed the code (read-only, CPU only) and found:
  - The RoPE scores are Toeplitz to 2.4e-6.
  - The ALiBi bias is exactly −slope_h·(i−j), unscaled, with −inf above the diagonal.
  - The rebinding of `design4x5.softmax_forward` takes effect when `train_arm` is called and is restored afterwards.
  - Apart from the checkpoint directory, nothing differs from the grid's (a).

## Against the registered bar

The bar is: PASS if a_rope recovers ≥ 0.8 at all three seeds. a_rope's lowest recovery is 0.9632, so the result is **PASS**. The KILL branch cannot fire, because neither arm recovers less than 0.5 at any seed.

## Test

- **Path:** `C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_j/N2/foreman/test_rpos.py`
- **RED:** `python test_rpos.py --stub` (NaN losses with the grid's digests) exits 1 with:
  `AssertionError: R-POS verdict NEITHER, not PASS: a_rope [nan, nan, nan] a_alibi [nan, nan, nan]`
- **GREEN:** `python test_rpos.py` exits 0 with `R-POS verdict PASS`.
- **Board:** the RED event precedes the rpos.py dispatch and every cell. Rows are in `rpos_results.jsonl` and the run log is `rpos_run.log`.

## What the result means

- **What the twin lacked was translation invariance.** RoPE builds no recency bias into the logits beyond a weak drift, yet it gets 99% of C_win. The twin's deficit was relative position as such. ALiBi adds an explicit recency window on top and gets a further ~9 points, past the family.
- **The zero-parameter ALiBi twin beats the family by 0.0206, 0.0160 and 0.0231 nats** at seeds 0, 1 and 2, in about 0.21 of the family's clock. On this bed, C_win against the correct control is negative at all three seeds.
- **FoX's input dependence buys nothing measurable here.** Fixed-slope ALiBi recovers 108.8% and 106.7% against FoX's 100.2% and 100.3% at seeds 0 and 1. FoX leads only at seed 2 (110.9% against 108.8%). ALiBi also uses 3,096 fewer parameters. The registered KILL clause, "data-dependent forgetting is required, and FoX owns it", is contradicted, not just unmet.
- **Replacement route: none owed**, because the leap did not die.
- **Standing consequence.** (a) was the only arm in the grid with no relative-position mechanism, so every bar scored against (a) measured that handicap. The zero-parameter control is now (a_alibi).

## OPEN

1. **Seeds 3 and 4** were not run; the row was registered at seeds 0–2.
2. **Rebase the family's surviving LM rows onto (a_alibi).** This covers R-STRAT abstention and R-XFER. The measurement becomes d_ALiBi = NLL_alibi − NLL_f, per site, on the seed-0 pair.
   - R-STRAT's decile profile was measured against (a). It may be the same positional handicap seen per site, so it stands unrebased until then.
   - **Default next row:** Foreman on the GPU, about 2 minutes per seed. The ALiBi checkpoints `ckpt_a_alibi_ss{0,1,2}` exist, but without weights (`save_model` is off in `train_arm`), so seed 0 must be re-run with `save_model=True` first.
3. **This bed rewards locality.** It is byte-level: 3 layers, width 128, 3,538 steps. The family and FoX attend 1.2–2.0 bytes back. Any operator claim now needs a bed where a fixed recency prior cannot reach the answer, for example long-range retrieval or copy. It also needs (a_alibi) as the floor, run before any bar is written.
4. **Harness defect:** `design4x5.poll_until_free` counts the caller's own CUDA context as busy (compute_procs=1, 201 MiB). An in-process multi-cell loop therefore stalls for the full timeout after its first cell. This run lost about 8 minutes to it before switching to one fresh process per cell.
