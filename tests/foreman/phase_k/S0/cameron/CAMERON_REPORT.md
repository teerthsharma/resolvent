# CAMERON S0: f-lite scalable kernel

f-lite scales. Both registered bars pass. f-lite is a fused Triton attention with forward and backward that never writes an S x S tensor to HBM. It matches the dense `ceq/arm_smprime.py` at theta = 0 to 2.9e-06, and it trains at 0.92x the softmax twin at 11.2M (1.63 GiB) and 0.927x at 26M. The dense family runs out of memory at both of those shapes. One precision cost comes with it, on the gate-logit gradient (see OPEN).

## Construction

At theta = 0: `W = softmax(s + L_i - L_j) * exp((1-beta) LSE_i)`.
- `L = cumsum(log m)` over nonzero gates. It is built in fp64 and passed to the kernel as fp32 (hi, lo) pairs.
- An exact zero is a segment boundary. The segment id is the running zero count, and a pair whose segment ids differ is set to -inf.
- The backward treats the LSE gradient as `D_i -> D_i - dLSE_i`.
- The gate gradient is `dL = rowsum(dS) - colsum(dS)`, summed over heads.
- The dots use `tf32x3`. Blocks are fwd (32,64,4w,1st) and bwd (32,32,4w,2st), tuned on the 4060 at d_head 64. Attention fwd+bwd takes 7.9 ms against 10.3 ms for SDPA at B8 H6 S1024 D64.

## Exactness (S=256, H=4, d=32, B=2, fp32, 32 exact-zero gates; rel = max|a-b|/max|b| vs dense fp32)

| beta | out | dq | dk | dv | du | dbeta | (dqk) | (dg) |
|---|---|---|---|---|---|---|---|---|
| 0.9  | 3.1e-07 | 6.6e-07 | 5.8e-07 | 4.8e-07 | 2.2e-06 | 2.4e-07 | 2.9e-07 | 1.6e-05 |
| 1.0  | 3.0e-07 | 4.5e-07 | 4.6e-07 | 3.7e-07 | 2.9e-06 | 1.4e-06 | 3.2e-07 | 8.3e-06 |
| 1.06 | 2.8e-07 | 6.1e-07 | 5.7e-07 | 5.3e-07 | 6.1e-07 | 2.2e-06 | 4.5e-06 | 3.2e-06 |

The worst barred value is 2.899e-06, against a bar of 1e-05: **GREEN**. dqk and dg are not barred. dg exceeds 1e-05 at beta 0.9. The fp64-oracle columns are in `exactness.json`.

## Cost (fp32, batch 8, AdamW, median of steps 4-12, each run in a fresh subprocess, twin re-measured in the same session)

| shape | arm | params | tok/s | step s | peak GiB | vs twin |
|---|---|---|---|---|---|---|
| 11.2M (384/6L/6H/1024) | twin | 11,228,160 | 64,493 | 0.1270 | 1.35 | 1 |
| 11.2M | f-lite | 11,232,798 | 59,356 (test run: 60,462) | 0.1380 | 1.63 | 0.920 |
| 26M (512/8L/8H/1024) | twin | 25,990,144 | 30,302 | 0.2703 | 2.42 | 1 |
| 26M | f-lite | 25,998,376 | 28,082 | 0.2917 | 2.92 | 0.927 |
| 725k (128/3L/8H/512) | twin | 724,608 | 306,099 | 0.0134 | 0.13 | 1 |
| 725k | f-lite | 725,391 | 301,609 | 0.0136 | 0.16 | 0.985 (dense family 31,004: 9.7x) |

The cost bar at 11.2M asks for at least 43,573 tok/s and at most 4 GiB: **GREEN**. At 26M, f-lite also clears 1/1.5 of the twin (20,251 tok/s) and 4 GiB.

## Tests (each ran RED on its stub before its GREEN; board events logged)

- Exactness: `SP/scale/S0_cameron/test_exactness.py`
  - RED (`--stub`, plain causal SDPA): `RED exactness: worst barred rel 1.012e+00 vs bar 1e-05 (stub)`
  - GREEN: `GREEN exactness: worst barred rel 2.899e-06 vs bar 1e-05 (flite)`
- Cost: `SP/scale/S0_cameron/test_cost.py`
  - RED (`--stub`, dense family): `RED cost@11.2M: family OOM vs bar >= 43573 tok/s, <= 4.0 GiB`
  - GREEN: `GREEN cost@11.2M: flite 60462 tok/s (0.925x twin), peak 1.63 GiB vs bar >= 43573 tok/s, <= 4.0 GiB`
- HF patch parity (Cameron's own check, not a registered bar): `SP/scale/S0_cameron/test_patch.py`
  - RED on its stub (softmax twin): 1.370e+00.
  - **Still RED** on f-lite: `RED patch: worst rel (loss + 50 param grads) 6.690e-05 vs bar 1e-05`
  - The loss is bitwise equal. The miss is one gradient, `layers.0.m_head.bias` (|g| = 1.1e-4, a 512-term cancelling sum). Against an fp64 oracle, f-lite is off by 5.46e-05 and dense fp32 by 1.23e-05. Both miss 1e-05 against the truth; f-lite is 4.4x worse.

## Kill and replacement

- Killed: fp64 accumulation of the row and column sums as the fix for the gate-gradient gap. It made the gap worse (1.63e-04) and was reverted.
- Replacement: compute the gate gradient in straddle form (sum dS only over pairs (i >= k > j)) with block-local prefix sums inside the dq kernel. The other option is to register the gate-gradient bar against fp64 truth at 1e-4, where dense fp32 would also be held to account.

## Deliverables

- `flite.py`: kernel, autograd, `flite_attention`, and `flite_forward`. Patch it in with `CEQAttention.forward = flite.flite_forward` on an `R.build_repaired(..., operator="smprime")` build.
- `measure.py`: the cost sweep. `python measure.py all` writes `cost.json`.
- `tune.py`: block-size tuning.
- Supporting files: `test_*.py`, `exactness.json`, `cost.json`.

## OPEN

1. Gate-logit gradient precision in the scan form (above). q, k, v, beta and the output sit at dense fp32's noise floor; u, g and the gate bias do not.
2. f-lite subtracts the row maximum before exponentiating, so it stays finite where the dense route's unshifted `Z = sum R e^s` overflows (s > 88 in fp32). The two routes agree only on beds where the dense route stays finite, and no overflow bed was run.
3. There is no segment-skip for dead key blocks yet. Once hard-concrete gates close often, skipping could make f-lite faster than the twin.
4. The 12-step losses (f-lite 3.80 vs twin 4.39 at 11.2M) come from one repeated random batch. They are memorisation, not evidence about quality.
5. `flite_forward` leaves `theta_head` gradient-free (None rather than f0's zeros), so AdamW weight decay does not touch it.
6. The standing default is to ship f-lite as the S1 family arm, since both bars are green. The straddle-form gate gradient comes next, before any claim that depends on gate-gradient precision.
