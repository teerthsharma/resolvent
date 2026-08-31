# V13 — independent audit of this session's numeric claims

Adversarial recomputation of the figures written into `V13_DAG_TASKLIST.md`,
`workdonenew.md` §5 and `MISTAKES.md` (`V-19`–`V-22`, `M-10`–`M-14`). Every row
below was recomputed from the raw journals, from source constants, or by
re-running the measurement — never by re-reading the number it checks.

**Result: 44 CONFIRMED, 9 DISCREPANT, 1 UNVERIFIABLE.**

No discrepancy overturns a headline conclusion. Eight are wrong or mislabelled
statistics inside otherwise sound arguments; one — the K-sweep harness control —
quotes figures that exist nowhere on disk and that the on-disk journal
contradicts.

Audit machinery: Python 3.11.9, scipy 1.17.1, numpy 1.26.4, torch 2.5.1+cu121,
same host, RTX 4060 Laptop idle at the time of the GPU re-measurement
(`nvidia-smi`: 0 MiB used, 0% util, 8188 MiB total).

---

## The table

| # | claim | as written | recomputed | verdict | route |
|---|---|---|---|---|---|
| 1 | two-sample 90% CI half-width in σ units | `t(.95,2N−2)·√(2/N)` | identical to the pooled-variance form `t·s_p·√(1/N+1/N)` | CONFIRMED | derivation, then `scipy.stats.t.ppf` |
| 2 | half-width at N=8 | `0.8807` | `0.880655` | CONFIRMED | `t.ppf(.95,14)·√(2/8)` |
| 3 | half-width at N=16 | `0.6001` | `0.600072` | CONFIRMED | same |
| 4 | half-width at N=23 | `0.4948` | `0.495473` | **DISCREPANT** | same |
| 5 | half-width at N=24 / N=40 | `0.4846` / `0.3722` | `0.484588` / `0.372221` | CONFIRMED | same |
| 6 | CI first fits the 0.5 margin at | `N=23` | N=22 → `0.507128`, N=23 → `0.495473` | CONFIRMED | scan N=2…500 |
| 7 | TOST power at true diff 0, N=8 | `0.000` | `0.0006` exact, `0.0005` MC | CONFIRMED | see §D1 |
| 8 | TOST power at N=23 | `≈0.03` | `0.0668` exact, `0.0675` MC | **DISCREPANT** | see §D2 |
| 9 | TOST power at N=24 | `0.042` | `0.0834` exact, `0.0833` MC, `0.0757` from the repo's own `tost()` | **DISCREPANT** | see §D2 |
| 10 | TOST power at N=40 | `0.431` | `0.4325` exact, `0.4326` MC | CONFIRMED | see §D1 |
| 11 | power first clears 0.80 at | `N=70` | N=69 → `0.7985`, N=70 → `0.8059`; normal approximation agrees on N=70 | CONFIRMED | two independent routes, §D1 |
| 12 | `8.8×` the registered N; `≈78 h` per arm | `8.8×`, `78 h` | `70/8 = 8.75`; `8.93 × 70/8 = 78.14` | CONFIRMED | arithmetic |
| 13 | `pivot_hop2(a,P) = a[:,P] @ a[P,:]` has rank ≤ \|P\| | rank ≤ 8 | max observed rank 8 over 16 examples | CONFIRMED | `torch.linalg.matrix_rank` |
| 14 | median `rank(a)` / `rank(a@a)` | `63` / `62` | `63` (all 16) / `62` (all 16) | CONFIRMED | `e3_t8`, s=64, n=16, model seed 0, batch seed 0 |
| 15 | median `rank(pivot_hop2)` | `8` | median `8`; per-example set `{7,8}`, 15 of 16 read 8 | CONFIRMED | same |
| 16 | `cos(hop2, a@a)` | `0.7398` (min `0.5481`, max `0.8385`) | mean `0.7398`, median `0.7394`, min `0.5481`, max `0.8385` | CONFIRMED | same; see §N1 |
| 17 | `‖hop2‖ / ‖a@a‖` | `0.1932` | mean-of-ratios `0.1932`; median `0.1451`; aggregate Frobenius `0.2350` | CONFIRMED | same; see §N1 |
| 18 | dimensions discarded | `54 of 62` | `62 − 8 = 54` | CONFIRMED | arithmetic |
| 19 | `‖a@x‖` identical both arms; `‖hop2@x‖`; ratio; nonzero fraction | `39.7479`, `7.0393`, `0.1771`, `38.54%` | `39.7479` both, `7.0393`, `0.1771`, `38.54%` | CONFIRMED | n=256, `e3_t8`, seeds (0,0) |
| 20 | `max\|softmax − pivot_unsigned\|` | `0.000e+00` | `0.0000e+00` at every seed pair tried | CONFIRMED | direct tensor diff, n=32 |
| 21 | `‖a‖` per arm (softmax / pivot_signed / pivot_unsigned / windowed_signed) | `12.3028 / 15.1003 / 12.3028 / 21.5133` | `12.302814 / 15.100266 / 12.302814 / 21.513271` | CONFIRMED | harness geometry S=64 D=24 d_model=16, n=32, seeds (0,0) |
| 22 | all four arms: min entry, negative fraction | `+0.000000`, `0.0000` | `+0.00000000`, `0.000000` for all four | CONFIRMED | same |
| 23 | other pairwise operator distances (`V13_DAG_TASKLIST`) | `2.273e-01` and `1.832e-01` | `2.2727e-01` and `1.8323e-01` | CONFIRMED | same |
| 24 | `workdonenew`: "`2.273e-01` and `1.832e-01` **from `pivot_unsigned`**" | both distances from `pivot_unsigned` | `pivot_signed↔pivot_unsigned = 2.2727e-01`; `windowed_signed↔pivot_unsigned = 2.2727e-01`; `1.8323e-01` is `pivot_signed↔windowed_signed` | **DISCREPANT** | see §D3 |
| 25 | softmax 0-step: min / max / mean / below-1.0 | `1.00055844 / 1.03349997 / 1.01178392 / 0 of 16` | identical to 8 dp | CONFIRMED | 16 untrained seeds, t\*=2, n=2048, n_eval=4096, no training |
| 26 | pivot_unsigned 0-step | `1.00055861 / 1.03350431 / 1.01178596 / 0 of 16` | identical to 8 dp | CONFIRMED | same |
| 27 | windowed_signed 0-step, seed 2 shortfall | `0.99997039 / 1.03467607 / 1.01216808 / 1 of 16`, short by `2.96e-5` | identical to 8 dp; argmin is seed 2; shortfall `2.9607e-5` | CONFIRMED | same |
| 28 | windowed_signed "wider **spread**" | `3.468e-2` against `3.350e-2` | ranges are `3.4706e-2` and `3.2942e-2`; the written pair is `max − 1.0` | **DISCREPANT** | see §D4 |
| 29 | thread pair readings | `0.971432426855954` @8, `0.9690874072329486` @6 | both present verbatim, `r10_it8_capacity_softmax_t2.jsonl` lines 4 and 17 | CONFIRMED | journal read |
| 30 | drift, seed sd, `Δ_eq`, ratio | `0.002345`, `0.010101`, `0.005051`, `0.464` | `0.0023450196230054`, `0.010101383890203`, `0.005050691945102`, `0.4642967` | CONFIRMED | `Decimal` subtraction + recomputed sd over the 8 threads=6 seeds |
| 31 | cells in `results/` with N=8 distinct seeds at fixed threads | exactly `8` | exactly `8` | CONFIRMED | own filter, §M |
| 32 | crossings of the 1-hop floor | `0` | `0` — no cell's CI upper bound is below its `floor_1` | CONFIRMED | own extraction fed to `cap_verdict` |
| 33 | `ĥ` peak and the five defined values | `0.389`; `0.186, 0.170, 0.267, 0.389, 0.074` | `0.3892096`; same five to 3 dp | CONFIRMED | `t*(1 − mean²)` recomputed |
| 34 | every mean / CI / floor in the census table | 8 rows | all 8 rows reproduce bit-for-bit | CONFIRMED | independent extraction → `cap_verdict` |
| 35 | contract line is `6.63×` below the calibrated model | `3.9 × (3.4/2.0) = 6.63` | `C_OPERATOR = 3.9`, `DTYPE_MODES['bf16_autocast'] = (2.2, 3.4)` → `6.63` exactly | CONFIRMED | `ceq/sizing.py:48,68` + arithmetic |
| 36 | at n=32, s=512, h=4: contract vs calibrated | `0.0671` vs `0.4449` GB/layer | `67,108,864 B = 0.06711 GB`; `3.9·4·32·512²·3.4 = 444,931,768 B = 0.44493 GB` | CONFIRMED | arithmetic against source constants |
| 37 | measured peaks at L=16, signed vs softmax | `2875.8` vs `1153.5` MiB, `2.49×` | `2875.8` and `1153.5` MiB; `2.4931` | CONFIRMED | re-ran `scripts/v13_b9_4060_probe.py --steps 50` on the idle card |
| 38 | contract shape, L=32 totals | softmax `5.806 GB`, signed `20.044 GB` | `5.806` and `20.044` GB at bs=32 (activation + state) | CONFIRMED | `sz.activation_bytes` + `sz.state_bytes` |
| 39 | "checkpointing rescues it to `0.645 GB`" | `0.645 GB` | `0.645 GB` is the checkpointed **activation**; the comparable **total** is `1.102 GB` | **DISCREPANT** | see §D5 |
| 40 | throughput at L=16: softmax / signed / signed+ckpt | `54,837 / 20,194 / 15,090` tok/s | re-run gave `32,433 / 13,401 / 9,981`; no artifact of the original run survives | **UNVERIFIABLE** | see §D6 |
| 41 | K-sweep harness control | `0.951348`, sd `0.014927`, gap `0.001001` | journal reads `0.950252`, sd `0.019256`, gap `0.002097` | **DISCREPANT** | see §D7 |
| 42 | B1 pricing "log-log slope" | `n^1.338` | endpoint slope `1.33813`; 3-point OLS `1.31700` | **DISCREPANT** | see §D8 |
| 43 | B1 extrapolated hours | `0.10 / 1.65 / 7.18 h`, `8.93 h` per arm | exact under slope 1.338 anchored at n=16384; `0.10 / 1.49 / 6.34 → 7.93 h` under OLS | CONFIRMED | see §D8 |
| 44 | V-19: first `bc: command not found` line | line `27` | line `30`; line 27 is whitespace | **DISCREPANT** | see §D9 |
| 45 | V-19: wave B completion and journal N=8 | lines `16–26`, `0.874834 / 0.972372 / 1.006066`, `WAVE B COMPLETE` | all four present at lines 16–26; t\*=8 and t\*=32 carry seeds 0–7 at n=32768; t\*=2 carries seed 0 only | CONFIRMED | `sed -n '14,28p'` + census |
| 46 | λ₂ band coverage | `0.9435 … 0.9499`, span `0.0064`, `12.9%` | `0.943451 … 0.949939`, span `0.006488`, `12.98%` | CONFIRMED | `r10_it15_dual_oracle.jsonl` `per_rung.lambda_2`; see §N2 |
| 47 | itemised ceiling | `41`, live `39` | `12+3+2+8+2+3+2+5+4 = 41`; `41 − 2 = 39` | CONFIRMED | arithmetic |
| 48 | crossing gaps and needed means | `0.700107 / 0.933779 / 0.982953`; `26.5% / 4.3% / 1.6%` | identical; needed mean is `floor_1 − 1.96·sd/√8` | CONFIRMED | reverse-derived then recomputed |
| 49 | first arm contrast | mean `0.956525`, sd `0.013133`, `+0.004176`, 90% CI `[−0.006189, +0.014542]` | identical; Welch df `13.135` | CONFIRMED | recomputed from `r10_v13pilot_capacity_pivot_unsigned_t2.jsonl`; all 8 seed values match |
| 50 | windowed_signed at t\*=2 | `1.021366`, `1.016935`, train `0.7747378913590046`, `payload_only 1.2272241529888959` | all four verbatim in the journal | CONFIRMED | journal read |
| 51 | `floor_2` at t\* = 2 / 8 / 32 | `0.000000 / 0.866025 / 0.968246` | `√((t*−2)/t*)` gives the same | CONFIRMED | arithmetic |
| 52 | `ĥ` at t\*=2, n=2048 | `0.186` softmax, `0.170` pivot_unsigned | `0.186064`, `0.170120` | CONFIRMED | arithmetic |
| 53 | host logical cores | `28` | `os.cpu_count() = 28`, `torch.get_num_threads() = 20` | CONFIRMED | direct read |
| 54 | M-12 sizing figures | `1.11/1.01/0.99`, `0.98/0.94/0.91`, `204.8 MiB` flat, `1.41/1.61/2.00`, `243.7` vs `1160.0` = `4.8×` | every figure reproduces at `dtype='fp32'` | CONFIRMED | `sz.activation_bytes` at fp32 against the re-run peaks; see §N3 |

---

## Discrepancies

### D1. The TOST power route, stated so the two disagreements below are readable

Power was recomputed two ways that agree with each other and were built without
reference to the repository's implementation.

*Exact.* Condition on `u = s_p/σ`, where `u²·df ~ χ²_df` and `df = 2N−2`. The
TOST rejects iff `c − m < d̄/σ < m − c`, where `c = t(1−α,df)·u·√(2/N)` and
`m = Δ_eq/σ = 0.5`. Integrating `2Φ((m−c)/√(2/N)) − 1` against the χ density of
`u` gives the power at a true difference of zero.

*Monte Carlo.* 400,000 replications of the actual two-sample pooled-variance
TOST at `α = 0.05`, drawing both samples from the same standard normal.

The two agree to within simulation error at every N tried (N=8: 0.0006 vs
0.0005; N=24: 0.0834 vs 0.0833; N=40: 0.4325 vs 0.4326; N=70: 0.8059 vs 0.8061).

### D2. Two power figures are the normal approximation, not the test's power

| N | as written | exact | Monte Carlo | repo's own `tost()` | normal approx (σ known) |
|---|---|---|---|---|---|
| 8 | `0.000` | 0.0006 | 0.0005 | 0.0008 | 0.0000 |
| 23 | `≈0.03` | **0.0668** | 0.0675 | — | 0.0122 |
| 24 | `0.042` | **0.0834** | 0.0833 | 0.0757 | **0.0426** |
| 40 | `0.431` | 0.4325 | 0.4326 | 0.4512 | 0.4323 |
| 70 | first `≥0.80` | 0.8059 | 0.8061 | 0.8223 (at N=72) | 0.8071 |

The written `0.042` at N=24 matches the σ-known normal approximation to three
decimals and matches nothing else. The exact TOST power there is `0.0834` —
almost exactly double. The written `≈0.03` at N=23 matches neither route: exact
is `0.0668`, normal approximation `0.0122`.

Direction of the error: both written values **understate** the design's power,
so `M-13`'s conclusion is conservative rather than flattering, and the N=70
headline is unaffected — the exact route and the normal approximation both put
the first crossing of 0.80 at N=70. What breaks is the middle of the table: at
N=24 the design is roughly twice as able to certify equivalence as `M-13` says,
which changes how a reader prices an intermediate seed count.

Fix: replace the N=23 and N=24 power cells with `0.067` and `0.083`, or state
that the column is a normal approximation and recompute it consistently
(N=23 → `0.012`, N=40 → `0.432`).

### D3. `workdonenew.md` attributes the wrong operator distance to the wrong pair

Written: "`pivot_signed` and `windowed_signed` are genuinely distinct
(`2.273e-01` and `1.832e-01` from `pivot_unsigned`)".

Measured at the harness geometry, n=32, model seed 0, batch seed 0:

| pair | max abs difference |
|---|---|
| softmax ↔ pivot_unsigned | `0.0000e+00` |
| softmax ↔ pivot_signed | `2.2727e-01` |
| softmax ↔ windowed_signed | `2.2727e-01` |
| pivot_signed ↔ pivot_unsigned | `2.2727e-01` |
| **pivot_unsigned ↔ windowed_signed** | **`2.2727e-01`** |
| **pivot_signed ↔ windowed_signed** | **`1.8323e-01`** |

`windowed_signed`'s distance from `pivot_unsigned` is `2.2727e-01`, not
`1.832e-01`. `1.8323e-01` is the distance between the two *sgate* arms, which is
the one pair the sentence does not name. `V13_DAG_TASKLIST.md`'s wording — "the
pairwise distances among the other combinations are `2.273e-01` and
`1.832e-01`" — is correct; the `workdonenew.md` row inherited the numbers and
added a wrong attribution.

Second-order note, not in the written record: `1.8323e-01` is the only one of
the six that moves with the draw. At batch seed 12345 it reads `2.0130e-01`
while every other entry is unchanged to four significant figures, because it is
a max over a banded-versus-unbanded difference rather than over a structural
one. A distance quoted to four figures from a single draw should carry that.

### D4. The 0-step "spread" figures are `max − 1.0`, not spreads

`M-14` writes: "`windowed_signed` has a marginally wider spread (`3.468e-2`
against `3.350e-2`)".

Measured over the same 16 untrained seeds:

| arm | max − min (the spread) | max − 1.0 (what was written) |
|---|---|---|
| softmax | `3.2942e-2` | `3.3500e-2` |
| windowed_signed | `3.4706e-2` | `3.4676e-2` |

The written pair reproduces `max − 1.0` to four significant figures for both
arms and reproduces neither range. The comparison's direction survives under
either definition (`windowed_signed` is wider both ways), so the entry's
argument holds; the label does not. The correct spreads are `3.471e-2` against
`3.294e-2`, a 5.4% difference rather than the 3.5% the written pair implies.

### D5. A checkpointed activation figure is compared against two totals

`workdonenew.md`: "softmax totals `5.806 GB` … while signed totals `20.044 GB`
… Gradient checkpointing rescues it to `0.645 GB`."

Recomputed at bs=32, seq=512, d=256, heads=4, L=32:

| arm | activation | state | total |
|---|---|---|---|
| softmax | 5.349 GB | 0.457 GB | **5.806 GB** |
| signed | 19.586 GB | 0.457 GB | **20.044 GB** |
| signed + checkpoint | **0.645 GB** | 0.457 GB | 1.102 GB |

The first two are totals; the third is an activation. The comparable total is
`1.102 GB`. Both fit the 8 GB card, so the sentence's conclusion is unaffected,
but as written it understates the checkpointed footprint by 1.71×.

### D6. The B9 throughput triple cannot be reproduced, and its artifact is empty

`results/r10_v13_b9_4060_probe.txt` is **0 bytes**. `scripts/v13_b9_4060_probe.py`
writes only to stdout, so nothing of the original run survives on disk. Grepping
`results/`, `scale/` and `scripts/` for `54,837`, `54837`, `20194` and `15090`
returns no match outside the prose that quotes them.

Re-running the script unmodified at `--steps 50` on the idle card:

| arm | L | ckpt | peak MiB (written) | peak MiB (re-run) | tok/s (written) | tok/s (re-run) |
|---|---|---|---|---|---|---|
| softmax | 16 | no | `1153.5` | **1153.5** | `54,837` | 32,433 |
| signed | 16 | no | `2875.8` | **2875.8** | `20,194` | 13,401 |
| signed | 16 | yes | — | 409.2 | `15,090` | 9,981 |

Peak bytes reproduce exactly — memory is deterministic and the memory claim is
CONFIRMED (row 37). Throughput does not: the re-run reads 59% of the published
softmax rate. The published ratios `2.72×` and `3.63×` recompute exactly from
the published triple (`54837/20194 = 2.7155`, `54837/15090 = 3.6340`), so the
arithmetic is internally sound; the re-run's own ratios are `2.42×` and `3.25×`.

The host was running sibling CPU training lanes throughout this audit, and this
script's launch path is CPU-bound between kernels, so the shortfall is
attributable to load rather than to the claim. That is exactly why it is marked
UNVERIFIABLE and not DISCREPANT: nothing here shows the written numbers are
wrong, and nothing here can show they are right. A throughput figure quoted to
five significant figures from a run whose only artifact is a zero-byte file
cannot be checked by anyone, including its author.

Fix: re-run with the output captured, and record the concurrent load.

### D7. The K-sweep harness control quotes numbers that are not on disk

`V13_DAG_TASKLIST.md` writes: "the standalone script's own softmax control was
checked against the published sweep at the same cell: `0.951348` (sd `0.014927`,
seeds 0-2) against `0.952349` (sd `0.010101`, N=8). The `0.001001` gap sits well
inside both spreads."

`results/r10_v13_kpivot_t2.txt`, the sweep's own journal, reads:

```
task=e3_t2 n=2048 steps=150 seeds=[0, 1, 2] threads=8  floor_2=0.000000
  softmax (1 hop)      mean 0.950252  sd 0.019256  ['0.971432', '0.933802', '0.945523']
```

| quantity | as written | on disk |
|---|---|---|
| control mean | `0.951348` | **`0.950252`** |
| control sd | `0.014927` | **`0.019256`** |
| gap against `0.952349` | `0.001001` | **`0.002097`** |

Neither `0.951348` nor `0.014927` appears anywhere in `results/`, `scale/` or
`scripts/`. The journal's mtime is `02:30`; `V13_DAG_TASKLIST.md`'s is `02:35`,
so the file was already in its present state five minutes before the paragraph
was written — a stale read does not explain it.

Three consequences, in increasing order of weight.

1. The gap is `0.002097`, not `0.001001` — 2.1× larger.
2. The sd is `0.019256`, not `0.014927`. Since the paragraph's whole argument is
   "the gap sits well inside both spreads", using a spread 23% too narrow makes
   the argument *harder*, so the conclusion survives its own error.
3. The control crosses thread lanes. The journal header records `threads=8`;
   the published N=8 softmax cell is `threads=6`. This is the pooling
   `it11_verdict.by_seed` refuses by design and that `M-10` was filed to
   prevent. The control's seed-0 value is `0.971432`, which is the *threads=8*
   reading recorded at `r10_it8_capacity_softmax_t2.jsonl:4`, not the
   `threads=6` reading at line 17. The measured gap `0.002097` is `0.894` of
   the thread floor `0.002345`, so at these thread counts the control cannot
   separate "harness differs" from "thread count differs" at all. It is not a
   harness control; it is a thread-count contrast wearing one.

Also worth recording because the surrounding prose treats the sweep as pending:
the journal already holds all four K rungs, and the reading gets *worse* as K
rises — `+0.010692` at K=8, `+0.018266` at K=16, `+0.015497` at K=32,
`+0.050076` at K=64, all against the control, all in the direction of higher
NRMSE. That is the second of the two branches the tasklist pre-registers, not an
open question.

### D8. "Fitted slope" is an endpoint slope

`V13_DAG_TASKLIST.md`: medians `n=2048 → 18.19 s`, `n=8192 → 94.74 s`,
`n=16384 → 293.94 s`, "a log-log slope of `secs ~ n^1.338`".

- Two-endpoint slope, n=2048 to n=16384: `ln(293.94/18.19)/ln 8 = 1.33813`.
- Ordinary least squares over all three points: **`1.31700`**.
- Adjacent pairwise slopes: `1.1904` and `1.6335` — the relation is convex over
  the measured range, which is why the endpoint slope and the fit differ.

The derived hours are internally exact under the endpoint slope anchored at the
n=16384 point (`0.102 / 1.651 / 7.181 h`, summing to `8.93`), so row 43 is
CONFIRMED as arithmetic. Under the OLS fit the same extrapolation gives
`0.10 / 1.49 / 6.34 h`, `7.93 h` per arm — 11% cheaper, and the N=70 price falls
from `78 h` to `69 h` per arm. The paragraph correctly calls its own result "a
floor rather than an estimate", so the direction is already caveated; the word
"fitted" is the part that is wrong.

### D9. V-19's line number for the first `bc` failure

`MISTAKES.md` V-19 and `V13_DAG_TASKLIST.md`'s B0 correction both state that
"the first `bc: command not found` appears at line 27, after it".

`grep -an "bc: command not found" results/r10_it8_waveB.log | head -1` returns
**line 30**. Line 27 is a whitespace-only line; lines 28–29 are the empty
substitution output `waveAC WROTE lines:  / 6`.

The claim this line number supports is unaffected and CONFIRMED: the completed
wave sits at lines 16–26 with `WAVE B COMPLETE` at 26, and 30 > 26. Only the
cited index is wrong. The entry's own prescribed check — quote the artefact used
to establish the absence — is satisfied in substance and off by three in the
citation.

---

## Notes on statistics that verify but are not what they appear to be

### N1. The hop-2 magnitude ratio is a mean of ratios in a column of medians

Row 17 verifies exactly, but the table it sits in is headed "median / mean" and
its two neighbouring rows are medians. `0.1932` is the **mean** of the 16
per-example ratios `‖hop2_i‖/‖(a@a)_i‖`. The median is `0.1451` and the
aggregate Frobenius ratio `‖hop2‖_F/‖a@a‖_F` over the whole batch is `0.2350`.
The distribution is right-skewed (min `0.0424`, max `0.5191`), so the three
summaries differ by up to 62%.

`workdonenew.md` §5 then writes "It retains `cos = 0.7398` of the full hop's
direction but only `0.1932` of its magnitude" in the same breath as "median rank
**62**" and "median rank **8**", which reads as four medians and is two medians,
a mean-of-cosines and a mean-of-ratios. The typical routed hop keeps `0.145` of
the magnitude, not `0.193`. The conclusion — a rank-8 approximation keeping a
fraction of the magnitude — is unaffected in every direction.

### N2. λ₂ band coverage is truncated, not rounded

`per_rung.lambda_2` in `results/r10_it15_dual_oracle.jsonl`: 12 rungs, min
`0.943451`, max `0.949939`, span `0.006488`, which is `12.98%` of the declared
width `0.050`. Written as span `0.0064` and `12.9%`; rounding to the quoted
precision gives `0.0065` and `13.0%`. Both are truncations toward the
narrative's direction. Individually negligible; recorded because the same
truncation appears in both files.

### N3. The B9 probe's own printed prediction is not the one `M-12` quotes

`scripts/v13_b9_4060_probe.py` runs the model in fp32 but calls
`sz.activation_bytes` at its `dtype='bf16_autocast'` default, so the script's
`model_MiB` and `ratio` columns are an fp32 measurement against a bf16
prediction. At L=16 the script prints `641.6 MiB` predicted for checkpointed
softmax and `153.7 MiB` for checkpointed signed.

`M-12` quotes `1160.0` and `204.8`, which are the **fp32** predictions — and
every one of `M-12`'s figures reproduces exactly at `dtype='fp32'`
(`1.11/1.01/0.99`, `0.98/0.94/0.91`, `204.8` flat in L, `1.41/1.61/2.00`,
`4.76×`). `M-12` is right and the script is the thing that is inconsistent: a
reader who reproduces `M-12` by running the script will get different numbers
than `M-12` reports and will conclude the entry is wrong. The entry is not
wrong; the script's default dtype does not match its own measurement.

---

## M. The census filter, stated so a disagreement is locatable

Row 31 is the count most likely to move under a different de-duplication rule,
so the rule used here is given in full.

- Source: every `results/*.jsonl`. Verified that no `"t": "cell"` row exists
  anywhere in `results/` outside `.jsonl` files.
- A row counts iff it parses as a JSON object with `"t" == "cell"` and carries
  `t_star`, `seed` and `eval_nrmse`.
- Arm: the row's own `arm` field; failing that the most recent `"t":"header"`
  row's `arm` in the same file; failing that the arm name in the filename.
- Group key: `(arm, t_star, n_train, threads, steps)` — threads included
  because the harness is deterministic given threads and not across them.
- De-duplication: first occurrence per `seed` within a group. Every repeated
  `(group, seed)` was checked for a conflicting `eval_nrmse`; **none** was
  found, so no de-duplication choice can change the result.
- Qualifying: `≥ 8` distinct seeds. Exactly 8 groups qualify, each with exactly
  seeds 0–7.

Robustness: dropping `steps` from the key changes nothing, because every group
at `steps ≠ 150` holds one seed. Dropping `threads` from the key also leaves 8,
because the merged groups gain only seed-0 rows they already contain.

The 8 groups and their source files:

| arm | t\* | n | threads | file |
|---|---|---|---|---|
| pivot_unsigned | 2 | 2048 | 6 | `r10_v13pilot_capacity_pivot_unsigned_t2.jsonl` |
| softmax | 2 | 2048 | 6 | `r10_it8_capacity_softmax_t2.jsonl` |
| softmax | 8 | 2048 | 6 | `r10_it8_capacity_softmax_t8.jsonl` |
| softmax | 8 | 16384 | 12 | `r10_it8_capacity_softmax_t8.jsonl` |
| softmax | 8 | 32768 | 12 | `r10_it8_capacity_softmax_t8.jsonl` |
| softmax | 32 | 2048 | 6 | `r10_it8_capacity_softmax_t32.jsonl` |
| softmax | 32 | 32768 | 12 | `r10_it8_capacity_softmax_t32.jsonl` |
| softmax | 32 | 49152 | 12 | `r10_it8_capacity_softmax_t32.jsonl` |

Nearest misses, for anyone auditing the boundary: `pivot_unsigned` at
`t*=8, n=32768, threads=12` holds 2 seeds, and `windowed_signed` at
`t*=2, n=2048, threads=6` holds 2. Both are correctly excluded.

---

## What the load-bearing claims survived

The four claims the round's argument rests on were attacked hardest and all
four hold.

**N=70.** Checked by three routes that share no code: an exact conditional
integration over the χ distribution of the pooled sd, a 400,000-replication
Monte Carlo of the literal two-sample TOST, and the σ-known normal
approximation. All three put the first crossing of 0.80 at N=70. This survived
despite two neighbouring cells in the same table being wrong, which is the
strongest evidence available that the headline was computed and not assumed.

**The census count of 8.** Reproduced from an independent extraction with a
stated filter, checked for duplicate-row conflicts (none), and shown invariant
to dropping either `steps` or `threads` from the grouping key.

**`max|softmax − pivot_unsigned| = 0.000e+00`.** Reproduced at three
independent seed pairs. The identity is structural — `Arm._operator` returns
`bench._softmax_operator(q,k)` for both kinds at `scale/m3_capability.py:118-124`
— so no draw can break it.

**The 0-step null.** All nine published statistics reproduce to eight decimal
places, and re-running at `threads=12` returns bit-identical values, so the
measurement is not thread-sensitive and the numbers are reproducible by anyone
with the repository.

## What would have been needed to catch more

The errors this audit found are of three kinds, and each was catchable by a
different discipline:

- **Two routes to the same number** (D2, D8). Every figure that agreed with
  itself and with nothing else was a figure computed once. The normal
  approximation and the exact power differ by 2× at N=24 and are
  indistinguishable at N=70, which is why the table's headline was safe and its
  middle was not.
- **Naming the statistic** (D4, N1, D5). Three separate figures verify exactly
  as `max − 1.0`, as a mean-of-ratios, and as an activation — while being
  written as a spread, as a median-column entry, and as a total.
- **Reading the artifact after writing the prose** (D6, D7, D9). The K-sweep
  control is the only finding that changes what a reader should believe about
  the instrument rather than about a number, and it was found by opening a
  478-byte file.

---

## Limits

- **Throughput on the card could not be checked.** Row 40 is UNVERIFIABLE, not
  CONFIRMED. The host ran sibling CPU lanes throughout, the re-run's absolute
  rates are 59–66% of the published ones, and the original run left a zero-byte
  artifact. Whether the published triple is right is undetermined either way.
- **Seed provenance for the operator and hop-2 probes was inferred, not
  recorded.** Rows 14–23 reproduce exactly at model seed 0 and batch seed 0, and
  the match to six significant figures across seven independent quantities makes
  the inference near-certain — but no script in the tree produces those numbers,
  so the settings were recovered by search rather than read. `‖a‖`, `cos` and
  the magnitude ratio all move with the draw; only `max|softmax −
  pivot_unsigned| = 0`, the ranks and the zero negative fraction are
  draw-independent.
- **The one figure in D3 that moves with the draw** (`1.8323e-01`) was checked
  at two batch seeds only.
- **`0.951348` and `0.014927` (D7) could not be traced to any source.** The
  audit establishes that they contradict the on-disk journal and appear nowhere
  in `results/`, `scale/` or `scripts/`; it does not establish where they came
  from. A run whose output was never captured is the likeliest explanation and
  is not evidence.
- **Nothing was re-trained.** Rows 29, 30, 34, 49 and 50 verify that the
  journals are internally consistent and that every derived statistic follows
  from them. They do not verify that the training runs which produced those
  journal rows were correct. A defect inside `train_with_checkpoints` would be
  invisible to this audit.
- **`V-20`, `V-21` and `V-22` were read but carry no recomputable figures** in
  the ranges checked, apart from `V-21`'s corroboration of the wave-B journal
  (row 45) and `V-22`'s scaling arithmetic, which was spot-checked
  (`0.398107/0.251189 = 1.58489` against `10^0.2 = 1.584893`;
  `(10^-2)^0.2 = 0.398107`, `(10^-3)^0.2 = 0.251189`) and holds. The prior-art
  claims inside them are outside this audit's remit.
- **The claims in `V13_DAG_TASKLIST.md` sections C and D** (contract defects
  C-1 through C-5, the conditioning limit) were not recomputed. They rest on
  `scripts/v13_derivation_check.py`, which was not re-run.
- **No file other than this one was written**, and no git command that writes
  was issued.
