# Section M — The inventory of measured facts, with provenance

*Role: WATSON (instruments, the journal). Repository HEAD `207e7b9`, branch `v17k-gate0`,
2026-09-03. Evidence classes: `RUN` = parsed or executed this session from `results/*.jsonl`
or `results/*.json` with read-only python one-liners; `READ path:line` = quoted from the
named file; `DERIVED` = arithmetic written out; `CITED` = external source (none are needed
in this section; every number here has a producer inside the tree or is marked NOT
MEASURED). No `house-events.jsonl` grep was run; nothing was written outside this file.*

This section exists so the paper never writes a number it cannot point at. Every row
carries: the number, the producing file and line (or journal path), the seed count, the
device, and the control it was compared against. Where the round-15 corrections index
(`V20_R15_JOURNAL.md` C1–C40, `READ V20_R15_JOURNAL.md:24-76`) overturned a figure, the
corrected figure is carried and the row is cited. The mechanism each table is designed
against is named from `MISTAKES.md` (66 `###` mechanism headings including `V-14a`,
`RUN grep -c '^### [VPMD]-' MISTAKES.md`; the record's own count is 65,
`READ workdonenewseal.md:359`).

---

## M.1 Identity binds — residual, bar, planted negatives, file

Design mechanism: **V-24** (an identity bind whose rejection region is empty,
`READ MISTAKES.md:1658`) — every bind below ships a planted mutilation that fails at O(1);
**V-25** (a theorem whose hypothesis no draw satisfies, `READ MISTAKES.md:1954`) — the
bind is measured on BED-M's *real* support `{−1, 0, +1}`, not on a toy draw.

| bind | residual | bar | shape / draws | device | control (planted negative → residual) | source |
|---|---|---|---|---|---|---|
| oracle gates ⇒ label, `arm_smprime`, path-product corner (`β=0`, QK off, `V=b` unrescaled) | `5.919777e-16` (real part exactly `0.000e+00`, all of it `polar(1,π)` imaginary dust) | `1e-6` | `n=64, s=64`, zero-hop fraction `0.967788`; `n=512, s=64` and `n=256, s=128` both `7.550528e-16` | cpu & cuda, float64/complex128 | `drop_phase 1.934830`, `drop_magnitude 0.466267`, `beta_one 1.335288`, `exp_scan nan` (133,120/133,120 NaN); honest cell `1.110223e-16`; manifest hash moves under all four | `READ V16_ARM_SMPRIME.md:25,195-199,238-244`; `READ workdonenewseal.md:91-94,113-122` |
| label bind, `arm_pl` (the telescope head, `l_ij = q_ij − C_j + s_j`, value-zero BOS) | `6.661338147750939e-16` at `s=8`; `7.549516567451064e-15` at `s=64` | `1e-6` | one draw per `s`, `np.random.default_rng(15)`, gates on `(0.15, 0.85)` | cpu | `drop_key_bias 0.97494590151405114`, `drop_value_rescale 0.9165274652308163`, `drop_bos_sink 1`, `half_key_bias 0.48449311856985267` — four for four | `RUN results/v15_r1.jsonl t="bind"`; `READ V15_ARM_PL.md:183-200` |
| label bind, `arm_phase` (complex chain) | `9.155133597044475e-16` at `s=8`; `5.2510145522368515e-15` at `s=64`; **`nan` on BED-M's band** (`|a|=1` ⇒ `log(1−1) = −inf`) | `1e-6` | one draw per `s` | cpu & cuda bitwise-identical | the band failure is the planted negative and is not adjusted away | `READ V15_ARM_PHASE.md:36`; `READ workdonenewseal.md:97-98` |
| softmax corner, `arm_smprime` (`β=1, g≡0, QK-on`) | **bitwise** against `#5a`'s own `softmaxAttn`; `1.110223e-16` on 19/64 entries against `ceq/lm.py::Attention("softmax_x")` (fused kernel subtracts the row max; not a tolerance, a named mechanism) | bitwise | `q,k ~ N(0,1)` at `[8,4]`, seed 3, float64 | cpu | wrong switch: `β=0` at the softmax corner `max|gap| > 0.5`; gate left on at `β=1` `> 0.1` | `READ V16_ARM_SMPRIME.md:27,266-293,336-339` |
| row sums at the softmax corner | `Σ_j Re W_ij = 2.220446e-16` from 1 at `β=1`, every row; at `β=0` rows read `1.312192 … 10.293107` | exact | 8 rows | cpu | the earlier statistic `Σ_j |W_ij| = 1.000000` was **struck** — it equals `Z_i^{1−β}` and is blind to `g` (read `4.440892e-16` with `g` fully on in a configuration whose real row sum is off by `1.750255`) | `READ README.md:261-275` (§4.7); `READ V16_ARM_SMPRIME.md:29` |
| standard-attention identity, `arm_phase` (`m=1, θ=0, s=0`) | `torch.equal(op.real, lm softmax_x) = True`; `op.imag` exactly 0; complex-gemm read-out **not** bitwise: `1.1102230246251565e-16` | bitwise | one probe | cpu | the read-out gap is named, not rounded | `READ V15_ARM_PHASE.md:35,241-243` |
| `|a| ≤ 1` by construction, `arm_phase` | worst `|a| = 1.0` exactly (the repr), `count(|a|>1) = 0` | exact | 2,200,000 draws (200,000 log-uniform `|u|` on `[3.78e-11, 2.65e10]`, 2,000,000 pinned at `m=1`); also at `u=±inf`, `±1e308`, and R1's own eight `â_max` including `285.0719` | cpu | none needed — an upper endpoint attained is the demonstration | `READ V15_ARM_PHASE.md:32` |
| band modulus `|Π e^{iθ}|` | `1.000000000000`; `9767/10⁴` prefixes exactly `1.0`, `233` one ulp low, `0` above | exact | `1e4` phases, `s=64` | cpu (cuda reads `7713` exactly-one — a summation-order split) | independent cumulative-product route `0.9999999999999843` | `READ V15_ARM_PHASE.md:33,169-172`; `READ V16_ARM_SMPRIME.md:315` |
| parity mask = `Z₂` winding | `torch.equal`, `0/4096` disagreements with `V15.lean::parity_sign`; integrality residual `2.131628e-14` | exact | `θ ∈ {0,π}`, `s=64` | cpu | continuous route `exp(i(Φ_i−Φ_j))` agrees only to `6.762526e-14` — reported beside | `READ V15_ARM_PHASE.md:34,203` |
| DAG resolvent vs brute-force path sums | light gates `(0.05,0.45)`, `p=0.6`: `2.7755575615628914e-17` abs (19 edges); heavy gates `(0.30,0.90)`, `p=1.0`: `3.5527136788005009e-15` abs, `2.1572667249528595e-16` relative, entries to `16.468587948382389` (36 edges) | `1.1e-16` inherited, re-measured | one draw per regime | cpu | the two routes share no code — dense LU vs enumeration of increasing vertex sequences (V-3 guard) | `READ V15_ARM_PL.md:153-177` |
| path-product corner, cpu vs cuda | cpu `0/64` entries moved; cuda `11/64` moved by `≤ 5.551115e-17` (`cumprod` is a parallel scan on cuda); **zeros identical on both devices** | bitwise (cpu), reported (cuda) | `[8,4]` probe | both | on BED-M's own support `{−1,0,+1}` the association order does not matter: `0/289` entries move, `max|gap| = 0.000000e+00` | `READ V16_ARM_SMPRIME.md:308-329` |
| corners distinct (`corners_are_distinct` in float64) | `|c₁−c₂| = 4.472918`, `|c₁−c₃| = 1.144938`, `|c₂−c₃| = 5.335671` | `> 0` | one probe | cpu | a containment whose corners coincide is decoration | `READ V16_ARM_SMPRIME.md:331-334` |

**What the binds do not claim.** The two-branch form `softmax(q)@V₁ + λ·X@V₂` passes
parity bitwise for the honest gate, for Gaussian noise, and for the label itself — an
empty rejection region, filed V-24 (`READ workdonenewseal.md:124-126`). The parity row
must read "the modification enters only through the key logit, additively, and vanishes
at zero", never "bitwise standard attention" unqualified (`READ workdonenew.md:137-146`).

---

## M.2 The deciding measurement R1, and its re-take on the certified device

Design mechanism: **M-10** thread-count floor (a cell's identity includes its thread
count, `READ workdonenew.md:363-366`), **M-16** thread lane, **V-26** (a marginal standing
in for a joint claim — the bimodal split is reported, not the mean alone), **P-3** (the CPU
journal is superseded by an append-only marker, never edited).

### M.2.1 R1 as taken (CPU) — `results/v15_r1.jsonl`

BED-M, `task=e3_t2`, `t*=2`, `s=64`, `d=24`, `d_model=16`, `n_train=2048`, `n_eval=4096`,
`steps=150`, `threads=8`, N=8 seeds per arm, device `cpu`; the CPU file keeps its 25
original lines plus one appended `t="supersede"` marker (26 lines, `RUN`; md5 of the first
25 lines `3559fd5f24ba0dfb67ac176c4584f58e`, `READ V17_R4_RETAKE.md:324-326`).

| quantity | value | source |
|---|---|---|
| `floor₁ = √((t*−1)/t*)` | `0.7071067811865476` (a value no line of `scripts/v15_r1.py` contains; identity at `:17`, computed at `:586`) | `READ V20_R15_THEORY_TABLE.md:71-72` |
| `arm_pl` per seed, `eval_nrmse` | 0: `0.645614`, 1: `0.644454`, 2: `1.152430`, 3: `1.113403`, 4: `0.634002`, 5: `0.641881`, 6: `0.662021`, 7: `1.139404` | `RUN results/v15_r1.jsonl` |
| `arm_pl` aggregate | mean `0.829151`, sd `0.253673`, 95 % CI `[0.617075, 1.041227]`, `ĥ = 0.625017`, `crosses = False`, achieved power `0.232077`, `â_max = 285.0719` | `RUN` (`t="agg"`) |
| `softmax` per seed | 0: `0.971432`, 1: `0.933802`, 2: `0.945523`, 3: `0.951174`, 4: `0.960759`, 5: `0.945679`, 6: `0.960282`, 7: `0.945482` | `RUN` |
| `softmax` aggregate | mean `0.951767`, sd `0.011824`, CI `[0.941881, 0.961652]`, `ĥ = 0.188` | `RUN` |
| **bimodal split** | crossed 5/8 (`0.634002 … 0.662021`, `ĥ 1.1235 … 1.1961`, gate-`R²` `0.97 … 0.99`, `â_max 1.10 … 1.51`); NO READING 3/8 (`1.113403 / 1.139404 / 1.152430`, gate-`R²` `0.011 / 0.627 / 0.047`, `â_max 20.31 / 49.66 / 285.07`); **nothing between `0.663` and `1.113`** — the reported mean is a value no seed produced | `READ V15_R1.md:177-188`; `READ README.md:166-173` |
| paired contrast (PL − softmax) | mean `−0.122616`, sd `0.257560`, resolution `Δ = t(.975,7)·sd/√8 = 0.215326` — "excludes a difference beyond Δ and nothing smaller" | `RUN` (`t="contrast"`) |
| the improvement against the noise floors | `0.122616 / 2.345e-3 = 52.3×` the thread-count floor; `0.122616 / 0.005051 = 24.3×` the equivalence margin `Δ_eq = 0.5σ_seed` | `DERIVED` from `RUN` and `READ workdonenew.md:391` |
| `sign(a)` probe | trained `0.917953` vs zero-step `0.943741` — **negative** trained gain; gate-`R²` `0.699299` trained vs `0.371839` zero-step; `loga_sst = 0.0` | `RUN` (`t="probe"`) |
| bar at this cell | `BAR CALIBRATED`: `predict_the_mean 1.0`, `payload_only 1.2272241529888959`, `oracle 0.0`, `flipper_dependence 1.4012436552018552`, `trained_two_feature 0.013981593578261986` | `RUN` (`t="bar"`) |
| wall clock | `1358.3225734233856 s` total for 16 cells; `arm_pl 91.633625 s`/150 steps; softmax `71.32 s` | `RUN` (`t="wall"`); `READ V15_R1.md:482-484` |
| seed-0 softmax reproduces the journal | `0.971432426855954` at `threads=8`, all fifteen digits, against `MISTAKES.md` M-10's recorded value | `READ V15_R1.md:193-198` |

**Verdict as recorded:** NOT CROSSED; the CI straddles `floor₁`; `C-CAP` not earned
(`READ workdonenewseal.md:136-154`). Two campaign firsts inside the negative verdict:
`ĥ = 0.625` against the nine-cell ceiling `0.389`, and five seeds below a floor no prior
arm had crossed at any cell (`READ workdonenew.md:220-233` for the nine-cell census).

### M.2.2 The re-take on the certified RTX 4060 — `results/v17k_r4_retake.jsonl`

24 cells, one invocation, `device="cuda"`, `threads=8`, `steps=150`, instrument hash
`5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309` on all 24,
`deterministic_algorithms: True`, `deterministic_warn_only: True`,
`CUBLAS_WORKSPACE_CONFIG=":4096:8"` (`RUN`, header record; `READ V17_R4_RETAKE.md:14-16`).

| arm | N | mean | sd | CI95 | `ĥ` | below `floor₁` | `crosses` | source |
|---|---|---|---|---|---|---|---|---|
| `arm_pl` | 8 | `0.830200` | `0.255414` | `[0.616669, 1.043731]` | `0.621535` | 5/8 (seeds 0,1,4,5,6) | `False` | `RUN` |
| `arm_smprime` (R1′, first-ever readings, no CPU counterpart) | 8 | `0.822193` | `0.250321` | `[0.612920, 1.031467]` | `0.647997` | **1/8** (seed 2 at `0.20391993939877656`) | `False` | `RUN`; `READ V17_R4_RETAKE.md:113-120` |
| `softmax` | 8 | `0.952561` | `0.011452` | `[0.942987, 0.962135]` | `0.185256` | 0/8 | `False` | `RUN` |

| quantity | value | control | source |
|---|---|---|---|
| paired contrast `arm_pl − softmax` (cuda) | `−0.122360`, sd `0.260159`, `Δ = 0.217498` | CPU `−0.122616`, `Δ = 0.215326`; move `+2.56e-04`, `+1.0 %` | `RUN`; `READ V17_R4_RETAKE.md:200-207` |
| paired contrast `arm_smprime − softmax` | `−0.130367`, sd `0.247507`, `Δ = 0.206921` | none (no CPU record exists) | `RUN` |
| verdicts moved across the device | **zero** sign flips in 16 paired cells; seeds 0/1/4/5/6 stay below, 2/3/7 stay above, all 8 softmax stay above | per-cell `dist_to_floor` sign | `READ V17_R4_RETAKE.md:194-196` |
| worst CPU↔CUDA delta | `9.522e-03` at `arm_pl` seed 7 (`1.139404 → 1.148927`); `4.06×` the thread floor `2.345e-03`; `4.42 %` of the resolution statement; `2.20 %` of the tightest crossing margin | two other non-learning seeds read `1.505e-04` (seed 2) and `6.329e-05` (seed 3) — "did not learn" is not the mechanism | `READ V17_R4_RETAKE.md:129-165,187-192` |
| what moved on seed 7 | `a_hat_max 285.0719 → 116.0061` (`2.5×`), `sign_acc` bitwise identical (`0.6905517578125`), `gate_r2` moves `1.5e-04`, `v_max` bitwise | the device changes how far an unbounded quantity ran, not what the seed does | `READ V17_R4_RETAKE.md:167-172` |
| identical-seed floor `δ_nrmse` | `0.0`, bitwise, 6 of 6 pairs (seeds 0,1 × three arms), `results/v17k_r4_floor.jsonl`, `44.7 s` | seed-to-seed spread `0.2554 / 0.2503 / 0.0115` is `≥ 10⁴×` larger and is a property of the task, not the instrument | `RUN results/v17k_r4_floor.jsonl` (six values equal to 16 digits); `READ V17_R4_RETAKE.md:259-286` |
| `warn_only` flag | changed no value: 6/6 flag-OFF cells (`results/r4_price_probe.json`) reproduce flag-ON bitwise | `==` on the float, not `allclose` | `READ V17_R4_RETAKE.md:225-238` |
| cost | `2.79 GPU-min` measured against `2.65` projected (+5.3 %); the ruling's `~2 GPU-h` was a `43×` over-book; peak `0.9739 GiB` = `12.2 %` of `7.996 GiB`; process wall `170.99 s`; CPU run this supersedes `1358.3 s` for 16 cells | `arm_smprime 16.161 s`/150 steps, `arm_pl 1.780`, `softmax 1.681` (means of 8) | `RUN` (`t="wall"`: `167.171 s`, `0.97388 GiB`); `READ V17_R4_RETAKE.md:292-313` |
| field coverage | CPU cell records carry 42 fields, CUDA 60; the 18 gate columns (`lambda_hat`, `unit_root`, `z_winding_*`, `frac_gate_annihilated`, `_0step` controls) and `instrument_hash` cannot be diffed across the device move | `RUN` (60 keys in the cuda cell record) | `READ V17_R4_RETAKE.md:174-179` |

### M.2.3 The round-15 extension: sixteen `arm_pl` cells and the paired contrast

| claim as filed | overturned / corrected figure | source |
|---|---|---|
| "no arm crosses BED-M's `floor₁`" (it.1) | **6 of 24 cells cross** at it.2; the it.1 citations were about other quantities | `READ V20_R15_JOURNAL.md:37` (C1) |
| "12 of 16" as the crossing headline (it.8) | the it.8 file's own `t="agg"` says `crosses: false`; pooled sd `0.2125` vs eight-cell `0.0203` — `10.5×`, all of it seed 9; the flip `n=9 → False`, `n=8 ex-seed-9 → True` is draw-invariant; the headline turns on an exclusion nobody has ruled | `READ V20_R15_JOURNAL.md:49` (C13) |
| "7 of 8 fresh vs `softmax` 0/8, `p = 6.730e-04`" | **unpaired** — softmax had never run past seed 7; the honest paired figure on the banked record is `5/8 vs 0/8, p = 0.012821` (a factor of 19); repaired by measurement at it.10: **8 of 9 `arm_pl` vs 0 of 9 `softmax`, paired, three draws**; gap between worst crossing `arm_pl` (`0.686874`) and best fresh `softmax` (`0.922856`) is `0.2360` with `floor₁` inside it; two fresh softmax cells read above `1.0` | `READ V20_R15_JOURNAL.md:50` (C14); `READ V20_R15_JOURNAL.md:1854-1873` |
| the sixteen `arm_pl` cells, deduplicated on `(kind, seed)` across `results/v17k_r4_retake.jsonl`, `results/v20_r15_it6_seeds8_15.jsonl`, `results/v20_r15_it8_armpl_b.jsonl` | 16 unique `arm_pl`: mean `0.778403`, sd `0.238706`, **12 of 16 below `floor₁`** (seeds 0,1,4,5,6,8,10,11,12,13,14,15; range `0.624869 … 0.680579`); the pooled verdict is `crosses: false` because four divergent cells drag the mean to `0.7784` before variance is considered | `RUN` (this session, 43 rows → 40 unique); `READ V20_R15_THEORY_TABLE.md:205-207` |
| `arm_smprime` seed 2 | crosses on all three eval draws: `0.203920 / 0.208055 / 0.216517`; worst reading `0.490590` below the floor; the only `arm_smprime` cell with `frac_gate_annihilated = 0.000000` and the only one that crosses; the other fifteen fail on all three draws (closest seed 8, `0.144954` above) | `READ V20_R15_JOURNAL.md:2113-2130` |
| every GPU-second and cost ratio quoted from it.3 onward (`0.3455 GPU-h`, `10.17×`, `41.9×`, the EXIT A table) | **struck**: `scripts/v15_r1.py` has no `torch.cuda.synchronize()`; `secs` is un-synchronised host wall clock whose strongest correlate is run order (`ρ = +0.7029, p = 0.0024`) above the gate correlation (`+0.5197`) | `READ V20_R15_JOURNAL.md:53` (C17); `READ V20_R15_THEORY_TABLE.md:99-102` |

**Caveat carried in the cell:** the eight new `arm_pl` cells (seeds 8–15) are uncontrolled
at the it.8 stage — softmax existed only for seeds 0–7 until the it.10 paired run
(`READ V20_R15_THEORY_TABLE.md:175,207`). The capped run at seeds 2, 3, 7 (`~6 GPU-s`)
that would settle Q3/W3 causality was priced three times and taken zero times
(`READ V20_R15_THEORY_TABLE.md:176`).

---

## M.3 The three corners, measured distinct

Design mechanism: **V-2** (a hand-built example where right and wrong coincide) and the
in-file refusal `corners_are_distinct` — a containment whose corners coincide is decoration
(`READ workdonenewseal.md:81`).

| corner | setting on `arm_smprime` | reference written from the theorem | result | source |
|---|---|---|---|---|
| c₁ softmax | `β=1, g≡0, QK-on` | `exp(w_ij) / Σ_{j'≤i} exp(w_ij')` | bitwise, `imag ≡ 0` bitwise | `READ V16_ARM_SMPRIME.md:268` |
| c₂ linear attention | `β=0, g≡0, QK-on` | `exp(w_ij)` (no normalizer) | bitwise, `imag ≡ 0` bitwise | `READ V16_ARM_SMPRIME.md:269` |
| c₃ path product | `β=0, QK-off` | explicit double loop, stated order (`k = i` down to `j+1`) | bitwise on cpu; cuda `11/64` at `≤ 5.551115e-17`, zeros identical | `READ V16_ARM_SMPRIME.md:270,308-317` |
| pairwise distances (float64) | — | — | `|c₁−c₂| = 4.472918`, `|c₁−c₃| = 1.144938`, `|c₂−c₃| = 5.335671` | `READ V16_ARM_SMPRIME.md:333` |
| switch effect sizes | `g`: `max|g=1 − g=0| = 0.673101`; `QK`: `max|qk=1 − qk=0| = 3.522037`; all three (`β, g, QK`) are `nn.Parameter`s on the shipped module | — | — | `READ V16_ARM_SMPRIME.md:29` |
| the `g ≡ 0` corner of the *original* v15 operator | row `i` sums to `i+1`, never `1`; smallest witness `i=1`, row `(1,1)`, sum `2`; normalized it is uniform `1/(i+1)`, which is attention only with constant QK logits — so `g ≡ 0` lands on **linear** attention, not softmax | Lean `gate_zero_row_sum`, `gate_zero_not_stochastic` (`lean/CEQ/V15.lean:221`) | refuted three ways | `READ workdonenew.md:54-74`; `READ README.md:198-201` |
| round-15 separation at trained settings | the "`≥ 0.30` on every pair" claim was withdrawn: the gate was drawn from W3's range and clamped, `99.109 %` at exactly `1.0`; re-drawn per cell `s=8` W1/W2 reads `2.623589e-01`; weakest reading still `2.6e11×` the tolerance, so `N ≥ 2` distinct primitives stands | — | corrected | `READ V20_R15_JOURNAL.md:55` (C19) |

The corners are theorems (`V16Domain.three_corners_containment`, `corners_are_distinct`,
`gate_zero_beta_zero_is_linear_attention`, `READ workdonenewseal.md:63-64`); the rows above
are their measured instances.

---

## M.4 The device certificate — RTX 4060 Laptop, sm_89

Design mechanism: **M-8** (pricing every arm at one arm's rate), **V-22** (a constant
certified under conditions the reading does not reproduce — the flag regime is journalled
on the header), **P-8** (an upper bound stated as a price — the R² gate refuses points that
paged over PCIe).

Box: `NVIDIA GeForce RTX 4060 Laptop GPU`, capability `8.9`, `8,585,216,000 B` total
(`7.996 GiB`), `torch 2.5.1+cu121`, Python `3.11.9`, `Windows-10-10.0.26200`,
`set_num_threads(8)`, `allow_tf32` matmul `False` / cudnn `True`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, run of record `2026-08-31 22:26:39`, git HEAD
`ab5b48547884e04258276e6e808d5a71ea65f917` (`RUN results/k_cert_local.json`;
`READ COSTS.md:55-63`).

### M.4.1 Throughput laws (forward + backward + `Adam.step`, `s=64`, `d_model=16`)

| arm | law | R² | fitted over | statistic | source |
|---|---|---|---|---|---|
| `softmax` | `s/step = exp(−12.18515596) · n^0.99625107` | `0.9999975` | `n ∈ {2048, 4096, 8192, 16384, 32768}` (5 points; `y = 0.010142 … 0.160665 s`) | median over 3 child processes of the median of ≥12 timed steps and ≥3.0 s of work, 2 warm-ups discarded | `RUN results/k_cert_local.json /throughput/laws/softmax` |
| `arm_smprime` | `s/step = exp(−10.01874582) · n^1.00258069` | `0.99999987` | `n ∈ {2048, 4096, 8192}` (3 points; `y = 0.093052 / 0.186518 / 0.373543 s`) | same | `RUN /throughput/laws/arm_smprime` |
| why three points | at `n = 16384` and `32768` the caching allocator reserves `10.578` and `13.969 GiB` from a `7.996 GiB` card and pages over PCIe; admitting them bends the exponent to `1.2341` at R² `0.976933` — "a throughput law fitted through swap is not a throughput law" | — | — | — | `READ COSTS.md:76-79`; `READ README.md:297-300` |
| the earlier CUDA law (V16 device round, `arm_phase`-era) | `s/step = exp(−11.9670) · n^0.9734`, R² `0.999384`; 40-iteration plan `68.35 GPU-h` at the 9,600-step ladder, `1.07 GPU-h` at the 150-step floor, against `815.88 / 12.75 CPU-h`; the decision `11.94×` at the point estimate, `29.0×` at the pessimistic end (CPU spread `2.55×`, CUDA `1.05×`); the contract's cited `13.6×` is the optimistic reading | `READ V16_DEVICE_CERT.md:474,804-806,856-864`; `READ workdonenewseal.md:277-282` |
| arm cost ratio at one length (`s=64` only, so **not a cost law**) | `arm_pl / softmax = 1.0589×`, range `0.987×–1.154×`; `arm_smprime ≈ 9.6×` softmax at 150 steps (`16.161 / 1.681`) | `READ V20_R15_THEORY_TABLE.md:189-190`; `DERIVED` from `READ V17_R4_RETAKE.md:299-301` |

### M.4.2 Memory model

| constant | `ceq/sizing.py` | re-solved on the card | R² | verdict | source |
|---|---|---|---|---|---|
| `C_RESIDUAL` (fp32) | 18 | `17.874` | `0.996373` | CONFIRMED, −0.7 % | `RUN /memory/law/c_residual_at_4B`; `READ COSTS.md:89` |
| `C_OPERATOR` (fp32) | 3.9 | `3.823` | `0.996373` | CONFIRMED, −2.0 % | `RUN /memory/law/c_operator_at_4B` |
| bf16-autocast operator B/element | 3.4 | `3.341` | `0.999830` | CONFIRMED, −1.7 % | `RUN /memory/law_bf16_autocast/c_operator_at_module_count` |
| bf16-autocast residual B/element | 2.2 | `2.383` | `0.999830` | **WRONG, +8.3 %, optimistic** (under-predicts); at the Q3 chunk shape `2.715` vs `2.738 GiB` (+0.85 %), `max_batch` 45 either way — no decision moves | `RUN /memory/law_bf16_autocast/c_residual_at_module_count`; `READ COSTS.md:92-100` |
| `measured/predicted`, real-valued arms | `0.950` for softmax and `arm_pl` across a 16× span in `n` | — | holds | `READ V16_DEVICE_CERT.md:279-286` |
| `measured/predicted`, complex arm (`arm_phase`) | `1.842` — the module under-predicts; `C_OPERATOR = 7.50` at 8 B/element, `4.29×` softmax's, not the `2×` a complex-is-two-floats argument gives; `C_RESIDUAL` NOT IDENTIFIED at `s=64` | — | breaks where it is new | `READ V16_DEVICE_CERT.md:281,318-334` |
| residency, `arm_smprime` | `n=2048`: alloc `0.814`, reserved `0.988 GiB`; `4096`: `1.597 / 1.908`; `8192`: `3.163 / 3.777`; `16384`: `6.296 / 7.504` (`10.578` under a training loop — NOT resident); `32768`: `12.559 / 13.951` — NOT resident; operator `complex64` at 8 B | — | — | `RUN /memory/cells` (peak bytes `874,290,688 … 13,484,888,576`); `READ COSTS.md:109-116` |
| the R2 cell at `n = 32768` for the phase arm | `7.893 GiB` against `6.939 GiB` free — REFUSED, not attempted; does not fit on CPU either; cap `22,106` at 20 % headroom, `27,633` at zero; last usable power of two `16,384`, where it is also cheaper (`5.29 h` vs `10.38 h`) | — | four contract cells affected | `READ V16_DEVICE_CERT.md:351-395`; `READ workdonenewseal.md:321-324` |
| the peak-activation formula `4·n·s·d·heads` | wrong tensor, `64×` low — the peak is the `[n,s,s]` operator; at batch 8192, `s=1024`: `512 MiB` claimed, `32,768 MiB` real | — | struck | `READ workdonenew.md:386` |
| "attention activation is `2·n·s²·heads` bytes bf16" | `6.63×` low (`3.9 × 3.4/2.0`); at `n=32, s=512, h=4`: `0.0671` vs `0.4449 GB/layer` | — | struck | `READ workdonenew.md:397` |

### M.4.3 Determinism

| finding | value | source |
|---|---|---|
| `cumsum` has no deterministic CUDA kernel in torch 2.5.1 and is the whole of the scan arms (`arm_pl`, `arm_phase`); `cumprod` **has** one, so `arm_smprime`'s forward runs under `use_deterministic_algorithms(True)` | forward `hop` and full forward bitwise, `max|Δ| = 0.0`, 8 repeats, `n=512`, reduction 64, both flag regimes | `RUN /determinism` (`flag_on.hop.bitwise = true`, `flag_on.forward.bitwise = true`) |
| the hole moved, not closed | `arm_smprime`'s **backward** raises under the strict flag: autograd differentiates `cumprod` with `cumsum` (`RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation`); gradient bitwise (`0.0`, 8 repeats) only with the flag off | `RUN /determinism/flag_on/gradient.executable = false`; `READ COSTS.md:149-154`; `READ README.md:305-314` |
| the hazard follows reduction length, not launch size | `cumsum(dim=-1)` bitwise at `[2048,64]`, at R2's `[32768,64]`, at `[262144,64]` (16.8 M elements), at `[512,4096]`; drifts only at a 1-D reduction of `1e6` (`6.82e-13` / `1.82e-12`); the arms reduce over 64 | `READ V16_DEVICE_CERT.md:625-645,658-659` |
| TF32 | asserted off and measured: enabling it moves the shipped arm's operator by `3.184e-04` relative (forward `3.064e-04`) against `3.237e-07` with it off on the bare product; two levels of the stack agree within 8 % | `READ V16_DEVICE_CERT.md:427-432` |
| run-to-run on the certified device | `δ_nrmse = 0.0` on 6 of 6 identical-seed pairs under `warn_only=True` (M.2.2) | `RUN results/v17k_r4_floor.jsonl` |
| cpu vs cuda `cumprod` | sequential on cpu, parallel scan on cuda: `11/64` entries move by `≤ 5.551115e-17`; `n_exactly_one` on the band splits `9767` cpu / `7713` cuda | `READ V16_ARM_SMPRIME.md:313-317` |

### M.4.4 Bar re-certification and the 0-step gate

| clause | worst δ (cpu vs cuda) | tolerance | δ/tol | source |
|---|---|---|---|---|
| `predict_the_mean` (`e3_t8`) | `8.580024779547557e-08` | `1e-6` | **8.58 %** against a 50 % HALT line, headroom `5.83×` | `RUN /bar/worst`; `READ COSTS.md:127-133` |
| `flipper_dependence` | `1.053e-07` | `0.05` | `2.1e-06` | `READ workdonenewseal.md:345` |
| `payload_only` | `9.130e-08` | margin `6.41e-02` | `1.4e-06` | `READ workdonenewseal.md:346` |
| `trained_two_feature` | `3.689e-08` | margin `9.86e-01` | `3.7e-08` | `READ workdonenewseal.md:347` |
| `oracle` | `0` exact on both devices, all four rungs | `1e-6` | `0` | `RUN /bar/cpu/e3_t*/cal/oracle = 0.0` |
| all four rungs × both devices | `BAR CALIBRATED`, HALT none; tolerances quoted from `bar_verdict`'s own body, text predating the node; 30 doubles bit-identical in IEEE-754 hex against the previous revision | — | — | `RUN /bar/verdict_cpu, /bar/verdict_cuda`; `READ workdonenewseal.md:350-352` |
| 0-step RED gate at R1′'s shapes (V16) | 32 cells (4 arms × 8 seeds), all PASS both devices; max `|cpu−cuda| = 1.678926e-07`; smallest margin `9.703933e-04` — `5,780×` the gap; binding cell `windowed_signed` seed 2 | `GATE_TOL = 1e-3` | — | `READ V16_BAR_RECERT.md:302-339` |
| 0-step RED gate in the K-cert run | 2 arms × 5 `n_train` × 3 seeds = 30 shapes, 30 reachable, 30 pass; worst margin `1.957e-03` = `1.96×` the tolerance | `GATE_TOL = 1e-3` | — | `READ COSTS.md:137-139` |

### M.4.5 The `5.8×` wall-clock gap — refuted

| step | measured | source |
|---|---|---|
| R1's softmax cell filed `71.32 s`/150 steps; the same arm, corpus, thread pin, steps and statistic re-read `12.3–13.6 s` | the `5.8×` does not reproduce | `READ V16_DEVICE_CERT.md:137-164`; `READ workdonenewseal.md:285-288` |
| eliminated by measurement | thread count (`0.99–1.26`), data source (`1.01×`), the arm (`1.06–1.11×` vs R1's `1.285`), the statistic (`1.012×`), thermal (wrong sign), dtype and shape (identical by construction) | `READ workdonenewseal.md:290-293` |
| what survives | i7-14700HX with 8 P + 12 E cores; `set_num_threads(8)` names a count, never a placement; E-cores alone `1.44×`, host contention alone `2.55×`, their cross `2.0×–4.0×` across six measurements; a `lake build` wrote `V15Source.olean` at `17:43:35` inside R1's softmax seed-3 window and the four cells after it are `13.1 %` cheaper | `41–83 %` of the gap in log terms; the residual is unexplained | `READ V16_DEVICE_CERT.md:174-211`; `READ workdonenewseal.md:295-304` |

---

## M.5 The calibration column — R11: 9 checked, 9 adverse, sign test

Design mechanism: **D-7** (a prediction filed without its counter, in a document whose
errors have a sign, `READ MISTAKES.md:2037`), **L-SIGN**, **M-2** (no numeric shrink factor
is fitted to nine rows).

| round | checked | adverse | signs | one-sided sign test | source |
|---|---|---|---|---|---|
| R11 | 9 (`V15_LEDGER.md:649-661`); `≥ 17` with the confirmations that table omits | 9 of 9 filed rows; 10 of `≥ 17` | 7 `+` (optimistic), 1 `−` (row 3: a sizing repair already made, reported as outstanding), 1 unsigned (row 8: two mutually inverse registrations) | `7/8`, `p = 0.0352` — direction established at `α = 0.05` | `READ V16_CALIBRATION.md:17-19,38-50,96-100` |
| the round entry's own "six wrong, all six optimistic" | the six honestly signed give `5/6`, `p = 0.1094` — **not** significant; "6/6" as filed gives `0.0156` but is not what the rows say | — | — | the headline was right about the direction and wrong about its evidence | `READ V16_CALIBRATION.md:98-105`; `READ workdonenewseal.md:434-436` |
| the one sub-census with a real denominator (`V15_CONTRACT_ARITHMETIC_AUDIT.md`, `A-0a…A-0g` confirmed, `A-1…A-4` findings) | 11 checked, 4 wrong, `36.4 %`, Wilson 95 % `[0.152, 0.646]` | — | — | `9/9` is a count, not a rate — the denominator was chosen after the audit and lists no confirmations | `READ V16_CALIBRATION.md:116-120` |
| size of the bias | Wilson 95 % on the optimism fraction at `7/8`: `[0.5291, 0.9776]` — excludes `0.5`, fixes nothing else; **no numeric shrink factor is authorised** (`D-CALIB-3`) | — | — | — | `READ V16_CALIBRATION.md:171-184` |
| the discount rule | `D-CALIB-1` the counter is the point estimate; `-2` a bare prediction is blocked, not discounted; `-3` sign, never size; `-4` the cheapest refutation of the optimistic half runs first; `-5` the row is appended whether or not it flatters | — | — | — | `READ V16_CALIBRATION.md:142-203` |
| R12 row | to be filed at it.38 — **NOT MEASURED** in the record read here | — | — | — | `READ V16_CALIBRATION.md:20` |

The nine rows, briefly (`READ V16_CALIBRATION.md:40-48`): (1) "`g == 0` gives bitwise
standard attention" — FALSE, row sums `i+1`; (2) "two-thirds of section-5 strikes were
`[V]`-as-theorem" — `4/40 = 10 %`, `6.67×` overstated; (3) sizing model "to be replaced" —
already calibrated (`−`); (4) "ceiling ≈ 38 from 22 %" — no live producer; (5) "TOST
retires to `N ≥ 23`" — power at `N = 23` is `0.0669`, `0.80` first at `N = 70`;
(6) `H = α + ½` shipped without `|d| < ½`; (7) `[RUN: 0.990]` best first-order recurrence —
matches neither bed (delay `R² = −0.000166/−0.000170`, power-law `0.604`/`0.755`);
(8) PART III vs PART IV mutually inverse registrations (M-20); (9) "carriers conserve mass
to `1e-12`" — false for the central carrier.

---

## M.6 The floors as measured

Design mechanism: **L-FLOOR** (an information floor beside every capability number),
**C15** of the corrections index (`floor₁` used as an information floor throughout,
`READ V20_R15_JOURNAL.md:51`), **V-10** (a gate satisfied by construction — the Hankel
`1/d` block is `I_d` with a zero row, and the node says so).

| floor | what it is | measured | source |
|---|---|---|---|
| `floor₁ = √((t*−1)/t*)` | a **one-hop capability threshold**, defined by `ĥ = t*(1 − NRMSE²) = 1`; not an information floor — a lower bound is never violated and this one is | **13 of 40** unique banked cells violate it (12 `arm_pl`, 1 `arm_smprime`); `s = 64` on 40/40; `dist_to_skyline` populated on 0/40; the published `6 of 34` is stale (it.8) | `RUN` this session over the three journals (43 rows, 40 unique on `(kind, seed)`, 13 with `eval_nrmse < 0.7071067811865476`, 13 with `eval_h_hat > 1.0`); `READ V20_R15_THEORY_TABLE.md:69-91` |
| `floor₁` at other `t*` | `t*=8`: `0.935414`; `t*=32`: `0.984251`; the nine pre-R11 N=8 cells never crossed any of them, `ĥ ≤ 0.389` | `READ workdonenew.md:220-233` |
| `floor₂ = √((t*−2)/t*)` | `0.866025` at `t*=8`; `pivot_unsigned` at `t*=8, n=32768` sits `0.110462` above it (`proven_hops = 0`) | `READ workdonenew.md:258-259` |
| **the exact oracle at `0.0`** — the real information floor on BED-M | `equilibrium_oracle` (`scale/negation_scope.py:286-304`) returns `z*_{s−1}` exactly; the bar clause `oracle` reads `0.0` on both devices, all rungs; distance from the banked wings to it is `0.203920` at best (`arm_smprime` seed 2) and `~0.85` modally; **no annex theorem predicts either wing's distance to it** | `RUN results/k_cert_local.json /bar/*/cal/oracle`; `READ V20_R15_THEORY_TABLE.md:87-91,197` |
| candidate floors that do not bind either frozen wing | M11 Fano and M12 rate–distortion: 0 of 3 beds, 0 producing `.py` (prose); M16 Hankel: BED-K only, both wings are BED-M | `READ V20_R15_THEORY_TABLE.md:87-90,199` |
| **the Hankel `1/d` law** (BED-K, delay `d`) | `R²₁ = 1/d` exact on `d ∈ {5,10,20}`: max−min of `R²₁·d = 0.000e+00`; `err_₁ = √(1 − R²₁) = 0.9746794345` at `d = 20`, bound to `5e-11`; domain hole at `d = 0` (`delay_zero_is_first_order`); the block is `I_d` with a zero row — "arithmetic, not evidence"; the AAK/Glover attribution was refused (correct source: Eckart–Young–Mirsky) | `READ V20_R15_IT6_JUPITER.md:450-465`; `READ V20_R15_THEORY_TABLE.md:149-152` |
| the thread-count floor | `2.345e-3` NRMSE drift across thread counts on CPU; `Δ_eq = 0.5σ_seed = 0.005051` at `t*=2, n=2048`, so the floor is `0.464` of the margin — any equivalence margin below `~4.7e-3` is indefensible | `READ workdonenew.md:391` |
| the TOST parity floor | 90 % CI half-width `0.8807σ` at N=8 against a `0.5σ` margin; CI first fits at N=23 (power `0.0669`); power `0.80` first at **N=70** (`8.8×` the registered count); two bit-identical arms return NO VERDICT at N=8 | `READ workdonenew.md:400`; `READ workdonenewseal.md:31` |
| the N=8 sign floor | `0.0078125` (`2⁻⁷`); the X₂₆ lead-time CI `[−13.88, +1.50]` includes zero, so the item was struck | `READ workdonenew.md:388` |

---

## M.7 BED-K and BED-1 — measured facts

Design mechanism: **V-11** (a precondition satisfied at every real draw — BED-K's delay bed
is the first corpus whose label carries a delayed cause), **M-19** (an invariant estimated
on a collapsed orbit — the Pesin arithmetic is stated), **V-3** (an identity checked against
itself — the committor is cross-checked by Kirchhoff and by the dynamics), **D-2** (the
oracle must not be the arm's own resolvent — BED-1's committor runs on the latent chain).

### M.7.1 BED-K (`ceq/beds/bed_k.py`; 15 tests)

| fact | measured | control | source |
|---|---|---|---|
| delay bed is scan-blind | best first-order fit using the label's own true previous value: `α = −0.0109, β = −0.0216, R² = −0.000170` (`−0.000166` in the sibling run; two runs, not a discrepancy); 30-seed sweep at `d=4, n=2048`: `R² ∈ [−0.0027, 0.0028]` | the **same fitting code** recovers AR(1) at `α = 0.8000, β = 1.0000, R² = 1.000000` | `READ V15_N4_BEDK.md:194-199`; `READ V16_CALIBRATION.md:228-230` |
| delay bed is attention-reachable | hand-set head reproduces the label over `i ≥ d` to `8.67e-19` | bar `≤ 1e-12` | `READ V15_N4_BEDK.md:204,291-293` |
| **power-law bed is NOT scan-blind** | first-order fit `R² = 0.604` at `H = 0.75`, `0.755` at `H = 0.9` | the Lean result `first_order_cannot_powerlaw` is an exact-identity theorem, not an approximation bound — a recurrence that cannot reproduce a kernel exactly may still fit 60 % of its variance, and does (debt D-APPROX) | `READ workdonenewseal.md:189,194-196,510` |
| Jacobian oracle | delay max err `2.88e-11`, power-law `4.88e-10` | both `≤ 1e-9` | `READ V15_N4_BEDK.md:211` |
| Hurst estimator (DFA, not R/S) | white noise `≈ 0.478`, AR(0.5) `≈ 0.486`; R/S's biased reading is `0.75`; recovery at `H_true = 0.75` reads `0.8121` (bias `+0.062`), `0.812 → 0.818` after the FARIMA correction — it did not shrink the bias, reported as such | — | `READ V15_N4_BEDK.md:223-258`; `READ workdonenewseal.md:191-192` |
| scan-blindness by theorem | `V15Kernel.first_order_cannot_delay` (arbitrary `f`: zero drive `0 = f(0)(0)`, impulse `1 = f(0)(0)`), `delay_forces_state_injective`, `linear_first_order_cannot_delay_beyond_state_dim`, `first_order_cannot_powerlaw`; the trivial reading `delay_realizable_at_dimension_d` refused in-file | — | `READ workdonenewseal.md:68,77-78` |

### M.7.2 BED-1 (`ceq/beds/bed_1.py`; 13 tests, 28 in the beds suite)

| fact | measured | control | source |
|---|---|---|---|
| committor is harmonic | `max|Lq| = 0.000000e+00` at `jitter = 0.0` (interior `q ∈ {0, ¼, ½, ¾, 1}`); `1.040834e-17` at `jitter = 0.05`, seed 11 (`q(C) = 0.3356422966`, `q(S_ac) = 0.1678211483`, `q(S_cb) = 0.6678211483`) | must-fire perturbation drives it to `1.000000e-06` — 11 orders (ratio `9.608e+10`) | `READ V15_BED1.md:123-124`; `READ workdonenewseal.md:214-215` |
| barrier-height vs committor labels disagree | `T* = ΔΔE‡/ln m = 0.62133493455961186`; at `0.9 T*` both say `lo`; at `1.1 T*` barrier says `lo`, committor says **`hi`** (rates `1.672502e-01 / 2.315116e-01` low-barrier, `1.398632e-01 / 2.679881e-01` for the `m`-fold bundle) | bisected crossover from committor flux `0.62133493455961175`, `|diff| = 1.110e-16`; third route from the dynamics: empirical `lo:hi = 0.86679 < 1` at `1.1 T*` (232,957 / 268,759 / 111,354 events), agreeing with the committor ratio to `0.34 %` | `READ V15_BED1.md:42,175-177,349`; `READ workdonenewseal.md:199-212` |
| the `a*` lowest-barrier-exit claim | FALSE on legal instances: one saddle at `E = 2.00` to B, four parallel at `E = 2.20` to C gives `q_C/q_B = 4e^{−0.2/T}` exact to `2.351e-15`; at `T = 0.50` the higher-barrier channel is `2.68×` faster (`q_B = 0.271644632`, `q_C = 0.728355368`); crossover `T* = 0.144269504088897` vs predicted `…896` (agreement `6.384e-16`); ranking flips back at `T = 0.10` (`q_B = 0.648786`) | independently reproduced in-session at the time | `READ workdonenew.md:390` |
| Chapman–Kolmogorov | `τ = 1` fails (`0.22702`), `τ = 64` passes (`0.00400`), factor 57 | both vacuity ends guarded at the passing lag: `‖T̂(64) − I‖ = 0.9482 > 0.5`, `‖T̂(128) − Π‖ = 0.1280 > 0.05` | `READ V15_BED1.md:196-211` |
| Pesin deficit (`T = 0.25`, `h = 0.081275` nats/step, exact) | generating (state) partition, 3 symbols: entropy `0.081042`, deficit **`0.000233`** (`0.29 %` of `h`); wrong guards `0.029889 / 0.030616 / 0.029915` — factor 128 | the contract's `q = ½` guard partition (8 symbols) has its own deficit `0.009654` = `11.9 %` of `h` — **not** near zero, so the contract's guards are not a generating partition; guards were not selected by an argmin over the deficit | `READ V15_BED1.md:248-272` |
| conservation census | 7 conserve (committor harmonicity `0.000000e+00`, reactive flux divergence-free `0.000000e+00`, both at `1e-12`), **2 do not**: guard-crossing balance (21 net on 65,481 events, `0.08 %`) and trap-channel share (`0.11005` vs `0.05837`, off by `1.885×`, because a channel with a metastable interior recrosses its own guard) | printed with mechanism | `READ V15_BED1.md:310-326` |
| Morse census (lower-link criterion on the 1-complex, 11 nodes, 16 edges) | `m₀ = 3, m₁ = 8, χ = V − E = −5 = m₀ − m₁, b₀ = 1, b₁ = 6`; Morse inequalities hold | must-fire `V(C) → 2.5` moves the classifier to `(2, 7)` with 2 regular points and leaves `χ = −5` unchanged | `READ V15_BED1.md:359-378` |
| mutation check | five mutants killed on the stated ground that 13/13 green on a first implementation run is not evidence | — | `READ V15_BED1.md:99`; `READ workdonenewseal.md:234-235` |
| what BED-1 has **not** produced | no arm has been trained on it; `C-TS` reads "BED BUILT, NOT RUN"; the state-metric cells Q6/W1 and Q6/W3 are F4 because the frozen wings return a scalar `[n]` and 0 of 40 banked cells journal a distributional object | — | `READ workdonenewseal.md:33`; `READ V20_R15_THEORY_TABLE.md:211-225` |

---

## M.8 X₃₅′ and X₃₇ — instrument facts

Design mechanism: **V-10** (the flatness gate has a demonstrated rejection region),
**M-2** (the persistence threshold was frozen before the run), **V-16** (the Z-winding
guard refuses aliased input rather than returning a plausible wrong integer), **M-15**
(every order-parameter reading is printed beside its null).

### M.8.1 X₃₅′ — hidden-cause detection (`tests/x35`, 14 tests; `tests/x35p`, 29 tests)

| instrument | measured | control / must-fire | source |
|---|---|---|---|
| Shewhart onset on the residual `r = z_obs − z_model(visible)` | onset `36` at `sd = 0` and `sd = 0.05`, true `36`, `mag_hat = 0.9809` (`m_true = 1.0`); 500 seeds at `sd = 0.05`: offsets `{0: 421, 1: 74, 2: 5}`, misses 0, **`0.9900` within ±1** against a pre-registered `0.95` (analytic prediction `~0.976`, bar not recomputed from the realised figure) | superposition, 200 seeds, sources at 36 and 80: first onset within ±1 of 36 at `0.9900` — the detector can only ever call the earliest source | `READ V15_X35A_RESIDUAL.md:144-146,184-186,205-213` |
| scope control — no plant ⇒ no onset | measured FAR `0.0120` (24/2000) vs calibrated `α = 0.01`; calibrated on `[0, 2000)`, scored out of sample on `[1e6, 1e6+2000)`; in-sample `0.0095` printed beside | truncated model at lag 64: FAR **`0.2667`** (`27×α`, corr `+0.8988`); lag 8: `1.0000` (`100×α`) — the flatness gate has a rejection region | `READ V15_X35A_RESIDUAL.md:147-151,234,273-274` |
| exact source solve through `(I − A)` | `8.882e-16`, two planted sources — the delta's cited `8.9e-16` | Lean `source_is_first_order_difference`, `two_sources_recovered` | `READ workdonenewseal.md:67,244` |
| exact inverse vs adjoint (`Wᵀr`) | crossover at `sd = 0.15`; by `sd = 0.50` it is 9 vs 266 with 67 exact misses against 5 — the exact solve is **retired to the noiseless regime** | — | `READ V15_X35P_SOURCE.md:44,380,490`; `READ workdonenewseal.md:245` |
| Kramers–Kronig causality residual | planted anticipating kernel `a√2 = 0.7 × 1.41421356 = 0.98994949366116647`; closed form `√2·‖anticausal‖₂`, independent route agrees to `1.11e-16`; no threshold, no tuning knob | null: 256 random 24-tap causal kernels, worst `1.3670e-15` (float64 FFT round-off); **margin `7.24e14`** | `READ V15_X35P_KK_CRB.md:61-67,81-90` |
| why the original KK probe read `0.000` | all four candidate mechanisms run: origin-forgetting `1.657e-16` (zero on every input), residual on `|H|` `3.269e-16` (zero by construction), no zero-padding `0.000000e+00` (conditionally), `1/M` normalisation eliminated (shrinks as `1/√M`; an exact zero would need `M > 1.3e7`) | — | `READ V15_X35P_KK_CRB.md:128-189`; `READ workdonenewseal.md:251-255` |
| Cramér–Rao floor on the onset | **DOES NOT EXIST** — the onset is an integer; embedding with the jump abrupt gives `I = 0` and a bound of `+∞`; smoothing makes the bound a property of the smoothing: `1/I` spans `0.18 … 443` (`2462×`) over ramp widths, and `1.08e7×` from sub-sample phase alone | — | `READ V15_X35P_KK_CRB.md:273-342,578` |
| Ziv–Zakai bound (ships instead) | calibrated exactly: `0.00e+00` difference at `n = 32/64/128` (`85.25 / 341.25 / 1365.25`); the Bayes-optimal detector tracks it across a `4096×` SNR span, ratio `[0.998, 1.329]` | — | `READ V15_X35P_KK_CRB.md:393-395,417-431` |
| prior-art status | X₃₅ is Basseville & Nikiforov 1993 §7.2.4 in closed form, equation for equation; the learned-model form is arXiv:2604.25655 Thm 3.1 — **occupied**, not novel | — | `READ workdonenewseal.md:463-464`; `READ README.md:216-220` |

### M.8.2 X₃₇ — topological certificates (`ceq/certs/topological.py`; `tests/certs`, 36 tests, 8 mutants killed)

| certificate | measured | control / must-fire | source |
|---|---|---|---|
| `Z` winding, per instance | correct on `k ∈ {−3, −1, 0, 1, 2, 5, 15}`, worst departure from an integer `4.4e-16` (at `k = 2`) against `INTEGRALITY_TOL = 1e-9` | refuses aliased input where the unguarded path (`max_step = π`) returns a plausible but wrong `−2` against true `k = 3`; the step guard is necessary and **not sufficient** — 63 turns in 64 samples has a wrapped step of `0.031π`, the healthiest reading in the file, and returns `−1`; a refinement check was added | `READ V15_X37_CERTS.md:155-169,184`; `READ workdonenewseal.md:261-262` |
| persistent `β₁` of a carrier trajectory (Rips) | RPS cyclic dominance: one `H₁` bar `(0.045414, 0.551497)`, persistence `0.506083`, count `1`, `True`; threshold frozen before the run (M-2) | coordination control: no finite `H₁` bar at any radius (`[RUN: 1 vs 0]`) | `READ V15_X37_CERTS.md:229-264,252` |
| stability of the persistence reading | `1.26e-3 / 1.07e-2 / 5.48e-2` against the `2ε` bound `2e-3 / 2e-2 / 1e-1` (ratios `0.534, 0.548`) | — | `READ V15_X37_CERTS.md:291-292`; `READ workdonenewseal.md:264` |
| Euler–Poincaré vs BED-1's Morse census | `Σ(−1)^k β_k = 1 − 6 = −5` = `m₀ − m₁ = −5` on both landscapes (default `(3,8)`, must-fire `(2,7)`) — AGREE | **half of it cannot fail**: on a 1-complex `β₀ − β₁ = V − E` identically; only the Poincaré–Hopf half carries content; on a Hopf normal form with a complete census `Σι = 1 = χ(D²)` while the Rips carrier reads `Σ(−1)^k β_k = 0` — the cross-check fires falsely where it can | `READ V15_X37_CERTS.md:362-366,425-436`; `READ workdonenewseal.md:265-266` |
| Kuramoto order parameter `r = |N⁻¹ Σ e^{iθ_j}|` | wrapped normal `σ = 0.02` (`N = 65536`): `0.999800` vs `exp(−σ²/2) = 0.999800` (`2.24e-07`); `σ = 0.05`: `0.998743` vs `0.998751`; `σ = 0.20`: `0.980110` vs `0.980199`; `N = 4096` equispaced → `5.55e-17` | `N = 4096` uniform, 8 seeds: `0.007794 – 0.022456`, mean `0.012916` vs the Rayleigh prediction `E[r] = √π/(2√N) = 0.013847` | `READ V15_X37_CERTS.md:478-491` |
| `bed_1.morse_census` returns `betti_0` and `betti_1` as constants | recorded as a weakness of the census, not of BED-1's answer | — | `READ V15_X37_CERTS.md:443` |
| the S² Rips corpus (`ceq/rips.py`) | exists in the tree (mujoco#3396); **NOT MEASURED** here — no journal row for it was located in the files this section reads | — | `READ BRIEF.md:195` |

---

## M.9 Test state (the invariant is the failure set, not the pass count)

| suite | state | source |
|---|---|---|
| `tests/loop` | 15 failed / 523 passed — the same 15 by name across five snapshots; the pass count moved `501 → 515 → 517 → 518 → 522 → 523` because two repo tests parametrize over every module | `READ workdonenewseal.md:487,498-502` |
| `tests/arm_smprime` | 35 passed CPU, 35 passed CUDA, same ids | `READ V16_ARM_SMPRIME.md:35,74-105` |
| `tests/arm_phase` / `tests/arm_pl` | ~~30 / 30 (CPU / CUDA); 17~~ **SUPERSEDED** (old cell kept: a count that moved is evidence). CPU at `03adf7f` + the V-29 `arm_phase` repair: **37 collected, 36 passed / 1 failed**, `python -m pytest tests/arm_phase -q`, `WIN-16QAL06O9GB`, python 3.11.9, torch 2.14.0+cpu, exit 1. Two separate moves: `+7` from the new gradient guard `test_the_shipped_init_has_a_live_gradient_everywhere.py` (all 7 pass), and `-1` from `test_bind2_the_band_path_product_has_modulus_one_over_1e4_phases`, which **fails at `03adf7f` unmodified** (29 passed / 1 failed with the repair stashed) and is therefore stale in the `30` above independently of this round. CUDA and `tests/arm_pl` not re-run — this box has no cuda | `READ workdonenewseal.md:489-490`; `READ MISTAKES.md` V-29 |
| `tests/beds` (BED-K 15 + BED-1 13), `tests/certs`, `tests/x35`, `tests/x35p`, `tests/mars_v15` | 28; 36 (8 mutants killed); 14; 29; 21 passed / 4 skipped (every skip must-fired) | `READ workdonenewseal.md:491-495` |
| Lean | 12 files, `lake build` exit 0 at `[1530/1531]`, 0 `sorry`, 0 `sorryAx`; every theorem through `#print axioms`: `[propext, Classical.choice, Quot.sound]` only; 169 `theorem`/`lemma` declarations (coordinator's grep, 2026-09-03) | `READ workdonenewseal.md:41-45`; `READ BRIEF.md:125-128` |

These counts are as of `91b862d` (2026-08-31, `READ workdonenewseal.md:6`) and were **not
re-run this session**; the paper should re-run `python -m pytest tests/loop -q` at its
assembly HEAD before quoting them.

---

## M.10 Numbers the paper must NOT use

### M.10.1 Struck constants — the registry (`STRUCK.md`, rendered from `tests/loop/test_no_struck_constant_ships.py:47` at `aa82df7`, 12 entries; `READ STRUCK.md:18-33`)

| constant | why it is struck | what to write instead |
|---|---|---|
| `−1.389` (R² `0.9938`) | M2 decay exponent, a `floor = 1e-6` artifact; least squares on `floor = 0` gives `−0.958` (R² `0.9990`) while the forcing audit reports `−1.221` (R² `0.9662`) — they disagree | **delete the claim**; no replacement is published |
| `−0.5173`, `−0.4160`, `−0.4654` | "live rows only" K1 slope interval and point — no producer exists in any `.py/.json/.jsonl/.txt` | the as-computed slope `−0.4137 [−0.4579, −0.3704]`, which reproduces from `scale/arm_a_k1.py` |
| `−1.826` | M2 slope as first reported; the shipped operator reads `−1.298` | `−1.298` (`READ RESEARCH.md`; brief §3) |
| `1.471448`, `1.343174` | M5 tail norms at `s = 128 / 512` — FABRICATED; a 1,800-setting sweep produced neither | measured `0.880500` (hops=2) / `0.882030` (hops=4) at 128; `1.292741` at 512 |
| `0.743864` (CI `0.656532 … 0.816955`) | U1/N3 pilot Spearman ρ — NO PRODUCER HAS EVER EXISTED; nine named functions never defined in any commit | nothing; the item is void |
| `5.4944e-13` | Karcher residual in float64 — asserted `[RUN]` from a throwaway script whose own output was `nan` | `7.481e-09 / 8.155e-09 / 8.405e-09` at published settings; `7.307e-13 / 7.958e-13 / 8.405e-13` at `--tol 1e-15 --steps 400` |

The test scans the shipped code and the lead documents for every entry; a struck constant
that reappears fails `tests/loop/test_no_struck_constant_ships.py`
(`READ STRUCK.md:6-10`).

### M.10.2 Stale traps the record names (each carries the corrected figure above)

| do not write | because | write instead | where corrected |
|---|---|---|---|
| "no arm crosses `floor₁`" | 6 of 24 crossed at it.2; 13 of 40 at it.14 | M.6 row 1 | `V20_R15_JOURNAL.md:37` (C1), `:51` (C15) |
| "6 of 34 cells violate `floor₁`" | stale it.8 count | `13 of 40` (`RUN`) | `V20_R15_THEORY_TABLE.md:80-86` |
| "12 of 16 cross" as a verdict | the `t="agg"` record says `crosses: false`; the flip turns on seed 9 | "12 of 16 cells sit below `floor₁`; the pooled verdict is `False`" | `V20_R15_JOURNAL.md:49` (C13) |
| `p = 6.730e-04` | unpaired | paired `p = 0.012821` on the banked record; `8/9 vs 0/9` on three draws at it.10 | `V20_R15_JOURNAL.md:50` (C14) |
| any GPU-second or cost ratio from `scripts/v15_r1.py`'s `secs` (`0.3455 GPU-h`, `10.17×`, `41.9×`, `2.85 / 118.8`, the EXIT A table) | un-synchronised host clock, run-order confound `ρ = +0.7029` | the K-cert laws (M.4.1), which use a separate timing statistic; and `arm_pl / softmax = 1.0589×` only as a ratio at `s = 64` | `V20_R15_JOURNAL.md:53` (C17) |
| "the `5.8×` wall-clock gap" / softmax `71.32 s` as a per-step cost | does not reproduce; `12.3–13.6 s` on a quiet host | M.4.5 | `V16_DEVICE_CERT.md:132-265` |
| "`13.6×` CPU→GPU decision" | the optimistic reading | `11.94×` point, `29.0×` pessimistic | `V16_DEVICE_CERT.md:804-806` |
| `Σ_j |W_ij| = 1.000000` as the softmax-corner certificate | equals `Z_i^{1−β}`, blind to `g` | `Σ_j Re W_ij` (`2.220446e-16`) | `README.md:261-275` |
| `label_cell = 1.440495` on BED-M | scored slot 0, a term no setting of the arm can move | `2.965913655728132e-16` | `README.md:244-259` |
| "a pole precisely on the unit circle" (`a_hat_max == 1.0`) | `torch.clamp(u, 0, 1)` at `ceq/arm_smprime.py:113` — a ceiling, not a converged pole; 6 of 16 cells are there at step 0; pre-clamp `u` exceeds `1.0` on 6 of 6 zero-step cells (margins `0.0129–0.1843`) and the driver is unknown | — | `V20_R15_JOURNAL.md:45` (C9) |
| "live-band decay under 3 % per position" | up to `4.76 %` (`−0.0476 … −0.0010`) | — | `V20_R15_JOURNAL.md:46` (C10) |
| `1.084523` (Q2/W3 excess) | a prose constant formed by arithmetic on another prose constant | `1.0845223424`, bound to `5e-10` | `V20_R15_THEORY_TABLE.md:158` |
| "`lambda_hat` as a magnitude" | mean of `log m` over every position — one `m_k = 0` sends it to `−inf`; it carries one bit | `lambda_hat_live` | `V20_R15_THEORY_TABLE.md:166` |
| "`N = 1` primitive, not 3" | struck as UNBOUND; re-bound at it.4 as two arena entries, one primitive | — | `V20_R15_JOURNAL.md:54` (C18) |
| "M14 at F1, HOW-BAD 108×" | M14 is `F4` (RULING J-17d); the `F1` was a scoreboard line | — | `V20_R15_JOURNAL.md:56` (C20) |
| "every pair separates by `≥ 0.30`" | withdrawn; `2.623589e-01` at `s = 8` W1/W2 | "weakest reading `2.6e11×` the tolerance" | `V20_R15_JOURNAL.md:55` (C19) |
| the `0.4899` K=8 common-mode share; "common-mode swamping" | does not reproduce (`0.7535 / 0.6517 / 0.6532`); the `0.7549` figure has `z = −0.30` against its null | withdrawn | `workdonenew.md:407,419` |
| "the label is `N(0, t*+1)`" | `N(0, t*)`; `b[s−1] = 0` kills the `m = 0` term; measured `Var(y) = 2.002719 / 7.984942 / 32.069818` | `N(0, t*)` | `workdonenew.md:381` |
| "`err(t) ≤ λ₂ᵗ`" in `l_∞` | violated 6 of 12, worst `1.290631` | true and tight in the degree-weighted 2-norm (12/12, worst slack `7.00e-04`) | `workdonenew.md:382` |
| "C-F fails 12/12" | `δ = 0.5` imported from another task's clause; ten of twelve cannot reach it by construction | `1/12` | `workdonenew.md:384` |
| "`ρ_P = √2` for a planted antisymmetric A" | `2.000000` exactly; `√2` is the random-matrix value (`1.411720` at `n = 256`) | `2` | `workdonenew.md:412` |
| "the corpus covers the λ₂ band `[0.90, 0.95]`" | 12 accepted rungs read `0.9435 … 0.9499`, `12.9 %` of the band | — | `workdonenew.md:396` |
| "the arms return a state distribution" | 0 of 40 banked cells journal one; both wings return `[n]` scalars; the chess witness is NOT a registered bed (`kdata.BED_SPECS = ['bed_m','bed_k','bed_1']`) | — | `V20_R15_THEORY_TABLE.md:213-224` |
| "S² Rips corpus measured", "Mapper cover measured", "isocommittor surfaces measured" | no journal row located in the files this section reads | `NOT MEASURED — needs <instrument>` | this section, M.8.2 last row |
| "beats softmax" on any BED-M cell | the deciding measurement's CI straddles; softmax is a fellow approximator on a bed that asks for a selective scan (`R-SKY`) | the paired contrast with its resolution statement (M.2) | `workdonenew.md:287-293`; `CEQ_V16_CONTRACT.md` |

---

## M.11 Mechanism map — what each table is designed against

| table | `MISTAKES.md` mechanism (`READ` line) | how the table discharges it |
|---|---|---|
| M.1 binds | V-24 (`:1658`), V-25 (`:1954`), V-3 (`:72`) | every bind row carries a planted negative at O(1) and is measured on BED-M's real support; the two routes of the resolvent check share no code |
| M.2 R1 | M-10 (`:1072`), M-16 (`:1346`), V-26 (`:2190`), P-3 (`:316`), M-6 (`:518`) | thread count and device are part of every cell's identity; the bimodal split is printed beside the mean; the CPU journal is superseded by an append-only marker; the N=8 adjudicator refuses below N=8 |
| M.3 corners | V-2 (`:58`) | distinctness is a measured distance, not a hand-built coincidence |
| M.4 device | M-8 (`:544`), V-22 (`:1140`), P-8 (`:387`) | one law per arm with its R²; the flag regime is journalled on the header; the R² gate refuses paged points |
| M.5 calibration | D-7 (`:2037`), M-2 (`:451`), P-1 (`:289`) | every prediction carries a counter; no shrink factor is fitted; every row cites its producing file |
| M.6 floors | C15, V-10 (`:168`), M-13 (`:1210`) | the capability threshold is named as such; the exact oracle at `0.0` is the floor; the equivalence margin is checked against the thread floor |
| M.7 beds | V-11 (`:178`), M-19 (`:1564`), D-2 (`:710`) | a label with a delayed cause; the Pesin arithmetic is stated; the committor is cross-checked by three routes |
| M.8 instruments | V-10 (`:168`), M-2 (`:451`), V-16 (`:828`), M-15 (`:1289`) | rejection regions demonstrated; thresholds frozen before data; the guard refuses rather than returns; every reading beside its null |
| M.10 struck | P-1 (`:289`), P-3 (`:316`), P-6 (`:364`) | the registry test fails on reappearance; every stale claim points at its correction row |

---

## M.12 Limits of this inventory

- Journal parses this session covered `results/v15_r1.jsonl`, `results/v17k_r4_retake.jsonl`,
  `results/v17k_r4_floor.jsonl`, `results/v20_r15_it6_seeds8_15.jsonl`,
  `results/v20_r15_it8_armpl_b.jsonl` and `results/k_cert_local.json`. Every other number
  is `READ` from a signed report and was **not** re-executed; the bed and instrument
  numbers (M.7, M.8) in particular rest on `V15_*.md` reports at `91b862d`.
- The `13 of 40` recount reproduces the theory table's figure exactly, including the three
  exact re-emissions (43 rows → 40 unique), but the dedupe key `(kind, seed)` is this
  session's reading of the table's recipe, not the table's own script.
- `RUN` on the throughput laws reproduces `COSTS.md` to the printed digits; the memory-law
  constants reproduce; the `arm_phase` complex-arm constants (`1.842`, `7.50`) are `READ`
  from `V16_DEVICE_CERT.md` and have no entry in `results/k_cert_local.json`.
- The test-state counts (M.9) are as of `91b862d` and were not re-run.
- No `CITED` rows: this section deliberately carries no external source; prior-art
  occupancy is another planet's table.
- `house-events.jsonl` was not opened.
