# V20 R15 it.8 — MERCURY (LESTRADE). TWO EXPERIMENTS RUN, ONE REFUSED, ONE STORY KILLED.

**Experiment A ran and is the round's symmetric `arm_pl` batch.
Experiment C is answered at ZERO GPU-s and the answer is the clamp.
Experiment B was NOT run: JUPITER's cap is not specified precisely enough to
execute, and this office declines to invent one. §3 prices the clarification.**

**Every number below carries the command that produced it.** No git write.
Nothing touched Kaggle — the run is local, on this box's own 4060.
**This office does not adjudicate any prediction.** VENUS's §2.2 filing and her
it.7 pre-registration are scored at it.29 and that row is hers.

---

## 0. THE COMMANDS, VERBATIM

**Experiment A, second attempt — the one that produced cells `[RUN]`:**

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u scripts/v15_r1.py \
  --device cuda --arms arm_pl \
  --seeds 0 8 9 10 11 12 13 14 15 \
  --tag v20_r15_it8_armpl_b            > /tmp/expA.log 2>&1 ; echo "REAL_EXIT=$?"
```

→ `REAL_EXIT=0`, 64 lines of stdout, `WROTE results/v20_r15_it8_armpl_b.jsonl`,
`=== WALL CLOCK: 19.6 s ===`.

**The first attempt aborted, and the abort is reported rather than hidden.**
The same invocation under `--tag v20_r15_it8_armpl_seeds8_15`, piped through
`| tail -40` inside a backgrounded shell, printed four lines (through
`=== 1. THE BIND …`) and stopped. `results/v20_r15_it8_armpl_seeds8_15.jsonl`
exists with **2 lines — one `header`, one `identity`, zero cells** `[READ]`, and
is left in place as the record of the abort. **The `EXIT=0` that shell reported
was `tail`'s exit code, not the instrument's** — `$?` after a pipeline is the
last stage's. That is this office's own harness defect, not the instrument's:
the identical command in the foreground, writing to a file, completed in
`19.589 s`. **A price that reads "0 cells in 90 s" is a pipeline artifact and
was diagnosed as one before a second GPU-second was spent.**

`scripts/v15_r1.py` **was not modified.** `--seeds` (`:547`), `--arms` (`:552`),
`--tag` (`:558`) are the shipped flags; `--n-train 2048`, `--n-eval 4096`,
`--steps 150` are the defaults at `:548-550` and are the retake's values.
**No cap flag exists** — see §3.

---

## 1. EXPERIMENT A — THE SYMMETRIC `arm_pl` RUN

### 1.1 The regime diff, field for field

`[RUN]` a 22-key diff of `results/v20_r15_it8_armpl_b.jsonl:1` against
`results/v17k_r4_retake.jsonl:1`:

```
HEADER FIELD COUNT new=22 retake=22
--- DIFF ---
 arms      new=['arm_pl']                      retake=['arm_pl','arm_smprime','softmax']
 seeds     new=[0,8,9,10,11,12,13,14,15]       retake=[0,1,2,3,4,5,6,7]
 tag       new='v20_r15_it8_armpl_b'           retake='v17k_r4_retake'
 when      new='2026-09-02 02:57:44'           retake='2026-08-31 23:08:44'
```

**Four fields differ. `arms` and `seeds` are what the experiment varies;
`tag` and `when` are stamps. The remaining EIGHTEEN are identical**, the
`instrument_hash 5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309`
included, along with `device`, `deterministic_algorithms`,
`deterministic_warn_only`, `cublas_workspace_config`, `threads`, `torch`,
`floor_1`, `steps`, `n_train`, `n_eval`, `lr`, `d_model`, `task`, `t_star`,
`s`, `d`.

**CORRECTION TO it.6, this office's own.** `V20_R15_IT6_MERCURY.md` §1 said the
regime test "diffs seventeen fields". **The header carries 22 keys and the
invariant set is 18.** The it.6 test node was correct; the prose count was one
short. `test_regime_matches_the_retake_field_for_field` now asserts
`len(set(new) - VARY) == 18` explicitly.

### 1.2 The nine cells

`arm_pl`, `t*=2`, `s=64`, `d=24`, `n_train=2048`, `n_eval=4096`, `steps=150`,
`lr=0.02`, `d_model=16`. Columns read off the cell records.

| seed | `eval_nrmse` | `frac_gate_annihilated` | `frac_..._0step` | `a_hat_min` | `a_hat_max` | `lambda_hat` | `lambda_hat_live` | `unit_root` | `gate_r2` | `v_max` | `secs` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** (control) | 0.6446726192039927 | 0.0 | 0.0 | 0.043322015553712845 | 1.4104527235031128 | −1.4324781894683838 | −1.4324781894683838 | True | 0.9888719478507275 | 4.585274696350098 | 1.881 |
| **8** | 0.6248685523735131 | 0.0 | 0.0 | 0.03543538227677345 | 1.6481777429580688 | −1.4479899406433105 | −1.4479899406433105 | True | 0.9732529553038906 | 4.585274696350098 | 1.715 |
| **9** | **1.281779592990027** | 0.0 | 0.0 | 0.00965781882405281 | **292.2945556640625** | **+0.30839410424232483** | **+0.30839410424232483** | True | **0.050886682051081156** | 4.585274696350098 | 1.710 |
| **10** | 0.6398332532283637 | 0.0 | 0.0 | 0.046166010200977325 | 1.255186915397644 | −1.4464397430419922 | −1.4464397430419922 | True | 0.9854557973057732 | 4.585274696350098 | 1.718 |
| **11** | 0.6408209504227382 | 0.0 | 0.0 | 0.042045947164297104 | 1.2995500564575195 | −1.46126127243042 | −1.46126127243042 | True | 0.9729963042022197 | 4.585274696350098 | 1.711 |
| **12** | 0.6762082068688114 | 0.0 | 0.0 | 0.041992831975221634 | 1.2870875597000122 | −1.4704492092132568 | −1.4704492092132568 | True | 0.9831772443281561 | 4.585274696350098 | 1.716 |
| **13** | 0.6348616577583769 | 0.0 | 0.0 | 0.04200531914830208 | 1.3961308002471924 | −1.459155559539795 | −1.459155559539795 | True | 0.9762957696220326 | 4.585274696350098 | 1.715 |
| **14** | 0.6338928504917678 | 0.0 | 0.0 | 0.034851837903261185 | 1.5189114809036255 | −1.4698615074157715 | −1.4698615074157715 | True | 0.9590363585727365 | 4.585274696350098 | 1.715 |
| **15** | 0.6805785772206486 | 0.0 | 0.0 | 0.0363335944712162 | 1.434257984161377 | −1.4400595426559448 | −1.4400595426559448 | True | 0.9812719581488555 | 4.585274696350098 | 1.718 |

### 1.3 Mechanical reads off those columns — no adjudication

- **Crossings, by the shipped `eval_nrmse <= floor_1` comparison against the
  header's `floor_1 = 0.7071067811865476`: `7` of the eight fresh seeds** —
  8, 10, 11, 12, 13, 14, 15. Seed 9 does not, at `1.281779592990027`.
- **`frac_gate_annihilated == 0.0` on 8 of 8 fresh cells** (9 of 9 including the
  control). `frac_gate_annihilated_0step == 0.0` on 9 of 9 as well.
- **`lambda_hat == lambda_hat_live` on all nine cells**, to the bit. That is
  JUPITER's it.7 §1.1 law read forward: no gate is annihilated, so no `−inf`
  enters the mean and the two columns cannot separate.
- **`lambda_hat > 0` on exactly one fresh seed, 9**, and that is the only fresh
  cell above `nrmse = 1.0`.
- **`a_hat_max` on the twelve non-diverging `arm_pl` cells spans
  `1.1029–1.6482`; the four diverging cells read `12.77 / 49.66 / 116.01 /
  292.29`** (retake seeds 2, 3, 7 and fresh seed 9). `unit_root == True` on
  **16 of 16** `arm_pl` cells, so on this arm the flag separates nothing.
- **`v_max` is `4.585274696350098` on all nine cells, identically.** It is a
  property of the drawn corpus at this shape, not of the fit.
- `[DERIVED]` **Over all sixteen distinct `arm_pl` cells, JUPITER's it.7 §3.1
  separation still has zero overlap and the margin tightens:**
  `max(nrmse | λ̂ < 0) = 0.680578577220649` (seed 15, this run) against
  `min(nrmse | λ̂ > 0) = 1.11333923299554` (retake seed 3). **Margin
  `0.432760655774893`**, against `0.451211` over the first eight. `13` cells
  below, `4` above, nothing between. **Whether that confirms his F1 constant is
  not this office's call.**
- `[DERIVED]` **Seed 9 is a fourth instance of the `a_hat_max` blow-up** he
  attributes to M2's open range, and it arrived in a batch that was not selected
  for it. The four diverging cells are 4 of 16, `gate_r2` `0.011–0.627` on the
  retake three and `0.050887` here.

### 1.4 CONTROL — THE INVOCATION REPRODUCES BITWISE

Seed 0 rode along and is compared against `results/v17k_r4_retake.jsonl` on the
stored floats, not the printed ones `[RUN]`:

| column | retake | this run | equal |
|---|---|---|---|
| `eval_nrmse` | 0.6446726192039927 | 0.6446726192039927 | **bitwise** |
| `a_hat_max` | 1.4104527235031128 | 1.4104527235031128 | **bitwise** |
| `a_hat_min` | 0.043322015553712845 | 0.043322015553712845 | **bitwise** |
| `lambda_hat` | −1.4324781894683838 | −1.4324781894683838 | **bitwise** |
| `lambda_hat_live` | −1.4324781894683838 | −1.4324781894683838 | **bitwise** |
| `gate_r2` | 0.9888719478507275 | 0.9888719478507275 | **bitwise** |
| `frac_gate_annihilated` | 0.0 | 0.0 | **bitwise** |
| `v_max` | 4.585274696350098 | 4.585274696350098 | **bitwise** |

**The control reproduces under a one-arm process against a three-arm one**, so
the eight fresh cells sit in the retake's frame.

### 1.5 THE PRICE — MEASURED, AND IT CAME IN UNDER

**Estimate: `14.240` GPU-s for eight cells (`8 × 1.780`, VENUS it.7 line 26).
Measured: `13.718` s of cell time for the eight fresh seeds. `−0.522 s`,
`−3.7 %`.**

| quantity | value | source |
|---|---|---|
| fresh-seed cell time (8–15) | **13.718 s** | sum of eight `secs` |
| all nine cells | **15.599 s** | sum of nine `secs` |
| mean per cell, nine cells | **1.7332222222222222 s** | `t="wall"` → `secs_per_run_by_arm.arm_pl` |
| mean per cell, fresh eight | **1.714750 s** | `[DERIVED]` |
| process wall clock | **19.588969469070435 s** | `t="wall"` → `secs` |
| non-cell fixed cost | **3.990 s** | `19.589 − 15.599` `[DERIVED]` |
| peak process working set | 1.2215042114257812 GiB | `t="wall"` |
| `cuda_max_memory_allocated` | 0.41673851013183594 GiB | `t="wall"` |

**Quote `arm_pl` as a range, not a point: `1.710–1.881 s` per cell on this box,
a `1.100×` spread.** That is far more stationary than `arm_smprime`, whose ten
it.6 cells spanned `15.907–22.069 s`, a `1.387×` spread. **The control cell is
the slowest of the nine at `1.881 s` and it is the first cell in the process**,
so `arm_pl`'s drift runs the other way from `arm_smprime`'s — a warm-up cost,
not a back-half cost. **A per-cell figure taken from `arm_pl` does not transfer
to `arm_smprime` and neither transfers as a point.**

**The it.6 overrun is not repeated:** `129 → 146.399` was `+13.2 %`;
`14.240 → 13.718` is `−3.7 %`. **The difference is that VENUS priced this one
off eight same-arm cells rather than off a six-cell cross-arm probe.**

---

## 2. EXPERIMENT C — THE CLAMP. THE POLE STORY IS DEAD, AND IT COST NOTHING.

**QUESTION.** Eight of ten it.6 cells read `a_hat_max == 1.0` exactly. Is that a
pole on the unit circle — marginal stability — or is it
`ceq/arm_smprime.py`'s `clamp(u, 0, 1)` saturating at its ceiling?

**ANSWER: IT IS THE CLAMP. Three independent measurements, no training run.**

**Line correction first.** The mandate cites `ceq/arm_smprime.py:109` for
`clamp(u, 0, 1)`. `[READ]` `:109` is `def magnitude(u: torch.Tensor) -> torch.Tensor:`;
the docstring naming the cap is `:110-112`; **the call is `:113`**:

```
113:    return torch.clamp(u, 0.0, 1.0)
```

### 2.1 The mechanism — the column IS a clamped magnitude

`[READ] scripts/v15_r1.py:360-362` — for `arm_smprime`, `gate_columns` takes

```
m, theta = arm_smprime.blend(*model.heads(x), model.g)
...
lg = torch.log(m)
```

and `:381-382`: `a_max = float(lg.max().exp())`, `a_hat_max=a_max`. So
**`a_hat_max` on this arm is `max_j m_j` and nothing else** — `theta` never
enters it. `blend` applies `magnitude()` **last** by design
(`ceq/arm_smprime.py:121-127`: *"THE CAP IS APPLIED AFTER THE BLEND, and the
ordering is load-bearing"*), so `m ∈ [0, 1]` for every input. **`a_hat_max`
cannot exceed `1.0` on `arm_smprime`, by construction.** And
`unit_root = bool(a_max >= 1.0)` (`:386`) therefore **can only ever fire by
equality on this arm — never by strict excess.** A flag that can only fire at
its own ceiling carries the ceiling's bit, not a spectral one.

### 2.2 The measurement that no convergence can explain — 0-step cells

`a_hat_max_0step` is the same function evaluated on the **as-constructed**
weights at **zero training steps** (`scripts/v15_r1.py:801-809`; the source at
`:796-800` says `identity_point` is deliberately *not* called there, so this is
the raw init). `[RUN]` over all **16 distinct `arm_smprime` cells** (retake
seeds 0–7 + it.6 seeds 8–15):

```
trained  a_hat_max == 1.0 :  seeds [0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15]   (12 of 16)
0-step   a_hat_max == 1.0 :  seeds [2, 3, 4, 7, 12, 14]                       ( 6 of 16)
max trained a_hat_max over 16 = 1.0     max 0-step a_hat_max over 16 = 1.0
```

**Six cells reach exactly `1.0` before a single gradient step is taken.** A
value present at initialisation is not something the arm converged to.
**Retake seed 2 is the cleanest single read: `a_hat_max_0step = 1.0`,
`a_hat_max = 0.5400443077087402` after training** — the head starts saturated
against the ceiling and training walks it **off**. That is the opposite of
convergence toward a pole.

### 2.3 The same clamp, saturating at its other endpoint, in the same cells

`a_hat_min == 0.0` on **15 of the 16** `arm_smprime` cells (seed 2 alone reads
`0.340760201215744`). `0.0` is the clamp's **lower** endpoint, and
`frac_gate_annihilated` names how much of the head sits on it:
`0.5032958984375` on twelve cells, with `n_zero_gates = 4123`.
`[DERIVED] 4123 / 0.5032958984375 = 8192.0` live positions — **so 4123 of 8192
positions of that head sit exactly on the lower cap.** The same two-sided clamp
is demonstrably saturating in the same cells, on the same tensor, at the same
step. `ceq/arm_smprime.py:46-48` says so in the source: *`clamp(u, 0, 1)` puts
half of a randomly-initialised head on the closed lower endpoint.*

### 2.4 The control: the same column on the UNCAPPED arm

`gate_columns` is one function. For `arm_pl` it reads `lg` off the head directly
(`scripts/v15_r1.py:359-361`) and `a_hat = exp(g)` has **no cap**. `[RUN]` over
all 16 `arm_pl` cells (retake 0–7 + this run 8–15), `a_hat_max` sorted:

```
1.1029  1.2551  1.2869  1.2871  1.2996  1.3961  1.4105  1.4342
1.4536  1.5052  1.5189  1.6482  12.7675  49.6605  116.0061  292.2946
```

**16 of 16 are strictly greater than 1.0, and the column reaches 292.29.**
Same instrument, same line, same iteration — the ceiling at `1.0` belongs to the
arm's clamp, not to the measurement.

### 2.5 WHAT THIS OFFICE REPORTS, AND WHAT IT DOES NOT

**REPORTED AS MEASURED:** `a_hat_max == 1.0` on an `arm_smprime` cell is the
`clamp(u, 0, 1)` ceiling. It is reachable at zero training steps, it is never
exceeded, and it is the only way `unit_root` can be `True` on this arm. **Any
reading of "marginal stability" or "an almost-all-pass filter" taken from
`a_hat_max == 1.0` alone is unsupported and must be said to die.**

**NOT DETERMINED, and named rather than papered over:** whether the pre-clamp
`u` at the argmax position sits strictly **above** `1.0` (hard saturation) or
lands on `1.0` exactly. Distinguishing those needs the pre-clamp tensor, which
no journal stores. **It does not change the answer** — either way the value is
the ceiling's and not a convergence fact — but it is a measurement this office
did not make. **Priced: journalling `max_j u_j` pre-clamp on both the trained
and the `_0step` record is a two-field addition on the path
`scripts/v15_r1.py:801-809` already walks, at 0 additional GPU-s**, the same
0-cost repair JUPITER's it.7 §2.5 asks for on `manifest.smp_values`.

---

## 3. EXPERIMENT B — NOT RUN. THE CAP IS UNDER-SPECIFIED, AND THE CLARIFICATION IS PRICED.

The mandate: *"use his, not one of your own; if he did not specify the mechanism
precisely enough to execute, say so and price the clarification rather than
inventing a cap."* **He did not, and this office does not.**

**Everything `V20_R15_IT7_JUPITER.md` says about the cap `[READ]`:** `:131`
(*"A norm cap arrests M2"*, quoting VENUS), `:133` (*"no capped run exists"*),
`:205-209` (*"capping `|w|` moves seeds 2, 3 and 7 into `λ̂ < 0`"* … *"three
capped cells at seeds 2, 3, 7 — ~6 GPU-s"*), and `:289` restating it. **No cap
value, no norm, no target tensor, no application rule, no code path.**

**Four things block execution, and each is a separate question:**

1. **`w` has no Python instance in the arm that was run.** The M2 theorem
   (`CEQ_V20_R15_CONTRACT.md:181-185`) is stated on `a = 2σ(w) − 1`.
   `[RUN] grep -rn "2 \* torch.sigmoid|2\*torch.sigmoid|2 \* sigmoid|2\*sigmoid|2\.0 \* torch.sigmoid" --include=*.py .`
   → **zero hits in the entire tree.** `arm_pl`'s gate is
   `g_head = nn.Linear(d_model, 1)` with `a_hat = exp(g)`
   (`ceq/arm_pl.py`; `scripts/v15_r1.py:333-337`). **`|w|` is therefore
   ambiguous over four different objects**: `g_head.weight`, `g_head.bias`, the
   head output `g`, or `a_hat = exp(g)`. Capping any two of them gives different
   cells.
2. **No norm is named.** L2 on the weight matrix, spectral norm, and L∞ on the
   per-position output are three different experiments.
3. **No value is named.** The cap the claim needs is precisely the one that
   moves `a_hat_max` from `12.77 / 49.66 / 116.01` below the threshold, and
   **choosing that value is choosing the result.** A cap fitted to make three
   cells pass is not a test of the causal claim; it is the claim assumed.
4. **No mechanism exists in the instrument, and adding one forfeits the frame.**
   `[RUN] grep -n "add_argument" scripts/v15_r1.py` → `:547-558`, **no cap, no
   clip, no max-norm flag.**
   `[RUN] grep -rn "clip_grad|clamp_|max_norm" scripts/v15_r1.py scale/m3_capability.py`
   → **zero hits.** Engaging a cap means editing the instrument, which changes
   `instrument_hash 5d41a63d…9a309` and takes the capped cells **out of the
   retake regime** — the one thing §1.1 exists to preserve. Whether that trade
   is acceptable is not a pricing decision and is not this office's to make.

**His price basis is also off, mildly, and it is now measured rather than
cited.** He prices at `1.884 s`/cell for `arm_pl` seed 0 from the retake. This
iteration measured seed 0 at **`1.881 s`** and the fresh eight at a mean of
**`1.714750 s`**.

**THE CLARIFICATION, PRICED.** Five answers are needed — target tensor, norm,
value, application rule (projection after each optimizer step, or a penalty
term), and a ruling on whether a `--gate-cap` flag may be added and a **new
`instrument_hash` accepted** for the capped cells. **With those five answers the
run is: 3 cells × `1.710–1.881 s` = `5.13–5.64` GPU-s of cell time, plus
`3.990 s` measured fixed cost → `9.1–9.6 s` wall**, one process, one journal.
**The GPU cost is not the obstacle. The specification is.**

---

## 4. TEST-BOUND, RED FIRST

`tests/mercury/test_v20_r15_it8_armpl_and_clamp.py`, six nodes.

**RED #1 — the pole story, written in the form it requires, and killed by the
journals `[RUN]`:**

```
    def test_a_hat_max_1_0_is_absent_before_training():
        """THE POLE STORY, written as it requires. RED."""
        smp = cells(load(IT6), "arm_smprime") + cells(load(RETAKE), "arm_smprime")
        saturated_at_init = sorted(
            c["seed"] for c in smp if c["a_hat_max_0step"] == 1.0)
>       assert saturated_at_init == [], saturated_at_init
E       AssertionError: [2, 3, 4, 7, 12, 14]
E       assert [2, 3, 4, 7, 12, 14] == []
E
E         Left contains 6 more items, first extra item: 2
tests\mercury\test_v20_r15_it8_armpl_and_clamp.py:67: AssertionError
```

**That node is §2's whole answer.** If `a_hat_max == 1.0` were a convergence
fact the list would be empty. It has six entries.

**RED #2 — a real defect in this office's first draft, recorded rather than
overwritten `[RUN]`:** the Experiment-B node globbed `results/*cap*.jsonl` to
assert no capped journal exists, and it matched
`r10_it8_capacity_softmax_t2.jsonl` and its siblings — **`*cap*` swallows
`capacity`.** A negative existence claim asserted through a loose glob reads
"capped cells exist" off files about capacity sweeps. Narrowed to
`v20_r15_it8_capped*.jsonl`.

```
2 failed, 5 passed in 0.49s
```

**GREEN after correction `[RUN]`:**

- `python -m pytest tests/mercury/test_v20_r15_it8_armpl_and_clamp.py -q` → **6 passed in 0.26s**
- `python -m pytest tests/mercury/ -q` → **99 passed in 9.91s** (93 at it.6 + 6 new)

---

## 5. PROVENANCE

| | |
|---|---|
| box | NVIDIA GeForce RTX 4060 Laptop GPU, `1073 MiB / 8188 MiB` used and `0 %` utilisation immediately before launch `[RUN] nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader` |
| instrument | `scripts/v15_r1.py`, working tree, **unmodified by this node**, `instrument_hash 5d41a63d…9a309` on all nine cell records |
| process exit | `0` (`REAL_EXIT`, taken from the process and not from a pipeline stage) |
| writing git commands | **none** |
| Kaggle | **not touched** — no kernel push, no dataset upload, no CLI call |
| frozen files touched | **none.** `results/v17k_r4_retake.jsonl` opened read-only, **424 lines before and after** `[RUN] wc -l`, absent from `git status --porcelain`. Clause (b)'s citations at `:161` and `:25` do not move |
| files created | `results/v20_r15_it8_armpl_b.jsonl`, `results/v20_r15_it8_armpl_seeds8_15.jsonl` (**aborted, 2 lines, 0 cells** — kept as the record of §0's abort), `tests/mercury/test_v20_r15_it8_armpl_and_clamp.py`, `V20_R15_IT8_MERCURY.md` |
| files appended | `house-events.jsonl` (harness convention) |
| not this node's | `M pytest.ini` was already modified at start and was not touched here |
| GPU time | Experiment A: `15.599 s` of cell time in a `19.589 s` process. Experiment C: **0 GPU-s**. Experiment B: not run |

**Not this office's to call:** whether seven crossings in eight fresh `arm_pl`
seeds confirms or refutes VENUS's it.7 pre-registration, whether the tightened
`0.432760655774893` margin holds JUPITER's Q3/W3 F1 constant, and whether the
death of the pole reading changes any grade. **The columns are above; the
scoring is VENUS's row at it.29.**
