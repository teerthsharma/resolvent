# MARS (MORIARTY) — CEQ v20 ROUND 15, it.7

Two strikes filed against the round's most load-bearing measurement. **Three of
the four attacks the brief commissioned did not fire**, and they are reported
first, because the experiment is now what everything else is built on and the
calibration is worth more than the count.

`[RUN]` `python -m pytest tests/mars_v20/test_lambda_hat_is_one_bit_and_the_fresh_gate_is_the_corpus.py -q`
→ **4 failed, 3 passed in 2.52s**. All three passes are CONTROLS; all four
failures are RED by design.

**No git write. Nothing touched Kaggle. No moon ran code; one read documents.**

---

## 0. MY OWN PREDICTOR CAME BACK VACUOUS, AND IT IS WORTH NOTHING

`frac_gate_annihilated_0step < 0.15 ⇒ eval_nrmse < 0.3`, scored on all 16
distinct cells `[RUN]`:

| | positives issued | crossings | correct |
|---|---|---|---|
| retake seeds 0–7 | 1 (seed 2) | 1 (seed 2) | 1/1 |
| **fresh seeds 8–15** | **0** | **0** | **vacuous** |

The fresh eight span `_0step` **0.3809–0.9946**. The threshold neighbourhood
**0.15–0.30 was never sampled**, so the boundary was not probed either. The
positive branch has support **n = 1**, and that cell is the one the threshold was
drawn around at it.5 — **fitted on one, tested on zero.** The negative branch is
nearly free: with a crossing base rate of 1/16, "predict no crossing" is right
15/16 times by asserting nothing.

**Verdict: nothing, not one supporting instance in sixteen.** An instance that
only re-scores the cell the rule was cut from is not an instance. **I withdraw
`_0step < 0.15` as a predictor** and restate it as what it is: a description of
seed 2. It should not be scored again at it.29.

**Why the experiment could not have settled it, and this is a bound.** `[DERIVED]`
`1 − (15/16)^8 = 0.403`: **~60 % of the time an 8-seed run issues no positive at
all.** Reaching 95 % confidence of one crossing needs `n ≥ 44`. Filed as **L-M2,
TERMINAL** in the ledger. This is the second strike I have withdrawn this round.

---

## 1. TASK A — THE ATTACKS ON THE EXPERIMENT THAT DID NOT FIRE

### 1.1 The control is real. `[RUN]`

`test_control_seeds_0_and_1_reproduce_the_retake_bitwise` **PASSES**. Seeds 0
and 1 agree with `results/v17k_r4_retake.jsonl` on **59 of 60 shared columns,
bitwise**, `eval_nrmse` included (`0.9260818361710341`, `0.8812466551692475`).
The single difference is wall-clock `secs` (16.104 vs 16.164; 15.957 vs 16.032).

MERCURY's supporting READ verifies: `[RUN]` `grep -n "torch.manual_seed(seed)"
scripts/v15_r1.py` → **`231`** and **`801`**, exactly as cited. A cell's init is a
function of its own seed.

**And the control proves more than the READ does.** The retake ran `arms =
['arm_pl', 'arm_smprime', 'softmax']`; the fresh run ran `['arm_smprime']`. The
arm list therefore *did* change between the two processes, and seeds 0–1 still
reproduce to the bit. **Independence of the arm list and the seed order is
measured, not merely read.** The ten cells are independent draws. **Attack does
not fire.**

### 1.2 The regime is identical. `[RUN]`

Field-by-field header diff: **18 of 22 fields identical**, including
`instrument_hash 5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309`,
`torch 2.5.1+cu121`, `device cuda`, `cublas_workspace_config :4096:8`,
`deterministic_algorithms True`, `s 64`, `d 24`, `t_star 2`, `n_train 2048`,
`n_eval 4096`, `steps 150`, `lr 0.02`, `task e3_t2`, `threads 8`,
`floor_1 0.7071067811865476`.

Four differ, all invocation metadata: `arms`, `seeds`, `tag`, `when`.

**Minor correction to MERCURY:** the claim is "all seventeen header fields
match"; the count is **eighteen**, and **four** differ rather than being absent.
The substance is right and the one that mattered — `arms` — is neutralised
empirically by 1.1. **The sixteen cells are poolable. Attack does not fire.**

### 1.3 `frac_gate_annihilated` is the corpus again — STRIKE 10

This one fires, and it is my it.1 STRIKE 1 territory holding on cells the corpus
argument had never seen.

#### The RED, verbatim

```
[RUN] python -m pytest tests/mars_v20/test_lambda_hat_is_one_bit_and_the_fresh_gate_is_the_corpus.py -q

E  AssertionError: 8/10 independently seeded cells publish a gate diagnostic equal to
   a CORPUS-ONLY statistic. corpus: neg=4123 (0.5032958984375) pos=4069
   (0.4967041015625) of n=8192. cells: [(0, 0.5032958984375), (1, 0.5032958984375),
   (8, 0.5032958984375), (9, 0.5032958984375), (10, 0.5032958984375),
   (12, 0.5032958984375), (14, 0.5032958984375), (15, 0.5032958984375)]
```

**The mechanism, from the producer.** `scripts/v15_r1.py:699-700` `[READ]`:

```python
x_ev, y_ev, f_ev, p_ev = batch_fn(a.n_eval, S, D, d_model=D_MODEL, seed=12345,
```

**The eval draw is at a hardcoded `seed=12345` and does not move with the cell
seed.** `gate_columns(kind, model, x_ev, live)` (`:729`) reads every gate column
on that one fixed draw. `live = range(head+1, S)` with `head = S-1-T_STAR` is
**2 positions**, so `n = 4096 × 2 = 8192`. `[RUN]` The census, counted with **no
arm at all**, is `neg = 4123 → 0.5032958984375`. Eight of ten cells publish that
float exactly.

**Settled: the column is measuring the arm on 2 of 10 cells and the corpus on 8.**

#### And what `0.99462890625` and `0.976806640625` are

`[RUN]` `test_the_two_escaping_cells_are_not_a_second_corpus_value` is RED:

```
E  AssertionError: cells whose gate column is an ARM reading rather than the census:
   [(11, 0.99462890625, 1.120603), (13, 0.976806640625, 1.203324)]. Both fail ABOVE
   1.0 while the eight census-valued cells fail at 0.85-0.93
```

`= 8148/8192` and `8002/8192`. **Neither is a corpus count**, so these two are the
only genuine arm readings in the file. **Both escape UPWARD** — nearly every
position driven onto the closed endpoint — and both fail *worse* than every
census cell.

**Replacement route.** The column cannot rank arms and must not enter an identity
hash while it does this. Two candidates already in every journalled row and
costing nothing to switch to: **`frac_gate_annihilated == 0.0`** as a boolean
(seed 2 alone across all 16, the only crosser) and **`lambda_hat_live`** for the
magnitude. Both are arm quantities; neither can equal a sign census. A third,
cheaper route: re-read the columns on the *train* draw, which is seeded at the
cell seed (`:708-709`), so a corpus coincidence cannot survive across cells.

---

## 2. TASK B — SEEDS 11 AND 13: THERE IS NO CONTRADICTION, AND THE READING IS THE CASUALTY

**STRIKE 9.** From the producer, `scripts/v15_r1.py:382-387` `[READ]`:

```python
col = dict(a_hat_max=a_max, a_hat_min=float(lg.min().exp()),
           lambda_hat=float(lg.mean()),
           lambda_hat_live=(float(lg[fin].mean()) if bool(fin.any())
                            else float("-inf")),
           frac_gate_annihilated=float((~fin).double().mean()),
           unit_root=bool(a_max >= 1.0),
```

with `a_max = float(lg.max().exp())` at `:381` and `fin = torch.isfinite(lg)` at
`:380`.

**`lambda_hat` is a MEAN over every live position, `-inf` included.** A single
non-finite entry drags an unweighted mean to `-inf`. **`unit_root` is a MAX**,
and a max ignores `-inf` unless every entry is `-inf`. **The two columns share no
arithmetic.** `unit_root == False` beside `lambda_hat == -inf` is not an anomaly;
it is the generic case, and the it.5 reading that said it should not occur was
wrong. An anomaly budget spent on seeds 11 and 13 on that basis is spent on
nothing.

**Which column is not measuring what the round reads it as: `lambda_hat`.** `[RUN]`

```
E  AssertionError: on 18/18 cells `lambda_hat == -inf` is EXACTLY the predicate
   `frac_gate_annihilated > 0` -- one zero gate in 8192 forces it.
```

**`lambda_hat` carries exactly one bit, and it is a bit already published beside
it.** It has no magnitude to read.

**The consequence for VENUS and JUPITER, and it is the opposite of what they
argue.** `V20_R15_IT5_VENUS.md:128` — "`lambda_hat` is `−inf` — the arm has no
fitted decay because **the hop has been switched off**."
`V20_R15_IT6_JUPITER.md:280-284` (Q3) — "the predicted descent to the exact
corner (`lambda_hat = -inf`, **gate dead**, landing at `0.88-0.93`)."

`lambda_hat_live` (`:384`) is the quantity both sentences describe, it is already
in every row, and on the eight cells the corner argument is about `[RUN]`:

| seed | `eval_nrmse` | `a_hat_max` | `lambda_hat_live` | `unit_root` |
|---|---|---|---|---|
| 0 | 0.926082 | **1.0** | −0.0195 | True |
| 1 | 0.881247 | **1.0** | −0.0118 | True |
| 8 | 0.852061 | **1.0** | −0.0320 | True |
| 9 | 0.902522 | **1.0** | −0.0035 | True |
| 10 | 0.892747 | **1.0** | −0.0254 | True |
| 12 | 0.919852 | **1.0** | −0.0137 | True |
| 14 | 0.909944 | **1.0** | −0.0072 | True |
| 15 | 0.891047 | **1.0** | −0.0072 | True |
| 11 | 1.120603 | 0.0699 | −4.1869 | **False** |
| 13 | 1.203324 | 0.7775 | −2.4989 | **False** |

**On every cell landing in the `0.88–0.93` band the gate is not dead.**
`a_hat_max == 1.0` exactly, `unit_root == True`, and the surviving half of the
band decays by **under 3 % per position**. The only two cells with a genuinely
collapsed gate are 11 and 13 — **and they are not in the band; they fail above
1.0.** The mechanism argument has the sign of its own evidence backwards.

**Replacement route.** Every use of `lambda_hat` in a mechanism claim becomes a
use of `lambda_hat_live`, which is already journalled on all 18 cells. `[DERIVED]`
Under that substitution VENUS's "descent to the exact corner" is not supported by
the eight band cells and *is* supported by 11 and 13 — a different claim about a
different population, which she is free to make.

---

## 3. TASK C — LEAP LEDGER

Five rows appended to `V20_R15_LEAP_LEDGER.md` (the file did not exist; created
with MARS's rows under their own heading so JUPITER's append merges):

| row | failure | grade | field / bound |
|---|---|---|---|
| **L-M1** | VENUS §2.2 body falsified 2/8; falsifier unfireable | **LEAPABLE** | error-statistical severity (Mayo) — tests whose rejection region is a proper subset of the hypothesis complement |
| **L-M2** | MARS `_0step` predictor vacuous | **TERMINAL** | base-rate power bound: `1−(15/16)^8 = 0.403`; `n ≥ 44` for 95 % |
| **L-M3** | seed 2, 1-in-16, no bound mechanism | **LEAPABLE** | anti-concentration / small-ball probability (Littlewood–Offord, Rudelson–Vershynin) |
| **L-M4** | the five observed gate states | **split** | TERMINAL (the draw) for the 3 census states; LEAPABLE (same field as L-M3) for seeds 11, 13 |
| **L-M5** | the round's reading of `lambda_hat` | **TERMINAL** | an instrument definition, not a missing theorem — `lambda_hat_live` is already there |

On **L-M1**: her rate claim survives. "Expected crossings: 1 of 8; 95 %
predictive interval 0–3" against 0 observed is **inside the interval**. The
falsified part is the *body* — the claimed value set for failing seeds — and her
named falsifier ("fails with a **live** gate") reads only the tail the
violation did not occur on. Seeds 11 and 13 fail with a gate that is *more*
annihilated. **The falsifier is not the negation of the prediction.** That is a
pre-registration hole, and it is the same class of hole my own L-M2 has.

---

## 4. SEARCH PROOF

Same invocation style, positive control first — the line MERCURY cites, found:

```
[RUN] grep -n "torch.manual_seed(seed)" scripts/v15_r1.py
231:    torch.manual_seed(seed)
801:                torch.manual_seed(seed)

[RUN] grep -n "batch_fn(a.n_eval" scripts/v15_r1.py
699:    x_ev, y_ev, f_ev, p_ev = batch_fn(a.n_eval, S, D, d_model=D_MODEL, seed=12345,

[RUN] grep -c "seed=12345" scripts/v15_r1.py
2
```

The positive control finds both cited lines. The same invocation then finds the
**single** eval-draw construction, and it is seed-constant — which is §1.3's
whole mechanism.

---

## 5. LIMITS

The census identity is an equality of published floats against a
no-arm count, which establishes that the values coincide; it does not prove the
arm computes the census, only that a diagnostic and its null agree, which is what
`MISTAKES.md` M-18 makes disqualifying. The `n ≥ 44` figure in L-M2 assumes the
crossing rate is the observed 1/16 and treats seeds as i.i.d. Bernoulli; the true
rate is unknown and 1/16 is itself an estimate from 16 draws. The header diff
covers the 22 journalled fields only — an unjournalled regime difference would
not appear in it, and the bitwise control in §1.1 is the stronger evidence
against one. `lambda_hat_live` is offered as the corrected magnitude column, not
as a crossing predictor: `[RUN]` across the 16 cells it does **not** separate
crossers (seed 2 at −0.795 crosses; seed 3 at −0.441 and seeds 11/13 at −2.50
and −4.19 do not). No claim is made that STRIKE 9 or 10 changes which arm wins;
they change what the round is entitled to say about *why*.
