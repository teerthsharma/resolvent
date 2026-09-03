# MARS (MORIARTY) — CEQ v20 ROUND 15, it.5

One strike filed. **Three of the four attacks the brief commissioned did not
fire**, and they are reported first, because the room has started agreeing with
MARS and the calibration is worth more than the count.

`[RUN]` `python -m pytest tests/mars_v20/test_seed2_is_decided_before_training.py -q`
→ **1 failed, 5 passed in 0.47s**. The failure is STRIKE 8, RED by design.
`[RUN]` `python -m pytest tests/mars_v20/ -q` → **17 failed, 19 passed in 6.09s**;
16 of the 17 are the standing it.1/it.2 RED-by-design nodes, unchanged.

**No git write. Nothing touched Kaggle. VENUS's `V20_R15_IT5_VENUS.md` had not
landed at close of clock (polled at 4m51s and again at file time).**

---

## STRIKE 8 — THE SEED-2 CELL IS DECIDED BEFORE THE FIRST GRADIENT STEP

**The claim struck:** that W1's tournament-best `eval_nrmse` `0.203920` is a
property of the path-product primitive, and therefore that it belongs to W1 in
VENUS's ranking.

### The RED, verbatim

```
[RUN] python -m pytest tests/mars_v20/test_seed2_is_decided_before_training.py -q

E  AssertionError: the winning cell is separated from all seven siblings at step 0,
   before any gradient: gap=0.0758056640625,
   frac_gate_annihilated_0step per seed={0: 0.5048828125, 1: 0.611328125,
     2: 0.124755859375, 3: 0.44287109375, 4: 0.5157470703125, 5: 0.5882568359375,
     6: 0.57958984375, 7: 0.2005615234375},
   trained frac_gate_annihilated={0: 0.5032958984375, 1: 0.5032958984375, 2: 0.0,
     3: 0.4967041015625, 4: 0.5032958984375, 5: 0.5032958984375,
     6: 0.5032958984375, 7: 0.5032958984375}
E  assert 0.0758056640625 <= 0.0

1 failed, 5 passed in 0.47s
```

### it.2 was wrong, and wrong in a useful direction

it.2 said the cause "needs weights not in the tree". `[RUN]` the search: **342
weight-bearing files in the tree (`*.safetensors|pt|pth|ckpt|bin|npz|npy`), and
0 of them name `arm_smprime`.** `results/m3_quintuple_v2_weights/` holds 122
`.pt` over arms `argmaxste settled settledrow softmax twin twinrow`, seeds 0–5;
`results/m3_quintuple_v2_cuda_weights/` 18 more over `settled softmax twin`;
`ceq/hf_artifact/weights/` five safetensors, all arm `twin`, seeds 0–4
(`ceq/hf_artifact/weights/MANIFEST.json`). `[READ]` `scripts/v15_r1.py` contains
no `torch.save` and no `save_file` — **the retake runner never checkpointed
anything.** The weights do not exist and never did.

They are also not needed. **`scripts/v15_r1.py` journals the UNTRAINED gate
state beside the trained one**, in the `*_0step` columns, and the answer was
sitting in `results/v17k_r4_retake.jsonl` the whole round.

### What the 0-step column says

| seed | 0 | 1 | **2** | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `frac_gate_annihilated_0step` | .5049 | .6113 | **.1248** | .4429 | .5157 | .5883 | .5796 | .2006 |
| `frac_gate_annihilated` trained | .5033 | .5033 | **.0000** | .4967 | .5033 | .5033 | .5033 | .5033 |
| `eval_nrmse` | .9261 | .8812 | **.2039** | .9163 | .9119 | .9182 | .8932 | .9267 |

Seed 2 begins with **4x less gate at the clamp floor than the median cell** and
is separated from its nearest sibling by `0.0758` with **zero overlap across all
eight seeds** — before a single optimizer step. Seven cells start at
`0.20`–`0.61` and are pulled to the `~0.5033` fixed point; seed 2 starts at
`0.1248` and escapes to an exact `0.0`. This is a **basin event fixed at
initialisation**, and its predictor is a journalled column.

### The corroboration, from the other arm

`test_seed2_pins_the_same_a_hat_min_in_two_structurally_different_arms` GREEN.
On seed 2 alone, `a_hat_min` is `0.340760201215744` for W1 and
`0.3407682776451111` for W3 — **agreement to `2.37e-5` relative between a clamped
path-product and an unclamped log-head prefix scan**, reached from *different*
starting points (`a_hat_min_0step` is `0.0` for W1 and `0.4297` for W3). Two
different operators do not converge to five significant figures by training
coincidence. Whatever pins `0.34076` on seed 2 is **upstream of the arm**.

And `test_seed2_is_w3s_worst_cell_while_it_is_w1s_best` GREEN: seed 2 is W1's
**best** cell (`0.2039`) and W3's **worst** (`1.1523`, with `a_hat_max 12.77`,
one of W3's three divergent seeds). A seed that made the primitive better would
not also make the other primitive diverge.

### The answer to the seed-2 question, which the brief asked to be stated plainly

**Neither "wing" nor "artifact". It is a 1-in-8 basin event whose predictor is
journalled, and it has `n = 1`.** The number is real — it was measured, not
mis-parsed, and it escapes STRIKE 1. What is *not* established is that the
primitive produced it: on the record, the thing that distinguishes seed 2 is a
draw that happened before training, and the same draw ruins the other wing.

**Consequence for VENUS.** W1's claim to the tournament rests on a cell whose
advantage is predicted by its initialisation. That does not make the ranking
trivial, and it does not make W1's number an artifact. It means the ranking
**cannot credit W1 for seed 2 at `n = 1`** without the one experiment that turns
a coincidence into a mechanism.

### The replacement route, priced

`REROUTE`, not retire. The hypothesis is sharp and falsifiable in one line:

> `frac_gate_annihilated_0step < 0.15` predicts `eval_nrmse < 0.3` for
> `arm_smprime`.

Currently `1/1` in favour and `7/7` against on the complement — a perfect
separation on eight points, which is exactly the sample size at which a perfect
separation means least. **Cost: `15.970 s`/cell (`V17_R4_RETAKE_PRICE.md:194`)
× 8 fresh seeds = `127.8` GPU-s**, ≈ 2 minutes on this box, no new instrument —
`frac_gate_annihilated_0step` is already emitted. `[DERIVED]`

The planted negative is in the file: `separation("a_hat_min_0step")` returns
exactly `0.0` on the same predicate over the same eight cells, because that
column is flat. **The `0.0758` is a measurement, not a separator that separates
everything** (`MISTAKES.md` V-7 class).

---

## TASK A — THE FREEZE. BOTH ATTACKS COMMISSIONED, NEITHER KILLS `N = 2`.

### A.1 SATURN's found-vs-named rule — **the rule is loose, the verdict is right**

The rule reads *"≥1 journalled record whose `kind` is the arm name."* `[RUN]`
the census by `(t, kind)` over `results/v17k_r4_retake.jsonl`:

```
('cell','arm_pl') 8      ('trace','arm_pl') 128    ('agg','arm_pl') 1
('bind','arm_pl') 2      ('contrast','arm_pl') 1   ('identity','arm_pl') 1
('probe','arm_pl') 1                               ('header',None) 1
```

**Of SATURN's 191 `arm_pl` records, 18 are trained cells** (across three files);
of 182 `arm_smprime`, 10. `[RUN]` moon census: 90.6% / 94.5% / 90.1% of the
three counted kinds are **not produced cells**.

The degeneracy the brief predicted is real and it is nameable: the `identity`
record carries keys `['all_ones_causal_mask','annihilating','device','kind',
'method','s','t']` — **no seed, no steps, no eval, no training at all.** It is a
structural probe of the operator. Under the rule as written, that one record
alone makes a wing FOUND. So does the `agg` summary row (`seed: None`), and so
does any one of the 128 heartbeat `trace` polls, which an arm crashing at step 1
would emit.

**The attack does not reach the verdict.** `[RUN]` the strict filter
`t=="cell" AND steps>0 AND finite eval_nrmse` gives `arm_pl` 18,
`arm_smprime` 10, `softmax` 18, **`arm_phase` 0** — the same verdict, with the
false confidence removed. And the exclusion half is clean: zero `arm_phase`
records under **any** of the 736 distinct top-level keys across
`results/**/*.jsonl`; the single repo hit is a prose note in
`results/k_cert_local.json` describing a contrast, not a result.
`CALIBRATION`: the identical invocation returns 3 hits for `arm_smprime` under
`results/`, so the zero is a measurement.

`REPRICE`, not retire: **tighten clause (b) to `t=="cell"`.** One predicate, no
new instrument, and it costs the freeze nothing because the digest already
covers a `cell` line (`results/v17k_r4_retake.jsonl:161` and `:25` are both
`{"t":"cell"}`). SATURN cited the right two lines under a rule that would have
admitted the wrong ones.

### A.2 JUPITER's exercised-map principle — **it survives, and it should**

The brief asked whether a rule that keeps two entries whenever the record has
not exercised the map rewards not running the experiment. **In isolation, yes.
In composition with the round's other binding kill, no**, and the honest answer
is that the principle holds.

Composed, `N(record) = #FOUND(record) − #(retirements the record exercises)`.
The found-vs-named rule can only *raise* N by running; the exercised-map rule
can only *lower* it by running. Neither direction pays for silence: an arm that
runs nothing is NAMED and scores 0, which is exactly what happened to W2.
Applied to the empty record the composition returns `N = 0`, not `2`. The
perverse incentive I went looking for is not there.

Two attempted kills that failed, reported because they failed:

1. **"W1's `a_hat_max ≤ 1.0` on 8/8 is the `1.0` filled-in default, not a
   measurement."** `[READ]` `scripts/v15_r1.py:373-378` — the hard-coded
   `a_hat_max=1.0` return is on the **`softmax`** branch only, and the source
   comment says so. `arm_smprime` goes through `lg = torch.log(m)` at `:365` and
   `a_max = float(lg.max().exp())` at `:381`. The six W1 cells at exactly `1.0`
   are the clamp endpoint reached, as JUPITER said. **Does not fire.**
2. **"The empty intersection is forced by the code, so the falsifier is
   unpurchasable."** `[READ]` `arm_pl`'s gate column is `lg = model.heads(x)[0]`
   — a raw unbounded head, so `a_hat_max ≤ 1.0` requires only `max(lg) ≤ 0` and
   nothing structurally forbids it. **Does not fire.** JUPITER's falsifier is
   genuinely purchasable.

**What does stand is a pricing asymmetry, and it is worth a number.** `[RUN]`
over the full journal, not the 8 terminal cells: **137 `a_hat_max`-bearing
records per arm**, 17x the sample the freeze read. The empty intersection holds
on all of it — W1's max over 137 records is `1.0`, W3's min is `1.0745`. So
JUPITER's fact is *stronger* than he claimed. But the boundary-nearest W3 point
is `log a_hat_max = 0.0719`, and **`N = 2` is one seed and seven hundredths of a
nat from `N = 1`**.

The round priced the route that would raise N to seven line items
(`V20_R15_WING_MANIFEST.md:110-128`, ≈41–130 GPU-s for W2) and **priced the
route that would lower it at nothing**. `[DERIVED]` eight more `arm_pl` seeds at
`1.614 s` (`V17_R4_RETAKE_PRICE.md:195`) = **`12.9` GPU-s** — the cheapest
unrun experiment in the round, **3.2x cheaper than the W2 route that was
priced**. `REPRICE`: the manifest should carry the collapse route beside the
raise route, at 12.9 GPU-s.

---

## TASK C — THE LOG. THE ATTACK MISSES THE INSTRUMENT AND HITS THE PROSE.

it.2 said a `status`-keyed grep sees none of MERCURY's `"state":"RED"` reds.
**That is wrong about the instrument, and I withdraw it.** `[READ]`
`scale/ledger.py:76-84`:

```python
def _status(event: dict[str, Any]) -> str | None:
    raw = event.get("status", event.get("state"))
    return raw.lower() if isinstance(raw, str) else None
```

It accepts **both** spellings and case-folds. `is_bound()` (`:183-185`) routes
through `tests(status="red")` (`:140-149`) which uses `_status`. **The
Inspector's shipped reader sees MERCURY's reds.** it.2's STRIKE was against a
grep, and the code had already been repaired.

**The blast radius, quantified anyway, because the number is not zero.**
`[RUN]` over `house-events.jsonl`: **11,780 lines, 11,776 parse, 4 do not.**
Union of every RED spelling: **1,378 overall / 1,276 in R15**. A literal
top-level-`status` search finds **214 / 150**. **Invisible to that search:
1,164 overall, 1,126 in R15** — `cameron` 1,025, `FOREMAN` 10, `Chase` 9,
`MERCURY` 4 (lines 9274, 9276, 9277, 9278), `NEPTUNE` 3, `HOUSE` 1.
`CALIBRATION`: the same method counts 345 records carrying a top-level `status`,
of which 214 are red — a positive control, so the 1,164 is a measurement.

**The residual defect is real and it is structural, not a spelling.**
`tests()` filters on `e.get("t") == "test"` **first**. The 116 records with no
`t` field are therefore **unbindable by construction** regardless of how their
status is spelled: 114 carrying `kind` (lines 399–405, 419–452, 489–548,
775–784, 993–995), 1 carrying `event` (`:11744`, appended this round), 1
carrying neither (`:11719`). A finding logged into any of those three schemas
cannot be bound and cannot be seen to be unbound.

Second residual: `scale/ledger.py:14` still documents
`grep -c '"agent":"HOUSE","status":"red"' house-events.jsonl -> 0` as the
cautionary example. It is correct as history, and it is also the string a reader
in a hurry copies. `REROUTE`: the file already ships `is_bound()`; the prose
should point at it rather than at a grep that returns 0 for three independent
reasons (`json.dumps` spacing, the `state` spelling, case).

---

## THE ATTACKS THAT DID NOT FIRE

| # | attack | outcome |
|---|---|---|
| 1 | W1's `a_hat_max = 1.0` is `gate_columns`' filled-in default | **NO.** That default is the `softmax` branch. `scripts/v15_r1.py:373-378`. |
| 2 | The empty W1/W3 intersection is forced by code, so JUPITER's falsifier is unpurchasable | **NO.** `arm_pl`'s gate head is unbounded. |
| 3 | The exercised-map principle rewards not running the experiment | **NO,** once composed with found-vs-named. `N(empty record) = 0`. |
| 4 | The intersection is empty only because the freeze read 8 cells of 137 | **NO** — and it backfires: it holds on all 137, so JUPITER's claim is stronger than stated. |
| 5 | A journalled `arm_phase` result hides under a key other than `kind` | **NO.** Zero across 736 distinct top-level keys. |
| 6 | The Inspector's binding reader cannot see `"state":"RED"` (MARS it.2) | **NO. WITHDRAWN.** `_status` normalises both. |
| 7 | Seed 2's `a_hat_min` matches W3's because both are an untrained init value | **NO.** They start at `0.0` and `0.4297` and *converge* to `0.34076`. Which makes it worse, not better. |

Seven attacks aimed, one landed. That is the calibration.

---

## LIMITS

STRIKE 8 establishes that seed 2 is separated at initialisation on one journalled
column; it does **not** establish that the initialisation *caused* the win —
`n = 1` on the low side, and the 127.8 GPU-s above is exactly the experiment that
would decide it. The cross-arm `a_hat_min` agreement is a two-point coincidence
with no third arm to check it against, since W2 was never produced. The
`frac_gate_annihilated_0step` figures are `[READ]` off the retake journal and
inherit whatever that journal inherits; nothing was re-trained here and nothing
ran on CUDA. The 1,164 / 1,126 invisible-RED counts are a moon's `[RUN]` and
were not independently reproduced by MARS; the `scale/ledger.py` line numbers and
the `_status` body were read directly and are `[READ]`. The A.1 strict-filter
counts (18 / 10 / 18) are a moon's; the `(t, kind)` breakdown of the retake file
is MARS's own `[RUN]`. VENUS's it.5 ranking had not landed and is unattacked.
