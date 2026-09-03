# V20 R15 · it.26 · MERCURY — THE THREE LANDING DEFECTS, AND THE CLASS THE CENSUS CANNOT SEE

**All readings 2026-09-02, 12:17–12:36Z, local CPU, branch `v17k-gate0`.**
**This office wrote exactly two files: this one and `tests/mercury/test_v20_r15_it26_landing_repairs.py`.**
Other modifications visible in `git status` predate this iteration and are not this office's.
**Content digests of every file read or written (sha256[:16], raw bytes, at the time of reading):**

| file | digest |
|---|---|
| `V20_R15_THEORY_TABLE.md` | `3c4d1b270049f6ff` |
| `scripts/v15_r1.py` | `2bf99fd91de1fd12` |
| `ceq/kdata.py` | `c810c887956af343` |
| `ceq/beds/bed_k.py` | `c4ae405b4c54a920` |
| `tests/mercury/test_v20_r15_it26_landing_repairs.py` | written this iteration |

**No HEAD SHA is published here.** MARS, it.25: `0 of 22` files under `tests/mercury/` are
known to git, so a HEAD SHA would not name the suite it is printed beside.

| line | it.26 |
|---|---|
| THREE it.25 DEFECTS | **3 of 3 REPAIRABLE.** Replacements named below; the table is JUPITER's to edit. |
| MARKER NODE | `tests/mercury/test_v20_r15_it26_landing_repairs.py` — **3 RED / 3 GREEN**, RED against the unmutated table |
| **UNCITED CLAIMS (new)** | **`29` uncited constants across `10` of `12` cells**, of `62` hand-read from a `63`-token screen |
| LANDINGS CENSUS | **`21 of 129`** — NOT ADVANCED. Wall clock spent on the new measurement. |

---

## 1 — WHY THE UNCITED CLASS IS THE RIGHT WORK, IN ONE LINE

The INSPECTOR: *`[RUN]` provenance is necessary and not sufficient — every anti-fabrication
device this round built authenticates that a reading HAPPENED, none that the SUBJECT HAD A
VALUE.* **An uncited claim is the limiting case: there is no reading to authenticate.**
Every instrument the round owns starts at a citation and walks outward, so a cell can be
`129 of 129` green and still assert three numbers nothing points at. **This one starts at
the claim.**

---

## 2 — THE THREE DEFECTS. DISPOSITIONS AND EXACT REPLACEMENTS

**All three are citation defects, not code defects.** Adopting JUPITER's it.24 ruling —
*a defect marker must assert against the thing the repair will change* — each marker below
asserts the TABLE, and each is paired with a code-side control that is GREEN now and proves
the repo was not mutated to make the marker pass.

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it26_landing_repairs.py -q   12:33Z
      3 failed, 3 passed in 0.51s
      RED  : m25c_admission_condition_cites_the_generator_not_the_registry
             m25a_nine_flags_are_cited_as_a_range
             m25b_computed_constant_cites_its_formula_and_its_input
      GREEN: m25c_control_475_is_a_registry_entry_and_236_is_the_generator
             m25a_control_547_carries_one_flag_and_the_nine_span_547_558
             m25b_control_586_computes_floor1_and_the_value_is_nowhere_in_the_file
```

### 2.1 `M-25c` — REPAIRABLE. IT IS THE ONE THAT COSTS SOMETHING.

`V20_R15_THEORY_TABLE.md:306`, the F4 admission-condition table for Q2/W1. The clause is
what tells the leap **what it would take to re-enter the cell**, and a leap following
`ceq/kdata.py:475` to run one BED-K cell arrives at a dict key.

```
[RUN] sed -n 475p ceq/kdata.py                                          12:19Z
          "bed_k": {"generator": "ceq.beds.bed_k.build_delay",
[RUN] grep -n "def build_delay" ceq/beds/bed_k.py
          236:def build_delay(n: int, d: int, seed: int, plant=None) -> dict:
```

**`:475` holds the generator's NAME as a string. The generator is `ceq/beds/bed_k.py:236`.**
The *zero callers under `scripts/`* half of the clause stands (verified it.25, unchanged).

> **REPLACEMENT — JUPITER'S TO APPLY, `V20_R15_THEORY_TABLE.md:306`:**
> from `the generator exists at ` + `ceq/kdata.py:475` + ` and has **zero callers under `scripts/`**`
> to   `the generator is `build_delay` @ `ceq/beds/bed_k.py:236`, registered @ `ceq/kdata.py:475`, and has **zero callers under `scripts/`**`

**A ruling this defect forces, offered for the INSPECTOR:** a *registry entry* and its
*artefact* are two citations, not one, and an admission condition must carry the second.
The registry line is worth keeping — it is where the fixed kwargs `{n: 500, d: 4, seed: 7}`
live — but it cannot stand alone in a clause that tells someone what to run.

### 2.2 `M-25a` — REPAIRABLE, AND THE ROUND ALREADY OWNS THE NOTATION.

`V20_R15_THEORY_TABLE.md:97`. Count right, landing wrong.

```
[RUN] sed -n 547p scripts/v15_r1.py    ->  ap.add_argument("--seeds", ...)      # ONE flag
[RUN] grep -n add_argument scripts/v15_r1.py | head -9
      547 548 549 550 551 552 556 557 558      # nine, spanning :547-558
```

`C131` established `:A-B` at it.24. It exists and it was not used.

> **REPLACEMENT — `V20_R15_THEORY_TABLE.md:97`:** `scripts/v15_r1.py:547` → `scripts/v15_r1.py:547-558`.
> Nothing else in the clause changes; `9` and `s = 64` on `40 of 40` are both correct.

### 2.3 `M-25b` — REPAIRABLE, AND IT NEEDS A RULING FIRST. HERE IS THE RULING.

```
[RUN] sed -n 586p scripts/v15_r1.py    ->  floor1 = math.sqrt((T_STAR - 1) / T_STAR)
[RUN] grep -c 0.7071067811865476 scripts/v15_r1.py   ->  0   (exit 1, no output)
[RUN] sed -n 138p scripts/v15_r1.py    ->  T_STAR = 2
```

**A citation can be correct about where a value comes from and wrong that the value is
there.** Three parts are asserted at `V20_R15_THEORY_TABLE.md:71`; one lands (the formula),
one is absent from the cited file (`0.7071067811865476`, computed and never written), and
one is real but uncited (`t* = 2` @ `:138`).

> ### RULING OFFERED — **CAN A COMPUTED CONSTANT BE CITED AT ALL, AND TO WHAT?**
> **Yes, to two lines and never to one.** A constant that no file contains is cited to
> **(a) the line that computes it** and **(b) the line that fixes the input that makes the
> computation take that value.** Citing only (a) is a claim the reader falsifies by opening
> the file — `grep` returns nothing — and citing only (b) loses the formula. A third,
> optional pointer is legitimate where the *evaluated* value was banked: the results
> record. In this repo the value's only home is the journal, not the source.

> **REPLACEMENT — `V20_R15_THEORY_TABLE.md:71-72`:**
> from: `floor₁ = sqrt((t*−1)/t*) = 0.7071067811865476` at `t* = 2` (`scripts/v15_r1.py:586`; identity at `scripts/v15_r1.py:17`)
> to: `floor₁ = sqrt((t*−1)/t*)`, computed @ `scripts/v15_r1.py:586` from `T_STAR = 2` @ `scripts/v15_r1.py:138`, evaluating to `0.7071067811865476` — **a value no line of that file contains**; identity @ `scripts/v15_r1.py:17`
>
> The two re-cites at `:198` and `:206` cite the FORMULA only and are **CORRECT as they
> stand** — `floor1 = math.sqrt(...)` is literally at `:586`. No edit there.

---

## 3 — THE UNCITED-CLAIM CLASS. `29` ACROSS `10` OF `12` CELLS.

**Scope.** §2 of the table, `V20_R15_THEORY_TABLE.md:126-228`, the twelve cells and their
six fields each. §0, §1, §3–§5 are prose and are **out of scope for this number** — stated
so the denominator is not mistaken for the whole file.

**Screen, then hand-read, and the hand count is the number.** The it.25 lesson is that a
mechanical screen returns an artefact of where constants live. It did so again:

```
[RUN]  regex screen, all numeric tokens outside `path:line`   ->  240 tokens
[RUN]  regex screen, float-shaped tokens only                 ->   63 tokens
[HAND] rejected as a section reference, not a constant        ->    1   (`§0.3` @ :180)
[HAND] read and ruled                                         ->   62
```

**240 is not the denominator and must not be quoted as one** — it is dominated by census
integers (`3 of 3`, `16 of 16`, `0 of 40`), shape constants (`64`, `24`, `4096`), clause
numbers and GPU-second prices, which are not the class. **`62` is the denominator.**

**RULE APPLIED, stated so it can be attacked:** a constant is **CITED** if the cell offers
*some* `path:line` in support of it — a producer, a bind node, or the record it was read
from — including a wildcard `:*` node. It is **UNCITED** if no pointer anywhere in the cell
attaches to it. `[RUN]` provenance inside the cell does **not** make it cited: that is the
INSPECTOR's point, and this instrument is built on it.

| cell | uncited | the constants |
|---|---|---|
| Q1/W1 | 0 | `1.387779e-16` carries the instance node `:*` |
| Q1/W3 | **2** | `1.110223e-16` (measured, no node), `1e-6` (the tolerance itself, no bind) |
| Q2/W1 | **1** | `0.000e+00` — max−min of `R²₁·d`, `[RUN]` only |
| Q2/W3 | **1** | `−1.0` — the `NEG_ENTRY` witness value; `ceq/hankel.py:75` is cited in the *sibling* cell, not this one |
| Q3/W1 | **1** | `0.916258` (seed 3's NRMSE). `−0.4411` keeps `scripts/v15_r1.py:384` |
| **Q3/W3** | **5** | **`0.451211`, `0.662128`, `1.113339`** — the INSPECTOR's exemplar, confirmed — plus `frac_gate_annihilated == 0.0` and the `1.884 s`/cell price |
| Q4/W1 | **4** | `±1.49×`, `ρ = +0.70`, `309.047`, `34.8%` — the last two derived in-cell from this office's own withdrawn `~275` |
| Q4/W3 | **4** | `1.0589×`, `0.987×`, `1.154×`, `0.0` — the cited lines are cost-derivation producers, not the timing record |
| Q5/W1 | **4** | `0.203920`, `0.852`, `+0.20`, `+0.27` — the last two **self-declared** in the cell as *"still prose with no node"* |
| Q5/W3 | **4** | `ci_hi = 0.6582633033`, clearance `0.0488`, `0.705184`, `0.7784` |
| Q6/W1 | 0 | the annex's `0.492` / `0.519` are quoted *as refuted*; `0.492188` is cited |
| Q6/W3 | **3** | `14.465410797679917` (ratio derived in prose), `1.413115257241014`, `1.411094757351561` |

> ### **`29 uncited claims across 10 of 12 cells`, of `62` numeric constants hand-read in the twelve cells of §2.**

**Three findings the number alone does not carry.**

1. **The census and this instrument are near-disjoint.** Q3/W3 is a cell whose *citations*
   land and whose *headline constants* have none. **A cell can be green on every instrument
   the round owns and assert five numbers nothing points at.**
2. **`√2 = 1.4142135623730951` and the exact oracle's `0.0` were excluded as definitional**,
   not counted. Ruling this differently moves the number to `31`; the exclusion is stated so
   it can be reversed by whoever disagrees.
3. **Two cells declare their own uncited constants in their own text** (Q5/W1's *"prose with
   no node"*, Q4/W1's *"does not evaluate to 275"*). **The round has been finding this class
   one instance at a time for nine iterations without naming it as a class.**

---

## 4 — WHAT WAS NOT REACHED

- **The landings census stands at `21 of 129` and did NOT advance.** The one clause priced
  for this iteration was not opened; the wall clock went to §3. **No extrapolation is
  offered from `21` to `129`, and the `~5 office-iterations` remainder price is unchanged.**
- **The uncited class was measured on §2 only.** §0, §3 and §4 carry constants of the same
  shape (§0.2's `0.7071067811865476` is one, and it is `M-25b`) and are **unmeasured**.
- **`named artifacts` were screened but not counted.** The number above is numeric constants
  only. Backticked identifiers with no `path:line` are a second population; the screen was
  written, the hand read was not done in time.

## 5 — LIMITS

The `29` is one office's hand ruling under the rule stated in §3, on a table whose digest is
`3c4d1b270049f6ff` and which has been edited mid-iteration twice this round — **any count
quoted against a different digest is a different count.** The three replacements in §2 are
named, not applied: `V20_R15_THEORY_TABLE.md` is JUPITER's artifact and this office did not
edit it. The three REDs go GREEN on his edit alone and on no code change, which is what they
were built to prove. The `62`-constant denominator is a float-shaped screen; an integer
constant asserted without a pointer would not be in it. No git writes. Nothing touched
Kaggle.
