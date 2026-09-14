<p align="center">
  <img alt="Lean 4 v4.7.0" src="https://img.shields.io/badge/Lean_4-v4.7.0-blue?style=flat-square" />
  <img alt="134 theorems and 41 lemmas" src="https://img.shields.io/badge/theorems_%2B_lemmas-134_%2B_41-success?style=flat-square" />
  <img alt="sorry: 0" src="https://img.shields.io/badge/sorry-0-success?style=flat-square" />
  <img alt="corner tests: 69 passing" src="https://img.shields.io/badge/corner_tests-69_passing-success?style=flat-square" />
  <img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" />
</p>

<h1 align="center">resolvent</h1>

<p align="center">
  <b>Softmax attention and Markov path composition are the same operator.</b><br/>
  <span>One causal head, three switches. Softmax attention, linear attention and the exact path product of a Markov chain are settings of it &mdash; proved in Lean 4, matched bitwise in code.</span><br/>
  <i>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></i><br/>
  <sub><a href="mailto:teerths57@gmail.com">teerths57@gmail.com</a> · <a href="https://github.com/teerthsharma/resolvent">github.com/teerthsharma/resolvent</a></sub>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#1-one-family-three-corners">The family</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#3-the-graded-read">The graded read</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#5-what-the-failures-built">What the failures built</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#6-contributing">Contributing</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="lean/CEQ/">Proofs</a>&nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="docs/FAILS.md">Research log</a>
</p>

<p align="center"><sub>Every number on this page comes with the command that reproduces it. Everything that did not hold (retractions, broken commands, open questions) is logged in one place: <a href="docs/FAILS.md">docs/FAILS.md</a>.</sub></p>

---

## Abstract

Attention mixes **weights across positions**. Recurrences and Markov chains
compose **values along paths**. Architectures usually pick one side. resolvent
is a single causal softmax head in which both sides are settings of the same
three switches, so one layer can sit at either end or anywhere between.

- **One operator, three regimes.** The switches `β`, `qk` and `g` move the head
  between softmax attention, linear attention and the exact path product. All
  three are live, trainable parameters on the shipped module.
- **Every theorem has a test.** Each corner is a Lean 4 theorem, and the
  float64 implementation is checked against each statement, bitwise at the
  corners. 175 declarations, zero `sorry`.
- **Gates that can close.** The hop is built as a path product, so a gate can
  reach exactly zero. Lean proves that no prefix scan in the logit can do that.
- **A read that knows when to refuse.** The resolvent read computes where a
  process ends up with one triangular solve. When a counterfactual has no
  defined answer it returns a refusal with the reason, at 100% sensitivity and
  100% specificity on the published check.
- **Relations, not only positions.** The graded read covers edge flows with
  curl, which no model that reads edges as differences of node values can
  represent, at any width, depth or budget.

**Keywords:** causal attention · softmax attention · linear attention · path
products · resolvent · committor functions · Hodge decomposition · selective
prediction · formal verification · Lean 4

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 resolvent · measured 2026-09-11 · Windows 11 · Python 3.11.9 · torch 2.14.0 (CPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Lean 4 proofs          13 files · 134 theorems + 41 lemmas · 0 sorry · build exit 0
 Corners in code        69 / 69 tests pass            tests/arm_smprime, tests/arm_pl
 Softmax corner         max |Δ| vs causal softmax     0.000e+00
 Committor closed form  max |Δ|                       4.441e-16
 Curl vs node models    share left unrepresented      mixed 0.6720 · pure curl 1.0000
 Planted negative       pure-gradient target          6.5e-16
 Refusal                sensitivity · specificity     100.00% · 100.00%  (138 + 262)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Quick start

```bash
git clone https://github.com/teerthsharma/resolvent && cd resolvent
pip install -r requirements.txt

python -m pytest tests/arm_smprime tests/arm_pl -q   # 69 passed: the Lean corners, in code
python -m ceqjepa.operator                           # ALL SELF-CHECKS PASSED
python -m ceqjepa.dr1                                # ALL SELF-CHECKS PASSED
python scripts/lean_count.py                         # 13 files: 134 theorems + 41 lemmas = 175
(cd lean && lake build CEQ)                          # exit 0
```

---

## 1. One family, three corners

`ceq/arm_smprime.py` implements the family:

```
W_ij = G_ij · exp(qk · q_i·k_j) / Z_i^β           j ≤ i, and 0 above the diagonal
G_ij = ∏_{k=j+1..i} m_k · e^{iθ_k}                 the gate's path product, m_k ∈ [0, 1]
Z_i  = Σ_{j≤i} |G_ij| · exp(qk · q_i·k_j)          the row normalizer
```

| setting | `β` | `g` | `qk` | what the head computes |
|---|---|---|---|---|
| softmax attention | 1 | 0 | on | rows sum to 1 |
| linear attention | 0 | 0 | on | no normalizer |
| exact path product | 0 | any | off | `W = G` |

### What each switch decides

There is no single word for this head because there is no single axis. The
three switches are independent, and each one turns off a different faculty:

| switch | off | on | what it decides |
|---|---|---|---|
| `qk` | `W` ignores content | dot-product logits | **whether positions are compared at all.** With `qk` off the head is pure structure, and the content path is gone. |
| `g` | `G ≡ 1` | path product `∏ m_k e^{iθ_k}` | **whether values compose along a path.** A gate at zero closes the path exactly; Lean proves no prefix scan in the logit can do that, because `exp(C_i − C_j)` is never zero. |
| `β` | `β = 0`, no normalizer | `β = 1`, rows sum to 1 | **whether the read is a mean or a total.** The read carries `N^(1−β)` in the token count, so `β = 1` is intensive and `β = 0` is extensive &mdash; the vocabulary `ceqjepa/pi_assign.py` uses for its own verdicts. |

`β` is a separate axis from the Hodge **grade** of section 3, which is what the
read is *about* (nodes are grade 0, edge flows grade 1). `β` is how it
normalises. Conflating the two is the most common way to misread the family.

Each claim below is proved in Lean and checked in code:

| Lean theorem | what it proves | code check |
|---|---|---|
| `three_corners_containment` | one family `(β, g, qk)` contains all three operators above | `tests/arm_smprime/test_arm_smprime.py::test_bind2_softmax_corner_is_bitwise_against_the_statement_of_5a`, `::test_bind2_linear_corner_is_bitwise`, `::test_bind2_path_product_corner_is_bitwise` |
| `corners_are_distinct` | the corners are different operators: at `(i, j) = (1, 0)` with the gate and `qk` off, `β = 1` reads `1/2` and `β = 0` reads `1` | `tests/arm_smprime/test_arm_smprime.py::test_bind2_the_corners_are_numerically_distinct` |
| `gate_zero_beta_zero_is_linear_attention` | with the gate off, `β` alone decides softmax-class membership | `tests/arm_smprime/test_arm_smprime.py::test_bind3_beta_decides_softmax_class_membership_by_row_sum` |
| `no_prefix_scan_represents_a_zero_gate` | `exp(C_i − C_j)` is never zero, so no prefix scan `C` represents a closed gate, and the path product does | `tests/arm_smprime/test_arm_smprime.py::test_no_prefix_scan_represents_bedm_hop_and_the_product_route_does` |
| `Asink_computes_chain` | a causal softmax head reproduces a chain's label exactly, for every gate with `a_k ≠ 1` | `tests/arm_pl/test_arm_pl.py::test_bind2_the_oracle_setting_reproduces_the_chain_label` |

### What each one can reach

Columns are the operators **as defined in the corner table above**, not a survey
of the literature: gated and hybrid variants of linear attention exist and are
not what this table scores. Every tick has a proof or a test behind it, named in
the last column. The final row is the one that is not yet earned.

| capability | softmax | linear | resolvent today | planned | evidence for the tick |
|---|:--:|:--:|:--:|:--:|---|
| rows sum to one (intensive read) | ✓ | ✗ | **✓** | | `beta = 1` corner, bitwise against causal softmax, max abs `0.000e+00` |
| no normalizer (extensive read) | ✗ | ✓ | **✓** | | `beta = 0` corner, bitwise |
| both, as settings of **one** head | ✗ | ✗ | **✓** | | `three_corners_containment` |
| a gate that closes **exactly** | ✗ | ✗ | **✓** | | `no_prefix_scan_represents_a_zero_gate`: `exp(C_i - C_j)` is never zero; the path product reaches it |
| exact path product of a Markov chain | ✗ | ✗ | **✓** | | path-product corner, `qk` off, bitwise |
| which outcome is reached **first**, in closed form | ✗ | ✗ | **✓** | | committor by one triangular solve, max abs `4.441e-16` |
| exact `do(a)` as a rank-1 edit | ✗ | ✗ | **✓** | | Sherman-Morrison arm, `ceqjepa/hf/` |
| refuses when a counterfactual is undefined | ✗ | ✗ | **✓** | | 100.00% sensitivity, 100.00% specificity, 138 + 262 cases |
| edge flows with curl (grade 1) | ✗ | ✗ | **✓** | | pure curl left unrepresented by every node-level model, share `1.0000` |
| corners proved in a proof assistant | ✗ | ✗ | **✓** | | 134 theorems + 41 lemmas, 0 `sorry` |
| **beats a trivial baseline at next-state prediction** | ✗ | ✗ | **✗** | **planned** | not earned. [docs/PI_JEPA_KAGGLE_CARD.md](docs/PI_JEPA_KAGGLE_CARD.md), [docs/COMPONENT_LEDGER.md](docs/COMPONENT_LEDGER.md) |

Everything above the last row says what the operator can **represent**. The last
row says what it can **predict**, and it is open. That gap is the programme.

The theorems live in `lean/CEQ/V16Domain.lean` and `lean/CEQ/V15Fork.lean`;
`ceq/arm_pl.py` is the float64 counterpart of `V15Fork.lean`. The whole proof
tree is 13 files and builds clean:

```bash
python -m pytest tests/arm_smprime tests/arm_pl -q   # 69 passed
python scripts/lean_count.py                         # 13 files: 134 theorems + 41 lemmas = 175
grep -n sorry lean/CEQ/*.lean lean/CEQ.lean          # 7 hits, each the words "No `sorry`" in a doc comment
```

---

## 2. The resolvent read

`ceqjepa/operator.py` turns attention into a question about where a process
ends up. It builds a causal row-stochastic softmax matrix `P`, makes the
declared boundary rows absorbing, and reads with triangular solves:

```
state read    z = (I − gP)⁻¹ Ṽ
committor     q = (I − Q)⁻¹ R       probability of ending in each boundary state
```

With `teleport = 0` the operator matches causal softmax to `0.000e+00`. Two
refusals are built into the solve itself: a singular transient block raises
`SingularTransientBlockError` instead of returning a number, and a non-finite
logit raises `ValueError` instead of spreading NaN into `q`.

```bash
python -m ceqjepa.operator
```

```
(a1) SHIP corner, teleport=0, |A|=3, n=64: max abs diff vs causal softmax = 0.000e+00
(b) q_floor (constant encoder, general solve) vs closed form: max abs diff 4.441e-16
(c) committor() raised SingularTransientBlockError as required
(d) build_operator(NaN logits) raised ValueError as required
(e) kappa: stashed 5.232541  dense-inverse 5.232541  rel err 1.697e-16
ALL SELF-CHECKS PASSED
```

(Abridged to five checks, each cut after its first clause. The run is seeded
and repeats bit for bit.)

---

## 3. The graded read

Data is graded: nodes, edges (relations) and triangles (interactions). The
edge space splits three ways (Hodge decomposition):

```
R^E  =  im(d0)  ⊕  im(d1ᵀ)  ⊕  ker(L1)
        gradient    curl        harmonic
```

A model that reads an edge as a difference of node values lives in `im(d0)`,
so the curl part of a target is outside its range. `ceqjepa/dr1.py` measures
that boundary as the least-squares optimum over the whole node-level class, so
no architecture in the class does better:

| target on a 9-node, 16-edge, 8-triangle complex | share a node-level model cannot represent |
|---|---|
| pure gradient (the planted negative) | `6.5e-16` |
| mixed | `0.6720` |
| pure curl | `1.0000` |

The same module makes refusal an output. Every counterfactual gets one of
three verdicts: **UNDEFINED** is refused with its reason, **NULL** is answered
with an exact zero, and **DEFINED** is answered. Scored against an independent
reachability oracle, with both trivial rules beside it:

| rule | sensitivity | specificity |
|---|---|---|
| **resolvent (reachability)** | **100.00%** | **100.00%** |
| refuse everything | 100.00% | 0.00% |
| answer everything | 0.00% | 100.00% |

```bash
python -m ceqjepa.dr1   # 138 undefined and 262 defined cases; ALL SELF-CHECKS PASSED
```

---

## 4. Built to be audited

Every self-check in this repository carries a planted negative, a case that
must fail and is seen to fail, so a passing check shows the instrument can
tell the difference. The project also keeps its own error record in the open:

- **[MISTAKES.md](MISTAKES.md)** names 66 failure mechanisms the project has
  hit, each with the instance, the rule and the check that stops it recurring.
- **[STRUCK.md](STRUCK.md)** lists 12 withdrawn constants, rendered from a
  registry the test suite enforces.
- **[docs/FAILS.md](docs/FAILS.md)** is the research log: retractions,
  commands that do not run clean yet, and open questions.

```bash
grep -oE '\b[VPMD]-[0-9]+[a-z]?\b' MISTAKES.md | sort -u | wc -l   # 66
grep -c '^| `' STRUCK.md                                          # 12
```

---

## 5. What the failures built

Most of what is on this page exists because an earlier version of it failed
and the failure stayed on the record. Each row pairs a failure with what it
built.

| what failed | what it built |
|---|---|
| The original parity clause said a zero gate gives standard attention. Lean refuted it (`gate_zero_not_stochastic`), and round R11 refuted it three independent ways. | `gate_zero_beta_zero_is_linear_attention`: with the gate off, `β` is the switch that decides softmax membership. It became the `β` column of the corner table. |
| The prefix-scan theorem `prefix_logit_mask` holds only for gates `a_k > 0`, while the corpus draws its gate from `{−1, 0, +1}`. | The path-product hop and `no_prefix_scan_represents_a_zero_gate`, restated on the corpus's real support (`prefix_logit_mask_restated`, `0 ≤ m_k ≤ 1`). |
| A depth-separation claim (Theorem 4) and the matched-baseline comparisons ended in ties or losses: a resolvent read reparametrises a hypothesis class, it does not add one. | The graded read, which changes the class instead. Section 3 scores it in closed form against the optimum of the whole node-level class, with ridge reported first. |
| Smoothing the Sherman–Morrison denominator with a teleport kept the solve safe but shrank the interventional signal. | Refusal as a decision rather than a smoothing: the three-way UNDEFINED / NULL / DEFINED verdict in section 3. |
| A raw `grep` miscounted the Lean proofs in both directions. | `scripts/lean_count.py`, which must count a planted file correctly before it counts the tree. |
| An absence proof (`git log -S` across all refs) began finding the audit's own recordings. | Every absence search now ships with a control symbol the same search must find (`docs/canon/CORRECTIONS.md`, row C5). |
| The per-coordinate corner rule `β = 1 − α` assigned each latent coordinate its own corner from a measured exponent. On the encoder that exponent is zero by construction; on a null that enumerates all 35 arrangements with the refused coordinates pinned, the assignment ranked 21st; and on the value axis `α` moves with the `β` it is measured at, at slope `−1.0027`, so `β = 1 − α` reduces to `0 = 1 − α₀` and has no solution. | [docs/CORNER_RULE_RETIREMENT.md](docs/CORNER_RULE_RETIREMENT.md), and the three laws in [MISTAKES.md](MISTAKES.md) that the round paid for: **L-PROSE** (a number in prose carries its producer, and a report publishes its bound-over-reported ratio), **L-NULL** (a permutation names everything it varies and everything it pins), **L-SURFACE** (a check that prints and then aborts is a failed check). |
| A GPU run scored the model against human moves and lost to a zero-parameter heuristic. The deeper defect was the metric: top-1 next move is next-token prediction, which the project's own north star rules out in favour of the next state toward equilibrium. | [docs/PI_JEPA_KAGGLE_CARD.md](docs/PI_JEPA_KAGGLE_CARD.md), which opens on the heuristic winning, and [docs/COMPONENT_LEDGER.md](docs/COMPONENT_LEDGER.md), which gates joint training on each component clearing its own bar and fixes two columns every future run carries: a frozen-random arm, and a trivial baseline on the real target. |

The full record, with every number, is in [docs/FAILS.md](docs/FAILS.md),
[MISTAKES.md](MISTAKES.md) and [STRUCK.md](STRUCK.md).

---

## 6. Contributing

resolvent is young and there is real work to claim. Good places to start:

- **Put `#print axioms` on every Lean file.** Three of the 13 files carry it
  today (`V15Phase`, `V15Source`, `V16Domain`). The other ten are mechanical
  and make each theorem's axiom footprint visible.
- **Pin a working environment.** A torch/torchvision version mismatch breaks
  `import transformers` and blocks the Hugging Face integration tests in
  `tests/gate0`. A tested pin set or a lockfile unblocks them.
- **Give the suite its first full run.** `python -m pytest tests/ -q` is too
  slow to finish on one CPU. Marking slow tests or adding a CI matrix would
  produce the project's first total pass count.
- **Take the graded read to real data.** `ceqjepa/dr1.py` shows exactly where
  node-level models run out of range. A grade-1 learner on real relational
  data (traffic, markets, citation flows) is the open research direction.

House rules for a pull request: every new number ships with the command that
produces it, and every new check ships with a planted negative that fails
without the change.

---

## Environment

Measured 2026-09-11 on Windows 11, Python 3.11.9, torch 2.14.0 (CPU).
Lean 4 v4.7.0. mathlib is fetched by `lake build` at the revision pinned in
`lean/lake-manifest.json` (`a45ae63747140c1b2cbad9d46f518015c047047a`); the
first build downloads about 4.2 GB (`du -sh lean/.lake/packages`).

## Citation

```bibtex
@software{sharma2026resolvent,
  author = {Sharma, Teerth},
  title  = {resolvent: one causal attention family spanning softmax attention,
            linear attention and exact path products},
  year   = {2026},
  url    = {https://github.com/teerthsharma/resolvent}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
