---
license: apache-2.0
library_name: pytorch
tags:
  - causal-inference
  - causality
  - markov-chain
  - chess
  - interpretability
pipeline_tag: other
---

# DCM-1 — a causal operator with an exact do(a) arm

DCM-1 (Dirichlet Chart Machine) predicts **which outcome is reached first**, and
what happens to that prediction when an action is **forced**. The prediction is
not a logit head: it is the committor of a learned causal Markov operator, solved
exactly, and the intervention is a rank-1 edit of that operator solved by
Sherman-Morrison.

1,285,771 parameters. Loads with plain `torch` + `safetensors` — **no
`transformers`, no other repo**.

## Read this first

**The headline result on this checkpoint's own benchmark is an artifact, and
three controls say so.** The do-arm beat the ignore-the-intervention bar by
**-7.0703 ± 0.5428 PPL (-13.02 SE, 3/3 seeds)** and that number is *not* causal
evidence:

| control | measured | what it means |
|---|---|---|
| move-permutation ablation | **-0.0044 ± 0.0138** (-0.32 SE) | randomising *which* move was forced changes nothing — the advantage carries no information about the intervention |
| bed oracle headroom | **+0.0935 ± 0.0464** (wrong sign) | a brute-force oracle handed the true interventional distribution does no better; **no model of any size could win on this bed** |
| marginal control (training-set outcome frequency) | PPL_obs **2.8935**, PPL_do **2.5374** | a predictor that learned nothing beats this model on *both* arms (model: 10.3872 / 2.7875; chance = 4.00) |

`load_example.py` shows the mechanism directly: forcing a move shifts `q` by
about **0.25**, but *changing which move* shifts it by about **0.0037** — a **95.3x**
ratio. The do-path applies a near-constant shift. `mv_row.weight` was
**zero-initialised** for this run, so every candidate move produced the identical
clamped row at step 0 and the loss had no move-to-move signal to sharpen. A
5-seed placebo experiment measured the same thing independently: real forced move
+0.0235, *permuted* move +0.0243, move-specific part **-0.0008 ± 0.0056**.

**Do not cite this artifact as a demonstration of consequence understanding.**
What it does demonstrate is that the *machinery* is exact and the *evaluation*
is honest enough to kill its own headline.

## What is exact here

These are properties of the construction, verified by execution, and they hold
regardless of what training did:

- **`q = (I - Q)^{-1} R` is a genuine probability.** Not a normalised score:
  it attains 0 and 1 exactly, and every row sums to 1 (measured `max|sum-1| =
  5.960e-07` in float32). One lower-triangular solve.
- **`do(a)` is a rank-1 edit, not a re-run.** `committor_do_batch` covers m
  candidate actions from **one** factorisation — measured **21.65x** faster than
  m separate full solves at n=512, m=32. Agreement with a full recompute:
  `max|diff| = 1.110e-16`.
- **The guard raises instead of lying.** The Sherman-Morrison denominator equals
  `(1 - P'_ii)/(1 - P_ii)` (verified against the closed form at rel err
  1.138e-16). Below `sqrt(eps)` the code raises rather than returning a
  committor amplified by `1/den`. **0 refusals in 384,000 training interventions
  and 0/1500 at eval** — because `teleport=0.0125` bounds `den >= 0.0125`
  against `den_min = 3.453e-04`, a 36.2x margin, so during training the guard is
  provably unreachable; it fires at `teleport=0`.
- **Conditioning is bounded by construction.** `teleport=c` gives
  `||(I-Q)^{-1}||_inf <= 1/c = 80` (realized `80*(1+6e-6)` in float32). Measured
  on the example boards: **5.62**.
- **At `teleport=0` the `g=0` read is the causal softmax bitwise** (max abs diff
  `0.000e+00`).

The vendored `modeling_dcm1.py` reproduces the training package **bitwise** on
these weights — `q_alpha`, `q_field`, `P`, `alpha`, `x_hat`, `logits` and
`do_read` all at `max|diff| = 0.000e+00`, `torch.equal` True.

## Usage

```bash
pip install -r requirements.txt
python load_example.py
```

```python
import torch
from modeling_dcm1 import DCM1Model, fen_to_vec, square

model = DCM1Model.from_pretrained(".")            # strict=True, zero missing/unexpected keys
x = fen_to_vec("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unsqueeze(0)

out = model(x)
out["q_alpha"]        # [B, 4] P(white_win, draw, black_win, sink) hit FIRST
out["q_field"]        # [B, 256, 4] the same read at every chart position
out["P"]              # [B, 256, 256] the causal row-stochastic operator

moves = torch.tensor([[[square("e2"), square("e4")]]])          # [B, m, 2]
q_do, q_base, i_star, ok = model.do_read(out, moves)
q_do[0, 0] - q_base[0]   # the model's stated CONSEQUENCE of forcing e2e4
```

`ok[b]` is False where Sherman-Morrison refused; drop those rows, never read
them as a value.

**Input format.** 769 floats: 12 piece planes × 64 squares (`PNBRQKpnbrqk`
order, a1=0 … h8=63), then side-to-move (1.0 = white). `fen_to_vec` builds it
from a FEN string with no `python-chess` dependency.

## Architecture

| | |
|---|---|
| chart positions `n` | 256 |
| absorbing sets `nA` | 4 — `white_win`, `draw`, `black_win`, `sink` (identity rows of P) |
| encoder | 769 → 128 → GELU → 128 |
| per-example operator | `L0 + a @ b.T`, rank 16 (LoRA-style factored modulation) |
| state solve | `z = (I - g P)^{-1} V`, g = 0.9, one triangular solve |
| do(a) head | 64+64 square embeddings (d=8) → additive bias on row `i` |
| parameters | 1,285,771 |

The intervention lands at `i* = nA + argmax(alpha[nA:])` — the transient chart
position the model's own read weights most. Clamping a declared-absorbing row is
refused: it would move `i` from A to T and change the shape of `q`.

## Training

Kaggle T4, `torch 2.10.0+cu128`, kernel `melowdramtic/ceq-jepa-dcm-1-causal-arm-t4`,
17,002 s total, one attempt.

- 5,000 paired chess positions (seed 0, m=8, R=4, max_plies=400); 1,500 held out
  (seed 12345, m=1, R=1, one-hot do-label)
- 12,000 steps, batch 32, lr 3e-4, Adam; `lambda_do=1.0`, `lambda_z=0.0`,
  `lambda_topo=0.0`
- `L = L_q + lambda_do * L_do`, both clamped BCE against the committor read
- this repo packages **model seed 0**, the best of three on both arms
- 1,816.7 s of training wall clock; the 5,000-position bed build was 10,304 s (61%)

`L_do` at init is **exactly 2.249340** for any model — with `L0 = 0` and a
zero-initialised delta the operator is the uniform causal chain and all four
absorbing sets are symmetric, so `q = 0.25` everywhere and the BCE is
`-(ln 0.25 + 3 ln 0.75)`. Only its descent carries information; the starting
value does not.

## Limits

`lambda_z` is 0 because the encoder-side auxiliary was measured to **hurt** the
committor read (L_q alone +0.5056 vs L_q + 0.5 L_z +0.4070, 3/3 seeds). `gamma`
is not a parameter — `q` contains no `g` and is bit-identical at
g ∈ {0.0, 0.5, 0.9, 0.99}. The `do` label is an empirical mean over R=4 rollouts,
so per-coordinate std reaches 0.25 and `L_do` can never reach 0; its absolute
scale is not comparable to `L_q`'s. Promotion piece is dropped from the move
encoding, so `e7e8q` and `e7e8n` share an embedding. The observational arm
**overfits to worse than chance** — held-out PPL_obs rises 3.9975 → 10.3872 while
train `L_q` falls 2.2491 → 0.1415, and ECE is 0.2301 with 1,120/6,000 points in
the top confidence bin at empirical 0.4795. The three checkpoints were never
reloaded and re-scored against the numbers reported for them. The evaluation bed
uses uniform-random self-play rollouts, under which forcing one ply moves the
outcome distribution by 0.0033 TV above a 0.1468 noise floor — that, not the
model, is why the causal claim could not be tested.

## Citation of prior art

- LoRA-style factored update, and the one-factor-zero initialisation:
  Hu et al., *LoRA*, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Committor functions and absorbing Markov chains: standard potential theory;
  `(I-Q)^{-1}R` is the classic absorption-probability solve.
- Sherman-Morrison rank-1 inverse update: Sherman & Morrison (1950).
- JEPA framing (predict a representation, not pixels): LeCun, *A Path Towards
  Autonomous Machine Intelligence* (2022).
- Differentiable topological terms used elsewhere in this project (not in this
  checkpoint, `lambda_topo=0`): Carriere et al.
  [arXiv:2010.08356](https://arxiv.org/abs/2010.08356); Hu et al.
  [arXiv:1910.01877](https://arxiv.org/abs/1910.01877); Moor et al.,
  *Topological Autoencoders*, PMLR v119.
- `python-chess` supplied the legal-move oracle and self-play during bed
  construction (not needed to load this artifact).

## Files

| file | |
|---|---|
| `config.json` | geometry, training budget, held-out numbers, the adverse note |
| `model.safetensors` | 1,285,771 params, float32, from `ceqjepa_causal_seed0.pt` |
| `modeling_dcm1.py` | the whole model, torch only, operator + do-arm vendored |
| `load_example.py` | strict-load check, both reads, the ratio that exposes the defect |
| `requirements.txt` | `torch`, `safetensors` |
