# ceqjepa — DCM-1 Prototype

## Subsystem

The `ceqjepa` module implements the primitives of DCM-1 (Dirichlet Chart Machine): a causal absorbing-chain operator and a training harness that supervises a committor head against exact float64 labels. Two files: `operator.py`, `train.py`.

## Provenance

The package is **untracked**. `git ls-files ceqjepa` returns empty (0 files) against the repo's current HEAD, commit `527ffca66be42c744c45b4229b4c085087d00dec` — checked live, both commands, today. Nothing under `ceqjepa/` has ever been committed, so `527ffca` is not this package's provenance; it is only the repo's HEAD at the time of writing. Every number below was produced by running the code on disk (torch `2.14.0+cpu`, numpy `2.4.6`, live-checked), not by inspecting a commit.

## ceqjepa/operator.py

A torch-only functional module (`import torch` only — no numpy, no transformers, no torchvision).

**Public API** (`__all__`, read from the file today):
- `causal_mask(n, device=None)` — `[n,n]` bool, lower-triangular-inclusive (`j <= i`)
- `build_operator(logits, absorbing_idx, teleport=TELEPORT)` — `P = softmax(logits)` under the causal mask, with a `teleport` fraction of every transient row's mass moved onto the causally-visible absorbing states before the boundary rows are overwritten to identity rows (`P[a,:] = e_a` for `a` in `absorbing_idx`). Raises `ValueError` on non-finite logits.
- `teleport_at(step, total_steps, start=TELEPORT, end=0.0)` — linear anneal of the teleport fraction, `start` at step 0 down to `end` by `total_steps`
- `state_solve(P, V, g)` — `z = (I - g*P)^{-1} V` via `torch.linalg.solve_triangular` (exact, since `P` is causal); `O = (1-g)*P@z`
- `committor(P, absorbing_idx, tol=None, kappa_max=None)` — splits `P` into `Q = P[T,T]`, `R = P[T,A]` and returns `q = (I-Q)^{-1} R` embedded to `[...,n,k]`; `tol=None` picks a dtype- and size-aware default (`max(1e-10, n_transient * finfo(dtype).eps)`) rather than a fixed `1e-10`; raises `SingularTransientBlockError` if the transient diagonal has an entry below `tol`, or (when `kappa_max` is given) if the exact `||(I-Q)^{-1}||_inf` exceeds it. Raises `ValueError` on non-finite `P`. Journals `committor.last_kappa_bound` as the exact norm (one extra triangular solve against the ones vector, computed detached under `no_grad`), not a lower-bound proxy.
- `q_floor_closed_form(n, absorbing_idx, dtype=torch.float64, device=None, teleport=TELEPORT)` — the collapse floor (committor field of the uniform causal chain) by a forward recursion, independent of `committor()`
- `SingularTransientBlockError` — raised when the transient block is singular or (with `kappa_max` set) too ill-conditioned
- `TELEPORT = 0.0125` — the training-time teleport floor
- `KAPPA_DESIGN_BOUND_F32 = 80.00048` (`80*(1+6e-6)`) — the float32-realized conditioning bound at `TELEPORT`; use this in asserts, not a bare `80.0`

**The teleport is a training scaffold, not part of the read.** At `teleport=0` (the ship/eval setting) `build_operator` returns the causal softmax bitwise — this is DCM-1's load-bearing corner. At `teleport=TELEPORT` (the training default) every transient row is displaced by up to `2*TELEPORT = 0.025` in row-L1, a displacement that does **not** shrink with chart size `n`; in exchange, the transient block is structurally non-singular during training (`1 - Q_ii >= TELEPORT` always), which is the fix for the crash described under Control Run below.

**Self-check** (`python ceqjepa/operator.py`, run today, verbatim):
```
(a1) SHIP corner, teleport=0, |A|=3, n=64: max abs diff vs causal softmax = 0.000e+00
(a2) TRAIN corner, teleport=0.0125: max row L1 displacement = 0.024682 (stated bound 2c = 0.025); transient rows moved 61/61, absorbing rows moved 0
(b) q_floor (constant encoder, general solve) vs closed form: max abs diff 4.441e-16
(c) committor() raised SingularTransientBlockError as required: transient block singular (I - Q has a diagonal entry below 1.000e-10 in torch.float64): some causally-self-only state is not in the declared absorbing set, or an annealed teleport has let a row saturate. With build_operator(teleport=c>0) this is unreachable -- a teleport of c bounds every transient diagonal below by c.
(c) q_floor_closed_form() raised SingularTransientBlockError as required: index 0 is not in the declared absorbing set: state 0 is always self-absorbing under a causal mask (only j <= 0 exists)
(d) build_operator(NaN logits) raised ValueError as required: build_operator: logits contain 1 non-finite entries (NaN/Inf); refusing to build P, because softmax would propagate NaN into q and into the conditioning number, where no threshold check catches it ((x < tol) is False for NaN)
(d) committor(NaN P) raised ValueError as required: committor: P contains 1 non-finite entries (NaN/Inf); refusing to solve, because the singularity test (diag.abs() < tol).any() is False for NaN, so a NaN P returns a full-NaN q and a NaN kappa with nothing raised
(e) kappa: stashed 5.232541  dense-inverse 5.232541  rel err 1.697e-16  |  the old diagonal quantity 2.467361 understates it by 2.12x
(f) kappa_max guard fired as required: transient block ill-conditioned: ||(I-Q)^{-1}||_inf = 5
(f) kappa journal under requires_grad=True: 0 warnings, kappa=4.247637, grad finite=True
(bonus) state_solve: z, O finite, correct shape. route=solve_triangular, delta=0.0
ALL SELF-CHECKS PASSED
```
Exit code 0. Checks (a1)/(a2) verify the teleport is a live switch (a prior version of this check used an empty absorbing set, which made the teleport branch unreachable and the corner test unable to fail no matter what the teleport did); (a2) verifies displacement stays inside the stated `2c` bound, not the corner's magnitude beyond that. Check (a2)'s displacement bound and check (b)'s closed-form agreement are the two independent things that would catch a broken teleport constant.

## ceqjepa/train.py

A training harness: a synthetic causal absorbing-chain corpus (`Bed`, generated on-the-fly per batch — there is no staged corpus file or fixed train/heldout/validation split on disk) and a model (`TinyCEQ`) trained against exact float64 committor labels.

**`Bed(n, nA, x_dim, z_dim, seed=0, dtype=torch.float64)`** — base tensors `L_env [n,n]`, `D [z_dim,n,n]`, `W_obs [x_dim,z_dim]`, `phi [n,x_dim]` fixed once at construction (seed 0). `Bed.batch(gen, B)` draws fresh latents each call, builds `logits = L_env + einsum('bm,mij->bij', u, D)`, solves the committor once per batch via `op.build_operator` / `op.committor`, and returns `(x, x_nx, q_star, v_idx)`. `Bed.q_floor()` returns `op.q_floor_closed_form(n, absorbing_idx, dtype=torch.float32)`.

**`TinyCEQ`** — encoder `e = MLP(x)`; a rank-`r` per-example modulation `delta_logits = einsum('bnr,bmr->bnm', delta_a(e), delta_b(e))` added to a learned base `L0` (zero-initialized, so `delta=0` at step 0 — see D6 below); `P = op.build_operator(logits, absorbing_idx)` at the **default** teleport (`TELEPORT=0.0125`) in both training and eval — `forward()` never passes `teleport=0`, so the bitwise ship corner described above is never exercised by this script (see Limits).

**Loss** (stage 1 only): `L_q` = BCE of the softmax-attention-weighted committor read `q_alpha` against `q_star`, clamped to `[1e-6, 1-1e-6]`; `L_z` = MSE of the forward-model output against the next observation; `L = L_q + 0.5*L_z`.

**Eval metric**: `S = 1 - mse_model/mse_bar` against the constant-predictor control `q_bar` (training-set channel mean); `collapse_floor = ||q_alpha - q_floor||_inf`; `var_across_examples` = `Var[e]` over the **batch dimension** (`dim=0`) of the encoder output `e`, averaged over channels — this, not `collapse_floor`, is the detector for an encoder that ignores its input (see Verification: Collapse Floor).

**Command-line interface** (`python -m ceqjepa.train --help`, current flags):
```
--steps N (200)            --seed S (0)                --device (cpu)
--out PATH (ceqjepa_checkpoint.pt)
--max-seconds SEC          --geometry {tiny,design} (tiny)
--n N (16)                 --nA K (4)                  --d-enc D (16)
--x-dim X (8)               --z-dim Z (4)               --z-dim-state ZS (6)
--g GAMMA (0.9)             --rank R (8)                --batch-size B (16)
--eval-every E (1)          --eval-n EN (32)            --lr LR (3e-4)
--ckpt-every N (0)          --kappa-ceiling K (80.0)
```
`--geometry design` sets `n=256, d-enc=128` (the DCM-1 frozen chart size), other dims stay at their flags. `--ckpt-every` writes an atomic checkpoint (`.tmp` + `os.replace`) every N steps in addition to the end-of-run write, so a killed run leaves usable weights. `--kappa-ceiling` prints a warning (never raises) when `committor.last_kappa_bound` exceeds it.

**Tiny-geometry example, run today:**
```bash
python -m ceqjepa.train --steps 40 --eval-every 10 --eval-n 64 --batch-size 32 \
  --n 16 --nA 4 --d-enc 16 --x-dim 8 --z-dim 4 --z-dim-state 6 --out ceqjepa_checkpoint.pt
```
```
[ceqjepa.train] n=16 nA=4 d_enc=16 x_dim=8 z_dim=4 z_dim_state=6 g=0.9 params=5480
[ceqjepa.train] q_bar=[0.1565, 0.2206, 0.3078, 0.315]
[eval] step=  10 train_loss=2.4486 S(vs const)=-0.2171 mse_model=1.7826e-02 mse_bar=1.4646e-02 collapse_floor=3.5538e-02 var_across_examples(enc_output,batch_dim)=1.8077e-02 kappa_bound=2.4521
[eval] step=  20 train_loss=2.4610 S(vs const)=-0.2702 mse_model=1.9385e-02 mse_bar=1.5261e-02 collapse_floor=3.6814e-02 var_across_examples(enc_output,batch_dim)=1.8359e-02 kappa_bound=2.4523
[eval] step=  30 train_loss=2.4412 S(vs const)=-0.0951 mse_model=2.3271e-02 mse_bar=2.1250e-02 collapse_floor=4.1297e-02 var_across_examples(enc_output,batch_dim)=1.6658e-02 kappa_bound=2.4525
[eval] step=  40 train_loss=2.4153 S(vs const)=-0.1576 mse_model=2.1606e-02 mse_bar=1.8665e-02 collapse_floor=3.7186e-02 var_across_examples(enc_output,batch_dim)=1.6600e-02 kappa_bound=2.4526
[ceqjepa.train] wrote checkpoint ceqjepa_checkpoint.pt
```
40 steps is too short to show a positive `S` reliably (see Control Run below for a longer, warmed-up run).

**Design-geometry, run today** (`--geometry design --steps 1`):
```
[ceqjepa.train] n=256 nA=4 d_enc=128 x_dim=8 z_dim=4 z_dim_state=6 g=0.9 params=646232
[ceqjepa.train] geometry=design target_params~325792 actual_params=646232 (D6 low-rank delta_logits, rank=8: delta_a+delta_b cost 2*d_enc*n*rank=524288 vs dense d_enc*n*n=8388608 that the pre-fix per-example modulation cost -- did NOT hit the target exactly, reporting actual)
```
646,232 params against the ~325,792 target: 2.0x over, not hit exactly (the print says so honestly). Before the rank-8 factorization this was 8,571,992 (26.3x over) because the per-example modulation was a dense `nn.Linear(d_enc, n*n)`.

**Checkpoint contents** (`torch.load` on a file written today, keys read live): `model_state_dict`, `geometry` (the full `argparse` namespace), `n_params`, `q_bar`, `final_eval`, `history`, `wall_s`. There is no git SHA and no torch-version field in the checkpoint — any claim that one is written there is false; if that provenance matters, capture it separately when a checkpoint is produced.

## Verification: Collapse Floor

`collapse_floor = ||q_alpha - q_floor||_inf`, where `q_floor` is the committor of the uniform causal chain, is monotone in **logit scale**. That is the same thing as encoder health only for collapse *across positions within one example* — it does not detect an encoder that ignores its input `x` entirely while its output still varies across positions (a batch-independent chart). For that failure mode, `var_across_examples` (`Var[e]` over the batch dimension, defined above) is the actual detector: a batch-independent encoder reads exactly 0 there by construction, regardless of what `collapse_floor` says. This scope limit is stated once, here, and does not get restated as a stronger claim ("verified to be monotone in encoder health") anywhere else in this file.

**Setup** (run today, `torch 2.14.0+cpu`, `numpy 2.4.6`, package untracked at HEAD `527ffca`): chart `n=64`, absorbing set `{0,40,50,60}`, `teleport=0` (the ship corner, so this measures the architecture's read, not the training scaffold). Encoder logits `H @ H.T / sqrt(16)` with `H = broadcast(h) + t*noise`, `h,noise ~ N(0,1)`; `t=0` gives every row of `H` identical (the uniform causal chain), `t=1.0` is a fully healthy encoder.

```
=== CFD vs logit-scale t, seed 0 ===
  t=0      CFD = 0.000000
  t=0.001  CFD = 0.000039
  t=0.1    CFD = 0.003950
  t=1      CFD = 0.078991
```
Monotone across 20 seeds (`t=0` fixed at `0.000000`; at `t=1.0`, min `0.031811`, median `0.078991`, max `0.320129` — every seed's own `t=0 <= t=1e-3 <= t=0.1 <= t=1.0`, and no seed's `t=1e-3` rung reaches into another seed's `t=0.1` rung or vice versa).

**Scope-limit demonstration**, same setup, replacing the random encoder with a fixed `[n,n]` logit pattern that has no dependence on any per-example input at all (so it is emitted identically for every example — `var_across_examples` on the encoder driving it is exactly 0), but still varies across chart positions:
```
CFD(batch-independent, position-varying encoder) = 0.044821
t=1.0 (healthy) band across 20 seeds: min 0.031811  median 0.078991  max 0.320129
inside healthy band: True
```
A collapsed encoder (one that never looks at `x`) can read as squarely healthy on `collapse_floor` alone. `var_across_examples` catches it because it is 0 for every batch built from this encoder, by construction.

## Verification: Control Run

**Setup**: `cpu`, `seed=0`, tiny geometry (`n=16, nA=4, d_enc=16`), `batch_size=16`, `--max-seconds 60`, run today:
```bash
python -m ceqjepa.train --steps 1000000 --seed 0 --max-seconds 60 --eval-every 2000 --eval-n 256 --batch-size 16 --out ceqjepa_checkpoint.pt
```
```
[ceqjepa.train] n=16 nA=4 d_enc=16 x_dim=8 z_dim=4 z_dim_state=6 g=0.9 params=5480
[ceqjepa.train] q_bar=[0.1565, 0.2206, 0.3078, 0.315]
[eval] step=2000 train_loss=2.1899 S(vs const)=+0.3829 mse_model=1.1571e-02 mse_bar=1.8749e-02 collapse_floor=5.0146e-01 var_across_examples(enc_output,batch_dim)=1.0186e-01 kappa_bound=2.4392
[eval] step=4000 train_loss=2.1219 S(vs const)=+0.3032 mse_model=1.1653e-02 mse_bar=1.6723e-02 collapse_floor=4.6590e-01 var_across_examples(enc_output,batch_dim)=1.0955e-01 kappa_bound=2.6934
[eval] step=6000 train_loss=2.1491 S(vs const)=+0.2835 mse_model=1.2716e-02 mse_bar=1.7747e-02 collapse_floor=3.4296e-01 var_across_examples(enc_output,batch_dim)=9.8863e-02 kappa_bound=3.1964
[eval] step=8000 train_loss=2.2382 S(vs const)=+0.3857 mse_model=1.1226e-02 mse_bar=1.8276e-02 collapse_floor=3.9477e-01 var_across_examples(enc_output,batch_dim)=1.3333e-01 kappa_bound=4.1654
[ceqjepa.train] stopping at step=8064: max-seconds=60.0 exceeded
[ceqjepa.train] wrote checkpoint ceqjepa_checkpoint.pt
```
8,064 steps in 60 seconds, zero crashes, `S` positive at every eval (+0.28 to +0.39), `kappa_bound` rising slowly (2.44 → 4.17) and nowhere near the 80 design ceiling. This does **not** reach scale — tiny geometry only.

An earlier version of this file reported a run that crashed at ≈20,400 steps with `SingularTransientBlockError` because `train.py`'s operator calls had no teleport floor: a transient row could saturate its causal-softmax diagonal to exactly 1.0, making `I - Q` singular. That crash mode is now structurally closed during training: `build_operator`'s default `teleport=TELEPORT=0.0125` keeps every transient diagonal entry `>= TELEPORT`, so `I - Q` cannot reach exact singularity from logit growth alone. `train.py` still has no `except SingularTransientBlockError` anywhere and never arms `committor(..., kappa_max=...)`, so a run whose exact conditioning climbs past what the teleport bounds (`<= 80*(1+6e-6)` in float32, see operator.py) is not caught explicitly — it would currently surface as a print (`--kappa-ceiling`, warning only) or, past `1/TELEPORT`, as an uncaught `SingularTransientBlockError` from the kappa side if `kappa_max` were ever armed, which it is not.

A separate, longer soak exists on disk today at `ceqjepa/artifacts/dcm1_design.pt` (design geometry, `n=256, d_enc=128`, 646,232 params) — read live for this section: 1,750 steps, 87 evals, `wall_s=817.2`. `S` was negative for the first 5 evals (cold start, step 20–100), crossed positive at step 120, and stayed positive for 82/87 evals thereafter (min after warm-up `+0.1171`, max `+0.8065`, final eval `S=+0.4686` at step 1740, `mse_model=3.2897e-03` vs `mse_bar=6.1907e-03`). Max `kappa_bound` over the whole run: `5.2555` — far under the 80 ceiling, because this run trained at the fixed default teleport the whole time; `operator.teleport_at` is defined but has zero callers anywhere in `train.py`, so the anneal toward `teleport=0` described in operator.py is not exercised by any run on disk, including this one.

## Limits

- **Package is untracked (D11).** `git ls-files ceqjepa` is empty at HEAD `527ffca`. Nothing here has a commit to point to.
- **`train.py` never evaluates at `teleport=0`.** `TinyCEQ.forward()` calls `op.build_operator(logits, absorbing_idx)` with the default `TELEPORT=0.0125` in both the training step and inside `evaluate()` (which just calls the same `forward()` in eval mode). The bitwise causal-softmax corner verified in operator.py's self-check is a property of the operator in isolation; no run in this file, and no checkpoint on disk, has ever been read at that corner.
- **The teleport anneal (`operator.teleport_at`) has no caller.** `train.py` takes no teleport-schedule argument; every run trains and evaluates at the constant `TELEPORT=0.0125`.
- **Design geometry misses its param target.** `--geometry design` reports 646,232 actual parameters against the ~325,792 DCM-1 target (2.0x over), from the rank-8 per-example modulation; the script reports this honestly rather than silently hitting the target. A smaller `--rank` would move it closer.
- **No exception handling for `SingularTransientBlockError` or `kappa_max` guard armed in `train.py`.** The default teleport keeps the diagonal guard structurally unreachable during training, but nothing in this file catches the error or arms `committor(..., kappa_max=...)`, so a conditioning excursion beyond the teleport's own guarantee, or a future change that runs training at `teleport=0`, would propagate an uncaught exception.
- **`collapse_floor` cannot see batch-independent collapse.** It is monotone in logit scale, not in "does the encoder look at its input" — see Verification: Collapse Floor above. `var_across_examples` is the metric that catches that failure mode; both are printed every eval, but only one detects it.
- **No stage-2 loss or refusal machinery.** Stage 1 only (committor and forward-model supervision) is implemented. Stage 2 (decision head, hinge loss, refusal by margin/reducibility/route) and its controls are not built.
- **Checkpoints carry no provenance fields.** No git SHA, no torch version, no dependency manifest is written into `--out`; only `model_state_dict`, `geometry`, `n_params`, `q_bar`, `final_eval`, `history`, `wall_s`.
