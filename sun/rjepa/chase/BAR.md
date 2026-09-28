# R-JEPA / Chase: registered bars and kills (written 2026-09-28 ~00:51 IST by shell clock, before any run)

## Object under test

`set_resolvent(x, mask)`: the repo resolvent (`ceq/attention.py::ceq_operator` + `path_sum`) moved from the
time axis to the candidate axis of a D-JEPA-style decision. Input: K candidate feature rows (goal-relative
predicted latent, ordinal evidence). A = rho * offdiag(q k^T) / rowL1, out = (I - A)^-1 v, score = w^T out.

## "Fastest", falsifiable form

F1. Unit = one decision = K candidates x H-step latent rollout + ranking. Report (i) analytic FLOPs per decision
    `K*H*F_pred + F_rank(K)`, (ii) wall-clock per decision, median of >= 30 timed reps after 10 warm-up reps,
    with IQR and the `nvidia-smi` load at the time.
F2. Speed is only compared between arms at MATCHED ranking quality: an arm's mean normalized top-1 regret
    (3 seeds) within 0.01 absolute of the best arm. An arm outside that is not "fast", it is "worse".
F3. Comparators: (a) DeepSets and a 1-layer set transformer (SDPA, no mask) at params within +-10% of the
    resolvent arm, (b) latent-distance ranking (0 params, the floor and D-JEPA's baseline), (c) SDPA itself
    as the mixer cost.
F4. "Fastest" survives only if the resolvent arm is at matched quality AND its per-decision wall-clock is
    <= every matched-quality comparator's.

## Kills (each fires on the stated measurement; replacement route written at registration)

K-AMDAHL  Ranker share of per-decision cost < 5 % at K=16, H=8 with the registered toy predictor
          (4-layer MLP, width 256, latent 64 -- deliberately SMALLER than any D-JEPA predictor, so the share is
          overstated). Fires => ranker speed cannot make the model fastest. Route: restate "fastest" as fewest
          predictor calls (K*H) per decision at matched success -- prune K or truncate H using the ranker.
K-SCALE   Exact solve cost vs K: fitted exponent of resolvent time over K in {16,64,256,1024} exceeds the
          SDPA set layer's by > 0.5, or resolvent > SDPA layer at K=256. Route: Neumann path sum at hops h
          (cost h*K^2*d, error <= rho^(h+1)/(1-rho)) with h registered from the bound.
K-FLOOR   Resolvent regret not below the latent-distance floor by >= 0.02 on 3/3 seeds. Route: retire the
          resolvent as ranker; keep latent distance (it is cheaper and equal).
K-MATCH   Resolvent regret worse than DeepSets or set transformer by > 0.01 (mean, 3 seeds). Route: "fastest"
          dies under F2; report the resolvent as "cheaper, worse" with the gap.
K-BED     Bed void if the floor's regret < 0.10 (nothing to fix) or oracle regret != 0.

## Ranking bed (registered)

d_z = 16 latent, K = 16, goal g ~ N(0, I). True terminals z_k ~ g + N(0, 1.5^2 I). Predictor error
e_k = s * tanh(B a_k) with a_k ~ N(0, I_8) observed action evidence, B fixed per bed (seed 1234), s = 0.8.
Observed per candidate: x_k = [zhat_k - g, a_k], zhat_k = z_k + e_k. Outcome y_k = -||z_k - g||.
Loss: cross-entropy over candidates with target argmax y. Train 20k decisions, eval 4k, 3 seeds (0,1,2),
Adam 3e-3, 3000 steps, batch 256, cpu. Normalized regret = (y_max - y_chosen)/(y_max - mean y).

## Correctness contracts (tda-tdd), each a RED test first

C1 permutation equivariance over candidates (float64, atol 1e-10).  C2 pole-freedom: rho >= 1 refused;
spectral radius of A < 1 on 1000 random draws; ||(I-A)^-1||_inf <= 1/(1-rho).  C3 K=1 and all-masked rows:
finite, and a masked candidate neither moves others nor gets rank above any unmasked.  C4 dense parity:
fp32 fast path vs float64 exact solve, rel err <= 1e-5; Neumann h vs exact within the stated bound.
C5 dtype walls: fp16/bf16 inputs; overflow of q k^T; rank flips vs float64.  C6 gradcheck float64.
C7 determinism (two calls bitwise).  C8 tile-boundary K in {1,2,3,7,8,9,15,16,17,31,32,33,63,64,65}.

## Amendment A1 (~00:55 IST by shell clock, before any bench run): D-JEPA opponent from Wilson's spec (sun/rjepa/wilson/facts.json)

D-JEPA code exists (github.com/NEBULIS-Lab/D-JEPA, Apache-2.0) and was NOT fetched; the opponent is reimplemented
from the paper spec relayed by Wilson. Single latent source (T half dropped). Token per candidate
[LN(zhat_i - g); a_i; r_i], r_i = (rank(||zhat_i - g||) - 1)/(K - 1); base b_i = r_i; encoder Linear->64, LN, GELU;
2 pre-norm layers, 4 heads, FF 128, no candidate position; head delta_i = 0.2 tanh(W_up tanh(W_down h_i)), 64->8->1,
W_up zero-init; s_i = b_i + delta_i, pick argmin s. Loss: CE on softmax(-s/0.05) to the best candidate
+ 0.25 mean softplus((0.02 + s_best - s_j)/0.05) + 0.1 mean delta^2. Recipe: AdamW 3e-4, wd 1e-4, clip 1.0.
Steps: 1000 (paper) AND 3000 (registered bed budget); both reported, verdicts read at 3000.

Arms sharing that exact shell, mixer the only free variable:
  djepa        softmax SDPA mixer (the opponent)
  rjepa_signed set_resolvent per head, rho = 0.9
  rjepa_stoch  stochastic_resolvent per head, g = 0.9 (Wilson's repo form)
K-MATCH and K-FLOOR are read on these three against floor_dist. The original Arm set (pointwise, deepsets,
settf, resolvent) is kept as a second, recipe-independent read.
Timing adds the same three arms. Extra registered read: bf16 argmax flip rate vs float64 on eval, for the
resolvent AND the settf control (a flip rate is meaningless without the control's).
