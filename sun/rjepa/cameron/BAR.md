# R-JEPA / Cameron bar — spread-aware ranking vs a D-JEPA-style operator (registered 2026-09-28, before any run)

Question: is there a way to rank K candidate futures that is cheaper than a learned D-JEPA-style
operator (arXiv 2609.24749, Liu et al.) and at least as good at picking the candidate that actually
succeeds? Every run records the sha256 of this file.

## Bed (exact truth, exact Bayes ceiling)

- World: Chirikov standard map on the torus T^2 = [0, 2pi)^2, p' = p + Ks sin(theta), theta' = theta + p',
  both mod 2pi, Ks = 1.5. Area-preserving, so the uniform starts are the invariant measure and the flat
  posterior is exact.
- Start x0 ~ U(T^2). Observation y = x0 + delta*xi (wrapped), delta = 1e-3.
- K = 8 candidates per start: impulse u_k ~ U([-1, 1]^2), applied at t = 0, then T map steps.
- Goal g = (0.5, 1.0); success_k = torus distance(F^T(x0 + u_k), g) < r, r = sqrt(pi/2) (area 1/K).
- Leads T in {2, 4, 6, 8, 10, 12, 16, 20, 24, 32}. Seeds {0, 1, 2}. N_train = 20,000, N_eval = 10,000 starts per lead per seed.
- Score of a policy = success rate of the picked candidate on the true x0.

## Floors and ceiling (run first)

- `blind`: random pick; its success is computed exactly as mean_k success_k.
- `floor` (latent distance): argmin_k dist(F^T(y + u_k), g). One predictor call per candidate.
- `bayes`: M = 256 posterior members x_m = y + delta*xi_m; argmax_k of the in-ball count, ties to `floor`.
- `oracle`: max_k success_k (any candidate succeeds).
- Normalised score NS = (s - blind) / (bayes - blind).

## Arms

- `DJ` (D-JEPA-style): per-candidate goal-relative features (sin/cos of z_k - g, d_k/r, sin/cos of y, u_k),
  shared MLP, mean-pooled set context (permutation-equivariant), score_k = -d_k/r + B*tanh(.), B = 2 (bounded
  correction on the latent distance). Trained listwise (softmax CE onto the executed successes) on N_train starts per lead.
- `DJs`: DJ plus one feature, log of the linearised spread sigma_k (below). Same width.
- `LIN` (Cameron, 0 parameters, no training): finite-difference Jacobian J_k of F^T at y + u_k (3 predictor calls
  per candidate); predictive ~ N(z_k, delta^2 J_k J_k^T), wrapped on the torus; P_k = in-ball fraction of S = 256
  Gaussian draws (no predictor calls); pick argmax P_k, ties to `floor`.
- `ENS3`: 3 posterior members per candidate, in-ball count, ties to `floor`. Call-matched control for LIN
  that lacks the linearisation.
- `CERT` (horizon certificate + abstain): `floor` if max_k delta*||J_k||_2 < r/2, else random. Same calls as LIN.
- `TOPO` (diagnostic): H0 persistence of the K-cloud {z_k} under the torus metric.

## Bed validity (must hold, else every row void)

1. Ceiling: at T = 2, bayes >= 0.98 * oracle (seed mean).
2. Gap exists: bayes - floor >= 0.03 at >= 2 leads (the decision-local gap is present).
3. Label-shuffle null: every arm scored against within-start shuffled successes lands within 0.01 of blind.

Deciding band: leads with bayes - floor >= 0.03 and bayes - blind >= 0.10 (seed means; floors/ceiling only).

## Learnability gate

Learned arms (DJ, DJs): NS on their own training set at T = 2 >= 0.90, per seed. A seed that fails voids that arm's rows.

## Bars (deciding: 3-seed means over the band)

- **LIN WIN:** NS_LIN >= NS_DJ - 0.01 at every band lead, and band-mean NS_LIN >= NS_floor + 0.05.
- **LIN KILL:** NS_LIN < NS_DJ - 0.01 at any band lead. Replacement registered now: `DJs` (reprice: the learned
  operator fed the spread).
- **Linearisation earns its calls:** band-mean NS_LIN > NS_ENS3 + 0.02. Else LIN is only a small ensemble; retire to ENS3.
- **CERT never hurts:** CERT >= floor - 0.01 at every lead; report the abstain fraction.
- **TOPO predicts the gap:** on eval starts in the band, AUC of the longest finite H0 bar (and of the shortest) for the
  event [floor pick fails and bayes pick succeeds] >= 0.65 and >= AUC of the spread feature log(delta*||J_floorpick||_2) - 0.02.
  **KILL** otherwise; replacement: the spread feature.
- **Speed:** report predictor calls per candidate, parameters, training starts, and measured eval wall clock per arm.

## TDA correctness (RED first, anthropic-skills:tda-tdd)

H0 code must pass: permutation invariance, isometry invariance (torus translations, axis swap, reflection),
scale equivariance (Euclidean), stability (bottleneck <= 2 eps), n-cluster ground truth with a Gaussian-blob
negative control, parity against ripser 0.6.14 (Euclidean and precomputed torus distance).

## Amendment A1 (2026-09-28 ~00:58, after Wilson's facts `sun/rjepa/wilson/facts.json`, before any run)

The `DJ` arm above is replaced by a reimplementation, from the paper's specification only (code not cloned), of the
D-JEPA ordinal operator: token v_i = [LN(zhat_i - z_g); r_i], r_i = (rank(c_i) - 1)/(K - 1) with c_i the latent
distance (one latent source, so the second half of the token is dropped); latent = (cos, sin) of both torus
coordinates (4-d); base b_i = r_i; 5 -> 64 encoder (LayerNorm, GELU), two pre-norm Transformer layers, 4 heads,
FF 128, no candidate positional encoding; head 64 -> 8 -> 1 zero-init; s_i = b_i + 0.2 tanh(W_up tanh(W_down h_i)),
lower is better. Loss -log sum_{y=1} softmax(-s/0.05) + 0.25 mean softplus((0.02 + s_pos - s_neg)/0.05) +
0.1 mean delta^2; AdamW lr 3e-4, wd 1e-4, clip 1.0, 1,000 updates of 8 starts (the paper's schedule).
The 64-start calibration checkpoint gate is not reproduced.
Added arms (same deciding bars as DJ where named):
- `DJL`: same operator, 1,000 updates of 256 starts (DJ's best shot; the LIN WIN bar is read against max(DJ, DJL)).
- `DJs`: token [LN(zhat - z_g); r; log sigma], base r.
- `LINDJ` (the combination Wilson asks for): the operator with base b_i = rank under LIN and token [LN(zhat - z_g); r_LIN; r_floor].
  Descriptive, not deciding.

## Amendment A2 (2026-09-28 ~01:08, during the arms run, before any band result of it was read)

Declared: a 2-lead smoke run of `arms_one` at N_eval = 300 (T = 2, 12) was seen; at T = 12 it read floor 0.520,
bayes 0.563, lin 0.557, dj 0.517, cert 0.287 (abstain 0.52). Added after it, descriptive:
- `LIN1` (the high-dimensional form of LIN): chance score Phi((r - d_k)/s_k), s_k = delta * ||J_k^T n_k||, n_k the unit
  goal direction at z_k. One vector-Jacobian product per candidate, no Gaussian draws, 0 parameters. It exists
  because LIN's S-draw Monte Carlo needs S JVPs once the observation lives in a high-dimensional space.
  Bar: band-mean NS_LIN1 >= NS_LIN - 0.02 means the chance rule survives the move to one VJP.
