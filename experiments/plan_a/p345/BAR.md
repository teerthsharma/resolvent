# P3 / P4 / P5 bars: bed, arms, budget (registered 2026-09-27, before any run; nothing here runs in this pass)

Owners were fetched 2026-09-27 unless marked [U]: HAE, Keurti et al., ICML 2023, arXiv 2207.12067;
DeltaProduct, arXiv 2502.10297; PaTH attention, arXiv 2505.16381; BitNet b1.58, arXiv 2402.17764
("can match the performance of the full precision baseline starting from a 3B size"); Alabdulmohsin,
Tran, Dehghani 2024, arXiv 2402.01825 (H approx 0.7). FineWeb-Edu [U].

## P3 Grassmann order: no novelty claimed

- **Bed:** the anchored A5 word problem that already exists in the tree,
  `tests/foreman/phase_j/L1/chase/beds_fixed/a5_bed.py`. It has 4 generator tokens, L = 64, and the
  target is the running product (60 classes). Two repairs carry over from Foreman L1 F1/F4: a BOS
  anchor, and a split seeded by `random.Random(f"a5|{split}|{seed}")`, never `hash()`.
- **Canon arm:** the geometric product in its even-Clifford form. Unit quaternions are Cl+(3) =
  Spin(3) = SU(2), which makes this the existing `(f_Q)` arm (`su2.py`). A hand-set lift reads A5
  500/500 at L 64 with no training (Chase L1). Trained and anchored, it reads 1.0 at position 64
  against FoX's 0.0156 on **one seed** (Foreman L1).
- **Owner arms:** HAE; DeltaProduct (n_h = 2); PaTH. The floors are the ALiBi twin, FoX and the
  majority class.
- **Learned gate:** held-out accuracy at position 64 must be >= 0.90. Every order row also prints the
  group-lift check (inverse-pair and order-3 errors < 0.1), per Foreman L1 F2.
- **Prediction:** Canon is within 0.02 of the best owner at position 64, over 3 seeds.
- **Counter:** Canon is > 0.05 below the best owner. The geometric product then leaves Canon for the
  owner's form.
- **Budget:** CPU or under 1 GB of GPU. Anchored runs used 512 sequences and 400 steps. At 6 arms x
  3 seeds, that is under 2 GPU-hours on this box. No cloud.

## P4 Self-similar mixing

- **Bed:** FineWeb-Edu [U] language modelling at R1 (6 x 320, 23.5M parameters, 0.47B tokens) and R2
  (8 x 512, 50.9M, 1.02B). The token store is the one Phase K built: 3,004,523,761 GPT-2 tokens,
  per `tests/foreman/phase_k/K0/wilson/WILSON_REPORT.md`.
- **Arms:** the ALiBi twin as floor. Canon-SS uses one mixing rule shared across window scales
  2^k, k = 0..log2(ctx), with half the unique non-embedding parameters of the twin. The owner is a
  multi-scale or hierarchical transformer [U, fetch before the run].
- **Learned gate:** each arm's eval loss must be below the unigram entropy of the eval split by at
  least 2 nats.
- **Prediction:** |loss(Canon-SS) - loss(ALiBi)| <= 0.01 at half the unique parameters, and the Hurst
  exponent of the model's own per-token loss series is within 0.05 of the data's. The Hurst estimator
  is the rescaled-range method of arXiv 2402.01825; it gets an estimator check on a synthetic series
  of known H before any model read.
- **Counter:** Canon-SS is worse by > 0.02.
- **House addition:** a twin at half the depth (the same half-parameter count, no sharing) is run as
  an arm. If it also lands within 0.01, the sharing is not the lever.
- **Budget, from the measured Phase K MFU on this box:** R1 is 2.6 h x 1.1-1.2, so 2.9-3.1 h per
  run. R2 is 12.3 h x 0.64-0.96, so 7.9-11.8 h per run. With 3 arms x 3 seeds, R1 costs 26-28 h and
  R2 costs 71-106 h of local GPU. **Blocked:** the GPU is shared with Plan B, and a multi-day local
  run needs the author's yes. No Kaggle.

## P5 Ternary plus growth

- **Step 1** reproduces BitNet b1.58 at R1: absmean ternary weights and 8-bit absmax activations,
  against the FP twin at equal size and tokens. The source claims parity only from 3B up, so the
  **prior at R1 is a loss gap**. This row measures the gap at R1; it does not test the source's 3B
  claim.
- **Step 2** applies ternary weights to Canon, then GMDH growth: a layer is kept only if held-out
  loss improves by >= 0.005.
- **Prediction:** at R2, ternary Canon is within 0.02 of FP Canon at <= 1/4 of the weight memory.
- **Counter:** the BitNet gap at R1 is > 0.10 nats. BitNet's claim then does not reach our scale,
  and ternary leaves the R-ladder.
- **Budget:** step 1 is 2 arms x 3 seeds x R1, 17-19 h. Step 2 at R2 costs another 48-71 h.
  **Blocked**, as P4.

## Integration gate

The Canon block is assembled only after >= 3 of P1-P5 pass. With P4 and P5 blocked on compute, at
most P1, P2 and P3 can be decided on this box.
