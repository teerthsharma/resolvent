# CAMERON — Addendum N, it.N1: R-COST-N and R-NEVER-LEN

**Short answer.** The diagonal theory holds on both of the rebuilt instances. It does not deliver a capability that only exact zeros have:
- **R-COST-N.** The exact read costs **1.78x** one SDPA pass. The ≤1.5x prediction is struck. The >3x counter does not hold. C4 is certified: the hop scheme pays 129.5x, and its Chebyshev form is **not exact** on the real operator.
- **R-NEVER-LEN.** f_N's false mass is exactly 0. The **counter holds**: SSMax at its paper init plus a threshold ties f_N, so NEVER is a certificate only at the logit level.

Out of scope this round: R-HOLONOMY and R-KR (not run).

**Setup.** Box: RTX 4060 Laptop, torch 2.14.0+cu126. Flash attention is unavailable in this build. Default SDPA dispatch is **EFFICIENT_ATTENTION** (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`), measured at 4.38–4.64 ms at B1 H8 S4096 D64 fp32 with tf32 off.

**Code.** All code is in `C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_j/N1/cameron/`. It is not committed (no git this round). sha256 prefixes:

| file | sha256 prefix |
|---|---|
| cost_n.py | 9ab22197eaad779a |
| test_cost_n.py | 321fb4e56008b35f |
| never_len.py | 1c02278f1d312cce |
| test_never_len.py | 6ada137f2cbffe57 |
| test_never_len_transfer.py | f807147b37c284f7 |
| dense_depth.py | 4e5fcfa43b8a77ca |
| test_dense_depth.py | 61b6d02db6544d39 |

never_len.py gained an `acc_curve` field after the main run. No metric changed.

---

## R-COST-N (§5)

- **Prediction:** ≤ 1.5x one SDPA pass.
- **Author's counter:** > 3x.
- **Bar, declared before any forward-substitution number:** ratio ≤ 1.5 **and** relative error ≤ 1e-5 against a float64 dense triangular solve.
- **Setup:** W is a real causal softmax from q, k drawn with `default_rng(0)` at B1 H8 S4096 D64, γ = 0.99.

### Measured

Timings are medians of 31 reps. Arms alternate with SDPA inside one loop and are timed with CUDA events.

| read | ms | x SDPA | rel err vs f64 | sequential steps |
|---|---|---|---|---|
| **fs5g_c256**: coarse forward substitution. Blocks of c = 256. Right-looking fused-attention pushes (flash-decoding lse merge) run on a side stream with one-block lookahead. The 256×256 diagonal uses batched cuBLAS trsm. Captured as a CUDA graph. | 7.86 | **1.777–1.797** (3 clean processes) | **2.96e-6** | 16 |
| fs2g (two-level, left-looking inner) | 10.26 | 2.27 | 3.2e-6 | 32 |
| fs_block256 (softmax rows materialized per block) | 14.07 | 3.13 | 3.9e-6 | 16 |
| fs_dense (one 4096 trsm) | 36.4 | 8.01 | 1.6e-6 | 1 |
| **hop, Chebyshev K = 130** (the contract's degree at γ = 0.99) | 582.6 | **129.5** | **4.46e+4 (diverges)** | 130 |
| hop, Neumann K = 130 | 571.7 | 128.6 | 0.268 | 130 |

### Why the Chebyshev hop fails

W is non-normal, so a polynomial that is small on [−γ, γ] can be large on W.
- **Symmetric control:** the same code converges to 1.0e-8 at K = 130 on a symmetric matrix with spectrum in [−1, 1]. The implementation is correct.
- **Real operator:** on a causal softmax W at n = 256 it needs K ≈ 400 for 1.5e-7 (`cheb_check.py`).

### Sequential cost of a blocked forward substitution

- The chain has **S/c** coarse steps: 16 at c = 256.
- Each step makes one c×c fused attention call over the previous block, then one c×c triangular solve.
- Total work is one causal attention pass plus S·c·D for the diagonal solves.

### Dense causal depth

**Structural:** D = n−1. Every subdiagonal entry of the jump chain is greater than 0. For example, log10 of their product is −2874 at n = 1024.

**Numerical:** in fp64 the jump chain reaches:

| n | hops to rel err 1e-6 | hops to rel err 1e-12 |
|---|---|---|
| 256 | 18 | 26 |
| 1024 | 22 | 30 |

The result is the same whether V has mean 0 or mean 1. Paths in diffuse attention reach BOS in O(log n) jumps. Each hop is still one attention pass, so the jump chain costs about 20–30 SDPA passes against 1.78x for the forward substitution.

### Tests

- **RED:** `test_cost_n.py hop_cheb` → `AssertionError: R-COST-N FAIL: hop_cheb ratio 129.538 > 1.5x one SDPA pass`
- **Still RED:** `test_cost_n.py fs5g_c256` → `AssertionError: R-COST-N FAIL: fs5g_c256 ratio 1.777 > 1.5x one SDPA pass`. The exact bar passes at 2.959e-06.
- **Depth RED:** `test_dense_depth.py` → `AssertionError: D5-as-framed FAIL at n=256: rel err <= 1e-12 after 26 hops, not n-1 = 255`

### Verdicts

- **Prediction (≤ 1.5x): STRUCK.** Measured 1.78x.
- **Counter (> 3x): does not hold.** §6's kill of the Mode B cost claim does not fire.
- **C4 ("the hop scheme pays ~130x for a triangular solve"): CERTIFIED** at 129.5x and 128.6x. It is sharper than written: the Chebyshev hop at degree 130 is not exact on a real causal W.
- **"D5 exactness needs n hops" on dense attention:** the structural statement is **CERTIFIED**. As a numerical cost it is **STRUCK** (26–30 hops reach 1e-12).

### Replacement route (reroute)

- **Object:** one fused Triton kernel (triton 3.7.1 is installed on this box). It does the far push and solves the diagonal tile in registers by the doubling product (I−N)⁻¹ = ∏(I+N^(2^k)). N ≥ 0 here, so the product has no cancellation. In fp64 it matches trsm to 3.4e-13.
- **Measurement:** the same `test_cost_n.py` bar, ≤ 1.5x.
- **Seat:** Cameron, it.N6.
- **Why this route is sharper:** the profile isolates the remaining excess.
  - The fused attention work is about one pass: 6.25 ms against 6.15 ms for SDPA under the profiler.
  - The excess is 16 batched trsm calls (1.69 ms) plus about 60 small launches on the critical path.
  - Two torch-level ways to remove it were measured and lost: a blocked-doubling solve (2.14x) and inverse-ahead on 3 streams (2.19x).
  - Fusion is the one lever left.

---

## R-NEVER-LEN (§5), logit level, as in T8

### Bed

The bed is rebuilt from the contract text:
- A planted DAG with fixed depth 8.
- Anchors are about 9.3% of tokens, matching T6's 6/64 and 95/1024.
- Each non-anchor has p ~ U{1, 2, 3} parents at the previous level.
- Logits are Δ = 6 on parents and 0 elsewhere.

### Arms

- **(a)** softmax.
- **(a_s)** SSMax. The equation was fetched from arxiv.org/html/2501.19399: logits × s·log n, with n = keys seen = i+1 and s = 1.0 at init. The post-training value ≈ 0.168 was read by a nurse and not re-verified. s was swept over {0.168, 0.5, 1.0, 2.0}.
- **(a″)** FoX-style **constant** forget, −λ(i−j) with λ ∈ {0.001, 0.01, 0.1}. This is a proxy, not learned FoX.
- **(f_N)** sparsemax.

### How the read is scored

- Every softmax-family arm gets **oracle-absorbing anchors**. f_N's anchors absorb on their own (checked, 15/15).
- The read is B = (I−Q)⁻¹R, computed in fp64.
- The oracle threshold is the best of 241 values per arm, n and seed.
- Numbers below are over 5 seeds. The ± is the 95% CI with t = 2.776.

### Measured, n = 256 / 1024 / 4096

| arm | exact-set acc | false-influence mass |
|---|---|---|
| **f_N** (τ = 0, no oracle) | **1.0000 / 1.0000 / 1.0000** | **exactly 0.0 (15/15)** |
| (a) + oracle τ | 0.7428±0.1381 / 0.3756±0.0520 / 0.1497±0.0565 | 0.0860 / 0.2066 / 0.4125±0.0136 |
| (a_s) s = 1.0 + oracle τ | 1.0000 / 1.0000 / **0.9998±0.0004** | ≤ 1.4e-5 |
| (a_s) s = 2.0 + oracle τ | 1.0000 / 1.0000 / 1.0000 | ≤ 3.0e-9 |
| (a_s) s = 0.5 + oracle τ | 0.9930 / 0.9693 / 0.9573 | — |
| (a_s) s = 0.168 + oracle τ | 0.6113 / 0.3211 / 0.2074±0.0487 | — |
| (a″) best λ + oracle τ | 0.7569 / 0.3822 / 0.1499 | — |

- **Leak law:** measured equals the formula to 1.4e-15. The mean one-hop leak is 0.1527 / 0.3728 / 0.6567.
- **T8's reference numbers** match the last-row reading of the law with p = 2: 0.5588 at n = 1024 against the contract's 0.559. The law gives 0.2394 and 0.8354 at n = 256 and 4096.
- **SSMax leak law:** (i+1−p)/(p·(i+1)^(sΔ) + i+1−p) holds to 1.9e-15. The leak **vanishes with length if and only if sΔ > 1**.
- **D-hop read on f_N:** exact after D = 8 hops (error 0.0). After 7 hops the error is 1.0.

### Tests

- **RED:** `test_never_len.py --stub --quick` → `AssertionError: R-NEVER-LEN P1 FAIL: f_N false mass max 6.830e-02 > 0`
- **P1: GREEN** on the real arms.
- **P2: still RED** → `AssertionError: R-NEVER-LEN COUNTER HOLDS: (a_s)+threshold within 0.02 (0.0000); NEVER is a certificate only`

### Verdicts

- **P1 (f_N false mass = 0 at every n): CERTIFIED.**
- **(a) grows as the leak law says: CERTIFIED.**
- **P2 (f_N beats a_s by ≥ 0.10 at n = 4096): STRUCK.**
- **The counter HOLDS.** By §6, NEVER is a certificate only, and it does not meet condition 2.

### The room's question

Does the tie of the leaps give a capability a forget gate lacks?
- **Against a data-independent forget, yes:** 1.0 against ≤ 0.15 at n = 4096, even with an oracle threshold.
- **The capability is not unique to exact zeros.** SSMax, a length-scaled temperature and not a forget gate, matches it at its paper init.
- **What exactness adds is the certificate.** f_N needs no threshold. SSMax's oracle τ at s = 1.0 ranges from 3.7e-8 to 3.1e-4 across seeds at n = 4096.

### Replacement route 1 (reprice by transfer): tried, killed by its own test, retired

The idea: choose τ once at n = 256, freeze it, and apply it at n = 1024 and 4096.
- **RED on the stub:** `AssertionError: R-NEVER-LEN' COUNTER HOLDS: transferred-tau (a_s) within 0.02 (-0.9984)`
- **Real run, still RED:** `AssertionError: R-NEVER-LEN' COUNTER HOLDS: transferred-tau (a_s) within 0.02 (0.0093)`
  - SSMax s = 1.0 scores 0.9907 at n = 4096. Per seed: 1, 1, 1, 0.9534, 1.

### Replacement route 2 (standing): reprice by s·Δ

- **Object:** trained (a_s) heads on the planted-chain bed. Record their learned s and their parent-versus-distractor logit gap Δ̂.
- **Measurement:** s·Δ̂, plus exact-set accuracy at a transferred τ, compared with f_N.
- **Seat:** the learned round, it.N4. Foreman trains and Cameron reads.
- **Why this route is sharper:** this round's verdict was decided by one scalar that was set by hand (Δ = 6). At s = 0.168, sΔ = 1.008 and the tie breaks by a gap of 0.79. At s = 1.0 it ties. Measuring s·Δ̂ on trained weights removes the free parameter that decided the outcome.

---

## OPEN

- **R-COST-N ≤ 1.5x is unmet.** It needs the fused Triton kernel. **Default:** build it at it.N6 against the unchanged `test_cost_n.py` bar.
- **Cost evidence has limits.** It uses a single q, k seed. The spread 1.777–1.797 comes from 3 processes and is not a 95% CI.
- **NEVER-LEN evidence has limits.**
  - It is logit-level only.
  - The FoX arm is constant decay, not learned data-dependent FoX.
  - All softmax-family arms got an oracle for absorbing anchors.
  - The contract's instance scripts were absent, so the bed was rebuilt from the text.
- **SSMax's post-training s ≈ 0.168 is nurse-fetched** and not re-read by me. The sΔ = 1.008 row depends on it.
- **Not run this round:** R-HOLONOMY and R-KR.
