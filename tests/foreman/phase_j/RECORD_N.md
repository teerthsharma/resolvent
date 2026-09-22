# Phase J — Addendum N (the diagonal): process record

Contracts: `ADDENDUM_N_CONTRACT.md` (v1) and `ADDENDUM_N_CONTRACT_v2.md` (v2, received
mid-round, 00:01). None of the four instance scripts the contract cites
(`diag_leap.py`, `crossties.py`, `kr_density.py`, `pf_regime.py`) exists anywhere on
this machine (Wilson: 233 directories and 164 repositories searched); every instance
was rebuilt from the text.

## Round N1 (2026-09-22 23:58 – 2026-09-23 00:3x): results as bound

Fellows on opus with nurses; one Inspector pass (58 audited, 13 struck). Reports:
`N1/<seat>/`; audit: `N1/inspector/FINAL.md`.

**Addendum N's mathematics holds at β = 1 on its own instances.** Rebuilt independently
(Chase): D1–D5, X1, X2a, X4, T10, and v2's X3a/X3b, KR and PF reproduce, every D/X error
below 1e−12 in f64, so the contract's kill line does not fire. Exceptions: X2b does not
reproduce and reverses (segmented scan 5.9× more accurate in fp32 worst case, not less:
VOID); T3/T3b's certification is struck (bar loosened after it failed); T8's softmax
cells are recorded but not asserted.

**On the trained family it does not describe the operator.**
- β is not 1: (f) learned 1.061 / 0.930 / 0.900 (Wilson, read from the checkpoint).
  At β ≠ 1 an absorber's self-weight is e^{(1−β)s_ii}, so the unit-diagonal count finds
  0 of 3 planted absorbers, and a self-weight above 1 puts a pole of the resolvent
  inside γ < 1 (Chase, bound). R-DIAG at trained β cannot fail (VOID).
- D3 as written is wrong for a gate with phase; the modulus form
  ρ_i = Σ_{j<i} |G_{i−1,j}| e^{s_ij−s_ii} holds to 2.2e−16 (Chase).
- The trained gate has no exact zeros (Wilson: 0 in every layer; minimum m 0.047 /
  0.097 / 0.093), so the absorber regime never fires.
- "The cost is the causal depth" is struck for dense heads: D = n − 1 structurally.

**H-SINK is struck** (Foreman, bound). P1: enrichment 0.0 at delimiters (bar ≥ 2);
P4: ρ = −0.020 (bar ≥ 0.3); P3: value-norm ratio 0.843 (bar ≤ 0.5). The softmax twin has
no first-token sink on this bed: its position-0 mass is 0.0113 against a uniform
0.01138. From the grid's records, the output-gated twin recovers −0.155 of C_win (all 5
split seeds negative). Evaluation-only, bound: (f)'s magnitude gate and FoX's forget gate
mark the same positions (Spearman 0.634 / 0.541 / 0.311); both arms lose at most
0.0011 nats when cut to a 16-byte window; replacing their data-dependent decay by a
constant costs 0.316 and 0.381 nats. On this bed the family's gate is a forget gate.

**R-COST-N** (Cameron, bound; Inspector GPU re-run). An exact blocked forward
substitution reads the causal resolvent at 1.757–1.797× one SDPA pass (relative error
2.96e−6 against float64): the ≤ 1.5× prediction is struck and the > 3× counter does not
hold. The current degree-130 Chebyshev hop scheme is 128.5–129.5× and diverges on the
real, non-normal W (relative error 4.46e4): C4 is certified, and worse than written.

**R-NEVER-LEN** (Cameron, logit level, bound). The sparsemax head's false-influence
mass is exactly 0 at n = 256 / 1024 / 4096, with no threshold. The counter holds: SSMax
at s = 1.0 with an oracle threshold reaches exact-set accuracy 1.0000 / 1.0000 /
0.9998, within 0.02. A threshold frozen at n = 256 still reaches 0.9907 at n = 4096.
NEVER is a certificate only; it does not meet condition 2.

**Owners** (Wilson, verified at source). Qiu et al. 2025, gated attention, is a NeurIPS
2025 Best Paper. PaTH puts its Householder-like products inside the q·k logit.
DeltaProduct (NeurIPS 2025) states in its body that two reflections make a rotation
and that A₅ needs only n_h = 2: the capability Addendum L claimed is owned.

## House's adversarial audit of D2 (contract §7) — `N1/house/house_d2_audit.py`

- D2 holds for the complex operator at any β: residual 2.0e−16 / 1.2e−16 / 4.0e−16 at
  β = 1 / 1.0906 / 0.8996. β ≠ 1 is not a counterexample to D2 (it breaks D1 and D3).
- **First counterexample, in our own head:** the model reads the real part
  (`ceq/hf/modeling_ceq.py:485`). Re(W_G) = W ∘ cos(Φ_i − Φ_j) to 1.1e−16. Its
  spectrum is unchanged (eigenvalues equal the diagonal exactly), but it is not a
  change of basis: 571 entries turn negative, entry magnitudes move by up to 0.379, and
  the equilibrium gain vector (the right eigenvector at eigenvalue 1) spans −0.366 to
  1.0 where the ungated operator's is exactly 1. Under the readout the model actually
  uses, phase is a signed attenuation of the off-diagonal weights, not a basis.
- A normalizer that sums G instead of |G| moves the diagonal by 2.62 (trivial).

## House's leap (hypothesis; unbound until round N2 binds it)

Dr House's fable run is disabled; the dispatcher makes the leap.

**Surprising fact.** The softmax twin attends 15–113 bytes back and collapses when cut
to 16 bytes (1.276 → 3.002 nats), while FoX and the family attend 1.2–2.0 bytes back,
lose at most 0.0011 nats at 16 bytes, and beat the twin by 0.23 nats — and FoX, added
to the twin with nothing else changed, recovers 100–111% of the family's win.

**Hypothesis.** The win is a relative-position prior, not the operator. Fixed structure
first: every arm receives position only as a learned absolute table added at the input
(`ceq/hf/modeling_ceq.py:567, 586`); there is no RoPE and no relative bias. The twin must
assemble recency from that table inside q·k; FoX and the family get recency
structurally from their gates. If so, the result is a matter of course.

**Own instance (`N1/house/leap_instance.py`, CPU, position-only layer-0 to 2 logits of
the trained checkpoints).** The twin does build a recency prior from its table: near
(1–16) minus far (64–256) logit gap up to 2.07 nats in one head, mean 0.34, Toeplitz
R² up to 0.58. FoX and the family leave their tables with no relative structure (gap
≤ 0.01 nats, R² ≤ 0.009): they hand recency entirely to the gate. By the criterion set
before running (a gap ≥ 2 nats in any head weakens the hypothesis), the strong form
"the twin has no recency" fails; the weaker form "the twin's recency is weak and costly
to learn" stands.

**Kill row (R-POS), registered before it runs.** Train two twins that add a fixed
relative prior and no parameters — RoPE, and ALiBi — at the grid shape, split seeds
0–2, CRN-paired with the grid's (a) and (f). Recovery = (loss(a) − loss(arm)) / C_win
per seed. PASS if the RoPE twin recovers ≥ 0.8 of C_win at all three seeds: the win is
positional encoding. KILL if both RoPE and ALiBi recover < 0.5: data-dependent
forgetting is required, and FoX owns it. Otherwise NEITHER.

## Round N2 (2026-09-23 00:37 – 01:1x): the leap bound

Fellows: Foreman (R-POS), Chase (R-DIAG′). One Inspector pass: 20 audited, 5 struck,
neither headline. Reports: `N2/<seat>/`; audit: `N2/inspector/FINAL.md`.

**R-POS: PASS — the leap binds.** Against the bar registered above before any cell ran,
the RoPE twin recovers 0.9632 / 0.9971 / 1.0143 of C_win at split seeds 0 / 1 / 2 (mean
0.9915) and the ALiBi twin 1.0884 / 1.0671 / 1.0882 (mean 1.0813), every arm at 724,608
parameters with CRN digests matching the grid's (a) and (f). The ALiBi twin beats the
family by 0.0206 / 0.0160 / 0.0231 nats. Recoveries recomputed by the Inspector from
`tests/foreman/design4x5/design4x5_results.jsonl`. Struck as unbound: "FoX's data
dependence buys nothing over ALiBi" (FoX leads at seed 2 and was not re-run), and "(a)
was the only arm without a relative prior" (a2 is also a softmax twin without one).

**R-DIAG′: PAST A POLE.** On the trained family (learned β 1.061 / 0.930 / 0.900), the
Mode B read at γ = 0.99 is ill-posed in 1,432 of 1,536 (window, layer, head) pairs,
93.2%; the largest self-weight is 2.0677 (γ* = 0.4836). No token is an absorber, and
no diagonal equals 1 — position 0 included (this bed has no BOS token). Pinning β = 1
makes the read well-posed and costs +0.0222 nats at evaluation.
