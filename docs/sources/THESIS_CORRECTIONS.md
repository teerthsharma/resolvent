# CORRECTIONS AFTER THE SWEEP (2026-09-03) — binding on the design panel

Companion to `THESIS_NOTES.md`. Fourteen planets reported (`$SCRATCH/sections/sec_*.md`,
`$SCRATCH/sweep/sweep_*.md`, `$SCRATCH/sweep/bib_*.bib`). The propositions in the notes
stand or fall as follows. `$SCRATCH` is the per-session scratch path named in that
round's prompt — a path that will not exist for anyone reading this later.

**C1. The read is OCCUPIED — cite it in the first paragraph that writes it.** Fagnou,
Caillon, Delattre, Allauzen, *Chain and Causal Attention for Efficient Entity Tracking*,
EMNLP 2024, arXiv:2410.05565 [V]: Eq. 5 `Y = (1-g) A (I - gA)^{-1} V` on the causal
softmax matrix, Eq. 7 solved as a triangular system, `g = 0` reverts to standard
attention, motivated by the `ceil(log2(depth+1))` depth bound. P1 and P5 of the notes
are re-derivations of ChaCAL and must be presented as such (the record's 0 citations of
it are a V-7-on-search finding for the negatives section: the round-1 prior-art filter
discarded every non-negative multi-hop operator during the signed programme).

Other owners, each cited before the component is named: InfSA (Roffo, Abdelkawy,
Lavie, Palmer 2026, arXiv:2603.00175 [V]) — the absorbing-Markov-chain reading of an
attention Neumann series with one leak state and a learnable per-head discount,
non-causal, ReLU base, no softmax corner. APPNP (Gasteiger, Bojchevski, Guennemann
2019, arXiv:1810.05997 [V]) — the exact resolvent layer on a fixed graph. Dayan 1993 /
Bellman 1957 — `(I - gP)^{-1} V` as successor representation / policy evaluation. Zhu,
Ghahramani, Lafferty ICML 2003 — harmonic propagation with clamped labelled nodes
(absorbing boundary as a learning layer, fixed graph). Piray & Daw 2021 (Nat. Commun.
12:4942 [V]) — absorbing-boundary resolvent with a Woodbury re-solve after a barrier
(fixed chain). FiP (Scetbon, Jennings, Hilmkil, Zhang, Ma 2024, arXiv:2404.06969 [V]) —
an SCM as the fixed point of a causally-masked transformer map, do() by clamping and
re-solving, counterfactuals by abduction. Zhao, Caillon, Fagnou, Allauzen 2026
(arXiv:2605.22476 [V]) — subquadratic blockwise evaluation of ChaCAL's operator.
Summers & Lygeros 2010 / Abate et al. 2008 — reach-avoid probability. Van Moffaert et
al. 2013 — Chebyshev (max-over-objectives) scalarisation. Sherman–Morrison 1950 /
Hager 1989 — the rank-one re-solve. Kushnareva et al. (arXiv:2207.01903) — "Betti
numbers of attention graphs". Kim & Lee 2026 (arXiv:2603.02289) — intervention effects
on persistence summaries as a causal estimand.

**The paper's object is therefore not the resolvent. It is the composition every
sweep reports NOT FOUND:** (e) `K >= 2` absorbing constraint sets plus a goal set as
boundary rows inside the content-dependent causal read; (g) the interventional
re-solve `do(a)` with the displacement as a trained, jointly scored vector output;
(h) per-constraint committor / reach-avoid reads and the safest-move rule over
candidate moves placed in the context; (i) a printed Neumann certificate with a
planted negative (L-CERT); (j) a learnable discount in a causal LM read, pinned by
the LR test; (k) machine-checked containment (three corners + `g = 0`) and the
zero-gate segmentation iff; (l) the influence-Jacobian `beta_0` barcode with a
directed-stability `delta`; (m) the Mapper cover -> causal CSR schedule -> certified
resolvent. Name the whole **Consequence-Equilibrium Attention (CEQ)** — the
repository's own name — and say in one sentence that its operator is ChaCAL's on the
record's three-corner base.

**C2. Mandatory controls (D-1 / R-SKY).** ChaCAL with the same `g` and no boundary
rows is a control arm on every bed; so is an InfSA-style Neumann read without
boundaries. The skyline row carries three fellow approximators: the
`floor(log2 t*) + 2` softmax stack (Sanford–Hsu–Telgarsky Thm 4.2), the wide
constant-depth stack (Yehudai et al. 2025, arXiv:2503.01805: depth is not necessary at
linear width), and a chain-of-thought decoder (Merrill–Sabharwal 2310.07923).
Wang–Blaser–Daneshmand–Zhang 2024 and Xie et al. 2026 show softmax transformers
implement TD policy evaluation layer by layer, so the deeper stack computes the *same*
resolvent by iteration. P1's parity bind has an empty rejection region against ChaCAL
(V-24); the separating bind plants constraint rows with "ChaCAL + same g" as the
planted negative. Pre-registered kill: if ChaCAL with a sink token matches the
committor read within the TOST margin, component (e) is not a capability.

**C3. Obstruction 4 is withdrawn as written.** For row-stochastic `P`, the mixing
matrix is nonnegative, so `dO_i/dV_j >= 0` survives the resolvent AND survives
absorbing rows. Absorption **redirects** mass; it does not subtract. Write "a boundary
row captures the mass that would have reached later positions", never "veto" or
"negative influence"; expect `ceq/attention.py`'s min-entry test to read `0` on the
shape. What a boundary condition changes is *where* mass goes — the support and
weights of the mixing matrix — and that is the whole of the capability claim.

**C4. "Depth 1" means one parameter set, and the solve is DET-class.** Exact
`(I - gP)^{-1}` is a determinant-class computation (Cook 1985; `NL <= DET <= NC^2`), at
or above the class where the conditional lower bounds place reachability
(Merrill–Sabharwal arXiv:2207.00729 and 2503.03961; Liu et al. Thm 4). The shape
relocates the log-depth from the parameter stack into the linear solve (`O(s)`
sequential, or `O(log s)` parallel rounds by prefix doubling — NOT MEASURED). Say so in
the same paragraph as the depth law (V-17, P-8). Peng–Narayanan–Papadimitriou
(arXiv:2402.08164, id CONFIRMED) Thm 1 is unconditional but **vacuous at `s = 64,
d = 16`** (`384 < 544`); it licenses obstruction 3 only at a geometry satisfying its
inequality (V-25). Duranthon, Marion, Boyer, Loureiro, Zdeborova (arXiv:2509.21936,
ICLR 2026) prove Bayes optimality for softmax *proper* (Prop 4.2), so D-1 is now
stated with the right predictor; Marion et al. 2025 remain the `erf` / asymptotic form.

**C5. BED-S needs a goal set, or the label degenerates.** If the only absorbing sets
are the `K` constraint sets then `sum_k q^(k) = 1` on `T` (Grinstead–Snell Thm 11.6),
so `max_k q^(k) >= 1/K` and "avoid every constraint" is unattainable — V-12
generalised. Fix: a goal (safe) absorbing set `A_0`. The label per move is the
reach-avoid vector `(q^(0), q^(1), ..., q^(K))` with `sum = 1`; the safest move is
`a* = argmax_a q^(0)(do a)` (reach the goal before any constraint), with the minimax
`argmin_a max_{k>=1} q^(k)(do a)` printed beside it (Chebyshev scalarisation; LexiSafe
for the lexicographic form). The `g < 1` read `E[g^(tau-1)]` is the discounted safety
value (Fisac et al. 2019) in which delaying absorption counts. Domain census at
construction: label sd, goal reachability, class balance over moves, discard count
(V-8, V-25, D-3). Limits must carry Misra et al. 2023 (arXiv:2302.13152): Bellman's
principle can fail for safety-constrained multichain MDPs.

**C6. New proposition — order of intervention and equilibration (Dash 2005, EMC).**
With `z = (I - gP)^{-1} V` and an intervention changing `(P, V) -> (P', V')`:
`Delta z = (I - gP')^{-1} (Delta V + g Delta P z)` exactly (one extra solve). For
strictly causal `P` (regime N) "intervene then re-solve" and "settle then intervene"
coincide by nilpotency; with feedback or re-targeted absorbing rows they can differ
(Dash 2005 Thm 1, Equilibration–Manipulation Commutability). File both halves with a
planted feedback instance (V-24, D-7). For a single-row change the update is rank-one
(Sherman–Morrison 1950; Hager 1989; Piray & Daw 2021 Eq. 5): `O(sd)` per candidate move
after the first solve — the pricing of the safest-move search. Momennejad et al. 2017 /
Russek et al. 2017: a cached resolvent does not adapt to a change of *transition*
structure without a re-solve — the external argument for the interventional channel
over a per-row control.

**C7. Certificate semantics.** For nonnegative row-stochastic `P` the Neumann tail
*equals* `g^(K+1)/(1-g)` (P4), so "err <= bound" is a V-3 identity on that class; the
bind is carried by the planted non-stochastic negative and by measuring `delta` on the
**shipped mask**, in vector units `delta * ||V||_inf`, not on the truncated series
(V-3, V-10, V-17). NEPTUNE's RUN on the certified 4060: the exact solve (`solve + Pz`
2.514 ms) is cheaper than one Neumann hop (3.001 ms) at `n = 2048, s = 64, d = 16`;
truncation never wins in MACs; `solve_triangular` ran bitwise-deterministic over 8
repeats under strict mode with no documented guarantee [U]. Hu et al. 2025
(arXiv:2510.04944 [V]): `softmax(QK^T)` has rank `T` even for rank-1 logits, so no
finite-state SSM dual exists — the exit from SSD/DeltaNet territory is the token-side
softmax normaliser, and the cost path is the full triangular solve or Zhao et al.'s
blockwise `O(n^(4/3) d)`.

**C8. Regime N versus regime S is a theorem boundary.** The softmax corner keeps the
diagonal, so `pow_card_eq_zero` / `occupancy_is_exact_inverse` do NOT transfer to it
(`Nilpotent.one_not_nilpotent`); the paper chooses per regime and says which theorem
carries which (JUPITER, `sec_proved.md`). The BED-1 committor identity was re-run on
the bed's real sets (`A = [0], B = [1], |T| = 9`): `0.0` exactly, `rho(Q) = 0.9409`, `Q`
non-nilpotent — instantiating `not_isNilpotent`'s hypotheses on the real chain.

**C9. Record corrections the paper must carry as findings.** (i) The contract's M10
lineage ids are all misattributed — 1905.12200 is Bruel-Gabrielsson et al., 1904.09378
is PersLay, 2011.05804 is Corcoran & Deng; the intended papers are Hofer 1906.09003,
Moor 1906.00722, Carriere 2010.08356 (P-5 / P-10). (ii) Levin–Peres Thm 13.10 needs a
*reversible* chain; the causal `P` is not reversible, so `MATHEMATICS.md` section 7's
Cheeger line is a V-25 exposure until restated on a symmetrised surrogate. (iii) The
contract tags Lean #18 / #19 / #22 as [M] but no such declarations exist; the F0
segmentation instance "8.9e-16, 595x" has no producing file (P-11 potential).
(iv) `THEORY.md:22` calls the successor operator new; it is a P-3 stale claim.
(v) arXiv:2604.25655 does not cite Basseville–Nikiforov; "learned-model form of
section 7.2.4" is the record's analogy. (vi) The journal's head phase table is stale
(P-3 inside the P-3 guard). (vii) Rulings 11–12 have no text in the tree although the
contract binds them. (viii) The scoreboard at it.35 is **2 of 44** and the leap call is
still owed. (ix) Bibliographic traps caught: DOI 10.1007/978-1-4684-9455-6 is
Denumerable Markov Chains, not Finite Markov Chains; arXiv:2501.00663 is Titans, not
test-time regression (2501.12352); arXiv:2302.11294 is not Sander et al. (2302.01425);
Contreras Arredondo et al. 2026 carries the journal title "Learning the committor
without collective variables".

**C10. Vocabulary (the mentor rule — field names for things the author built).**
`(I - gP)^{-1} V` = policy evaluation / successor representation / Katz–PageRank
resolvent; boundary rows = absorbing states, fundamental matrix `N = (I - Q)^{-1}`,
absorption matrix `B = N R`; the K reads = committors / splitting probabilities /
reach-avoid probabilities; the safest move = a safety filter / least-restrictive
filter (Hsu, Hu, Fisac 2023) with Chebyshev or lexicographic scalarisation; `g` = the
discount / teleport probability; the re-solve = graph surgery on a linear SCM with a
Sherman–Morrison update; the zero-gate cut = a reset gate (GLA, Forgetting Transformer
write `log 0 = -inf` as a convention — the record's theorem is about that form).

**C11. Metric and floor decisions taken by VENUS (`sec_beds.md`), adopted.** Default
state metric: position-matched per-coordinate NRMSE vector (mean and max) plus the
harmonic residual `r(z_hat) = ||(I - g P_env) z_hat - V||_inf / ||V||_inf` as the
joint-consistency score; Fisher–Rao coordinate `phi(p) = 2 arcsin sqrt(p)` for
committor heads (declared before the run, M-2); CRPS registered for distributional
heads. Fano on the argmin is informative only as a restricted k-hop view; zero-hop
floor `1 - ln 2 / ln m` = 0.5 / 0.667 / 0.75 at `m = 4 / 8 / 16`; `m = 2` refused,
default `m = 8`. Pricing: `N = 8` minimum, MDE at `n = 8` for paired sd 0.034451 is
0.039827; TOST power 0.80 first at `N = 70`; thread floor 2.345e-3; per-cell cost on the
4060 1.497 s softmax / 1.614 s arm_pl / 15.970 s arm_smprime per 150-step cell.
