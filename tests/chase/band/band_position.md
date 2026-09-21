# Band-position leap: STEP 0 result (CHASE)

Scope: reads only, CPU only, scratchpad + board log only. `ceq/` and `ceqjepa/`
read, never written. No STEP 1 (the gamma sweep). No Kaggle. No test suite.

## Fixed structure (before any scored quantity)

| item | value |
|---|---|
| checkpoint (primary) | `ceqjepa/artifacts/curriculum/chess.pt` — `bed=chess`, `phase=chess`, step 550 (resumed from `english.pt`, +300 steps), `n=16`, `nA=4`, `g=0.9`, `rank=8` |
| checkpoint (cross-check) | `ceqjepa/space/chess_checkpoint.pt` — `bed=chess`, step 400, standalone, `n=32`, `nA=4`, `g=0.9` |
| checkpoint (excluded) | `ceqjepa_checkpoint.pt` (repo root) — `geometry.bed='synthetic'`, 20 steps; the `phase='chess'` label is cosmetic, not chess-trained |
| model | `ceqjepa.train.TinyCEQ`, forward exactly as shipped; teleport left at its default 0.0125 (never overridden for eval in `train.py`) |
| P | `ceqjepa.operator.build_operator(L0 + delta_logits(x), absorbing_idx)`, `[n,n]`, per held-out chess FEN, drawn fresh from `ceqjepa.beds.chess.ChessBed` (seed 0, 40 games, 64-example batch — a distributional repro, not a byte-identical replay: the bed itself isn't checkpointed) |
| gamma | checkpoint's own `g` buffer, frozen (never trained): 0.9 on both checkpoints |
| support graph | directed, n nodes, edge `j -> i` (`j<=i`, causal) iff `P[i,j] >= threshold`; threshold swept (0, 1/n², 1/n, 2/n) because P is a dense causal softmax — every causally-visible entry is analytically nonzero |
| diameter | max shortest-path hops over reachable ordered pairs (this is a causal DAG, not the undirected king-move board the leap's illustrative example used; "no path" is a legitimate outcome here) |

Script: `band_position.py` (this directory). Output: `band_position_results.json`.

## Result 1 — literal reading (full P)

`rho(P) = 1.000000` **exactly**, for every one of 64 held-out examples, on
both checkpoints. This is not a measurement — it is Perron–Frobenius applied
to a row-stochastic matrix (softmax rows + absorbing identity rows sum to 1
by construction), and no amount of training can move it. So
`gamma*rho(P) = gamma = 0.9` always, and `xi(P) = -1/ln(0.9) = 9.4912`,
constant, unconditionally.

Support-graph diameter(P): 1 (nonzero test — degenerate, the causal region
is dense) up to 2–3 at the tighter thresholds.

**xi(P) >= diameter(P) at every threshold on both checkpoints → LEAP DEAD**
under the literal reading. But this kill is vacuous: it never used the
checkpoint's trained weights at all, since rho(P)=1 holds for a random,
untrained, or adversarial P equally.

## Result 2 — steelman (transient block Q)

The leap's own worked figure (rho=0.2596, no fit) cannot be `rho(P)`; it
must refer to something sub-stochastic. The natural candidate this
architecture actually computes is `Q = P` restricted to the transient rows
and columns (absorbing rows/cols dropped) — exactly the matrix
`ceqjepa.operator.committor` inverts as `(I-Q)^-1 R`. Q is not
row-stochastic, so `rho(Q) < 1` is not forced.

Measured:

| checkpoint | rho(Q) mean | rho(Q) max | gamma*rho(Q) max | xi(Q) worst-case |
|---|---|---|---|---|
| primary (n_T=12) | 0.0243 | 0.2315 | 0.2083 | 0.637 |
| cross-check (n_T=28) | 0.1944 (identical across all 64 examples — see caveat) | 0.1944 | 0.1749 | 0.574 |

Support-graph diameter(Q): 1–3 across thresholds on both checkpoints.

**xi(Q) < diameter(Q) at every threshold on both checkpoints → leap SURVIVES
STEP 0** under the steelmanned reading, and the worst-case measured rho(Q)
(0.19–0.23) lands close to the leap's cited 0.2596.

Caveat on the cross-check checkpoint: `rho(Q)` is bit-identical across all
64 examples, meaning `P` did not vary with the input at all. `ceqjepa/train.py`
documents a real, previously-shipped bug (fixed 2026-09-08, LoRA-style
zero-init on both `delta_a`/`delta_b` factors makes the per-example operator
pathway dead — "every number in this repo's ledger up to 2026-09-08 was
produced with the per-example operator pathway dead"). This checkpoint
(`space/chess_checkpoint.pt`, no date recorded) may predate that fix; treat
its number as one degenerate data point, not corroboration. The primary
checkpoint (`curriculum/chess.pt`) does show per-example variation
(rho(Q) range 0.00005–0.2315) and is the more trustworthy of the two.

## Verdict

Not a clean kill either way. The literal STEP-0 instruction names a
quantity (`rho(P)` for this repo's actual row-stochastic P) that is a fixed
mathematical identity, so testing it against the trained checkpoint is a
category error — it kills the leap without ever consulting the weights.
The physically-motivated substitute (Q, the transient block that the
architecture's own committor solve uses) does not kill it, and its measured
worst-case value is close to the leap's own cited figure. **Recommendation:
if this row continues, re-run STEP 0 with Q named explicitly as the object
under test, not P** — the current STEP-0 instruction should be corrected
upstream rather than silently reinterpreted twice.

## FREE READS (a) / (b): blocked, not run

Both reads presuppose a discrete "gate" (open/closed) attached to the chess
checkpoint, binned by hop distance, with a per-step training log of gate
values. No such object exists for the resolvent P (P is continuous softmax
weight, never binary/gated). The repo does have a genuine gate mechanism —
`tests/chase/gate/r1_gate.py` (hard-concrete / straight-through carry gates,
commit `db701c9`) — but it: (1) trains on an unrelated synthetic corpus, not
chess, so there is no "chess checkpoint" with gates to bin by hop distance;
(2) its own log, `tests/chase/gate/r1_results.jsonl`, holds only aggregate
pre/post snapshots (init vs. step 800), not a step-indexed series, so
`mean(r)` and `mean(ln r)` over `r = g_{t+1}/g_t` cannot be computed from it.
Reporting this rather than fabricating numbers for either read. If reads
(a)/(b) matter, they need to be re-scoped to a real per-step gate log on a
real chess-trained gated model, neither of which exists yet in this repo.

## Citations: fetched, not from memory

| citation | bibliographic match | content match |
|---|---|---|
| Anderson 1958, Phys. Rev. 109, 1492 | CONFIRMED (APS): "Absence of Diffusion in Certain Random Lattices" | abstract confirms localization/no-diffusion at low density; did not confirm the specific "locator expansion / hop series" framing (paywalled beyond abstract) |
| Weinberg 1963, Phys. Rev. 131, 440 | CONFIRMED (APS): "Quasiparticles and the Born Series" | abstract confirms: perturbation theory (Born series) fails exactly when composite/bound states are present — matches the leap's paraphrase |
| Combes–Thomas 1973 | CONFIRMED (Crossref): Combes, J.M. & Thomas, L., "Asymptotic behaviour of eigenfunctions for multiparticle Schrödinger operators", *Commun. Math. Phys.* **34**, 251–270 (1973) — journal/vol/page were unstated in the leap text, now supplied | UNREACHED beyond metadata (403 on direct fetch, no login attempted) |
| Lewontin & Cohen 1969, PNAS 62, 1056 | CONFIRMED (Crossref): "On Population Growth in a Randomly Varying Environment", PNAS **62**, 1056–1060 (1969) — exact page match | UNREACHED beyond metadata (403 on direct fetch) |
| Ash & Nicholls 1972, Nature | CONFIRMED (Crossref): "Super-resolution Aperture Scanning Microscope", *Nature* **237**, 510–512 (1972) — exact page match | UNREACHED beyond metadata (login redirect, not followed) |

No citation is contradicted; none is yet fully content-verified beyond
Anderson and Weinberg. Per FETCH-BEFORE-CITE, none of these should enter
`docs/` on the strength of this pass alone — the three UNREACHED-content rows
need an actual full-text read (library/institutional access) before they are
cited as supporting specific numeric claims (the exp(-kappa|i-j|) decay form,
the lambda/60 number, the sigma²/2 drift rate).
