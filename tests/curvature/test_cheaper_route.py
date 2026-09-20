"""Pricing the barcode's competition, before any barcode is built.

`docs/CEQ_SHAPE.md:2142-2145` (card J-D5) prices the directed-stability
influence barcode at six owed items before a number off it may be read, and
`:2436` already records one of the six as KILLED on this bed: `beta_0(eps=0)
= 1` on `100%` of softmax draws (row 14, "the attention barcode" -- one weak
component, no skippable structure at threshold zero). Row 19 (`:2800`) is
`OCCUPIED BY THE AUTHOR` by a persistence-derived CSR schedule feeding a
forward-only kernel. Neither row is built or invoked here -- this file prices
what either would have to beat, using only what the bed already hands out for
free: `ceq/arm_smprime.py`'s own exact per-entry cache.

THE TWO FREE SCORERS. `ceq.arm_smprime.hop` returns `R`, the real modulus row
`R_ij = prod_{k=j+1}^{i} m_k` (clause 1 of `path_product`), and `numerator`
returns `mod_ij = R_ij * exp(w_ij)`, the exact per-entry term `Z_i` sums. Tile
the columns into blocks of `BLOCK` positions. For a block `b` strictly before
row `i`'s own block:

    carry[b, i]    = R[i, right_edge(b)]            -- SCORER A
    tile_max[b, i] = max_j in b of mod[i, j]         -- SCORER B, at eps

`carry` is EXACT: `mod[i, right_edge(b)] = carry[b, i] * exp(w)`, so
`carry[b, i] = 0` (bitwise, `m` is `{0, 1}` on this bed) forces every entry in
block `b` -- not just its right edge -- to `mod = 0`, and the converse holds
too (the right-edge entry alone is `carry * exp(w) > 0` whenever `carry != 0`).
So `tile_max[b, i] = 0 <=> carry[b, i] = 0`: scorer A is scorer B's `eps -> 0`
limit, not a different mechanism, which `test_scorer_b_converges_onto_scorer_a`
measures directly. Skipping on `carry == 0` costs nothing (the block's true
sum is `0`); skipping on `tile_max[b, i] < row_max[i] * eps` costs at most
`BLOCK * eps * row_max[i]` per block (every live entry in it is `<=
tile_max[b, i] < row_max[i] * eps`), which is the bound `M - log(1/eps)`
reads in log space.

RED FIRST. The naive hypothesis, before the two mechanisms were told apart,
was that BOTH scorers are exact on this bed, since BED-M's own gate is `{0,
1}` and not a continuous decay:

    def test_the_naive_hypothesis_both_free_scorers_are_exact():
        ...
        assert err == 0.0

and it failed, verbatim, at commit `1d7863f` on `WIN-16QAL06O9GB`, python
3.11.9, torch 2.14.0+cpu, `SEED=4051, N=64, BLOCK=8, QK=6.0, eps=1e-3`:

    E       assert 80964.80823258816 == 0.0

Scorer B is a genuine approximation with its own live regime -- content logits
scaled by `qk=6.0` give blocks a real spread of magnitudes, so a block can be
"small but not zero" and `tile_max` prunes it at a real, measured, non-zero
cost. `test_scorer_b_error_is_within_its_predicted_bound` is the corrected
test: the bound holds, not "the error is zero".

THE CONTROL. A same-budget random skip has no way to tell a dead block from a
live one, so at `SEED=4051` it costs `9.1e6` where scorer A costs `0.0`
(`|A| = 104` blocks), and `7.8e9` where scorer B at `eps=1e-3` costs `8.1e4`
(`|B| = 130` blocks) -- four to five orders of magnitude, which is the margin
either row of the barcode's competition would have to close.

`ponytail:` `N=64` is a probe size chosen to keep the O(N^2/BLOCK) reference
loop instant; a schedule kernel at the bed's real `s` is N-15's job, not this
file's -- this file only prices the free scorers a schedule would have to
beat.
"""
import torch

import ceq.arm_smprime as arm

SEED = 4051
N = 64            # bedm_draw's `s + 1`
DK = 8
BLOCK = 8
QK = 6.0          # content-logit scale: large enough that a block's true
                  # contribution can be small-but-nonzero, not just "on" or
                  # "off" -- BED-M's own {0, 1} gate alone never exercises
                  # scorer B's approximate regime (see the docstring's RED).
TOL = 1e-9        # float64 slack on a "<=" bound comparison, not a result


def _draw():
    """One real draw from the tracked bed (BED-M) plus content `q, k`.

    `bedm_draw` is `ceq/arm_smprime.py`'s own tracked-bed draw; `q, k` are
    drawn fresh (BED-M carries no embeddings of its own) with a seed offset
    by one so the gate draw and the content draw never share a stream.
    """
    a, b = arm.bedm_draw(seed=SEED, s=N - 1)
    u, th = arm.oracle_heads(a)
    g = torch.Generator().manual_seed(SEED + 1)
    q = torch.randn(N, DK, generator=g, dtype=arm.DTYPE)
    k = torch.randn(N, DK, generator=g, dtype=arm.DTYPE)
    return u, th, q, k


def _dense(u, th, q, k):
    """`(mod, R)`, both `[N, N]` real float64, read off the shipped arm.

    `mod[i, j] = R[i, j] * exp(w_ij)` is exactly the per-entry term `Z_i`
    sums (`numerator`'s second return, `qk` on); `R` is `hop`'s real modulus
    row, `path_product(m)`. Read-only calls into `ceq/arm_smprime.py` -- this
    file builds no new arm and no new route.
    """
    _, mod = arm.numerator(q, k, u, th, qk=QK, g=1.0, route="product")
    _, R = arm.hop(u, th, g=1.0, route="product")
    return mod, R


def _candidates(n=N, block=BLOCK):
    """Every `(b, i)` with key-block `b` fully strictly before row `i`'s own
    block -- the skip candidates. The diagonal block itself is never a
    candidate, matching every real block-sparse scheme (`docs/CEQ_SHAPE.md`
    card N-15's own guard)."""
    return [(b, i) for i in range(n) for b in range(i // block)]


def _per_candidate(mod, R, cands, block=BLOCK):
    """`(contrib, carry, tile_max, row_max)`, one float per candidate (plus
    one `row_max` per row) -- the whole reference computation, read once."""
    n = mod.shape[-1]
    row_max = {i: float(mod[i, : i + 1].max()) for i in range(n)}
    contrib, carry, tile_max = {}, {}, {}
    for (b, i) in cands:
        sl = slice(b * block, (b + 1) * block)
        contrib[(b, i)] = float(mod[i, sl].sum())
        carry[(b, i)] = float(R[i, (b + 1) * block - 1])
        tile_max[(b, i)] = float(mod[i, sl].max())
    return contrib, carry, tile_max, row_max


def _random_control_error(cands, contrib, budget, seed):
    """Skip `budget` candidates chosen uniformly at random (no replacement,
    same accounting as a scorer: the true contribution of every skipped pair
    is simply dropped) -- the do-nothing baseline either scorer must beat."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(cands), generator=g).tolist()
    return sum(contrib[cands[j]] for j in perm[:budget])


def test_scorer_a_carry_is_exact_and_beats_random_control():
    """Scorer A (`carry == 0.0`, bitwise): zero error, non-trivial skip
    fraction, and a matched-budget random control that is not free."""
    u, th, q, k = _draw()
    mod, R = _dense(u, th, q, k)
    cands = _candidates()
    contrib, carry, _, _ = _per_candidate(mod, R, cands)

    a_skip = [c for c in cands if carry[c] == 0.0]
    a_err = sum(contrib[c] for c in a_skip)
    assert a_err == 0.0, f"scorer A must be EXACT; got error {a_err}"
    assert 0 < len(a_skip) < len(cands), (
        f"expected a non-trivial skip fraction, got {len(a_skip)}/{len(cands)}")

    ctrl_err = _random_control_error(cands, contrib, len(a_skip), SEED + 100)
    assert ctrl_err > a_err, (
        f"random control at the same budget ({len(a_skip)} blocks) should "
        f"cost more than scorer A's exact {a_err}; got {ctrl_err}")


def test_scorer_b_error_is_within_its_predicted_bound():
    """Scorer B (`tile_max < row_max * eps`) at `eps in {1e-3, 1e-6, 1e-9}`:
    the realised error never exceeds the predicted `BLOCK * eps * row_max`
    bound, and a matched-budget random control costs at least as much."""
    u, th, q, k = _draw()
    mod, R = _dense(u, th, q, k)
    cands = _candidates()
    contrib, _, tile_max, row_max = _per_candidate(mod, R, cands)

    for eps in (1e-3, 1e-6, 1e-9):
        b_skip = [c for c in cands if tile_max[c] < eps * row_max[c[1]]]
        b_err = sum(contrib[c] for c in b_skip)
        pred_bound = sum(BLOCK * eps * row_max[c[1]] for c in b_skip)
        assert b_err <= pred_bound + TOL, (
            f"eps={eps:g}: realised error {b_err} exceeded the predicted "
            f"bound {pred_bound}")

        ctrl_err = _random_control_error(cands, contrib, len(b_skip), SEED + 200)
        assert b_err <= ctrl_err, (
            f"eps={eps:g}: scorer B ({b_err}) should not cost more than a "
            f"random control at the same budget ({len(b_skip)} blocks, "
            f"{ctrl_err})")


def test_scorer_b_converges_onto_scorer_a_as_eps_shrinks():
    """`tile_max[b, i] = 0 <=> carry[b, i] = 0` (the docstring's clause):
    at a small enough `eps`, scorer B's skip set is bitwise scorer A's."""
    u, th, q, k = _draw()
    mod, R = _dense(u, th, q, k)
    cands = _candidates()
    contrib, carry, tile_max, row_max = _per_candidate(mod, R, cands)

    a_skip = {c for c in cands if carry[c] == 0.0}
    b_skip = {c for c in cands if tile_max[c] < 1e-9 * row_max[c[1]]}
    assert b_skip == a_skip, (
        f"expected scorer B at eps=1e-9 to equal scorer A's exact skip set; "
        f"symmetric difference had {len(b_skip ^ a_skip)} entries")
    assert sum(contrib[c] for c in b_skip) == 0.0
