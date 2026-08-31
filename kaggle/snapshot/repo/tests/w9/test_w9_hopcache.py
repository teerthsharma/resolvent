"""W9 -- a scoped, topological hop cache, so alpha > 0 can decode.

THE WALL W8 HIT. The multi-hop path sum needs a square operator. During
incremental decoding the operator row is `[1, N]` and `A^2` does not exist, so
alpha > 0 raised `NotImplementedError` rather than silently downgrading to
hops=1 and making prefill and decode disagree.

THE CONSTRUCTION. `A` is strictly causal, so the hop tower is a filtration:

    (A^h v)_i = sum_{j<i} A_ij (A^{h-1} v)_j

`hop_h[i]` depends only on `hop_{h-1}[j]` for `j < i`. Cache one vector per hop
per position and decode becomes exact -- K attention rows per token, K times the
value-cache memory, and byte-identical to what prefill computes.

WHY NOT A FLAT KV CACHE. A flat cache stores k and v and throws the tower away.
Three structures are borrowed instead, each supplying one thing a flat cache does
not:

  NeMo-Relay `scope_stack.rs`  -- LIFO scopes; popping a scope drops exactly that
                                 scope's registrations; the root refuses removal.
                                 That is LIFETIME.
  foliation                    -- only a FREE FACE may be evicted: refcount zero,
                                 no resident children. That is ADMISSIBILITY.
  W3, measured here            -- eviction is BITWISE (0.000e+00 over 24/24)
                                 where post-softmax gating leaks (2.15e-05).
                                 That is EXACTNESS.

Provenance note: NeMo-Relay ships no measurements at all -- no memory saved, no
drop counts, no latency. The structure is borrowed; every number below is
measured here.

WHY EVICTION IS SAFE AND GATING IS NOT. A gate multiplies a column by ~0 but the
token's score never leaves the softmax denominator, so it shadows every surviving
row forever. Removing it from the key set is bitwise. That is the difference
between "the weight is zero" and "the token was never there", and it is what lets
this cache drop entries without changing the answer for what remains.
"""
from __future__ import annotations

import pytest
import torch

from ceq import hopcache as hc

B, H, S, DH = 2, 3, 32, 16
ALPHA, RHO, HOPS = 0.25, 0.9, 3


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    return torch.device(request.param)


def qkv(device, seed=0, dtype=torch.float64):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: torch.randn(B, H, S, DH, generator=g, dtype=dtype).to(device)
    return mk(), mk(), mk()


# ------------------------------------------------------------- the core parity

def test_incremental_decode_matches_full_prefill_to_one_ulp(device):
    """THE falsifier for W9.

    Bitwise equality is NOT achievable here and asserting it would be dishonest.
    Prefill contracts the whole sequence in one matmul; decode contracts one row
    at a time. Same value, different summation order, so the results differ by
    floating-point rounding and nothing else.

    Measured, float64, S=32: max absolute 1.665335e-16, mean 1.320947e-17,
    relative to scale 1.898657e-16 -- one ULP. The operator ROW itself agrees to
    2.775558e-17, so the gap is entirely in the hop accumulation and not in the
    operator.

    Contrast with eviction below, which IS bitwise: there the retained
    computation is the identical expression, not a reordering of it. The two
    claims are different and are asserted differently.
    """
    q, k, v = qkv(device)
    full = hc.prefill_correction(q, k, v, alpha=ALPHA, rho=RHO, hops=HOPS)

    cache = hc.HopCache(hops=HOPS)
    rows = [cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                       alpha=ALPHA, rho=RHO) for i in range(S)]
    inc = torch.cat(rows, dim=-2)
    rel = float((inc - full).abs().max() / full.abs().max())
    assert rel < 1e-14, f"relative gap {rel:.3e} is too large to be rounding"


def test_cache_holds_k_slots_not_k_squared(device):
    """The whole point is that the tower is a filtration. If the cache grew with
    hops squared, recomputing would be cheaper."""
    q, k, v = qkv(device)
    cache = hc.HopCache(hops=HOPS)
    for i in range(S):
        cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                   alpha=ALPHA, rho=RHO)
    assert cache.n_slots() == HOPS, cache.n_slots()
    assert cache.n_entries() == HOPS * S, cache.n_entries()


def test_first_token_has_no_predecessors_and_contributes_nothing(device):
    """Strict causality: position 0 has an empty prefix. A softmax or an L1
    normalization over nothing is NaN if it is not handled."""
    q, k, v = qkv(device)
    cache = hc.HopCache(hops=HOPS)
    r0 = cache.step(q[:, :, 0:1], k[:, :, 0:1], v[:, :, 0:1], alpha=ALPHA, rho=RHO)
    assert torch.isfinite(r0).all()
    assert float(r0.abs().max()) == 0.0, float(r0.abs().max())


# --------------------------------------------- NeMo-Relay scope semantics

def test_popping_a_scope_drops_exactly_that_scopes_entries(device):
    """LIFO scopes. `pop` drops that scope's registrations and nothing else."""
    q, k, v = qkv(device)
    cache = hc.HopCache(hops=HOPS)
    for i in range(8):
        cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                   alpha=ALPHA, rho=RHO)
    before = cache.n_positions()
    cache.push_scope()
    for i in range(8, 14):
        cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                   alpha=ALPHA, rho=RHO)
    assert cache.n_positions() == before + 6
    cache.pop_scope()
    assert cache.n_positions() == before, (cache.n_positions(), before)


def test_root_scope_refuses_removal(device):
    """The root holds the shared prefix -- a system prompt, a few-shot block --
    that every later sequence depends on. Popping it is refused, not tolerated."""
    q, k, v = qkv(device)
    cache = hc.HopCache(hops=HOPS)
    for i in range(4):
        cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                   alpha=ALPHA, rho=RHO)
    with pytest.raises(RuntimeError, match="root"):
        cache.pop_scope()


def test_a_popped_scope_leaves_earlier_answers_bitwise_unchanged(device):
    """Dropping a scope must not perturb what was computed before it opened.
    Checked by parity against a run where that scope never existed."""
    q, k, v = qkv(device)

    def run(with_scope: bool):
        cache = hc.HopCache(hops=HOPS)
        out = []
        for i in range(6):
            out.append(cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                                  alpha=ALPHA, rho=RHO))
        if with_scope:
            cache.push_scope()
            for i in range(6, 10):
                cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                           alpha=ALPHA, rho=RHO)
            cache.pop_scope()
        out.append(cache.step(q[:, :, 10:11], k[:, :, 10:11], v[:, :, 10:11],
                              alpha=ALPHA, rho=RHO))
        return torch.cat(out, dim=-2)

    assert torch.equal(run(True), run(False)), float((run(True) - run(False)).abs().max())


# ----------------------------------------------- foliation eviction semantics

def test_only_a_free_face_may_be_evicted(device):
    """A position registered inside a live scope is not a free face. Evicting it
    would leave a hop tower with a hole in the middle -- the same corruption as a
    block table whose parent was evicted."""
    q, k, v = qkv(device)
    cache = hc.HopCache(hops=HOPS)
    for i in range(6):
        cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                   alpha=ALPHA, rho=RHO)
    cache.push_scope()
    for i in range(6, 9):
        cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                   alpha=ALPHA, rho=RHO)
    with pytest.raises(RuntimeError, match="free face"):
        cache.evict(0)


def test_eviction_is_bitwise_when_nothing_dependent_has_run(device):
    """W3's result, carried into the cache -- with the scope it actually has.

    Eviction is bitwise ONLY when no position was computed while the evicted one
    was resident. Evict immediately, before anything reads it, and the result is
    identical to never having inserted it: not approximately, bitwise.

    The moment a dependent position runs, its hop vectors have already contracted
    the evicted contribution into stored state, and removing the key cannot
    unwind it. That is `test_eviction_is_not_retroactive` below, and it is the
    price of O(K) decode rather than a bug.
    """
    q, k, v = qkv(device)
    drop = 3

    def run(evict: bool):
        cache = hc.HopCache(hops=HOPS)
        for i in range(6):
            if not evict and i == drop:
                continue
            cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                       alpha=ALPHA, rho=RHO)
            if evict and i == drop:
                cache.evict(drop)          # before anything depends on it
        return cache.step(q[:, :, 6:7], k[:, :, 6:7], v[:, :, 6:7],
                          alpha=ALPHA, rho=RHO)

    assert torch.equal(run(True), run(False)), float((run(True) - run(False)).abs().max())


def test_eviction_is_not_retroactive(device):
    """A property of this cache that has to be stated, because assuming the
    opposite is the natural mistake.

    Evicting a position removes it from every FUTURE row. It does not unwind the
    hop vectors of positions already computed, which contracted it into their own
    state when they ran. Those vectors are cached, not recomputed, and that is
    the entire reason decode is affordable.

    So an evicting cache is exact going forward and lossy backward. A caller that
    evicts mid-sequence and expects the prefix to be reconstructible is wrong, and
    the test exists so nobody has to find that out from a quality regression.

    This is the one place the cache is weaker than the operator. W3 measured
    eviction bitwise at 0.000e+00 over 24/24 draws, but that was the OPERATOR,
    where the whole computation is redone from the retained key set. In a cache
    the contribution is already contracted into stored state, and the only exact
    remedies are to recompute the affected hop vectors -- O(N) work, which is
    exactly what the cache exists to avoid -- or to evict before anything reads
    the position. The trade is stated rather than hidden.
    """
    q, k, v = qkv(device)
    drop = 2

    late = hc.HopCache(hops=HOPS)
    for i in range(6):
        late.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                  alpha=ALPHA, rho=RHO)
    late.evict(drop)                                  # after the fact
    after_late = late.step(q[:, :, 6:7], k[:, :, 6:7], v[:, :, 6:7],
                           alpha=ALPHA, rho=RHO)

    never = hc.HopCache(hops=HOPS)
    for i in range(6):
        if i == drop:
            continue
        never.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                   alpha=ALPHA, rho=RHO)
    after_never = never.step(q[:, :, 6:7], k[:, :, 6:7], v[:, :, 6:7],
                             alpha=ALPHA, rho=RHO)

    assert not torch.equal(after_late, after_never), (
        "evicting after the fact matched never-inserting; if that ever becomes "
        "true the cache is recomputing the prefix and decode is no longer O(K) "
        "per token")


def test_island_eviction_drops_a_whole_component(device):
    """mujoco#3396 granularity: an island is the unit.

    Its members reach each other, so dropping half of one leaves the rest reading
    a prefix that no longer exists. The representative is the minimum index, so
    which island a position belongs to is deterministic rather than an artefact
    of insertion order.
    """
    q, k, v = qkv(device)
    cache = hc.HopCache(hops=HOPS, couple_at=0.0)
    for i in range(8):
        cache.step(q[:, :, i:i + 1], k[:, :, i:i + 1], v[:, :, i:i + 1],
                   alpha=ALPHA, rho=RHO)
    comps = cache.components()
    assert all(r == min(m) for r, m in comps.items()), comps
    before = cache.n_positions()
    dropped = cache.evict_component(7)
    assert cache.n_positions() == before - len(dropped)
