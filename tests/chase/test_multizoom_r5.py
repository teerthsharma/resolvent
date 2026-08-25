"""R5's falsifiers. These decide whether R5 lives or is deleted.

R5 -- multi-zoom reading. Fine nearby, coarse far, self-similar, near-linear cost,
with a STATED ACCURACY BOUND for the coarsening.

Two measured facts from REQUIREMENTS.md that these tests are built to attack:
  * the obvious hierarchical dyadic schedule scored 0.39 on the random->oracle axis
    against 0.975 for plain sliding-window + sinks, and won 0 of 15 cells;
  * its builder cost 53x the attention it scheduled (42.12 ms vs 0.792 ms at 8192).

The structural reason given: an H-matrix far-field is LOW-RANK, not sparse, and a
block schedule can only keep or drop. So the falsifier that matters most is
`test_summarising_beats_dropping_at_matched_budget`. If summarising does not beat
dropping on the same geometry at the same cost, the whole mechanism bought
nothing and R5 is deleted.

The random->oracle axis is reproduced here so the numbers are comparable to the
0.39 / 0.975 already on record:

    score = (err_random - err_method) / (err_random - err_oracle)

0 = no better than spending the budget at random. 1 = as good as an oracle that
picked blocks by their true attention mass. Error is relative L2 against exact
dense causal attention.

Every test parametrizes over device.
"""

import math
import time

import pytest
import torch

from conftest import DEVICES, requires_triton

from ceq.multizoom import (
    coarsening_bound,
    drop_reference,
    exact_causal_attention,
    multizoom_reference,
    plan_for,
    window_sink_reference,
)

BLOCK = 32
DIM = 32


# ------------------------------------------------------------------ workloads


def clustered_kv(seq, dim, n_clusters, spread, device, seed=0):
    """Keys/values in `n_clusters` contiguous runs with within-run radius `spread`.

    `spread` is the knob the accuracy bound is written in terms of. Small spread =
    a far field that is genuinely summarisable. Large spread = iid noise, where
    the bound goes vacuous and pooling should NOT be expected to work. Testing
    both is how the bound earns the right to be called predictive.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    per = math.ceil(seq / n_clusters)
    ck = torch.randn(n_clusters, dim, generator=g)
    cv = torch.randn(n_clusters, dim, generator=g)
    idx = torch.arange(seq) // per
    k = ck[idx] + spread * torch.randn(seq, dim, generator=g)
    v = cv[idx] + spread * torch.randn(seq, dim, generator=g)
    q = torch.randn(seq, dim, generator=g)
    return [x.to(device)[None, None] for x in (q, k, v)]


def budgeted_block_attention(q, k, v, plan, chooser):
    """Causal attention restricted to whichever RAW blocks `chooser` returns.

    The keep-or-drop family: window+sinks, random, and oracle are all this
    function with a different chooser. Budget is expressed in blocks so it is
    directly comparable to the number of units multizoom spends.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = torch.zeros_like(q)
    pos = torch.arange(plan.seq, device=q.device)
    for qb in range(plan.num_blocks):
        qs, qe = qb * plan.block, min((qb + 1) * plan.block, plan.seq)
        blocks = chooser(qb, len(plan.units(qb)))
        rows = torch.cat([
            torch.arange(b * plan.block, min((b + 1) * plan.block, plan.seq),
                         device=q.device) for b in blocks
        ])
        kk, vv = k.index_select(-2, rows), v.index_select(-2, rows)
        scores = (q[..., qs:qe, :] @ kk.transpose(-1, -2)) * scale
        scores = scores.masked_fill(rows[None, :] > pos[qs:qe, None], float("-inf"))
        out[..., qs:qe, :] = torch.softmax(scores, dim=-1) @ vv
    return out


def _choosers(q, k, v, plan, sink_blocks, seed=0):
    scale = 1.0 / math.sqrt(q.shape[-1])
    g = torch.Generator(device="cpu").manual_seed(seed)

    def recent(qb, n):
        sinks = list(range(min(sink_blocks, qb + 1)))
        rest = [b for b in range(qb, -1, -1) if b not in sinks][: max(0, n - len(sinks))]
        return sorted(set(sinks + rest + [qb]))

    def random_blocks(qb, n):
        sinks = list(range(min(sink_blocks, qb + 1)))
        pool = [b for b in range(qb + 1) if b not in sinks and b != qb]
        take = max(0, n - len(sinks) - 1)
        perm = torch.randperm(len(pool), generator=g).tolist()[:take]
        return sorted(set(sinks + [pool[i] for i in perm] + [qb]))

    def oracle(qb, n):
        qs, qe = qb * plan.block, min((qb + 1) * plan.block, plan.seq)
        logits = (q[..., qs:qe, :] @ k.transpose(-1, -2)) * scale
        pos = torch.arange(plan.seq, device=q.device)
        logits = logits.masked_fill(pos[None, :] > pos[qs:qe, None], float("-inf"))
        mass = torch.softmax(logits, dim=-1).sum(dim=(0, 1, 2))
        nb = qb + 1
        pad = (-plan.seq) % plan.block
        bm = torch.nn.functional.pad(mass, (0, pad)).reshape(-1, plan.block).sum(-1)
        chosen = torch.topk(bm[:nb], k=min(n, nb)).indices.tolist()
        return sorted(set(chosen + [qb]))

    return recent, random_blocks, oracle


def axis_score(err_method, err_random, err_oracle):
    denom = err_random - err_oracle
    if denom <= 0:
        return float("nan")
    return (err_random - err_method) / denom


def rel_err(a, ref):
    return float((a - ref).norm() / ref.norm())


# ------------------------------------------------- THE test: summarize vs drop


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spread", [0.05, 0.3, 1.5])
def test_summarising_beats_dropping_at_matched_budget(device, spread):
    """Identical geometry, identical token budget, summarise vs keep-or-drop.

    `drop_reference` reads exactly the same units multizoom does and spends
    exactly the same number of attended tokens; the only difference is that a
    coarse unit contributes its first `block` RAW tokens instead of `block`
    pooled means. That isolates the one variable R5's hazard note is about.

    Deletion criterion: if dropping is as good as summarising, the coarse level
    is decoration and R5 dies.
    """
    seq = 64 * BLOCK
    q, k, v = clustered_kv(seq, DIM, n_clusters=16, spread=spread, device=device)
    plan = plan_for(seq, block=BLOCK, window_blocks=2, sink_blocks=1)
    ref = exact_causal_attention(q, k, v)

    e_sum = rel_err(multizoom_reference(q, k, v, plan), ref)
    e_drop = rel_err(drop_reference(q, k, v, plan), ref)
    print(f"\n[spread={spread}] summarise={e_sum:.4f} drop={e_drop:.4f} "
          f"ratio={e_drop / max(e_sum, 1e-12):.2f}x")
    assert e_sum < e_drop, (
        f"summarising ({e_sum:.4f}) did not beat dropping ({e_drop:.4f}) at "
        f"spread={spread}. The coarse level bought nothing; delete R5."
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spread", [0.05, 0.3, 1.5])
def test_random_to_oracle_axis_against_window_plus_sinks(device, spread):
    """The axis the dyadic schedule scored 0.39 on and window+sinks scored 0.975.

    Not an assertion of victory -- a measurement, printed, at matched budget.
    The only hard assertion is that multizoom is not WORSE than random, which is
    what 0.39 effectively was.
    """
    seq = 64 * BLOCK
    q, k, v = clustered_kv(seq, DIM, n_clusters=16, spread=spread, device=device)
    plan = plan_for(seq, block=BLOCK, window_blocks=2, sink_blocks=1)
    ref = exact_causal_attention(q, k, v)
    recent, rnd, oracle = _choosers(q, k, v, plan, sink_blocks=1)

    e_rand = rel_err(budgeted_block_attention(q, k, v, plan, rnd), ref)
    e_orac = rel_err(budgeted_block_attention(q, k, v, plan, oracle), ref)
    e_win = rel_err(budgeted_block_attention(q, k, v, plan, recent), ref)
    e_mz = rel_err(multizoom_reference(q, k, v, plan), ref)

    budget = sum(len(plan.units(qb)) for qb in range(plan.num_blocks))
    dense = plan.num_blocks * (plan.num_blocks + 1) // 2
    print(f"\n[spread={spread}] blocks {budget}/{dense} = "
          f"{budget / dense:.1%} of dense")
    print(f"  err   random={e_rand:.4f} oracle={e_orac:.4f} "
          f"window+sinks={e_win:.4f} multizoom={e_mz:.4f}")
    print(f"  axis  window+sinks={axis_score(e_win, e_rand, e_orac):.3f} "
          f"multizoom={axis_score(e_mz, e_rand, e_orac):.3f}")

    # The axis is UNDEFINED whenever the block-granular oracle falls below
    # random, which it does here: a single top-k choice is shared by all
    # `block` queries in the query block, and those queries want different
    # blocks. Verified as a real oracle, not a bug -- at full budget it
    # reproduces dense to 2.5e-07 and its error falls monotonically with budget
    # (0.61 -> 0.22 -> 0.0000 at 1x/2x/4x). Reported rather than papered over.
    assert e_mz < e_rand, (
        f"multizoom ({e_mz:.4f}) is no better than spending the same budget at "
        f"random ({e_rand:.4f}); delete R5."
    )
    assert e_mz < e_win, (
        f"multizoom ({e_mz:.4f}) did not beat sliding-window+sinks ({e_win:.4f}), "
        f"the baseline that scored 0.975 and beat the dyadic schedule 15/15; "
        f"delete R5."
    )
    assert e_mz < e_orac, (
        f"multizoom ({e_mz:.4f}) did not beat the same-budget block oracle "
        f"({e_orac:.4f}); the coarse level is no better than an unimplementable "
        f"top-k, so it has no reason to exist; delete R5."
    )


# --------------------------------------------------------- the accuracy bound


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("spread", [0.02, 0.1, 0.5])
def test_measured_coarsening_error_never_exceeds_the_stated_bound(device, spread):
    """R5 demands a STATED ACCURACY BOUND. This is it, checked against reality.

    If the measured infinity-norm error ever exceeds the predicted bound, the
    bound is wrong, R5 has no accuracy bound, and R5 is deleted.
    """
    seq = 16 * BLOCK
    for seed in range(6):
        q, k, v = clustered_kv(seq, DIM, 8, spread, device, seed=seed)
        plan = plan_for(seq, block=BLOCK, window_blocks=2, sink_blocks=1)
        got = multizoom_reference(q, k, v, plan)
        ref = exact_causal_attention(q, k, v)
        measured = float((got - ref).abs().max())
        predicted = coarsening_bound(q, k, v, plan)
        assert measured <= predicted + 1e-6, (
            f"BOUND VIOLATED at spread={spread} seed={seed}: measured "
            f"{measured:.6e} > predicted {predicted:.6e}. The stated accuracy "
            f"bound is wrong; delete R5."
        )


@pytest.mark.parametrize("device", DEVICES)
def test_bound_is_non_vacuous_when_the_far_field_is_tight(device):
    """A bound that never says anything is not a bound. Measure where it bites.

    Vacuous means "no better than the trivial statement that both outputs lie in
    the convex hull of the values", i.e. >= the value range. The bound must beat
    that for SOME tight far field, or R5's accuracy-bound clause is decoration.
    """
    seq = 16 * BLOCK
    rows = []
    tight = False
    for spread in (0.001, 0.005, 0.02, 0.1, 0.5):
        q, k, v = clustered_kv(seq, DIM, 8, spread, device, seed=1)
        plan = plan_for(seq, block=BLOCK, window_blocks=2, sink_blocks=1)
        trivial = float((v.amax(-2) - v.amin(-2)).amax())
        b = coarsening_bound(q, k, v, plan)
        measured = float((multizoom_reference(q, k, v, plan)
                          - exact_causal_attention(q, k, v)).abs().max())
        rows.append((spread, measured, b, trivial))
        if b < trivial:
            tight = True
    print("\n  spread   measured      bound      trivial   informative")
    for s, m, b, t in rows:
        print(f"  {s:<7} {m:.3e}  {b:.3e}  {t:.3e}  {'YES' if b < t else 'no'}")
    assert tight, (
        "the coarsening bound is vacuous at every spread tested -- it never says "
        "anything the convex hull does not already say. R5 has no usable accuracy "
        "bound; delete R5."
    )


# ------------------------------------------------------------- cost structure


@pytest.mark.parametrize("device", DEVICES)
def test_pool_builder_is_cheap_against_the_attention_it_feeds(device):
    """The dyadic builder cost 53x the attention it scheduled. Measure this one.

    Deletion criterion: a builder that costs more than the attention it enables
    is not a speedup mechanism no matter what it does to quality.
    """
    from ceq.multizoom import build_pyramid

    seq = 128 * BLOCK
    q, k, v = clustered_kv(seq, DIM, 32, 0.2, device)
    plan = plan_for(seq, block=BLOCK, window_blocks=2, sink_blocks=1)

    def timeit(fn, n=5):
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    t_build = timeit(lambda: (build_pyramid(k, plan.levels),
                              build_pyramid(v, plan.levels)))
    t_attn = timeit(lambda: exact_causal_attention(q, k, v))
    print(f"\n  pyramid build {t_build * 1e3:.3f} ms | dense attention "
          f"{t_attn * 1e3:.3f} ms | ratio {t_build / t_attn:.3f}x")
    assert t_build < t_attn, (
        f"the pooling builder costs {t_build / t_attn:.1f}x the attention it "
        f"feeds. The dyadic schedule died of exactly this at 53x; delete R5."
    )


@pytest.mark.parametrize("device", DEVICES)
def test_attended_tokens_grow_as_n_log_n_not_quadratically(device):
    """R5: 'Cost grows near-linearly with context. Quadratic cost fails.'

    Counted exactly, not timed, so this is free of clock noise. The claim being
    tested is the specific one the construction makes: units per query block is
    LINEAR IN log2(seq), which makes total attended tokens Theta(N log N).

    A raw log-log slope threshold would have been a tunable number, so the test
    asserts the model instead: fit units-per-query-block against log2(seq) and
    require the fit to be near-exact. Quadratic cost cannot pass that fit, and
    neither can anything else that is not N log N.
    """
    pts = []
    for shift in range(4, 15):
        seq = (1 << shift) * BLOCK
        plan = plan_for(seq, block=BLOCK, window_blocks=2, sink_blocks=1)
        units = sum(len(plan.units(qb)) for qb in range(plan.num_blocks))
        pts.append((seq, units, units / plan.num_blocks, units * plan.block))

    xs = torch.tensor([math.log2(s) for s, _, _, _ in pts], dtype=torch.float64)
    ys = torch.tensor([u for _, _, u, _ in pts], dtype=torch.float64)
    xc, yc = xs - xs.mean(), ys - ys.mean()
    slope = float((xc * yc).sum() / (xc ** 2).sum())
    resid = yc - slope * xc
    r2 = float(1 - (resid ** 2).sum() / (yc ** 2).sum())

    lx = torch.tensor([math.log(s) for s, _, _, _ in pts], dtype=torch.float64)
    ly = torch.tensor([math.log(t) for _, _, _, t in pts], dtype=torch.float64)
    lxc, lyc = lx - lx.mean(), ly - ly.mean()
    ll_slope = float((lxc * lyc).sum() / (lxc ** 2).sum())

    print("\n     seq  units/qblk   attended tok   frac of dense")
    for seq, _, upb, tok in pts:
        print(f"  {seq:>7} {upb:>11.2f} {tok:>14} {tok / (seq * (seq + 1) / 2):>14.4%}")
    print(f"  units/qblk = {slope:.3f} * log2(seq) + c, R^2 = {r2:.6f}")
    print(f"  total attended tokens log-log slope = {ll_slope:.3f} (dense causal = 2.0)")

    assert r2 > 0.999, (
        f"units per query block is not linear in log2(seq) (R^2={r2:.4f}); the "
        f"cost is not N log N. Quadratic cost fails R5; delete R5."
    )
    assert ll_slope < 1.5, (
        f"attended tokens scale as seq^{ll_slope:.2f}, too close to quadratic; "
        f"delete R5."
    )


@requires_triton
def test_kernel_wall_clock_scales_near_linearly():
    """The same claim, on the clock, through the real kernel."""
    from ceq.mz_kernel import multizoom_attention

    pts = []
    for shift in (5, 6, 7, 8, 9):
        seq = (1 << shift) * BLOCK
        q, k, v = [torch.randn(1, 4, seq, 64, device="cuda", dtype=torch.float16)
                   for _ in range(3)]
        plan = plan_for(seq, block=BLOCK, window_blocks=2, sink_blocks=1)
        multizoom_attention(q, k, v, plan)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            multizoom_attention(q, k, v, plan)
        torch.cuda.synchronize()
        pts.append((seq, (time.perf_counter() - t0) / 10))
    xs = torch.tensor([math.log(s) for s, _ in pts])
    ys = torch.tensor([math.log(t) for _, t in pts])
    slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum()
                  / ((xs - xs.mean()) ** 2).sum())
    print("\n  " + ", ".join(f"{s}:{t * 1e3:.3f}ms" for s, t in pts))
    print(f"  wall-clock log-log slope = {slope:.3f}")
    assert slope < 1.4, f"kernel wall clock scales as seq^{slope:.2f}"
