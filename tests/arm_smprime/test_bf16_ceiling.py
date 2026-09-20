"""bf16 ceiling probe for `ceq/arm_smprime.py::numerator`, the shipped path
ending at line 247 (`return gh * e.to(gh.dtype), rh * e`). ONE FILE, THIS ONE:
`ceq/arm_smprime.py` belongs to another agent this round and is read here,
never edited; the bed this file needs is reproduced from the round's own
recipe rather than imported from `tests/curvature/` or `tests/cameron/`,
neither of which this file touches either.

THE OPEN QUESTION, verbatim from the round. The already-landed dead-entry
guard (`live = rh > 0` / `e = torch.where(live, e, 0)` at :245-246) stops a
DEAD entry's overflowing logit from poisoning the numerator: wherever the
gate already killed a cell (`rh == 0`), `e` is forced to `0` before the
complex multiply, so `gh * e` is `0 * 0` and never `0 * inf`. It does nothing
for a LIVE entry's own overflow, and one live entry is unconditional: the
diagonal, where the path product is empty (`rh_ii = 1`, `gh_ii = 1+0j`
exactly) regardless of `theta` anywhere else in the row. Complex
multiplication expands `(a+bi)(c+di)` to `(ac-bd)+(ad+bc)i`; at
`gh = 1+0j`, `e = inf+0j`, that is `(1*inf - 0*0) + (1*0 + 0*inf)i`, and
`0 * inf` is `nan` -- not a bf16 rounding artifact, a property of complex
multiplication by an infinite operand in every complex dtype torch ships.
Float64's overflow wall (`709.782712893384`) needed the round's bed scaled
`q`/`k` by x40 to reach; bf16 and float32 share their 8-bit exponent field
but NOT a wall -- bf16's shorter mantissa also shrinks its own largest
finite value below float32's, so bf16's own wall (computed below,
`~88.7189`) sits marginally under float32's (`~88.7228`) and bf16 overflows
`exp` FIRST. Both are still ~8x closer to zero than float64's wall, so the
round predicts x5.00 reaches the same collision. Whether it does, and whether
`torch.autocast(cpu, bfloat16)` even keeps `exp` in bf16 rather than quietly
promoting it, are MEASURED below by patching `torch.exp` for the duration of
one real call into the shipped function and reading the dtypes of the
tensors that actually passed through it -- never a second copy of the
arithmetic, never an edit to the file that owns it.

STATE THE DTYPE. Every number this file prints carries the `torch.dtype` of
the tensor it came from, on the same line, because a number without one is
not a number in this round.
"""
from __future__ import annotations

import contextlib
import math

import torch

from ceq import arm_smprime as smp

#: The two dtypes share an 8-bit exponent field but NOT an overflow wall:
#: bf16's shorter mantissa shrinks its own largest finite value below
#: float32's, so bf16 overflows `exp` FIRST. Both computed from
#: `torch.finfo(...).max` cast to float64 before `log`, never hardcoded --
#: a float32-precision `log` (no `dtype=torch.float64`) silently rounds the
#: wall itself and is what produced the retracted `88.72283935546875`.
F32_EXP_WALL = float(torch.log(torch.tensor(
    torch.finfo(torch.float32).max, dtype=torch.float64)))
BF16_EXP_WALL = float(torch.log(torch.tensor(
    torch.finfo(torch.bfloat16).max, dtype=torch.float64)))
FLOAT64_CEILING = 709.782712893384  # this round's own float64 measurement

#: (name, d_model, provenance) -- three shapes actually registered elsewhere
#: in this repository, not invented for this file. `ArmSMPrime` takes one
#: width (`d_model`); it is not multi-head, so a "config" here is that one
#: number, and the causal logit's scale is `1/sqrt(d_model)` regardless of
#: what a caller elsewhere calls "d_head".
REGISTERED_CONFIGS = (
    ("d16_arm_smprime_excess_count", 16,
     "ceq/hf/modeling_ceq.py:414 and tests/gate0/test_g10_arm_wiring.py:287, "
     "both citing d_model=16 as the shape the arm's own +37-parameter excess "
     "was counted at"),
    ("d24_registered_training_config", 24,
     "scout report: a registered training config for the argmax/glance/"
     "settled/softmax/twin family runs d_head=24, seq=64, hops=8"),
    ("d64_lm_and_hf_train_defaults", 64,
     "ceq/lm.py (D_MODEL=256, N_HEADS=4) and ceq/hf/train.py DEFAULTS "
     "(hidden_size=512, n_heads=8) -- two independent modules, same d_head"),
)

SEQ = 64     # ceq/arm_smprime.py::gradient_finiteness's own default `s`,
             # and this round's own overflow bed's `s`
BATCH = 8    # this round's own overflow bed's `n`
SEED = 3     # this round's own overflow bed's `seed`


@contextlib.contextmanager
def _exp_spy():
    """Patches `torch.exp` for the block's duration and yields a list this
    fills with one `(input_dtype, output_dtype, input_tensor, output_tensor)`
    tuple per call -- the only way to read `ceq/arm_smprime.py:232`'s `s`/`e`
    off the SHIPPED `numerator` without a second copy of its arithmetic and
    without editing the file (owned by another agent this round). Restores
    the real `torch.exp` on the way out, exception or not."""
    calls: list[tuple[torch.dtype, torch.dtype, torch.Tensor, torch.Tensor]] = []
    orig = torch.exp

    def spy(t, *a, **kw):
        out = orig(t, *a, **kw)
        calls.append((t.dtype, out.dtype, t, out))
        return out

    torch.exp = spy
    try:
        yield calls
    finally:
        torch.exp = orig


def _max_abs_finite(t: torch.Tensor) -> float:
    """`max |s|` over the entries `s` actually carries a number at -- the
    causal triangle, `masked_fill`'d to `-inf` above it. Excluding those is
    not a filter of convenience: `(-inf).abs()` is `inf`, and including the
    masked triangle would report `inf` for every step regardless of the live
    logits, which is a statement about the mask and not about training."""
    live = torch.isfinite(t)
    return float(t[live].detach().abs().max()) if live.any() else float("nan")


def _causal_mask(s: int) -> torch.Tensor:
    idx = torch.arange(s)
    return idx.unsqueeze(-1) >= idx.unsqueeze(-2)


def _cositing_bed(scale: float, dtype: torch.dtype, *, seed: int = SEED,
                   n: int = BATCH, s: int = SEQ, d: int = 16):
    """This round's own recipe, REPRODUCED and not imported -- the private
    draw it was originally measured on is not in the tree, and this file
    touches neither `tests/curvature/` nor `tests/cameron/` to borrow one.
    Batch `n`, sequence `s`, head dim `d`, `q`/`k` scaled by `scale`, gate
    closed (`u <- 0`) wherever a draw lands under `0.55`.

    Drawn once in float64 and cast down at the end, so `dtype` changes what
    the OPERATOR sees and not what got drawn: the float64 and bf16 readings
    this file compares are the same bed bitwise through the cast, up to
    `scale` itself.

    Occupancy is MEASURED and returned, not asserted to the round's own
    `0.555816650390625` -- that figure is `gradient_finiteness`'s default
    `n=512` bed, a different draw entirely; this file's `n=8` bed lands close
    (measured below) and is reported rather than forced to match."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, s, d, generator=g, dtype=torch.float64) * scale
    k = torch.randn(n, s, d, generator=g, dtype=torch.float64) * scale
    u = torch.rand(s, generator=g, dtype=torch.float64)
    dead = u < 0.55
    u = torch.where(dead, torch.zeros_like(u), u)
    th = (torch.rand(s, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    occupancy = float(dead.double().mean())
    return q.to(dtype), k.to(dtype), u.to(dtype), th.to(dtype), occupancy


# ================================================== THE DECISIVE MEASUREMENT

def test_exp_runs_in_bfloat16_under_autocast_in_both_regimes(capsys):
    """WHERE DOES `exp(s)` EVALUATE? Two regimes, because the round's own
    facts say they must differ: `blend`/`magnitude` (`clamp`, `lerp`) are not
    autocast-listed ops, so `u`/`theta` keep whatever dtype they ARRIVE at.
    `numerator`'s public callers hand it `u`/`theta` straight off
    `m_head`/`theta_head`, two `nn.Linear`s -- an autocast-listed op -- so
    under real training they arrive already cast to bf16. A raw-tensor call
    (this file's own co-siting bed, `_dead_shift_bed`-style fixtures) never
    passes through a `Linear`, so `u`/`theta` stay float32 there instead.

    MEASURED, not assumed, by patching `torch.exp` around one real call to
    `smp.numerator` and reading `(input dtype, output dtype)` straight off
    the tensors `ceq/arm_smprime.py:232` actually built."""
    torch.manual_seed(0)
    n, s, d = 4, SEQ, 16
    q_raw = torch.randn(n, s, d)
    k_raw = torch.randn(n, s, d)
    u_raw = torch.rand(s)
    th_raw = (torch.rand(s) * 2 - 1) * math.pi

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16), _exp_spy() as raw_calls:
        gh_raw, rh_raw = smp.numerator(q_raw, k_raw, u_raw, th_raw)
    assert len(raw_calls) == 1, ("expected exactly one torch.exp call inside "
                                  "numerator", raw_calls)
    s_dtype_raw, e_dtype_raw, _, _ = raw_calls[0]

    m_head = torch.nn.Linear(d, 1)
    theta_head = torch.nn.Linear(d, 1)
    wq = torch.nn.Linear(d, d, bias=False)
    wk = torch.nn.Linear(d, d, bias=False)
    x = torch.randn(n, s, d)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16), _exp_spy() as lin_calls:
        u_lin = m_head(x).squeeze(-1)
        th_lin = theta_head(x).squeeze(-1)
        q_lin, k_lin = wq(x), wk(x)
        gh_lin, rh_lin = smp.numerator(q_lin, k_lin, u_lin, th_lin)
    assert len(lin_calls) == 1, ("expected exactly one torch.exp call inside "
                                  "numerator", lin_calls)
    s_dtype_lin, e_dtype_lin, _, _ = lin_calls[0]

    with capsys.disabled():
        print("\n  [torch.autocast(cpu, bfloat16), raw float32 u/theta]")
        print(f"    s.dtype (dtype=torch.dtype) = {s_dtype_raw}")
        print(f"    e.dtype (dtype=torch.dtype) = {e_dtype_raw}")
        print(f"    gh.dtype = {gh_raw.dtype}   rh.dtype = {rh_raw.dtype}")
        print("  [torch.autocast(cpu, bfloat16), Linear-sourced u/theta]")
        print(f"    s.dtype (dtype=torch.dtype) = {s_dtype_lin}")
        print(f"    e.dtype (dtype=torch.dtype) = {e_dtype_lin}")
        print(f"    gh.dtype = {gh_lin.dtype}   rh.dtype = {rh_lin.dtype}")

    # THE ANSWER: exp runs IN bfloat16 in both regimes -- autocast does not
    # promote it to float32. Pinned so a torch upgrade that changes this is
    # caught rather than silently re-read as the opposite finding.
    assert s_dtype_raw == torch.bfloat16, s_dtype_raw
    assert e_dtype_raw == torch.bfloat16, e_dtype_raw
    assert s_dtype_lin == torch.bfloat16, s_dtype_lin
    assert e_dtype_lin == torch.bfloat16, e_dtype_lin
    # `gh`'s dtype depends on the regime: `gate`'s `torch.polar` is not
    # autocast-listed, so autocast runs it in float32 regardless of the
    # ambient cast -- `_ctype` (ceq/arm_smprime.py:108-109) then sees "not
    # float32" whenever `u`'s OWN dtype was bfloat16 (the Linear regime) and
    # returns complex128, a double-precision gate multiplied against a
    # bfloat16 `e`. The raw regime keeps `u` float32 throughout, so `_ctype`
    # takes its other branch and `gh` stays complex64.
    assert gh_raw.dtype == torch.complex64, gh_raw.dtype
    assert gh_lin.dtype == torch.complex128, gh_lin.dtype
    assert rh_raw.dtype == torch.float32, rh_raw.dtype       # float32 * bf16 -> float32
    assert rh_lin.dtype == torch.bfloat16, rh_lin.dtype       # bfloat16 * bf16 -> bfloat16


def test_the_two_walls_have_different_sources(capsys):
    """bf16 and float32 share their 8-bit exponent field but NOT an overflow
    wall: it is the MANTISSA, not the exponent, that decides `finfo(...).max`
    once the exponent field is fixed, and bf16's shorter mantissa gives it a
    smaller largest finite value than float32's, hence a smaller (earlier)
    `exp` wall. THE SURVIVING CLAIM IS THE STRONGER ONE: bf16 narrows the
    ceiling, it does not widen or share it. Both walls, and a value in the
    interval where the two dtypes disagree, are measured here rather than
    trusted, so the constants cannot drift from the arithmetic again."""
    f32_wall = float(torch.log(torch.tensor(torch.finfo(torch.float32).max,
                                             dtype=torch.float64)))
    bf16_wall = float(torch.log(torch.tensor(torch.finfo(torch.bfloat16).max,
                                              dtype=torch.float64)))
    f64_wall = float(torch.log(torch.tensor(torch.finfo(torch.float64).max,
                                             dtype=torch.float64)))
    with capsys.disabled():
        print(f"\n  ln(float32.max), computed in float64 (dtype=torch.float64)"
              f" = {f32_wall!r}")
        print(f"  ln(bfloat16.max), computed in float64 (dtype=torch.float64)"
              f" = {bf16_wall!r}  (smaller: bf16's own mantissa shrinks "
              f"its representable max below float32's)")
        print(f"  ln(float64.max), computed in float64 (dtype=torch.float64)"
              f" = {f64_wall!r}")
        print(f"  ratio f64_wall / f32_wall = {f64_wall / f32_wall!r}"
              f"  (the round's own 'exactly 8x closer')")
    assert f32_wall == F32_EXP_WALL, f32_wall
    assert bf16_wall == BF16_EXP_WALL, bf16_wall
    assert f64_wall == FLOAT64_CEILING, f64_wall
    # THE SURVIVING CLAIM, bitwise checkable: bf16's wall is strictly the
    # smaller one. Not "close to" or "shares" -- strictly less than.
    assert bf16_wall < f32_wall, (bf16_wall, f32_wall)
    #: the round states the f64/f32 ratio as `7.999999978`, not `8` exactly --
    #: float64's wall is float64's OWN max's log (double precision), while
    #: float32's wall is float32's max's log (smaller number, same double
    #: precision now that it is no longer rounded to float32 on the way
    #: out), so the two carry different rounding error and the ratio is
    #: close to, not exactly, 8.
    assert abs(f64_wall / f32_wall - 8.0) < 5e-8, f64_wall / f32_wall

    # THE DISAGREEMENT INTERVAL [bf16_wall, f32_wall): a real x in here is
    # small enough that exp(x) is finite in float32, but once that finite
    # float32 result is represented in bfloat16 -- exactly what autocast
    # does per test_exp_runs_in_bfloat16_under_autocast_in_both_regimes --
    # it exceeds bfloat16's own largest finite value and reads inf. This is
    # the measurable form of "bf16 overflows first", not a restatement of it.
    x = 88.7215
    assert bf16_wall < x < f32_wall, (bf16_wall, x, f32_wall)
    f32_exp = torch.exp(torch.tensor(x, dtype=torch.float32))
    bf16_exp = f32_exp.to(torch.bfloat16)
    with capsys.disabled():
        print(f"\n  x = {x!r}, inside [bf16_wall, f32_wall)")
        print(f"    exp(x) in float32 (dtype=torch.float32) = {f32_exp.item()!r}")
        print(f"    that same value cast to bfloat16 (dtype=torch.bfloat16)"
              f" = {bf16_exp.item()!r}")
    assert torch.isfinite(f32_exp), f32_exp.item()
    assert torch.isinf(bf16_exp), bf16_exp.item()


# ============================================= THE CO-SITING, AT x5.00 vs x40

def test_cositing_reproduces_on_the_live_diagonal_at_bf16s_wall(capsys):
    """Does the round's float64 finding -- a dead entry co-sited with an
    overflowing logit poisons the LIVE diagonal with `inf+nan*j`, past the
    already-landed guard -- reproduce in bf16 at x5.00 rather than float64's
    x40? MEASURED on this file's own reproduction of the bed (`_cositing_bed`,
    seed 3, n=8, s=64, d=16), at three cells: float64 x5 (a negative control:
    nowhere near float64's own wall), float64 x40 (the round's own float64
    finding, reproduced independently), and bf16 x5.00 under
    `torch.autocast(cpu, bfloat16)` (the round's open question)."""
    s = SEQ

    def _measure(scale, dtype, autocast):
        q, k, u, th, occupancy = _cositing_bed(scale, dtype)
        ctx = (torch.autocast(device_type="cpu", dtype=torch.bfloat16)
               if autocast else contextlib.nullcontext())
        with ctx, _exp_spy() as calls:
            gh, rh = smp.numerator(q, k, u, th)
        s_dtype, e_dtype, s_tensor, e_tensor = calls[0]
        cm = _causal_mask(s).unsqueeze(0).expand(q.shape[0], -1, -1)
        diagmask = torch.eye(s, dtype=torch.bool).unsqueeze(0).expand(q.shape[0], -1, -1)
        nan_any = (torch.isnan(gh.real) | torch.isnan(gh.imag)) & cm
        inf_any = (torch.isinf(gh.real) | torch.isinf(gh.imag)) & cm
        return dict(
            occupancy=occupancy, s_dtype=s_dtype, e_dtype=e_dtype,
            max_abs_s=_max_abs_finite(s_tensor), e_overflowed=bool(torch.isinf(e_tensor).any()),
            total_causal=int(cm.sum()), nan_total=int(nan_any.sum()),
            nan_on_diagonal=int((nan_any & diagmask).sum()),
            nan_off_diagonal=int((nan_any & ~diagmask).sum()),
            inf_only=int((inf_any & ~nan_any).sum()),
        )

    ctrl = _measure(5.0, torch.float64, autocast=False)
    f64 = _measure(40.0, torch.float64, autocast=False)
    bf16 = _measure(5.0, torch.float32, autocast=True)

    with capsys.disabled():
        for label, r in (("float64, x5.00 (control)", ctrl),
                         ("float64, x40  (round's own wall)", f64),
                         ("bf16 via autocast, x5.00 (this file's question)", bf16)):
            print(f"\n  {label}")
            print(f"    occupancy (dtype=float, measured) = {r['occupancy']!r}")
            print(f"    max|s| over causal cells (dtype={r['s_dtype']}) = "
                  f"{r['max_abs_s']!r}")
            print(f"    exp() overflowed to inf (dtype={r['e_dtype']}) = "
                  f"{r['e_overflowed']}")
            print(f"    causal cells total (dtype=int) = {r['total_causal']}")
            print(f"    non-finite gh: nan-any = {r['nan_total']} "
                  f"(on diagonal = {r['nan_on_diagonal']}, "
                  f"off diagonal = {r['nan_off_diagonal']}), "
                  f"inf-only = {r['inf_only']}")

    # THE CONTROL: float64 at x5.00 never reaches float64's own wall, so no
    # entry is non-finite. A bed that already breaks here would be measuring
    # the construction, not the ceiling (`tests/curvature`'s planted negative
    # (c), reproduced as a control rather than borrowed).
    assert ctrl["nan_total"] == 0 and ctrl["inf_only"] == 0, ctrl

    # THE ROUND'S OWN FLOAT64 FINDING, reproduced independently: the landed
    # guard keeps the diagonal's unconditionally-`1+0j` gate from turning the
    # overflow into a nan, so what reaches gh at x40 is inf, not nan.
    assert f64["nan_total"] == 0, f64
    assert f64["nan_on_diagonal"] == 0, f64
    assert f64["nan_off_diagonal"] == 0, f64
    assert f64["inf_only"] > 0, f64

    # THE ANSWER: it reproduces in bf16, at x5.00 -- exp() overflows there too,
    # and the same guard keeps the result inf rather than nan, confirming the
    # round's 8x-closer prediction rather than merely restating it.
    assert bf16["e_overflowed"], bf16
    assert bf16["nan_total"] == 0, bf16
    assert bf16["nan_on_diagonal"] == 0, bf16
    assert bf16["nan_off_diagonal"] == 0, bf16
    assert bf16["inf_only"] > 0, bf16


# ===================================== TASK 1's BAR: 200 STEPS, g LEARNABLE

def test_registered_configs_stay_under_the_bf16_ceiling_at_200_steps(capsys):
    """BAR: `max|s| < BF16_EXP_WALL` for every registered config, over
    200 gradient steps with `g` left learnable (`trainable_heads()`, the
    module's own instruction: `identity_heads()` leaves `g` at a critical
    point with an exactly-zero gradient -- "Train from `trainable_heads()`
    instead" is the docstring's own line). `BF16_EXP_WALL`, not
    `F32_EXP_WALL`, is the bar: this forward runs with no autocast so `s`
    is measured in float32, but production training runs under
    `torch.autocast(cpu, bfloat16)` (`test_exp_runs_in_bfloat16_under_...`),
    where bf16's narrower wall is the one that actually binds.

    KILL: any registered config crossing the wall during ORDINARY training
    (no adversarial q/k scale, no planted dead gate) would mean the ceiling
    is OBSERVED rather than merely constructible, which is the round's own
    framing of what a crossing here would mean.

    Ordinary AdamW training at `lr=1e-3` (`gradient_finiteness`'s own
    setting) on random data has no reason to push a causal logit within 20x
    of an overflow wall in 200 steps; this is the measurement that says
    whether that expectation holds for the shapes this repository actually
    registers, rather than an assumption standing in for one."""
    n = BATCH
    results = []
    for name, d_model, provenance in REGISTERED_CONFIGS:
        torch.manual_seed(0)
        arm = smp.ArmSMPrime(SEQ, d_model=d_model).trainable_heads()
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(n, SEQ, d_model, generator=gen)
        y = torch.randn(n, generator=gen)
        opt = torch.optim.AdamW(arm.parameters(), lr=1e-3)
        trace: list[float] = []
        for _ in range(200):
            opt.zero_grad()
            with _exp_spy() as calls:
                pred = arm(x)
            assert len(calls) == 1, ("expected exactly one torch.exp call "
                                      "per forward", len(calls))
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            opt.step()
            trace.append(_max_abs_finite(calls[0][2]))
        results.append((name, d_model, provenance, max(trace), trace[-1]))

    with capsys.disabled():
        print(f"\n  BAR: max|s| (dtype=torch.float32) < {BF16_EXP_WALL!r} "
              f"over 200 AdamW steps, g learnable, n={n}, s={SEQ}")
        for name, d_model, provenance, worst, last in results:
            print(f"    {name} (d_model={d_model}): "
                  f"max over 200 steps = {worst!r}, final step = {last!r}")
            print(f"      registered at: {provenance}")

    for name, d_model, provenance, worst, last in results:
        assert worst < BF16_EXP_WALL, (
            f"KILL: {name} (d_model={d_model}) crossed the bf16 exp "
            f"wall ({BF16_EXP_WALL}) during ordinary 200-step training "
            f"(max|s| = {worst}) -- the wall is OBSERVED, not merely "
            f"constructible.")
