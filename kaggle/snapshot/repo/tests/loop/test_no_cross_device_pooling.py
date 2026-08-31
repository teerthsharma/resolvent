"""CONDITION 1, V15_NEPTUNE_SYSTEMS.md: a CUDA cell may never be pooled with a
CPU-taken cell.

WHY THIS IS A SEPARATE GUARD FROM `it11_verdict.by_seed`. MISTAKES.md M-10
measured *thread count alone* moving `eval_nrmse` by 2.345e-3 -- 0.464 of the
pre-registered equivalence margin `Delta_eq` -- and `by_seed` already refuses a
same-seed, same-`threads` disagreement for that reason. But it buckets rows by
`threads` ONLY: two rows sharing a seed and a thread count, one taken on cpu
and one on cuda, fall into the SAME bucket as far as `by_seed` is concerned. It
would in fact raise on such a pair too, purely by accident -- because the
values happen to differ -- but its message blames "something no field of the
cell record names", which stopped being true the moment `device` became a
field, and it would say nothing if two devices' readings ever landed within
1e-9 of each other by coincidence.

`scale.r10_capacity_sweep.refuse_cross_device_pool` is the guard actually named
by condition 1: it inspects `device` directly and refuses ANY spread,
unconditionally -- not "pick the majority device", the way `by_seed` picks the
majority thread count. Device is named a LARGER perturbation than threads, and
condition 1 states no accommodation for it.
"""
from __future__ import annotations

import pytest

from scale.r10_capacity_sweep import refuse_cross_device_pool


def _cell(device: str, seed: int = 0, eval_nrmse: float = 0.95) -> dict:
    return dict(t="cell", t_star=2, n_train=2048, seed=seed, threads=8,
               steps=150, eval_nrmse=eval_nrmse, device=device)


def test_must_fire_a_pool_spanning_two_devices_raises():
    """THE MUST-FIRE. Eight cpu seeds and eight cuda seeds, same (t*, n,
    threads, steps) -- exactly the shape a careless verdict function would
    concatenate before computing a seed CI. The guard must raise before any
    mean or CI is ever computed over it."""
    rows = ([_cell("cpu", seed=s, eval_nrmse=0.95 + 0.001 * s) for s in range(8)]
            + [_cell("cuda", seed=s, eval_nrmse=0.95 + 0.001 * s) for s in range(8)])
    with pytest.raises(ValueError, match="devices"):
        refuse_cross_device_pool(rows)


def test_a_single_device_pool_does_not_raise():
    """THE CONTROL. A guard that raises on everything would pass the must-fire
    above for the wrong reason. A same-device pool -- the ordinary, intended
    case -- must return cleanly and name that device."""
    rows = [_cell("cpu", seed=s) for s in range(8)]
    assert refuse_cross_device_pool(rows) == "cpu"

    rows = [_cell("cuda", seed=s) for s in range(8)]
    assert refuse_cross_device_pool(rows) == "cuda"


def test_a_row_missing_device_is_read_as_cpu():
    """Every journal this file wrote before the `--device` flag existed has no
    `device` field at all, and it was cpu-only by construction. Treating a
    missing field as an unknown third bucket would make an old journal refuse
    to pool with a fresh `--device cpu` run of the same cell, which is not the
    defect this guard exists to catch -- so old and new cpu rows must still
    pool together, and old rows must still refuse against a cuda row."""
    old = dict(t="cell", t_star=2, n_train=2048, seed=0, threads=8, steps=150,
              eval_nrmse=0.95)                                    # no "device"
    new_cpu = _cell("cpu", seed=1)
    assert refuse_cross_device_pool([old, new_cpu]) == "cpu"

    new_cuda = _cell("cuda", seed=1)
    with pytest.raises(ValueError, match="devices"):
        refuse_cross_device_pool([old, new_cuda])


def test_empty_pool_reports_cpu_rather_than_crashing():
    """No rows to disagree over -- returns the default rather than raising
    KeyError/IndexError on `.pop()` of an empty set."""
    assert refuse_cross_device_pool([]) == "cpu"


def _bar(device: str, trained_two_feature: float = 0.0139816) -> dict:
    """A `t="bar"` journal record as `r10_capacity_sweep.main()` now writes it."""
    return dict(t="bar", ok=True, why="BAR CALIBRATED", device=device,
                predict_the_mean=1.0, payload_only=1.2272242, oracle=0.0,
                flipper_dependence=1.4012437,
                trained_two_feature=trained_two_feature)


def test_must_fire_a_pool_of_bar_records_spanning_two_devices_raises():
    """V16: `calibrate_bar` now takes a `device` and the `bar` record carries
    one, so the BAR is a per-device measurement and not a property of the task.

    This is the must-fire one level above the cell rows. A cpu bar and a cuda
    bar are two readings of two instruments; crediting a cuda cell against the
    cpu bar is exactly MISTAKES.md V-22 -- a threshold carried in from another
    system -- and the guard has to refuse the set before any of it is read.
    The two records below are the MEASURED cpu and cuda `trained_two_feature`
    at `e3_t2, n=4096, steps=600, seed=0` (V16_BAR_RECERT.md), i.e. a pair that
    differs only in the last significant figures, which is precisely the pair a
    value-comparing guard would wave through."""
    rows = [_bar("cpu", 0.013981593578261986),
            _bar("cuda", 0.013981630466969725)]
    with pytest.raises(ValueError, match="devices"):
        refuse_cross_device_pool(rows)


def test_a_bar_record_pools_with_the_cells_it_certified():
    """THE CONTROL, and the case that must NOT refuse. A cuda bar and the cuda
    cells it gated are one device and belong together -- a guard that refused
    this would make the certified path unrunnable rather than safe."""
    rows = [_bar("cuda")] + [_cell("cuda", seed=s) for s in range(8)]
    assert refuse_cross_device_pool(rows) == "cuda"


def test_a_cuda_cell_may_not_be_read_against_a_cpu_bar():
    """The mixed shape the V15 abort was protecting against, now expressed as
    data rather than as a `return 1`: one cpu bar, eight cuda cells."""
    rows = [_bar("cpu")] + [_cell("cuda", seed=s) for s in range(8)]
    with pytest.raises(ValueError, match="devices"):
        refuse_cross_device_pool(rows)


def test_the_analytic_ceiling_row_is_the_documented_false_positive():
    """PIN THE TRAP, do not pretend it is absent.

    `main()` writes a `t="ceiling"` record with no `device` field, on purpose:
    `sqrt((t*-1)/t*)` is a closed form and stamping a device on it would claim
    it had been measured on one. Read as `"cpu"` by the missing-field rule, it
    refuses against the cuda cells of its own journal -- so a caller must
    filter by `t` first. This test exists so that behaviour is a decision on
    the record rather than a surprise, and so that a later change which starts
    device-stamping the ceiling has to come here and say why."""
    ceiling = dict(t="ceiling", t_star=2, hop_budget=1,
                   nrmse_ceiling=0.7071067811865476)          # no "device"
    with pytest.raises(ValueError, match="devices"):
        refuse_cross_device_pool([ceiling] + [_cell("cuda", seed=s) for s in range(8)])

    readings = [r for r in ([ceiling] + [_cell("cuda", seed=s) for s in range(8)])
                if r["t"] in ("cell", "bar", "instrument_broken")]
    assert refuse_cross_device_pool(readings) == "cuda"
