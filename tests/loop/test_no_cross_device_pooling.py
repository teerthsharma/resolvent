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
