"""C5 and C6, measured. Utilisation, wall clock, and memory against a tuned baseline.

C5 -- >=30% accelerator utilisation. An elegant module at 5% is dead.
C6 -- honest cost. Wall clock and memory against a TUNED attention baseline, not
      quality deltas alone.

Method notes, so the numbers mean something:

  * Peak is MEASURED on this device with a large cuBLAS fp16 GEMM, not read off a
    spec sheet. sigmoid's probe on this same RTX 4060 Laptop reported
    "22.1 TFLOP/s fp16 measured"; this file re-measures rather than citing it.
  * The utilisation numerator is the FLOPs the kernel ACTUALLY executes --
    4 * units * BLOCK_M * BLOCK_N * head_dim per batch-head -- not the FLOPs a
    dense kernel would have executed. Counting dense-equivalent FLOPs for a
    kernel that skips work is how a sparse kernel reports 300% utilisation.
  * The baseline is `F.scaled_dot_product_attention(is_causal=True)`, which
    dispatches to torch's tuned fused backends. It is a DIFFERENT algorithm
    (exact, quadratic), so the comparison is cost-only and is reported as such.
"""

import math
import time

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_triton

from ceq.multizoom import plan_for

pytestmark = requires_triton

HEAD_DIM = 64


def _sync():
    torch.cuda.synchronize()


def _time(fn, warmup=5, rounds=20):
    """Best-of-N with CUDA events.

    Best rather than median, and CUDA events rather than perf_counter, because
    this is a LAPTOP GPU: a 300-GEMM sustained loop measured 6.70 TFLOP/s against
    17.73 in burst on the same silicon. Median mixes the kernel's speed with the
    thermal state of the room. Best-of-N is the only figure that is a property of
    the kernel, and the same estimator is used for the peak, so the ratio is
    apples to apples.
    """
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        _sync()
        ts.append(a.elapsed_time(b) / 1e3)
    return min(ts)


_PEAKS = {}


def _measured_peaks():
    """(burst, large, sustained) fp16 TFLOP/s, measured, cached for the session.

    THE DENOMINATOR IS THE WHOLE ARGUMENT for a utilisation number, so all three
    are reported and C5 is judged against the LARGEST -- the most conservative
    choice, the one hardest for the kernel to clear.
    """
    if _PEAKS:
        return _PEAKS
    out = {}
    for tag, n in (("burst", 2048), ("large", 4096)):
        a = torch.randn(n, n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, n, device="cuda", dtype=torch.float16)
        out[tag] = 2 * n ** 3 / _time(lambda: torch.mm(a, b), 20, 50) / 1e12
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    _sync()
    t0 = time.perf_counter()
    for _ in range(200):
        torch.mm(a, b)
    _sync()
    out["sustained"] = 2 * n ** 3 / ((time.perf_counter() - t0) / 200) / 1e12
    _PEAKS.update(out)
    return _PEAKS


def _peak_fp16_tflops():
    return max(_measured_peaks().values())


def _peak_bandwidth_gbs():
    n = 1 << 26
    x = torch.empty(n, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    t = _time(lambda: y.copy_(x), warmup=10, rounds=30)
    return 2 * x.numel() * 2 / t / 1e9


def test_report_measured_device_peaks():
    pk = _measured_peaks()
    bw = _peak_bandwidth_gbs()
    p = torch.cuda.get_device_properties(0)
    print(f"\n  device {p.name}  sm_{p.major}{p.minor}  {p.multi_processor_count} SMs  "
          f"{p.total_memory / 1e9:.2f} GB")
    for tag in ("burst", "large", "sustained"):
        print(f"  fp16 GEMM peak, {tag:<9} {pk[tag]:6.2f} TFLOP/s")
    print(f"  C5 denominator (largest = worst case for the kernel) "
          f"{max(pk.values()):.2f} TFLOP/s")
    print(f"  measured copy bandwidth  {bw:.1f} GB/s")
    assert max(pk.values()) > 1.0, "peak measurement failed"


@pytest.mark.parametrize("block", [32, 64, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_utilisation_by_block_and_warps(block, num_warps):
    """C5 sweep. Reported for every configuration; the assertion is only that the
    measurement is well-formed. `test_c5_best_configuration_reaches_30_percent`
    is the one that passes or fails C5."""
    from ceq.mz_kernel import attention_flops, multizoom_attention

    seq, bh = 16384, 8
    q, k, v = [torch.randn(1, bh, seq, HEAD_DIM, device="cuda", dtype=torch.float16)
               for _ in range(3)]
    plan = plan_for(seq, block=block, window_blocks=2, sink_blocks=1)
    fn = lambda: multizoom_attention(q, k, v, plan, num_warps=num_warps)  # noqa: E731
    t = _time(fn)
    fl = attention_flops(plan, HEAD_DIM, bh)
    peak = _peak_fp16_tflops()
    ach = fl / t / 1e12
    print(f"\n  block={block:<4} warps={num_warps}  {t * 1e3:7.3f} ms  "
          f"{ach:6.2f} TFLOP/s  util {ach / peak:6.1%}")
    assert t > 0 and fl > 0


def test_c5_best_configuration_reaches_30_percent_utilisation():
    """C5: >=30% accelerator utilisation, or the kernel is dead."""
    from ceq.mz_kernel import attention_flops, multizoom_attention

    seq, bh = 16384, 8
    q, k, v = [torch.randn(1, bh, seq, HEAD_DIM, device="cuda", dtype=torch.float16)
               for _ in range(3)]
    peak = _peak_fp16_tflops()
    best = (0.0, None)
    for block in (32, 64, 128):
        for warps in (2, 4, 8):
            for stages in (1, 2, 3):
                plan = plan_for(seq, block=block, window_blocks=2, sink_blocks=1)
                try:
                    t = _time(lambda: multizoom_attention(
                        q, k, v, plan, num_warps=warps, num_stages=stages),
                        warmup=3, rounds=10)
                except Exception as exc:  # noqa: BLE001
                    print(f"  block={block} warps={warps} stages={stages}: {exc}")
                    continue
                ach = attention_flops(plan, HEAD_DIM, bh) / t / 1e12
                if ach > best[0]:
                    best = (ach, (block, warps, stages, t))
    ach, cfg = best
    pk = _measured_peaks()
    print("\n  utilisation against each measured denominator:")
    for tag in ("burst", "large", "sustained"):
        print(f"    vs {tag:<9} peak {pk[tag]:6.2f} TFLOP/s -> {ach / pk[tag]:6.1%}")
    print(f"  C5 judged against the largest: {peak:.2f} TFLOP/s")
    print(f"  best config block={cfg[0]} warps={cfg[1]} stages={cfg[2]} "
          f"-> {cfg[3] * 1e3:.3f} ms, {ach:.2f} TFLOP/s, "
          f"utilisation {ach / peak:.1%}")
    assert ach / peak >= 0.30, (
        f"C5 FAILED: best achieved utilisation is {ach / peak:.1%} of the measured "
        f"{peak:.2f} TFLOP/s peak, below the 30% floor. An elegant kernel at this "
        f"rate is dead."
    )


def test_c6_wall_clock_and_memory_against_tuned_sdpa():
    """C6: wall clock and peak memory against `F.scaled_dot_product_attention`.

    Reports the whole curve rather than one flattering point. sigmoid measured a
    crossover near 8k-16k positions for selection-based sparsity on this same
    GPU; this is the same question asked of a summarising scheme.
    """
    from ceq.mz_kernel import multizoom_attention

    print("\n     seq   sdpa ms   mz ms   speedup   sdpa MB   mz MB   mem ratio")
    rows = []
    for shift in (10, 11, 12, 13, 14, 15):
        seq = 1 << shift
        bh = 8
        q, k, v = [torch.randn(1, bh, seq, HEAD_DIM, device="cuda",
                               dtype=torch.float16) for _ in range(3)]
        plan = plan_for(seq, block=64, window_blocks=2, sink_blocks=1)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t_base = _time(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                       warmup=3, rounds=10)
        m_base = torch.cuda.max_memory_allocated() / 1e6

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t_mz = _time(lambda: multizoom_attention(q, k, v, plan, num_warps=4),
                     warmup=3, rounds=10)
        m_mz = torch.cuda.max_memory_allocated() / 1e6

        rows.append((seq, t_base, t_mz, m_base, m_mz))
        print(f"  {seq:>6}  {t_base * 1e3:8.3f}  {t_mz * 1e3:6.3f}  "
              f"{t_base / t_mz:7.2f}x  {m_base:8.1f}  {m_mz:6.1f}  "
              f"{m_mz / m_base:9.2f}x")

    crossover = next((s for s, tb, tm, _, _ in rows if tb / tm > 1.0), None)
    print(f"  wall-clock crossover at seq = {crossover}")
    assert rows[-1][1] / rows[-1][2] > 1.0, (
        f"C6 FAILED: at the longest context measured the module is still slower "
        f"than tuned SDPA ({rows[-1][1] * 1e3:.3f} ms vs {rows[-1][2] * 1e3:.3f} ms). "
        f"A near-linear scheme that never overtakes quadratic attention on real "
        f"hardware has no cost story."
    )


def test_pyramid_build_is_a_small_share_of_end_to_end_cost():
    """The builder cost, on the GPU, inside the real path.

    The dyadic schedule died at 53x. This measures the same ratio for a mean pool,
    end to end, including the concatenate the kernel needs.
    """
    from ceq.multizoom import build_pyramid
    from ceq.mz_kernel import multizoom_attention

    seq, bh = 16384, 8
    q, k, v = [torch.randn(1, bh, seq, HEAD_DIM, device="cuda", dtype=torch.float16)
               for _ in range(3)]
    plan = plan_for(seq, block=64, window_blocks=2, sink_blocks=1)

    t_build = _time(lambda: (torch.cat(build_pyramid(k, plan.levels), dim=-2),
                             torch.cat(build_pyramid(v, plan.levels), dim=-2)))
    t_total = _time(lambda: multizoom_attention(q, k, v, plan))
    print(f"\n  pyramid build {t_build * 1e3:.3f} ms of {t_total * 1e3:.3f} ms total "
          f"= {t_build / t_total:.1%} of the fused call")
    assert t_build < t_total, (
        f"the pooling builder ({t_build * 1e3:.3f} ms) costs more than the whole "
        f"fused attention call ({t_total * 1e3:.3f} ms)"
    )
