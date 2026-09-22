"""COST-B' row: time Mode B' vs plain SDPA, S=4096, float32, CUDA.
Runs as a fresh subprocess (own CUDA context)."""
import time
import json
import torch
import torch.nn.functional as F

from costb_impl import qmul, qconj, hillis_steele_prefix_quat, rotate_blocks

DEV = "cuda"
DTYPE = torch.float32
B, H, S, HD = 1, 8, 4096, 64
D_MODEL = H * HD  # 512
WARMUP, STEPS = 5, 20


def sync():
    torch.cuda.synchronize()


def build_inputs():
    torch.manual_seed(0)
    x = torch.randn(B, S, D_MODEL, device=DEV, dtype=DTYPE)
    w_head = torch.randn(4, D_MODEL, device=DEV, dtype=DTYPE) * 0.1
    q = torch.randn(B, H, S, HD, device=DEV, dtype=DTYPE, requires_grad=True)
    k = torch.randn(B, H, S, HD, device=DEV, dtype=DTYPE, requires_grad=True)
    v = torch.randn(B, H, S, HD, device=DEV, dtype=DTYPE, requires_grad=True)
    x = x.requires_grad_(True)
    w_head = w_head.requires_grad_(True)
    return x, w_head, q, k, v


def baseline_sdpa(q, k, v):
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.MATH,
         torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
         torch.nn.attention.SDPBackend.FLASH_ATTENTION]
    ):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def observed_backend(q, k, v):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for name, backend in [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]:
        try:
            with sdpa_kernel([backend]):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return name
        except RuntimeError:
            continue
    return "NONE_RAN"


def mode_bprime_stages(x, w_head, q, k, v):
    """Returns (out, stage_times_ms dict) for ONE forward pass, with
    torch.cuda.synchronize() bracketing each stage."""
    times = {}

    sync(); t0 = time.perf_counter()
    quat = x @ w_head.T
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (B,S,4)
    sync(); times["head"] = (time.perf_counter() - t0) * 1000

    sync(); t0 = time.perf_counter()
    quat_h = quat.unsqueeze(1).expand(B, H, S, 4)  # head-shared
    Pi = hillis_steele_prefix_quat(quat_h)  # (B,H,S,4)
    sync(); times["scan"] = (time.perf_counter() - t0) * 1000

    sync(); t0 = time.perf_counter()
    v_rot = rotate_blocks(v, qconj(Pi))
    sync(); times["pre_rotate"] = (time.perf_counter() - t0) * 1000

    sync(); t0 = time.perf_counter()
    out = F.scaled_dot_product_attention(q, k, v_rot, is_causal=True)
    sync(); times["sdpa"] = (time.perf_counter() - t0) * 1000

    sync(); t0 = time.perf_counter()
    out_rot = rotate_blocks(out, Pi)
    sync(); times["post_rotate"] = (time.perf_counter() - t0) * 1000

    return out_rot, times


def time_forward_only(fn, *args, steps=STEPS, warmup=WARMUP):
    for _ in range(warmup):
        out = fn(*args)
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = fn(*args)
    sync()
    return (time.perf_counter() - t0) * 1000 / steps, out


def time_fwd_bwd(fn, args, loss_fn, steps=STEPS, warmup=WARMUP):
    for _ in range(warmup):
        out = fn(*args)
        loss = loss_fn(out)
        loss.backward()
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = fn(*args)
        loss = loss_fn(out)
        loss.backward()
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    sync()
    return (time.perf_counter() - t0) * 1000 / steps


def main():
    x, w_head, q, k, v = build_inputs()

    backend = observed_backend(q, k, v)

    def bprime_forward(x, w_head, q, k, v):
        out, _ = mode_bprime_stages(x, w_head, q, k, v)
        return out

    def sdpa_forward(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # forward-only timing
    t_sdpa_fwd, _ = time_forward_only(sdpa_forward, q, k, v)
    t_bprime_fwd, _ = time_forward_only(bprime_forward, x, w_head, q, k, v)

    # staged breakdown, averaged over STEPS (forward only, post-warmup)
    for _ in range(WARMUP):
        mode_bprime_stages(x, w_head, q, k, v)
    stage_sums = {}
    for _ in range(STEPS):
        _, st = mode_bprime_stages(x, w_head, q, k, v)
        for kk, vv in st.items():
            stage_sums[kk] = stage_sums.get(kk, 0.0) + vv
    stage_avg = {kk: vv / STEPS for kk, vv in stage_sums.items()}

    # forward+backward timing
    loss_fn = lambda o: o.float().sum()
    t_sdpa_fb = time_fwd_bwd(sdpa_forward, (q, k, v), loss_fn)
    t_bprime_fb = time_fwd_bwd(bprime_forward, (x, w_head, q, k, v), loss_fn)

    result = {
        "row": "COST-B'",
        "shape": {"B": B, "H": H, "S": S, "head_dim": HD, "dtype": "float32"},
        "backend_observed": backend,
        "flash_available_in_build": torch.backends.cuda.flash_sdp_enabled(),
        "ms_sdpa_fwd": t_sdpa_fwd,
        "ms_bprime_fwd": t_bprime_fwd,
        "ratio_fwd": t_bprime_fwd / t_sdpa_fwd,
        "ms_sdpa_fwd_bwd": t_sdpa_fb,
        "ms_bprime_fwd_bwd": t_bprime_fb,
        "ratio_fwd_bwd": t_bprime_fb / t_sdpa_fb,
        "stage_ms_fwd": stage_avg,
        "bar": 1.3,
        "strike_above": 2.0,
    }
    ratio = result["ratio_fwd"]
    if ratio <= 1.3:
        verdict = "PASS"
    elif ratio > 2.0:
        verdict = "FAIL"
    else:
        verdict = "NEITHER"
    result["verdict"] = verdict

    print(json.dumps(result, indent=2))
    with open("costb_timing_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
