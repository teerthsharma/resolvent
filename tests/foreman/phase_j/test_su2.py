import numpy as np
import torch
from su2 import qmul, qconj, qnormalize, prefix_scan, sequential_prefix


def random_unit_quats(n, seed, dtype=torch.float64):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 4))
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return torch.tensor(v, dtype=dtype)


def test_row_a_scan_vs_sequential_f64():
    q = random_unit_quats(4096, 0, torch.float64)
    a = prefix_scan(q)
    b = sequential_prefix(q)
    err = (a - b).abs().max().item()
    assert err <= 1e-12, err


def test_row_b_gij_direct_vs_pi():
    q = random_unit_quats(4096, 0, torch.float64)
    Pi = prefix_scan(q)  # Pi_i = q_i...q_1
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 4096, size=(200, 2))
    worst = 0.0
    for a_, b_ in idx:
        i, j = int(max(a_, b_)), int(min(a_, b_))
        if i == j:
            i += 1
        # direct path product q_i ... q_{j+1}
        direct = q[j + 1]
        for k in range(j + 2, i + 1):
            direct = qmul(q[k], direct)
        gij = qmul(Pi[i], qconj(Pi[j]))
        err = (direct - gij).abs().max().item()
        worst = max(worst, err)
    assert worst <= 1e-12, worst


def test_row_c_f32_scan_renorm_vs_f64():
    q64 = random_unit_quats(4096, 0, torch.float64)
    q32 = q64.to(torch.float32)

    ref = prefix_scan(q64)

    def scan_with_renorm(qq, every):
        x = qq.clone()
        n = x.shape[-2]
        d = 1
        while d < n:
            shifted = x[..., :-d, :]
            head = x[..., :d, :]
            combined = qmul(x[..., d:, :], shifted)
            x = torch.cat([head, combined], dim=-2)
            d *= 2
            if every and d % every == 0:
                x = qnormalize(x)
        return qnormalize(x)

    out_renorm = scan_with_renorm(q32, 256)
    err_renorm = (out_renorm.to(torch.float64) - ref).abs().max().item()
    assert err_renorm <= 1e-4, err_renorm

    out_norenorm = prefix_scan(q32)
    err_norenorm = (out_norenorm.to(torch.float64) - ref).abs().max().item()

    q16 = q64.to(torch.bfloat16)
    out_bf16_norenorm = prefix_scan(q16)
    err_bf16_norenorm = (out_bf16_norenorm.to(torch.float64) - ref).abs().max().item()
    out_bf16_renorm = scan_with_renorm(q16, 256)
    err_bf16_renorm = (out_bf16_renorm.to(torch.float64) - ref).abs().max().item()

    print("INFO f32 no-renorm err:", err_norenorm)
    print("INFO bf16 no-renorm err:", err_bf16_norenorm)
    print("INFO bf16 renorm err:", err_bf16_renorm)


if __name__ == "__main__":
    test_row_a_scan_vs_sequential_f64()
    test_row_b_gij_direct_vs_pi()
    test_row_c_f32_scan_renorm_vs_f64()
    print("ALL PASS")
