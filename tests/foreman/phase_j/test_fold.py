import torch
from fold import make_inputs, direct_out, fold_out

torch.manual_seed(0)


def test_fold_matches_direct_float64():
    q, k, v, Pi = make_inputs(2, 2, 256, 16, seed=0, dtype=torch.float64)
    d = direct_out(q, k, v, Pi)
    f = fold_out(q, k, v, Pi)
    err = (d - f).abs().max().item()
    assert err <= 1e-12, f"max abs err {err} > 1e-12"


def test_fold_matches_direct_float32():
    q, k, v, Pi = make_inputs(2, 2, 256, 16, seed=0, dtype=torch.float32)
    d = direct_out(q, k, v, Pi)
    f = fold_out(q, k, v, Pi)
    err = (d - f).abs().max().item()
    assert err <= 1e-5, f"max abs err {err} > 1e-5"


def test_fold_gradcheck_float64_small():
    q, k, v, Pi = make_inputs(1, 1, 16, 16, seed=1, dtype=torch.float64)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    def f(q, k, v):
        return fold_out(q, k, v, Pi)

    assert torch.autograd.gradcheck(f, (q, k, v), eps=1e-6, atol=1e-5)


if __name__ == "__main__":
    test_fold_matches_direct_float64()
    print("float64 fold-vs-direct: PASS")
    test_fold_matches_direct_float32()
    print("float32 fold-vs-direct: PASS")
    test_fold_gradcheck_float64_small()
    print("gradcheck: PASS")
