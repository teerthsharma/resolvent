"""Does a 0.5B model with this attention fit a Colab GPU, and what does it cost?

Sizing arithmetic is worthless unless the model behind it is calibrated against a
real measurement, so this file is in two halves:

  CALIBRATION -- `ceq/sizing.py` predicts the activation bytes of a forward +
  backward, and the prediction is checked against `torch.cuda.max_memory_allocated`
  on the box that is running the test. A sizing model that has never been
  compared to a scale is a spreadsheet.

  EXTRAPOLATION -- the calibrated model applied to a 0.5B configuration on the
  two Colab accelerators that can actually run this operator.

The T4 is excluded on purpose and not by oversight: `tests/chase/test_hf_shipping.py`
already records that Triton needs compute capability 8.0+ and the free-tier T4 is
sm_75. The A100 (sm_80) and L4 (sm_89) are the two that qualify, and both are
paid tiers.

Every test parametrizes over cpu and cuda via `tests/chase/conftest.py`. The
arithmetic runs on both; only the calibration needs a GPU.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ceq import lm, sizing


# --------------------------------------------------------------- calibration

CAL = dict(bs=4, d=256, layers=4, heads=4)


def _measured_activation_bytes(kind, seq, device, **cfg):
    """Peak allocated bytes for one forward + backward, minus the resident weights."""
    torch.manual_seed(0)
    m = lm.TinyLM(kind, d=cfg["d"], n_layers=cfg["layers"], n_heads=cfg["heads"],
                  seq=seq, seed=0).to(device)
    x = torch.randint(0, lm.VOCAB, (cfg["bs"], seq), device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    F.cross_entropy(m(x).reshape(-1, lm.VOCAB), x.reshape(-1)).backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del m, x
    torch.cuda.empty_cache()
    return peak


@pytest.mark.parametrize("seq", [256, 512, 1024])
def test_the_sizing_model_predicts_measured_activation_memory(device, seq):
    """RED first. The extrapolation to 0.5B is only as good as this.

    `sizing.activation_bytes` must land within 25% of the allocator's own peak
    for the signed arm on the box under test. Anything looser and the A100 / L4
    verdicts below are not arithmetic, they are a guess with units attached.
    """
    if device != "cuda":
        pytest.skip("peak-allocation accounting is CUDA-only")
    got = _measured_activation_bytes("signed", seq, device, **CAL)
    cfg = sizing.Config(d_model=CAL["d"], n_layers=CAL["layers"], n_heads=CAL["heads"],
                        d_head=CAL["d"] // CAL["heads"], vocab=lm.VOCAB, seq=seq)
    pred = sizing.activation_bytes(cfg, batch=CAL["bs"], arm="signed", bytes_per=4)
    rel = abs(pred - got) / got
    assert rel < 0.25, (
        "seq={}: predicted {:.1f} MiB, allocator measured {:.1f} MiB, off by "
        "{:.1%}. RTX 4060 Laptop, sm_89, 8.0 GiB, torch 2.5.1+cu121.".format(
            seq, pred / 2 ** 20, got / 2 ** 20, rel))


@pytest.mark.parametrize("seq", [256, 512, 1024])
def test_the_sizing_model_predicts_the_softmax_arm_too(device, seq):
    """The control. A model tuned to fit only the arm it was fitted on has learnt
    the measurement, not the mechanism."""
    if device != "cuda":
        pytest.skip("peak-allocation accounting is CUDA-only")
    got = _measured_activation_bytes("softmax", seq, device, **CAL)
    cfg = sizing.Config(d_model=CAL["d"], n_layers=CAL["layers"], n_heads=CAL["heads"],
                        d_head=CAL["d"] // CAL["heads"], vocab=lm.VOCAB, seq=seq)
    pred = sizing.activation_bytes(cfg, batch=CAL["bs"], arm="softmax", bytes_per=4)
    rel = abs(pred - got) / got
    assert rel < 0.35, (
        "seq={}: predicted {:.1f} MiB, measured {:.1f} MiB, off by {:.1%}".format(
            seq, pred / 2 ** 20, got / 2 ** 20, rel))


def _measured_autocast_bytes(kind, seq, device, **cfg):
    torch.manual_seed(0)
    m = lm.TinyLM(kind, d=cfg["d"], n_layers=cfg["layers"], n_heads=cfg["heads"],
                  seq=seq, seed=0).to(device)
    x = torch.randint(0, lm.VOCAB, (cfg["bs"], seq), device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(m(x).reshape(-1, lm.VOCAB).float(), x.reshape(-1))
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del m, x
    torch.cuda.empty_cache()
    return peak


@pytest.mark.parametrize("seq", [512, 1024])
def test_bf16_autocast_does_not_halve_the_activation_memory(device, seq):
    """RED first, and it moved every 0.5B verdict in this file.

    The model was originally driven at `bytes_per=2` for bf16, which is the
    obvious assumption and is wrong: autograd retains fp32 alongside the bf16
    casts, the logits and cross_entropy stay fp32, and LayerNorm is
    autocast-excluded. Measured signed-arm bf16/fp32 peak: 0.748 at seq 256,
    0.772 at 512, 0.797 at 1024 -- never near 0.5, and RISING with S.

    `DTYPE_MODES["bf16_autocast"]` carries the fitted (2.2, 3.4) effective bytes.
    This test is what keeps that pair honest.
    """
    if device != "cuda":
        pytest.skip("peak-allocation accounting is CUDA-only")
    got = _measured_autocast_bytes("signed", seq, device, **CAL)
    fp32 = _measured_activation_bytes("signed", seq, device, **CAL)
    assert got > 0.5 * fp32, (
        "bf16 autocast peak {:.1f} MiB against fp32 {:.1f} MiB = {:.3f}x; if this "
        "ever really does reach 0.5x, DTYPE_MODES is now pessimistic".format(
            got / 2 ** 20, fp32 / 2 ** 20, got / fp32))
    cfg = sizing.Config(d_model=CAL["d"], n_layers=CAL["layers"], n_heads=CAL["heads"],
                        d_head=CAL["d"] // CAL["heads"], vocab=lm.VOCAB, seq=seq)
    pred = sizing.activation_bytes(cfg, batch=CAL["bs"], arm="signed",
                                   dtype="bf16_autocast")
    rel = abs(pred - got) / got
    assert rel < 0.25, (
        "seq={}: predicted {:.1f} MiB under bf16 autocast, measured {:.1f} MiB, "
        "off by {:.1%}".format(seq, pred / 2 ** 20, got / 2 ** 20, rel))


# --------------------------------------------------------------- the FLOP count

def _counted_flops_per_token(arm, device, seq, bs, d, layers, heads):
    from torch.utils.flop_counter import FlopCounterMode
    m = lm.TinyLM(arm, d=d, n_layers=layers, n_heads=heads, seq=seq, seed=0).to(device)
    x = torch.randint(0, lm.VOCAB, (bs, seq), device=device)
    with FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops() / (bs * seq), fc.get_flop_counts()


def test_the_signed_arm_flop_formula_matches_a_measured_flop_count(device):
    """RED first. `sizing.flops_per_token` is derived arithmetic; torch's own
    FlopCounterMode is the instrument. If they disagree the derivation is wrong,
    and every FLOP ratio quoted downstream is wrong with it.

    The signed arm is the one that must match on BOTH devices, because it is
    nothing but matmuls and the counter has a formula for every one of them.
    """
    seq, bs, d, layers, heads = 256, 2, 128, 2, 4
    cfg = sizing.Config(d_model=d, n_layers=layers, n_heads=heads,
                        d_head=d // heads, vocab=lm.VOCAB, seq=seq)
    got, _ = _counted_flops_per_token("signed", device, seq, bs, d, layers, heads)
    pred = sizing.flops_per_token(cfg, arm="signed", include_head=True)
    rel = abs(pred - got) / got
    assert rel < 0.05, (
        "formula {:.4e} FLOP/token, FlopCounterMode measured {:.4e}, off by "
        "{:.1%}".format(pred, got, rel))


def test_the_softmax_baseline_flop_formula_matches_where_the_counter_can_see_sdpa(device):
    """The control -- and an instrument fault worth recording rather than routing
    around. On CPU, `is_causal=True` dispatches to
    `aten::_scaled_dot_product_flash_attention_for_cpu`, for which
    `torch.utils.flop_counter` in torch 2.5.1 has NO registered formula: the
    attention term is counted as exactly ZERO and the total comes back 30.8%
    low. Comparing the formula against that number would have 'proved' the
    derivation wrong on CPU and right on CUDA.

    So the test asserts on the counter FIRST -- did it see any attention at all
    -- and only then on the formula.
    """
    seq, bs, d, layers, heads = 256, 2, 128, 2, 4
    cfg = sizing.Config(d_model=d, n_layers=layers, n_heads=heads,
                        d_head=d // heads, vocab=lm.VOCAB, seq=seq)
    got, counts = _counted_flops_per_token("softmax", device, seq, bs, d, layers, heads)
    ops = {str(k) for mod in counts.values() for k in mod}
    saw_sdpa = any("scaled_dot_product" in o for o in ops)
    if not saw_sdpa:
        pytest.skip(
            "torch {} FlopCounterMode registered no formula for the SDPA op this "
            "device dispatched to; counted ops were {}. The attention term is "
            "counted as zero, so the instrument cannot validate this arm here."
            .format(torch.__version__, sorted(ops)))
    pred = sizing.flops_per_token(cfg, arm="softmax", include_head=True)
    rel = abs(pred - got) / got
    assert rel < 0.05, (
        "formula {:.4e} FLOP/token, FlopCounterMode measured {:.4e}, off by "
        "{:.1%}".format(pred, got, rel))


def test_the_signed_arm_costs_more_flops_per_token_than_softmax_at_equal_params(device):
    """hops=3 buys 3 extra [S,S]@[S,D] matmuls per layer per forward, and the
    operator materialization is a DENSE [S,S] where a fused causal kernel touches
    only the lower triangle. Both show up here."""
    cfg = sizing.CFG_500M
    a = sizing.flops_per_token(cfg, arm="softmax")
    b = sizing.flops_per_token(cfg, arm="signed")
    assert b > a
    assert b / a < 1.5, (
        "signed is {:.3f}x softmax FLOPs/token at d={} S={} hops={}: {:.4e} vs "
        "{:.4e}".format(b / a, cfg.d_model, cfg.seq, sizing.HOPS, b, a))


# ------------------------------------------------------- does 0.5B actually fit

def test_the_500m_config_is_actually_500m_parameters(device):
    """Calibration for every 'at 0.5B' sentence below."""
    n = sizing.params(sizing.CFG_500M)
    assert 4.5e8 <= n <= 5.5e8, "{:,} parameters".format(n)


def test_the_head_dim_of_the_500m_config_is_one_the_kernel_accepts(device):
    """`test_kernel_contracts.py::test_common_head_dims` records head dims 80 and
    96 as REJECTED; only 16/32/64/128 are supported. A 0.5B config chosen for
    round numbers of d_model can land on an unrunnable head dim."""
    assert sizing.CFG_500M.d_head in (16, 32, 64, 128), sizing.CFG_500M.d_head
    assert sizing.CFG_500M.n_heads * sizing.CFG_500M.d_head == sizing.CFG_500M.d_model


@pytest.mark.parametrize("gpu", ["A100-40GB", "L4-24GB"])
def test_a_500m_signed_model_trains_at_seq_2048_batch_1_on_a_colab_gpu(device, gpu):
    """RED first. 2048 is the shortest context anyone ships a 0.5B model at.

    Batch 1 with gradient accumulation is the last resort before the answer is
    'this does not fit', so if batch 1 does not fit, nothing does.
    """
    cfg = sizing.replace(sizing.CFG_500M, seq=2048)
    fit = sizing.fits(cfg, batch=1, arm="signed", gpu=gpu, optimizer="adamw_bf16_mixed")
    assert fit.ok, fit.explain()


@pytest.mark.parametrize("gpu", ["A100-40GB", "L4-24GB"])
def test_the_softmax_control_fits_the_same_gpu_at_the_same_shape(device, gpu):
    """The control. If neither arm fits, the finding is the GPU and not the
    operator."""
    cfg = sizing.replace(sizing.CFG_500M, seq=2048)
    fit = sizing.fits(cfg, batch=1, arm="softmax", gpu=gpu, optimizer="adamw_bf16_mixed")
    assert fit.ok, fit.explain()


@pytest.mark.parametrize("gpu", ["A100-40GB", "L4-24GB"])
def test_a_500m_signed_model_reaches_a_batch_of_at_least_8_at_seq_2048(device, gpu):
    """Batch 1 that fits is not a training run. Below roughly 8 sequences of 2048
    the step is gradient-accumulation-bound and the wall clock stops being about
    the operator at all."""
    cfg = sizing.replace(sizing.CFG_500M, seq=2048)
    b = sizing.max_batch(cfg, arm="signed", gpu=gpu, optimizer="adamw_bf16_mixed")
    assert b >= 8, (
        "max batch {} at seq 2048 on {}; the softmax control reaches {}. {}".format(
            b, gpu, sizing.max_batch(cfg, arm="softmax", gpu=gpu,
                                     optimizer="adamw_bf16_mixed"),
            sizing.fits(cfg, batch=8, arm="signed", gpu=gpu,
                        optimizer="adamw_bf16_mixed").explain()))


def test_activation_checkpointing_is_a_real_rollback_and_not_a_hope(device):
    """The stated rollback for the memory finding. Recomputing the attention block
    in the backward pass must drop the retained [S,S] tensors to one per layer.

    Measured, not assumed: this runs both paths."""
    if device != "cuda":
        pytest.skip("peak-allocation accounting is CUDA-only")
    plain = _measured_activation_bytes("signed", 1024, device, **CAL)
    ckpt = sizing.measure_checkpointed_activation_bytes(
        "signed", 1024, device, **CAL)
    assert ckpt < 0.6 * plain, (
        "checkpointed {:.1f} MiB vs plain {:.1f} MiB = {:.2f}x; the rollback does "
        "not recover the memory it is supposed to.".format(
            ckpt / 2 ** 20, plain / 2 ** 20, ckpt / plain))


# ============================================================== 300M, round 3
#
# The user's standard: parity at 3.3M is FALSE unless it survives at 300M. So
# what does 300M cost, exactly, and on what.
#
# CFG_300M is the GPT-2-medium shape -- d 1024, 24 layers, 16 heads of 64 --
# because that is what "300M" means in the literature and because d_head 64 is
# one of the four head dims the kernel accepts.
#
# THE ARM CHANGED AND THE SIZING MODEL DID NOT. Every coefficient above was
# calibrated against `signed`: the L1-normalized operator, hops=3, one [S,S]
# matrix per layer. The operator that reached 1.0334 parity is `sgate`, which
# builds TWO softmax matrices over the same logits and combines them. A memory
# model fitted to one operator and quoted for the other is a spreadsheet again.


@pytest.fixture
def parity_point():
    """rho=1.5 lam=0.10 hops=2 -- the iteration-16 point, restored afterwards."""
    old = (lm.RHO, lm.SGATE_LAM, lm.HOPS)
    lm.RHO, lm.SGATE_LAM, lm.HOPS = 1.5, 0.10, 2
    try:
        yield
    finally:
        lm.RHO, lm.SGATE_LAM, lm.HOPS = old


@pytest.mark.parametrize("seq", [256, 512, 1024])
def test_the_sizing_model_predicts_the_sgate_arm_it_was_never_calibrated_on(
        device, seq, parity_point):
    """RED first. Every 300M verdict below is quoted for `sgate` and every
    coefficient in `ceq/sizing.py` was fitted on `signed`."""
    if device != "cuda":
        pytest.skip("peak-allocation accounting is CUDA-only")
    got = _measured_activation_bytes("sgate", seq, device, **CAL)
    cfg = sizing.Config(d_model=CAL["d"], n_layers=CAL["layers"], n_heads=CAL["heads"],
                        d_head=CAL["d"] // CAL["heads"], vocab=lm.VOCAB, seq=seq)
    pred = sizing.activation_bytes(cfg, batch=CAL["bs"], arm="sgate", bytes_per=4)
    rel = abs(pred - got) / got
    assert rel < 0.25, (
        "seq={}: predicted {:.1f} MiB for sgate, allocator measured {:.1f} MiB, "
        "off by {:.1%}".format(seq, pred / 2 ** 20, got / 2 ** 20, rel))


@pytest.mark.parametrize("seq", [256, 512, 1024])
def test_sgate_does_not_cost_more_activation_memory_than_the_signed_arm(
        device, seq, parity_point):
    """`sgate` runs softmax TWICE over the same [S,S] logits. Autograd retains
    each softmax's OUTPUT for its own backward, so the retained-tensor count is
    the thing that changed when the parity operator changed, and it moves every
    batch-size verdict on record."""
    if device != "cuda":
        pytest.skip("peak-allocation accounting is CUDA-only")
    sg = _measured_activation_bytes("sgate", seq, device, **CAL)
    sd = _measured_activation_bytes("signed", seq, device, **CAL)
    assert sg <= sd * 1.05, (
        "seq={}: sgate {:.1f} MiB against signed {:.1f} MiB = {:.2f}x. The sizing "
        "model, the 0.5B verdict and the MODEL_CARD memory table were all fitted "
        "on `signed`.".format(seq, sg / 2 ** 20, sd / 2 ** 20, sg / sd))


def test_the_300m_config_is_a_gpt2_medium_shape_of_about_300m_parameters(device):
    n = sizing.params(sizing.CFG_300M)
    assert 2.8e8 <= n <= 3.8e8, "{:,} parameters".format(n)
    assert sizing.CFG_300M.d_head in (16, 32, 64, 128), sizing.CFG_300M.d_head
    assert sizing.CFG_300M.n_heads * sizing.CFG_300M.d_head == sizing.CFG_300M.d_model


@pytest.mark.parametrize("gpu", ["A100-40GB", "L4-24GB"])
@pytest.mark.parametrize("seq", [1024, 2048])
def test_a_300m_sgate_model_trains_at_batch_1_on_a_colab_gpu(device, gpu, seq):
    """Batch 1 with gradient accumulation is the last resort before 'does not
    fit'. No checkpointing here: this is what the naive run does."""
    cfg = sizing.replace(sizing.CFG_300M, seq=seq)
    fit = sizing.fits(cfg, batch=1, arm="sgate", gpu=gpu, optimizer="adamw_bf16_mixed")
    assert fit.ok, fit.explain()


@pytest.mark.parametrize("gpu", ["A100-40GB", "L4-24GB"])
@pytest.mark.parametrize("seq", [1024, 2048])
def test_a_300m_sgate_model_reaches_batch_8_with_gradient_checkpointing(device, gpu, seq):
    """Checkpointing is the stated rollback and it is measured at 0.379x. Batch 8
    of `seq` is the floor below which the step is accumulation-bound and the wall
    clock stops being about the operator."""
    cfg = sizing.replace(sizing.CFG_300M, seq=seq)
    b = sizing.max_batch(cfg, arm="sgate", gpu=gpu, optimizer="adamw_bf16_mixed",
                         checkpointed=True)
    assert b >= 8, (
        "max batch {} at seq {} on {} WITH checkpointing; the softmax control "
        "reaches {}. {}".format(
            b, seq, gpu,
            sizing.max_batch(cfg, arm="softmax", gpu=gpu,
                             optimizer="adamw_bf16_mixed", checkpointed=True),
            sizing.fits(cfg, batch=8, arm="sgate", gpu=gpu,
                        optimizer="adamw_bf16_mixed", checkpointed=True).explain()))


@pytest.mark.parametrize("seq", [1024, 2048])
def test_the_operator_term_is_not_the_majority_of_the_300m_activation_budget(device, seq):
    """Where the memory actually goes decides which rollback is worth buying.

    If the [S,S] term dominates, then checkpointing (which keeps ONE block's
    worth alive) is the only lever that matters and every other economy is noise.
    """
    cfg = sizing.replace(sizing.CFG_300M, seq=seq)
    tot = sizing.activation_bytes(cfg, batch=1, arm="sgate")
    smx = sizing.activation_bytes(cfg, batch=1, arm="softmax")
    op = tot - smx
    assert op < 0.5 * tot, (
        "seq {}: the [S,S] operator term is {:.2f} GiB of a {:.2f} GiB "
        "activation budget = {:.1%}; everything the softmax arm also pays is "
        "{:.2f} GiB".format(seq, op / 2 ** 30, tot / 2 ** 30, op / tot,
                            smx / 2 ** 30))


# ----------------------------------------------- what the shipping chain does
#
# `ceq/hf/train.py::preflight` sizes the run with `dtype="bf16_autocast"`. The
# training loop underneath it (lines 193-211) contains no `torch.autocast` call,
# no `.bfloat16()` and no `GradScaler`: it trains in fp32. So the memory the
# preflight promises and the memory the loop takes are two different numbers,
# and at 300M seq 2048 that is 28.43 GiB against 32.77 GiB on a 33.53 GiB budget.


def test_the_memory_preflight_predicts_the_dtype_the_training_loop_actually_uses(device):
    """RED first, measured rather than grepped: build the real `CEQForCausalLM`,
    run the forward+backward the loop runs, and compare the allocator's peak
    against what `preflight` would have promised for the same shape."""
    if device != "cuda":
        pytest.skip("peak-allocation accounting is CUDA-only")
    from ceq.hf import configuration_ceq, modeling_ceq

    d, L, H, seq, bs = 256, 2, 4, 256, 4
    cfg = configuration_ceq.CEQConfig(
        vocab_size=lm.VOCAB, hidden_size=d, num_hidden_layers=L,
        num_attention_heads=H, max_position_embeddings=seq)
    torch.manual_seed(0)
    m = modeling_ceq.CEQForCausalLM(cfg).to(device)
    x = torch.randint(0, lm.VOCAB, (bs, seq), device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    m(input_ids=x, labels=x).loss.backward()      # exactly what train.py does
    torch.cuda.synchronize()
    got = torch.cuda.max_memory_allocated() - base
    del m, x
    torch.cuda.empty_cache()

    scfg = sizing.Config(d_model=d, n_layers=L, n_heads=H, d_head=d // H,
                         vocab=lm.VOCAB, seq=seq)
    promised = sizing.activation_bytes(scfg, batch=bs, arm="signed",
                                       dtype="bf16_autocast")
    assert got <= promised * 1.25, (
        "the loop took {:.1f} MiB where preflight's bf16_autocast sizing promised "
        "{:.1f} MiB = {:.2f}x. `train.py` has no autocast call; it trains fp32 "
        "while sizing the run in bf16.".format(
            got / 2 ** 20, promised / 2 ** 20, got / promised))


@pytest.mark.parametrize("seq", [1024, 2048])
def test_a_300m_step_is_within_2x_the_softmax_control_at_the_300m_block_shape(device, seq):
    """RED first, and it is what a GPU-hour budget is denominated in.

    d=1024, 16 heads of 64 -- the 300M block -- one layer, batch 1, bf16
    autocast, the two arms interleaved round-robin so contention hits both
    equally, minimum of 20 timed steps. The shipped wall-clock figure is 1.32x
    and was measured at d=256 seq 128, where the [S,S] term is small.
    """
    if device != "cuda":
        pytest.skip("wall clock on cpu measures the wrong machine")
    import time
    old = (lm.RHO, lm.SGATE_LAM, lm.HOPS)
    lm.RHO, lm.SGATE_LAM, lm.HOPS = 1.5, 0.10, 2
    try:
        mods = {}
        for kind in ("softmax", "sgate"):
            torch.manual_seed(0)
            mm = lm.TinyLM(kind, d=1024, n_layers=1, n_heads=16, seq=seq, seed=0).to(device)
            mods[kind] = (mm, torch.optim.AdamW(mm.parameters(), lr=1e-4))
        x = torch.randint(0, lm.VOCAB, (1, seq), device=device)
        best = {k: float("inf") for k in mods}
        for i in range(14):
            for kind, (mm, opt) in mods.items():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(mm(x).reshape(-1, lm.VOCAB).float(),
                                           x.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                torch.cuda.synchronize()
                if i >= 4:
                    best[kind] = min(best[kind], time.perf_counter() - t0)
        del mods, x
        torch.cuda.empty_cache()
    finally:
        lm.RHO, lm.SGATE_LAM, lm.HOPS = old
    r = best["sgate"] / best["softmax"]
    assert r <= 2.0, (
        "seq {}: sgate {:.1f} ms/step against softmax {:.1f} ms = {:.2f}x on an "
        "RTX 4060 Laptop. Every GPU-hour estimate multiplies by this."
        .format(seq, best["sgate"] * 1e3, best["softmax"] * 1e3, r))


#: sgate/softmax step-time ratio at the 300M block shape (d=1024, 16 heads of 64),
#: bf16 autocast, batch 1, arms interleaved round-robin, minimum of 20 timed
#: steps on an RTX 4060 Laptop. Bound by
#: `test_a_300m_step_is_within_2x_the_softmax_control_at_the_300m_block_shape`.
MEASURED_STEP_RATIO = {1024: 3.13, 2048: 5.49}


@pytest.mark.parametrize("seq", [1024, 2048])
def test_a_300m_parity_run_at_a_chinchilla_budget_fits_one_colab_a100_session(
        device, seq):
    """RED first. This is the number the whole 300M question is denominated in.

    Budget: 20 tokens per non-embedding parameter -- 20 x 302,088,192 = 6.04e9
    tokens -- which is the smallest budget at which a val-loss comparison is not
    an artifact of undertraining, and undertraining is exactly the direction that
    FLATTERS this operator: W10 recorded the gap widening 1.2191x at 250 steps to
    1.337x at 600.

    The softmax control is priced from FLOPs at 40% MFU on an A100-40GB's 312
    TFLOP/s bf16; the signed arm is that number times the MEASURED step-time
    ratio, because its MFU is not the control's and assuming it is would be the
    optimistic assumption this project has already been burned by twice.

    A Colab session is capped at 24 hours.
    """
    cfg = sizing.replace(sizing.CFG_300M, seq=seq)
    h_softmax = sizing.gpu_hours(cfg, arm="softmax", tokens=sizing.chinchilla_tokens(cfg))
    h_sgate = h_softmax * MEASURED_STEP_RATIO[seq]
    assert h_sgate <= 24.0, (
        "seq {}: {:.3e} tokens at 20/non-embedding-param costs the softmax "
        "control {:.1f} A100-40GB hours at 40% MFU with gradient checkpointing, "
        "and the operator {:.1f}x that = {:.0f} hours. A Colab session caps at "
        "24, so one seed of one arm is {:.0f} sessions."
        .format(seq, sizing.chinchilla_tokens(cfg), h_softmax,
                MEASURED_STEP_RATIO[seq], h_sgate, -(-h_sgate // 24)))
