"""Round 4: make `ceq/hf/` publishable, and make every claim in it survive a
hostile reader.

Four defects were measured before a line of this file's implementation existed,
and each one is a silent wrong answer rather than a crash. That ordering matters:
a public artifact that raises is an inconvenience, a public artifact that returns
a plausible number is a retraction.

  P1  the package does not implement the operator that reached parity.
      `modeling_ceq.ceq_operator` is `rho*w/||w||_1` at rho=0.9 hops=3; the
      1.0334 median came from `sgate` at rho=1.5 lam=0.10 hops=2. Relative
      difference 1.818e+00, already on record in `test_scale_hazards.py`.

  P2  an additive `attention_mask` of -1e4 -- the historical transformers
      convention -- is BITWISE ignored. `ceq_operator` tests `m < finfo.min/2`,
      which is -1.7e38 in float32, so -1e4 is read as "not masked". Measured:
      masked columns keep 0.48290979862213135 of weight and the returned
      operator is `torch.equal` to the unmasked one.

  P3  a chunked prefill leaks the future. `ceq.hybrid.stock_attention` sets
      `is_causal = (q_len == kv_len)`, so at q_len=4 kv_len=16 it passes
      `is_causal=False` and every query reads every key. Perturbing key 15 moved
      the output of query 0 by 1.8396726846694946.

  P4  `tie_word_embeddings=True` strands `lm_head.weight` on the META device and
      the forward still runs, returning 1.0053620544046395e+30 -- uninitialized
      memory -- against the model that was saved. Nothing raises at save, at
      load, or at forward.

  P5  `truncation_bound(rho=1.5, hops=2)` returns **-6.75**. A negative error
      bound, silently, at the operating point the package now ships.

Every test parametrizes over cpu and cuda and skips cuda when absent.
"""

import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

transformers = pytest.importorskip("transformers")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: rho=1.5 lam=0.10 hops=2, the iteration-16 parity point, 1.0334 median over
#: 5 seeds at 3,319,296 parameters exactly.
PARITY_RHO, PARITY_LAM, PARITY_HOPS = 1.5, 0.10, 2


@pytest.fixture
def parity_point():
    """`ceq/lm.py` reads its operator constants from module globals. The parity
    run set them; the module defaults are the pre-campaign values."""
    from ceq import lm
    old = (lm.RHO, lm.SGATE_LAM, lm.HOPS)
    lm.RHO, lm.SGATE_LAM, lm.HOPS = PARITY_RHO, PARITY_LAM, PARITY_HOPS
    try:
        yield
    finally:
        lm.RHO, lm.SGATE_LAM, lm.HOPS = old


def _tiny(**kw):
    from ceq.hf.configuration_ceq import CEQConfig
    base = dict(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                num_attention_heads=4, max_position_embeddings=32)
    base.update(kw)
    return CEQConfig(**base)


# ---------------------------------------------------------------- P1, the operator


def test_the_package_implements_the_operator_that_reached_parity(device, parity_point):
    """RED first, and it is the claim the whole package rests on.

    A reader who downloads this repo gets whatever `modeling_ceq` computes. If
    that is not the operator the 1.0334 was measured on, the number on the model
    card is about code that is not in the box. Bitwise, against `ceq/lm.py`,
    which is where the parity run actually ran.
    """
    from ceq import lm
    from ceq.hf.modeling_ceq import sgate_operator
    torch.manual_seed(0)
    a = lm.Attention("sgate", 64, 4).to(device)
    x = torch.randn(2, 24, 64, device=device)
    q, k, _ = a.qkv_heads(x)
    ref = a.operator(q, k)
    got = sgate_operator(q, k, rho=PARITY_RHO, lam=PARITY_LAM)
    assert torch.equal(ref, got), (
        "the Hub copy of the PARITY operator has drifted from ceq/lm.py: max abs "
        "difference {:.3e}".format(float((ref - got).abs().max())))


def test_the_shipped_default_config_is_the_parity_point(device):
    """A default that is not the measured operating point is a trap: it runs, it
    trains, and it is not the thing anybody published a number about."""
    cfg = _tiny()
    assert cfg.operator == "sgate", cfg.operator
    assert (cfg.rho, cfg.lam, cfg.hops) == (PARITY_RHO, PARITY_LAM, PARITY_HOPS), (
        (cfg.rho, cfg.lam, cfg.hops))


def test_a_config_written_before_the_operator_key_existed_is_refused(device):
    """RED first, and this hazard was CREATED by changing the default.

    A `config.json` saved before 2026-08-25 carries `"rho": 0.9, "hops": 3` and
    no `"operator"` key, because there was only one operator. `from_dict` calls
    `cls(**config_dict)`, so loading that checkpoint after the default moved
    would build `sgate` at rho=0.9 hops=3 -- a point nobody has measured -- and
    run it on weights trained with `signed`. Nothing about the shapes would
    complain.

    The rule that catches it costs one line to state: setting any operator knob
    without naming the operator is refused.
    """
    from ceq.hf.configuration_ceq import CEQConfig
    with pytest.raises(ValueError, match="operator"):
        CEQConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                  num_attention_heads=4, rho=0.9, hops=3)


def test_selecting_the_control_operator_does_not_inherit_the_parity_rho(device):
    """RED first. rho=1.5 belongs to `sgate`; the L1 operator was measured at
    rho=0.9 and every number on record for it assumes that. A shared default
    would silently move the control."""
    from ceq.hf.configuration_ceq import CEQConfig
    signed = CEQConfig(operator="signed")
    assert (signed.rho, signed.hops) == (0.9, 3), (signed.rho, signed.hops)
    sgate = CEQConfig(operator="sgate")
    assert (sgate.rho, sgate.lam, sgate.hops) == (PARITY_RHO, PARITY_LAM, PARITY_HOPS)


def test_an_unknown_operator_name_is_refused_at_construction(device):
    """A typo in `operator` must not fall through to a default. Silently training
    the wrong operator for a week is the expensive version of this bug."""
    with pytest.raises(ValueError, match="operator"):
        _tiny(operator="sgaet")


def test_the_model_runs_end_to_end_on_both_operators(device):
    """Both names have to be live code paths, and they have to be DIFFERENT code
    paths. `PretrainedConfig.__init__` swallows unknown keywords into attributes
    without complaint, so a config that merely *stores* `operator="sgate"` while
    the block keeps computing the L1 form passes any test that only checks the
    forward is finite. The two arms are therefore compared against each other
    from identical weights."""
    from ceq.hf.modeling_ceq import CEQForCausalLM
    logits = {}
    for name in ("sgate", "signed"):
        torch.manual_seed(0)
        m = CEQForCausalLM(_tiny(operator=name)).to(device)
        x = torch.randint(0, 64, (2, 16), device=device, generator=None) * 0 + 3
        out = m(input_ids=x, labels=x)
        assert torch.isfinite(out.logits).all(), name
        out.loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in m.parameters()), name
        logits[name] = out.logits.detach()
    assert not torch.allclose(logits["sgate"], logits["signed"]), (
        "both operator names produced identical logits from identical weights; "
        "`operator` is being stored and not read")


# ------------------------------------------------------------- P2, the mask hazard


@pytest.fixture(params=["ceq.hf.modeling_ceq", "ceq.attention"])
def operator_module(request):
    """The operator is DELIBERATELY duplicated -- the Hub copies the modeling
    file and cannot import `ceq` -- so a guard added to one copy and not the
    other is the drift the duplication was already known to cause."""
    import importlib
    return importlib.import_module(request.param)


def test_an_additive_mask_that_is_not_the_min_convention_is_refused(device, operator_module):
    """RED first. Measured: with `-1e4` on columns 3+, the returned operator is
    `torch.equal` to the operator computed with NO mask at all, and the masked
    columns keep 0.48290979862213135 of weight.

    This is the same failure class as `BlockMask.from_kv_blocks` reporting a
    sparse pattern while computing fully dense: the caller is told the constraint
    was applied and it was not. `-10000.0` is the historical transformers mask
    value and still appears in live code, so this is reachable rather than
    theoretical.
    """
    ceq_operator = operator_module.ceq_operator
    torch.manual_seed(0)
    q = torch.randn(1, 1, 6, 8, device=device)
    k = torch.randn(1, 1, 6, 8, device=device)
    m = torch.zeros(1, 1, 1, 6, device=device)
    m[..., 3:] = -1e4
    with pytest.raises(ValueError, match="mask"):
        ceq_operator(q, k, m)


def test_both_legal_mask_conventions_still_agree(device, operator_module):
    """Calibration for the test above. A guard that rejects the two conventions
    that DO work would be worse than the hazard it replaces."""
    ceq_operator = operator_module.ceq_operator
    torch.manual_seed(0)
    q = torch.randn(1, 1, 6, 8, device=device)
    k = torch.randn(1, 1, 6, 8, device=device)
    keep = torch.ones(1, 1, 1, 6, dtype=torch.bool, device=device)
    keep[..., 3:] = False
    add = torch.zeros(1, 1, 1, 6, device=device)
    add[..., 3:] = torch.finfo(torch.float32).min
    a, b = ceq_operator(q, k, keep), ceq_operator(q, k, add)
    assert torch.equal(a, b), float((a - b).abs().max())
    assert float(a[..., 3:].abs().max()) == 0.0, "a masked key kept weight"


def test_the_operator_counts_the_key_positions_it_masked(device):
    """An observable counter, not only a raise. The mask class of bug is the one
    that reports the right pattern and computes another, so the package carries a
    number a caller can read back and compare against what it asked for."""
    from ceq.hf import modeling_ceq
    modeling_ceq.STATS.update(calls=0, masked_key_positions=0)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 6, 8, device=device)
    k = torch.randn(1, 1, 6, 8, device=device)
    keep = torch.ones(1, 1, 1, 6, dtype=torch.bool, device=device)
    keep[..., 4:] = False
    modeling_ceq.ceq_operator(q, k, keep)
    assert modeling_ceq.STATS["calls"] == 1, modeling_ceq.STATS
    assert modeling_ceq.STATS["masked_key_positions"] == 2, modeling_ceq.STATS


# ---------------------------------------------------------- P3, decode causality


def test_a_chunked_prefill_is_refused_rather_than_silently_uncausal(device):
    """RED first, measured behaviourally rather than by reading a mask.

    `is_causal=True` is correct only when q_len == kv_len, so the code guards it
    with `q_len == kv_len` -- which is right for a 1-token decode and WRONG for a
    chunk of 4 against a cache of 16, where it silently drops causality
    altogether. Perturbing key 15 moved query 0's output by 1.8396726846694946.
    """
    from ceq.hybrid import stock_attention
    torch.manual_seed(1)
    q = torch.randn(1, 2, 4, 8, device=device)
    k = torch.randn(1, 2, 16, 8, device=device)
    v = torch.randn(1, 2, 16, 8, device=device)
    with pytest.raises(NotImplementedError, match="causal"):
        stock_attention(q, k, v, None)


def test_a_single_token_decode_step_is_still_allowed(device):
    """Calibration. q_len == 1 against a cache of N is the one case where
    "attend to everything cached" IS the causal answer, and it must keep
    working or the guard has eaten the decode path."""
    from ceq.hybrid import stock_attention
    torch.manual_seed(1)
    q = torch.randn(1, 2, 1, 8, device=device)
    k = torch.randn(1, 2, 16, 8, device=device)
    v = torch.randn(1, 2, 16, 8, device=device)
    out = stock_attention(q, k, v, None)
    assert out.shape == (1, 2, 1, 8) and torch.isfinite(out).all()


def test_a_masked_chunked_prefill_is_allowed_because_the_mask_carries_causality(device):
    """The guard is about a MISSING constraint, not about the shape. When the
    caller supplies an explicit mask, torch never consults `is_causal` and the
    chunk is fine."""
    from ceq.hybrid import stock_attention
    torch.manual_seed(1)
    q = torch.randn(1, 2, 4, 8, device=device)
    k = torch.randn(1, 2, 16, 8, device=device)
    v = torch.randn(1, 2, 16, 8, device=device)
    mask = torch.zeros(1, 1, 4, 16, device=device)
    mask[..., :] = torch.finfo(torch.float32).min
    for i in range(4):
        mask[..., i, : 12 + i + 1] = 0.0
    out = stock_attention(q, k, v, mask)
    assert out.shape == (1, 2, 4, 8) and torch.isfinite(out).all()


# --------------------------------------------------- P4, the unmaterialized head


def test_an_unmaterialized_lm_head_raises_instead_of_returning_garbage(device, tmp_path):
    """RED first, and this is the worst of the four.

    `PreTrainedModel.is_remote_code()` is `cls._auto_class is not None`, so
    `register_for_auto_class()` -- mandatory for shipping modeling code -- routes
    the load onto a branch that refuses to tie. `lm_head.weight` stays on the
    META device, the forward RUNS, and the logits are uninitialized memory:
    measured 1.0053620544046395e+30 at [0,0,0], max abs difference
    1.0053620544046395e+30 against the model that was saved. transformers 5.3.0
    logs a warning and raises nothing.

    The shipped rollback is `tie_word_embeddings=False`. This test covers the
    caller who overrides it anyway.
    """
    from ceq.hf.modeling_ceq import CEQForCausalLM
    d = str(tmp_path / "tied")
    CEQForCausalLM(_tiny(tie_word_embeddings=True)).save_pretrained(d)
    m = CEQForCausalLM.from_pretrained(d)
    assert m.lm_head.weight.is_meta, (
        "the tying defect did not reproduce on transformers {}; this guard may "
        "now be dead code".format(transformers.__version__))

    # `.to()` is already loud -- torch refuses to copy out of a meta tensor, on
    # cpu and cuda alike. That is NOT the path that produced 1.005e+30: the
    # default CPU path never calls `.to()` at all, and that is the one where
    # nothing raised.
    with pytest.raises(NotImplementedError, match="meta"):
        m.to(device)
    with pytest.raises(RuntimeError, match="meta"):
        m(input_ids=torch.randint(0, 64, (1, 8)))


def test_the_untied_default_forwards_normally(device, tmp_path):
    """Calibration: the guard must not fire on the configuration that ships."""
    from ceq.hf.modeling_ceq import CEQForCausalLM
    d = str(tmp_path / "untied")
    ref = CEQForCausalLM(_tiny())
    ref.save_pretrained(d)
    m = CEQForCausalLM.from_pretrained(d).to(device).eval()
    x = torch.randint(0, 64, (1, 8), device=device)
    with torch.no_grad():
        got = m(input_ids=x).logits
        want = ref.to(device).eval()(input_ids=x).logits
    assert torch.equal(got, want), float((got - want).abs().max())


# -------------------------------------------------------- P5, the negative bound


def test_the_truncation_bound_refuses_a_rho_it_cannot_bound(device):
    """RED first. `truncation_bound(1.5, 2)` returns **-6.75** today: a negative
    error bound at the exact operating point the package now ships.

    The formula `rho^(K+1)/(1-rho)` is the geometric tail and needs rho < 1. At
    rho = 1.5 the series does not converge and the only thing that makes the path
    sum exact is nilpotency, which needs hops >= S-1, not 2. So hops=2 at rho=1.5
    is a genuine truncation with NO bound, and saying so is the honest output.
    """
    from ceq.attention import truncation_bound
    assert truncation_bound(0.9, 4) > 0.0
    with pytest.raises(ValueError, match="rho"):
        truncation_bound(PARITY_RHO, PARITY_HOPS)


# ----------------------------------------------- the adoptability bar, restated


def _tiny_llama(n_kv_heads):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=n_kv_heads, max_position_embeddings=64,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


@pytest.mark.parametrize("n_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_alpha_zero_is_bitwise_the_models_own_attention(device, n_kv_heads):
    """RED first on the GQA case, and it is a correction I owe on my own round-6
    report.

    "alpha = 0 is BITWISE stock" was asserted against `ceq.hybrid.stock_attention`
    -- this repository's own reimplementation of SDPA -- and not against the
    attention the model actually runs. Measured against the real `sdpa` path on
    a randomly-initialized Llama: **0.000e+00 at 4 kv heads and 1.490e-07 max
    abs / 2.609e-07 relative at 2 kv heads.** The cause is grouped-query
    attention: `_repeat_kv` materializes the expanded keys while transformers
    passes `enable_gqa=True` to torch, and those two reduce in different orders
    (measured 4.768e-07 between them on bare tensors).

    Qwen2.5-0.5B, the checkpoint the 9.2149-vs-9.2150 perplexity was measured on,
    is a GQA model. So the recorded number was internally consistent in exactly
    the way the output-layout bug was, and the fix is the same one: stop
    reimplementing the library and call it.
    """
    from ceq import hybrid
    m = _tiny_llama(n_kv_heads).to(device)
    x = torch.randint(0, 64, (1, 16), device=device)
    with torch.no_grad():
        m.set_attn_implementation("sdpa")
        stock = m(input_ids=x).logits
        hybrid._install(m, 0.0, hybrid.DEFAULT_RHO, hybrid.DEFAULT_HOPS)
        gated = m(input_ids=x).logits
        m.set_attn_implementation("sdpa")
    assert torch.equal(stock, gated), (
        "alpha=0 is not bitwise the model's own attention at {} kv heads: max "
        "abs {:.3e}, relative {:.3e}".format(
            n_kv_heads, float((stock - gated).abs().max()),
            float((stock - gated).abs().max() / stock.abs().max())))


@pytest.mark.parametrize("n_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_alpha_zero_generates_identically_to_stock(device, n_kv_heads):
    """The decode half of the same bar. Teacher-forced parity is blind to
    `is_causal`, which was correct in prefill and wrong at decode and produced
    ' The following the 1: 1: 1: 1:' against ' The capital of France is Paris.'
    with perplexity matching to 0.0000%."""
    from ceq import hybrid
    m = _tiny_llama(n_kv_heads).to(device)
    x = torch.randint(0, 64, (1, 8), device=device)
    with torch.no_grad():
        m.set_attn_implementation("sdpa")
        a = m.generate(x, max_new_tokens=8, do_sample=False)
        hybrid._install(m, 0.0, hybrid.DEFAULT_RHO, hybrid.DEFAULT_HOPS)
        b = m.generate(x, max_new_tokens=8, do_sample=False)
        m.set_attn_implementation("sdpa")
    assert torch.equal(a, b), (a.tolist(), b.tolist())


# ------------------------------------------- signedness is not free at init


def test_signedness_is_emergent_with_logit_spread_not_structural_at_lam_0p10(device):
    """A FINDING, asserted positively so it cannot quietly stop being true.

    Tier 3 -- an operator entry that can go negative -- is the property that
    justifies this module existing at all, and at `lam = 0.10` it is NOT
    structural. An entry is negative only when `softmax(w)_ij < lam *
    softmax(-w)_ij`, which needs that entry to sit far enough below its own row.
    At `nn.Linear` default initialization the logits are too flat to get there:
    over 20 draws the negative fraction was mean **3.91e-04 (cpu) / 4.99e-04
    (cuda)**, max 1.085e-03, at a mean within-row logit spread of 0.68. At
    unit-variance logits (spread 3.26) it is 5.25e-02 -- more than a hundred
    times higher -- and 0.225 at 8x.

    `min A = -0.1080` on record is the TRAINED 3.3M model. The operator earns its
    sign during training; it does not start with it. At `lam = 1.0` signedness IS
    structural -- rows sum to exactly zero -- and that is the trade the `lam` knob
    makes. Both ends are checked here so the claim is a curve, not an anecdote.
    """
    import math
    from ceq import lm
    from ceq.hf.modeling_ceq import sgate_operator

    def neg_fraction(q, k, lam=PARITY_LAM):
        op = sgate_operator(q, k, rho=PARITY_RHO, lam=lam)
        return float((op < 0).sum()) / op.numel()

    torch.manual_seed(0)
    a = lm.Attention("sgate", 64, 4).to(device)
    at_init = [neg_fraction(*lm.Attention("sgate", 64, 4).to(device)
                            .qkv_heads(torch.randn(2, 24, 64, device=device))[:2])
               for _ in range(8)]
    q = torch.randn(1, 1, 64, 32, device=device)
    k = torch.randn(1, 1, 64, 32, device=device)
    at_unit, at_8x = neg_fraction(q, k), neg_fraction(8.0 * q, k)

    assert max(at_init) < 3e-3, at_init
    assert at_unit > 30 * max(at_init), (max(at_init), at_unit)
    assert at_8x > at_unit, (at_unit, at_8x)
    # lam = 1 makes it structural: rows sum to exactly zero, so any row with a
    # positive entry must carry a negative one.
    assert float(sgate_operator(q, k, rho=PARITY_RHO, lam=1.0).min()) < 0.0
    rows = sgate_operator(q, k, rho=PARITY_RHO, lam=1.0).sum(-1)[..., 1:]
    assert float(rows.abs().max()) < 1e-6, float(rows.abs().max())


def test_the_package_states_that_signedness_is_not_free_at_initialization(device):
    """RED first. A caller reading `sgate_operator`'s docstring learns that the
    operator is signed. They must also learn how little of it is there at step
    zero, in the file that ships rather than in a markdown file that does not."""
    from ceq.hf.modeling_ceq import COSTS
    s = COSTS["signedness"]
    assert s["negative_fraction_at_nn_linear_init"] == {"cpu": 3.91e-04,
                                                       "cuda": 4.99e-04}
    assert s["negative_fraction_at_unit_variance_logits"] == 5.25e-02
    assert s["min_A_trained_3p3M"] == -0.1080
    assert s["structural_at_lam"] == 1.0


# ------------------------------------------------------------- the cost, in-band


def test_the_package_states_its_costs_in_the_file_that_ships(device):
    """A cost that lives only in README.md does not travel. The Hub copies
    `configuration_ceq.py` and `modeling_ceq.py` and nothing else, so a
    downloader who never opens this repo has to be able to read the price from
    the module they imported."""
    from ceq.hf.modeling_ceq import COSTS
    mem = COSTS["train_peak_memory_vs_sdpa"]
    assert mem == {128: 1.44, 256: 1.74, 512: 2.65, 1024: 4.45, 2048: 8.06}, mem
    assert COSTS["wall_clock_vs_softmax_at_parity"] == 1.32
    assert COSTS["gradient_checkpointing_peak_ratio"] == 0.379
    assert COSTS["kernel"] == "forward-only; the training path is pure torch"
    # EVERY entry, not the endpoints. The first version of this assertion checked
    # s=8 and s=128 only, and the dict it was guarding had 0.10547 at s=16 and
    # 0.03516 at s=32 -- both invented, against DONE.md's 0.08887 and 0.02637 --
    # and it passed. A guard that samples a table does not guard the table.
    decay = COSTS["content_conditional_sign_decay"]
    assert decay == {8: 0.17480, 16: 0.08887, 32: 0.02637, 64: 0.01172,
                     128: 0.00391, "slope": -1.389, "r2": 0.9938,
                     "softmax_at_every_s": 0.0}, decay
    assert COSTS["parity"] == {"ratio": 1.0334, "params": 3319296, "seeds": 5,
                               "size": "3.3M", "seq": 128}
    # RED first, and the most important entry in the dict. A price list that
    # carries the favourable number (1.0334x val loss) and omits the capability
    # result (a LOSS, one-sided Fisher p = 2.8e-05) is advertising, not a cost
    # statement -- and it is the one a downloader has no other way to see.
    cap = COSTS["capability"]
    assert cap["cogs_gen_exact_match"] == {"softmax": 0.0293, "sgate": 0.0000}
    assert cap["cogs_in_distribution"] == {"softmax": 0.9258, "sgate": 0.7734}
    assert cap["fisher_one_sided_p"] == 2.8e-05
    assert cap["params_both_arms"] == 3652096
    assert cap["verdict"] == "a loss, not a tie, and already present in-distribution"


def _evidence_paragraphs():
    """Paragraphs of the run-log corpus that CARRY RUN EVIDENCE.

    TWO DEFECTS ARE BEING REPAIRED HERE AND THE SECOND IS THE REAL ONE.

    1. THE CORPUS WAS ONE FILE. `DONE.md` rotates; when it does, the
       measurements move to `DONE_ARCHIVE_ROUND*.md` and every provenance
       assertion silently loses its target. Measured: `1.44x`, `1.0334`,
       `0.379x`, `3,319,296` appear 2/7/2/4 times in the round-1 archive. The
       corpus is now every `DONE*.md`.

    2. A MENTION WAS ACCEPTED AS A MEASUREMENT. `assert "3,319,296" in done`
       passes on ANY occurrence -- including the sentence of a finding that
       REPORTED THE NUMBER MISSING. That is exactly what happened: after the
       finding was written up, all four figures appeared in `DONE.md` exactly
       once each, inside the text reporting their absence, so the bind would
       pass ON THE REPORT OF ITS OWN FAILURE. A provenance test that a
       meta-discussion can satisfy is not a provenance test.

    So a hit counts only inside a paragraph that carries evidence: a `[RUN]`
    marker, or a markdown table row. Both are how this repo records a
    measurement; neither appears in prose about a number's absence.
    """
    out = []
    for f in sorted(pathlib.Path(REPO).glob("DONE*.md")):
        out += f.read_text(encoding="utf-8").split("\n\n")
    return out


def _is_evidence(block):
    """A paragraph carries run evidence iff it has a `[RUN]` marker or a
    markdown table row. Prose about a number being absent has neither."""
    return ("[RUN]" in block
            or any(ln.lstrip().startswith("|") for ln in block.splitlines()))


def _measured(needle, blocks):
    """THE FILTER LIVES HERE, NOT IN THE CORPUS BUILDER, AND THAT WAS A REAL
    DEFECT. The first version filtered while READING FILES and left `_measured`
    as a bare containment test -- so the must-fire control below, which passes
    literal blocks, bypassed the filter entirely and the control FAILED. A
    control that cannot reach the logic it guards is not guarding it."""
    return any(needle in b and _is_evidence(b) for b in blocks)


def test_provenance_rejects_a_mention_that_carries_no_run_evidence():
    """MUST-FIRE. The repair above is only worth having if it can tell a
    measurement from a mention, so it is shown doing that on planted text."""
    meta = ("**F4 EVERY SHIPPED COSTS NUMBER HAS LOST ITS PROVENANCE.** "
            "`3,319,296` occurs **0** times in `DONE.md`.")
    table = "| softmax | 3,319,296 | 1.8528 | **1.8838** | 69.9 s |"
    run = "[RUN] parameters 3,319,296 on both arms, 5 seeds."
    assert not _measured("3,319,296", [meta]), (
        "a paragraph REPORTING the number missing was accepted as provenance")
    assert _measured("3,319,296", [table]), "a table row was rejected"
    assert _measured("3,319,296", [run]), "a [RUN] paragraph was rejected"


def test_every_cost_the_package_states_is_reachable_from_a_recorded_measurement(device):
    """The numbers in COSTS are quoted, so they can be quoted WRONG. Each is
    checked against RUN EVIDENCE in the run-log corpus -- not against any
    mention anywhere in one file. See `_evidence_paragraphs` for why."""
    from ceq.hf.modeling_ceq import COSTS
    ev = _evidence_paragraphs()
    assert ev, "no run-evidence paragraphs found; the corpus glob is broken"
    missing = []

    def want(needle, label):
        if not _measured(needle, ev):
            missing.append((label, needle))

    for seq, ratio in COSTS["train_peak_memory_vs_sdpa"].items():
        if not (_measured("{:.2f}x".format(ratio), ev)
                or _measured("{:.2f}".format(ratio), ev)):
            missing.append(("train_peak_memory_vs_sdpa[{}]".format(seq), ratio))
    for s_, rate in COSTS["content_conditional_sign_decay"].items():
        if isinstance(s_, int):
            want("{:.5f}".format(rate), "content_conditional_sign_decay[{}]".format(s_))
    want("1.32x", "wall_clock_vs_softmax_at_parity")
    want("0.379x", "gradient_checkpointing_peak_ratio")
    want("1287.5", "parity wall-clock")
    want("487.7", "parity wall-clock")
    want("1.0334", "parity ratio")
    want("3,319,296", "parity params")
    cap = COSTS["capability"]
    want("0.0293", "cogs_gen_exact_match softmax")
    want("0.9258", "cogs_in_distribution softmax")
    want("0.7734", "cogs_in_distribution sgate")
    want("2.8e-05", "fisher_one_sided_p")
    want("{:,}".format(cap["params_both_arms"]), "params_both_arms")
    assert not missing, (
        "shipped COSTS figures with no run evidence anywhere in DONE*.md:\n  "
        + "\n  ".join("{}: {}".format(a, b) for a, b in missing))


# ------------------------------- invariants the shipped docstrings assert
#
# These two were GREEN on first run and are NOT build claims of this round. They
# are here because both facts are ASSERTED IN PROSE inside files that ship, and a
# claim in a shipped file that no test checks is the thing this round exists to
# remove.


def test_the_modeling_file_has_no_device_conditional_path(device):
    """`modeling_ceq.py` says "there is no `if q.is_cuda` branch anywhere in this
    file". Grepped rather than trusted: a CPU fallback that exists as a sentence
    is not a CPU fallback."""
    import ast
    src = open(os.path.join(REPO, "ceq", "hf", "modeling_ceq.py"),
               encoding="utf-8").read()
    # AST, not grep: the module docstring says the words "if q.is_cuda" while
    # explaining that there is no such branch, and a grep cannot tell the
    # sentence from the code. The first version of this test could not either.
    attrs = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    for banned in ("is_cuda", "cuda"):
        assert banned not in attrs, banned


def test_the_gate_registers_both_the_attention_and_its_mask(device):
    """`ceq/attention.py` has this guard in tests/w6; `ceq/hybrid.py` -- the path
    a pretrained checkpoint actually adopts -- had none. Registering an attention
    implementation WITHOUT its mask makes transformers pass
    `attention_mask=None`, which silently drops padding, packing and
    sliding-window constraints."""
    from transformers.masking_utils import AttentionMaskInterface
    from transformers.modeling_utils import AttentionInterface
    from ceq import hybrid
    hybrid.register()
    assert hybrid.NAME in AttentionInterface._global_mapping, "attention not registered"
    assert hybrid.NAME in AttentionMaskInterface._global_mapping, "MASK not registered"


# ------------------------------------------------------- the stranger's command


def test_the_smoke_test_passes_on_a_cuda_free_interpreter(device):
    """One command, CPU only, no Triton, no network. Run in a fresh interpreter
    with CUDA hidden, because that is the machine that downloads this."""
    from conftest import run_isolated
    rc, out, err = run_isolated("""
        import os, subprocess, sys
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        p = subprocess.run([sys.executable, "-m", "ceq.hf.smoke"],
                           capture_output=True, text=True, cwd=r"{repo}",
                           env=dict(os.environ, PYTHONPATH=r"{repo}"))
        sys.stdout.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise SystemExit(p.returncode)
    """.format(repo=REPO))
    assert rc == 0, "smoke test failed:\n{}\n{}".format(out[-4000:], err[-4000:])
    for token in ("torch", "transformers", "python", "ok: all 7 checks passed"):
        assert token in out.lower(), (token, out[-2000:])


def test_the_smoke_test_reports_the_version_matrix_it_was_verified_against(device):
    """A green smoke test that does not say WHAT it was green against is not
    evidence. torch 2.5.1+cu121 / transformers 5.3.0 / python 3.11.9 is the
    matrix every number in this repo was measured on."""
    src = open(os.path.join(REPO, "ceq", "hf", "smoke.py"), encoding="utf-8").read()
    for pinned in ("2.5.1", "5.3.0", "3.11"):
        assert pinned in src, pinned
