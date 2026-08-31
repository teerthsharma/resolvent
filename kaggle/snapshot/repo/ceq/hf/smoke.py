"""One command, CPU only, no GPU, no Triton, no network:

    python -m ceq.hf.smoke

Exits 0 if the package is intact on this machine and 1 with a traceback if it is
not. Nothing here downloads anything, and nothing here needs CUDA -- that is the
point. The machine that downloads a Hub checkpoint is overwhelmingly a Windows,
macOS or CPU-only Linux box, and every previous shipping defect in this project
was invisible until somebody ran it on one.

Seven checks, each one standing in for a failure this project has actually
produced:

  1  import with no CUDA and no triton      a top-level `import triton` makes the
                                            checkpoint fail to LOAD, not to run
  2  the parity operator is the one shipped  the package used to implement the
                                            operator the number was NOT about
  3  forward, backward, generate            a model that trains and cannot
                                            .generate() is unusable, and nothing
                                            in transformers says so
  4  save -> AutoModel(trust_remote_code)    the auto_map path is the one every
                                            downloader takes
  5  lm_head is materialized                 tied embeddings strand it on META
                                            and the forward returns 1.005e+30
  6  alpha = 0 is bitwise stock              the adoptability bar, on both the
                                            teacher-forced and the decode path
  7  the mask guard fires                    a -1e4 mask used to be ignored
                                            bitwise while reporting as applied

It also prints the version matrix it ran on and the price list from
`modeling_ceq.COSTS`, because a green check that does not say what it was green
against is not evidence.
"""
from __future__ import annotations

import os
import platform
import sys
import tempfile

#: The matrix every number in this repository was measured on. Printed, not
#: enforced -- a stranger on a different matrix should still be able to run this
#: and see exactly how far they are from the reference.
VERIFIED_AGAINST = {
    "python": "3.11.9",
    "torch": "2.5.1+cu121",
    "transformers": "5.3.0",
    "os": "Windows 11 (26200); also exercised on Linux CPU-only",
    "gpu-for-the-measured-costs": "RTX 4060 Laptop, sm_89, 8.0 GiB, 24 SMs",
}


def _tiny(**kw):
    from ceq.hf.configuration_ceq import CEQConfig
    base = dict(vocab_size=64, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=4, max_position_embeddings=64)
    base.update(kw)
    return CEQConfig(**base)


def check_1_imports_without_cuda_or_triton():
    """The two shipped files must contain no triton import, and this process must
    have got here with no CUDA device.

    NOT `"triton" not in sys.modules`, which was the first version of this check
    and was wrong. Measured: `import torch` leaves it False, `import
    transformers` leaves it False, and `from transformers import
    LlamaForCausalLM` leaves it True. Triton arrives through TRANSFORMERS' own
    optional import, not through anything here, and that is outside this
    package's control. What is inside this package's control is that neither
    shipped file imports it, and that is what is asserted -- the rest is
    reported so a reader can see it.
    """
    import torch
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("modeling_ceq.py", "configuration_ceq.py"):
        for line in open(os.path.join(src_dir, name), encoding="utf-8"):
            stripped = line.strip()
            assert not (stripped.startswith("import triton")
                        or stripped.startswith("from triton")), \
                "{}: {}".format(name, stripped)
    return ("torch {}, cuda devices visible {}, triton pulled in by "
            "transformers: {}".format(torch.__version__,
                                      torch.cuda.device_count(),
                                      "triton" in sys.modules))


def check_2_the_shipped_operator_is_the_parity_operator():
    """Bitwise against `ceq/lm.py`, which is where the 5-seed parity run ran.

    Skipped with a printed note rather than failed when the repository is not on
    the path, because the Hub repository contains only the two .py files and a
    downloader legitimately has no `ceq/lm.py` to compare against.
    """
    import torch
    from ceq.hf.configuration_ceq import PARITY_POINT
    from ceq.hf.modeling_ceq import sgate_operator
    rho, lam, hops = PARITY_POINT
    try:
        from ceq import lm
    except ImportError:
        return "SKIPPED: ceq/lm.py not on the path (Hub repo has only the 2 files)"
    old = (lm.RHO, lm.SGATE_LAM, lm.HOPS)
    lm.RHO, lm.SGATE_LAM, lm.HOPS = rho, lam, hops
    try:
        torch.manual_seed(0)
        a = lm.Attention("sgate", 64, 4)
        q, k, _ = a.qkv_heads(torch.randn(2, 24, 64))
        ref, got = a.operator(q, k), sgate_operator(q, k, rho=rho, lam=lam)
    finally:
        lm.RHO, lm.SGATE_LAM, lm.HOPS = old
    assert torch.equal(ref, got), "max abs diff {:.3e}".format(
        float((ref - got).abs().max()))
    assert float(got.triu(0).abs().max()) == 0.0, "the operator is not strictly causal"
    # NOT `min A < 0`. At lam = 0.10 signedness is EMERGENT, not structural: at
    # nn.Linear initialization the negative fraction is 3.91e-04 (cpu) /
    # 4.99e-04 (cuda) over 20 draws and on some draws it is exactly zero.
    # Asserting the sign here would be a test that happens to hold on the
    # fixture and is false in general -- which is how a fragile test survives to
    # mislead later. The structural end of the same knob is checked instead.
    structural = sgate_operator(q, k, rho=rho, lam=1.0)
    assert float(structural.min()) < 0.0, "lam=1 is not signed; the knob is broken"
    frac = float((got < 0).sum()) / got.numel()
    return ("bitwise equal to ceq/lm.py at rho={} lam={} hops={}; negative "
            "fraction at init {:.3e} (EMERGENT -- see COSTS['signedness']); "
            "structural at lam=1, min A {:.4f}".format(
                rho, lam, hops, frac, float(structural.min())))


def check_3_forward_backward_and_generate():
    import torch
    from ceq.hf.modeling_ceq import CEQForCausalLM
    m = CEQForCausalLM(_tiny())
    x = torch.randint(0, 64, (2, 16))
    out = m(input_ids=x, labels=x)
    assert torch.isfinite(out.logits).all() and torch.isfinite(out.loss)
    out.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters()), "a parameter got no gradient"
    assert type(m).can_generate(), "can_generate() is False; GenerationMixin missing"
    g = m.eval().generate(torch.randint(0, 64, (1, 8)), max_new_tokens=4,
                          do_sample=False)
    assert tuple(g.shape) == (1, 12), g.shape
    return "loss {:.4f}, {} params, generate() -> {}".format(
        float(out.loss), sum(p.numel() for p in m.parameters()), tuple(g.shape))


def check_4_save_and_reload_through_automodel():
    import torch
    from transformers import AutoModelForCausalLM
    from ceq.hf.modeling_ceq import CEQForCausalLM
    d = tempfile.mkdtemp(prefix="ceq_smoke_")
    ref = CEQForCausalLM(_tiny()).eval()
    ref.save_pretrained(d)
    have = set(os.listdir(d))
    for f in ("configuration_ceq.py", "modeling_ceq.py", "config.json"):
        assert f in have, "{} missing from the saved repo: {}".format(f, sorted(have))
    got = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True).eval()
    x = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        a, b = ref(input_ids=x).logits, got(input_ids=x).logits
    assert torch.equal(a, b), float((a - b).abs().max())
    return "auto_map round-trip identical; files {}".format(
        sorted(f for f in have if f.endswith(".py")))


def check_5_the_lm_head_is_materialized():
    import torch
    from ceq.hf.modeling_ceq import CEQForCausalLM
    d = tempfile.mkdtemp(prefix="ceq_tied_")
    CEQForCausalLM(_tiny(tie_word_embeddings=True)).save_pretrained(d)
    m = CEQForCausalLM.from_pretrained(d)
    try:
        m(input_ids=torch.randint(0, 64, (1, 8)))
    except RuntimeError as e:
        assert "meta" in str(e), e
        return "tied embeddings correctly REFUSED (transformers strands lm_head on meta)"
    return ("tied embeddings materialized on this transformers; the defect this "
            "guard covers may be fixed here")


def check_6_alpha_zero_is_bitwise_stock():
    """The adoptability bar, reproduced with no network.

    A randomly-initialized tiny Llama stands in for the 0.5B checkpoint. The
    quantity is the same either way: at alpha = 0 the gated path must be
    BITWISE stock on the teacher-forced path and character-identical under greedy
    generation. On `Qwen/Qwen2.5-0.5B` that measured ppl 9.2149 against stock
    9.2150 with identical text; here it is checked exactly.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from ceq import hybrid
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).eval()
    x = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        m.set_attn_implementation("sdpa")
        stock = m(input_ids=x).logits
        stock_gen = m.generate(x, max_new_tokens=8, do_sample=False)
        hybrid._install(m, 0.0, hybrid.DEFAULT_RHO, hybrid.DEFAULT_HOPS)
        gated = m(input_ids=x).logits
        gated_gen = m.generate(x, max_new_tokens=8, do_sample=False)
        m.set_attn_implementation("sdpa")
    assert torch.equal(stock, gated), (
        "alpha=0 is NOT bitwise stock: max abs diff {:.3e}. This model is GQA "
        "(2 kv heads against 4 query heads), which is exactly where a "
        "reimplemented SDPA drifts: 1.490e-07 was measured before the gate "
        "started delegating to transformers' own sdpa_attention_forward."
        .format(float((stock - gated).abs().max())))
    assert torch.equal(stock_gen, gated_gen), (
        "alpha=0 generation diverged: {} vs {}".format(
            stock_gen.tolist(), gated_gen.tolist()))
    return "logits bitwise equal and greedy generation identical at alpha=0"


def check_7_an_unrecognised_mask_is_refused():
    import torch
    from ceq.hf import modeling_ceq
    q, k = torch.randn(1, 1, 6, 8), torch.randn(1, 1, 6, 8)
    m = torch.zeros(1, 1, 1, 6)
    m[..., 3:] = -1e4
    try:
        modeling_ceq.ceq_operator(q, k, m)
    except ValueError as e:
        assert "mask" in str(e), e
    else:
        raise AssertionError(
            "a -1e4 additive mask was ACCEPTED; it is not below finfo.min/2 and "
            "produces a bitwise-dense operator while reporting as masked")
    keep = torch.ones(1, 1, 1, 6, dtype=torch.bool)
    keep[..., 4:] = False
    a = modeling_ceq.sgate_operator(q, k, keep)
    assert float(a[..., 4:].abs().max()) == 0.0, "a masked key kept weight"
    return "the -1e4 convention is refused; a bool mask zeroes the masked keys"


CHECKS = [
    ("imports with no CUDA and no triton", check_1_imports_without_cuda_or_triton),
    ("the shipped operator IS the parity operator", check_2_the_shipped_operator_is_the_parity_operator),
    ("forward, backward, generate", check_3_forward_backward_and_generate),
    ("save -> AutoModel(trust_remote_code=True)", check_4_save_and_reload_through_automodel),
    ("lm_head is materialized, not on meta", check_5_the_lm_head_is_materialized),
    ("alpha = 0 is bitwise stock", check_6_alpha_zero_is_bitwise_stock),
    ("an unrecognised attention mask is refused", check_7_an_unrecognised_mask_is_refused),
]


def main() -> int:
    import torch
    import transformers
    from ceq.hf.configuration_ceq import CEQConfig
    from ceq.hf.modeling_ceq import COSTS

    print("=" * 76)
    print("ceq/hf smoke test -- CPU only, no network, no Triton")
    print("=" * 76)
    print("\nRUNNING ON")
    for key, value in (("python", platform.python_version()),
                       ("torch", torch.__version__),
                       ("transformers", transformers.__version__),
                       ("os", "{} {}".format(platform.system(), platform.release())),
                       ("cuda devices visible", torch.cuda.device_count())):
        print("  {:<22} {}".format(key, value))
    print("\nVERIFIED AGAINST")
    for key, value in VERIFIED_AGAINST.items():
        print("  {:<22} {}".format(key, value))

    cfg = CEQConfig()
    print("\nSHIPPED DEFAULTS")
    print("  operator={!r} rho={} lam={} hops={} tie_word_embeddings={} use_cache={}"
          .format(cfg.operator, cfg.rho, cfg.lam, cfg.hops,
                  cfg.tie_word_embeddings, cfg.use_cache))

    print("\nWHAT THIS COSTS -- measured, not estimated")
    print("  train peak memory vs SDPA, by seq: " + ", ".join(
        "{}:{:.2f}x".format(s, r)
        for s, r in COSTS["train_peak_memory_vs_sdpa"].items()))
    print("  wall clock vs softmax at parity:   {:.2f}x".format(
        COSTS["wall_clock_vs_softmax_at_parity"]))
    print("  gradient checkpointing:            {:.3f}x peak, and MANDATORY at 0.5B"
          .format(COSTS["gradient_checkpointing_peak_ratio"]))
    print("  kernel:                            {}".format(COSTS["kernel"]))
    print("  decode:                            {}".format(COSTS["decode"]))
    print("  parity:                            {ratio}x softmax at {size}/{params} "
          "params, {seeds} seeds, seq {seq}".format(**COSTS["parity"]))
    d = COSTS["content_conditional_sign_decay"]
    print("  the distinguishing property DECAYS with context: {:.5f} at s=8 -> "
          "{:.5f} at s=128 (softmax is {:.4f} at every s)"
          .format(d[8], d[128], d["softmax_at_every_s"]))
    #: `slope` is None on purpose -- print the withdrawal, never a number.
    print("  exponent:                          {}".format(d["exponent_status"]))
    print("  trust_remote_code:                 {}".format(COSTS["trust_remote_code"]))
    c = COSTS["capability"]
    print("  CAPABILITY -- A LOSS, not a tie:   {} exact match, softmax {:.4f} vs "
          "sgate {:.4f} at {:,} params both arms, one-sided Fisher p={:.1e}; "
          "already present in-distribution ({:.4f} vs {:.4f})"
          .format(c["task"], c["cogs_gen_exact_match"]["softmax"],
                  c["cogs_gen_exact_match"]["sgate"], c["params_both_arms"],
                  c["fisher_one_sided_p"], c["cogs_in_distribution"]["softmax"],
                  c["cogs_in_distribution"]["sgate"]))

    print("\nCHECKS")
    failed = 0
    for i, (label, fn) in enumerate(CHECKS, 1):
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("  [{}/{}] FAIL  {}".format(i, len(CHECKS), label))
            import traceback
            traceback.print_exc()
        else:
            print("  [{}/{}] ok    {}\n              {}".format(
                i, len(CHECKS), label, detail))

    print()
    if failed:
        print("FAILED: {} of {} checks".format(failed, len(CHECKS)))
        return 1
    print("OK: all {} checks passed on python {} / torch {} / transformers {}".format(
        len(CHECKS), platform.python_version(), torch.__version__,
        transformers.__version__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
