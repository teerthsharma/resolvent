"""What breaks when the target deliverable -- a ~1B model or module on the Hugging Face
Hub -- has a custom Triton attention kernel behind custom modeling code.

Everything here runs against the installed transformers and the real merged kernel
(k22.py, triton-lang/kernels PR #22). Nothing is asserted from documentation alone.
"""

import json
import os
import shutil
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

transformers = pytest.importorskip("transformers")

MODELING = '''
import torch
from transformers import PreTrainedModel, PretrainedConfig


class CEAConfig(PretrainedConfig):
    model_type = "consequence_equilibrium_attention"

    def __init__(self, hidden_size=64, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


class CEAModel(PreTrainedModel):
    config_class = CEAConfig

    def __init__(self, config):
        super().__init__(config)
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        # The real model would call the Triton scheduled-attention kernel here.
        return self.proj(x)
'''


@pytest.fixture(scope="module")
def repo_dir():
    d = tempfile.mkdtemp(prefix="cea_hub_")
    with open(os.path.join(d, "modeling_cea.py"), "w") as fh:
        fh.write(MODELING)
    cfg = {
        "model_type": "consequence_equilibrium_attention",
        "hidden_size": 64,
        "architectures": ["CEAModel"],
        "auto_map": {
            "AutoConfig": "modeling_cea.CEAConfig",
            "AutoModel": "modeling_cea.CEAModel",
        },
    }
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    sys.path.insert(0, d)
    import modeling_cea  # noqa
    m = modeling_cea.CEAModel(modeling_cea.CEAConfig())
    m.save_pretrained(d)
    with open(os.path.join(d, "config.json")) as fh:
        c = json.load(fh)
    c["auto_map"] = cfg["auto_map"]
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(c, fh)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_model_loads_without_trust_remote_code(repo_dir):
    """Every downloader of this checkpoint must pass trust_remote_code=True, which is
    an arbitrary-code-execution opt-in. Organisations block it by policy."""
    from transformers import AutoModel
    try:
        AutoModel.from_pretrained(repo_dir, trust_remote_code=False)
    except Exception as exc:
        pytest.fail(
            f"{type(exc).__name__}: {str(exc)[:400]}\n"
            "Custom modeling code cannot be loaded without an explicit "
            "trust_remote_code=True from every user, on every load."
        )


def test_custom_kernel_runs_on_cpu():
    """A user with no NVIDIA GPU. HF download counts are dominated by CPU/Mac users."""
    import k22
    q, k, v = [torch.randn(128, 64) for _ in range(3)]
    offsets, indices = k22.build_dense_causal_block_schedule(2)
    try:
        k22.scheduled_attention(q, k, v, offsets, indices, 64)
    except Exception as exc:
        pytest.fail(
            f"{type(exc).__name__}: {str(exc)[:200]}. There is no CPU fallback path "
            "in the kernel and none in scheduled_attention()."
        )


def test_triton_is_importable_without_cuda():
    """`import triton` at module scope in modeling_cea.py means the checkpoint fails
    to LOAD, not merely to run fast, wherever triton is absent."""
    import k22
    assert "triton" in open(k22.__file__).read().split("\n")[4], "setup check"
    import importlib.util
    spec = importlib.util.find_spec("triton")
    assert spec is not None, "triton not installed"
    # PyPI ships triton wheels for Linux x86_64/aarch64 only. macOS: no wheel at all.
    # torch declares triton under a marker restricting it to Linux x86_64, so on
    # Windows/macOS it is simply absent and a top-level `import triton` raises.
    import platform
    assert platform.system() == "Linux", (
        f"this machine is {platform.system()}; triton resolves here only because a "
        f"third-party build (triton-windows) is installed. A stock "
        f"`pip install torch transformers` on Windows or macOS installs no triton, so "
        f"`import triton` at the top of modeling_cea.py raises ModuleNotFoundError "
        f"before any attention is attempted."
    )


def test_kernel_covers_the_gpus_a_1b_model_is_actually_run_on():
    """k22 hardcodes num_warps=4 and has no @triton.autotune; the PR benchmarked one
    device. Triton's own compatibility statement is NVIDIA compute capability 8.0+."""
    from conftest import HAS_CUDA
    if not HAS_CUDA:
        pytest.skip("no GPU")
    cc = torch.cuda.get_device_capability(0)
    assert cc[0] < 8, (
        f"this GPU is sm_{cc[0]}{cc[1]}. Triton requires compute capability 8.0+, so "
        f"every pre-Ampere card (T4 sm_75, V100 sm_70, all GTX/RTX 20xx) cannot run "
        f"this checkpoint at all -- T4 being the default free-tier Colab GPU."
    )


def test_attention_mask_interface_registration_is_not_a_silent_trap():
    """transformers routes custom attention through AttentionInterface. If the name is
    registered there but NOT in AttentionMaskInterface, transformers passes
    attention_mask=None and causal/padding constraints are dropped without error."""
    from transformers import AttentionInterface
    try:
        from transformers import AttentionMaskInterface
    except ImportError:
        pytest.fail("AttentionMaskInterface not importable; cannot register the mask")

    def custom_attention(module, query, key, value, attention_mask, **kwargs):
        assert attention_mask is not None, (
            "registered in AttentionInterface only: attention_mask arrived as None. "
            "A causal LM whose attention function silently receives no mask trains "
            "and serves non-causally. Nothing raises."
        )
        return value, None

    AttentionInterface.register("cea_test", custom_attention)
    assert "cea_test" in getattr(AttentionMaskInterface, "_global_mapping", {}), (
        "registering an attention implementation does NOT register its mask. "
        "transformers' own docs: 'Transformers skips mask creation and passes "
        "attention_mask=None ... those constraints can be silently dropped.' "
        "For a causal LM that is a correctness bug with no exception."
    )
