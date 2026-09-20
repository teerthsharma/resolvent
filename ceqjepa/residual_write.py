"""THE RED INSTRUMENT for the residual-write leap.

THE LEAP, restated from the task that launched this module: context in the
wired `smprime` arm is claimed to live in the residual stream as a per-token,
per-layer BINARY OVERWRITE -- `Z^{1-beta}` is not a row gain but a RESET, and
a token whose `(1-beta) * log Z` crosses the dtype's `LOG_MAX` has its residual
erased and replaced by the read, rather than merely nudged.

WHAT IS ALREADY PROVED AND WHAT IS NOT. `tests/curvature/test_beta_axis_is_a_
row_gain.py` measures, on isolated q/k/v, that beta is a per-row scalar GAIN
`Z_i^(1-beta)` on the softmax read-out and nothing else -- that is arithmetic,
not a hypothesis. What THAT test does not measure is magnitude: whether the
gain stays a perturbation of the residual stream or swamps it, on the WIRED
24-layer arm, at its own shipped init. That is this module's only job.

    r_i(beta, layer) = || Z_i^{1-beta} * (a @ v)_i || / || x_residual_i ||

`x_residual` is the tensor `ceq/hf/modeling_ceq.py::CEQBlock.forward` is about
to add to (`x` in `x = x + self.self_attn(self.input_layernorm(x), ...)`).
`a @ v` is `ceq/arm_smprime.py::readout`'s own return, called through this
module rather than re-derived: `capture_forward` hooks `CEQAttention`'s own
submodules to pull the exact q/k/v/u/theta the model computes, and
`residual_write` calls `ceq.arm_smprime.numerator` / `readout` on them
UNMODIFIED. Nothing here reimplements the arm; `ceq/arm_smprime.py` is not
edited by this lane.

ONE FORWARD PASS, BETA SWEPT AFTER THE FACT. The model's own per-layer `beta`
is left at the shipped corner (`1.0`) for the one real forward pass that
produces `x_residual`, and the three beta values are applied only to the
POST-HOC `Z^{1-beta}` factor and to `readout`'s own beta argument -- the same
protocol `test_beta_axis_is_a_row_gain.py` uses (one q/k/v draw, beta swept),
extended here from an isolated cell to the full 24-layer wired model. This
means "at the shipped corner" is exact (the whole point is that `Z^0 = 1`, so
`r` there is just `||a@v|| / ||x_residual||` at what the model actually ran)
and "at an interior beta" answers "how large would this layer's write have
been here, at its own real activations, had beta been smaller" -- a real
question about the mechanism, not a claim that a beta-0.5 model would have
produced the same `x_residual` upstream (it would not; that is a different,
much more expensive experiment this module does not run).

ENVIRONMENT NOTE, NOT A REPO FIX. `requirements.txt` documents that this box's
installed `torchvision` is stranded against `torch==2.14.0`
(`operator torchvision::nms does not exist`), and `transformers==5.3.0`
imports it unconditionally at module scope through
`modeling_utils -> loss.loss_utils -> loss_d_fine -> ... -> image_utils`, so
plain `from transformers import PreTrainedModel` fails before any of this
module's own code runs. The documented repair is a global
`pip uninstall torchvision torchaudio`; mutating the shared environment is out
of scope for a one-file lane and would reach concurrent lanes mid-run, so
`_stub_torchvision_if_broken` does the process-local equivalent instead: a
fake `torchvision` satisfying the two symbols that one import chain touches
(`torchvision.transforms.InterpolationMode`, `torchvision.ops.nms`), which
this instrument never calls either way.
"""
from __future__ import annotations

import enum
import importlib.machinery
import json
import math
import pathlib
import sys
import types

import torch


def _stub_torchvision_if_broken() -> None:
    try:
        import torchvision  # noqa: F401
        return
    except Exception:
        pass

    def _mk(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return m

    tv, tv_t, tv_ops = _mk("torchvision"), _mk("torchvision.transforms"), _mk("torchvision.ops")
    tv_t.InterpolationMode = enum.Enum(
        "InterpolationMode",
        ["NEAREST", "NEAREST_EXACT", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS"])

    def _dead_nms(*_a, **_k):
        raise RuntimeError("stub torchvision.ops.nms: real torchvision is "
                            "broken in this environment (see requirements.txt); "
                            "this instrument never calls it")

    tv_ops.nms = _dead_nms
    tv.transforms, tv.ops = tv_t, tv_ops
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = tv_t
    sys.modules["torchvision.ops"] = tv_ops


_stub_torchvision_if_broken()

from ceq.arm_smprime import numerator as _smp_numerator  # noqa: E402
from ceq.arm_smprime import readout as _smp_readout  # noqa: E402
from ceq.hf.configuration_ceq import CEQConfig  # noqa: E402
from ceq.hf.modeling_ceq import CEQModel  # noqa: E402

#: `ceq/hf/configuration_ceq.py::SMPRIME_CORNER` -- the arm's shipped init.
SHIPPED_BETA = 1.0
#: Strictly between the shipped corner and the arm's OTHER named corner
#: (beta=0, `ceq/arm_smprime.py`'s own bind point, already swept by
#: `tests/curvature/test_beta_axis_is_a_row_gain.py` at 0.0/0.37/1.0). Chosen
#: to not duplicate that file's own points: one at the midpoint, one close to
#: the beta=0 corner without landing on it.
INTERIOR_BETAS = (0.5, 0.1)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def find_smprime_checkpoint() -> str | None:
    """Any on-disk `config.json` naming `operator: "smprime"`. Every shipped
    CEQ checkpoint on this box (`ceq/hf_artifact/config.json`,
    `ceqjepa/hf/config.json`) is `"sgate"` or a different `model_type`
    entirely, so this is expected to read `None`; it is a real search rather
    than an assumption so a future `smprime` checkpoint is picked up without
    editing this function."""
    for p in ROOT.rglob("config.json"):
        if "kaggle" in p.parts:
            continue
        try:
            cfg = json.loads(p.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if cfg.get("model_type") == "ceq" and cfg.get("operator") == "smprime":
            return str(p)
    return None


def _log_max(dtype: torch.dtype) -> float:
    return math.log(torch.finfo(dtype).max)


def _shape(t: torch.Tensor, n_heads: int, d_head: int) -> torch.Tensor:
    b, s, _ = t.shape
    return t.view(b, s, n_heads, d_head).transpose(1, 2)


class _Capture:
    """One block's worth of tensors, filled in by `capture_forward`'s hooks.
    `qk`/`g` are plain floats (the arm's other two switches, read off the
    module so a future non-shipped `qk`/`g` config is still measured
    correctly) and everything else is a tensor, bitwise what
    `CEQAttention._smprime` built for this forward pass."""

    __slots__ = ("x_residual", "q", "k", "v", "u", "theta", "qk", "g")


def capture_forward(model: CEQModel, input_ids: torch.Tensor) -> list[_Capture]:
    """One real forward pass through `model`, returning one `_Capture` per
    block in layer order. Hooks `CEQBlock` (for the residual it is about to
    add to) and `CEQAttention.qkv` / `.m_head` / `.theta_head` (for exactly
    the q/k/v/u/theta `_smprime` computes) -- nothing here recomputes what the
    model already computed."""
    caps: list[_Capture] = []
    handles = []

    def on_block(_block, args):
        c = _Capture()
        c.x_residual = args[0].detach()
        caps.append(c)

    def make_on_qkv(attn):
        def on_qkv(_module, _args, out):
            q, k, v = out.chunk(3, dim=-1)
            c = caps[-1]
            c.q = _shape(q, attn.n_heads, attn.d_head).detach()
            c.k = _shape(k, attn.n_heads, attn.d_head).detach()
            c.v = _shape(v, attn.n_heads, attn.d_head).detach()
            c.qk, c.g = float(attn.qk), float(attn.g)
        return on_qkv

    def on_u(_module, _args, out):
        caps[-1].u = out.squeeze(-1).unsqueeze(-2).detach()

    def on_theta(_module, _args, out):
        caps[-1].theta = out.squeeze(-1).unsqueeze(-2).detach()

    for block in model.layers:
        attn = block.self_attn
        if attn.operator != "smprime":
            raise ValueError(
                "capture_forward is wired for operator='smprime' only, got "
                "{!r}".format(attn.operator))
        handles.append(block.register_forward_pre_hook(on_block))
        handles.append(attn.qkv.register_forward_hook(make_on_qkv(attn)))
        handles.append(attn.m_head.register_forward_hook(on_u))
        handles.append(attn.theta_head.register_forward_hook(on_theta))
    try:
        model.eval()
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    return caps


def residual_write(cap: _Capture, beta: float) -> dict:
    """The RED's own quantities for one captured block at one beta.

      z                 `Z_i`, `ceq.arm_smprime.operator`'s own `mod.sum(-1)`
                         before it is raised to any power -- real, `[B,H,S]`.
      r                 `||Z^{1-beta} (a@v)|| / ||x_residual||`, per
                         (batch, token), heads concatenated the way
                         `CEQAttention.forward` concatenates them before
                         `o_proj` (`[B,S]`).
      crosses_log_max   `(1-beta) * log(Z) > log(dtype.max)` -- the leap's own
                         overflow condition, evaluated in the log domain so it
                         stays checkable even where `Z**(1-beta)` itself would
                         already have rounded to `inf` (`[B,H,S]` bool).
    """
    num, mod = _smp_numerator(cap.q, cap.k, cap.u, cap.theta, qk=cap.qk, g=cap.g)
    del num
    z = mod.sum(-1)  # [B, H, S], real, same dtype as q/k/v
    #: THE SOFTMAX READ, beta=1.0, NOT beta=beta. `readout(beta)` already
    #: divides by `Z^beta`, so multiplying IT by `Z^(1-beta)` applies the
    #: exponent twice -- `Z^(2(1-beta)) * softmax_read`, which overstated this
    #: instrument's own `r` by 2.9x at beta=0.5 and 10.4x at beta=0.1 on the
    #: shipped config. The identity the leap is about is
    #: `readout(beta) == Z^(1-beta) * readout(1)`, so the gain multiplies the
    #: SOFTMAX read and `written` is then bitwise the tensor the block adds.
    read = _smp_readout(cap.q, cap.k, cap.v, cap.u, cap.theta,
                        beta=1.0, qk=cap.qk, g=cap.g).real  # [B, H, S, Dh]
    exponent = 1.0 - beta
    gain = z ** exponent  # matches operator()'s own `** beta`, exponent flipped
    written = read * gain.unsqueeze(-1)
    b, h, s, dh = written.shape
    written_full = written.transpose(1, 2).reshape(b, s, h * dh)
    r = written_full.norm(dim=-1) / cap.x_residual.norm(dim=-1)  # [B, S]
    log_z = torch.log(z.clamp_min(torch.finfo(z.dtype).tiny))
    crosses = (exponent * log_z) > _log_max(z.dtype)
    return {"z": z, "r": r, "crosses_log_max": crosses}


def bimodality_bic(r_positive_finite: torch.Tensor) -> dict:
    """One- vs two-component 1-D Gaussian-mixture BIC on `log10(r)`, over
    values the caller has already filtered to finite and positive.

    THE STATISTIC AND WHY THIS ONE. `r` spans many orders of magnitude by
    construction (the leap predicts a mass above `1e3` and a mass below `1`),
    so the fit is on `log10(r)`, not `r`. No dip-test package is installed in
    this environment (`import diptest` fails); `sklearn` (already a project
    dependency's own dependency: `pip show scikit-learn` reads `1.9.0` here)
    ships `GaussianMixture.bic`, which is the two-cluster-vs-one-cluster test
    the task names as the alternative to a dip test. `delta_bic_one_minus_two
    > 10` ("very strong" on the conventional Kass & Raftery scale) is
    NECESSARY but not sufficient for "bimodal" here: at n in the hundreds,
    BIC prefers 2 Gaussians over 1 on almost any merely-skewed unimodal
    distribution, because its `k * log(n)` penalty for the two extra
    parameters is cheap next to the log-likelihood gain from fitting one more
    tail -- a `delta_bic` of hundreds can come from two components 0.15
    log10-units apart (a factor of 1.4x), which is not what "mass below 1,
    mass above 1e3" (a >=3-decade split) means. So `bimodal` also requires
    the fitted means to be `> 1` decade apart (`log10_mean_separation`,
    reported alongside so a caller can re-threshold it), and it is a
    CONJUNCTION OF TWO THRESHOLDS ON NAMED STATISTICS, not an eyeball.
    """
    from sklearn.mixture import GaussianMixture

    x = r_positive_finite.detach().cpu().double().numpy()
    if x.size < 8:
        return {"statistic": "gmm_bic_on_log10_r", "n": int(x.size),
                "bimodal": None,
                "verdict": "fewer than 8 finite positive samples to fit"}
    x = torch.from_numpy(x).log10().numpy().reshape(-1, 1)
    gm1 = GaussianMixture(n_components=1, random_state=0).fit(x)
    gm2 = GaussianMixture(n_components=2, random_state=0).fit(x)
    bic1, bic2 = float(gm1.bic(x)), float(gm2.bic(x))
    order = gm2.means_.ravel().argsort()
    means_sorted = gm2.means_.ravel()[order]
    separation = float(means_sorted[-1] - means_sorted[0])
    delta_bic = bic1 - bic2
    return {
        "statistic": "gmm_bic_on_log10_r",
        "n": int(x.size),
        "bic_one_component": bic1,
        "bic_two_component": bic2,
        "log10_mean_separation": separation,
        "delta_bic_one_minus_two": bic1 - bic2,
        "two_component_means_log10_r": gm2.means_.ravel()[order].tolist(),
        "two_component_weights": gm2.weights_[order].tolist(),
        "bimodal": delta_bic > 10.0 and separation > 1.0,
    }


def layer_stats(r: torch.Tensor) -> dict:
    """min/median/max over finite entries, plus the two named fractions,
    plus `bimodality_bic` on the finite positive subset. `nan` (none expected
    at init; `Z` is strictly positive by `ceq/arm_smprime.py`'s own argument)
    is excluded from BOTH `frac_above_1e3` and `frac_below_1` -- IEEE
    comparisons against `nan` are `False` either way -- and counted only in
    `frac_non_finite`, so the three fractions do not sum to `1.0` were a `nan`
    ever present; that is reported rather than silently absorbed into one
    side."""
    r = r.reshape(-1)
    dtype_name = str(r.dtype).replace("torch.", "")
    finite = r[torch.isfinite(r)]
    positive_finite = finite[finite > 0]
    stats = {
        "n": int(r.numel()),
        "dtype": dtype_name,
        "frac_non_finite": float((~torch.isfinite(r)).float().mean()),
        "frac_above_1e3": float((r > 1e3).float().mean()),
        "frac_below_1": float((r < 1.0).float().mean()),
        "min": float(finite.min()) if finite.numel() else None,
        "median": float(finite.median()) if finite.numel() else None,
        "max": float(finite.max()) if finite.numel() else None,
    }
    stats["bimodality"] = bimodality_bic(positive_finite)
    return stats


def measure(*, seed: int = 0, batch: int = 4, seq: int = 64,
           betas: tuple[float, ...] | None = None,
           config: CEQConfig | None = None) -> dict:
    """The full RED: build the wired `smprime` arm at its shipped init (or at
    `config` if the caller passes a tiny one for a fast check), run ONE real
    forward pass, and report `layer_stats` per (beta, layer)."""
    if betas is None:
        betas = (SHIPPED_BETA,) + INTERIOR_BETAS
    torch.manual_seed(seed)
    if config is None:
        config = CEQConfig(operator="smprime")  # shipped defaults
    model = CEQModel(config).eval()
    ids = torch.randint(0, config.vocab_size, (batch, seq),
                        generator=torch.Generator().manual_seed(seed + 1))
    caps = capture_forward(model, ids)
    param_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    act_dtype = str(caps[0].q.dtype).replace("torch.", "") if caps else None

    checkpoint = find_smprime_checkpoint()
    report = {
        "n_layers": len(caps),
        "batch": batch, "seq": seq, "seed": seed,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "param_dtype": param_dtype,
        "activation_dtype": act_dtype,
        "checkpoint": checkpoint if checkpoint is not None else (
            "NONE FOUND -- searched every config.json under the repo for "
            "operator=='smprime'; every on-disk CEQ checkpoint is 'sgate'. "
            "Measured AT INIT: CEQModel(CEQConfig(operator='smprime')).eval() "
            "at the shipped corner (beta=qk=g=1.0), torch.manual_seed({})."
            .format(seed)),
        "log_max_reference": {"float32": _log_max(torch.float32),
                              "float64": _log_max(torch.float64)},
        "betas": {},
    }
    for beta in betas:
        per_layer = []
        for i, cap in enumerate(caps):
            out = residual_write(cap, beta)
            row = layer_stats(out["r"])
            row["layer"] = i
            row["frac_crosses_log_max"] = float(
                out["crosses_log_max"].float().mean())
            per_layer.append(row)
        report["betas"][beta] = per_layer
    return report


def main() -> None:
    print(json.dumps(measure(), indent=2, default=str))


if __name__ == "__main__":
    main()
