"""R1 -- clamp(u,0,1) has zero gradient outside (0,1): straight-through vs
hard-concrete as drop-in replacements for ceq/arm_smprime.py::magnitude.

ceq/ is NEVER edited. Both new forms are installed as a MONKEYPATCH of the
module-global name `ceq.arm_smprime.magnitude` -- exactly the technique
tests/foreman/gated/gate_init_repair.py already uses on `ceq.hf.train.build`,
for the identical reason: `blend()` looks `magnitude` up as a module global at
call time, so replacing the name on the module object changes what every
caller (`blend` -> `hop` -> `numerator`/`operator`/`readout` -> the HF
attention block) executes, with no source file touched.

Run stages, each gated on the previous one:
  (1) reproduce_f1()        -- numpy only, no torch model. MUST pass first.
  (2) verify_ste_forward()  -- straight-through forward == torch.clamp bitwise.
  (3) spurious_zero_selftest() -- the R7 flag+cumsum check, on a synthetic
      case with a known zero count, before it is trusted on real gate values.
  (4) run_variant(form) x2  -- train at the reference config, reusing
      tests/foreman/gated/gate_init_repair.py's T.train()/T.build() plumbing.
  (5) pairing_check()       -- detrended per-step loss correlation vs shuffled
      control, same seed/data/shape/steps for both variants.
  (6) must_fire() + the row + the kill.

Command: python r1_gate.py
"""
import io
import json
import math
import os
import sys
import time

import numpy as np

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
RESULTS_PATH = os.path.join(SCRATCH, "r1_results.jsonl")
sys.path.insert(0, REPO)


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Chase",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", row)


def result(**kw):
    with io.open(RESULTS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(kw, default=str) + "\n")


# ============================================================ (1) REPRODUCE F1

def reproduce_f1():
    """default_rng(0), d=512, n=20000, LayerNorm-normalised rows,
    w ~ N(0, 1/d), float64. u = layernorm(x) @ w + bias, magnitude = clamp(u,0,1).

    Target (from the task, itself measured off the author's own instance):
      bias 0     -> P(u<=0)=0.501, P(u>=1)=0.160, grad-carrying=33.8%
      bias 0.999 -> P(u<=0)=0.159ish, P(u>=1)=0.498, grad-carrying=34.3%
      hard-concrete (zeta=1.1, gamma=-0.1 stretch) at bias 0: exact-0=0.9%,
      exact-1=0.9%.
    """
    rng = np.random.default_rng(0)
    d, n = 512, 20000
    x = rng.standard_normal((n, d)).astype(np.float64)
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)  # ddof=0, LayerNorm's own biased estimator
    xn = (x - mean) / np.sqrt(var)      # no affine, no eps (var>0 a.s. at d=512)
    w = rng.standard_normal(d).astype(np.float64) / math.sqrt(d)  # N(0, 1/d)

    def cell(bias):
        u = xn @ w + bias
        le0 = float((u <= 0.0).mean())
        ge1 = float((u >= 1.0).mean())
        grad = 1.0 - le0 - ge1
        return dict(bias=bias, p_le0=le0, p_ge1=ge1, frac_grad=grad)

    c0 = cell(0.0)
    c1 = cell(0.999)

    zeta, gamma = 1.1, -0.1  # Louizos, Welling, Kingma 2018, stretch = 1.2
    u0 = xn @ w + 0.0
    s = 1.0 / (1.0 + np.exp(-u0))
    sbar = s * (zeta - gamma) + gamma
    hc_exact0 = float((sbar <= 0.0).mean())
    hc_exact1 = float((sbar >= 1.0).mean())

    out = dict(
        d=d, n=n, seed=0,
        bias0=c0, bias999=c1,
        hard_concrete_bias0=dict(exact0=hc_exact0, exact1=hc_exact1,
                                  zeta=zeta, gamma=gamma, stretch=zeta - gamma),
    )

    def close(a, b, tol=0.01):
        return abs(a - b) <= tol

    checks = dict(
        bias0_le0=close(c0["p_le0"], 0.501),
        bias0_ge1=close(c0["p_ge1"], 0.160),
        bias0_grad=close(c0["frac_grad"], 0.338),
        bias999_ge1=close(c1["p_ge1"], 0.498),
        bias999_grad=close(c1["frac_grad"], 0.343),
        hc_exact0=close(hc_exact0, 0.009, tol=0.01),
        hc_exact1=close(hc_exact1, 0.009, tol=0.01),
    )
    out["checks"] = checks
    out["all_pass"] = all(checks.values())
    return out


# =================================================== (2) THE TWO PARAMETRIZATIONS

import torch  # noqa: E402  (after sys.path insert, matches gate_init_repair.py)
import ceq.arm_smprime as arm_smprime  # noqa: E402

_ORIG_MAGNITUDE = arm_smprime.magnitude
ZETA, GAMMA = 1.1, -0.1  # hard-concrete stretch constants, Louizos et al. 2018


class _STEClamp(torch.autograd.Function):
    """Straight-through (Bengio, Leonard, Bengio 2013): forward is the exact
    clamp(u,0,1); backward is the identity, no zeroing outside (0,1)."""

    @staticmethod
    def forward(ctx, u):
        return torch.clamp(u, 0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def magnitude_ste(u: torch.Tensor) -> torch.Tensor:
    return _STEClamp.apply(u)


def magnitude_hardconcrete(u: torch.Tensor) -> torch.Tensor:
    """Louizos, Welling, Kingma 2018 (L0 regularisation): stretch the sigmoid
    to (gamma, zeta) = (-0.1, 1.1) then clamp to [0,1]. No stochastic gate
    noise term -- this instantiates the deterministic/eval-mode transform,
    matching the F1 instance measured above; ordinary autograd clamp backward
    (rare exact endpoints under the stretch, unlike bare clamp(u,0,1))."""
    s = torch.sigmoid(u)
    sbar = s * (ZETA - GAMMA) + GAMMA
    return torch.clamp(sbar, 0.0, 1.0)


FORMS = {"straight_through": magnitude_ste, "hard_concrete": magnitude_hardconcrete}


def verify_ste_forward():
    """Forward equality, bitwise, against torch.clamp -- a straight-through
    that changes the forward is a different model, not a different gradient."""
    g = torch.Generator().manual_seed(0)
    u = torch.randn(4096, generator=g, dtype=torch.float64) * 3.0
    a = magnitude_ste(u)
    b = torch.clamp(u, 0.0, 1.0)
    exact = torch.equal(a, b)
    return dict(bitwise_equal=exact, n=u.numel())


# ===================================================== (3) R7 SPURIOUS ZEROS

def spurious_zero_count(m: torch.Tensor) -> dict:
    """`m`: [..., S] magnitudes in [0,1]. Ground truth for "the window
    (j, i] contains a true zero" is flag+cumsum (Blelloch 1989 segmented
    scan): flag_k = (m_k == 0), cum_k = cumsum(flag), window (j,i] has a zero
    iff cum_i > cum_j. Compare against the float path_product's own zero
    reads (R_ij == 0 via `arm_smprime.path_product`). A float-zero where the
    flag+cumsum says no true zero exists is a SPURIOUS zero -- underflow, not
    an actual closed gate."""
    s = m.shape[-1]
    flag = (m == 0.0).to(torch.float64)                # [..., S]
    cum = torch.cumsum(flag, dim=-1)                    # cum[k] = #zeros in m[0..k]
    idx = torch.arange(s, device=m.device)
    le = (idx.unsqueeze(1) <= idx.unsqueeze(0))          # le[j, i] : j <= i  (row j, col i)
    # window for pair (i, j<=i) is k in (j, i] i.e. cum[i] - cum[j]
    cum_i = cum.unsqueeze(-2).expand(*cum.shape[:-1], s, s)          # [..., j, i] broadcasting i along last dim
    cum_j = cum.unsqueeze(-1).expand(*cum.shape[:-1], s, s)          # [..., j, i] j along second-to-last
    window_zero_count = cum_i - cum_j                    # [..., j, i] = cum[i]-cum[j]
    causal = le.expand_as(window_zero_count)              # j<=i valid region
    true_zero = (window_zero_count > 0) & causal
    R = arm_smprime.path_product(m.to(arm_smprime.DTYPE))       # [..., S, S] float64, R[..., i, j]
    R = R.transpose(-1, -2)                               # -> [..., j, i] to match true_zero's axis order
    float_zero = (R == 0.0) & causal
    spurious = float_zero & (~true_zero)
    return dict(n_pairs=int(causal.sum()), n_float_zero=int(float_zero.sum()),
                n_true_zero=int(true_zero.sum()),
                n_spurious=int(spurious.sum()))


def spurious_zero_selftest():
    """Synthetic case with a KNOWN zero count: S=16, m all 0.7 except one
    true zero at position 5. flag+cumsum must find exactly the windows that
    cross position 5 as true zeros, and (since 0.7 to any power up to 16
    never underflows float64) the float product must read zero on exactly
    those and nowhere else -- n_spurious must be 0."""
    m = torch.full((16,), 0.7, dtype=torch.float64)
    m[5] = 0.0
    out = spurious_zero_count(m)
    # windows (j,i] that contain k=5: j<5<=i, j in [0,4], i in [5,15] -> 5*11=55
    expected_true = 5 * 11
    ok = (out["n_spurious"] == 0) and (out["n_true_zero"] == expected_true)
    out["expected_true_zero"] = expected_true
    out["self_test_pass"] = ok
    return out


def gradient_frac_nonzero(u_raw: torch.Tensor, form: str) -> float:
    """Fraction of gate positions where d(magnitude)/d(u) != 0, computed by
    vjp with an all-ones cotangent (m is elementwise in u, so this recovers
    dm/du exactly per position, cheaper and more precise than backprop
    through the whole model). `u_raw` is the REAL m_head pre-activation
    captured off the model's own forward pass (gate_values' hook), not a
    freshly re-run linear layer -- the embedding pipeline needs input_ids,
    not a hidden-state tensor, to reach that layer honestly."""
    u = u_raw.detach().clone().requires_grad_(True)
    m = FORMS[form](u)
    grad, = torch.autograd.grad(m.sum(), u, retain_graph=False)
    return float((grad != 0).to(torch.float64).mean())


# =========================================== (4) TRAIN AT THE REFERENCE CONFIG

import ceq.hf.train as T  # noqa: E402
from ceq.hf.modeling_ceq import CEQForCausalLM  # noqa: E402
from ceq.arm_smprime import GATE_INIT_OFF  # noqa: E402

OUT_DIR_TMPL = os.path.join(SCRATCH, "r1_ckpt_{form}")
HIDDEN, LAYERS, HEADS, SEQ, BATCH = 512, 4, 8, 512, 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
STEPS, SAVE_EVERY = 800, 100

REFERENCE = dict(  # repaired-clamp, quoted from the task text verbatim
    loss_last=1.3122, last50_mean=1.2665, last50_std=0.0399,
    trained_exact_zero=0.3121, trained_max_backward_run=42,
    backward_reach_init=dict(mean=0.672, median=0.0, p95=3.0, max=10.0),
    backward_reach_trained=dict(mean=2.146, median=1.0, p95=7.0, max=42.0),
)

_ORIG_BUILD = T.build


def build_repaired(*, hidden_size, n_layers, n_heads, seq, vocab_size=256, **overrides):
    """Byte-identical to gate_init_repair.py::repaired_build -- same init
    (m_head.bias=1-off, theta_head.bias=off), so the ONLY variable this
    script changes relative to that reference run is `arm_smprime.magnitude`
    itself, via the module-global monkeypatch installed by run_variant()."""
    model = _ORIG_BUILD(hidden_size=hidden_size, n_layers=n_layers,
                        n_heads=n_heads, seq=seq, vocab_size=vocab_size, **overrides)
    off = GATE_INIT_OFF
    with torch.no_grad():
        for layer in model.model.layers:
            attn = layer.self_attn
            attn.m_head.bias.fill_(1.0 - off)
            attn.theta_head.bias.fill_(off)
    return model


@torch.no_grad()
def gate_values(model, x_ids):
    """Returns (m_layers, u_raw_layers): the realised magnitude AND the raw
    m_head pre-activation it came from, both per-layer [B,S] off one real
    forward pass -- u_raw is what gradient_frac_nonzero needs, read from the
    model's own hidden states rather than re-run on input_ids."""
    model.eval()
    captured_m, captured_u = [], []

    def make_hook():
        def hook(mod, inputs):
            x = inputs[0]
            u_raw = mod.m_head(x).squeeze(-1)
            th_raw = mod.theta_head(x).squeeze(-1)
            m, th = arm_smprime.blend(u_raw, th_raw, mod.g)
            captured_m.append(m.detach().float().cpu())
            captured_u.append(u_raw.detach().float().cpu())
        return hook

    handles = [layer.self_attn.register_forward_pre_hook(make_hook())
               for layer in model.model.layers]
    model(input_ids=x_ids)
    for h in handles:
        h.remove()
    return captured_m, captured_u


def backward_reach_stats(m_layers):
    """BACKWARD reach: for row i, the length of the consecutive nonzero run
    of m ENDING at i (not the maximal run through i in either direction --
    G_ij is prod_{k=j+1}^i m_k, so live reach for row i only ever extends
    backward from i)."""
    reach = []
    for m in m_layers:  # [B, S]
        nz = (m > 0).numpy()
        B, S = nz.shape
        for b in range(B):
            row = nz[b]
            run = 0
            for i in range(S):
                run = run + 1 if row[i] else 0
                reach.append(run)
    t = torch.tensor(reach, dtype=torch.float64)
    qs = torch.tensor([0.5, 0.95], dtype=torch.float64)
    med, p95 = torch.quantile(t, qs).tolist()
    return dict(n=t.numel(), mean=float(t.mean()), median=med, p95=p95, max=float(t.max()))


def exact_zero_frac(m_layers):
    allm = torch.cat([m.reshape(-1) for m in m_layers])
    return float((allm == 0.0).to(torch.float64).mean())


def run_variant(form: str):
    board("start_variant", task="r1_gate", form=form, shape=dict(
        hidden=HIDDEN, layers=LAYERS, heads=HEADS, seq=SEQ, batch=BATCH),
        device=DEVICE, seed=SEED, steps=STEPS)

    arm_smprime.magnitude = FORMS[form]  # THE monkeypatch under test
    try:
        data = T.ByteBatches(T._corpus_text(None, 64 * 1024 * 1024), vocab_size=256)
        eval_gen = torch.Generator().manual_seed(12345)
        x_eval, _ = data.batch("val", BATCH, SEQ, eval_gen, DEVICE)

        torch.manual_seed(SEED)
        model0 = build_repaired(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                                seq=SEQ, vocab_size=256, operator="smprime").to(DEVICE)
        m_init, u_init = gate_values(model0, x_eval)
        frac_grad_init = gradient_frac_nonzero(u_init[0], form)
        spur_init = spurious_zero_count(m_init[0][0])
        reach_init = backward_reach_stats(m_init)
        ez_init = exact_zero_frac(m_init)
        bias_m_init = float(model0.model.layers[0].self_attn.m_head.bias[0])
        del model0
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        board("gate_at_init", form=form, frac_grad_nonzero=frac_grad_init,
              exact_zero=ez_init, backward_reach=reach_init, spurious=spur_init)

        out_dir = OUT_DIR_TMPL.format(form=form)
        T.build = build_repaired
        try:
            t0 = time.time()
            record = T.train(out_dir=out_dir, steps=STEPS, batch=BATCH, seq=SEQ,
                             hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                             device=DEVICE, vocab_size=256, data_path=None,
                             max_bytes=64 * 1024 * 1024, lr=3e-4, seed=SEED,
                             grad_checkpoint=False, log_every=20,
                             save_every=SAVE_EVERY, operator="smprime")
            train_s = time.time() - t0
        finally:
            T.build = _ORIG_BUILD

        model1 = CEQForCausalLM.from_pretrained(out_dir).to(DEVICE)
        m_trained, u_trained = gate_values(model1, x_eval)
        frac_grad_trained = gradient_frac_nonzero(u_trained[0], form)
        spur_trained = spurious_zero_count(m_trained[0][0])
        reach_trained = backward_reach_stats(m_trained)
        ez_trained = exact_zero_frac(m_trained)
        bias_m_trained = float(model1.model.layers[0].self_attn.m_head.bias[0])

        losses = record["losses"]
        last50 = losses[-50:]
        last50_mean = float(np.mean(last50))
        last50_std = float(np.std(last50))

        board("gate_after_training", form=form, frac_grad_nonzero=frac_grad_trained,
              exact_zero=ez_trained, backward_reach=reach_trained, spurious=spur_trained,
              loss_last=losses[-1], last50_mean=last50_mean, last50_std=last50_std)

        out = dict(
            form=form, device=DEVICE, seed=SEED, steps=STEPS,
            dtype="float32", shape=dict(hidden=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                                         d_head=HIDDEN // HEADS, seq=SEQ, batch=BATCH),
            command="python r1_gate.py",
            bias_m_head_init=bias_m_init, bias_m_head_trained=bias_m_trained,
            bias_move=abs(bias_m_trained - bias_m_init),
            frac_grad_nonzero_init=frac_grad_init, frac_grad_nonzero_trained=frac_grad_trained,
            exact_zero_init=ez_init, exact_zero_trained=ez_trained,
            backward_reach_init=reach_init, backward_reach_trained=reach_trained,
            spurious_zero_init=spur_init["n_spurious"], spurious_zero_trained=spur_trained["n_spurious"],
            train_seconds=train_s, loss_first=losses[0], loss_last=losses[-1],
            last50_mean=last50_mean, last50_std=last50_std,
            n_params=record["n_params"], out_dir=out_dir, loss_curve=losses,
        )
        return out
    finally:
        arm_smprime.magnitude = _ORIG_MAGNITUDE  # restore, whether or not train() raised


# ===================================================== (5) PAIRING + (6) VERDICT

def detrended_corr(a, b):
    """Per-step loss correlation, DETRENDED (first-difference removes the
    shared downward trend both curves share from optimisation itself, which
    inflates raw correlation regardless of pairing)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    da, db = np.diff(a[:n]), np.diff(b[:n])
    if da.std() == 0 or db.std() == 0:
        return 0.0
    return float(np.corrcoef(da, db)[0, 1])


def pairing_check(losses_a, losses_b, seed=0):
    real = detrended_corr(losses_a, losses_b)
    rng = np.random.default_rng(seed)
    shuffled = np.array(losses_b, dtype=np.float64).copy()
    rng.shuffle(shuffled)
    control = detrended_corr(losses_a, shuffled)
    return dict(real=real, shuffled_control=control)


def must_fire(res: dict) -> dict:
    checks = dict(
        i_grad_frac_init=res["frac_grad_nonzero_init"] >= 0.95,
        i_grad_frac_trained=res["frac_grad_nonzero_trained"] >= 0.95,
        ii_bias_moves=res["bias_move"] >= 0.05,
        iii_exact_zero_init=res["exact_zero_init"] < 0.05,
    )
    checks["passes"] = all(checks.values())
    checks["iv_trained_exact_zero_fraction"] = res["exact_zero_trained"]  # reported, no threshold
    return checks


def main():
    t0 = time.time()
    board("start", task="r1_gate", note="R1: clamp gradient death, "
          "straight-through vs hard-concrete")

    f1 = reproduce_f1()
    result(stage="f1", **f1)
    board("f1", **{k: v for k, v in f1.items() if k != "checks"}, checks=f1["checks"])
    if not f1["all_pass"]:
        board("VOID", reason="F1 did not reproduce; the whole row rests on "
              "this diagnosis, stopping before any model is touched",
              checks=f1["checks"])
        print("F1 FAILED TO REPRODUCE -- stopping.", f1["checks"])
        return

    ver = verify_ste_forward()
    result(stage="ste_forward_equality", **ver)
    board("ste_forward_equality", **ver)
    if not ver["bitwise_equal"]:
        board("VOID", reason="straight-through forward != clamp bitwise; "
              "that is a different model, not a different gradient")
        print("STE FORWARD MISMATCH -- stopping.", ver)
        return

    selftest = spurious_zero_selftest()
    result(stage="spurious_zero_selftest", **selftest)
    board("spurious_zero_selftest", **selftest)
    if not selftest["self_test_pass"]:
        board("VOID", reason="spurious-zero self-test failed on a known "
              "case; the R7 instrument is not trustworthy on real gates")
        print("SPURIOUS-ZERO SELF-TEST FAILED -- stopping.", selftest)
        return

    board("checks_that_cannot_fail", note=(
        "gate_init_repair.py verify_repair uses frac_zero_exact<0.05 against "
        "a bf16/fp32-scale tolerance appropriate to float32 gate reads -- "
        "matches on-disk precision. The spurious-zero self-test above is "
        "constructed so a broken flag+cumsum (e.g. off-by-one window) FAILS "
        "it (expected_true_zero=55, not 0 or all-pairs), so it is not a "
        "check that cannot fail. The pairing control below (shuffled) is "
        "the one instrument whose whole job is to read near-zero -- it is "
        "reported beside the real correlation, not alone, so a control that "
        "\"passed\" (near 0) cannot be mistaken for a positive result."))

    results = {}
    for form in ("straight_through", "hard_concrete"):
        res = run_variant(form)
        results[form] = res
        result(stage="variant", **{k: v for k, v in res.items() if k != "loss_curve"})

    pair = pairing_check(results["straight_through"]["loss_curve"],
                         results["hard_concrete"]["loss_curve"])
    result(stage="pairing", **pair)
    board("pairing", **pair)

    mf = {form: must_fire(results[form]) for form in results}
    for form, m in mf.items():
        result(stage="must_fire", form=form, **m)
        board("must_fire", form=form, **m)

    both_fail = not any(m["passes"] for m in mf.values())
    winner = None
    if not both_fail:
        candidates = [f for f, m in mf.items() if m["passes"]]
        winner = min(candidates, key=lambda f: results[f]["loss_last"])

    verdict = dict(
        both_fail_must_fire=both_fail,
        winner=winner,
        kill_fired=both_fail,
        must_fire=mf,
        reference=REFERENCE,
        results={f: {k: v for k, v in r.items() if k != "loss_curve"}
                for f, r in results.items()},
        pairing=pair,
        wall_clock_total_s=time.time() - t0,
    )
    result(stage="verdict", **verdict)
    board("done", **{k: v for k, v in verdict.items()
                    if k not in ("must_fire", "results")})

    with open(os.path.join(SCRATCH, "r1_verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    print("VERDICT", json.dumps({k: v for k, v in verdict.items()
                                if k not in ("results",)}, indent=2, default=str))


if __name__ == "__main__":
    main()
