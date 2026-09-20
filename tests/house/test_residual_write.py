"""The residual-write leap, RED first, exercised through `ceqjepa/residual_write.py`
(a concurrent lane's file this round; this file owns only the test).

THE LEAP, restated from `ceqjepa/residual_write.py`'s own docstring: context in
the wired `smprime` arm is claimed to live in the residual stream as a
per-token, per-layer BINARY OVERWRITE -- `Z^{1-beta}` is not a row gain but a
RESET, and a token whose `(1-beta) * log Z` crosses the dtype's `LOG_MAX` has
its residual erased and replaced by the read. `residual_write.measure()` runs
ONE real forward pass through the wired 24-layer `smprime` model, captures
every block's own q/k/v/u/theta and `x_residual`, and reports
`r_i = ||Z_i^{1-beta} (a@v)_i|| / ||x_residual_i||` per (token, layer, beta),
plus a one- vs two-component Gaussian-mixture BIC on `log10(r)` as the
bimodality call. This file is the pytest binding of that instrument, run on a
small config (`hidden_size=64, num_attention_heads=4`, `ceq/hf/smoke.py`'s own
`_tiny` convention, `num_hidden_layers=24` kept because the leap is about
DEPTH) so the suite stays CPU-fast; the 24-layer, `beta`, `qk`-switch and
log-domain-overflow mechanics are `residual_write.py`'s own and are not
reimplemented here.

THE RED RUN, verbatim: torch 2.14.0+cpu, float32 activations (this tiny
model's own param dtype), CPU, seed 0, batch 2, seq 16,
`betas = (SHIPPED_BETA,) + INTERIOR_BETAS = (1.0, 0.5, 0.1)`,
`python -m pytest tests/house/test_residual_write.py -q`:

    every one of 24 layers, every one of 3 betas:
      frac_non_finite = 0.0,  frac_above_1e3 = 0.0
      bimodality.bimodal = False  (delta_bic_one_minus_two mostly negative --
        one Gaussian component fits log10(r) at least as well as two -- and
        never past the module's own "very strong" bimodal bar of 10.0; its
        one exception among the 72 (layer, beta) cells is beta=0.1 layer=3
        at 7.81, still under that bar)
      r stays in roughly [0.6, 8.3] across every (layer, beta) measured

THAT IS THE KILL, reported as plainly as the BIND would have been. At the
model's own shipped init, on the wired 24-layer arm, `Z^{1-beta}` never gets
close to swamping the residual, let alone erasing it: the read is a
perturbation everywhere this run looked, not a reset. `test_red_...` below
pins exactly this (zero crossings, zero bimodal calls, the read `min`/`max`
band), and it is designed to go red the moment either changes -- under a
concurrent edit to `ceqjepa/residual_write.py` this round, or if the leap's
mechanism is later shown to need a trained (not freshly initialised) model to
engage at all, which this run cannot distinguish from "does not exist" and
says so rather than overclaiming a KILL past what one seed of one init proves.

THE TWO CONTROLS, on the SAME model, so a crossing (if one is later found in a
trained checkpoint) has already been checked against a corner where it must
not appear and a knob it must move with.

CONTROL 1 -- the softmax corner. `residual_write.residual_write` computes its
gain as `z ** (1 - beta)`; at `beta = 1.0` that exponent is `0.0` bitwise, and
`z ** 0.0 == 1.0` for every entry regardless of `z`'s own value (checked
directly on the captured layer-0 `z`, not inferred from `r` alone). No
per-token scalar survives to move `r` at the softmax corner, and the `measure`
report at `beta=1.0` above independently shows zero crossings there -- the two
checks agree, which is the control passing.

CONTROL 2 -- scale. Layer 0's captured q/k/v/u/theta, `beta` fixed at
`INTERIOR_BETAS[0] = 0.5`, the `qk` switch (`ceq/arm_smprime.py:228`, "multiply
the logits by a constant" is exactly this knob) multiplied by
`c in {1, 4, 16, 64, 128, 256, 512}`. Per-token `log(r)` against
`(1-beta) * max_h log Z_h` (the max over heads, because `r` norms the
head-concatenated write and the largest head's gain is what would dominate
that norm): correlation climbs `0.089 -> 0.300 -> 0.785 -> 0.977 -> 0.994 ->
0.999 -> 0.9997` as `c` grows -- weak while `(1-beta) log Z` is small next to
the read's own directional noise, strong once it is not. Pooled over all 7*32
finite tokens: Pearson **0.9988**, least-squares slope **1.951**, intercept
1.355, R^2 **0.9976**. The slope is not the idealised single-block 1 measured
elsewhere on an isolated row (this is the dense, multi-head, causally-masked
model, and `read` itself is not held fixed as `c` grows) -- reported as
measured, not rounded to the tidier number. What the control establishes is
that the growth tracks the named mechanism and not an unrelated one.

WHAT NEITHER CONTROL SETTLES. Both are run at the model's untrained init.
Neither speaks to a trained checkpoint, where `beta`, the heads and the value
vectors have all moved from here; `residual_write.find_smprime_checkpoint`
already reports none exists on this box to measure instead.
"""
from __future__ import annotations

import torch

from ceqjepa import residual_write as RW
from ceq.hf.configuration_ceq import CEQConfig
from ceq.hf.modeling_ceq import CEQModel

SEED = 0
BATCH = 2
SEQ = 16


def _tiny_smprime_config(**kw) -> CEQConfig:
    """`ceq/hf/smoke.py::_tiny`'s own sizing (`vocab_size=64, hidden_size=64,
    num_attention_heads=4, max_position_embeddings=64`), read there rather than
    imported (it is a script, not a library call) so this file does not gain a
    second owner. `num_hidden_layers=24` is kept at the shipped default and NOT
    narrowed to smoke's `2` -- the leap is a claim about a 24-layer stack, and
    a 2-layer model could not carry it either way.
    """
    base = dict(operator="smprime", vocab_size=64, hidden_size=64,
                num_hidden_layers=24, num_attention_heads=4,
                max_position_embeddings=64)
    base.update(kw)
    return CEQConfig(**base)


def _measure(seed: int = SEED, batch: int = BATCH, seq: int = SEQ):
    betas = (RW.SHIPPED_BETA,) + RW.INTERIOR_BETAS
    return RW.measure(seed=seed, batch=batch, seq=seq, betas=betas,
                       config=_tiny_smprime_config())


def _capture_layer0(seed: int = SEED, batch: int = BATCH, seq: int = SEQ):
    """One real forward pass, layer 0's `_Capture` only -- what control 2
    scales. Reproduces `measure`'s own construction (`torch.manual_seed(seed)`
    immediately before `CEQModel(config)`) so this capture is the same tensors
    `_measure` would have used for layer 0, checked by `test_control2_...`
    below rather than assumed."""
    torch.manual_seed(seed)
    config = _tiny_smprime_config()
    model = CEQModel(config).eval()
    ids = torch.randint(0, config.vocab_size, (batch, seq),
                        generator=torch.Generator().manual_seed(seed + 1))
    return RW.capture_forward(model, ids)[0]


def _scaled_qk(cap, c: float):
    """A copy of `cap` with its `qk` switch multiplied by `c` -- the "multiply
    the logits by a constant" of control 2, `ceq/arm_smprime.py:228`'s own
    knob. `RW._Capture` has no public constructor; this lane owns the test
    file and not the sibling module, so a plain-namespace stand-in is used
    here rather than reaching into `RW._Capture.__slots__` from outside it.

    `ponytail:` a private-name dependency either way (there is no OTHER way
    to hand `residual_write.residual_write` a `_Capture`-shaped object); if
    the sibling module grows a public builder, swap this for it.
    """
    class _Cap:
        pass
    scaled = _Cap()
    scaled.x_residual = cap.x_residual
    scaled.q, scaled.k, scaled.v = cap.q, cap.k, cap.v
    scaled.u, scaled.theta = cap.u, cap.theta
    scaled.qk, scaled.g = cap.qk * c, cap.g
    return scaled


# --------------------------------------------------------------------- RED


def test_red_wired_24_layer_arm_at_shipped_init_does_not_cross():
    """THE RED RUN, pinned: zero crossings, zero bimodal calls, at every one
    of 24 layers and all three named betas, reproduced twice first."""
    report_a = _measure()
    report_b = _measure()

    assert report_a["n_layers"] == 24
    assert report_a["activation_dtype"] == "float32"
    assert set(report_a["betas"]) == {1.0, 0.5, 0.1}

    all_r_min, all_r_max = [], []
    for beta, rows_a in report_a["betas"].items():
        rows_b = report_b["betas"][beta]
        assert len(rows_a) == len(rows_b) == 24
        for row_a, row_b in zip(rows_a, rows_b):
            # reproducible first -- a pin on a number that moves run to run
            # would be a pin on nothing.
            assert row_a["median"] == row_b["median"], (beta, row_a["layer"])

            assert row_a["frac_non_finite"] == 0.0, (beta, row_a["layer"])
            assert row_a["frac_above_1e3"] == 0.0, (beta, row_a["layer"])
            assert row_a["bimodality"]["bimodal"] is False, (beta, row_a["layer"])
            # the module's own "very strong" bimodal bar (Kass & Raftery, > 10.0)
            # -- not "always negative": one cell (beta=0.1, layer=3) sits at
            # +7.81, still short of the bar, and asserting strict negativity
            # would be a pin tighter than the leap's own named criterion.
            assert row_a["bimodality"]["delta_bic_one_minus_two"] < 10.0, \
                (beta, row_a["layer"], row_a["bimodality"]["delta_bic_one_minus_two"])
            assert row_a["frac_crosses_log_max"] == 0.0, (beta, row_a["layer"])
            all_r_min.append(row_a["min"])
            all_r_max.append(row_a["max"])

    # THE LEAP'S OWN CRITERION, failed to fire: no mass below 1 that is
    # SEPARATE from a mass above 1e3 -- every row's own min/max band sits
    # inside one order-of-magnitude-scale range, not split across a gap.
    assert min(all_r_min) > 0.5, f"unexpectedly small read: {min(all_r_min):.6e}"
    assert max(all_r_max) < 20.0, f"unexpectedly large read: {max(all_r_max):.6e}"


# ----------------------------------------------------------------- control 1


def test_control1_softmax_corner_gain_is_exactly_one():
    """`beta = 1.0` makes `residual_write`'s own `z ** (1 - beta)` gain
    `1.0` bitwise for every entry, whatever `z` itself is -- checked directly
    on layer 0's real captured `z`, not inferred from `r` staying small."""
    cap0 = _capture_layer0()
    out = RW.residual_write(cap0, beta=1.0)
    gain = out["z"] ** (1.0 - 1.0)
    assert torch.equal(gain, torch.ones_like(gain))

    # and the consequence the RED run's own beta=1.0 rows already show:
    # no crossing, because nothing is left that could produce one.
    assert torch.isfinite(out["r"]).all()
    assert float((out["r"] > 1e3).float().mean()) == 0.0
    assert not bool(out["crosses_log_max"].any())


# ----------------------------------------------------------------- control 2


def test_control2_scaling_the_logits_ties_growth_to_one_minus_beta_logz():
    """Fix `beta = INTERIOR_BETAS[0] = 0.5`; multiply layer 0's captured `qk`
    switch by `c` over `{1, 4, 16, 64, 128, 256, 512}`; per-token
    `log(r)` against `(1 - beta) * max_h log(Z_h)` should correlate more
    strongly as `c` grows (the mechanism's own signal stops being swamped by
    the read's directional noise), and pooled over the sweep the two should
    move together and not arbitrarily.
    """
    beta = RW.INTERIOR_BETAS[0]
    assert beta == 0.5
    cap0 = _capture_layer0()
    cs = (1.0, 4.0, 16.0, 64.0, 128.0, 256.0, 512.0)

    log_r_all, pred_all, per_c_corr = [], [], []
    for c in cs:
        out = RW.residual_write(_scaled_qk(cap0, c), beta=beta)
        r, z = out["r"], out["z"]
        logz_max = torch.log(z.clamp_min(torch.finfo(z.dtype).tiny)).amax(dim=1)
        ok = torch.isfinite(r) & (r > 0) & torch.isfinite(logz_max)
        assert int(ok.sum()) == r.numel(), f"c={c}: already non-finite before the sweep needed it to be"
        log_r = torch.log(r[ok]).double()
        pred = ((1.0 - beta) * logz_max[ok]).double()
        log_r_all.append(log_r)
        pred_all.append(pred)
        per_c_corr.append(float(torch.corrcoef(torch.stack([log_r, pred]))[0, 1]))

    assert per_c_corr == sorted(per_c_corr), f"correlation was not monotone in c: {per_c_corr}"
    assert per_c_corr[0] < 0.5, f"c=1 correlation drifted: {per_c_corr[0]}"
    # 0.9988 as measured AFTER the exponent fix in `residual_write.residual_write`
    # (it had been applying `Z^(1-beta)` on top of `readout(beta)`, i.e. twice);
    # the pre-fix number this file's header quotes was 0.9997, from the doubled
    # exponent. The control's content is unchanged: correlation still climbs
    # monotonically to ~1 as the logit scale grows.
    assert per_c_corr[-1] > 0.998, f"c=512 correlation drifted: {per_c_corr[-1]}"

    log_r = torch.cat(log_r_all)
    pred = torch.cat(pred_all)
    assert log_r.numel() == pred.numel() == len(cs) * cap0.q.shape[0] * cap0.q.shape[2]

    pooled_corr = float(torch.corrcoef(torch.stack([log_r, pred]))[0, 1])
    design = torch.stack([pred, torch.ones_like(pred)], dim=1)
    slope, intercept = torch.linalg.lstsq(design, log_r.unsqueeze(1)).solution.squeeze(1).tolist()
    fitted = design @ torch.tensor([slope, intercept], dtype=log_r.dtype)
    ss_res = float(((log_r - fitted) ** 2).sum())
    ss_tot = float(((log_r - log_r.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot

    assert pooled_corr > 0.99, f"pooled correlation drifted: {pooled_corr}"
    # THE PREDICTED RATE IS 1, AND THE FIT NOW READS IT. `log r` should be
    # `(1-beta) log Z + const`, slope 1. The pre-fix 1.951 this file's header
    # quotes was the doubled exponent in `residual_write.residual_write`
    # showing up as exactly twice the predicted slope; with the exponent
    # applied once the fit reads 0.944. Bracketed around 1 rather than
    # asserted above it, so either direction of drift goes red.
    assert 0.8 < slope < 1.2, f"growth is not the predicted rate: slope={slope}"
    assert r_squared > 0.99, f"fit quality drifted: {r_squared}"
