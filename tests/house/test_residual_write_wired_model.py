"""Self-check for `ceqjepa/residual_write.py`, the RED instrument for THE LEAP
(see that module's docstring). This file checks the INSTRUMENT, not the
hypothesis: whether `capture_forward` pulls one record per block off the real
wired `ceq.hf.modeling_ceq.CEQModel`, whether `residual_write` reduces to the
already-proved row-gain identity at beta=1
(`tests/curvature/test_beta_axis_is_a_row_gain.py`'s own finding, checked here
on the wired model's captured tensors instead of an isolated draw), whether
`layer_stats`'s two named fractions mean what they say, and whether
`bimodality_bic` can actually tell a two-cluster draw from a one-cluster draw
-- a statistic that always reports "unimodal" would pass the real measurement
silently and be exactly as uninformative as no statistic.

NAMED `..._wired_model.py` AND NOT `test_residual_write.py`. A second file
already occupies that exact path when this one was written -- a
`ceq.arm_smprime.block_summary`/`read_summary` stress test, self-described as
deliberately NOT importing this lane's `ceqjepa/residual_write.py` to avoid a
write race. Both files test the same leap; they are not the same check --
that one exercises the log-domain SURVIVOR merge
(`block_summary`+`read_summary`, `ceq/arm_smprime.py:285-367`) directly,
adversarially scaled, while this one exercises the DENSE `operator`/`readout`
path that `ceq/hf/modeling_ceq.py:478` actually calls from the wired
24-layer model, at its shipped init. Overwriting either would repeat the
exact collision the launching task named ("a concurrent round already lost a
producer that way"), so this file keeps its own name instead.

TINY and CPU throughout, so this runs in under a second; the real measurement
at the shipped 1280/24/20 config is `ceqjepa/residual_write.py::main`, not
this file.
"""
from __future__ import annotations

import torch

from ceq.arm_smprime import readout as smp_readout
from ceqjepa.residual_write import (
    CEQConfig,
    CEQModel,
    bimodality_bic,
    capture_forward,
    layer_stats,
    measure,
    residual_write,
)

TINY = dict(vocab_size=32, hidden_size=32, num_hidden_layers=3,
            num_attention_heads=4, max_position_embeddings=16,
            operator="smprime")


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = CEQConfig(**TINY)
    return cfg, CEQModel(cfg).eval()


def test_capture_forward_returns_one_record_per_block_with_the_right_shapes():
    cfg, model = _tiny_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    caps = capture_forward(model, ids)
    assert len(caps) == cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads
    for cap in caps:
        assert cap.q.shape == (2, cfg.num_attention_heads, 8, d_head)
        assert cap.v.shape == cap.q.shape
        assert cap.x_residual.shape == (2, 8, cfg.hidden_size)
        assert cap.u.shape == (2, 1, 8)


def test_residual_write_at_beta_one_matches_the_plain_readout_norm():
    """`Z^{1-beta} = Z^0 = 1` exactly at the shipped corner, so the numerator
    of `r` must be bitwise the plain `||a @ v||` there -- the row-gain
    identity, re-derived here on the wired arm's own captured q/k/v/u/theta
    rather than on an isolated cell."""
    cfg, model = _tiny_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    cap = capture_forward(model, ids)[1]

    out = residual_write(cap, 1.0)
    read = smp_readout(cap.q, cap.k, cap.v, cap.u, cap.theta,
                       beta=1.0, qk=cap.qk, g=cap.g).real
    b, h, s, dh = read.shape
    read_full = read.transpose(1, 2).reshape(b, s, h * dh)
    expected = read_full.norm(dim=-1) / cap.x_residual.norm(dim=-1)
    assert torch.allclose(out["r"], expected, atol=1e-6, rtol=1e-5)
    assert not torch.equal(out["z"], torch.ones_like(out["z"]))  # Z itself is not trivially 1


def test_residual_write_at_an_interior_beta_moves_away_from_beta_one():
    """PLANTED-NEGATIVE-style check: an interior beta must not silently
    collapse to the same `r` as the shipped corner -- if it did, the beta
    argument would be dead and every reading below it vacuous."""
    cfg, model = _tiny_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    cap = capture_forward(model, ids)[0]
    r1 = residual_write(cap, 1.0)["r"]
    r_interior = residual_write(cap, 0.5)["r"]
    assert not torch.allclose(r1, r_interior)


def test_bimodality_bic_prefers_two_components_on_a_bimodal_draw():
    g = torch.Generator().manual_seed(0)
    lo = torch.exp(torch.randn(200, generator=g) * 0.05 - 3.0)   # ~ e^-3
    hi = torch.exp(torch.randn(200, generator=g) * 0.05 + 8.0)   # ~ e^8, >> lo
    result = bimodality_bic(torch.cat([lo, hi]))
    assert result["bimodal"] is True
    assert result["delta_bic_one_minus_two"] > 10.0


def test_bimodality_bic_prefers_one_component_on_a_unimodal_draw():
    g = torch.Generator().manual_seed(1)
    r = torch.exp(torch.randn(400, generator=g) * 0.3)
    result = bimodality_bic(r)
    assert result["bimodal"] is False


def test_bimodality_bic_reports_rather_than_crashes_on_too_few_samples():
    result = bimodality_bic(torch.tensor([1.0, 2.0, 3.0]))
    assert result["bimodal"] is None
    assert result["n"] == 3


def test_layer_stats_fractions_match_their_own_definition():
    r = torch.tensor([0.1, 0.5, 2.0, 5000.0, float("inf")])
    stats = layer_stats(r)
    # float32 `.mean()` of a 5-element boolean tensor is not bit-exact 0.4;
    # tolerance covers that rounding, not the logic under test.
    assert abs(stats["frac_above_1e3"] - 2 / 5) < 1e-6
    assert abs(stats["frac_below_1"] - 2 / 5) < 1e-6
    assert abs(stats["frac_non_finite"] - 1 / 5) < 1e-6
    assert stats["max"] == 5000.0  # inf excluded from min/median/max


def test_measure_runs_end_to_end_and_reports_every_beta_and_layer():
    cfg = CEQConfig(**TINY)
    report = measure(seed=0, batch=2, seq=8, betas=(1.0, 0.5, 0.1), config=cfg)
    assert report["n_layers"] == cfg.num_hidden_layers
    assert report["checkpoint"].startswith("NONE FOUND")  # no smprime checkpoint ships
    assert set(report["betas"]) == {1.0, 0.5, 0.1}
    for per_layer in report["betas"].values():
        assert len(per_layer) == cfg.num_hidden_layers
        for row in per_layer:
            assert {"min", "median", "max", "frac_above_1e3", "frac_below_1",
                    "bimodality", "dtype"} <= row.keys()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
