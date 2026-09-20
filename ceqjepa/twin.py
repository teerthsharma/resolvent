"""The twin-pair constructor for the L-VARY comparison (Addendum F-N sec. 6).

OURS is `ArmSMPrime`'s three-switch body (`beta`, `qk`, `g`, already in this
codebase at `ceq/arm_smprime.py`) with `beta` learnable, its two gate heads
(`m_head`, `theta_head`) trainable, and the C2 route constraint on (both
arms call `operator()` at its default `route="product"` -- the only route
`arm_smprime.py`'s own docstring proves correct; nothing here ever passes
`route="exp_scan"`). THE TWIN is the SAME module, `beta` pinned at exactly
1 and both gate heads pinned at the identity gate (`m=1, theta=0`), all
five `requires_grad_(False)`. That pin/unpin is L-VARY's one factor.

C1-C8 (Addendum F-N sec. 6) are named checks a training loop runs before
trusting an OURS row -- metadata, not extra weights. They add no parameter
and this module does not enforce them; enforcing them is a training-loop
concern, out of scope for a constructor. THE TWIN has no live gate for a
route constraint to bind, so C1-C8 and C2 are both moot for it, not
violated.

Both arms read the same three heads on the same features (the strawman
trap, sec. 8: a twin denied a head OURS has is not a twin): a WDL-5
classifier, a tau-hat regression in plies, and an r_gamma read against
T = {[WIN]} (sigmoid -- a probability read, not a logit). The rule
decision (`rule_decision`) is derived from tau-hat, not a fourth head.

NO TRAINING HERE (L-LEAN, same as `arm_smprime.py`). Run this file
directly for the self-check: it builds one pair, asserts the parameter
counts match and that exactly one config field differs, and prints both.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from ceq.arm_smprime import ArmSMPrime

__all__ = [
    "TwinConfig", "ThreeSwitchTwin", "make_twin_pair",
    "rule_decision", "diff_fields", "count_params", "C1_C8", "ROUTE",
]

#: The route both arms use (`operator()`'s default). Naming it here so a
#: future edit that starts passing `route=` explicitly has one constant to
#: change, not two call sites that can drift apart into a second factor.
ROUTE = "product"

#: Addendum F-N sec. 6's named checks a training loop runs before trusting
#: an OURS row. Listed here as the set OURS runs "in force" under; see the
#: module docstring for why they add no parameter and bind nothing here.
C1_C8 = tuple("C%d" % i for i in range(1, 9))


@dataclasses.dataclass(frozen=True)
class TwinConfig:
    """Everything OURS and THE TWIN share, plus the one factor that varies.

    `gates_on=True` is OURS: `beta` learnable, the gate heads trainable.
    `gates_on=False` is THE TWIN: `beta` pinned at 1, the gate heads pinned
    at the identity gate. Every other field is architecture, seed and data
    shared by construction -- `make_twin_pair` builds both arms from the
    same `TwinConfig` with only this field flipped.
    """
    seq: int
    d_model: int = 16
    hidden: int = 128
    seed: int = 0
    gates_on: bool = True


class ThreeSwitchTwin(nn.Module):
    """One arm: `ArmSMPrime`'s three-switch body plus the three shared
    heads. `cfg.gates_on` picks OURS or THE TWIN; nothing else about the
    module changes shape between the two, so the parameter count is equal
    by construction (freezing a parameter does not remove it)."""

    def __init__(self, cfg: TwinConfig):
        super().__init__()
        self.cfg = cfg
        self.body = ArmSMPrime(cfg.seq, d_model=cfg.d_model, hidden=cfg.hidden)
        self.body.trainable_heads()          # beta/qk/g=1, gate heads near-identity, all live
        self.wdl_head = nn.Linear(cfg.d_model, 5)
        self.tau_head = nn.Linear(cfg.d_model, 1)
        self.rgamma_head = nn.Linear(cfg.d_model, 1)
        if not cfg.gates_on:
            self._pin_gates_off()

    def _pin_gates_off(self) -> None:
        """THE TWIN setting: `beta` -> exactly 1, gate heads -> the identity
        gate (`m=1, theta=0`), all five `requires_grad=False` so nothing
        here ever moves off that corner. Same layers, same shapes as OURS."""
        with torch.no_grad():
            self.body.beta.fill_(1.0)
            self.body.m_head.weight.zero_()
            self.body.m_head.bias.fill_(1.0)
            self.body.theta_head.weight.zero_()
            self.body.theta_head.bias.zero_()
        for p in (self.body.beta, self.body.m_head.weight, self.body.m_head.bias,
                  self.body.theta_head.weight, self.body.theta_head.bias):
            p.requires_grad_(False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The last position's hidden state -- what all three heads read.
        `ArmSMPrime.forward`'s own body, minus its scalar `readout` (which
        stays a registered, unused parameter so both arms' counts match
        the plain `ArmSMPrime` shape they were built from)."""
        n, seq, _ = x.shape
        u, th = self.body.heads(x)
        a = self.body._operator(self.body.wq(x), self.body.wk(x), u, th)
        h = self.body.mlp(x + (a @ x.to(a.dtype)).real)
        return h[:, seq - 1]

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.features(x)
        return {
            "wdl": self.wdl_head(feat),
            "tau_hat": self.tau_head(feat).squeeze(-1),
            "r_gamma": torch.sigmoid(self.rgamma_head(feat)).squeeze(-1),
        }


def rule_decision(win: torch.Tensor, tau_hat: torch.Tensor) -> torch.Tensor:
    """cursed iff win and tau_hat > 100 plies (Addendum F-N sec. 6) --
    derived from tau-hat, never a separately learned label."""
    return win.bool() & (tau_hat > 100)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def diff_fields(a: TwinConfig, b: TwinConfig) -> list:
    """Field names where `a` and `b` disagree -- checked, not trusted."""
    return [f.name for f in dataclasses.fields(a)
            if getattr(a, f.name) != getattr(b, f.name)]


def make_twin_pair(cfg: TwinConfig):
    """Build OURS (`gates_on=True`) and THE TWIN (`gates_on=False`) from
    one config, asserting the two things the strawman trap (sec. 8) says a
    lane must never just trust: that only one field varies, and that the
    parameter counts match. Either failing is Addendum F-N's own finding,
    not a bug to route around -- rule 0 applies, so this raises rather than
    silently shipping a confounded pair.
    """
    ours_cfg = dataclasses.replace(cfg, gates_on=True)
    twin_cfg = dataclasses.replace(cfg, gates_on=False)

    diffs = diff_fields(ours_cfg, twin_cfg)
    if diffs != ["gates_on"]:
        raise RuntimeError(
            "L-VARY violated: OURS/THE TWIN configs differ in %r, not "
            "exactly ['gates_on'] -- report this as the finding, rule 0 "
            "applies, do not ship a row against these two arms." % (diffs,)
        )

    torch.manual_seed(cfg.seed)
    ours = ThreeSwitchTwin(ours_cfg)
    torch.manual_seed(cfg.seed)               # same seed -> same shared init
    twin = ThreeSwitchTwin(twin_cfg)

    n_ours, n_twin = count_params(ours), count_params(twin)
    if n_ours != n_twin:
        raise RuntimeError(
            "PARAMETER COUNT MISMATCH: OURS=%d THE TWIN=%d -- the "
            "comparison as specified is confounded. Report this as the "
            "finding, rule 0 applies, do not ship a row against these two "
            "arms." % (n_ours, n_twin)
        )
    return ours, twin


if __name__ == "__main__":
    ours, twin = make_twin_pair(TwinConfig(seq=8))
    n_ours, n_twin = count_params(ours), count_params(twin)
    diffs = diff_fields(ours.cfg, twin.cfg)
    print("OURS params:", n_ours)
    print("THE TWIN params:", n_twin)
    print("differing attributes:", diffs)
    assert n_ours == n_twin
    assert diffs == ["gates_on"]
    assert bool(ours.body.beta.requires_grad) is True
    assert bool(twin.body.beta.requires_grad) is False
    assert float(twin.body.beta) == 1.0
