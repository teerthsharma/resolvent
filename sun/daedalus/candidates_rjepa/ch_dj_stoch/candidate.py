"""Wrapped COPY of Chase's round-1 D-JEPA shell DJ(mixer="stoch") (sun/rjepa/chase/bench.py:78-110) with
Wilson's stochastic_resolvent mixer (sun/rjepa/chase/rjepa_ops.py:51-70), logic unedited.
Changes: DZ = F (2), DA = 0 (this bed has no action evidence); width h = 24 (Chase ran 64) to fit 20,000 * 1.02;
DJ returns (s lower-better, delta) -> the ranker contract's score is -s."""
import torch
import torch.nn as nn


def stochastic_resolvent(q, k, v, g=0.9, mask=None):
    if not g < 1.0:
        raise ValueError(f"g={g!r}: P is row-stochastic, rho(gP) = g, pole at g = 1")
    ct = torch.promote_types(q.dtype, torch.float32)
    q, k, vv = q.to(ct), k.to(ct), v.to(ct)
    w = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        w = w.masked_fill(~mask[..., None, :], float("-inf"))
        vv = vv * mask[..., None].to(ct)
    P = torch.nan_to_num(torch.softmax(w, -1), nan=0.0)
    if mask is not None:
        P = P * mask[..., :, None].to(ct)
    eye = torch.eye(P.shape[-1], dtype=ct, device=P.device)
    return ((1 - g) * P @ torch.linalg.solve(eye - g * P, vv)).to(v.dtype)


class DJ(nn.Module):
    def __init__(self, DZ, DA=0, h=24, nh=4):
        super().__init__()
        self.DZ, self.h, self.nh = DZ, h, nh
        self.lnz = nn.LayerNorm(DZ)
        self.enc = nn.Sequential(nn.Linear(DZ + DA + 1, h), nn.LayerNorm(h), nn.GELU())
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            ln1=nn.LayerNorm(h), qkv=nn.Linear(h, 3 * h), o=nn.Linear(h, h),
            ln2=nn.LayerNorm(h), ff=nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, h))))
            for _ in range(2))
        self.down, self.up = nn.Linear(h, 8), nn.Linear(8, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):
        DZ, Kc = self.DZ, x.shape[-2]
        c = x[..., :DZ].norm(dim=-1)
        r = c.argsort(-1).argsort(-1).to(x.dtype) / max(Kc - 1, 1)
        e = self.enc(torch.cat([self.lnz(x[..., :DZ]), x[..., DZ:], r[..., None]], -1))
        sh = lambda t: t.unflatten(-1, (self.nh, self.h // self.nh)).transpose(-2, -3)
        for L in self.layers:
            q, k, v = map(sh, L["qkv"](L["ln1"](e)).chunk(3, -1))
            m = stochastic_resolvent(q, k, v, g=0.9)
            e = e + L["o"](m.transpose(-2, -3).flatten(-2))
            e = e + L["ff"](L["ln2"](e))
        delta = 0.2 * torch.tanh(self.up(torch.tanh(self.down(e))))[..., 0]
        return r + delta, delta


class Ranker(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.dj = DJ(F)

    def forward(self, z):
        return -self.dj(z)[0]


def build(spec):
    return Ranker(spec["F"])
