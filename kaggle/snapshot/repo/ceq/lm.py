"""A small from-scratch language model with a pluggable attention operator.

Two arms, identical in every way except one line:

    softmax   out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    signed    out = v + A v + A^2 v + ... + A^K v      A strictly causal, signed

The operator carries no parameters of its own -- q, k, v and o projections belong
to the block and are shared -- so the two arms have IDENTICAL parameter counts,
not merely similar ones. Same initialization seed, same data, same steps, same
optimizer. The operator is the only free variable, which is what makes the
comparison a statement about the operator rather than about a training run.

WHY BYTE-LEVEL. Vocabulary 256, no tokenizer. Nothing about the comparison can
hide in a merge table, and there is no tokenizer dependency to argue about. It
also removes the injectivity hazard that has bitten this project before: a
tokenizer that maps distinct answers to the same first token silently destroys
whatever is being measured.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

D_MODEL, N_LAYERS, N_HEADS, SEQ = 256, 4, 4, 128
RHO, HOPS = 0.9, 3
SGATE_LAM = 1.0   # weight on the negative half; 1.0 makes row sums exactly zero

# --- softmax's own knobs, so the baseline is armed as well as the challenger ---
# A parity claim where the challenger got four tuned knobs and the control got one
# is worth less than one that survives a control with the same surface. These are
# the direct analogues: SMX_TAU is a logit temperature (what rho buys the signed
# arm by rescaling row mass), SMX_RHO an output scale, SMX_HOPS multi-hop
# propagation, SMX_ID the identity term.
#
# SMX_TAU=1, SMX_RHO=1, SMX_HOPS=1, SMX_ID=False is EXACTLY standard causal
# softmax attention, so the default recovers the untouched baseline.
SMX_TAU, SMX_RHO, SMX_HOPS, SMX_ID = 1.0, 1.0, 1, False
VOCAB = 256


class ByteCorpus:
    """Raw bytes, 90/10 train/val, contiguous split so val is unseen text."""

    def __init__(self, text: str, val_frac: float = 0.1):
        b = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        cut = int(len(b) * (1 - val_frac))
        self.train, self.val = b[:cut], b[cut:]

    def batch(self, split: str, bs: int, seq: int, gen: torch.Generator, device):
        d = self.train if split == "train" else self.val
        i = torch.randint(len(d) - seq - 1, (bs,), generator=gen)
        x = torch.stack([d[j:j + seq] for j in i])
        y = torch.stack([d[j + 1:j + seq + 1] for j in i])
        return x.to(device), y.to(device)


class Attention(nn.Module):
    """One attention block. `kind` selects the operator and nothing else."""

    def __init__(self, kind: str, d: int, n_heads: int):
        super().__init__()
        assert kind in ("softmax", "softmax_x", "signed", "sgate", "sgate_nores")
        self.kind, self.n_heads, self.d_head = kind, n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def operator(self, q: torch.Tensor, k: torch.Tensor | None = None) -> torch.Tensor:
        """The signed, strictly causal coupling matrix.

        Two forms, and the difference between them is the whole parity campaign.

        `signed` -- raw logits normalized by their row L1 norm. Signed, but
        LINEAR in the logits and therefore flat: it cannot concentrate the way
        `exp` does. Measured cost of that flatness: val loss 2.1103 against
        softmax 1.5780 at 600 steps, and the gap WIDENS with budget.

        `sgate` -- a difference of two softmaxes over the SAME logits, with the
        negative half weighted by `lam`:

            A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)

        This docstring said `A = (rho/2) * (softmax(w) - softmax(-w))` until
        2026-08-25. That is the `lam = 1` special case, and `lam` is exactly the
        knob that got the operator from 1.2099 to parity -- the shipped point is
        `lam = 0.10`. The formula below the docstring always had `lam`; the
        docstring did not.

        Both halves keep `exp` sharpening, so selectivity is restored, and the
        result is signed: positive weight concentrates on the most similar keys,
        negative on the least. No new parameters -- the arms stay identical in
        parameter count, which is what makes the comparison a statement about
        the operator rather than about capacity.

        SIGNEDNESS IS ONLY STRUCTURAL AT `lam = 1`, where rows sum to exactly
        zero. At `lam = 0.10` an entry goes negative only when
        `softmax(w)_ij < lam * softmax(-w)_ij`, so it needs the entry to sit far
        below its own row: at `nn.Linear` initialization the measured negative
        fraction is 3.91e-04 (cpu) / 4.99e-04 (cuda), against 5.25e-02 at
        unit-variance logits. The `min A = -0.1080` on record is the TRAINED
        model.

        Row L1 is bounded by `rho * (1 + lam) / (1 + lam) = rho` in both forms,
        so the operator stays strictly causal with bounded rows and
        `CEQ.Nilpotent.pow_card_eq_zero` applies unchanged -- it is stated over
        `[CommRing R]` with the single hypothesis `forall i j, i <= j -> A i j =
        0` and is blind to both sign and magnitude. `CEQ.Contraction` is NOT
        available at the shipped `rho = 1.5`, and neither is the geometric
        truncation bound `rho^(K+1)/(1-rho)`, which is -6.75 there;
        `ceq.attention.truncation_bound` raises rather than returning it.

        THE MODULE GLOBALS BELOW ARE NOT THE PARITY POINT. `RHO, HOPS = 0.9, 3`
        and `SGATE_LAM = 1.0` are the pre-campaign values; the 5-seed run at
        1.0334 set `1.5, 0.10, 2` before calling this. Every test and benchmark
        that wants the parity point sets them explicitly, and that asymmetry is
        recorded as open rather than fixed here, because these globals are also
        the defaults every earlier measurement in this file was taken at.
        """
        k = q if k is None else k
        s = q.shape[-2]
        w = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        m = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(-1)

        if self.kind == "softmax_x":
            # Standard causal softmax -- diagonal INCLUDED, unlike the signed
            # arms -- with a temperature, an output scale and optional multi-hop.
            # This is the armed control.
            mi = torch.ones(s, s, dtype=torch.bool, device=w.device).tril(0)
            neg = torch.finfo(w.dtype).min
            p = torch.softmax((w / SMX_TAU).masked_fill(~mi, neg), dim=-1)
            return SMX_RHO * p.masked_fill(~mi, 0.0)

        if self.kind == "sgate":
            # A = rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)
            #
            # Two obstructions, measured, and lam trades between them.
            #
            # TEMPERATURE. `rho * w / sum|w|` is homogeneous of degree ZERO --
            # A(t*q, k) = A(q, k) -- so peak row weight was 0.100336 at logit
            # scales 0.25, 1.0, 4.0 and 16.0 alike, spread exactly 0.000e+00 over
            # a 64x sweep, against softmax 0.0504 -> 1.0000. grad_q . q was
            # 1.088e-14 against softmax's 10.561. Every query permanently loses a
            # degree of freedom and there is no temperature channel. Softmax is
            # not degree-zero homogeneous, so both halves here restore it.
            #
            # DC. At lam = 1 the two halves each sum to 1, so rows sum to exactly
            # zero and A annihilates the constant vector: measured row sum
            # 0.000000e+00 and |A @ 1| = 1.192e-07. The path sum can then add only
            # deviations, never signal level. lam < 1 leaves row sum
            # rho*(1-lam)/(1+lam) > 0 and restores it.
            lam = SGATE_LAM
            neg = torch.finfo(w.dtype).min
            pp = torch.softmax(w.masked_fill(~m, neg), dim=-1).masked_fill(~m, 0.0)
            pm = torch.softmax((-w).masked_fill(~m, neg), dim=-1).masked_fill(~m, 0.0)
            return RHO * (pp - lam * pm) / (1.0 + lam)

        w = w.masked_fill(~m, 0.0)
        l1 = w.abs().sum(-1, keepdim=True)
        return RHO * w / l1.clamp_min(torch.finfo(w.dtype).tiny)

    def qkv_heads(self, x: torch.Tensor):
        """Split the fused projection into per-head q, k, v."""
        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = lambda t: t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        return shape(q), shape(k), shape(v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv_heads(x)

        if self.kind == "softmax":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.kind == "softmax_x":
            o = path_sum_terms(self.operator(q, k), v, SMX_HOPS,
                               include_identity=SMX_ID)
        else:
            # `sgate_nores` drops the k=0 term. The block already adds a residual
            # (`x = x + attn(n1(x))`), and the softmax arm's output `P v` contains
            # no bare copy of `v`, so a path sum that starts at `v` gives the
            # signed arms an identity path the control does not have -- counted
            # twice against the same block residual. That is a defect in the
            # comparison, not a property of the operator, and it was present in
            # every number measured before this file existed.
            o = path_sum_terms(self.operator(q, k), v, HOPS,
                               include_identity=self.kind != "sgate_nores")
        return self.proj(o.transpose(1, 2).reshape(b, s, d))


def path_sum_terms(a: torch.Tensor, v: torch.Tensor, hops: int,
                   *, include_identity: bool = True) -> torch.Tensor:
    """`v + Av + ... + A^K v`, or `Av + ... + A^K v` without the identity.

    Dropping the k=0 term gives `A (I - A)^-1 v`, a polynomial in `A`, so it is
    still strictly causal, still signed and still nilpotent --
    `CEQ.Nilpotent.pow_card_eq_zero` applies to any such polynomial unchanged.
    """
    o = v.clone() if include_identity else torch.zeros_like(v)
    term = v
    for _ in range(hops):
        term = a @ term
        o = o + term
    return o


class Block(nn.Module):
    def __init__(self, kind: str, d: int, n_heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(kind, d, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, kind: str, d: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, seq: int = SEQ, seed: int | None = None,
                 vocab: int = VOCAB):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.kind, self.n_heads, self.d_head, self.seq = kind, n_heads, d // n_heads, seq
        # VOCAB is the byte-level default, not a design commitment. COGS has 874
        # word types; a fixed 256-way head would have to truncate them, and a
        # truncated COGS reports a number that looks like a COGS score and is not
        # one. The operator carries no parameters either way, so both arms grow by
        # exactly the same amount and stay parameter-identical.
        self.vocab = vocab
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(kind, d, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, s = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(s, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


def train_one(kind: str, corpus: ByteCorpus, *, steps: int, device, seed: int = 0,
              bs: int = 32, lr: float = 3e-4, **model_kw) -> dict:
    """Identical budget for both arms: same steps, lr, batch size, seed, data.

    `model_kw` (`d`, `n_layers`, `n_heads`, `seq`) is forwarded to `TinyLM` so a
    size ladder can be run without a second training loop. It must be passed
    explicitly rather than by rebinding `D_MODEL`/`SEQ`: those are default
    argument values, bound once at definition time, so a module-level rebind is
    silently ignored.
    """
    m = TinyLM(kind, seed=seed, **model_kw).to(device)
    gen = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)

    m.train()
    for _ in range(steps):
        x, y = corpus.batch("train", bs, m.seq, gen, device)
        loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    m.eval()
    vg = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        vals = []
        for _ in range(20):
            x, y = corpus.batch("val", bs, m.seq, vg, device)
            vals.append(F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1)))
        val = float(torch.stack(vals).mean())
    return dict(kind=kind, val_loss=val, train_loss=float(loss),
                n_params=m.n_params(), model=m)


def compare(corpus: ByteCorpus, *, steps: int, device, seed: int = 0) -> dict:
    return compare_kinds(corpus, ("softmax", "signed"), steps=steps, device=device,
                         seed=seed)


def compare_kinds(corpus: ByteCorpus, kinds, *, steps: int, device,
                  seed: int = 0, **kw) -> dict:
    """Every arm gets the same budget, the same seed and the same data. The
    operator is the only free variable -- `kw` reaches every arm identically."""
    return {k: train_one(k, corpus, steps=steps, device=device, seed=seed, **kw)
            for k in kinds}
