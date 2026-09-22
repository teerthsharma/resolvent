"""R-POS: the grid's softmax twin (a) plus a zero-parameter relative prior.

a_rope  : RoPE on q,k inside attention (d_head 16, base 10000, rotate-half form).
a_alibi : ALiBi bias -slope_h*(i-j), slopes 2^(-8h/H), h=1..H, H=8.
Everything else is design4x5.train_arm("a") verbatim: the "a" branch patches
CEQAttention.forward with the module-global `softmax_forward`, so this file
rebinds that one global and calls train_arm itself -- same build, seed, CRN,
steps, eval subsample. The absolute position table stays.

python rpos.py --check          # CPU self-checks, no training
python rpos.py --run            # 6 cells + one (a) reproduction cell
"""
import hashlib
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SP = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
sys.path.insert(0, SP)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import design4x5 as D  # noqa: E402
import crn_build as CB  # noqa: E402
import r3_eval as RE  # noqa: E402

ROWS = os.path.join(HERE, "rpos_results.jsonl")
GRID = os.path.join(D.REPO, "tests", "foreman", "design4x5", "design4x5_results.jsonl")
STEPS = 3538
_SOFTMAX = D.softmax_forward
_CACHE = {}


def rope_cs(s, dh, device, dtype, base=10000.0):
    key = ("rope", s, dh, str(device), dtype)
    if key not in _CACHE:
        inv = 1.0 / (base ** (torch.arange(0, dh, 2, dtype=torch.float32) / dh))
        f = torch.arange(s, dtype=torch.float32)[:, None] * inv[None, :]
        emb = torch.cat([f, f], dim=-1)
        _CACHE[key] = (emb.cos().to(device, dtype), emb.sin().to(device, dtype))
    return _CACHE[key]


def rotate(t, cos, sin):
    h = t.shape[-1] // 2
    return t * cos + torch.cat([-t[..., h:], t[..., :h]], dim=-1) * sin


def alibi_mask(s, n_heads, device, dtype):
    key = ("alibi", s, n_heads, str(device), dtype)
    if key not in _CACHE:
        slopes = torch.tensor([2.0 ** (-8.0 * h / n_heads) for h in range(1, n_heads + 1)])
        i = torch.arange(s)[:, None]
        j = torch.arange(s)[None, :]
        m = -slopes[:, None, None] * (i - j).float()[None]
        m = m.masked_fill((j > i)[None], float("-inf"))
        _CACHE[key] = m[None].to(device, dtype)          # [1, H, S, S]
    return _CACHE[key]


def rope_forward(self, x, attention_mask=None):
    if attention_mask is not None:
        raise NotImplementedError
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)

    def shape(t):
        return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    cos, sin = rope_cs(s, self.d_head, x.device, x.dtype)
    o = F.scaled_dot_product_attention(rotate(shape(q), cos, sin), rotate(shape(k), cos, sin),
                                       shape(v), is_causal=True)
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


def alibi_forward(self, x, attention_mask=None):
    if attention_mask is not None:
        raise NotImplementedError
    b, s, d = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)

    def shape(t):
        return t.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    o = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                       attn_mask=alibi_mask(s, self.n_heads, x.device, x.dtype))
    return self.o_proj(o.transpose(1, 2).reshape(b, s, d))


FORWARDS = {"a_rope": rope_forward, "a_alibi": alibi_forward, "a": _SOFTMAX}


def check():
    """RoPE: logits depend on i-j only (Toeplitz for position-constant q,k);
    ALiBi with slopes forced to 0 equals the grid's softmax_forward; zero new params."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 16).expand(1, 1, 64, 16)
    k = torch.randn(1, 1, 1, 16).expand(1, 1, 64, 16)
    cos, sin = rope_cs(64, 16, "cpu", torch.float32)
    L = rotate(q, cos, sin) @ rotate(k, cos, sin).transpose(-1, -2)
    toe = max(float((L[0, 0, i + 1:, 1:] - L[0, 0, i:-1, :-1]).abs().max()) for i in range(0, 1))
    assert toe < 1e-4, toe
    m = D._ORIG_BUILD(hidden_size=D.HIDDEN, n_layers=D.LAYERS, n_heads=D.HEADS,
                      seq=D.SEQ, vocab_size=D.VOCAB, operator="sgate")
    attn = m.model.layers[0].self_attn
    x = torch.randn(2, 64, D.HIDDEN)
    key = ("alibi", 64, D.HEADS, "cpu", torch.float32)
    mm = torch.zeros(1, D.HEADS, 64, 64).masked_fill(
        (torch.arange(64)[None, :] > torch.arange(64)[:, None])[None, None], float("-inf"))
    _CACHE[key] = mm
    with torch.no_grad():
        dz = float((alibi_forward(attn, x) - _SOFTMAX(attn, x)).abs().max())
    del _CACHE[key]
    assert dz < 1e-5, dz
    slopes = [2.0 ** (-8.0 * h / 8) for h in range(1, 9)]
    print("[CHECK] rope toeplitz max dev={:.2e}  alibi(slope=0)-softmax={:.2e}  slopes={}  "
          "params={}".format(toe, dz, slopes, sum(p.numel() for p in m.parameters())), flush=True)


def recomputed_digest(ss, text):
    tr, va, _ = CB.doc_order(text, ss)
    return CB.digest_order(tr, va)


def run():
    grid = {}
    for l in io.open(GRID, encoding="utf-8"):
        r = json.loads(l)
        grid[(r["arm"], r["split_seed"])] = r
    crn = D.load_crn()
    text = RE._corpus_text(None, 64 * 1024 * 1024)
    D.OUT_DIR_TMPL = os.path.join(HERE, "ckpt_{arm}_ss{ss}")
    cells = [(a, ss) for ss in (0, 1, 2) for a in ("a_rope", "a_alibi")] + [("a", 0)]
    if "--cell" in sys.argv:  # one cell per fresh process: the poll counts this process's own context as busy
        i = sys.argv.index("--cell")
        cells = [(sys.argv[i + 1], int(sys.argv[i + 2]))]
    for arm, ss in cells:
        if not D.poll_until_free(timeout_s=900):
            print("[HELD] card busy, stopping at", arm, ss, flush=True)
            break
        D.softmax_forward = FORWARDS[arm]
        D.OUT_DIR_TMPL = os.path.join(HERE, "ckpt_" + arm + "_ss{ss}")
        t1 = time.time()
        try:
            rec = D.train_arm("a", ss, ss, STEPS, crn)
        finally:
            D.softmax_forward = _SOFTMAX
        dt = time.time() - t1
        torch.cuda.empty_cache()
        dg_re = recomputed_digest(ss, text)
        g = grid[("a", ss)]
        row = dict(arm=arm, split_seed=ss, seed=ss, n_params=rec["n_params"],
                   final_eval_loss=rec["final_eval_loss"], loss_last_train=rec["losses"][-1],
                   crn_digest_file=rec["crn_digest"], crn_digest_recomputed=dg_re,
                   crn_digest_grid=g["crn_digest"],
                   crn_match=(dg_re == g["crn_digest"] == rec["crn_digest"]),
                   n_params_eq_grid_a=(rec["n_params"] == g["n_params"]),
                   steps=STEPS, run_seconds=dt, grid_a_run_seconds=g["run_seconds"],
                   ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with io.open(ROWS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print("[CELL]", json.dumps(row), flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    if "--run" in sys.argv:
        run()
