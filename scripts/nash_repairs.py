"""Issue #2: does the logit-Nash stance fail only because of its two known defects?

The nash arm lost both W7 kill conditions (`tests/w7/test_w7_nash.py`). Two
defects were known before this ran, each with its fix already stated:

  tau   `nash_operator` takes `safe_tau` over the whole batch, one max, 2.2x the
        median per-example value (tests/deimos/test_deimos_r9_iteration1.py);
        the fix is per-example tau, as scale/negation_scope.py computes it.
  bias  `ceq/arms.py` built a learned game bias `wb` and never read it; the
        game bias was `value.mean(-1)`. The fix is to read it.

This trains the 2x2 of those fixes against the signed arm on the W7 corpus
(corpus seed 0, training seeds 0-4, 400 steps, the test's own budget) and
prints OOD NRMSE. The control asserts that the (batch, mean) cell reproduces
the shipped `Arm.operator` bit for bit at seed 0, so the table measures the
shipped arm and not a lookalike. Run: python scripts/nash_repairs.py (~6 min CPU)
"""
import math
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ceq import arms, corpus, nash  # noqa: E402

SEEDS, STEPS = range(5), 400
SHIPPED_OP, SHIPPED_BUILD = arms.Arm.operator, arms.build


def variant(tau_mode, bias_mode):
    def op(self, tokens):
        x = self.emb(tokens)
        q, k = self.wq(x), self.wk(x)
        s_len, d = q.shape[-2], q.shape[-1]
        m = nash._causal_bool(s_len, x.device)
        w = ((q @ k.transpose(-2, -1)) / math.sqrt(d)).masked_fill(~m, torch.finfo(x.dtype).min)
        p = torch.softmax(w, dim=-1).masked_fill(~m, 0.0)
        g = (x @ x.transpose(-2, -1)) / math.sqrt(d)
        game = (g + g.transpose(-2, -1)) / 2.0
        bias = self.wb(x).squeeze(-1) if bias_mode == "wb" else x.mean(-1)
        if tau_mode == "batch":
            t = nash.safe_tau(game.reshape(-1, s_len, s_len))
        else:
            t = (1.25 * torch.linalg.matrix_norm(game.detach(), ord=2) / 4.0).unsqueeze(-1)
        stance = 2.0 * nash.qre_stance(game, bias, tau=t) - 1.0
        return self.rho * p * stance.unsqueeze(-2)
    return op


def build_with_wb(kind, vocab, device=None, **kw):
    m = SHIPPED_BUILD(kind, vocab, device=device, **kw)
    m.wb = torch.nn.Linear(m.emb.embedding_dim, 1, bias=False).to(device)
    return m


def train(kind, data, seed, op=SHIPPED_OP, bias_mode="mean"):
    arms.Arm.operator = op
    arms.build = build_with_wb if bias_mode == "wb" else SHIPPED_BUILD
    try:
        return arms.train_one(kind, data, "cpu", steps=STEPS, seed=seed)["ood_nrmse"]
    finally:
        arms.Arm.operator, arms.build = SHIPPED_OP, SHIPPED_BUILD


def main():
    data = corpus.build(n_train=384, n_test=128, seed=0)
    cells = {f"nash tau={t} bias={b}": (t, b) for t in ("batch", "perex") for b in ("mean", "wb")}
    rows = {"signed": [train("signed", data, s) for s in SEEDS]}
    for name, (t, b) in cells.items():
        rows[name] = [train("nash", data, s, variant(t, b), b) for s in SEEDS]

    shipped = train("nash", data, 0)
    ok = shipped == rows["nash tau=batch bias=mean"][0]
    print(f"control: shipped Arm.operator {shipped!r} vs (batch, mean) cell "
          f"{rows['nash tau=batch bias=mean'][0]!r} -> {'IDENTICAL' if ok else 'DIFFERENT'}")

    print(f"\n{'arm':>26} {'mean OOD':>9} {'sd':>7} {'min':>7} {'max':>7} {'beats signed':>13} {'< 1.0':>6}")
    for k, o in rows.items():
        wins = "-" if k == "signed" else f"{sum(a < b for a, b in zip(o, rows['signed']))}/5"
        print(f"{k:>26} {statistics.mean(o):>9.4f} {statistics.stdev(o):>7.4f} {min(o):>7.4f} "
              f"{max(o):>7.4f} {wins:>13} {sum(x < 1.0 for x in o)}/5")
    print(f"torch {torch.__version__}  threads {torch.get_num_threads()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
