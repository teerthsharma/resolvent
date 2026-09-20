"""load_example.py -- load DCM-1 from this directory in a fresh process and
run both reads. Needs only torch + safetensors: no transformers, no ceqjepa.

    python load_example.py

Prints, and asserts:
  (1) strict load: zero missing keys, zero unexpected keys, 1,285,771 params
  (2) the committor is a genuine probability -- every row sums to 1
  (3) ||(I-Q)^{-1}||_inf stays under the teleport design bound 80*(1+6e-6)
  (4) the do(a) arm: q(do a) - q for two different forced moves, by
      Sherman-Morrison, one factorisation per position
  (5) THE ADVERSE FACT, printed beside (4): that difference was MEASURED to
      carry no information about which move was forced.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_dcm1 import DCM1Model, committor, fen_to_vec, square  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPA_DESIGN_BOUND_F32 = 80.0 * (1 + 6e-6)   # teleport=0.0125 -> 1/c = 80, f32 rounding
NAMES = ("white_win", "draw", "black_win", "sink")

# Two real boards, encoded without python-chess (12 piece planes x 64 + side-to-move).
FENS = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("italian ", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
]
MOVES = [("e2", "e4"), ("g1", "f3"), ("a2", "a3")]


def main():
    assert "transformers" not in sys.modules, "this artifact must not need transformers"

    model = DCM1Model.from_pretrained(HERE)          # asserts strict=True internally
    n_params = sum(p.numel() for p in model.parameters())
    print("[1] loaded DCM1Model  n=%d nA=%d d_enc=%d rank=%d teleport=%.4f params=%d"
          % (model.n, model.nA, model.enc[0].out_features, model.rank,
             model.teleport, n_params))

    x = torch.stack([fen_to_vec(f) for _, f in FENS])
    with torch.no_grad():
        out = model(x)
    q_field, q_alpha = out["q_field"], out["q_alpha"]
    kappa = committor.last_kappa_bound

    row_sums = q_field.sum(-1)
    print("[2] forward OK  x=%s  q_field=%s  q_alpha=%s"
          % (tuple(x.shape), tuple(q_field.shape), tuple(q_alpha.shape)))
    print("    committor row sums: min=%.8f max=%.8f  max|sum-1|=%.3e"
          % (row_sums.min(), row_sums.max(), (row_sums - 1).abs().max()))
    for (label, _), q in zip(FENS, q_alpha):
        print("    q_alpha[%s] = %s" % (label, {k: round(float(v), 4) for k, v in zip(NAMES, q)}))
    print("[3] ||(I-Q)^-1||_inf = %.6f  (design bound %.6f at teleport=0.0125)"
          % (kappa, KAPPA_DESIGN_BOUND_F32))

    # --- the do(a) arm -----------------------------------------------------
    moves = torch.tensor([[[square(a), square(b)] for a, b in MOVES]] * x.shape[0])
    with torch.no_grad():
        q_do, q_base, i_star, ok = model.do_read(out, moves)
    print("[4] do(a) via Sherman-Morrison, one factorisation per position; "
          "refused %d/%d" % (int((~ok).sum()), ok.numel()))
    for b, (label, _) in enumerate(FENS):
        print("    %s  intervened at chart position i*=%d   q(no intervention) = %s"
              % (label, int(i_star[b]),
                 [round(float(v), 4) for v in q_base[b]]))
        for j, (fr, to) in enumerate(MOVES):
            dq = q_do[b, j] - q_base[b]
            print("        do(%s%s): q = %s   delta = %s   max|delta| = %.3e"
                  % (fr, to, [round(float(v), 4) for v in q_do[b, j]],
                     [round(float(v), 4) for v in dq], float(dq.abs().max())))
    spread = float((q_do.unsqueeze(2) - q_do.unsqueeze(1)).abs().max())
    shift = float((q_do - q_base.unsqueeze(1)).abs().max())
    print("    max|q(do a) - q|          , i.e. HOW MUCH intervening moves q  = %.6e" % shift)
    print("    max|q(do a) - q(do a')|   , i.e. how much WHICH move matters   = %.6e" % spread)
    print("    ratio = %.1fx: the do-arm applies a near-CONSTANT shift and barely "
          "distinguishes the moves." % (shift / max(spread, 1e-12)))
    print("[5] ADVERSE, MEASURED, read this before citing [4]: on the held-out chess bed")
    print("    (1500 positions) permuting WHICH move was forced moved the score by")
    print("    -0.0044 +- 0.0138 PPL (-0.32 SE). The shift in [4] is real movement that")
    print("    carries NO outcome information -- the ratio above is that same defect")
    print("    visible in three moves. The bed's own oracle ceiling is +0.0935 +- 0.0464")
    print("    -- the WRONG SIGN -- so no model of any size could win on it.")

    assert (row_sums - 1).abs().max() < 1.5e-4, "committor rows do not sum to 1"
    assert kappa <= KAPPA_DESIGN_BOUND_F32, "teleport guarantee violated: kappa=%r" % kappa
    assert bool(ok.all()), "Sherman-Morrison refused: %s" % ok.tolist()
    assert n_params == 1285771, n_params
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
