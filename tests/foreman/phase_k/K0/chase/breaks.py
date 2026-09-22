"""What breaks (Chase K0). CPU float64 unless stated. Rows -> breaks_rows.jsonl.
B1 ALiBi prior heads (slopes 2^-h, h=1..8 = Phase J a_alibi's 2^(-8h/H), H=8), content-free Toeplitz, n=4096:
   dressed mass (instrument vs Y2), range, and the BOS weight of the read w0(i) = (1-g) G(i,0) at i = 255, 1023, 4095.
   NO trained ALiBi weights exist: tests/foreman/phase_j/N2/foreman/ckpt_a_alibi_ss{0,1,2}/ hold run_record.json only.
B2 same slopes + random content (q,k ~ N(0,1), D=64, SSMax a=1 b=0), n=4096, g=.99/.999: instrument mass, BOS weight.
B3 exact unit self-weights: rows with W_ii == 1.0 in fp32 softmax (SSMax a in .5..8) and in sparsemax, n=4096.
B4 bf16 weights: row-sum defect and the read's gain (1-g)(I-gW)^-1 1 (exact answer 1) with W rounded to bf16."""
import json, math, sys, os
import numpy as np, torch
from scipy.linalg import solve_triangular
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rrange, resolvent as R
out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "breaks_rows.jsonl"), "w")
def emit(d):
    print(json.dumps(d), flush=True); out.write(json.dumps(d) + "\n")

n = 4096
slopes = [2.0 ** (-h) for h in range(1, 9)]
rows = np.arange(3500, 4001, 100); r = np.arange(30, 400)
e0 = np.zeros(n); e0[0] = 1.0
for g in (0.9, 0.99, 0.999):
    for lam in slopes:
        W = rrange.toeplitz_head(lam, n)
        m_fit = rrange.head_mass(W, g, rows, r)
        m_y2 = math.log(g + (1 - g) * math.exp(lam))
        w0 = (1 - g) * solve_triangular(np.eye(n) - g * W, e0, lower=True)
        emit(dict(t="B1", g=g, slope=lam, m_fit=m_fit, m_y2=m_y2, rel=abs(m_fit - m_y2) / m_y2, range=1 / m_fit,
                  bos_w_255=w0[255], bos_w_1023=w0[1023], bos_w_4095=w0[4095], exp_m_i_4095=math.exp(-m_y2 * 4095)))

g0 = torch.Generator().manual_seed(7)
q = torch.randn(1, 8, n, 64, generator=g0, dtype=torch.float64); k = torch.randn(1, 8, n, 64, generator=g0, dtype=torch.float64)
qs = R.ssmax_q(q, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
for h in range(8):
    Wh = R.weights(qs[:, h:h + 1], k[:, h:h + 1], alibi=False)[0, 0]
    z = R._logits(qs[:, h:h + 1], k[:, h:h + 1])[0, 0]
    i_ = torch.arange(n, dtype=torch.float64)
    z = z - slopes[h] * (i_[:, None] - i_[None, :]).clamp(min=0)
    W = torch.softmax(z, -1).numpy(); del z, Wh
    for g in (0.99, 0.999):
        m_fit = rrange.head_mass(W, g, rows, r)
        w0 = (1 - g) * solve_triangular(np.eye(n) - g * W, e0, lower=True)
        emit(dict(t="B2", g=g, slope=slopes[h], ssmax_a=1.0, m_fit=m_fit, range=1 / m_fit if m_fit > 0 else float("inf"),
                  m_y2_bare_slope=math.log(g + (1 - g) * math.exp(slopes[h])), bos_w_1023=w0[1023], bos_w_4095=w0[4095],
                  bos_w_mean=float(w0.mean()), max_Wii_offBOS=float(np.diag(W)[1:].max())))

q32, k32 = q.float(), k.float()
for a in (0.5, 1.0, 2.0, 4.0, 8.0):
    qs32 = R.ssmax_q(q32, torch.tensor(a), torch.tensor(0.0))
    d = torch.cat([R.diag_weights(qs32[:, h:h + 1], k32[:, h:h + 1]) for h in range(8)], 1)
    Ws = R.sparsemax(R._logits(qs32[:, :2], k32[:, :2]))
    ds = torch.diagonal(Ws, dim1=-2, dim2=-1)
    emit(dict(t="B3", ssmax_a=a, softmax_rows=8 * n, softmax_unit_self=int((d == 1.0).sum()),
              softmax_unit_self_excl_row0=int((d[..., 1:] == 1.0).sum()), softmax_max_Wii_excl_row0=float(d[..., 1:].max()),
              sparsemax_rows=2 * n, sparsemax_unit_self=int((ds == 1.0).sum()),
              sparsemax_onehot_rows=int((Ws.max(-1).values == 1.0).sum())))

for a in (0.0, 1.0):
    for alibi in (False, True):
        qs16 = R.ssmax_q(q32[:, :2], torch.tensor(a), torch.tensor(1.0 if a == 0 else 0.0))
        z = R._logits(qs16, k32[:, :2], alibi=alibi)[:, :2]
        Wb = torch.softmax(z.bfloat16(), -1).double()[0]
        defect = (Wb.sum(-1) - 1)
        for g in (0.99, 0.999):
            gain = [(1 - g) * solve_triangular(np.eye(n) - g * Wb[h].numpy(), np.ones(n), lower=True) for h in range(2)]
            gain = np.stack(gain)
            emit(dict(t="B4", ssmax_a=a, alibi=alibi, g=g, max_rowsum_defect=float(defect.abs().max()),
                      frac_rows_g_rowsum_ge_1=float((g * Wb.sum(-1) >= 1).double().mean()),
                      read_gain_max=float(gain.max()), read_gain_min=float(gain.min()),
                      gamma_rho_bf16=float(g * torch.diagonal(Wb, dim1=-2, dim2=-1).max())))
out.close()
