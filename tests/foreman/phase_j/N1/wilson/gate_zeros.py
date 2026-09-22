"""Wilson N1: read d45_ckpt_f_ss0_pair on CPU; count exact zeros of the realised
magnitude gate per layer on DocByteBatches val (split_seed 0), run_j.py's eval seed.
Gate form = hard_concrete (design4x5.py:334/342 swap it in for arm f)."""
import os, sys, json
REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCR = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
for p in (REPO, os.path.join(REPO, "tests", "chase", "gate"), os.path.join(REPO, "tests", "foreman", "eval")):
    sys.path.insert(0, p)
import torch
import ceq.arm_smprime as A
import r1_gate as R
import r3_eval as RE

torch.set_grad_enabled(False)
A.magnitude = R.FORMS["hard_concrete"]
m = R.build_repaired(hidden_size=128, n_layers=3, n_heads=8, seq=512, vocab_size=256, operator="smprime")
blob = torch.load(os.path.join(SCR, "d45_ckpt_f_ss0_pair", "model.pt"), map_location="cpu", weights_only=False)
m.load_state_dict(blob["state_dict"], strict=True)
m.eval()
data = RE.DocByteBatches(RE._corpus_text(None, 64 * 1024 * 1024), 256, val_frac=0.1, split_seed=0)
eg = torch.Generator().manual_seed(20260921)
xs = [data.batch("val", 8, 512, eg, "cpu")[0] for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2)]
caps = {}
layers = list(m.model.layers)
hs = [b.self_attn.register_forward_pre_hook((lambda i: lambda mod, inp: caps.setdefault(i, []).append(inp[0]))(i)) for i, b in enumerate(layers)]
for x in xs:
    m(input_ids=x)
for h in hs:
    h.remove()
out = []
for i, b in enumerate(layers):
    at = b.self_attn
    x = torch.cat(caps[i])
    mr, _ = A.blend(at.m_head(x).squeeze(-1), at.theta_head(x).squeeze(-1), at.g)
    out.append(dict(layer=i, beta=float(at.beta), qk=float(at.qk), g=float(at.g), n=mr.numel(),
                    exact_zero=int((mr == 0).sum()), exact_one=int((mr == 1).sum()),
                    below_0p05=int((mr < 0.05).sum()), m_min=float(mr.min()), m_median=float(mr.median())))
print(json.dumps(dict(batches=len(xs), batch_shape=list(xs[0].shape), layers=out), indent=1))
