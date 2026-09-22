"""Foreman T2d: train the as-built fq arm (copied a5_bed.A5Arm, unedited
class) with and without a BOS anchor token, at the bed budget and at 8x data.
Only difference in the anchored arm: Embedding(5,d) and token 4 prepended;
loss/accuracy on the original 64 positions only. CPU float32, seed 0.
argv: tag n_train steps bos(0/1)
"""
import json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import a5_bed as A

tag, n_train, steps, bos = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); variant = sys.argv[5] if len(sys.argv) > 5 else "fq"
A.N_TRAIN = n_train
tr_tok, tr_lab = A.gen_split("train", 0); ev_tok, ev_lab = A.gen_split("eval", 0)
torch.manual_seed(0)
m = A.A5Arm(A.D, A.N_HEADS, A.N_LAYERS, variant)
if bos:
    m.emb = torch.nn.Embedding(5, A.D)
def prep(t):
    t = torch.tensor(t)
    return torch.cat([torch.full((t.shape[0], 1), 4), t], 1) if bos else t
def fwd(t):
    return m(prep(t))[:, 1:] if bos else m(prep(t))
opt = torch.optim.AdamW(m.parameters(), lr=A.LR, weight_decay=A.WEIGHT_DECAY)
trY, evY = torch.tensor(tr_lab), torch.tensor(ev_lab)
t0 = time.time()
for _ in range(steps):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(fwd(tr_tok).reshape(-1, 60), trY.reshape(-1))
    loss.backward(); opt.step()
m.eval()
with torch.no_grad():
    pr = fwd(ev_tok).argmax(-1)
    acc = {str(p): float((pr[:, p - 1] == evY[:, p - 1]).float().mean()) for p in [1, 2, 4, 8, 16, 32, 64]}
    acc["all"] = float((pr == evY).float().mean())
out = {"tag": tag, "variant": variant, "n_train": n_train, "steps": steps, "bos": bos, "final_train_loss": float(loss),
       "eval_acc": acc, "seconds": round(time.time() - t0, 1)}
print(json.dumps(out))
with open(os.path.join(HERE, "a5_anchor_train.jsonl"), "a") as f:
    f.write(json.dumps(out) + "\n")
