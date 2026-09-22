"""Foreman: hand-set the as-built (f_Q) layer (copied a5_bed.Layer, unedited)
to the exact A5 lift and read held-out running products.

T2a  anchor forced (every query attends key 0) -> expect 1.0  (fold+lift+convention OK)
T2b  as-built SDPA, same weights, Q=K=0 (uniform; layer 1 has no key-0 signal) -> RED
T2c  trained as-built arm (seed 0, bed budget): did it learn ANY A5 lift? -> RED
"""
import itertools, json, os, sys, types, faulthandler; faulthandler.dump_traceback_later(100)
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a5_bed as A  # the foreman copy (identical to SPJ/a5_bed.py)
from su2 import qmul, qconj, qnormalize

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
res = {}


# ---- 2I and a lift of the bed's generators (consistent mod +-1 on all of A5)
def aq(deg, ax):
    ax = np.array(ax, float); ax /= np.linalg.norm(ax); h = np.radians(deg) / 2
    return np.concatenate([[np.cos(h)], np.sin(h) * ax])


def qm(a, b):
    return qmul(torch.tensor(a), torch.tensor(b)).numpy()


key = lambda q: tuple(np.round(q, 6) + 0.0)
E = {key(np.array([1., 0, 0, 0])): np.array([1., 0, 0, 0])}
fr = list(E.values())
while fr:
    nx = []
    for p in fr:
        for g in (aq(72, [0, 1, (1 + 5 ** 0.5) / 2]), aq(120, [1, 1, 1])):
            r = qm(g, p)
            if key(r) not in E:
                E[key(r)] = r; nx.append(r)
    fr = nx
E2I = np.array(list(E.values())); assert len(E2I) == 120
cand = [e for e in E2I if abs(abs(e[0]) - 0.5) < 1e-9]  # 120-degree rotations (3-cycles)


def try_lift(a, b):
    gq = [a, b, np.array([a[0], *-a[1:]]), np.array([b[0], *-b[1:]])]
    seen = {A.IDENTITY: np.array([1., 0, 0, 0])}
    fr = [A.IDENTITY]
    while fr:
        nx = []
        for p in fr:
            for g, q in zip(A.GENS, gq):
                pp, qq = A._perm_compose(g, p), qm(q, seen[p])  # bed: prod=g∘prod ; fold: Pi=q*Pi
                if pp in seen:
                    if min(np.abs(seen[pp] - qq).max(), np.abs(seen[pp] + qq).max()) > 1e-6:
                        return None
                else:
                    seen[pp] = qq; nx.append(pp)
        fr = nx
    return gq, seen


lift = None
for a, b in itertools.product(cand, cand):
    lift = try_lift(a, b)
    if lift:
        break
assert lift, "no consistent 2I lift of the bed generators"
GQ, PERM2Q = lift
CLS_Q = torch.tensor(np.array([PERM2Q[p] for p in A.ELEMS]))  # [60,4], index = bed label
res["lift_generators"] = [q.round(4).tolist() for q in GQ]


def classify(u):  # sign-invariant oracle readout
    u = u / u.norm(dim=-1, keepdim=True)
    return (u @ CLS_Q.T).abs().argmax(-1)


# ---- hand-set one as-built fq Layer
layer = A.Layer(A.D, A.N_HEADS, "fq")
emb = torch.nn.Embedding(4, A.D)
with torch.no_grad():
    X = layer.ln1(emb.weight)                        # [4,d] what the layer sees per token
    Xa = torch.cat([X, torch.ones(4, 1)], 1)
    tgt = torch.tensor(np.array(GQ))                 # [4,4]
    Wb = torch.linalg.pinv(Xa) @ tgt                 # min-norm exact fit (4 points)
    layer.quat_head.weight.copy_(Wb[:-1].T); layer.quat_head.bias.copy_(Wb[-1])
    layer.qkv.weight.zero_(); layer.qkv.bias.zero_()  # Q=K=0 -> uniform causal attention
    d = A.D
    layer.qkv.weight[2 * d:2 * d + 4] = Wb[:-1].T     # V head0 block0 = generator quaternion
    layer.qkv.bias[2 * d:2 * d + 4] = Wb[-1]
    layer.o_proj.weight.copy_(torch.eye(d)); layer.o_proj.bias.zero_()
    fit_err = (qnormalize(layer.quat_head(X)) - tgt).abs().max().item()
res["quat_head_fit_err"] = fit_err

ev_tok, ev_lab = A.gen_split("eval", 0)
ev_x, ev_y = torch.tensor(ev_tok), torch.tensor(ev_lab)


def read(tag):
    with torch.no_grad():
        o = layer.attn(layer.ln1(emb(ev_x)))[..., 0:4]
    pred = classify(o)
    acc = {"all": float((pred == ev_y).double().mean())}
    for p in A.REPORT_POSITIONS:
        acc[str(p)] = float((pred[:, p - 1] == ev_y[:, p - 1]).double().mean())
    res[tag] = acc
    print(tag, acc)
    return acc


real_F = A.F
A.F = types.SimpleNamespace(scaled_dot_product_attention=lambda Q, K, V, is_causal: V[..., :1, :].expand_as(V))
t2a = read("T2a_anchor_forced")
A.F = real_F
t2b = read("T2b_as_built_uniform")

# ---- T2c: train the as-built arm exactly as the bed does (seed 0), inspect its generators
torch.set_default_dtype(torch.float32)
tr_tok, tr_lab = A.gen_split("train", 0)
ev_tok, ev_lab = A.gen_split("eval", 0)
r = None
torch.manual_seed(0)
m = A.A5Arm(A.D, A.N_HEADS, A.N_LAYERS, "fq")
opt = torch.optim.AdamW(m.parameters(), lr=A.LR, weight_decay=A.WEIGHT_DECAY)
trx, trY = torch.tensor(tr_tok), torch.tensor(tr_lab)
for _ in range(A.STEPS):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(m(trx).reshape(-1, 60), trY.reshape(-1))
    loss.backward(); opt.step()
m.eval()
with torch.no_grad():
    tr_acc = float((m(trx).argmax(-1) == trY).float().mean())
    ev_acc = float((m(torch.tensor(ev_tok)).argmax(-1) == torch.tensor(ev_lab)).float().mean())
    L0 = m.layers[0]
    q = qnormalize(L0.quat_head(L0.ln1(m.emb.weight))).double()  # tokens g0,g1,g0inv,g1inv
    one = torch.tensor([1., 0, 0, 0], dtype=torch.float64)
    pm = lambda z: min((z - one).norm().item(), (z + one).norm().item())
    inv_err = max(pm(qmul(q[0], q[2])), pm(qmul(q[1], q[3])))
    cube_err = max(pm(qmul(q[i], qmul(q[i], q[i]))) for i in range(4))
res["T2c_trained"] = {"final_train_loss": float(loss.detach()), "train_acc_all_pos": tr_acc, "eval_acc_all_pos": ev_acc,
                      "layer0_quats": q.numpy().round(4).tolist(), "inverse_pair_err": inv_err, "order3_err": cube_err}
print("T2c", res["T2c_trained"])
json.dump(res, open(os.path.join(HERE, "a5_handset.json"), "w"), indent=2)

assert t2a["all"] == 1.0, f"anchor-forced hand-set arm not exact: {t2a}"
print("GREEN T2a: fold + 2I lift + bed convention read every held-out running product (anchor forced)")
fails = []
if t2b["64"] < 0.9:
    fails.append(f"RED T2b: as-built SDPA, same hand-set lift, acc@64={t2b['64']:.4f} all={t2b['all']:.4f} (<0.9)")
if not (inv_err < 0.1 and cube_err < 0.1):
    fails.append(f"RED T2c: trained layer-0 generators are no A5 lift: inverse_pair_err={inv_err:.3f} order3_err={cube_err:.3f} (bar <0.1)")
for f in fails:
    print(f)
assert not fails, " | ".join(fails)
