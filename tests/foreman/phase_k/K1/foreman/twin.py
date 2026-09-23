# Foreman K1 candidate (c): the (a_L) block trained long on bed_k's train distribution. Bars: test_twin.py.
#   python twin.py --gen-check          CPU: vectorised generator vs bed_k.make_bed -> twin_gen.json
#   python twin.py --L 4 --steps 10000  GPU job under the house lock -> twin_L4.json, twin_L4.pt
import argparse, datetime, json, math, os, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.dont_write_bytecode = True
K0 = "C:/Users/seal/Desktop/New folder (32)/tests/foreman/phase_k/K0"
sys.path[:0] = [K0 + "/cameron", K0 + "/wilson"]
import bed_k                                               # noqa: E402
from train_ladder import Block, AlibiAttention, get_lr     # noqa: E402

HERE = Path(__file__).parent
SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
LOCK = SP + "/phase_k/K1/gpu.lock.d"
BOARD = "C:/Users/seal/Desktop/New folder (32)/house-events.jsonl"
V, NULL = bed_k.V, bed_k.NULL


def gen(chain, cap):
    """bed_k.make_bed, vectorised over a batch of lane draws (B, n): parent (-1 for roots), depth, root (positions)."""
    B, n = chain.shape
    sc, order = torch.sort(chain, dim=1, stable=True)
    k = torch.arange(n, device=chain.device).expand(B, n)
    start = torch.ones_like(sc, dtype=torch.bool)
    start[:, 1:] = sc[:, 1:] != sc[:, :-1]
    within = k - torch.cummax(torch.where(start, k, torch.zeros_like(k)), dim=1).values
    dep = within % (cap + 1) if cap else within
    root_s = torch.gather(order, 1, k - dep)
    par_s = torch.where(dep > 0, torch.gather(order, 1, (k - 1).clamp(min=0)), torch.full_like(k, -1))
    out = [torch.empty_like(x).scatter_(1, order, x) for x in (par_s, dep, root_s)]
    return out


def gen_check():
    bad, draws = 0, 0
    for s, (n, cap) in enumerate([(1024, 32)] * 8 + [(4096, None)] * 8):
        chain = np.random.default_rng([100, s]).integers(0, 16, n)
        ref = bed_k.make_bed(n, 16, np.random.default_rng([100, s]), cap=cap)
        got = gen(torch.from_numpy(chain)[None], cap)
        bad += sum(int((g[0].numpy() != r).sum()) for g, r in zip(got, ref))
        draws += 1
    (HERE / "twin_gen.json").write_text(json.dumps({"mismatches": bad, "draws": draws}))
    return bad, draws


class Twin(nn.Module):
    def __init__(self, L, d=128, ctx=1024):
        super().__init__()
        self.E = nn.Embedding(V + 1, d)
        self.P = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList([Block(d, d // 64, i, L, ctx, AlibiAttention) for i in range(L)])
        self.ln = nn.LayerNorm(d)
        self.O = nn.Linear(d, d, bias=False)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=0.02)
        for blk in self.blocks:
            nn.init.normal_(blk.mlp[2].weight, std=0.02 / math.sqrt(2 * L))
            nn.init.normal_(blk.attn.out.weight, std=0.02 / math.sqrt(2 * L))
        nn.init.normal_(self.E.weight, std=1.0)
        nn.init.orthogonal_(self.P.weight)   # CPU pilot (pilot_cpu2_L4.log): pid readable from step 0

    def forward(self, ids, pids):
        e = self.E(ids)
        x = e + self.P(self.E(pids))
        for blk in self.blocks:
            x = blk(x)
        lg = (self.O(self.ln(x)) @ e.transpose(1, 2)).float()     # content keys only: vocabulary = the bed's ids
        n = ids.shape[1]
        return lg.masked_fill(torch.ones(n, n, dtype=torch.bool, device=ids.device).triu(1), float("-inf"))


def batch(g, B, n, cap, dev):
    chain = torch.randint(0, 16, (B, n), generator=g, device=dev)
    parent, depth, root = gen(chain, cap)
    ids = torch.argsort(torch.rand(B, V, generator=g, device=dev), dim=1)[:, :n]
    pids = torch.where(parent >= 0, torch.gather(ids, 1, parent.clamp(min=0)), torch.full_like(ids, NULL))
    return ids, pids, depth, root


@torch.no_grad()
def evaluate(model, dev):
    model.eval()
    res = {}
    tr = [bed_k.make_train(np.random.default_rng([91, k])) for k in range(64)]
    ok, dep = [], []
    for i in range(0, 64, 16):
        bs = tr[i:i + 16]
        ids = torch.tensor(np.stack([b["ids"] for b in bs]), device=dev)
        pids = torch.tensor(np.stack([b["pids"] for b in bs]), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(ids, pids).argmax(-1)
        ok.append((torch.gather(ids, 1, pred).cpu().numpy() == np.stack([b["target"] for b in bs])).ravel())
        dep.append(np.stack([b["depth"] for b in bs]).ravel())
    ok, dep = np.concatenate(ok), np.concatenate(dep)
    res.update(acc_all=float(ok.mean()), acc_0_16=float(ok[dep <= 16].mean()), acc_17_32=float(ok[dep >= 17].mean()))
    ok, dep = [], []
    for k in range(8):
        b = bed_k.make_test(np.random.default_rng([92, 4096, k]), 4096)
        ids = torch.tensor(b["ids"][None], device=dev)
        pids = torch.tensor(b["pids"][None], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(ids, pids).argmax(-1)
        ok.append(torch.gather(ids, 1, pred).cpu().numpy()[0] == b["target"])
        dep.append(b["depth"])
    ok, dep = np.concatenate(ok), np.concatenate(dep)
    res.update(test4096_all=float(ok.mean()), test4096_beyond16=float(ok[dep > 16].mean()),
               test4096_beyond128=float(ok[dep > 128].mean()), test4096_0_16=float(ok[dep <= 16].mean()))
    model.train()
    return res


def acquire(tag):
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(LOCK)
                txt = open(LOCK + "/owner.txt").read()
                pid = int(txt.split("pid=")[1].split()[0])
                import psutil
                if age > 3 * 3600 and not psutil.pid_exists(pid):
                    shutil.rmtree(LOCK, ignore_errors=True)
                    with open(BOARD, "a") as f:
                        f.write(json.dumps({"t": "finding", "agent": "Foreman", "text": "removed stale gpu.lock.d (> 3 h, pid dead): " + txt.strip()}) + "\n")
                    continue
            except Exception:
                pass
            time.sleep(60)
    with open(LOCK + "/owner.txt", "w") as f:
        f.write(f"seat=Foreman pid={os.getpid()} start={datetime.datetime.now().isoformat(timespec='seconds')} tag={tag}\n")


SEED = {4: 101, 5: 102}          # as registered in test_twin.py's header


def train(L, steps, B=32, n=1024, lr=1e-3, warmup=500, budget_s=40 * 60, dev="cuda", seed=None):
    dev = torch.device(dev)
    seed = SEED[L] if seed is None else seed
    torch.manual_seed(seed)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    model = Twin(L).to(dev)
    decay = [p for nm, p in model.named_parameters() if p.dim() == 2 and not nm.startswith("E.")]
    rest = [p for nm, p in model.named_parameters() if not (p.dim() == 2 and not nm.startswith("E."))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    marks = {max(1, steps // 8): 0.125, steps // 4: 0.25, steps // 2: 0.5, steps: 1.0}
    ckpts, losses, run = [], [], 0.0
    t0 = time.time()
    done = 0
    for step in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = get_lr(step, steps, warmup, lr)
        ids, pids, depth, root = batch(g, B, n, 32, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, pids)
        loss = F.cross_entropy(logits.view(-1, n), root.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += loss.item() if step % 50 == 0 else 0.0
        if step % 500 == 0:
            losses.append([step, run / 10, round(time.time() - t0, 1)])
            print(f"L{L} step {step} loss {run / 10:.4f} {time.time() - t0:.0f}s", flush=True)
            run = 0.0
        done = step
        over = time.time() - t0 > budget_s
        if step in marks or over:
            c = evaluate(model, dev)
            c.update(step=step, frac=marks.get(step, step / steps), tokens=step * B * n, wall_s=round(time.time() - t0, 1))
            ckpts.append(c)
            print(json.dumps(c), flush=True)
        if over:
            break
    out = {"L": L, "d": 128, "heads": 2, "steps_planned": steps, "steps": done, "batch": B, "n": n, "tokens": done * B * n,
           "lr": lr, "warmup": warmup, "seed": seed, "params": sum(p.numel() for p in model.parameters()),
           "stopped_by_budget": done < steps, "losses": losses, "ckpts": ckpts, "gpu": torch.cuda.get_device_name(0)}
    torch.save(model.state_dict(), HERE / f"twin_L{L}.pt")
    (HERE / f"twin_L{L}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-check", action="store_true")
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.gen_check:
        print(gen_check())
    else:
        acquire(f"foreman.k1.twin_L{a.L}")
        try:
            train(a.L, a.steps)
        finally:
            shutil.rmtree(LOCK, ignore_errors=True)
