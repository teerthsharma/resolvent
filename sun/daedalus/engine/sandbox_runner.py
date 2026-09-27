"""Sandboxed candidate execution (L-SEALED-TRUTH). Run only by verifier.py:

    python -I -B sandbox_runner.py <job.pt> <out.pt>

The job holds train (X, Y) per seed and eval/probe X only -- never eval labels,
so a candidate cannot read truth from memory. Before any candidate code is
imported an audit hook is installed that records and denies: any write outside
the job's scratch dir, any read under the repository other than the candidate's
own directory (sealed/, engine/, registry/, stored results), sockets,
subprocesses, os.system/exec/spawn, ctypes. The parent additionally hashes the
protected trees before and after, and scores every number itself.

ponytail: an in-process audit hook is not a hard boundary against code that
patches the runner's memory; the hard guarantees are (a) no eval labels in this
process, (b) the parent's before/after hashes. Upgrade to an OS sandbox
(AppContainer / job object / container) before any untrusted LLM proposer runs.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

job_path, out_path = sys.argv[1], sys.argv[2]
JOB = torch.load(job_path, weights_only=True)
CAND = os.path.normcase(os.path.abspath(JOB["cand_dir"]))
SCRATCH = os.path.normcase(os.path.dirname(os.path.abspath(out_path)))
REPO = os.path.normcase(os.path.abspath(JOB["repo"]))
DEV = JOB["device"]
def _warm():
    """Trigger torch's lazy imports (CUDA DLLs; triton via the optimizer, which dlopens kernel32) before the hook."""
    m = torch.nn.GRU(4, 4, batch_first=True).to(DEV)
    F.cross_entropy(m(torch.randn(2, 3, 4, device=DEV))[0].reshape(-1, 4), torch.zeros(6, dtype=torch.long, device=DEV)).backward()
    torch.optim.AdamW(m.parameters()).step()
    torch.nn.utils.clip_grad_norm_(list(m.parameters()), 1.0)


_warm()
VIOL = []
_log = open(os.path.join(SCRATCH, "violations.log"), "a", buffering=1)

_BLOCK_PREFIX = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
                 "os.startfile", "ctypes.", "winreg.", "urllib.Request", "http.client.connect")
_MUTATE = ("os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate",
           "os.link", "os.symlink", "shutil.")


def _under(p, root):
    return p == root or p.startswith(root + os.sep)


def _deny(what):
    VIOL.append(what)
    _log.write(what + "\n")
    raise PermissionError("daedalus sandbox: " + what)


def _hook(event, args):
    if event == "open":
        path, mode, flags = args[0], args[1], args[2]
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        p = os.path.normcase(os.path.abspath(os.fsdecode(path)))
        writing = (mode is not None and any(c in str(mode) for c in "wax+")) or bool((flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if writing and not _under(p, SCRATCH):
            _deny(f"write {p}")
        if _under(p, REPO) and not _under(p, CAND) and not _under(p, SCRATCH):
            _deny(f"read {p}")
    elif event.startswith(_BLOCK_PREFIX):
        _deny(f"{event}")
    elif event.startswith(_MUTATE):
        p = os.path.normcase(os.path.abspath(os.fsdecode(args[0]))) if args and isinstance(args[0], (str, bytes, os.PathLike)) else ""
        if not _under(p, SCRATCH):
            _deny(f"{event} {p}")


sys.dont_write_bytecode = True
sys.addaudithook(_hook)

RES = {"violations": VIOL, "error": ""}
try:
    import importlib.util
    sys.path.insert(0, JOB["cand_dir"])
    spec = importlib.util.spec_from_file_location("candidate", os.path.join(JOB["cand_dir"], JOB["entry"]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if JOB["mode"] == "bed":
        tr = mod.generate(JOB["n_train"], JOB["seeds"][0])
        ev = mod.generate(JOB["n_eval"], JOB["seeds"][1])
        RES.update(Xtr=torch.as_tensor(np.asarray(tr[0])), Ytr=torch.as_tensor(np.asarray(tr[1])),
                   Xev=torch.as_tensor(np.asarray(ev[0])), Yev=torch.as_tensor(np.asarray(ev[1])),
                   band=torch.as_tensor(np.asarray(tr[2], dtype=bool)))
    else:
        spec_d = dict(JOB["spec"])
        probe = JOB["probe_X"].to(DEV)
        runs = []
        for k, seed in enumerate(JOB["seeds"]):
            torch.manual_seed(seed)
            model = mod.build(spec_d).to(DEV)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_hidden = sum(p.numel() for p in model.parameters() if not p.requires_grad) + sum(b.numel() for b in model.buffers())
            if k == 0:
                with torch.no_grad():
                    out0 = model(probe)
                RES["probe_shape"] = torch.tensor(list(out0.shape))
                RES["n_params"] = n_params
            Xtr, Ytr = JOB["train_X"][k].to(DEV), JOB["train_Y"][k].to(DEV)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=JOB["lr"]) if params else None
            g = torch.Generator(device="cpu").manual_seed(seed)
            losses = []
            model.train()
            for step in range(JOB["steps"]):
                idx = torch.randint(0, len(Xtr), (JOB["batch"],), generator=g).to(DEV)
                logits = model(Xtr[idx])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Ytr[idx].reshape(-1))
                if opt is not None:
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                if step % 50 == 0 or step == JOB["steps"] - 1:
                    losses.append(float(loss))
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(x.to(DEV)).argmax(-1).cpu() for x in JOB["eval_X"][k].split(1024)])
                # V1 causality instrument: resample the future, the past must not move
                base = model(probe)
                gp = torch.Generator(device="cpu").manual_seed(seed + 99)
                causal_dev = 0.0
                T = probe.shape[1]
                for t0 in (T // 4, T // 2, 3 * T // 4):
                    pert = probe.clone()
                    pert[:, t0 + 1:] = torch.randint(0, JOB["spec"]["vocab"], pert[:, t0 + 1:].shape, generator=gp).to(DEV)
                    causal_dev = max(causal_dev, float((model(pert)[:, : t0 + 1] - base[:, : t0 + 1]).abs().max()))
            run = {"seed": seed, "pred": pred, "losses": torch.tensor(losses), "causal_dev": causal_dev,
                   "n_params": n_params, "n_hidden": n_hidden}
            if hasattr(mod, "mixing_weights"):
                with torch.no_grad():
                    run["mixing_W"] = mod.mixing_weights(model, probe).detach().double().cpu()
            if hasattr(mod, "resolvent_params"):
                with torch.no_grad():
                    W, gamma = mod.resolvent_params(model, probe)
                run["resolvent_W"] = W.detach().double().cpu()
                run["resolvent_gamma"] = float(gamma)
            runs.append(run)
        RES["runs"] = runs
except PermissionError as e:
    RES["error"] = "PermissionError: " + str(e)
except Exception as e:  # candidate crash is a V0 failure, reported not hidden
    RES["error"] = f"{type(e).__name__}: {e}"

RES["violations"] = list(VIOL)
torch.save(RES, out_path)
