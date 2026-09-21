"""Opus checker, (D): the missing number. chase_cogs.py died before it scored
anything and its replacement (cogs_control.py) scores the CONTROL only, so the
gated arm has still never been put against the bar. This does that.

Identical protocol to ceq/capability.py::run_split at BUDGET["cogs"]:
steps=3000, bs=32, lr=3e-4, gate on `test` max_eval=256, generalization on
`gen` max_eval=512, max_new=192, fp32, cuda. The ONLY difference from the
shipped runner is kind="smprime", admitted past ceq/lm.py:66 through the
scratchpad copy lm_patched.py, monkeypatched into ceq.harness in-process.
ceq/ is not written to.

Bar (pre-registered, ceq/capability.py:39-40): 0.02930 + 0.02 = 0.04930 on
COGS generalization, gated on the control clearing RESOLUTION_FLOOR 0.20
in-distribution.
"""
import io, json, pathlib, sys, time
REPO = pathlib.Path(r"C:\Users\seal\Desktop\New folder (32)")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
import torch
import lm_patched
from ceq import harness
harness.lm = lm_patched

BOARD = pathlib.Path(r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl")
OUT = HERE / "opus_smprime_bar.jsonl"
STEPS, BS, LR = 3000, 32, 3e-4
GATE_EVAL, GEN_EVAL, MAX_NEW = 256, 512, 192
BAR = 0.02930 + 0.02

def board(**e):
    with io.open(BOARD, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(agent="Chase", ts=time.strftime("%Y-%m-%dT%H:%M:%S"), **e)) + "\n")

def run(kind, seed, device):
    t0 = time.time()
    m = harness.train_model(kind, "cogs", steps=STEPS, device=device, seed=seed,
                            bs=BS, lr=LR)
    t_train = time.time() - t0
    gate = harness.score(m, "cogs", "test", device=device, max_eval=GATE_EVAL,
                         seed=seed, max_new=MAX_NEW)
    gen = harness.score(m, "cogs", "gen", device=device, max_eval=GEN_EVAL,
                        seed=seed, max_new=MAX_NEW)
    row = dict(event="opus_bar_row", row="cogs_smprime_bar", kind=kind, seed=seed,
               device=str(device), dtype="fp32", steps=STEPS, bs=BS, lr=LR,
               seq=m.seq, vocab=m.vocab, n_params=m.n_params(),
               in_distribution=gate["exact_match"],
               n_solved_in_distribution=gate["n_solved"], n_eval_in=gate["n_eval"],
               generalization=gen["exact_match"],
               n_solved_generalization=gen["n_solved"], n_eval_gen=gen["n_eval"],
               train_loss=m.train_loss, train_seconds=round(t_train, 1),
               bar=BAR, clears_bar=gen["exact_match"] >= BAR,
               command=f"python opus_smprime_bar.py  # kind={kind} seed={seed} steps={STEPS}")
    with io.open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    board(**row)
    print(json.dumps(row), flush=True)
    del m
    torch.cuda.empty_cache()
    return row

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")]
    board(event="opus_bar_start", row="cogs_smprime_bar", seeds=seeds,
          steps=STEPS, bar=BAR, device=str(dev),
          note="gated arm scored against the pre-registered bar for the first time")
    for s in seeds:
        run("smprime", s, dev)
    board(event="opus_bar_end", row="cogs_smprime_bar")
