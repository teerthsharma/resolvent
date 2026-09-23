# Foreman K1 bars on a TRAINED ALiBi twin given a long budget (candidate (c)). Written 2026-09-23 03:01 IST, BEFORE
# twin.py exists or runs. Never edited after its RED. Reads twin_L4.json, twin_L5.json, twin_gen.json. Exit 1 on any fail.
#
# Twin: the registered (a_L) block, imported read-only from K0/wilson/train_ladder.py (Block + AlibiAttention: fixed
# ALiBi slopes, no position table, pre-LN, GELU MLP 4d), d = 128 (R0 width, 2 heads of 64), L = 4 (the R0 twin) and
# L = 5 (control). Input per token: E[id] + P(E[pid]), one shared learned table over V + 1 = 32,769 ids (NULL = V).
# Readout: logits over the bed's own tokens, <O(LN(h_t)), E[id_j]> for j <= t (content keys only: a vocabulary
# restriction, not an extra hop); prediction = id of the argmax token. Loss: cross-entropy to the root's position.
# Train stream: bed_k's train distribution (n = 1024, 16 lanes, depth capped at 32), regenerated on the GPU by a
# vectorised copy of make_bed, torch seed 101 (L4) / 102 (L5). bf16 autocast, AdamW.
# Evaluation (float32 weights, bf16 autocast): 64 train-distribution beds bed_k.make_train(default_rng([91, k])) and
# 8 test beds bed_k.make_test(default_rng([92, 4096, k]), 4096), at 1/8, 1/4, 1/2 and all of the budget.
#
# foreman.k1.twin_machinery : the vectorised generator equals bed_k.make_bed (parent, depth, root) on 16 identical
#     lane draws, capped at 32 and uncapped (0 mismatches).
# foreman.k1.twin_budget : each arm trained on >= 0.28e9 tokens (twice the R0 rung's 0.14B language-model budget).
# foreman.k1.twin_gate : the L=4 twin learned the part doubling can reach: accuracy on depth <= 16 >= 0.9 at the end.
# foreman.k1.twin_control : the L=5 twin reaches beyond depth 16 on the same stream: accuracy on 17 <= depth <= 32 >= 0.9.
# foreman.k1.twin_wall : the L=4 twin stays at the wall: accuracy on 17 <= depth <= 32 <= 0.2 at the end (credited
#     doubling on this distribution is 0.095, the oracle C2 line 0.380). VOID -> FAIL if twin_gate fails.
# foreman.k1.twin_plateau : the L=4 wall is not a budget artifact: acc(17..32) at the end minus at 1/4 <= 0.05.
# foreman.k1.twin_kill_line : the L=4 twin on n = 4096 test beds, accuracy on depth > 16 <= 0.5 (the registered kill).
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
suffix = sys.argv[1] if len(sys.argv) > 1 else ""
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


G = json.loads((HERE / f"twin_gen{suffix}.json").read_text())
T4 = json.loads((HERE / f"twin_L4{suffix}.json").read_text())
T5 = json.loads((HERE / f"twin_L5{suffix}.json").read_text())
check("foreman.k1.twin_machinery generator == bed_k.make_bed", G["mismatches"] == 0 and G["draws"] >= 16, G)
for T in (T4, T5):
    check(f"foreman.k1.twin_budget L={T['L']} tokens >= 0.28e9", T["tokens"] >= 0.28e9, T["tokens"])
f4, f5 = T4["ckpts"][-1], T5["ckpts"][-1]
gate = f4["acc_0_16"] >= 0.9
check("foreman.k1.twin_gate L=4 acc(depth<=16) >= 0.9", gate, f4)
check("foreman.k1.twin_control L=5 acc(17..32) >= 0.9", f5["acc_17_32"] >= 0.9, f5)
check("foreman.k1.twin_wall L=4 acc(17..32) <= 0.2", gate and f4["acc_17_32"] <= 0.2,
      ("" if gate else "VOID (gate failed) ") + str(f4["acc_17_32"]))
q = [c for c in T4["ckpts"] if c["frac"] == 0.25][0]
check("foreman.k1.twin_plateau L=4 acc(17..32) end - quarter <= 0.05", f4["acc_17_32"] - q["acc_17_32"] <= 0.05,
      (q["acc_17_32"], f4["acc_17_32"]))
check("foreman.k1.twin_kill_line L=4 n=4096 acc(depth>16) <= 0.5", f4["test4096_beyond16"] <= 0.5, f4["test4096_beyond16"])
print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
