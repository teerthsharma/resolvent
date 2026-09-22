# K0_logit_law_unidentified. Bars written 2026-09-23 01:58, before any train-bed resolvent number existed.
# Claim: training at one length (n = 1024, depth <= 32) does not pin the logit law the test needs.
# The hand-set one-layer softmax resolvent (gamma 0.999) at s_train = the smallest grid s scoring >= 0.99 on
# the K1 TRAIN bed (8 beds, worst bed) scores < 0.5 on the 16k TEST bed (8 beds, median), and
# s_99(16k) - s_train > 2.0. So an arm that learns "just enough" scale in training fails at test; an SSMax
# coefficient a in s = a ln n + b is not identifiable from a single training n.
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from k1_floors import s_needed, GRID, train_beds, test_beds
from bed_k import resolvent_rec

s_tr = max(s_needed(b, 0.99) for b in train_beds())
tb = test_beds(16384)
acc16 = float(np.median([resolvent_rec(b["parent"], b["root"], np.array([s_tr]), 0.999)[0] for b in tb]))
s99 = float(np.median([s_needed(b, 0.99) for b in tb]))
print(f"s_train (worst of 8 train beds, acc>=0.99) = {s_tr:.2f}; 16k median acc at s_train = {acc16:.4f}; "
      f"16k median s_99 = {s99:.2f}; implied a = {(s99 - s_tr) / np.log(16):.3f} per ln n")
json.dump({"s_train": s_tr, "acc16_at_s_train": acc16, "s99_16k": s99, "a_implied": (s99 - s_tr) / np.log(16)},
          open("logit_law.json", "w"), indent=1)
ok = acc16 < 0.5 and s99 - s_tr > 2.0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
