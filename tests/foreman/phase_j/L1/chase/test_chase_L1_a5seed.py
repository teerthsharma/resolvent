"""GREEN route for A5-RERUN: with the split seeded by crc32 instead of hash(), two runs
under different PYTHONHASHSEED give identical per-seed losses and accuracies."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

def test_a5_stable_seed_reproduces():
    r1 = json.load(open(os.path.join(HERE, "beds_fixed", "r1.json")))["results"]
    r2 = json.load(open(os.path.join(HERE, "beds_fixed", "r2.json")))["results"]
    for k in r1:
        a = [(x["final_train_loss"], x["per_position_accuracy"]) for x in r1[k]]
        b = [(x["final_train_loss"], x["per_position_accuracy"]) for x in r2[k]]
        assert a == b, (k, a, b)
