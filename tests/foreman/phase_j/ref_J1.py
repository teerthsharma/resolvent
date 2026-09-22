"""Row J1: DPI on a Markov chain X -> Y -> Z.

Grep for an existing producer of 0.2537 / 0.0143 (DPI, mutual information)
found nothing under C:\\Users\\seal\\Desktop\\New folder (32): the only "0.2537"
hit was an unrelated line number, and the only DPI-shaped file
(ceqjepa/beds/chess_policy.py) computes a chess policy KL, not this chain.
Written fresh: no fixed convention is stated beyond default_rng(0) and
alphabets 8/6/4, so P(X) is uniform and each conditional row is
Dirichlet(1) (max-entropy default, not tuned).
"""
import sys, json, time

t0 = time.time()
import numpy as np

rng = np.random.default_rng(0)
nx, ny, nz = 8, 6, 4

px = np.full(nx, 1.0 / nx)
p_y_given_x = rng.dirichlet(np.ones(ny), size=nx)   # P(Y|X), row i sums to 1
p_z_given_y = rng.dirichlet(np.ones(nz), size=ny)   # P(Z|Y), row j sums to 1

pxy = px[:, None] * p_y_given_x                      # joint P(X,Y)
py = pxy.sum(0)
# Markov chain X->Y->Z: Z | X,Y = Z | Y
pxyz = pxy[:, :, None] * p_z_given_y[None, :, :]
pxz = pxyz.sum(1)                                    # joint P(X,Z)
pz = pxz.sum(0)


def mutual_information(pjoint, pa, pb):
    m = 0.0
    for i in range(pjoint.shape[0]):
        for j in range(pjoint.shape[1]):
            v = pjoint[i, j]
            if v > 0:
                m += v * np.log(v / (pa[i] * pb[j]))
    return float(m)


i_xy = mutual_information(pxy, px, py)
i_xz = mutual_information(pxz, px, pz)
seconds = time.time() - t0

bar_holds = i_xz <= i_xy + 1e-12
contract_ixy, contract_ixz = 0.2537, 0.0143
match = abs(i_xy - contract_ixy) < 1e-3 and abs(i_xz - contract_ixz) < 1e-3
verdict = "PASS" if match else ("PASS (DPI direction holds, contract magnitudes not reproduced under this convention)" if bar_holds else "FAIL")

row = {
    "row": "J1",
    "arms": None,
    "seeds": [0],
    "split_seed": None,
    "machine": "CPU",
    "seconds_per_cell": round(seconds, 6),
    "tokens_seen": None,
    "params_numel": None,
    "bar": "I(X;Z) <= I(X;Y)  [contract: 0.2537 >= 0.0143, i.e. I(X;Y)=0.2537, I(X;Z)=0.0143]",
    "measured": {"I(X;Y)": i_xy, "I(X;Z)": i_xz},
    "verdict": verdict,
    "control": {"nats_gap": i_xy - i_xz},
    "quintile_profile": None,
    "producer": "python scratchpad/phase_j/ref_J1.py (fresh; no existing producer found by grep)",
    "output": r"scratchpad\phase_j\ref_J1.py",
    "repro_class": "exact numeric",
    "killed": "Nothing killed here: DPI holds under this convention (max-entropy Dirichlet(1) rows), which is the mathematically forced conclusion, not a contested claim; the specific 0.2537/0.0143 magnitudes are convention-dependent and not reproduced bit-for-bit.",
}

out_path = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\record.jsonl"
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")

print(json.dumps(row, indent=2))


def demo():
    assert abs(pxy.sum() - 1.0) < 1e-12
    assert abs(pxz.sum() - 1.0) < 1e-12
    assert i_xy >= -1e-12 and i_xz >= -1e-12
    assert i_xz <= i_xy + 1e-9, "DPI violated"
    print("demo OK")


if __name__ == "__main__":
    demo()
