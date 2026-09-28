# Planted checks for p1.py: the metric must read 1.0 on an exact embedding and low on a random one.
import numpy as np
import p1


def preorder_coords(hop):
    n = hop.shape[0]
    kids = [list(np.nonzero(hop[u] == 1)[0]) for u in range(n)]
    def pre(rev):
        o, st, c = np.zeros(n, int), [0], 0
        while st:
            u = st.pop(); o[u] = c; c += 1
            st.extend(kids[u] if rev else kids[u][::-1])
        return o
    return np.stack([pre(False), pre(True)], 1)


def test_split_is_a_partition():
    lvl, hop = p1.world(0, True)
    S = p1.split(hop, 0)
    n = len(lvl)
    parts = [S["near"], S["far"], S["trneg"], S["honeg"]]
    assert (sum(p.astype(int) for p in parts) == 1 - np.eye(n, dtype=int)).all()
    assert (S["near"] | S["far"] == S["A"]).all()


def test_planted_order_embedding_reads_one_on_tree():
    lvl, hop = p1.world(1, False)
    S = p1.split(hop, 1)
    x = preorder_coords(hop)
    pred = ((x[:, None, :] < x[None, :, :]).all(-1))
    r = p1.score(pred, S)
    assert all(abs(v - 1.0) < 1e-12 for v in r.values()), r


def test_random_embedding_is_low():
    lvl, hop = p1.world(2, False)
    S = p1.split(hop, 2)
    x = np.random.default_rng(0).random((len(lvl), 2))
    r = p1.score((x[:, None, :] < x[None, :, :]).all(-1), S)
    assert r["far_cons"] < 0.5, r


if __name__ == "__main__":
    test_split_is_a_partition(); test_planted_order_embedding_reads_one_on_tree(); test_random_embedding_is_low()
    print("p1 planted checks: 3/3 pass")
