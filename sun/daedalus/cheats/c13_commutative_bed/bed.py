import numpy as np


def generate(n, seed, T=16):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, T))
    Y = np.cumsum(X * np.array([1, 2, 3, 5])[X] % 7, axis=1) % 7   # looks order-dependent, is not
    return X, Y.astype(np.int64), np.arange(T) >= 8
