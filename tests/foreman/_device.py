"""Device parametrization for every FOREMAN test.

A CUDA-only test cannot be reproduced without a GPU, and the CPU path is the
parity oracle for the GPU path. So every numeric test runs on both.
"""

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device available"),
    ),
]


def t(x, device="cpu"):
    """numpy/list -> float64 torch tensor on `device`."""
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def n(x):
    """torch tensor -> numpy, so existing numpy assertions keep working."""
    return x.detach().cpu().numpy()
