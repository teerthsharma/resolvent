"""Shared fixtures for W2.

The measurement protocol is deliberately imported from tests/cameron/perron.py
rather than reimplemented: W2's whole claim is a comparison against the numbers
Cameron measured (attention ratio 1565.111), and a fresh protocol would produce
a number that cannot be compared to it.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "tests" / "cameron")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    return torch.device(request.param)
