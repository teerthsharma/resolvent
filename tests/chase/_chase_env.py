"""Device/CUDA/Triton probes and subprocess isolation, shared by tests/chase.

Route (a) of tests/loop/test_conftest_import_is_order_dependent.py:33-43.
Extracted out of conftest.py because `conftest` is a bare module name claimed
by 26 test directories and resolves by whichever one pytest's collector
inserted onto `sys.path` first -- a subset run can bind the wrong directory's
conftest and fail an import that a full-tree run never sees. `_chase_env` is
claimed only by this directory, so it cannot be shadowed. conftest.py imports
these back for its own fixtures; test files that used to write
`from conftest import ...` import from here directly instead.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch


def _has_cuda():
    """`is_available()` alone is not enough.

    With CUDA_VISIBLE_DEVICES="" this torch build still reports
    `is_available() == True` while `device_count() == 0`, and the next call to
    `get_device_capability(0)` raises "Invalid device id" -- from conftest, at
    import time, which takes the WHOLE directory down before a single test runs.
    A CPU-only machine is the common case for a reader reproducing this, so the
    probe has to survive it.
    """
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:  # noqa: BLE001
        return False


HAS_CUDA = _has_cuda()
DEVICES = ["cpu"] + (["cuda"] if HAS_CUDA else [])


def has_triton():
    if not HAS_CUDA:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    try:
        return torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:  # noqa: BLE001
        return False


HAS_TRITON = has_triton()

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
requires_triton = pytest.mark.skipif(
    not HAS_TRITON, reason="Triton needs CUDA with compute capability >= 8.0"
)


def run_isolated(source, timeout=180):
    """Run `source` in a fresh interpreter. Returns (returncode, stdout, stderr).

    For probes that can trigger a CUDA illegal memory access. An IMA poisons the
    context for the whole process, so the only way to assert on one without
    destroying the rest of the run is to give it its own process.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    here = os.path.dirname(os.path.abspath(__file__))
    prelude = f"import sys; sys.path.insert(0, r{root!r}); sys.path.insert(0, r{here!r})\n"
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    proc = subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(source)],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr
