"""Gate-0 shared fixtures.

THE COMPARATORS MOVED OUT, to `tests/gate0/helpers.py`. They lived here and were
reached as `from conftest import ...`, which is the form
`tests/chase/conftest.py` uses for `requires_triton` -- and which
`tests/loop/test_conftest_import_is_order_dependent.py` records as a defect
rather than a convention: 24 of 26 test directories carry a `conftest.py` and
only 2 carry an `__init__.py`, so the bare name `conftest` binds to whichever
directory collection order reached first. Import them as
`from tests.gate0.helpers import bitwise_diff, load_state`.

The re-export below is kept because `tests/gate0/test_g08_autopilot.py` names
this file as where `load_state` and `bitwise_diff` live. It is a compatibility
shim, not the address: new call sites should use `tests.gate0.helpers`.
"""
import os as _os

# CUBLAS_WORKSPACE_CONFIG MUST BE SET BEFORE CUDA INITIALISES, NOT BEFORE THE
# TEST RUNS. `torch.use_deterministic_algorithms(True)` requires it for cuBLAS
# reductions; set it after CUDA init and strict mode fails the FORWARD too,
# measured in `V17_R4_RETAKE_PRICE.md`. The symptom is order dependence, not a
# hard error: `test_bind_d_deterministic_mode_on_cuda` passes alone and FAILS in
# a full `tests/gate0` run, because by then another module has already touched
# the device. Measured on this box: without this line the suite reads
# `1 failed, 188 passed`; with `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before
# the process, `189 passed`. `setdefault` so an explicit export still wins.
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tests.gate0.helpers import bitwise_diff, load_state  # noqa: F401
