"""No tracked module outside the package may be importable as `resolvent`.

Until the restructure, tests/foreman/phase_k/K0/chase/resolvent.py was bare-imported as `resolvent`
by 17 Phase K files. Collected in one process with this suite, whichever of the two sys.modules
cached first shadowed the other (6 collection errors, `cannot import name 'standard_map' from
'resolvent' (.../K0/chase/resolvent.py)`). It is now k0_resolvent.py; this keeps the name free.
"""
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_no_tracked_module_outside_the_package_is_named_resolvent():
    r = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    files = [f for f in r.stdout.decode("utf-8", "replace").split("\0") if f]
    shadows = [f for f in files if not f.startswith("resolvent/")
               and (f.endswith("/resolvent.py") or f == "resolvent.py" or "/resolvent/__init__.py" in "/" + f)]
    assert shadows == []
