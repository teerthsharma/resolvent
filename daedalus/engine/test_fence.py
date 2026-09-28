"""The sandbox read fence is the repository root, wherever daedalus/ sits.

verifier.REPO is passed to sandbox_runner.py as the job's `repo` and is the fence: candidate code may not read
anything under it outside its own directory. It was computed by counting parents from daedalus/, which was correct
only while daedalus/ sat exactly two levels down, as it did before the restructure. One level down, the count lands on the directory
ABOVE the repository; two moves the other way and the fence shrinks to a subtree, and candidates can read the
Phase J bed truth under tests/. The fence must be found, not counted.

    python -m pytest daedalus/engine/test_fence.py -q
"""
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # daedalus/engine/test_fence.py -> repository root (intended layout)
sys.path.insert(0, HERE)

import beds  # noqa: E402
import verifier as V  # noqa: E402

SEALED_FILE = os.path.join(ROOT, "daedalus", "sealed", "a5_T16_fixed_eval.npz")
BED_TRUTH = os.path.join(ROOT, "tests", "foreman", "phase_j", "a5_bed.py")
nc = lambda p: os.path.normcase(os.path.abspath(p))


def test_the_fence_is_the_repository_root():
    assert os.path.isfile(os.path.join(ROOT, "pytest.ini"))
    assert nc(V.REPO) == nc(ROOT), (V.REPO, ROOT)
    assert nc(beds.REPO) == nc(ROOT), (beds.REPO, ROOT)
    assert os.path.isfile(os.path.join(beds.PHASE_J, "a5_bed.py"))


def test_a_candidate_cannot_read_sealed_draws_or_bed_truth():
    assert os.path.isfile(SEALED_FILE) and os.path.isfile(BED_TRUTH)
    src = ("got = []\n"
           f"for p in ({SEALED_FILE!r}, {BED_TRUTH!r}):\n"
           "    try:\n        got.append(len(open(p, 'rb').read(1)))\n"
           "    except PermissionError:\n        got.append(-1)\n"
           "assert got == [-1, -1], got\n")
    with tempfile.TemporaryDirectory(prefix="fence_cand_") as cdir:
        with open(os.path.join(cdir, "candidate.py"), "w") as f:
            f.write(src)
        out, before, after = V._run_sandbox(cdir, {}, {"device": "cpu", "mode": "none", "spec": {}, "probe_X": None,
                                                       "seeds": []}, timeout=600)
    if "infra" in out:
        pytest.skip("sandbox runner died: " + str(out["infra"])[:200])
    denied = " ".join(out["violations"])
    assert nc(SEALED_FILE) in os.path.normcase(denied), out["violations"]
    assert nc(BED_TRUTH) in os.path.normcase(denied), out["violations"]
    assert "AssertionError" not in out.get("error", ""), out.get("error")
    assert before == after
