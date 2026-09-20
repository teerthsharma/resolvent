"""Must-fires for the fractal sort wing.

L-COUNT: this file collects EXACTLY 7 tests.  test_collected_count asserts it,
so a test silently lost to a rename or an import error turns the suite red
instead of green-and-empty.

L-SURFACE: the CLI's exit code is read from the process, never through a pipe.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ceqjepa import fractal_sort as fs

EXPECTED_TESTS = 7
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def world():
    return fs.build_world(n_train=3000, n_query=60)


def test_collected_count(pytestconfig):
    """L-COUNT -- the suite's own size is an assertion, not a summary line."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(Path(__file__))],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    n = sum(1 for ln in out.stdout.splitlines() if "::test_" in ln)
    assert n == EXPECTED_TESTS, f"collected {n}, expected {EXPECTED_TESTS}"


def test_l0_trie_is_brute_force_exactly(world):
    """MUST-FIRE.  A trie shipped without this is STRUCK."""
    acc = fs.run_arms(world, 0)
    assert np.array_equal(acc["trie"]["idx"], acc["ref_i"])
    assert acc["trie"]["work_mean"] == float(acc["N"])
    assert acc["trie"]["err_max"] == 0.0
    assert acc["trie"]["fallbacks"] == 0


def test_hash_and_trie_are_the_same_partition(world):
    """The counter's structural claim: the trie's leaves ARE the hash buckets."""
    acc = fs.run_arms(world, 6)
    assert np.array_equal(acc["hash"]["idx"], acc["trie"]["idx"])
    assert acc["hash"]["work_mean"] == acc["trie"]["work_mean"]


def test_lambda1_equals_minus_entropy_rate(world):
    """Holliday-Goldsmith-Glynn Prop. 1: the product's norm IS the likelihood."""
    assert abs(world.lam1 + world.h) < 1e-3, (world.lam1, world.h)


def test_two_chi_estimators_agree(world):
    """QR spectrum gap against the filter's measured forgetting rate."""
    rel = abs(world.chi_qr - world.chi_fit) / world.chi_qr
    assert rel < 0.05, (world.chi_qr, world.chi_fit)


def test_collage_bound_holds_on_the_planted_ifs():
    """MUST-FIRE, and it can fail: Sierpinski, s = 1/2 exactly."""
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8660254037844386]])
    maps = [(0.5 * np.eye(2), 0.5 * v) for v in V]
    A = fs.chaos_game(maps, 800, seed=11)
    B = fs.chaos_game(maps, 800, seed=12)
    c = max(float(np.linalg.norm(M, 2)) for M, _ in maps)
    assert c == pytest.approx(0.5)
    union = np.concatenate([A @ M.T + t[None, :] for M, t in maps], axis=0)
    bound = fs.hausdorff(A, union) / (1.0 - c)
    assert fs.hausdorff(A, B) <= bound


def test_cli_exits_zero(tmp_path):
    """L-SURFACE -- the exit code is asserted off the process, not a pipeline."""
    out = subprocess.run(
        [sys.executable, "-m", "ceqjepa.fractal_sort", "--test", "trie",
         "--n-train", "2000", "--n-query", "40", "--repeats", "1"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    assert "MUST-FIRE FIRED" in out.stdout
