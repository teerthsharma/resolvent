"""R1 -- "settle, don't glance".

REQUIREMENTS.md R1 states the falsifier as:

    "iterate on held input. Output must change meaningfully across early
     iterations and stop changing at convergence. If pass 1 already equals the
     settled state, R1 is dead weight -- delete it."

and the hazard, three lines later, as:

    "if the update is affine, the fixed point has a closed form and the
     iteration computes nothing (residual 1.05e-15)."

Those two paragraphs contradict each other, and this file measures the
contradiction rather than arguing it. Any gamma-contraction, affine included,
satisfies ||z_1 - z*|| ~ gamma/(1-gamma) * ||b||. So the stated falsifier is
passed with room to spare by exactly the construction the hazard note kills.
The falsifier does not falsify.

The doc's own replacement -- "best affine fit to the learned update at the
settled state" -- has its own hole, and it is measurable: a max-plus operator
is piecewise affine, and z* sits in the interior of a linearity cell. Probe
below the cell radius and the probe reports "affine" for a non-affine operator.
The radius is not a free parameter, and REQUIREMENTS.md does not name one.

Two further measurements decide R1 on cost rather than on rhetoric:
the combinatorial work the iteration does (distinct greedy policies visited)
against the numerical work it does (iterations), and value iteration against
the direct method for the same fixed point.
"""

import time

import torch
import pytest

from _device import DEVICES, HAS_CUDA
from _ceq import (bellman, greedy_policy, maxplus_star, policy_iteration,
                  discounted_resolvent, affine_fit_residual, random_instance,
                  random_row_stochastic)


def _affine_control(n, gamma, seed, device):
    P = random_row_stochastic(n, seed=seed, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    b = torch.randn(n, generator=g, dtype=torch.float64).to(device)
    f = lambda z: gamma * (P @ z) + b
    z_star = discounted_resolvent(P, b, gamma)
    return f, z_star, P, b


def _rel_gap(a, b):
    return float((a - b).norm() / max(float(b.norm()), 1e-300))


# ==========================================================================
# CLAIM AS WRITTEN -- RED
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_the_stated_R1_falsifier_rejects_the_affine_update_it_is_meant_to_kill(device):
    """R1's falsifier asks only that pass 1 differ from the settled state. Run
    it against the affine update whose fixed point is np.linalg.solve -- the
    case R1's own hazard note declares dead weight. If the falsifier works, the
    affine update must fail it."""
    gamma = 0.9
    rows = []
    for seed in range(8):
        f, z_star, _, _ = _affine_control(32, gamma, seed, device)
        z1 = f(torch.zeros_like(z_star))
        rows.append(_rel_gap(z1, z_star))
    affine_gap = sum(rows) / len(rows)

    mp = []
    for seed in range(8):
        R, A, g = random_instance(n=32, n_act=4, gamma=gamma, seed=seed, device=device)
        z, info = maxplus_star(R, A, g)
        mp.append(_rel_gap(info["first_pass"], z))
    maxplus_gap = sum(mp) / len(mp)

    print("\n  mean ||z_1 - z*|| / ||z*||   affine (dead by R1's own note): %.4f"
          % affine_gap)
    print("                               max-plus (the candidate):       %.4f"
          % maxplus_gap)
    assert affine_gap < 0.1 * maxplus_gap, (
        f"the affine update moves {affine_gap:.3f} between pass 1 and the "
        f"settled state against the max-plus candidate's {maxplus_gap:.3f}: "
        f"R1's stated falsifier is passed by a closed-form linear solve, so it "
        f"separates nothing. Every gamma-contraction passes it."
    )


@pytest.mark.parametrize("device", DEVICES)
def test_the_collapse_probe_detects_the_maxplus_nonlinearity_at_the_settled_state(device):
    """R1's replacement probe: "best affine fit to the learned update at the
    settled state". Run it on the max-plus update at z*, at the radii a probe
    would plausibly use."""
    R, A, gamma = random_instance(n=16, n_act=4, gamma=0.9, seed=3, device=device)
    z, _ = maxplus_star(R, A, gamma)
    f = lambda w: bellman(w, R, A, gamma)
    rows = [(r, affine_fit_residual(f, z, r, seed=1))
            for r in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)]
    print("  max-plus affine-fit residual vs probe radius:")
    for r, v in rows:
        print("    radius %-8.0e  relative residual %.3e" % (r, v))
    small = [v for r, v in rows if r <= 1e-2]
    assert max(small) > 1e-6, (
        f"the collapse probe reports residual {max(small):.2e} at radius <= 1e-2 "
        f"on a genuinely non-affine operator: the max-plus update is piecewise "
        f"affine and z* is interior to a linearity cell, so any probe below the "
        f"cell radius certifies 'affine'. REQUIREMENTS.md R1 names no radius."
    )


@pytest.mark.parametrize("device", DEVICES)
def test_settling_does_combinatorial_work_in_proportion_to_its_iteration_cost(device):
    """If settling is a mechanism rather than a linear solver run slowly, the
    iteration must be discovering something. The only thing the max-plus
    iteration discovers that a solve cannot is WHICH policy wins. Count it."""
    rows = []
    for seed in range(6):
        R, A, gamma = random_instance(n=24, n_act=4, gamma=0.9, seed=seed, device=device)
        _, info = maxplus_star(R, A, gamma, log=True)
        rows.append((info["iters"], info["n_distinct_policies"]))
    ratios = [p / it for it, p in rows]
    print("  (iterations, distinct greedy policies) per instance: %s" % rows)
    print("  combinatorial fraction: %s" % ["%.4f" % r for r in ratios])
    assert min(ratios) > 0.1, (
        f"only {min(ratios):.4f} of the iterations change the greedy policy "
        f"(worst instance {min(rows, key=lambda r: r[1] / r[0])}): the remaining "
        f"iterations are numerical refinement of a linear system whose matrix "
        f"was already fixed. That is a solver, not a settling mechanism."
    )


@pytest.mark.parametrize("device", DEVICES)
def test_value_iteration_is_not_beaten_by_the_direct_method_for_the_same_point(device):
    """"Settle" is only a mechanism if the settled state is not available by a
    direct algorithm at comparable cost. For the max-plus star it is: Howard
    policy iteration terminates finitely and each step is one linear solve."""
    rows = []
    for seed in range(6):
        R, A, gamma = random_instance(n=64, n_act=4, gamma=0.95, seed=seed, device=device)
        t0 = time.perf_counter()
        z_vi, info = maxplus_star(R, A, gamma)
        t_vi = time.perf_counter() - t0
        t0 = time.perf_counter()
        z_pi, k_pi = policy_iteration(R, A, gamma)
        t_pi = time.perf_counter() - t0
        rows.append((info["iters"], t_vi, k_pi, t_pi, float((z_vi - z_pi).abs().max())))
    print("  seed | VI iters   VI s    | PI steps   PI s    | agreement")
    for i, r in enumerate(rows):
        print("   %d   | %6d  %.4f  | %6d  %.4f  | %.2e" % (i, r[0], r[1], r[2], r[3], r[4]))
    assert max(r[4] for r in rows) < 1e-9, "the two methods must find the same point"
    speedup = sum(r[1] for r in rows) / sum(r[3] for r in rows)
    print("  policy iteration is %.1fx faster for the identical fixed point" % speedup)
    assert speedup < 1.0, (
        f"the direct method reaches the identical settled state {speedup:.1f}x "
        f"faster in {max(r[2] for r in rows)} steps against "
        f"{max(r[0] for r in rows)} iterations: iterating to settle is a slow "
        f"algorithm for a quantity with a direct method, not a mechanism"
    )


# ==========================================================================
# CALIBRATION AND INSTRUMENTS -- GREEN
# ==========================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_the_collapse_probe_reproduces_the_measured_affine_residual(device):
    """Calibration: on the affine update the probe must read ~0 at every
    radius, INCLUDING the large radii where the max-plus operator jumps to
    3.6e-01. REQUIREMENTS.md records 1.05e-15 for this case; the small-radius
    readings here sit on the float64 lstsq conditioning floor (relative
    residual scales as noise/radius), which is why the discriminating radius
    is a large one, not a small one."""
    f, z_star, _, _ = _affine_control(24, 0.9, 42, device)
    rows = [(r, affine_fit_residual(f, z_star, r, seed=2))
            for r in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)]
    print("  affine control residual vs radius: "
          + ", ".join("%.0e:%.2e" % r for r in rows))
    assert max(v for _, v in rows) < 1e-8           # conditioning floor at 1e-6
    assert max(v for r, v in rows if r >= 1.0) < 1e-14


@pytest.mark.parametrize("device", DEVICES)
def test_a_fixed_iteration_budget_returns_a_non_fixed_point_with_no_alarm(device):
    """The measured hazard, reproduced: a DEQ blew a 30-NFE budget at step 47
    and nothing complained. Cap the budget and read what comes back."""
    R, A, gamma = random_instance(n=32, n_act=4, gamma=0.99, seed=9, device=device)
    z_true, info_full = maxplus_star(R, A, gamma)
    z_30, info_30 = maxplus_star(R, A, gamma, max_iter=30)
    resid_30 = info_30["residuals"][-1]
    err_30 = float((z_30 - z_true).abs().max())
    print("  budget 30: exit residual %.3e, true error %.3e, converged at %d iters"
          % (resid_30, err_30, info_full["iters"]))
    assert info_full["iters"] > 30
    assert err_30 > 1e-3
    # the residual is the alarm; it is above tol and says so
    assert resid_30 > 1e-6
    # and the geometric bound resid/(1-gamma) covers the true error
    assert err_30 <= resid_30 / (1 - gamma) + 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_the_residual_decays_at_the_contraction_modulus(device):
    """The instrument R1's hazard note asks for: log the residual per step.
    For a gamma-contraction the ratio must sit at gamma, and a departure from
    it is the alarm."""
    R, A, gamma = random_instance(n=32, n_act=4, gamma=0.9, seed=17, device=device)
    _, info = maxplus_star(R, A, gamma, max_iter=120)
    res = info["residuals"]
    ratios = [res[k + 1] / res[k] for k in range(20, 60)]
    lo, hi = min(ratios), max(ratios)
    print("  residual ratio over steps 20-60: [%.6f, %.6f] against gamma=%.2f"
          % (lo, hi, gamma))
    assert lo > gamma - 1e-6 and hi < gamma + 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_pass_one_is_not_the_settled_state_for_either_operator(device):
    """R1's falsifier, run and reported rather than argued. Both operators pass
    it. That is the point of the RED test at the top of this file."""
    gamma = 0.9
    f, z_star, _, _ = _affine_control(32, gamma, 0, device)
    assert _rel_gap(f(torch.zeros_like(z_star)), z_star) > 0.05
    R, A, g = random_instance(n=32, n_act=4, gamma=gamma, seed=0, device=device)
    z, info = maxplus_star(R, A, g)
    assert _rel_gap(info["first_pass"], z) > 0.05


@pytest.mark.parametrize("device", DEVICES)
def test_the_probe_works_when_read_against_its_control_at_the_same_radius(device):
    """The corrected instrument, and the shape of the update that survives R1.

    The probe's absolute residual is meaningless -- it sits on a conditioning
    floor that scales with 1/radius. Read as a RATIO against the affine control
    at the SAME radius it becomes radius-robust, and it separates the three
    cases at every radius rather than at a lucky one:

        affine              ratio 1        (by construction)
        max-plus            ratio ~1 below the cell radius, then large
        smooth non-monotone ratio large everywhere

    tanh(Mz + b) with M signed is the only one of the three that is both
    non-affine everywhere and non-monotone. It is what R1 needs and what R2
    needs, which is the same requirement arrived at twice."""
    n = 24
    g = torch.Generator(device="cpu").manual_seed(77)
    M = torch.randn((n, n), generator=g, dtype=torch.float64)
    M = (0.9 * M / torch.linalg.matrix_norm(M, 2)).to(device)
    b = torch.randn(n, generator=g, dtype=torch.float64).to(device)
    f_tanh = lambda z: torch.tanh(M @ z + b)
    z = torch.zeros(n, dtype=torch.float64, device=device)
    for _ in range(4000):
        z = f_tanh(z)

    f_aff, z_aff, _, _ = _affine_control(n, 0.9, 78, device)
    R, A, gamma = random_instance(n=n, n_act=4, gamma=0.9, seed=79, device=device)
    z_mp, _ = maxplus_star(R, A, gamma)
    f_mp = lambda w: bellman(w, R, A, gamma)

    print("  radius   | affine   | max-plus  ratio | tanh      ratio")
    tanh_ratios, mp_small = [], []
    for r in (1e-6, 1e-4, 1e-2, 1e-1, 1.0):
        a = affine_fit_residual(f_aff, z_aff, r, seed=4)
        m = affine_fit_residual(f_mp, z_mp, r, seed=4)
        t = affine_fit_residual(f_tanh, z, r, seed=4)
        print("  %-8.0e | %.2e | %.2e  %8.1f | %.2e  %8.1f"
              % (r, a, m, m / a, t, t / a))
        # radius 1e-6 is conditioning-floor-dominated for BOTH operators
        # (relative residual scales as noise/radius), so it is printed but not
        # asserted on -- a threshold there sits on numerical noise and flips
        # between runs. The signal rows start at 1e-4.
        if r >= 1e-4:
            tanh_ratios.append(t / a)
            if r <= 1e-2:
                mp_small.append(m / a)
    assert min(tanh_ratios) > 1e3, "the probe must flag tanh at every signal radius"
    assert min(mp_small) < 10.0, "and must miss max-plus below the cell radius"
    assert max(mp_small) < min(tanh_ratios) / 1e3, "the two must not overlap"


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device available")
def test_collapse_probe_cpu_cuda_parity():
    """The CPU path is the parity oracle for the probe as well."""
    R, A, gamma = random_instance(n=16, n_act=4, gamma=0.9, seed=3, device="cpu")
    z, _ = maxplus_star(R, A, gamma)
    Rg, Ag = R.cuda(), A.cuda()
    zg, _ = maxplus_star(Rg, Ag, gamma)
    for radius in (1e-4, 1.0, 10.0):
        a = affine_fit_residual(lambda w: bellman(w, R, A, gamma), z, radius, seed=1)
        b = affine_fit_residual(lambda w: bellman(w, Rg, Ag, gamma), zg, radius, seed=1)
        print("  radius %-8.0e cpu %.3e  cuda %.3e" % (radius, a, b))
        assert abs(a - b) < 1e-9 + 1e-6 * max(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "--tb=short",
                                  "-p", "no:cacheprovider"]))
