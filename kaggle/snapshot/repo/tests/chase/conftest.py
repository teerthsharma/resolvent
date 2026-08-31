"""Shared fixtures for tests/chase.

Part C. Three jobs, all of them about one thing: a test must not be able to
change the result of another test.

1. DEVICE PARAMETRISATION. `cuda` is skipped when absent rather than being the
   only path. A CPU reference is the only honest parity oracle for a GPU kernel.

2. GLOBAL TORCH STATE IS RESTORED AROUND EVERY TEST. Measured leak:
   `tests/chase/test_stochastic_P.py` called `torch.set_default_dtype(torch.float64)`
   at MODULE SCOPE. pytest imports every test module during collection, so that
   line ran before any test did, and every other module's `torch.eye(...)` came
   back float64. `test_kernel_contracts.py::test_residual_path_masks_the_missing_backward`
   then died with "expected mat1 and mat2 to have the same dtype, but got:
   float != double" -- an error with nothing whatsoever to do with what it tests.
   Worse, it was COLLECTION-ORDER DEPENDENT, so it appeared and vanished with the
   `-k` filter. The autouse fixture below is the fix that covers every caller,
   present and future, rather than the one module that happened to do it.

3. CONTEXT-POISONING ISOLATION. An illegal memory access on CUDA is not
   recoverable in-process: every later kernel launch in the same process fails,
   whatever its own inputs are. Measured this session: one out-of-range gather
   took 9 subsequent, unrelated CUDA tests down with it. `run_isolated` runs such
   a probe in a FRESH INTERPRETER so the blast radius is one subprocess.

4. KNOWN-RED LEDGER. Most failures in this directory are findings, not defects:
   the test's failure message IS the recorded measurement. They are marked
   `xfail(strict=True)` from one list below, so the suite is green when the world
   is as documented AND a finding that silently stops reproducing turns into a
   failure instead of quietly disappearing.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


# ------------------------------------------------- 2. global torch state guard

_PRISTINE_DTYPE = torch.get_default_dtype()


@pytest.fixture(autouse=True)
def _restore_global_torch_state():
    """Snapshot and restore process-global torch settings around every test."""
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(_PRISTINE_DTYPE)
    try:
        yield
    finally:
        torch.set_default_dtype(dtype if dtype is not None else _PRISTINE_DTYPE)
        torch.set_default_dtype(_PRISTINE_DTYPE)


# ------------------------------------------------------- 3. subprocess isolation


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


# --------------------------------------------------------- 4. known-red ledger

# node id -> what the failure records. These fail ON PURPOSE; the message is the
# finding. `strict=True` means a finding that stops reproducing fails the suite.
KNOWN_RED = {
    # --- R1 hazard: a DEQ loses its fixed point with no alarm in the loss
    "test_deq_divergence.py::test_deq_solver_keeps_converging_and_loss_curve_reflects_it":
        "inner solver blew its 30-NFE budget at step 47 while loss was still falling",
    "test_deq_divergence.py::test_loss_curve_is_a_usable_alarm_for_equilibrium_loss":
        "loss 3.1364 -> 1.4568 over 922 steps after the fixed point was lost",
    "test_deq_divergence.py::test_outer_deq_still_contracts_when_gamma_grows[0.999]":
        "rho(outer)=1.280 at gamma=0.999, residual 1.564e+22",
    "test_deq_divergence.py::test_saturating_outer_deq_equilibrium_carries_signal[0.9]":
        "equilibrium is the trivial fixed point: ||z*|| = 5.80e-12, tanh' ~ 1",
    "test_deq_divergence.py::test_saturating_outer_deq_equilibrium_carries_signal[0.99]":
        "equilibrium is the trivial fixed point: ||z*|| = 6.23e-11, tanh' ~ 1",
    "test_deq_divergence.py::test_saturating_outer_deq_equilibrium_carries_signal[0.999]":
        "did not converge; ||z*|| = 1.70, tanh' = 0.758",
    # --- R3 hazard: a row-stochastic operator cannot amplify
    "test_stochastic_P.py::test_resolvent_output_can_leave_the_convex_hull_of_b":
        "600/600 draws stayed inside [min(b),max(b)]/(1-gamma); it cannot amplify",
    "test_stochastic_P.py::test_reachable_set_is_not_a_scaled_convex_hull":
        "8/32 target coordinates lie outside the reachable set; NRMSE floor 0.590",
    "test_stochastic_P.py::test_simplex_projection_preserves_gradient_direction":
        "projection destroyed 82.1% of the gradient norm and zeroed 95.3% of P",
    "test_stochastic_P.py::test_resolvent_solve_is_cheap_at_1b_scale":
        "dense resolvent solve at d=2048 costs 11.10 ms per layer per forward",
    # --- R5 hazard: top-k is an argsort, and the builder costs more than the work
    "test_schedule_rebuild.py::test_schedule_selection_receives_gradient":
        "d(loss)/d(salience) is None; top-k is an argsort",
    "test_schedule_rebuild.py::test_loss_is_continuous_in_the_schedule_score":
        "loss jumps 5.9618 at a rank flip; median step-to-step change 0.00e+00",
    "test_schedule_rebuild.py::test_gradient_survives_a_schedule_rebuild_between_forward_and_backward":
        "9.4% of mask entries differed; gradient rel err 0.433, cosine 0.9066",
    "test_schedule_rebuild.py::test_mismatched_backward_still_descends_the_loss":
        "the mismatched step still descends, so nothing surfaces the mismatch",
    "test_schedule_rebuild.py::test_schedule_rebuild_is_cheaper_than_the_attention_it_schedules":
        "rebuild 42.63 ms vs 0.097 ms of attention = 440x",
    # --- caustic Theorem 1 as an objective
    "test_floor_objective.py::test_floor_does_not_prefer_a_noise_candidate_over_the_correct_one":
        "the certified floor selected 'pure_noise'; the bound is blind to correctness",
    "test_floor_objective.py::test_floor_separates_a_perfect_candidate_from_a_wholly_wrong_one":
        "floor(all correct) = floor(all wrong) = 0",
    "test_floor_objective.py::test_optimising_the_floor_does_not_degrade_accuracy":
        "soft floor 20.68 -> 1.11 while certified floor stayed 6 and accuracy 70%",
    "test_floor_objective.py::test_injectivity_guard_does_not_require_an_answer_key":
        "verify_injective consumes the gold answers it claims to avoid needing",
    "test_floor_objective.py::test_non_injective_ground_relation_is_caught_before_it_certifies":
        "certifies 19 errors on a model that answered all 20 correctly (token 220)",
    "test_floor_objective.py::test_certified_floor_has_a_usable_gradient":
        "d(floor)/d(params) is None; the exact bound is a step function",
    # --- the merged kernel, triton-lang/kernels#22
    "test_kernel_contracts.py::test_gradient_reaches_qkv_through_scheduled_attention":
        "output is detached; autograd fails OPEN, no exception, gradients never arrive",
    "test_kernel_contracts.py::test_residual_path_masks_the_missing_backward":
        "a residual connection makes the missing backward invisible",
    "test_kernel_contracts.py::test_empty_schedule_row_does_not_produce_nan":
        "acc/0 with l_i == 0 -> 4096 non-finite elements, silently",
    "test_kernel_contracts.py::test_scheduled_attention_rejects_empty_schedule_row":
        "an empty CSR row is not rejected before launch",
    # test_negative_block_index_* and test_out_of_range_offsets_* are NOT here:
    # they were moved into subprocess isolation and now assert their finding
    # positively, so they pass instead of failing on purpose.
    # Only the INVALID parameters are findings. seq 64/128 and head dim 128 are
    # supported and must keep passing, so these are keyed on the full
    # parametrised id rather than the bare test name.
    "test_kernel_contracts.py::test_tile_boundary_sequence_lengths[1]":
        "seq=1 is rejected: sequence length must be divisible by block_size",
    "test_kernel_contracts.py::test_tile_boundary_sequence_lengths[63]":
        "seq=63 is rejected: sequence length must be divisible by block_size",
    "test_kernel_contracts.py::test_tile_boundary_sequence_lengths[129]":
        "seq=129 is rejected: sequence length must be divisible by block_size",
    "test_kernel_contracts.py::test_common_head_dims[80]":
        "head dim 80 is rejected; only 16/32/64/128 are supported",
    "test_kernel_contracts.py::test_common_head_dims[96]":
        "head dim 96 is rejected; only 16/32/64/128 are supported",
    # --- shipping the thing
    "test_hf_shipping.py::test_model_loads_without_trust_remote_code":
        "a custom kernel checkpoint requires trust_remote_code",
    "test_hf_shipping.py::test_custom_kernel_runs_on_cpu":
        "no CPU fallback path exists in scheduled_attention()",
    "test_hf_shipping.py::test_triton_is_importable_without_cuda":
        "the merged module imports triton at module scope",
    "test_hf_shipping.py::test_kernel_covers_the_gpus_a_1b_model_is_actually_run_on":
        "Triton needs cc >= 8.0; T4 (sm_75), the free-Colab GPU, cannot run it",
    "test_hf_shipping.py::test_attention_mask_interface_registration_is_not_a_silent_trap":
        "registering an attention impl does not register its mask; causality drops",
    # --- CHASE round 2: training, sizing and the Hub package
    "test_signed_operator_trainability.py::test_the_row_l1_spread_is_narrow_enough_for_a_1_over_l1_jacobian":
        "row L1 spans 1.10e-03 to 1.67e+01 at init = 1.5e+04x; the Jacobian is 1/l1",
    "test_signed_operator_trainability.py::test_the_row_l1_spread_stays_bounded_throughout_training":
        "median/min row L1 peaks at 1.20e+05 over 200 steps on real text",
    "test_signed_operator_trainability.py::test_forward_backward_memory_is_within_2x_of_the_softmax_arm[512]":
        "signed fwd+bwd 379.9 MiB vs softmax 143.5 MiB = 2.65x at seq 512",
    "test_signed_operator_trainability.py::test_forward_backward_memory_is_within_2x_of_the_softmax_arm[1024]":
        "signed fwd+bwd 1271.2 MiB vs softmax 285.8 MiB = 4.45x at seq 1024",
    "test_scale_sizing.py::test_a_500m_signed_model_trains_at_seq_2048_batch_1_on_a_colab_gpu[A100-40GB]":
        "0.5B signed at seq 2048 batch 1 needs 36.31 GiB against a 33.53 GiB A100-40GB budget",
    "test_scale_sizing.py::test_a_500m_signed_model_trains_at_seq_2048_batch_1_on_a_colab_gpu[L4-24GB]":
        "0.5B signed at seq 2048 batch 1 needs 36.31 GiB against a 20.25 GiB L4 budget",
    "test_scale_sizing.py::test_a_500m_signed_model_reaches_a_batch_of_at_least_8_at_seq_2048[A100-40GB]":
        "max batch 0 on A100-40GB against the softmax control's 8; batch 8 is 230.01 GiB",
    "test_scale_sizing.py::test_a_500m_signed_model_reaches_a_batch_of_at_least_8_at_seq_2048[L4-24GB]":
        "max batch 0 on L4-24GB against the softmax control's 4",
    "test_ceq_hub_package.py::test_registering_for_auto_class_does_not_break_weight_tying":
        "is_remote_code() is `_auto_class is not None`, so shipping code un-ties lm_head",
    # --- CHASE round 3: does parity scale, and what does 300M cost
    "test_scale_hazards.py::test_the_shipped_conditioning_rollback_covers_the_parity_operator":
        "the HF path computes rho*w/||w||_1; the parity operator is sgate, rel diff 1.818e+00",
    "test_scale_hazards.py::test_the_hazard_the_eps_rollback_was_built_for_exists_on_the_parity_operator":
        "min sgate row L1 is 1.227272 = the constructive floor at every d and S; no 1/l1 to bound",
    "test_scale_axes.py::test_the_attention_branch_does_not_dominate_the_residual_at_depth[4]":
        "last-layer branch/residual RMS 0.5060 sgate vs 0.0876 softmax = 5.78x at L=4, init",
    "test_scale_axes.py::test_the_attention_branch_does_not_dominate_the_residual_at_depth[24]":
        "0.2284 vs 0.1126 = 2.03x at L=24; the gap NARROWS with depth, 5.78x -> 2.03x",
    "test_scale_sizing.py::test_a_300m_sgate_model_trains_at_batch_1_on_a_colab_gpu[2048-L4-24GB]":
        "300M sgate seq 2048 batch 1 needs 28.43 GiB bf16 against a 20.25 GiB L4 budget",
    "test_scale_sizing.py::test_the_operator_term_is_not_the_majority_of_the_300m_activation_budget[1024]":
        "the [S,S] term is 4.97 GiB of 6.14 GiB = 80.9% at seq 1024",
    "test_scale_sizing.py::test_the_operator_term_is_not_the_majority_of_the_300m_activation_budget[2048]":
        "the [S,S] term is 19.89 GiB of 22.23 GiB = 89.5% at seq 2048",
    "test_scale_sizing.py::test_the_memory_preflight_predicts_the_dtype_the_training_loop_actually_uses":
        "train.py sizes with bf16_autocast and trains fp32: 86.6 MiB against 48.3 promised = 1.79x",
    "test_scale_sizing.py::test_a_300m_step_is_within_2x_the_softmax_control_at_the_300m_block_shape[1024]":
        "sgate 256.8 ms/step vs softmax 82.0 ms = 3.13x at d=1024 H=16 seq 1024",
    "test_scale_sizing.py::test_a_300m_step_is_within_2x_the_softmax_control_at_the_300m_block_shape[2048]":
        "5.41x-5.56x at seq 2048 on a quiet GPU; 19.81x under allocator pressure",
    "test_scale_sizing.py::test_a_300m_parity_run_at_a_chinchilla_budget_fits_one_colab_a100_session[1024]":
        "6.04e9 tokens costs the control 41.4 A100-hours and the operator 130; the cap is 24",
    "test_scale_sizing.py::test_a_300m_parity_run_at_a_chinchilla_budget_fits_one_colab_a100_session[2048]":
        "6.04e9 tokens costs the control 46.8 A100-hours and the operator 257; the cap is 24",
    # --- the flex_attention rollback
    "test_rollback_flex_attention.py::test_adapted_flex_attention_matches_the_merged_kernel":
        "the CSR -> BlockMask adapter is 151x worse than the kernel's own parity",
}


def _ledger_keys(nodeid):
    """Every form a KNOWN_RED entry may be written in, most specific first.

    `test_x.py::test_y[cuda-512]` is looked up as itself, then with the DEVICE
    token stripped (`test_y[512]`), then bare (`test_y`). The middle form exists
    because every test in this directory is parametrised over device, so a
    finding that depends on a non-device parameter -- a sequence length, a GPU
    name -- has no other way to be keyed without also naming a device it has
    nothing to do with.
    """
    rel = nodeid.split("/")[-1]
    yield rel
    name, sep, rest = rel.partition("[")
    if sep:
        for dev in ("cpu-", "cuda-"):
            if rest.startswith(dev):
                yield name + "[" + rest[len(dev):]
    yield name


def pytest_collection_modifyitems(config, items):
    for item in items:
        for key in _ledger_keys(item.nodeid):
            reason = KNOWN_RED.get(key)
            if reason is not None:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
                break
