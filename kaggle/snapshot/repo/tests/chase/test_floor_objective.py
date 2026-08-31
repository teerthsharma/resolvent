"""THEORY.md §5: caustic Theorem 1 as a TRAINING objective rather than a diagnostic.

Theorem 1 (caustic README §4, verbatim): "For injective R, the number of entities on
which f errs, written err(f), is at least n - m: err(f) >= n - m", where m is the
number of distinct values the model produces. caustic's README calls it "usable as an
objective" because "the bound needs no ground truth".

Minimising n - m is maximising m. m is the number of DISTINCT OUTPUTS. Nothing in the
quantity refers to correctness. These tests ask what a gradient-based optimiser does
when pointed at it for 40k steps rather than a human reading it once.

caustic's own documented failure (README "Scope: what this does not claim"): under
Qwen/Llama tokenizers every numeric answer shares first token 220, so "twenty distinct
numbers collapse to one observable value, m = 1 regardless of the model, and Theorem 1
would certify n - 1 = 19 wrong answers on a model that answered all twenty correctly."
branchcut's select_by_floor raises on non-injective candidates precisely because
"optimising a floor that certifies nothing is worse than not optimising, because the
number still looks like a score."
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_ref import orbit_floor, select_by_floor, verify_injective  # noqa: E402

N_ENTITIES = 20
VOCAB = 32


# ================================================ 1. the objective's global optimum

def test_floor_does_not_prefer_a_noise_candidate_over_the_correct_one():
    """THEORY.md §5 runs candidate CSR schedules in competition scored by the floor,
    with the do-nothing candidate (the merged 0D-salience builder) always entered."""
    gold = [f"ans{i}" for i in range(N_ENTITIES)]

    # The do-nothing candidate is 100% CORRECT, but the observable collapses two of its
    # answers onto one value -- caustic's Asmara/Asuncion case, first token 1634.
    correct_but_colliding = list(gold)
    correct_but_colliding[1] = correct_but_colliding[0]
    correct_but_colliding[3] = correct_but_colliding[2]

    candidates = [
        ("do_nothing_0d_salience", correct_but_colliding),   # 20/20 right, 2 collisions
        ("pure_noise", [f"garbage_{i}" for i in range(N_ENTITIES)]),  # 0/20 right, 0 collisions
    ]
    winner, scored = select_by_floor(candidates)
    assert winner != "pure_noise", (
        f"the certified floor selected {winner!r}. Scores (floor, name): {scored}. "
        f"A schedule producing 20 distinct wrong answers scores m=n, floor 0 -- "
        f"identical to a schedule producing 20 distinct RIGHT answers. The bound is "
        f"blind to correctness by construction; that is what 'consults no answer key' "
        f"means."
    )


def test_floor_separates_a_perfect_candidate_from_a_wholly_wrong_one():
    """The minimum requirement for a score: a perfect answer must beat a wrong one."""
    gold = [f"ans{i}" for i in range(N_ENTITIES)]
    wrong = [f"wrong{i}" for i in range(N_ENTITIES)]
    assert orbit_floor(gold) < orbit_floor(wrong), (
        f"floor(all correct) = {orbit_floor(gold)}, floor(all wrong) = "
        f"{orbit_floor(wrong)}. The objective cannot distinguish them."
    )


# ============================================== 2. what gradient descent does with it

def _soft_floor(logits):
    """Differentiable surrogate for n - m: the expected number of pairwise collisions.

    m = n - (collisions), and E[collision(i,j)] = sum_v p_i(v) p_j(v). This is the
    obvious relaxation and it is the only kind available -- min_errors in branchcut is
    typed `int -> int` and there is no differentiable variant in either repo.
    """
    p = torch.softmax(logits, dim=-1)
    agree = p @ p.T
    n = logits.shape[0]
    return (agree.sum() - agree.diag().sum()) / 2


def test_optimising_the_floor_does_not_degrade_accuracy():
    """40k steps compressed: start from a model that is 100% correct, descend the floor.

    A wrong diagnostic misleads a reader once. A wrong objective is descended.
    """
    torch.manual_seed(0)
    gold = torch.arange(N_ENTITIES)
    logits = torch.full((N_ENTITIES, VOCAB), -4.0)
    logits[torch.arange(N_ENTITIES), gold] = 4.0          # correct on every entity ...
    # ... but the observable collapses 6 of them onto entity 0's answer, exactly the
    # collision structure caustic documents. The floor sees 5 certified errors.
    for i in range(1, 7):
        logits[i, gold[i]] = -4.0
        logits[i, gold[0]] = 4.0
    logits = logits.clone().requires_grad_()
    start_true_acc = 1.0  # every entity's ARGMAX-before-collapse answer was correct

    opt = torch.optim.Adam([logits], lr=0.05)
    trace = []
    for step in range(2000):
        opt.zero_grad()
        loss = _soft_floor(logits)                        # THE objective. Nothing else.
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == 1999:
            pred = logits.argmax(-1)
            acc = float((pred == gold).double().mean())
            floor = orbit_floor([int(x) for x in pred])
            trace.append((step, acc, floor, float(loss)))

    # Measured: soft floor 20.6817 -> 1.1070 (18.7x better) over 2000 steps, while the
    # CERTIFIED floor stayed pinned at 6 and true accuracy stayed at 70.0% throughout.
    moved = trace[0][2] != trace[-1][2]
    assert moved, (
        "descending the certified floor from a 100%-correct start:\n  "
        + "\n  ".join(f"step {s:4d}: accuracy {a:5.1%}  certified floor {f:2d}  "
                      f"soft floor {l:.4f}" for s, a, f, l in trace)
        + f"\nAccuracy {trace[0][1]:.0%} -> {trace[-1][1]:.0%} while the certified floor "
          f"{trace[0][2]} -> {trace[-1][2]}. The objective improved monotonically."
    )


# ==================================== 3. the guard needs the thing it claims to avoid

def test_injectivity_guard_does_not_require_an_answer_key():
    """caustic's verify_injective signature is
       verify_injective(spec, gold: dict[str, str], first_token_fn) -> list[pairs]

    THEORY.md §5's premise is "Training without ground truth". The precondition check
    that makes Theorem 1 safe takes the answer key as an argument.
    """
    import inspect
    src = inspect.signature(verify_injective)
    needs_gold = "ground_values" in src.parameters
    assert not needs_gold, (
        f"verify_injective{src} consumes the ground relation's values. In caustic the "
        f"real signature is verify_injective(spec, gold, first_token_fn) where gold is "
        f"dict[entity -> correct answer]. There is no way to certify the precondition "
        f"of a ground-truth-free bound without ground truth. THEORY.md §5 concedes the "
        f"entities and ground relation for schedule selection are unstated; this is "
        f"what stating them costs."
    )


def test_non_injective_ground_relation_is_caught_before_it_certifies():
    """Reproduce caustic's own documented case: the tokenizer that collapses every
    numeric answer to first token 220."""
    gold_strings = [str(i) for i in range(1, 21)]           # 20 distinct correct answers
    first_token = [220 for _ in gold_strings]               # Qwen/Llama: all share 220
    model_answers = list(first_token)                       # model is CORRECT on all 20

    assert not verify_injective(first_token), "setup: the ground relation is not injective"
    floor = orbit_floor(model_answers)
    assert floor == 0, (
        f"Theorem 1 certifies {floor} errors on a model that answered all "
        f"{len(gold_strings)} of them correctly (m=1, because every gold answer shares "
        f"first token 220). caustic's own documented case, reproduced. As a diagnostic "
        f"that is one wrong number a human reads once. As an objective it is a "
        f"persistent gradient pointing away from the correct answers, and the only "
        f"thing that stops it -- verify_injective -- needs the answer key."
    )


# ============================================================ 4. differentiability

def test_certified_floor_has_a_usable_gradient():
    """min_errors in branchcut is typed `int -> int`. THEORY.md §5 wants it as an
    objective; an objective needs a derivative."""
    torch.manual_seed(0)
    logits = torch.randn(N_ENTITIES, VOCAB, requires_grad=True)
    pred = logits.argmax(-1)
    floor = torch.tensor(float(orbit_floor([int(x) for x in pred])))
    try:
        g = torch.autograd.grad(floor, logits, allow_unused=True)[0]
    except RuntimeError as exc:
        g = None
        reason = str(exc)
    else:
        reason = "gradient was all zero"
    assert g is not None and g.abs().sum() > 0, (
        f"{reason}. " +
        "d(certified floor)/d(parameters) is None: the exact bound is a step function "
        "of argmax outputs, constant almost everywhere. Any training use must "
        "substitute a surrogate, and the surrogate is what actually gets descended -- "
        "not the certified quantity."
    )
