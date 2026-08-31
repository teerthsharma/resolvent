"""Bitwise comparators for Gate-0 items G0.2 (K-RESUME) and G0.3 (K-PERSIST).

NOT IN `conftest.py`, AND NOT IMPORTED AS A BARE `conftest`. That form is what
`tests/chase/conftest.py` does for `requires_triton`, and it was copied here
first -- but `tests/loop/test_conftest_import_is_order_dependent.py` records it
as a DEFECT, not a convention: 24 of 26 test directories carry a `conftest.py`
and only 2 carry an `__init__.py`, so the bare name `conftest` binds to
whichever directory collection order reached first. The chase file is one of
that guard's standing baseline failures, so it is precedent for the bug and not
for the fix. These live in a module whose name is reached by its full path,
`tests.gate0.helpers`, which no other directory can claim.

WHY A WALKER AND NOT A LOOP OVER `named_parameters()`. The contract asks for
FOUR state components, and only one of them is a flat dict of tensors. The
optimizer's `state_dict()` is `{"state": {param_index: {"exp_avg": T,
"exp_avg_sq": T, "step": T}}, "param_groups": [ {...python scalars...} ]}`, and
a comparison that reaches only the top level of that would pass while the
moments drifted. The walker descends into every dict and list and compares the
leaves, so `optimizer.state.0.exp_avg` is a path the report can name.

`torch.equal`, never `allclose`. Two known edges, neither reachable here but
both worth naming: `torch.equal` calls `+0.0` and `-0.0` equal although their
bits differ, and calls NaN unequal to itself although the bits agree. AdamW's
moments and torch's RNG byte buffers produce neither.
"""
import os

import torch


def _walk(a, b, path, out):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            out.append(path + "  [one side is not a tensor]")
        elif a.dtype != b.dtype or a.shape != b.shape:
            out.append(path + "  [{}{} vs {}{}]".format(
                a.dtype, tuple(a.shape), b.dtype, tuple(b.shape)))
        elif not torch.equal(a, b):
            out.append(path + "  [max |delta| = {:.6e}]".format(
                float((a.double() - b.double()).abs().max())
                if a.is_floating_point() else float("nan")))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(path + "  [keys {} vs {}]".format(sorted(a), sorted(b)))
            return
        if not a:
            # A dict with no keys compares equal to another dict with no keys
            # having examined nothing. That is the shape a vacuous control
            # takes, so it is REPORTED rather than passed over.
            out.append(path + "  [VACUOUS: empty dict on both sides]")
            return
        for k in a:
            _walk(a[k], b[k], "{}.{}".format(path, k), out)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(path + "  [len {} vs {}]".format(len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, "{}[{}]".format(path, i), out)
        return
    if a != b:
        out.append(path + "  [{!r} vs {!r}]".format(a, b))


def bitwise_diff(a, b, root="<root>"):
    """Every path at which two nested structures differ, `torch.equal` at leaves."""
    out = []
    _walk(a, b, root, out)
    return out


def load_state(d):
    """The four checkpoint state components of a `train()` out_dir, plus `step`.

    Read back from DISK, not from the live objects a run happened to leave in
    memory: what survives a Kaggle session cap is the directory, and that is the
    only thing whose equality means anything for resume.
    """
    from ceq.hf.modeling_ceq import CEQForCausalLM
    ck = torch.load(os.path.join(d, "trainer_state.pt"), map_location="cpu",
                    weights_only=True)
    return dict(model=dict(CEQForCausalLM.from_pretrained(d).named_parameters()),
                optimizer=ck["optimizer"],
                torch_rng_state=ck["torch_rng_state"],
                data_gen_state=ck["data_gen_state"],
                step=ck["step"])
