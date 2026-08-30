"""X-R1: the identity manifest -- a content hash of what was actually measured.

WHY. Nine or more times this project has reported a number measured under one
object beneath another object's name: `tgate` measured under `sgate`'s, a
ParaFormer arm silently running softmax, `impact_hetero` registered against
`make_impact_batch` so a heterogeneous-plant arm drew the homogeneous corpus,
and the published intervals splitting into two families because the estimator
was dropped between the JSON and the rendered row. Every instance is the same
shape and none of them was detectable from the artifact.

WHAT THE JOURNAL KEY ALREADY DOES, AND WHERE IT STOPS. `m3_quintuple._key`
encodes `cell, k, s, d, steps, n_train, n_eval, t_max, seed, task`. That is
already the config half and three consumers parse it. It does NOT encode
`beta`, `n_neumann`, `d_model` or `reserve_query` -- and `load_unit`'s own
docstring names `cell, k_piv, beta, t_max, n_neumann` as the arguments "that
make a `QuintArm` the cell it is", warning that "a consumer that re-derives them
derives them wrong exactly once and then reports a `twin` number under
`settled`". Two runs differing only in `n_neumann` collide on one key and the
second silently overwrites the first. `beta` is worse: it is a module constant
(`BETA = 0.5`), so changing it moves every published cell with no change to any
key, any record field, or anything a reader can see.

THE FOUR COMPONENTS, one per thing the contract names.

  config  every declared scalar that makes the arm the arm, including the four
          the key drops.
  code    `module:qualname` plus a BYTECODE fingerprint of each callable the arm
          dispatches through. Bytecode and not source, deliberately: a docstring
          or comment edit must NOT invalidate a published cell, or the refusal
          becomes one nobody can obey and gets switched off. A changed branch
          moves it.
  shapes  parameter shapes in construction order.
  rng     the declared seeding plan AND a digest of the tensors it produced.
          The plan alone is a claim; the tensors are the measurement of it.

WHY THE RNG HALF IS NOT OPTIONAL. `paired_arm.train_and_predict` calls
`torch.manual_seed(seed)` and then constructs the arm, so initialisation draws
from the GLOBAL generator. Anything that consumes from that generator in between,
or any change to the order in which parameters are created, moves every
published number while config, key and code all stay put. X-R6 names this as
"G2 inversion via a `torch.Generator` passed by reference". Hashing the tensors
is the only artifact that records what the generator actually did.

G2 -- NO FIX MAY MOVE A PUBLISHED NUMBER. Everything here is a pure read: no
tensor is written, no RNG is advanced, no file is opened.
`tests/cameron/test_identity_manifest.py` asserts that bitwise against a
published cell and against the committed journal.

STATUS: this module is standalone by design. `scale/m3_quintuple.py` is held by
another agent mid-run, so the integration points are reported rather than
landed -- see the report for the three call sites.
"""
from __future__ import annotations

import hashlib
import json
from types import CodeType

#: Every field that makes a measured cell the cell it is. A field missing here
#: is a field two different arms can collide on, which is the whole defect, so
#: this list is deliberately wider than `_key`'s: it adds `beta`, `n_neumann`,
#: `d_model` and `kind`, and it keeps `torch_version` because a number taken
#: under a different torch IS a number taken under a different object. That last
#: one will fire on an interpreter upgrade; that is the mechanism working, and
#: the refusal names the field so the reader can see in one line why.
CONFIG_FIELDS = (
    "cell", "kind", "task", "s", "d", "d_model", "k_piv", "beta",
    "t_max", "n_neumann", "steps", "n_train", "n_eval", "seed",
    "device", "torch_version",
)

#: The shipped seeding plan, read from `scale/paired_arm.py`: the train batch is
#: drawn at `seed`, the eval batch at `seed + 12345`, and the model is then
#: initialised from the GLOBAL generator after `torch.manual_seed(seed)`.
#: Declared here so a change to it is a change to a value under version control
#: rather than an unrecorded change in behaviour.
RNG_PLAN = {"train_seed": "seed", "eval_seed": "seed + 12345",
            "init": "global torch.manual_seed(seed)"}


class IdentityMismatch(RuntimeError):
    """A cell whose stored identity is not the identity of the arm in hand."""


def _sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
        h.update(b"\x00")
    return h.hexdigest()


def _code_fingerprint(fn) -> str:
    """`module:qualname` plus a recursive bytecode digest.

    The docstring is dropped from `co_consts` so that documenting a function
    does not invalidate every cell measured through it. Nested code objects
    (comprehensions, closures) are recursed into, because a changed
    comprehension is a changed implementation.
    """
    code: CodeType = fn.__code__
    doc = getattr(fn, "__doc__", None)

    def walk(c: CodeType, top: bool) -> str:
        consts = []
        for i, k in enumerate(c.co_consts):
            #: CPython reserves `co_consts[0]` of a FUNCTION body as the
            #: docstring slot: it holds the string when one is written and
            #: `None` when one is not. Dropping only the string would leave the
            #: two cases asymmetric -- `(None, 1)` against `(1,)` -- and a
            #: docstring edit would still move the digest, which is the failure
            #: this rule exists to prevent. Both are dropped. The rule applies
            #: at the top level ONLY: a nested code object (a comprehension, a
            #: closure) has no docstring slot, so its first constant is real
            #: logic and dropping it would blind the digest to a changed
            #: comprehension.
            if top and i == 0 and (k is None or (isinstance(k, str) and k == doc)):
                continue
            consts.append(walk(k, False) if isinstance(k, CodeType) else repr(k))
        return _sha(c.co_code, repr(c.co_names).encode(),
                    repr(c.co_varnames).encode(), repr(consts).encode())

    name = f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', fn)}"
    return _sha(name.encode(), walk(code, True).encode())


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, default=repr).encode()


def manifest(record: dict, *, callables, params, rng_plan: dict | None = None) -> dict:
    """The identity of one measured cell. Pure read.

    `record` is the saved unit dict (or any mapping carrying CONFIG_FIELDS),
    `callables` the functions the arm dispatches through, `params` its
    parameter tensors. Returns the four component digests, the config values
    themselves so a refusal can name the field that moved, and the combined
    `hash` that a table cell stores.
    """
    cfg = {k: record[k] for k in CONFIG_FIELDS if k in record}
    plan = RNG_PLAN if rng_plan is None else rng_plan

    shapes = [(k, tuple(v.shape)) for k, v in params.items()]
    #: `.cpu()` before `.numpy()` so a cuda-lane cell hashes identically to the
    #: CPU copy that is actually saved -- the cuda lane already serialises CPU
    #: tensors, and an identity that depended on device would refuse the
    #: artifact it just wrote.
    values = _sha(*(v.detach().cpu().numpy().tobytes() for _, v in params.items()))

    config = _sha(_canon(cfg))
    code = _sha(*(_code_fingerprint(f).encode() for f in callables))
    shape = _sha(_canon(shapes))
    rng = _sha(_canon(plan), values.encode())
    return {"config": config, "code": code, "shapes": shape, "rng": rng,
            "values": cfg, "plan": plan,
            "hash": _sha(config.encode(), code.encode(), shape.encode(),
                         rng.encode())}


def refuse_if_changed(stored: dict, current: dict) -> None:
    """Raise unless `current` is the same measured object as `stored`.

    THE REFUSAL NAMES WHAT MOVED. A bare "hash differs" sends the reader back to
    bisecting, which is how the original defects survived: every one of them was
    visible in principle and nothing pointed at it. Every component that moved
    is listed, and for `config` the individual fields are named with both
    values.
    """
    moved = [c for c in ("config", "code", "shapes", "rng")
             if stored.get(c) != current.get(c)]
    if not moved:
        return
    lines = [f"IDENTITY MISMATCH: {', '.join(moved)} moved -- this cell was "
             f"measured under a different object than the one in hand"]
    if "config" in moved:
        a, b = stored.get("values", {}), current.get("values", {})
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                lines.append(f"  config.{k}: {a.get(k)!r} -> {b.get(k)!r}")
    if "rng" in moved and stored.get("plan") != current.get("plan"):
        lines.append(f"  rng.plan: {stored.get('plan')} -> {current.get('plan')}")
    elif "rng" in moved:
        lines.append("  rng: the seeding plan is unchanged, so the initial "
                     "tensors it produced are not the ones this cell was "
                     "measured from")
    lines.append(f"  stored {stored.get('hash')} != current {current.get('hash')}")
    raise IdentityMismatch("\n".join(lines))
