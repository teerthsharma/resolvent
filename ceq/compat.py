"""K-COMPAT -- the bridge between the box that DECIDES and the box that RUNS.

TWO STACKS, AND THEY ARE NOT THE SAME ONE.

    decided on   python 3.11.9   torch 2.5.1+cu121   transformers 5.3.0   RTX 4060
    runs on      python 3.12.13  torch 2.10.0+cu128  transformers ?       Kaggle GPU

The right-hand row is `[RUN]`, read off Kaggle run 1's own log
(`results/kaggle_v17k_output/ceq-v17-k.log`, lines 8 and 10). Every
`cumsum_cuda_kernel` determinism finding that Rulings 1 and 9 rest on was
measured on the left-hand row. **There is no 3.12/2.10 interpreter on the
deciding box**, so nothing in this module could be tested against the stack it
exists for. That constraint is the whole design:

  * everything this module reports it OBSERVES on the interpreter that is
    running it -- there is not one inherited version string in any return value;
  * where it cannot adapt it raises `CompatError` naming the module and the
    exception, at a cell that costs one minute, rather than letting a later cell
    produce a number downstream of a precondition that silently did not hold;
  * it changes NO numerical behaviour to make a version work. The one thing it
    rebinds is an import PATH (`transformers.generation.GenerationMixin`), which
    moves no float. If a version needed the arms to compute something different,
    that is the author's ruling and this module would report it, not do it.

PUBLIC API

    REFERENCE                     the matrix the local campaign was measured on
    PRIVATE_ATTRS                 the six undocumented PreTrainedModel internals
    REQUIRED_MODULES              what the Kaggle run must be able to import
    CompatError                   the one exception; carries module + cause

    version_record()              -> dict, JSON-safe, for run headers/manifests
    matches_reference(rec)        -> bool
    transformers_report()         -> dict: the six attributes + the mixin path
    attr_binding(base, sub, name) -> dict (the pure checker the report is built of)
    ensure_generation_mixin()     -> dict; installs the path alias if it can
    probe_determinism(device)     -> dict: MEASURED regime on the live stack
    regime(fwd, strict_bwd, warn_bwd) -> str (the pure classifier)
    selfcheck(verbose=True)       -> dict; raises CompatError

WHERE IT IS CALLED. `kaggle/ceq_v17k.ipynb`, at the foot of the version-pin cell
(step "3a", cell index 4), after the pins are settled and before step "3b" and
K-CERT. It is APPENDED to that cell rather than inserted as its own, because
`tests/gate0/test_g16_lrt_pinned.py::test_the_notebook_computes_lambda_at_the_
end_of_q3_without_renumbering_anything` binds `len(cells) == 22`, `cells[15]`
and `cells[18]`, and an inserted cell renumbers all three.

WHAT THIS MODULE DOES NOT REACH. The alias installed by
`ensure_generation_mixin` is PROCESS-LOCAL. Step "3b" runs
`python -m ceq.hf.smoke` in a SUBPROCESS, which gets a fresh interpreter and no
alias -- which is why K-COMPAT runs BEFORE it: on a stack where the import path
has moved, the legible halt happens here and the subprocess's raw traceback
never has to be read. Recorded as a boundary, not fixed.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import warnings

import torch

#: The matrix every local number was measured on. NOT retyped -- it is the same
#: dict `ceq/hf/smoke.py` already prints under "VERIFIED AGAINST", so a stack
#: that moves cannot be recorded in two places and drift between them.
from ceq.hf.smoke import VERIFIED_AGAINST as REFERENCE

#: The six undocumented `PreTrainedModel` internals `ceq/hf/modeling_ceq.py`
#: sets (lines 445-449 and 630). `V17_G04_ENV.md` B2 flags `transformers==5.3.0`
#: as load-bearing for exactly these. They are the reason that pin is tight, and
#: they are what this module measures instead of assuming.
PRIVATE_ATTRS = ("_no_split_modules", "_supports_sdpa", "_supports_flash_attn",
                 "_supports_flex_attn", "_can_compile_fullgraph",
                 "_tied_weights_keys")

#: Every module the Kaggle queue reaches. `scripts.k_cert` is a NAMESPACE
#: package import (`scripts/` carries no `__init__.py`) and therefore depends on
#: the repo root being on `sys.path` -- which is the notebook's cell 2 job and a
#: real thing to fail on.
REQUIRED_MODULES = ("ceq.kdata", "ceq.hf.modeling_ceq", "ceq.hf.train",
                    "ceq.autopilot", "scripts.k_cert")

#: What Rulings 1 and 9 rest on, stated as a triple so the probe can compare
#: against it instead of a prose paragraph nobody evaluates.
#: (strict forward executable, strict backward executable, warn_only backward
#: executable) for `cumprod` on CUDA, measured on torch 2.5.1+cu121 / RTX 4060.
INHERITED_CUDA_CUMPROD = (True, False, True)

_GENERIC_TOKENS = {"supports", "can", "modules", "keys"}


class CompatError(RuntimeError):
    """A stack difference this bridge cannot adapt to. Always names the thing
    that failed and the exception that failed it."""


# --------------------------------------------------------------- version record

def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _parse_arch(arch: str):
    """`"sm_89"` -> `(8, 9)`, `"sm_100"` -> `(10, 0)`. torch writes the minor as
    the LAST digit and everything before it as the major, which is the only
    reading that keeps Blackwell's `sm_100` apart from a hypothetical `sm_10`."""
    digits = arch.split("_")[-1]
    if not digits.isdigit() or len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


def device_usable_from(capability, arch_list) -> bool:
    """Can a torch built for `arch_list` actually run on a device of `capability`?

    NOT `capability in arch_list`, which is the check that reads False on the
    CERTIFIED LOCAL 4060: `sm_89` is absent from torch 2.5.1+cu121's arch list
    (`sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90`) `[MEASURED]` on a box
    where every CUDA op in this repository demonstrably runs. A cubin built for
    `sm_XY` runs on `sm_XZ` for `Z >= Y` within the same major `X`, so the test
    is same-major and minor-not-greater. The P100 case Kaggle run 1 hit stays
    False under it -- `sm_60` against a list with no major 6 `[RUN]`.
    """
    want = _parse_arch(capability) if capability else None
    if want is None:
        return False
    return any(p is not None and p[0] == want[0] and p[1] <= want[1]
               for p in (_parse_arch(a) for a in arch_list))


def version_record() -> dict:
    """Everything needed to say whether two numbers are comparable at all.

    A number whose environment is not recorded beside it cannot be compared
    across the two boxes, and this round compares across two boxes by
    construction. Every value is read off the live interpreter; JSON-safe so a
    run header or a manifest can embed it whole.
    """
    import transformers

    # BEFORE anything below touches the device: `get_device_capability` would
    # initialise CUDA itself and this field would then be True on every call.
    initialized = bool(torch.cuda.is_initialized())
    cuda = bool(torch.cuda.is_available())
    name = capability = None
    arches: list = []
    device_error = None
    if cuda:
        try:
            major, minor = torch.cuda.get_device_capability(0)
            capability = "sm_{}{}".format(major, minor)
            name = torch.cuda.get_device_name(0)
            arches = list(torch.cuda.get_arch_list())
        except Exception as exc:                     # pragma: no cover - device dependent
            device_error = "{}: {}".format(type(exc).__name__, exc)

    record = {
        "python": platform.python_version(),
        "python_full": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "numpy": _module_version("numpy"),
        "datasets": _module_version("datasets"),
        "huggingface_hub": _module_version("huggingface_hub"),
        "cuda_available": cuda,
        "device_count": int(torch.cuda.device_count()) if cuda else 0,
        "device_name": name,
        "device_capability": capability,
        "arch_list": arches,
        # Kaggle run 1 was handed a P100 (sm_60) that Kaggle's own torch cannot
        # use: `is_available()` was True and every CUDA op would have failed
        # later. Usability is the arch comparison, never the availability flag.
        "device_usable": device_usable_from(capability, arches) if capability else None,
        # The verbatim membership test, recorded BESIDE the real one because
        # they disagree on the certified local 4060 and the disagreement is a
        # finding about the gate in notebook cell 1, not about this device.
        "device_in_arch_list_verbatim": (capability in arches) if capability else None,
        "device_error": device_error,
        "cuda_initialized": initialized,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "threads": int(torch.get_num_threads()),
        "reference": dict(REFERENCE),
    }
    record["matches_reference"] = matches_reference(record)
    return record


def _module_version(name: str):
    try:
        return getattr(importlib.import_module(name), "__version__", "UNKNOWN")
    except Exception:
        return None


def matches_reference(record: dict) -> bool:
    """Is this the stack the local campaign's numbers were measured on?

    Three fields decide it. `False` is not a failure -- it is the flag that says
    a cross-box `|Delta|` is being read across a stack change as well as a
    device change, which is a different claim from the one the round was
    designed around.
    """
    return all(record.get(k) == REFERENCE[k]
               for k in ("python", "torch", "transformers"))


# ---------------------------------------------- the six private HF attributes

def _tokens(name: str) -> list:
    return [p for p in name.strip("_").split("_")
            if len(p) >= 3 and p not in _GENERIC_TOKENS]


def attr_binding(base, subject, name: str) -> dict:
    """Does `subject`'s override of the private attribute `name` actually bind?

    Pure, so it can be fired at a planted class rather than only at the one
    class that happens to work today. Two distinct failures, reported apart:

      `unknown_to_base`  the name is on NO class in `base`'s MRO -- the release
                         renamed or dropped it, and the subject is setting an
                         attribute nothing reads. `rename_candidates` carries
                         the private names on the base that share a token with
                         it, so a human gets the successor instead of a bare
                         False.
      `honored` False    the name resolves, but to a value the BASE supplied --
                         the subject's override was normalised away (an
                         `__init_subclass__` rewrite is the realistic
                         mechanism). This is the silent one.
    """
    base_mro = list(base.__mro__)
    declared = next((c for c in base_mro if name in c.__dict__), None)
    owner = next((c for c in subject.__mro__ if name in c.__dict__), None)

    candidates: list = []
    if declared is None:
        tokens = _tokens(name)
        seen = set()
        for klass in base_mro:
            for attr in klass.__dict__:
                if (attr.startswith("_") and attr != name and attr not in seen
                        and any(t in attr for t in tokens)):
                    seen.add(attr)
                    candidates.append(attr)
        candidates = sorted(candidates)[:8]

    return {
        "declared_on_base": declared.__name__ if declared is not None else None,
        "base_value": _jsonable(getattr(base, name, "<ABSENT>")),
        "effective_owner": owner.__name__ if owner is not None else None,
        "effective_value": _jsonable(getattr(subject, name, "<ABSENT>")),
        "effective_type": type(getattr(subject, name, None)).__name__,
        "honored": owner is not None and owner not in base_mro,
        "unknown_to_base": declared is None,
        "rename_candidates": candidates,
    }


def ensure_generation_mixin(paths=("transformers.generation",
                                   "transformers.generation.utils",
                                   "transformers")) -> dict:
    """Resolve `GenerationMixin`, and alias it onto the path the model imports.

    `ceq/hf/modeling_ceq.py:94` is `from transformers.generation import
    GenerationMixin`, and that file is read-only to this module. If a release
    moves the symbol, that import raises and the model cannot be built at all.
    `paths[0]` is the path the model asks for; the rest are where the symbol has
    lived. When it is found elsewhere the module object gets the name bound onto
    it, which is an import-path alias to the SAME class -- no numerical
    behaviour changes, and the alias is reported rather than done quietly.

    IDEMPOTENT, and PROCESS-LOCAL: a subprocess gets none of it.
    """
    want = paths[0]
    found_in = None
    klass = None
    for path in paths:
        try:
            module = importlib.import_module(path)
        except Exception:
            continue
        obj = getattr(module, "GenerationMixin", None)
        if isinstance(obj, type):
            found_in, klass = path, obj
            break

    if klass is None:
        return {"import_path": want, "found_in": None, "class_name": None,
                "shim_installed": False,
                "error": "GenerationMixin is on none of {} -- transformers {} "
                         "moved it, and ceq/hf/modeling_ceq.py:94 imports it "
                         "from {!r}".format(list(paths),
                                            _module_version("transformers"), want)}

    shim = False
    if found_in != want:
        try:
            target = importlib.import_module(want)
        except Exception as exc:
            return {"import_path": want, "found_in": found_in,
                    "class_name": klass.__name__, "shim_installed": False,
                    "error": "GenerationMixin found in {} but {} is not "
                             "importable: {}: {}".format(found_in, want,
                                                         type(exc).__name__, exc)}
        setattr(target, "GenerationMixin", klass)
        shim = True

    return {"import_path": want, "found_in": found_in,
            "class_name": klass.__name__, "shim_installed": shim, "error": None}


def transformers_report() -> dict:
    """Which of the six the RUNNING transformers actually honors -- measured."""
    import transformers

    mixin = ensure_generation_mixin()
    if mixin["error"] is not None:
        raise CompatError("K-COMPAT HALT: " + mixin["error"])
    from transformers import PreTrainedModel
    from ceq.hf.modeling_ceq import CEQForCausalLM

    attrs = {n: attr_binding(PreTrainedModel, CEQForCausalLM, n)
             for n in PRIVATE_ATTRS}
    return {
        "transformers": transformers.__version__,
        "reference_transformers": REFERENCE["transformers"],
        "subject": "ceq.hf.modeling_ceq.CEQForCausalLM",
        "private_attrs": attrs,
        "all_honored": all(a["honored"] for a in attrs.values()),
        "not_honored": sorted(n for n, a in attrs.items() if not a["honored"]),
        "unknown_to_base": sorted(n for n, a in attrs.items()
                                  if a["unknown_to_base"]),
        "generation_mixin": mixin,
    }


# ---------------------------------------------------------- determinism regime

def regime(forward_ok: bool, strict_backward_ok: bool,
           warn_only_backward_ok: bool) -> str:
    """The classifier, pure, so it can be fired at planted observations.

    It reads the OBSERVATION and nothing else. A classifier keyed off the torch
    version would reproduce this box's answer on every stack and would report a
    torch that fixed the kernel as though it had not.
    """
    if not forward_ok:
        return "FORWARD_BLOCKED"
    if strict_backward_ok:
        return "STRICT"
    if warn_only_backward_ok:
        return "WARN_ONLY"
    return "BLOCKED"


def _attempt(fn) -> dict:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
        return {"executable": True, "error": None,
                "warned": any("deterministic" in str(w.message) for w in caught)}
    except Exception as exc:
        return {"executable": False, "warned": False,
                "error": "{}: {}".format(type(exc).__name__,
                                         str(exc).splitlines()[0][:300])}


def probe_determinism(device=None, n: int = 4, length: int = 64) -> dict:
    """MEASURE the determinism regime on whatever stack is running. Never inherit it.

    Rulings 1 and 9 rest on `cumsum_cuda_kernel` having no deterministic CUDA
    implementation in torch 2.5.1, which puts the hole in the BACKWARD of
    `cumprod` -- the workhorse arm's reduction -- and is why the round trains
    under `warn_only=True` and holds only forward-only cells to strict mode.
    **torch 2.10 may have fixed that.** If it has, that is a finding that
    improves the round, and it can only surface as a measurement.

    Two ops, because they fail for different reasons and the ruling names both:
    `cumprod` is `ceq/arm_smprime.py`'s hop (the shipped arm), `cumsum` is what
    `arm_phase`/`arm_pl` reduce with and is the kernel actually missing.

    Restores the process-global determinism flags on the way out.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    was_on = torch.are_deterministic_algorithms_enabled()
    was_warn = torch.is_deterministic_algorithms_warn_only_enabled()

    def forward(op):
        def go():
            t = torch.randn(n, length, device=device)
            with torch.no_grad():
                op(t, dim=-1)
        return go

    def backward(op):
        def go():
            t = torch.randn(n, length, device=device, requires_grad=True)
            op(t, dim=-1).sum().backward()
        return go

    ops = {}
    try:
        for label, op in (("cumprod", torch.cumprod), ("cumsum", torch.cumsum)):
            torch.use_deterministic_algorithms(True, warn_only=False)
            entry = {"strict_forward": _attempt(forward(op)),
                     "strict_backward": _attempt(backward(op))}
            torch.use_deterministic_algorithms(True, warn_only=True)
            entry["warn_only_backward"] = _attempt(backward(op))
            ops[label] = entry
    finally:
        torch.use_deterministic_algorithms(was_on, warn_only=was_warn)

    def verdict(label):
        e = ops[label]
        return regime(e["strict_forward"]["executable"],
                      e["strict_backward"]["executable"],
                      e["warn_only_backward"]["executable"])

    cp = ops["cumprod"]
    observed = (cp["strict_forward"]["executable"],
                cp["strict_backward"]["executable"],
                cp["warn_only_backward"]["executable"])
    return {
        "device": device,
        "torch": torch.__version__,
        "shape": [n, length],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "ops": ops,
        "regime": verdict("cumprod"),
        "regime_cumsum": verdict("cumsum"),
        "inherited": "torch 2.5.1+cu121 / RTX 4060, cumprod on CUDA: strict "
                     "forward OK, strict backward RAISES cumsum_cuda_kernel, "
                     "warn_only backward OK (Rulings 1 and 9, COSTS.md 1.6)",
        # None, not False, off CUDA: the inherited claim is CUDA-scoped and a
        # CPU probe is not a refutation of it.
        "matches_inherited": (observed == INHERITED_CUDA_CUMPROD
                              if device == "cuda" else None),
    }


# ------------------------------------------------------------------- selfcheck

def _tiny_forward_backward(device: str = "cpu", operator: str = "smprime") -> dict:
    """One forward and one backward on the WORKHORSE arm, at toy size.

    `operator="smprime"` and not the shipped default, because that branch is the
    one that imports `ceq.arm_smprime` from inside `_smprime` -- a seam that
    exists on no other operator -- and its backward is the one the determinism
    hole is in. Toy shapes: this is a stack check, never a reading.
    """
    from ceq.hf.configuration_ceq import CEQConfig
    from ceq.hf.modeling_ceq import CEQForCausalLM

    torch.manual_seed(0)
    model = CEQForCausalLM(CEQConfig(
        vocab_size=64, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=64,
        operator=operator)).to(device)
    x = torch.randint(0, 64, (2, 16), device=device)
    out = model(input_ids=x, labels=x)
    out.loss.backward()
    grads = [p for p in model.parameters() if p.grad is not None]
    return {
        "operator": operator,
        "device": device,
        "loss": float(out.loss),
        "finite_loss": bool(torch.isfinite(out.loss)),
        "finite_logits": bool(torch.isfinite(out.logits).all()),
        "params": int(sum(p.numel() for p in model.parameters())),
        "params_with_grad": len(grads),
        "finite_grads": bool(all(torch.isfinite(g.grad).all() for g in grads)),
    }


def _halt(what: str, exc: BaseException, record) -> "CompatError":
    return CompatError(
        "K-COMPAT HALT: {}\n  {}: {}\n  stack: python {} / torch {} / "
        "transformers {} (reference: python {} / torch {} / transformers {})\n"
        "  This is a STACK difference, not a research result. Nothing below "
        "this point may run.".format(
            what, type(exc).__name__, exc,
            record.get("python"), record.get("torch"), record.get("transformers"),
            REFERENCE["python"], REFERENCE["torch"], REFERENCE["transformers"]))


def selfcheck(verbose: bool = True, device=None) -> dict:
    """The one minute that a stack mismatch is allowed to cost.

    Imports every module the run needs, builds a tiny model, takes one forward
    and one backward, prints the version record and the MEASURED determinism
    regime. Raises `CompatError` naming the module and the exception on any
    failure -- so a stack difference is diagnosed here and not deep in Q3.
    """
    record = version_record()
    if verbose:
        _print_record(record)

    mixin = ensure_generation_mixin()
    if mixin["error"] is not None:
        raise CompatError("K-COMPAT HALT: " + mixin["error"])

    modules = {}
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise _halt("`import {}` failed.".format(name), exc, record) from exc
        modules[name] = True

    try:
        attrs = transformers_report()
    except Exception as exc:
        raise _halt("the transformers private-attribute report failed.",
                    exc, record) from exc

    try:
        step = _tiny_forward_backward()
    except Exception as exc:
        raise _halt("the tiny model could not take a forward and a backward.",
                    exc, record) from exc
    if not (step["finite_loss"] and step["finite_grads"]
            and step["params_with_grad"] > 0):
        raise CompatError("K-COMPAT HALT: the tiny model ran but its step is not "
                          "usable: {}".format(json.dumps(step)))

    try:
        det = probe_determinism(device=device)
    except Exception as exc:
        raise _halt("the determinism probe failed.", exc, record) from exc

    out = {"version": record, "modules": modules, "transformers": attrs,
           "forward_backward": step, "determinism": det}
    if verbose:
        _print_attrs(attrs)
        _print_determinism(det)
        print("[K-COMPAT OK] {} modules imported, tiny {} step took a forward "
              "and a backward ({} params, {} grads), determinism regime {}"
              .format(len(modules), step["operator"], step["params"],
                      step["params_with_grad"], det["regime"]))
    return out


# ---------------------------------------------------------------------- output

def _print_record(record: dict) -> None:
    print("=" * 78)
    print("K-COMPAT -- version record (MEASURED on the interpreter running this)")
    print("=" * 78)
    for key in ("python", "torch", "torch_cuda", "transformers", "numpy",
                "datasets", "huggingface_hub", "platform", "threads",
                "cublas_workspace_config", "cuda_available", "device_name",
                "device_capability", "device_usable", "cuda_initialized"):
        print("  {:<26} {}".format(key, record[key]))
    print("  {:<26} {}".format("arch_list", record["arch_list"]))
    print("\n  REFERENCE (what the local campaign's numbers were measured on)")
    for key, value in REFERENCE.items():
        print("    {:<24} {}".format(key, value))
    if record["matches_reference"]:
        print("\n  SAME STACK as the reference: a cross-box |Delta| is a DEVICE "
              "delta only.")
    else:
        print("\n  *** DIFFERENT STACK from the reference. Every cross-box "
              "|Delta| below carries a\n      STACK change as well as a device "
              "change, and must say so where it is read. ***")
    if record["device_capability"] and record["device_usable"] is False:
        print("  *** the visible GPU ({}) has no compatible arch in this "
              "torch's build list {}\n      -- CUDA ops will fail. This is "
              "Kaggle run 1's P100 failure. ***"
              .format(record["device_capability"], record["arch_list"]))
    elif (record["device_usable"] is True
            and record["device_in_arch_list_verbatim"] is False):
        print("  note: {} is not listed VERBATIM in {} but is minor-compatible "
              "with it.\n        A gate written as `capability in arch_list` "
              "would halt here wrongly."
              .format(record["device_capability"], record["arch_list"]))
    if (record["cuda_available"]
            and record["cublas_workspace_config"] != ":4096:8"):
        print("  *** CUBLAS_WORKSPACE_CONFIG is {!r}, not ':4096:8'. "
              "`use_deterministic_algorithms(True)`\n      needs it for cuBLAS "
              "reductions, and setting it after CUDA has initialised makes "
              "strict\n      mode fail the FORWARD too -- order dependence, not "
              "an error (V17_R4_RETAKE_PRICE.md).\n      Every strict-mode "
              "reading taken in this process is suspect. ***"
              .format(record["cublas_workspace_config"]))


def _print_attrs(report: dict) -> None:
    print("\nTHE SIX PRIVATE PreTrainedModel ATTRIBUTES, on transformers {} "
          "(reference {})".format(report["transformers"],
                                  report["reference_transformers"]))
    for name, entry in report["private_attrs"].items():
        print("  {:<24} honored={:<5} on_base={:<20} value={}".format(
            name, str(entry["honored"]), str(entry["declared_on_base"]),
            entry["effective_value"]))
        if entry["unknown_to_base"]:
            print("      *** NOT DECLARED BY ANY CLASS IN PreTrainedModel's MRO. "
                  "This transformers does not\n          read it; the override "
                  "is inert. Candidates that may have replaced it: {}"
                  .format(entry["rename_candidates"] or "none found"))
        elif not entry["honored"]:
            print("      *** the override did NOT bind -- the effective value "
                  "comes from {} ***".format(entry["effective_owner"]))
    mixin = report["generation_mixin"]
    print("  GenerationMixin           imported from {!r}, found in {!r}, "
          "alias installed: {}".format(mixin["import_path"], mixin["found_in"],
                                       mixin["shim_installed"]))


def _print_determinism(det: dict) -> None:
    print("\nDETERMINISM REGIME -- MEASURED on this stack, device {}, torch {}"
          .format(det["device"], det["torch"]))
    for op, entry in det["ops"].items():
        for phase in ("strict_forward", "strict_backward", "warn_only_backward"):
            r = entry[phase]
            print("  {:<8} {:<20} executable={:<5} {}".format(
                op, phase, str(r["executable"]),
                r["error"] or ("(warned)" if r["warned"] else "")))
    print("  regime(cumprod, the shipped arm's reduction): {}".format(det["regime"]))
    print("  regime(cumsum, arm_phase/arm_pl's reduction): {}"
          .format(det["regime_cumsum"]))
    print("  inherited: {}".format(det["inherited"]))
    if det["matches_inherited"] is None:
        print("  matches_inherited: N/A -- the inherited claim is CUDA-scoped "
              "and this probe ran on {}".format(det["device"]))
    elif det["matches_inherited"]:
        print("  matches_inherited: TRUE -- Rulings 1 and 9's basis reproduces "
              "on this stack.")
    else:
        print("  *** matches_inherited: FALSE -- this stack does NOT behave as "
              "the stack Rulings 1\n      and 9 were measured on. That is a "
              "FINDING, and every determinism claim in the\n      round must be "
              "re-read against the table above before it is quoted. ***")


def main(argv=None) -> int:
    try:
        selfcheck(verbose=True)
    except CompatError as exc:
        print("\n" + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
