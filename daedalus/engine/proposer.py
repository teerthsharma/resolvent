"""Candidate contract and the M1 proposer stub (no LLM is called at M0).

A candidate is a directory the proposer writes; the verifier (verifier.py) is
the only reader of truth and the only trainer. The proposer has no write path
into engine/, registry/ or sealed/.

manifest.json
    name        str
    kind        "operator" | "bed"
    bed         operator only: a registered bed name ("a5_word_T16")
    entry       python file, default "candidate.py"
    claims      optional list, each needs a V1 spec: "causal", "row_stochastic", "pole_free"
    reported    optional {"band_acc": float} -- re-measured by V4 (L-REPRO)
    citations   optional list of owner ids; each must be a fetched owner in owners.json
    family      bed only: "pointer_chase" | "group_reset_x3" | "custom"
    params      bed only: family parameters
  Forbidden anywhere in the manifest (the verifier owns them): bar(s), threshold(s),
  margin, opponent(s), baseline(s), seed(s), steps, lr, budget, eval, test.

candidate.py (operator)
    build(spec) -> torch.nn.Module; spec = {"vocab", "n_classes", "T", "param_budget"}
        forward(tokens LongTensor[B, T]) -> logits [B, T, n_classes]; causal.
        trainable params within +-2% of spec["param_budget"]; buffers <= 10% of it.
    mixing_weights(model, tokens) -> Tensor[..., T, T]      required iff claims row_stochastic
    resolvent_params(model, tokens) -> (W[..., T, T], gamma) required iff claims pole_free
  Pure module: no file, network or process I/O.

candidate.py (bed, family "custom")
    generate(n, seed) -> (X[n, T] int, Y[n, T] int, band[T] bool)
"""
import json
import os


def propose(parent_dirs, out_dir):  # M1: an LLM writes a diff of a parent candidate here
    raise NotImplementedError("M1: proposer not started at M0 (no LLM tokens spent)")


def write_candidate(out_dir, manifest, source):
    """The only write path the proposer has: a fresh candidate directory."""
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, manifest.get("entry", "candidate.py")), "w") as f:
        f.write(source)
    return out_dir
