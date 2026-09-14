"""ceqjepa/beds/english.py -- phase-2 (ENGLISH) bed for the chess -> English
-> stocks/prediction-markets curriculum.

ENGLISH IS THE INPUT INTERFACE, NOT A GENERATION CAPABILITY. TinyCEQ at the
measured DCM-1 geometry is 315,992 parameters; GPT-2 small is 124,000,000 --
392x larger -- and 315,992 parameters cannot generate fluent English. Any
design that promises free-form English generation from this model is wrong.
What this phase trains is an encoder that reads an English question
("will white win from this position?", "what is the safest move?") into the
SAME committor / state-read structure the chess phase trained, so the demo
is committor-in, template-rendered-answer-out -- never token-by-token
generation.

INTERFACE PARITY WITH ceqjepa.train.Bed. This class exposes the identical
`batch(gen, B) -> (x, x_nx, q_star, v_idx)` and `q_floor()` methods as
ceqjepa/train.py's synthetic Bed, so a caller (a future phase-2 script; this
file does not edit train.py) can point TinyCEQ's training loop at either bed
with no interface change. `x`, `x_nx` are float32 [B, x_dim]; `q_star` is
float32 [B, nA]; `v_idx` is long [B].

ABSORBING SETS FOR TEXT -- THE DECISION, AND WHAT IT COSTS.
Text does not resolve the way a game does: there is no win/loss/draw a
sentence is heading toward. What a sentence in progress DOES resolve into is
one of a small number of terminal punctuation marks, so this bed uses that
as the "discrete continuation classes" option named in the brief, rather
than bare end-of-sequence or a single generic sentence-boundary flag --
generic EOS collapses '.', '!' and '?' into one undifferentiated class and
throws away a distinction the corpus actually carries.

Measured on data/tinystories_20k.txt (17,930,904 chars): '.' occurs 345,457
times, '!' 31,494 times, '?' 11,990 times -- roughly 88.6% / 8.1% / 3.1% of
the three marks combined. That imbalance is a real cost: a collapsed encoder
that always predicts the class prior gets most of its committor mass right
for free on '.', and the '?' class is the one likely to be noisiest at small
sample sizes. It is reported here, not hidden.

A 4th, always-absorbing abstract index 0 is kept as "the sink", matching
ceqjepa.operator's own convention (absorbing_teleport requires index 0 to be
visible from every row for the teleport target to be well-formed, and
q_floor_closed_form raises unless 0 is absorbing). Index 0 carries no
linguistic meaning; indices 1..3 are '.', '!', '?' in that fixed order. So
nA = 4: q_star[:, 0] is sink mass (small, from the shared teleport/prior
baseline), q_star[:, 1:4] is the real per-example committor over the three
sentence-terminal classes.

THE GROUND-TRUTH CHAIN THIS BED SOLVES, EXACTLY, VIA THE SHARED OPERATOR.
Each example is a window of n abstract causal positions: the first nA are
the abstract absorbing indices above (no text content), and positions
nA..n-1 map one-to-one, in reading order, onto n - nA REAL consecutive
characters drawn from tinystories_20k.txt starting at a random offset. The
causal logits driving each transient row i are:
  - a small fixed random baseline (self.L0, seeded once at construction,
    shared by every example -- this bed's analogue of Bed's L_env prior);
  - PLUS, into the three linguistic absorbing columns only, -ALPHA * d,
    where d is the REAL number of characters from position i's actual
    character forward to the next real occurrence of that class in the
    actual corpus text (capped at LOOKAHEAD chars). Nearer real punctuation
    gets a less-negative logit and so more softmax mass.
No random synthetic latent stands in for the text: d is computed by
scanning the real string. ceqjepa.operator.build_operator then makes this
row-stochastic and causal, and ceqjepa.operator.committor solves
q = (I - Q)^{-1} R EXACTLY (float64) -- the same solve the chess bed and the
synthetic Bed use, never touched by the model. This is a genuine, if
simple, real-text-grounded committor: "given real upcoming punctuation
distances at this point in real English, which terminal class is reached
first" -- not a claim about narrative meaning or plot resolution, which is
a different and much harder question this bed does not attempt.

x / x_nx: a dependency-free byte-hash feature of a short trailing character
context ending at the query position (`x`) and at the very next real
character (`x_nx`) -- the next-token-style observation pair the state-read
loss (L_z) supervises against, built from real corpus bytes, no teacher.

SECOND PATH -- FROZEN TEACHER, GATED, NOT EXERCISED LOCALLY. Local torch is
2.14.0+cpu and `transformers` is BROKEN locally (PreTrainedModel import
fails), so this path is written but never run here. Pass
`teacher=callable` to EnglishBed.__init__: a callable taking a list of
`(text, char_pos)` pairs and returning a `[len(list), x_dim]` float tensor.
On Kaggle (where transformers works), that callable would be:
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModel.from_pretrained("distilgpt2").eval()
    with torch.no_grad():
        hidden = model(**tok(texts, return_tensors="pt")).last_hidden_state
    feat = hidden[:, -1, :] @ fixed_random_projection   # frozen, [768] -> x_dim
frozen (`.eval()`, no grad, no fine-tuning) and late-bound (imported only
inside this branch, so its absence never breaks the default path). Nothing
else in this file imports transformers.

torch only (+ stdlib pathlib); no numpy needed.
"""
from __future__ import annotations

import pathlib

import torch

import ceqjepa.operator as op

__all__ = ["EnglishBed", "CLASSES", "NA"]

_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PATH = _ROOT / "data" / "tinystories_20k.txt"

CLASSES = (".", "!", "?")          # absorbing indices 1, 2, 3, in this order
NA_SINK = 1                        # abstract index 0, per operator.py's own sink convention
NA = NA_SINK + len(CLASSES)        # = 4
LOOKAHEAD = 2048                   # cap on the forward scan for "distance to next mark"
ALPHA = 0.02                       # ponytail: hand-picked decay, tune if S(text) reads flat


class EnglishBed:
    """Phase-2 English bed. See module docstring for the full design and
    its cost. `n` is total abstract causal positions (>= NA + 1 so at least
    one real transient position exists); `x_dim` is the observation width."""

    def __init__(self, n=32, x_dim=8, seed=0, path=DEFAULT_PATH, dtype=torch.float64,
                 teacher=None):
        if n <= NA:
            raise ValueError(f"n={n} must exceed NA={NA} (need at least one real "
                              "transient character position)")
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- the default path expects "
                "data/tinystories_20k.txt at the repo root")
        self.text = path.read_text(encoding="utf-8", errors="replace")
        if len(self.text) < LOOKAHEAD + n:
            raise ValueError(f"{path} is too short ({len(self.text)} chars) for "
                              f"n={n}, LOOKAHEAD={LOOKAHEAD}")
        self.n, self.nA, self.x_dim, self.dtype = n, NA, x_dim, dtype
        self.absorbing_idx = torch.arange(NA)
        self.teacher = teacher  # optional frozen callable, see module docstring; unused by default
        if teacher is not None and not callable(teacher):
            raise TypeError("teacher must be a callable(list[(text, char_pos)]) -> "
                             f"[len, x_dim] float tensor, got {type(teacher)!r}")
        g = torch.Generator().manual_seed(seed)
        # Fixed random baseline prior, shared across every example -- this bed's
        # analogue of train.py's Bed.L_env. Small scale: the real per-example
        # signal is the punctuation-distance term added in batch(), not this prior.
        self.L0 = torch.randn(n, n, generator=g, dtype=dtype) * 0.1

    # -- real-text featurisers, dependency-free -----------------------------
    def _dist_to_classes(self, char_pos):
        """Real char distance from `char_pos` forward to each class in
        CLASSES, scanning the actual corpus text, capped at LOOKAHEAD."""
        window = self.text[char_pos:char_pos + LOOKAHEAD]
        out = []
        for ch in CLASSES:
            j = window.find(ch)
            out.append(float(j if j >= 0 else LOOKAHEAD))
        return out

    def _byte_feat(self, char_pos):
        """Deterministic byte-hash context feature: last <=8 real chars up to
        and including char_pos, hashed into x_dim buckets. No learned embedding,
        no randomness beyond the fixed hash -- the default, teacher-free path."""
        lo = max(0, char_pos - 8)
        ctx = self.text[lo:char_pos + 1]
        v = torch.zeros(self.x_dim, dtype=torch.float32)
        for k, ch in enumerate(ctx):
            v[(ord(ch) * 131 + k) % self.x_dim] += 1.0
        n_ctx = max(1, len(ctx))
        return v / n_ctx

    def _feat(self, char_pos):
        if self.teacher is not None:
            return self.teacher([(self.text, char_pos)])[0]
        return self._byte_feat(char_pos)

    def batch(self, gen, B):
        """One batch, ONE solve. Returns (x, x_nx, q_star, v_idx) matching
        ceqjepa.train.Bed.batch's interface exactly."""
        n, nA, n_t = self.n, self.nA, self.n - self.nA
        hi = len(self.text) - n_t - LOOKAHEAD - 1
        starts = torch.randint(0, hi, (B,), generator=gen)
        v_idx = torch.randint(nA, n, (B,), generator=gen)
        logits = self.L0.unsqueeze(0).repeat(B, 1, 1)
        x = torch.zeros(B, self.x_dim, dtype=torch.float32)
        x_nx = torch.zeros(B, self.x_dim, dtype=torch.float32)
        for b in range(B):
            s = int(starts[b])
            for i in range(nA, n):
                char_pos = s + (i - nA)
                d = self._dist_to_classes(char_pos)
                for k in range(len(CLASSES)):
                    logits[b, i, NA_SINK + k] = -ALPHA * d[k]
            vi = int(v_idx[b])
            char_pos_v = s + (vi - nA)
            x[b] = self._feat(char_pos_v)
            x_nx[b] = self._feat(char_pos_v + 1)
        P = op.build_operator(logits, self.absorbing_idx)          # [B,n,n]
        q_full = op.committor(P, self.absorbing_idx)                # [B,n,nA], EXACT
        rows = torch.arange(B)
        q_star = q_full[rows, v_idx].float()
        return x, x_nx, q_star, v_idx

    def q_floor(self):
        """Closed-form committor of the uniform causal chain over this bed's
        (n, absorbing_idx) -- no encoder, no solve. Requires index 0
        absorbing, which the sink convention above guarantees."""
        return op.q_floor_closed_form(self.n, self.absorbing_idx, dtype=torch.float32)


if __name__ == "__main__":
    torch.manual_seed(0)
    bed = EnglishBed(n=32, x_dim=8, seed=0)
    gen = torch.Generator().manual_seed(0)
    x, x_nx, q_star, v_idx = bed.batch(gen, 8)

    print(f"[english bed] path={DEFAULT_PATH} corpus_chars={len(bed.text)}")
    print(f"[english bed] n={bed.n} nA={bed.nA} classes={CLASSES} x_dim={bed.x_dim}")
    print(f"x.shape={tuple(x.shape)} x_nx.shape={tuple(x_nx.shape)} "
          f"q_star.shape={tuple(q_star.shape)} v_idx.shape={tuple(v_idx.shape)}")
    print("v_idx:", v_idx.tolist())
    print("q_star (sink, '.', '!', '?'):")
    for row in q_star.tolist():
        print("  ", [round(c, 4) for c in row])
    print("q_star row sums (must be 1, real committor is a proper distribution):",
          [round(s, 6) for s in q_star.sum(-1).tolist()])
    print("mean class mass over batch:", [round(c, 4) for c in q_star.mean(0).tolist()],
          "  -- corpus marginal frequency of ('.', '!', '?') is roughly "
          "(0.886, 0.081, 0.031); sink+prior noise means these will not match exactly")
    print("x[0]:", [round(v, 3) for v in x[0].tolist()])
    print("x_nx[0]:", [round(v, 3) for v in x_nx[0].tolist()])

    # self-check: shapes, finiteness, and that q_star is a real probability vector
    assert x.shape == (8, 8) and x_nx.shape == (8, 8)
    assert q_star.shape == (8, NA) and v_idx.shape == (8,)
    assert torch.isfinite(x).all() and torch.isfinite(x_nx).all() and torch.isfinite(q_star).all()
    assert torch.allclose(q_star.sum(-1), torch.ones(8), atol=1e-4), \
        "committor rows must sum to 1 -- absorption is certain on this finite triangular chain"
    assert (q_star >= -1e-6).all() and (q_star <= 1 + 1e-6).all()

    floor = bed.q_floor()
    assert floor.shape == (bed.n, NA)
    assert torch.allclose(floor.sum(-1), torch.ones(bed.n), atol=1e-4)
    print(f"[english bed] q_floor shape={tuple(floor.shape)} OK, sums to 1 per row")

    teacher_raised = False
    try:
        EnglishBed(n=32, x_dim=8, teacher="not callable")
    except TypeError as e:
        teacher_raised = True
        print(f"[english bed] non-callable teacher correctly rejected: {e}")
    assert teacher_raised

    print("ALL SELF-CHECKS PASSED")
