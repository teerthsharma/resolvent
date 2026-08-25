"""CPU eval harness: the model, a matched softmax control, a published number.

Runs on CPU. Needs no GPU, no Triton, no `trust_remote_code`, and no network
once `data/scan_*.txt` is cached. That is deliberate: the free-Colab T4 is sm_75
and cannot run this project's Triton kernel at all, so an eval that needs the
kernel is an eval most readers cannot run.

WHY SCAN. It is exact-match scored, like ARC-AGI, but its length split has 3,920
test items against ARC-AGI-2's 120 tasks. `tests/cameron/test_arc_reality.py`
measures what that costs: ARC needs 5 solved tasks out of 120 before a gap
clears p < 0.05, so two arms near zero return Fisher p = 1.0 and the run says
nothing. SCAN resolves a gap over thirty times smaller for the same exact-match
scoring rule, and it has a published from-scratch transformer baseline that a
parameter-matched model is legitimately comparable to.

WHAT THIS HARNESS REFUSES TO DO. There is no frontier-model column. A 0.5B model
against o3 or Claude Opus is a category error unless it is framed as
parameter-matched, and at that framing the frontier number is not the
comparison -- the from-scratch transformer on the same split is. `run()` raises
rather than build a table for a split with no published reference, so the third
column cannot quietly go missing.

THE SCORER SHIPS WITH IT. `exact_match` is five lines and it is the whole
difference between a dataset someone can use and one they cannot: `openai/gsm8k`
sees 1,076,823 downloads/month, `tasksource/corr2cause` 49.
"""
from __future__ import annotations

import pathlib

import torch
import torch.nn.functional as F

from . import lm

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
SEQ = 64

# Published reference points. Every entry names a paper and the model that
# produced the number. A reference without a citation is a memory, and this
# project has already had to strike claims that turned out to be memories.
REFERENCES = {
    ("length", "transformer_from_scratch"): dict(
        score=0.00, sd=None, model="vanilla Transformer, trained from scratch",
        source="arXiv:2108.12284",
        note="Csordas, Irie & Schmidhuber 2021, SCAN length cutoff=26. Their "
             "relative-positional Universal Transformer reaches 1.00 +/- 0.00 "
             "on the same split, so the split has a demonstrated ceiling and a "
             "floor of exactly zero -- maximum resolving power."),
    ("length", "rpe_universal_transformer"): dict(
        score=1.00, sd=0.00, model="relative-positional Universal Transformer",
        source="arXiv:2108.12284",
        note="the ceiling: this split is solvable, so a zero is a real failure "
             "rather than an impossible task."),
    ("addprim_jump", "transformer_from_scratch"): dict(
        score=0.034, sem=0.020, sd=None, model="unmodified Transformer",
        source="arXiv:2107.01366",
        note="Chaabouni, Dessi & Kharitonov, Table 3, jump column: 3.4 +/- 2.0, "
             "and the +/- is 1 SEM, NOT a standard deviation -- this file said "
             "sd for a round. Best hyperparameter configuration per architecture. "
             "NONZERO, unlike the length split, so there is a baseline to beat "
             "rather than a floor to share. Lake & Baroni's original "
             "LSTM+attention best was 0.012 (arXiv:1711.00350, prose, not a "
             "table)."),
    ("addprim_jump", "second_vanilla_baseline"): dict(
        score=0.003, sd=0.004, model="Transformer, absolute position encodings",
        source="arXiv:2108.04378",
        note="Ontanon, Ainslie, Cvicek & Fisher, Tables 1/8 mean and Table 10 "
             "sd, SCAN-aj row `abs`. A SECOND published vanilla number on the "
             "same split, an order of magnitude below the first. Two papers "
             "disagree about what an unmodified Transformer scores here by "
             "0.003 against 0.034, which is the honest width of this reference "
             "and is wider than any margin a single run of this harness could "
             "claim. Their own best config reaches 0.017 mean / 0.039 max."),
    ("addprim_jump", "altered_attention"): dict(
        score=0.430, sem=0.095, sd=None,
        model="Transformer + T5-style relative-position attention bias",
        source="arXiv:2107.01366",
        note="Same Table 3, `+T5` row, jump column: 43.0 +/- 9.5 SEM from 3.4. "
             "THE STRONGER PRECEDENT. Changing the attention moves SCAN "
             "addprim_jump by ~40 points, against ~4 points on COGS "
             "sequence-generation (arXiv:2108.12284 Table 3). If the question "
             "is which benchmark responds most to an attention change, the "
             "answer measured here is this split, not COGS."),

    # COGS is the benchmark this harness recommends first. Its from-scratch
    # baseline is nonzero AND far from the ceiling, and somebody has already
    # changed the attention mechanism on it and moved the number 52 points.
    ("cogs", "transformer_from_scratch"): dict(
        score=0.35, sd=0.06, model="2-layer encoder-decoder Transformer, 9.5M params",
        source="arXiv:2010.05465",
        note="Kim & Linzen 2020, Table 2, generalization set, 21,000 items, 5 "
             "seeds. The SAME row reads Dev 0.96 / Test 0.96, so 0.35 is a "
             "generalization gap and not an unlearnable task. ENCODER-DECODER: "
             "2 encoder + 2 decoder layers, 4 heads, ff 512. Scoring is exact "
             "match on the whole logical form."),
    ("cogs", "in_distribution_ceiling"): dict(
        score=0.96, sd=None, model="the same 2-layer encoder-decoder Transformer",
        source="arXiv:2010.05465",
        note="Table 2, Test column. The resolution gate: a run whose control "
             "cannot learn the in-distribution split has no business reporting "
             "a generalization column. The 0.96-0.99 range quoted in the "
             "abstract spans all three models; the Transformer's own number is "
             "0.96."),
    ("cogs", "altered_attention"): dict(
        score=0.81, sd=0.01, model="relative-positional Transformer",
        source="arXiv:2108.12284",
        note="Csordas, Irie & Schmidhuber 2021, Table 3. THE PRECEDENT, and it "
             "is a modest one stated honestly: within their own tuned setup the "
             "attention variants span Trafo 0.80 +/- 0.00, Rel. Trafo 0.81 +/- "
             "0.01, Uni. Trafo 0.78 +/- 0.03, Rel. Uni. Trafo 0.77 +/- 0.01. So "
             "changing the attention moves this benchmark by about 4 points at "
             "a fixed training recipe -- not the 52 points a previous draft of "
             "this file claimed."),
    ("cogs", "edge_transformer_graph_prediction"): dict(
        score=0.874, sd=0.004, model="Edge Transformer (triangular attention)",
        source="arXiv:2112.00578",
        note="CORRECTION, iteration 17. Bergen, O'Donnell & Bahdanau 2021 Table "
             "4 TOP section is the GRAPH-PREDICTION version of COGS, not the "
             "sequence-generation task scored here, and its 0.784 Universal "
             "Transformer comparison row is quoted from Ontanon et al. 2021 "
             "with no sd. This project cited 0.874-against-0.784 as a "
             "sequence-generation precedent for three rounds. It is not one. "
             "Kept in the table so the retraction is visible rather than "
             "deleted."),
    ("cogs", "no_decoder_only_reference"): dict(
        score=None, sd=None, model="none published",
        source="none",
        note="Searched and not found: no COGS number for a decoder-only causal "
             "LM trained from scratch. arXiv:2310.19956 uses decoder-only "
             "models but pretrains them on C4 for 131B tokens first. The model "
             "in this harness is decoder-only and from scratch, so the 0.35 is "
             "an ARCHITECTURE-MISMATCHED reference and must be read as a "
             "landmark, never as a matched control. The matched control is the "
             "softmax arm in the same table."),
}

# COGS. Measured on the vendored files rather than quoted: 874 word types over
# train+test+gen once PAD/SEP/EOS are added, longest TRAIN sequence 177 tokens,
# longest GENERALIZATION sequence 543. COGS_SEQ covers the training split whole
# -- nothing on the supervised side is ever truncated -- and the decoder slides
# its window for the generalization items that run past it, which is the same
# sliding window both arms get.
COGS_VOCAB, COGS_MAX_LEN, COGS_SEQ = 874, 543, 192
EVAL_WHICH = {'cogs': 'gen'}   # COGS scores on the generalization set


# ------------------------------------------------------------------ the data

def load_scan(split: str, which: str) -> list[tuple[list[str], list[str]]]:
    """Parse the canonical SCAN files: `IN: <command> OUT: <actions>`.

    Vendored from github.com/brendenlake/SCAN rather than regenerated from the
    grammar. A locally regenerated SCAN would be a different dataset wearing the
    same name, and every published baseline attached to it would become a guess.
    """
    path = DATA / f"scan_{split}_{which}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; fetch the SCAN splits first")
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cmd, act = line.split(" OUT: ")
        pairs.append((cmd[len("IN: "):].split(), act.split()))
    return pairs


def load_cogs(which: str) -> list[tuple[list[str], list[str]]]:
    """COGS as (sentence tokens, logical-form tokens).

    Vendored from github.com/najoungkim/COGS, the canonical repository, rather
    than the community Hub mirror -- `Punchwe/COGS` sees 56 downloads a month
    and is not the authors' upload, so a silent difference there would break
    comparability with the published 0.35.

    The third TSV column is the generalization category; kept out of the return
    because scoring is exact match on the logical form, but it is what would
    let a later run report per-category accuracy instead of one aggregate.
    """
    path = DATA / f"cogs_{which}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; fetch the COGS splits first")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src, tgt, _category = line.split("	")
        out.append((src.split(), tgt.split()))
    return out


def load_pairs(split: str, which: str) -> list[tuple[list[str], list[str]]]:
    """SCAN and COGS behind one call, so nothing downstream branches on split."""
    return load_cogs(which) if split == "cogs" else load_scan(split, which)


def split_seq(split: str) -> int:
    return COGS_SEQ if split == "cogs" else SEQ


def exact_match(pred: list[str], gold: list[str]) -> bool:
    """SCAN's scoring rule. Whole sequence, in order, no partial credit.

    Partial credit would report a healthy number for a model emitting the right
    actions in the wrong order, which is the exact failure SCAN exists to catch.
    """
    return list(pred) == list(gold)


# --------------------------------------------------------------- tokenisation

SEP, EOS, PAD = "<sep>", "<eos>", "<pad>"


def build_vocab(split: str) -> dict[str, int]:
    """Word level, not byte level. SCAN's vocabulary is about 20 words, so a
    byte tokenizer would make every sequence 8x longer for nothing and push the
    length split past any reasonable context.

    Ids stay under `lm.VOCAB` so `lm.TinyLM` is reused unmodified -- the arms
    being compared must be the arms this project actually ships, not a local
    reimplementation that could drift from them.
    """
    words = {PAD, SEP, EOS}
    for which in ("train", "test", "gen") if split == "cogs" else ("train", "test"):
        for cmd, act in load_pairs(split, which):
            words.update(cmd)
            words.update(act)
    return {w: i for i, w in enumerate(sorted(words))}


def encode(pair, vocab: dict[str, int]) -> list[int]:
    cmd, act = pair
    return ([vocab[w] for w in cmd] + [vocab[SEP]]
            + [vocab[w] for w in act] + [vocab[EOS]])


def _batch(pairs, vocab, idx, device, seq: int = SEQ):
    rows = [encode(pairs[i], vocab)[:seq + 1] for i in idx]
    pad = vocab[PAD]
    rows = [r + [pad] * (seq + 1 - len(r)) for r in rows]
    t = torch.tensor(rows, dtype=torch.long, device=device)
    return t[:, :-1], t[:, 1:]


# --------------------------------------------------------------------- eval

@torch.no_grad()
def greedy_batch(model, cmds, vocab, inv, device, max_new: int = 48,
                 seq: int | None = None, chunk: int = 64) -> list[list[str]]:
    """Greedy decode many prompts at once, stopping each at its own `<eos>`.

    COGS has 21,000 generalization items and logical forms with a median of 52
    tokens. One forward pass per output token per item is not a run anybody
    finishes, and an eval nobody finishes is an eval nobody checks.

    Right-padding is exact here rather than approximately exact, and that is
    load-bearing. Both arms are strictly causal -- the softmax arm through
    `is_causal=True`, the signed arms through `.tril(-1)` -- so a pad token at a
    later position cannot reach an earlier one, and every row of the batch gets
    bit-identical logits to the same prompt decoded alone.
    `test_greedy_decode_batched_matches_one_at_a_time` is that check.

    Decoding is restricted to the task alphabet. The head is sized to the task
    vocabulary now, but SCAN's 27 words still sit inside a wider table once PAD
    and the sentinels are counted, and an untrained model happily argmaxes into
    ids that mean nothing here. Masking removes that artifact IDENTICALLY for
    both arms, so it cannot favour either.
    """
    seq = seq or model.seq
    valid = torch.full((model.vocab,), float("-inf"), device=device)
    valid[torch.tensor(sorted(vocab.values()), device=device)] = 0.0
    pad, sep, eos = vocab[PAD], vocab[SEP], vocab[EOS]

    out: list[list[str]] = []
    for start in range(0, len(cmds), chunk):
        grp = cmds[start:start + chunk]
        ids = [[vocab[w] for w in c] + [sep] for c in grp]
        acts: list[list[str]] = [[] for _ in grp]
        live = list(range(len(grp)))
        for _ in range(max_new):
            if not live:
                break
            win = [ids[b][-seq:] for b in live]
            width = max(len(w) for w in win)
            batch = torch.full((len(win), width), pad, dtype=torch.long, device=device)
            for r, w in enumerate(win):
                batch[r, :len(w)] = torch.tensor(w, dtype=torch.long, device=device)
            logits = model(batch)
            picks = (logits[torch.arange(len(win), device=device),
                            torch.tensor([len(w) - 1 for w in win], device=device)]
                     + valid).argmax(-1).tolist()
            still = []
            for r, b in enumerate(live):
                if picks[r] == eos:
                    continue
                ids[b].append(picks[r])
                acts[b].append(inv[picks[r]])
                still.append(b)
            live = still
        out.extend(acts)
    return out


def greedy_actions(model, cmd: list[str], vocab: dict[str, int], inv, device,
                   max_new: int = 48, seq: int | None = None) -> list[str]:
    """One prompt. The single-item spelling of `greedy_batch`, and its oracle."""
    return greedy_batch(model, [cmd], vocab, inv, device, max_new=max_new,
                        seq=seq)[0]


def eval_indices(n_test: int, max_eval: int, seed: int) -> list[int]:
    """A fixed random subsample, not the first `max_eval` rows.

    `data/cogs_gen.tsv` is ordered BY GENERALIZATION CATEGORY. Taking the first
    256 rows would score one category and call it COGS. The generator is seeded
    off the split size alone, so both arms are scored on exactly the same items
    at every seed and the subsample cannot drift between them.
    """
    if max_eval >= n_test:
        return list(range(n_test))
    g = torch.Generator().manual_seed(20260825 + n_test)
    return torch.randperm(n_test, generator=g)[:max_eval].sort().values.tolist()


def train_model(kind: str, split: str, *, steps: int, device, seed: int = 0,
                bs: int = 32, lr: float = 3e-4):
    """Train one arm. Identical budget for every arm: same steps, lr, batch, seed.

    Kept separate from scoring so one trained model can answer more than one
    question. COGS asks two -- can the control learn the in-distribution split
    at all, and what happens on the generalization split -- and they have to be
    the same model or the first does not gate the second.
    """
    vocab = build_vocab(split)
    seq = split_seq(split)
    train = load_pairs(split, "train")

    model = lm.TinyLM(kind, seq=seq, vocab=len(vocab), seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)

    model.train()
    loss = torch.zeros((), device=device)
    for _ in range(steps):
        idx = torch.randint(len(train), (bs,), generator=g).tolist()
        x, y = _batch(train, vocab, idx, device, seq)
        loss = F.cross_entropy(model(x).reshape(-1, len(vocab)), y.reshape(-1),
                               ignore_index=vocab[PAD])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.train_loss = float(loss)
    return model.eval()


@torch.no_grad()
def score(model, split: str, which: str, *, device, max_eval: int = 256,
          seed: int = 0, max_new: int = 128) -> dict:
    """Exact match on `which`, over a fixed subsample identical across arms."""
    vocab = build_vocab(split)
    inv = {i: w for w, i in vocab.items()}
    test = load_pairs(split, which)
    pick = eval_indices(len(test), max_eval, seed)
    preds = greedy_batch(model, [test[i][0] for i in pick], vocab, inv, device,
                         max_new=max_new, seq=split_seq(split))
    solved = sum(exact_match(p, test[i][1]) for p, i in zip(preds, pick))
    return dict(kind=model.kind, exact_match=solved / len(pick), n_solved=solved,
                n_eval=len(pick), eval_split=which, n_params=model.n_params(),
                vocab=len(vocab), seq=model.seq,
                train_loss=getattr(model, "train_loss", None))


def train_and_score(kind: str, split: str, *, steps: int, device, seed: int = 0,
                    bs: int = 32, lr: float = 3e-4, max_eval: int = 256,
                    eval_which: str | None = None, max_new: int = 128) -> dict:
    """One arm, trained and scored on its default split."""
    model = train_model(kind, split, steps=steps, device=device, seed=seed,
                        bs=bs, lr=lr)
    return score(model, split, eval_which or EVAL_WHICH.get(split, "test"),
                 device=device, max_eval=max_eval, seed=seed, max_new=max_new)


def run(split: str, *, steps: int, device, seed: int = 0, max_eval: int = 256,
        arms: tuple[str, ...] = ("softmax", "sgate"), lr: float = 3e-4,
        eval_which: str | None = None, max_new: int = 128) -> dict:
    """The table. Three columns, or no table.

    Refuses a split with no published reference rather than emit a two-column
    result that invites the reader to supply a baseline from memory.
    """
    ref = REFERENCES.get((split, "transformer_from_scratch"))
    if ref is None:
        raise KeyError(
            f"no published from-scratch reference for split {split!r}; "
            f"have {sorted(k for k, _ in REFERENCES)}. Add one with a citation "
            "or pick a split that has one -- the comparison column is not "
            "optional.")
    return dict(
        split=split, steps=steps, seed=seed, device=str(device).split(":")[0],
        arms={k: train_and_score(k, split, steps=steps, device=device,
                                 seed=seed, max_eval=max_eval, lr=lr,
                                 eval_which=eval_which, max_new=max_new)
              for k in arms},
        reference=ref,
        # framing, carried in the artifact so it cannot be dropped in the retell
        comparison="parameter-matched from-scratch, NOT frontier-model")
