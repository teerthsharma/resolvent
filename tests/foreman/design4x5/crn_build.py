"""CRN order file builder + pairing prover. Step (1) of design4x5.

Builds ONE data order per split seed in {0,1,2,3,4}: the document
train/val index order DocByteBatches (r3_eval.py, REUSED not reimplemented)
produces for that split_seed, plus a SHA256 digest of that order. Writes
crn_order.json. Then PROVES the pairing: constructs DocByteBatches twice
per split_seed (simulating two different arms loading data) and confirms
byte-identical train/val tensors and byte-identical first-K train batches
drawn off gen=torch.Generator().manual_seed(seed+1) -- the exact generator
r3_eval.train_with_eval() uses, with split_seed TIED to seed per the design.

No GPU touched: this is data-layer only (index arithmetic + byte tensors),
so it runs regardless of what the shared card is doing.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys
import time

REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
BOARD = r"C:\Users\seal\Desktop\New folder (32)\house-events.jsonl"
CRN_PATH = os.path.join(SCRATCH, "crn_order.json")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "foreman", "eval"))

import torch  # noqa: E402
import r3_eval as RE  # noqa: E402 -- REUSED, not edited


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Foreman",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", row, flush=True)


SPLIT_SEEDS = [0, 1, 2, 3, 4]
VAL_FRAC = 0.1
K_PROOF_BATCHES = 8   # "a handful of steps"
BATCH, SEQ = 8, 512


def doc_order(text, split_seed, val_frac=VAL_FRAC):
    """Byte-identical replica of DocByteBatches.__init__'s ordering logic
    (r3_eval.py lines ~68-74) -- index arithmetic only, no corpus bytes
    materialized here, so the digest is cheap and the SAME order the real
    class will produce when train_with_eval() builds it for real."""
    docs = [d for d in text.split("\n\n") if d.strip()]
    order = list(range(len(docs)))
    random.Random(split_seed).shuffle(order)
    cut = int(len(order) * (1 - val_frac))
    train_ix, val_ix = sorted(order[:cut]), sorted(order[cut:])
    return train_ix, val_ix, len(docs)


def digest_order(train_ix, val_ix):
    payload = json.dumps({"train_ix": train_ix, "val_ix": val_ix}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_crn():
    text = RE._corpus_text(None, 64 * 1024 * 1024)
    orders = {}
    for ss in SPLIT_SEEDS:
        train_ix, val_ix, n_docs = doc_order(text, ss)
        dg = digest_order(train_ix, val_ix)
        orders[str(ss)] = dict(
            split_seed=ss, n_docs=n_docs,
            n_train_docs=len(train_ix), n_val_docs=len(val_ix),
            train_ix=train_ix, val_ix=val_ix, digest=dg,
        )
        print("[CRN] split_seed={} n_docs={} train={} val={} digest={}".format(
            ss, n_docs, len(train_ix), len(val_ix), dg[:16]), flush=True)
    out = dict(built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               val_frac=VAL_FRAC, split_seeds=SPLIT_SEEDS, orders=orders)
    with io.open(CRN_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    board("crn_built", path=CRN_PATH,
          digests={k: v["digest"] for k, v in orders.items()})
    return out


def prove_pairing(crn):
    """For each split_seed: build DocByteBatches TWICE (arm-A load, arm-B
    load) at seed=split_seed (TIED, per the design), confirm the digest
    matches the CRN file, confirm the two loads are byte-identical, and
    confirm the first K_PROOF_BATCHES train batches drawn with
    gen=manual_seed(seed+1) are byte-identical x AND y tensors."""
    text = RE._corpus_text(None, 64 * 1024 * 1024)
    all_ok = True
    proof_rows = []
    for ss in crn["split_seeds"]:
        seed = ss  # TIED: model seed == split seed, per the design
        dA = RE.DocByteBatches(text, 256, val_frac=VAL_FRAC, split_seed=ss)
        dB = RE.DocByteBatches(text, 256, val_frac=VAL_FRAC, split_seed=ss)
        train_eq = torch.equal(dA.train, dB.train)
        val_eq = torch.equal(dA.val, dB.val)
        my_digest = digest_order(*doc_order(text, ss)[:2])
        digest_match = (my_digest == crn["orders"][str(ss)]["digest"])

        genA = torch.Generator().manual_seed(seed + 1)
        genB = torch.Generator().manual_seed(seed + 1)
        batches_eq = True
        for _ in range(K_PROOF_BATCHES):
            xA, yA = dA.batch("train", BATCH, SEQ, genA, "cpu")
            xB, yB = dB.batch("train", BATCH, SEQ, genB, "cpu")
            if not (torch.equal(xA, xB) and torch.equal(yA, yB)):
                batches_eq = False
                break

        ok = train_eq and val_eq and digest_match and batches_eq
        all_ok = all_ok and ok
        row = dict(split_seed=ss, seed=seed, digest=my_digest[:16],
                   digest_matches_crn_file=digest_match,
                   train_tensor_byte_identical=train_eq,
                   val_tensor_byte_identical=val_eq,
                   first_k_batches_byte_identical=batches_eq,
                   k=K_PROOF_BATCHES, pairing_verified=ok)
        proof_rows.append(row)
        print("[PROOF] split_seed={} pairing_verified={} (train_eq={} val_eq={} "
              "digest_match={} batches_eq={})".format(
                  ss, ok, train_eq, val_eq, digest_match, batches_eq), flush=True)
        if not ok:
            board("crn_pairing_VOID", **row)
    board("crn_pairing_proof", all_split_seeds_verified=all_ok, rows=proof_rows)
    return all_ok, proof_rows


if __name__ == "__main__":
    crn = build_crn()
    ok, rows = prove_pairing(crn)
    print(json.dumps(dict(all_verified=ok), indent=2), flush=True)
    if not ok:
        print("CRN PAIRING VOID -- night is void per pre-registered kill.", flush=True)
        sys.exit(1)
