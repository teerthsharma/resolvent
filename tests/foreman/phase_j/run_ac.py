import sys, time, json
sys.path.insert(0, r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad")
sys.path.insert(0, r"C:\Users\seal\Desktop\New folder (32)")
import design4x5 as D

CEQAttention = D.CEQAttention
softmax_forward = D.softmax_forward
ORIG_FORWARD = D._ORIG_ATTN_FORWARD
RE = D.RE

if not D.poll_until_free(timeout_s=1800):
    print(json.dumps({"error": "GPU not free"}))
    sys.exit(1)

crn = D.load_crn()
dg = D.crn_digest(crn, 0)

HIDDEN = 512
out_dir = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad\phase_j\d45_ckpt_ac_ss0"

common = dict(out_dir=out_dir, steps=3538, batch=D.BATCH, seq=D.SEQ,
              hidden_size=HIDDEN, n_layers=D.LAYERS, n_heads=D.HEADS,
              device=D.DEVICE, vocab_size=D.VOCAB, lr=D.LR, seed=0,
              split_seed=0, eval_every=3538, eval_batches=D.EVAL_BATCHES,
              log_every=200, save_model=True)

CEQAttention.forward = softmax_forward
t0 = time.time()
try:
    rec = RE.train_with_eval(operator="sgate", **common)
finally:
    CEQAttention.forward = ORIG_FORWARD
wall = time.time() - t0

rec["crn_digest"] = dg
rec["arm"] = "a_c"
rec["hidden_used"] = HIDDEN
rec["wall_seconds"] = wall
tokens_seen = 3538 * D.BATCH * D.SEQ
rec["tokens_seen"] = tokens_seen

with open("ac_result.json", "w") as f:
    json.dump(rec, f, indent=2)

print(json.dumps(dict(final_eval_loss=rec["final_eval_loss"], seconds=rec["seconds"],
                       wall_seconds=wall, tokens_seen=tokens_seen,
                       n_params=rec["n_params"], crn_digest=dg), indent=2))
