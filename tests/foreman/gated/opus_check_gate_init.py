"""Opus checker on the gate-init repair row.

(A) read the gate MYSELF at the four states, one script, one eval batch,
one metric implementation: reference-init, repaired-init, reference-trained,
repaired-trained.  The reference run-length numbers quoted in the task were
measured by a DIFFERENT script at eval_seed=999; everything here is at
eval_seed=12345, the seed both training scripts used.

Two reach metrics, because they are not the same thing:
  stretch = length of the maximal contiguous nonzero-m stretch containing
            the position (the Foreman's metric), 0 at a zero position;
  back    = prod_{k=j+1}^{i} m_k is the actual path weight, so row i's live
            reach is how many steps BACK from i before a zero factor --
            the backward run ending at i.  <= stretch, always.

L-REPRO: seed 0 models, float32, h512/L4/H8/d_head64/seq512/batch8,
eval gen seed 12345, device cuda if available.
"""
import io, json, math, os, sys, time
REPO = r"C:\Users\seal\Desktop\New folder (32)"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
BOARD = os.path.join(REPO, "house-events.jsonl")
sys.path.insert(0, REPO)
import torch
from ceq.hf import train as T
from ceq.hf.modeling_ceq import CEQForCausalLM
from ceq.arm_smprime import blend, GATE_INIT_OFF

HIDDEN, LAYERS, HEADS, SEQ, BATCH = 512, 4, 8, 512, 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
REF_CKPT = os.path.join(SCRATCH, "gated_ckpt")
REP_CKPT = os.path.join(SCRATCH, "gate_repair_ckpt")


def board(event, **kw):
    row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), agent="Opus-checker",
               event=event, **kw)
    with io.open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print("[board]", json.dumps(row)[:400])


@torch.no_grad()
def gate_values(model, x_ids):
    model.eval()
    cap = []

    def hook(mod, inputs):
        x = inputs[0]
        m, th = blend(mod.m_head(x).squeeze(-1),
                      mod.theta_head(x).squeeze(-1), mod.g)
        cap.append(m.detach().float().cpu())

    hs = [l.self_attn.register_forward_pre_hook(hook) for l in model.model.layers]
    model(input_ids=x_ids)
    for h in hs:
        h.remove()
    return cap


def stats(m_layers):
    a = torch.cat([m.reshape(-1) for m in m_layers])
    nz = a[a > 0]
    return dict(n=int(a.numel()), n_zero=int((a == 0).sum()),
                frac_zero=float((a == 0).float().mean()),
                geomean_nonzero=float(torch.exp(torch.log(nz).mean())) if nz.numel() else 0.0,
                frac_one=float((a == 1.0).float().mean()),
                mean=float(a.mean()))


def reaches(m_layers):
    """stretch + backward run, per position, pooled over layers and batch."""
    stretch, back = [], []
    for m in m_layers:                       # [B,S]
        nz = (m > 0)
        B, S = nz.shape
        b = torch.zeros(B, S, dtype=torch.int32)
        run = torch.zeros(B, dtype=torch.int32)
        for t in range(S):
            run = torch.where(nz[:, t], run + 1, torch.zeros_like(run))
            b[:, t] = run
        f = torch.zeros(B, S, dtype=torch.int32)
        run = torch.zeros(B, dtype=torch.int32)
        for t in range(S - 1, -1, -1):
            run = torch.where(nz[:, t], run + 1, torch.zeros_like(run))
            f[:, t] = run
        s = torch.where(nz, b + f - 1, torch.zeros_like(b))
        stretch.append(s.reshape(-1))
        back.append(b.reshape(-1))

    def q(parts):
        t = torch.cat(parts).to(torch.float64)
        med, p95 = torch.quantile(
            t, torch.tensor([0.5, 0.95], dtype=torch.float64)).tolist()
        return dict(n=int(t.numel()), mean=float(t.mean()), median=med,
                    p95=p95, max=float(t.max()))

    return dict(stretch=q(stretch), back=q(back))


_ORIG = T.build


def repaired_build(**kw):
    model = _ORIG(**kw)
    with torch.no_grad():
        for l in model.model.layers:
            l.self_attn.m_head.bias.fill_(1.0 - GATE_INIT_OFF)
            l.self_attn.theta_head.bias.fill_(GATE_INIT_OFF)
    return model


def main():
    t0 = time.time()
    board("start", task="opus_check_gate_init", device=DEVICE, eval_gen_seed=12345)
    data = T.ByteBatches(T._corpus_text(None, 64 * 1024 * 1024), vocab_size=256)
    x_eval, _ = data.batch("val", BATCH, SEQ,
                           torch.Generator().manual_seed(12345), DEVICE)

    out = {}

    def measure(name, model):
        mv = gate_values(model, x_eval)
        out[name] = dict(stats=stats(mv), reach=reaches(mv),
                         per_layer_frac_zero=[float((m == 0).float().mean()) for m in mv])
        print(name, json.dumps(out[name])[:700], flush=True)
        del model, mv
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    measure("ref_init", _ORIG(hidden_size=HIDDEN, n_layers=LAYERS, n_heads=HEADS,
                              seq=SEQ, vocab_size=256, operator="smprime").to(DEVICE))
    torch.manual_seed(SEED)
    measure("rep_init", repaired_build(hidden_size=HIDDEN, n_layers=LAYERS,
                                       n_heads=HEADS, seq=SEQ, vocab_size=256,
                                       operator="smprime").to(DEVICE))
    measure("ref_trained", CEQForCausalLM.from_pretrained(REF_CKPT).to(DEVICE))
    measure("rep_trained", CEQForCausalLM.from_pretrained(REP_CKPT).to(DEVICE))

    from safetensors.torch import load_file
    cfg_ref = json.load(open(os.path.join(REF_CKPT, "config.json")))
    cfg_rep = json.load(open(os.path.join(REP_CKPT, "config.json")))
    sd_ref = load_file(os.path.join(REF_CKPT, "model.safetensors"))
    sd_rep = load_file(os.path.join(REP_CKPT, "model.safetensors"))
    out["paired"] = dict(
        configs_equal=(cfg_ref == cfg_rep),
        config_diff={k: (cfg_ref.get(k), cfg_rep.get(k))
                     for k in set(cfg_ref) | set(cfg_rep)
                     if cfg_ref.get(k) != cfg_rep.get(k)},
        same_key_set=(set(sd_ref) == set(sd_rep)),
        n_keys=len(sd_ref),
        trained_bias=[dict(
            layer=i,
            ref_m=float(sd_ref["model.layers.{}.self_attn.m_head.bias".format(i)]),
            rep_m=float(sd_rep["model.layers.{}.self_attn.m_head.bias".format(i)]),
            ref_th=float(sd_ref["model.layers.{}.self_attn.theta_head.bias".format(i)]),
            rep_th=float(sd_rep["model.layers.{}.self_attn.theta_head.bias".format(i)]))
            for i in range(LAYERS)],
        switches=[dict(
            layer=i,
            ref=[float(sd_ref["model.layers.{}.self_attn.{}".format(i, s)])
                 for s in ("beta", "qk", "g")],
            rep=[float(sd_rep["model.layers.{}.self_attn.{}".format(i, s)])
                 for s in ("beta", "qk", "g")])
            for i in range(LAYERS)],
        m_head_weight_l2_delta=[
            float((sd_ref["model.layers.{}.self_attn.m_head.weight".format(i)]
                   - sd_rep["model.layers.{}.self_attn.m_head.weight".format(i)]).norm())
            for i in range(LAYERS)],
    )

    rr_ref = json.load(open(os.path.join(REF_CKPT, "run_record.json")))
    rr_rep = json.load(open(os.path.join(REP_CKPT, "run_record.json")))
    keys = [k for k in rr_ref if not isinstance(rr_ref[k], (list, dict))]
    out["run_record_scalar_diff"] = {k: (rr_ref[k], rr_rep.get(k))
                                     for k in keys if rr_ref[k] != rr_rep.get(k)}
    out["run_record_scalar_same"] = {k: rr_ref[k] for k in keys
                                     if rr_ref[k] == rr_rep.get(k)}
    lr_ref, lr_rep = rr_ref.get("losses"), rr_rep.get("losses")
    out["loss"] = dict(ref_first=lr_ref[0], ref_last=lr_ref[-1],
                       rep_first=lr_rep[0], rep_last=lr_rep[-1],
                       ref_mean_last50=sum(lr_ref[-50:]) / 50.0,
                       rep_mean_last50=sum(lr_rep[-50:]) / 50.0,
                       n_ref=len(lr_ref), n_rep=len(lr_rep))
    out["wall_s"] = time.time() - t0
    p = os.path.join(SCRATCH, "opus_check_gate_init.json")
    json.dump(out, open(p, "w"), indent=1, default=str)
    board("done", path=p,
          ref_trained_frac_zero=out["ref_trained"]["stats"]["frac_zero"],
          rep_trained_frac_zero=out["rep_trained"]["stats"]["frac_zero"],
          ref_trained_back_max=out["ref_trained"]["reach"]["back"]["max"],
          rep_trained_back_max=out["rep_trained"]["reach"]["back"]["max"],
          ref_trained_stretch_max=out["ref_trained"]["reach"]["stretch"]["max"],
          rep_trained_stretch_max=out["rep_trained"]["reach"]["stretch"]["max"],
          wall_s=out["wall_s"])
    print("WROTE", p)


if __name__ == "__main__":
    main()
