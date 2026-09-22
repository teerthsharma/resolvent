import json, os, torch
SP = r"C:\Users\seal\AppData\Local\Temp\claude\C--Users-seal-Desktop-New-folder--32-\870edeb1-6409-4414-98e7-00d0537b75dd\scratchpad"
sd = torch.load(os.path.join(SP, "d45_ckpt_f_ss0_pair", "model.pt"), map_location="cpu", weights_only=False)["state_dict"]
r = dict(beta=[float(sd["model.layers.{}.self_attn.beta".format(l)]) for l in range(3)])
json.dump(r, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "beta.json"), "w")); print(r)
