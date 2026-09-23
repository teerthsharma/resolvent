"""Builds one Kaggle kernel folder from a plan. Usage: python mk_kernels.py <plan.json>
plan: {"slug", "title", "version", "lanes": [[job names], ...], "est": {name: s}, "soft_s", "hard_s"}.
Jobs: Cameron's jobs_remaining.json (paths SP -> /tmp/sp; the control's --out -> runs_kaggle_control/) plus smoke
variants (smoke_* = the registered argv of the *_s1 job with --steps 50; smoke_fR_ga_g999_s1 also --gamma_sched 0.999)."""
import base64, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).parent
SP = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad"
REPO = "C:/Users/seal/Desktop/New folder (32)"
CAM = Path(SP) / "phase_k/K2/cameron"
JR = json.loads((CAM / "jobs_remaining.json").read_text())
EXP = {("phase_k/" + k) if not k.startswith("REPO/") else k: v for k, v in JR["dependency_sha256"].items()}
EXP["REPO/tests/foreman/phase_j/N1/cameron/cost_n.py"] = "9ab22197eaad779a51a1b8e1cff294babe7811f12bca2aba53ea2d481fbec93a"  # not in Cameron's list; imported by K0 resolvent.fs5c
sha = lambda b: hashlib.sha256(b).hexdigest()
files = {}
for rel, want in EXP.items():
    b = Path((REPO + "/" + rel[5:]) if rel.startswith("REPO/") else (SP + "/" + rel)).read_bytes()
    assert sha(b) == want, (rel, sha(b), want)
    files[rel] = {"sha": want, "b64": base64.b64encode(b).decode()}

mp = lambda a: [x.replace(SP, "/tmp/sp") for x in a]
JOBS = {}
for j in JR["jobs"]:
    argv = mp(j["argv"])
    if j["kind"] == "walk":
        JOBS[j["name"]] = {"name": j["name"], "argv": argv, "kind": "walk", "out": argv[3], "after": j["after"], "keep_model": True}
        continue
    i = argv.index("--out")
    if j["kind"] == "control":
        argv[i + 1] = argv[i + 1].replace("/runs/", "/runs_kaggle_control/")
    JOBS[j["name"]] = {"name": j["name"], "argv": argv, "kind": j["kind"], "out": argv[i + 1],
                       "keep_model": "--hook" in argv or j["kind"] == "control"}


def smoke(base, new, extra=()):
    j = json.loads(json.dumps(JOBS[base]))
    a = j["argv"]
    a[a.index("--steps") + 1] = "50"
    a[a.index("--out") + 1] = j["out"] = j["out"].replace(base, new)
    for k, v in extra:
        a[a.index(k) + 1] = v
    j["name"] = new
    JOBS[new] = j
    if base + "_walk" in JOBS:
        w = json.loads(json.dumps(JOBS[base + "_walk"]))
        w["argv"][3] = w["out"] = j["out"]
        w["name"], w["after"] = new + "_walk", new
        JOBS[new + "_walk"] = w


smoke("id_aL4_s1", "smoke_aL4_s1")
smoke("id_fR_sp_s1", "smoke_fR_sp_s1")
smoke("id_fR_ga_s1", "smoke_fR_ga_s1")
smoke("id_fR_ga_s1", "smoke_fR_ga_g999_s1", [("--gamma_sched", "0.999")])
JOBS.pop("smoke_fR_ga_g999_s1_walk", None)

plan = json.loads(Path(sys.argv[1]).read_text())
lanes = [[{**JOBS[n], "est_s": float(plan["est"].get(n, 0))} for n in lane] for lane in plan["lanes"]]
cfg = {"slug": plan["slug"], "version": plan["version"], "files": files, "fps": {k: v for k, v in JR["bed_fingerprints"].items() if k != "how"},
       "lanes": lanes, "soft_s": plan["soft_s"], "hard_s": plan["hard_s"]}
src = (HERE / "driver_template.py").read_text()
src = src.replace("PAYLOAD", repr(base64.b64encode(json.dumps(cfg).encode()).decode()), 1)
d = HERE / "kernels" / plan["slug"].split("/")[1]
d.mkdir(parents=True, exist_ok=True)
(d / "run.py").write_text(src, newline="\n")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": plan["slug"], "title": plan["title"], "code_file": "run.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=1))
print(d, sha((d / "run.py").read_bytes()), len(src))
for i, lane in enumerate(lanes):
    print(" lane", i, [(j["name"], j["est_s"]) for j in lane])
