"""RED if a5_bed.gen_split is not a pure function of (split, seed) across interpreter runs."""
import subprocess, sys, os
code = "import sys; sys.path.insert(0, r'%s'); import a5_bed as A; t,_=A.gen_split('train',0); print(t[0,:16].tolist())" % os.path.dirname(os.path.abspath(__file__))
outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": s}).stdout.strip() for s in ("1", "2")]
print(outs)
assert outs[0] == outs[1], f"RED: gen_split('train',0) differs by PYTHONHASHSEED: {outs[0]} vs {outs[1]}"
print("GREEN")
