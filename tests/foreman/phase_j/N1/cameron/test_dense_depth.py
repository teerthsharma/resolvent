"""D5 on a dense causal softmax W (random q, k, n = 256 and 1024, fp64, gamma = 0.99).
Assertion as the room's brief frames D5: the D-hop exactness needs n-1 jump-chain hops,
i.e. the rel err against the triangular solve stays > 1e-12 until hop n-1.
Structural half: every subdiagonal entry of J > 0 (so J^(n-1) != 0 and D = n-1)."""
import subprocess, sys, re
out = subprocess.run([sys.executable, "dense_depth.py", "1.0"], capture_output=True, text=True).stdout
print(out)
for line in out.strip().splitlines():
    n = int(re.search(r"n=(\d+)", line).group(1))
    logsub = float(re.search(r"subdiag J = (-?[\d.]+)", line).group(1))
    h12 = int(re.search(r"1e-12: (\d+)", line).group(1))
    assert logsub > -1e300, "structural: a zero subdiagonal entry"
    assert h12 >= n - 1, f"D5-as-framed FAIL at n={n}: rel err <= 1e-12 after {h12} hops, not n-1 = {n-1}"
