"""R-COST-N bar. Exit non-zero on failure.
python test_cost_n.py <impl>
Bar (declared before any fs number): ratio impl/SDPA <= 1.5 AND rel err vs float64 dense
solve <= 1e-5 (max abs err / max |ref|). Counter: ratio > 3."""
import sys, json, os
sys.path.insert(0, r"C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad")
import design4x5
assert design4x5.poll_until_free(timeout_s=900), "GPU not free"
import cost_n
impl = sys.argv[1]
row = cost_n.main([impl])[0]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_n_rows.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")
ok_cost = row["ratio"] <= 1.5
ok_exact = row["rel_err_vs_f64"] <= 1e-5
print(f"RESULT impl={impl} ratio={row['ratio']:.3f} rel_err={row['rel_err_vs_f64']:.3e} "
      f"cost_bar={'PASS' if ok_cost else 'FAIL'} exact_bar={'PASS' if ok_exact else 'FAIL'}")
assert ok_cost, f"R-COST-N FAIL: {impl} ratio {row['ratio']:.3f} > 1.5x one SDPA pass"
assert ok_exact, f"R-COST-N FAIL: {impl} rel err {row['rel_err_vs_f64']:.3e} > 1e-5"
