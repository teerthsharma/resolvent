"""Inspector N2: recompute R-POS recoveries from the grid jsonl, independent of test_rpos.py."""
import json
G = r"C:/Users/seal/Desktop/New folder (32)/tests/foreman/design4x5/design4x5_results.jsonl"
R = r"../foreman/rpos_results.jsonl"
grid = {}
arms = set()
for l in open(G, encoding="utf-8"):
    if not l.strip(): continue
    r = json.loads(l); arms.add((r.get("stage"), r.get("arm")))
    if r.get("stage") == "cell": grid.setdefault((r["arm"], r["split_seed"]), []).append(r)
print("grid stages/arms:", sorted(map(str, arms)))
dups = {k: len(v) for k, v in grid.items() if len(v) > 1}
print("duplicate grid cells:", dups)
rows = [json.loads(l) for l in open(R, encoding="utf-8") if l.strip()]
for r in rows:
    s = r["split_seed"]; a = grid[("a", s)][-1]; f = grid[("f", s)][-1]
    la, lf = a["final_eval_loss"], f["final_eval_loss"]
    rec = (la - r["final_eval_loss"]) / (la - lf)
    dg = r["crn_digest_recomputed"] == r["crn_digest_file"] == r["crn_digest_grid"] == a["crn_digest"] == f["crn_digest"]
    print("%-8s ss%d loss %.7f  la %.7f lf %.7f Cwin %.6f  rec %.4f  f-arm %+.4f  digest_all_equal %s  params %d vs a %d  sec %.1f" % (
        r["arm"], s, r["final_eval_loss"], la, lf, la - lf, rec, lf - r["final_eval_loss"], dg, r["n_params"], a["n_params"], r["run_seconds"]))
for arm in ("fox", "FoX", "a_fox", "a+fox", "g"):
    pass
print("all grid arms seed0-2 with recovery vs C_win:")
for (arm, s), v in sorted(grid.items()):
    if s in (0, 1, 2) and arm not in ("a", "f"):
        la, lf = grid[("a", s)][-1]["final_eval_loss"], grid[("f", s)][-1]["final_eval_loss"]
        x = v[-1]; print("  %-10s ss%d loss %.6f rec %.4f params %s sec %s" % (arm, s, x["final_eval_loss"], (la - x["final_eval_loss"]) / (la - lf), x.get("n_params"), round(x.get("run_seconds", 0), 1)))
for s in (0,1,2):
    print("f ss%d sec %.1f" % (s, grid[("f", s)][-1]["run_seconds"]))
