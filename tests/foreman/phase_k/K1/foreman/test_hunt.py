# Foreman K1 bars on hand-set constructions over bed_k. Written 2026-09-23 02:56 IST, BEFORE hunt.py exists or runs.
# Never edited after its RED; a changed bar gets a new name. Reads hunt.json (written by hunt.py). Exit 1 if any row fails.
#
# Beds: bed_k (K0/cameron/bed_k.py, imported read-only). Test beds are Cameron's K1.F seeds default_rng([12, n, k]),
# k = 0..7, so every oracle number is comparable to his line. Credit rule (Cameron's): an unresolved token guesses the
# most recent root at or before its final pointer. "R0 configs" = k=3 W<=10 and k=4 W<=6, anchored and centered.
#
# foreman.k1.hunt_machinery : my window-in-any-coordinate hybrid with coordinate = position reproduces K0's
#     shortcut.hybrid pointer-for-pointer (0 mismatches over the checked beds/configs), and reproduces Cameron's
#     K1.F oracle cell (L=7, n=4096, centered k3 W10, credited beyond 2^L) = 0.4085875984251969 to 1e-12.
# foreman.k1.ids_eq_pos (retires/binds the struck K0 claim "the shortcut relies on ids = positions"):
#     (i) on the pinned wald.py bed (ids ARE positions) the id-addressed hybrid (window placed in id space, the K0
#         mechanism verbatim: "the token whose id is v1 - j") equals the positional one: 0 pointer mismatches at
#         L=6 W=26 anchored k3, all-token accuracy within 5e-5 of K0's 0.3982;
#     (ii) on bed_k (n=4096, 8 beds) the same id-addressed hybrid, best over R0 configs, scores credited beyond 2^L
#         within 0.005 of credited doubling at L=4 and L=7, and its window fires on <= 1e-3 of (token, layer) pairs.
# foreman.k1.content_closure (the 2^L assumption for content addressing, unlimited heads and width): each token
#     holds a SET of ancestor tokens; per layer it fetches every held token's set (id lookup) AND every earlier
#     token whose set overlaps its own (shared-id predicate). Resolved iff its root is in the set. On 8 bed_k-structure
#     beds (n=1024, 16 lanes, uncapped), L=1..5: the resolved set equals {depth <= 2^L} exactly (0 mismatches),
#     with and without the overlap predicate.
# foreman.k1.no_oracle_ceiling : constructions that need no absolute-position oracle -- (a) a fresh window of the
#     W positions before the token ITSELF (ALiBi-local) and (b) a one-layer-stale bundle of the W positions before
#     v1, stored by v1 -- best over R0 configs, credited beyond 2^L <= 0.25 at L in {4, 7} and n in {4096, 8192, 16384}.
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "hunt.json")).read_text())
fails = []


def check(name, ok, got=""):
    print(("PASS " if ok else "FAIL ") + name + " : " + str(got))
    if not ok:
        fails.append(name)


m = R["machinery"]
check("foreman.k1.hunt_machinery space==hybrid mismatches", m["space_vs_hybrid_mismatches"] == 0, m["space_vs_hybrid_mismatches"])
check("foreman.k1.hunt_machinery Cameron cell L7 n4096 centered k3 W10",
      abs(m["cameron_cell"] - 0.4085875984251969) < 1e-12, m["cameron_cell"])

w = R["wald"]
check("foreman.k1.ids_eq_pos (i) wald id-addressed == positional pointers", w["mismatches"] == 0, w["mismatches"])
check("foreman.k1.ids_eq_pos (i) wald id-addressed all ~ 0.3982", abs(w["id_addr_all"] - 0.3982) < 5e-5,
      f"{w['id_addr_all']:.6f} (positional {w['pos_all']:.6f})")
for L in ("4", "7"):
    c = R["bedk"]["4096"][L]
    check(f"foreman.k1.ids_eq_pos (ii) n=4096 L={L} id-addressed best - credited doubling <= 0.005",
          c["id_best"] - c["dbl_cred"] <= 0.005, f"id {c['id_best']:.4f} [{c['id_best_cfg']}] dbl {c['dbl_cred']:.4f} oracle {c['oracle_best']:.4f}")
    check(f"foreman.k1.ids_eq_pos (ii) n=4096 L={L} id-window hit rate <= 1e-3", c["id_hit_rate"] <= 1e-3, c["id_hit_rate"])

for L, v in R["closure"].items():
    check(f"foreman.k1.content_closure L={L} plain", v["mismatch_plain"] == 0, v)
    check(f"foreman.k1.content_closure L={L} with overlap predicate", v["mismatch_overlap"] == 0, v)

for n in ("4096", "8192", "16384"):
    for L in ("4", "7"):
        c = R["bedk"][n][L]
        best = max(c["local_best"], c["stale_best"])
        check(f"foreman.k1.no_oracle_ceiling n={n} L={L} <= 0.25", best <= 0.25,
              f"local {c['local_best']:.4f} [{c['local_best_cfg']}] stale {c['stale_best']:.4f} [{c['stale_best_cfg']}] "
              f"dbl {c['dbl_cred']:.4f} oracle {c['oracle_best']:.4f}")

print(("RED " if fails else "GREEN ") + f"{len(fails)} failing rows")
sys.exit(1 if fails else 0)
