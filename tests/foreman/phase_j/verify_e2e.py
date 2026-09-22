import json
import numpy as np

# Load r_do_e2e.json
with open("r_do_e2e.json") as f:
    data = json.load(f)

# Extract seeds
SEEDS = data["seeds"]
FLOOR_PLUGIN = data["floor_plugin"]
BAR_THRESHOLD = data["bar_threshold_0p8x_floor"]
results = data["results"]

# Recompute means from per-seed do_error values
arms = ["a", "a2F", "f"]
computed_means = {}
for arm in arms:
    errors = [results[arm][str(s)]["do_error"] for s in SEEDS]
    computed_means[arm] = float(np.mean(errors))

print("=== MEANS VERIFICATION ===")
print(f"Recorded means: {data['means']}")
print(f"Computed means: {computed_means}")
print(f"Match: {np.allclose(list(data['means'].values()), list(computed_means.values()))}")

# Recompute verdict logic
f_errs = [results["f"][str(s)]["do_error"] for s in SEEDS]
f_pass_5of5 = all(e <= BAR_THRESHOLD for e in f_errs)
fox_above_floor = computed_means["a2F"] > FLOOR_PLUGIN
fox_at_or_below_thr = computed_means["a2F"] <= BAR_THRESHOLD
any_clears_floor = any(computed_means[arm] < FLOOR_PLUGIN for arm in arms)

print("\n=== VERDICT LOGIC ===")
print(f"f_pass_5of5 (all f <= {BAR_THRESHOLD:.6f}): {f_pass_5of5}")
print(f"  f_errs: {f_errs}")
print(f"fox_above_floor ({computed_means['a2F']:.6f} > {FLOOR_PLUGIN}): {fox_above_floor}")
print(f"fox_at_or_below_thr ({computed_means['a2F']:.6f} <= {BAR_THRESHOLD:.6f}): {fox_at_or_below_thr}")
print(f"any_clears_floor (any < {FLOOR_PLUGIN}): {any_clears_floor}")

# Apply verdict logic
if f_pass_5of5 and fox_above_floor:
    computed_verdict = "PASS: condition 2 reopened"
elif f_pass_5of5 and fox_at_or_below_thr:
    computed_verdict = "FAIL: closed for good, consequence struck"
elif not any_clears_floor:
    computed_verdict = "DO DEAD AT EVERY READ: Phase J closes as a kernel phase"
else:
    order = sorted(computed_means, key=lambda arm: computed_means[arm])
    computed_verdict = ("NEITHER: bar not cleanly resolved, ordering {} ({:.6f} < {:.6f} < {:.6f})".format(
        "<".join(order), computed_means[order[0]], computed_means[order[1]], computed_means[order[2]]))

print(f"\nRecorded verdict: {data['verdict']}")
print(f"Computed verdict: {computed_verdict}")
print(f"Match: {data['verdict'] == computed_verdict}")

# Verify clock_ratio
a_wall_first = results["a"]["0"]["wall_seconds"]
for arm in arms:
    arm_mean = float(np.mean([results[arm][str(s)]["wall_seconds"] for s in SEEDS]))
    computed_ratio = arm_mean / a_wall_first
    recorded_ratio = data["clock_ratio"][arm]
    match = np.isclose(computed_ratio, recorded_ratio)
    print(f"\nClock ratio {arm}: computed={computed_ratio:.16f}, recorded={recorded_ratio:.16f}, match={match}")

# Verify seconds_per_cell
all_wall = [results[arm][str(s)]["wall_seconds"] for arm in arms for s in SEEDS]
computed_seconds_per_cell = float(np.mean(all_wall))
print(f"\n=== SECONDS_PER_CELL ===")
print(f"Computed: {computed_seconds_per_cell:.16f}")
print(f"Recorded: {data['started']} to {data['finished']}")

print("\nCLEAN: R-DO-E2E" if (
    np.allclose(list(data['means'].values()), list(computed_means.values())) and
    data['verdict'] == computed_verdict
) else "\nSTRUCK: R-DO-E2E")
