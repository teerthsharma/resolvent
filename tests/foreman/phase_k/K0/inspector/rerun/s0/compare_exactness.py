#!/usr/bin/env python3
import json
import sys

orig_path = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/scale/S0_cameron/exactness.json"
rerun_path = "C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_k/K0/inspector/rerun/s0/exactness.json"

try:
    with open(orig_path) as f:
        orig_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: {orig_path} not found")
    sys.exit(1)

try:
    with open(rerun_path) as f:
        rerun_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: {rerun_path} not found")
    sys.exit(1)

if not isinstance(orig_data, list):
    orig_data = [orig_data]
if not isinstance(rerun_data, list):
    rerun_data = [rerun_data]

# Index by cfg_beta
orig_by_beta = {row.get("cfg_beta"): row for row in orig_data if isinstance(row, dict)}
rerun_by_beta = {row.get("cfg_beta"): row for row in rerun_data if isinstance(row, dict)}

all_betas = set(orig_by_beta.keys()) | set(rerun_by_beta.keys())
all_betas = sorted(all_betas, key=lambda x: float(x) if x else 0)

total_keys = 0
exactly_equal_count = 0

for beta in all_betas:
    orig_row = orig_by_beta.get(beta, {})
    rerun_row = rerun_by_beta.get(beta, {})

    all_keys = set(orig_row.keys()) | set(rerun_row.keys())

    for key in sorted(all_keys):
        if key == "cfg_beta":
            continue
        total_keys += 1
        orig_val = orig_row.get(key)
        rerun_val = rerun_row.get(key)
        is_equal = orig_val == rerun_val
        if is_equal:
            exactly_equal_count += 1
        print(f"beta={beta} key={key}: orig={orig_val} rerun={rerun_val} equal={is_equal}")

print(f"exactly equal leaves: {exactly_equal_count} of {total_keys}")
