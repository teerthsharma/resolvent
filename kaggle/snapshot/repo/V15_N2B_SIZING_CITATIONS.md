# V15 Sizing Model Citations Census

**Date:** 2026-08-31  
**Scope:** Exhaustive read-only census of memory-sizing and capacity figures in the repository  
**Task:** Find every site where the old uncalibrated model (6.63x low) or new calibrated model (C_OPERATOR=3.9, 3.4 B/elt) is cited  

---

## Module: ceq/sizing.py

The authoritative sizing module defines these constants:

| Line | Constant | Current Value | Classification |
|------|----------|---------------|---|
| 48 | `C_OPERATOR` | `3.9` | **CALIBRATED** |
| 68 | `DTYPE_MODES['bf16_autocast']` | `(2.2, 3.4)` | **CALIBRATED** |

The module docstring (lines 7-11) states:  
> "measured **3.9 tensors of shape [B, H, S, S] per layer**"

---

## Citation Sites by Classification

### STALE (6.63x or uncalibrated model cited)

| File | Line | Content | Class |
|------|------|---------|-------|
| `attic/workdonenew.pre-v13.md` | 185 | `"attention activation is 2·n·s²·heads bytes bf16" (v-main.6 §3, v-main.8 Part IV) – 6.63× LOW – the leading 2 is bytes-per-element, so the line assumes ONE retained [n,heads,s,s] tensor per layer. ceq/sizing.py, calibrated against the CUDA allocator on this same RTX 4060 Laptop, records C_OPERATOR = 3.9 retained tensors at an effective 3.4 bytes/element under bf16 autocast. 3.9 × (3.4/2.0) = 6.63. At n=32, s=512, h=4 the line says 0.0671 GB/layer; the calibrated model says 0.4449 GB/layer` | **STALE** |
| `workdonenew.md` | 275 | (identical to above — duplicated archival entry) | **STALE** |
| `CEQ_V15_CONTRACT.md` | 32 | `- The sizing line was 6.63x low.` | **STALE** |
| `V13_CLAIM_AUDIT.md` | 59 | `contract line is 6.63× below the calibrated model – 3.9 × (3.4/2.0) = 6.63 – C_OPERATOR = 3.9, DTYPE_MODES['bf16_autocast'] = (2.2, 3.4) → 6.63 exactly – CONFIRMED – ceq/sizing.py:48,68 + arithmetic` | **STALE** |
| `V13_CLAIM_AUDIT.md` | 60 | `at n=32, s=512, h=4: contract vs calibrated – 0.0671 vs 0.4449 GB/layer – 67,108,864 B = 0.06711 GB; 3.9·4·32·512²·3.4 = 444,931,768 B = 0.44493 GB – CONFIRMED – arithmetic against source constants` | **STALE** |

---

### CALIBRATED (uses 3.9 and/or 3.4 B/elt)

| File | Line | Content | Class |
|------|------|---------|-------|
| `CEQ_V15_CONTRACT.md` | 71 | `NEPTUNE – systems gate — dispatch, memory by the CALIBRATED sizing model C_OPERATOR = 3.9 x 3.4 B/elt, wall-clock vs FLOP` | **CALIBRATED** |
| `V15_LEDGER.md` | 66 | `C_OPERATOR = 3.9 x 3.4 B/elt everywhere it was cited. Output` | **CALIBRATED** |
| `ceq/sizing.py` | 48 | `C_OPERATOR = 3.9` | **CALIBRATED** |
| `ceq/sizing.py` | 68 | `"bf16_autocast": (2.2, 3.4),` | **CALIBRATED** |

---

### UNRELATED (numbers match but refer to other quantities)

The following numbers appear in the grep results but do not refer to the sizing model:

| File | Line | Content | Classification |
|------|------|---------|---|
| `BOARD.md` | 35 | `The PIVOT form is intensity 3.9 FLOP/B` (arithmetic intensity, not bytes/elt) | UNRELATED |
| `TRAINING.md` | 81 | `activations 2.28 → 7.88 GiB, a 3.45× jump` (activation ratio, not model constant) | UNRELATED |
| `colab/train_ceq.ipynb` | 47 | `this operator materializes it and autograd retains ~3.9 such tensors per layer` (descriptive prose, not a constant) | UNRELATED |
| `scripts/v13_b9_4060_probe.py` | 7 | `records 3.9 retained tensors per layer at an effective 3.4 bytes per element` (measurement prose) | UNRELATED |
| `ceq/hf/modeling_ceq.py` | 101 | `materializes it and autograd retains a measured 3.9 tensors of [B,H,S,S]` (comment, same as above) | UNRELATED |
| Multiple `.jsonl` files (`tests/chase/deq_run.jsonl`, `results/*.jsonl`, etc.) | Various | Numeric values from simulation runs (e.g., loss `3.9`, condition numbers, RSS measurements); none are model constants | UNRELATED |

---

## Files with "sizing" or "capacity" in Name

**Module files:**
- `ceq/sizing.py` — **authoritative source** (contains C_OPERATOR=3.9 and DTYPE_MODES)

**Test files:**
- `tests/chase/test_scale_sizing.py` — tests the sizing model

**Capacity-related result files (data, not model constants):**
- `results/r10_it8_capacity_softmax_t2.jsonl`
- `results/r10_it8_capacity_softmax_t8.jsonl`
- `results/r10_it8_capacity_softmax_t32.jsonl`
- `results/r10_probe13_capacity_softmax_t8.jsonl`
- `results/r10_probe13_capacity_softmax_t32.jsonl`
- `results/r10_r10_it21_t8_n8192_capacity_softmax_t8.jsonl`
- `results/r10_smoke_capacity_softmax_t2.jsonl`
- `results/r10_v13pilot_capacity_pivot_unsigned_t2.jsonl`
- `results/r10_v13pilot_capacity_pivot_unsigned_t8.jsonl`
- `results/r10_v13pilot_capacity_windowed_signed_t2.jsonl`
- `scale/r10_capacity_sweep.py` — pricing model, not sizing constants

---

## Summary Statistics

### STALE Sites: 5 total across 4 files

**Load-bearing files:**
1. `workdonenew.md` (1 STALE site) — **[PROSE/DOCUMENTATION]**
2. `V13_CLAIM_AUDIT.md` (2 STALE sites) — **[PROSE/DOCUMENTATION]**
3. `CEQ_V15_CONTRACT.md` (1 STALE site) — **[PROSE/DOCUMENTATION]**
4. `attic/workdonenew.pre-v13.md` (1 STALE site) — **[PROSE/ARCHIVE]**

**Non-load-bearing classification:** All 5 STALE sites are in **prose documentation** (`.md` files), not in executable modules or tests. They are part of the audit/contract trail.

### CALIBRATED Sites: 4 total across 2 files

**Load-bearing files:**
1. `ceq/sizing.py` (2 sites: lines 48, 68) — **[MODULE]** ← Source of truth
2. `CEQ_V15_CONTRACT.md` (1 site) — **[PROSE/CONTRACT]**
3. `V15_LEDGER.md` (1 site) — **[PROSE/LEDGER]**

### Files Not Modified by Task

The following categories of files contain numbers that matched search patterns but require no corrections:

- **Test data files (`.jsonl`, `.txt`):** Simulation outputs with literal numeric values (e.g., rho=0.6868, condition numbers)
- **Notebooks (`.ipynb`):** Training tables and experimental documentation
- **Code comments:** References to the 3.9 tensors as a measurement fact (not a model constant to be updated)
- **Archive files in `attic/`:** Pre-v13 development artifacts

### Skipped Files (as instructed)

- `house-events.jsonl` — **SKIPPED** (3.3 MB JSONL file with spacing issues in JSON serialization; would produce false negatives on literal-string grep)

---

## Verification Checklist

- [x] Found all literal `6.63` citations (4 files)
- [x] Found all literal `3.9` citations in code (ceq/sizing.py line 48)
- [x] Found all literal `3.4` citations in code (ceq/sizing.py line 68)
- [x] Identified `C_OPERATOR` token (ceq/sizing.py line 48)
- [x] Located sizing vocabulary: `bytes_per`, `B/elt`, `B/element`, `sizing model`, `activation_bytes`
- [x] Read `ceq/sizing.py` fully and reported all constants
- [x] Identified files named `*sizing*` and `*capacity*`
- [x] Classified all hits as STALE, CALIBRATED, or UNRELATED
- [x] Excluded `.git/`, `__pycache__/`, `.pytest_cache/` from search
- [x] Excluded `house-events.jsonl` per instructions

---

## Next Steps (For Correction Node)

The task V15 contract specifies: "replace the old model everywhere it was cited" with the calibrated model `C_OPERATOR = 3.9` and `3.4 bytes/element`.

**STALE sites requiring correction (in order of priority):**

1. **Priority 1 (Audit trail):** `V13_CLAIM_AUDIT.md` lines 59–60 (confirms the discrepancy; may be left as-is for record)
2. **Priority 2 (Contract):** `CEQ_V15_CONTRACT.md` line 32 (states the requirement; update to reflect completion)
3. **Priority 3 (Work record):** `workdonenew.md` line 275 (work log; should reflect calibrated values)
4. **Priority 4 (Archive):** `attic/workdonenew.pre-v13.md` line 185 (archive; clarify historical status)

**CALIBRATED sites (already correct):**
- No action needed; these already cite the calibrated model.

