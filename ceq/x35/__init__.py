"""ceq.x35 -- residual inference of hidden causes (CEQ_V15_1_DELTA.md X35).

`residual.py` is X35a, THE INSTRUMENT: `r = z_obs - z_model(visible)`, onset by
a memoryless (Shewhart) comparator on residual energy, threshold calibrated on
no-plant runs. Its two must-fires live in tests/x35/test_residual_onset.py and
its reading is filed in V15_X35A_RESIDUAL.md.
"""
__all__ = ["residual"]
