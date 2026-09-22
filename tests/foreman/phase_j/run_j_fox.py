import os
import sys

PHASE_J = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PHASE_J)
import run_j as RJ  # noqa: E402

if __name__ == "__main__":
    os.makedirs(PHASE_J, exist_ok=True)
    import arms_j as J
    RJ.run_row("R-FoX", "a2F", J.train_a2F,
               dict(desc="C_forget >= 0.5*C_win(ss0)", threshold=RJ.BAR), do_poll=True)
