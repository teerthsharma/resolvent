import datetime, json, os, sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    """Every test parametrizes over device; cuda skips when absent.

    REQUIREMENTS.md standing rule: a test that only runs on CUDA cannot be
    reproduced by a reader without an A100, and a CPU reference is the only
    honest parity oracle for a GPU kernel anyway.
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available")
    return torch.device(request.param)


# --------------------------------------------------------------- board logging
# Round 1's log was struck for truncated JSON and for 10 REDs against zero
# GREENs. The fix is not discipline, it is automation: this hook fires on every
# test in this directory, red and green alike, and writes one complete JSON
# object per line with the real pytest node id. Nothing is transcribed by hand,
# so nothing can be selectively omitted.
BOARD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "house-events.jsonl")


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return
    state = {"passed": "GREEN", "failed": "RED", "skipped": "SKIP"}[report.outcome]
    detail = ""
    if report.longrepr is not None:
        detail = " ".join(str(report.longrepr).split())[-600:]
    line = json.dumps({"t": "test", "agent": "cameron",
                       "node": report.nodeid, "state": state,
                       "detail": detail, "dur": round(report.duration, 3),
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    with open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
