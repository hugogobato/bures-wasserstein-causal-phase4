#!/usr/bin/env python3
"""Run frozen Phase 4 core jobs with resumable, atomic checkpoints."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

from bw_causal.phase4.runner import main


if __name__ == "__main__":
    raise SystemExit(main('core'))
