#!/usr/bin/env python3
"""Verify the real-weight smoke ledger and record the 512-step replay estimate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    smoke = run / "qualification/smoke"
    rows = [json.loads(line) for line in (smoke / "train_metrics.jsonl").read_text().splitlines() if line.strip()]
    if [row.get("global_step") for row in rows] != [1, 2]:
        raise ValueError("Smoke must complete exactly two optimizer updates")
    times = [float(row["step_time_seconds"]) for row in rows]
    if not all(math.isfinite(value) and value > 0 for value in times):
        raise ValueError("Smoke step times must be finite and positive")
    for step in (1, 2):
        checkpoint = smoke / f"checkpoints/global_step_{step}"
        if not all((checkpoint / folder / ".metadata").is_file() for folder in ("model", "optimizer")):
            raise ValueError(f"Smoke DCP metadata incomplete at step {step}")
        if len(list((checkpoint / "extra_state").glob("extra_state_rank_*.pt"))) != 8:
            raise ValueError(f"Smoke DCP rank state incomplete at step {step}")
    resumed = run / "qualification/restored_from_step_1"
    resumed_rows = [json.loads(line) for line in (resumed / "train_metrics.jsonl").read_text().splitlines() if line.strip()]
    if [row.get("global_step") for row in resumed_rows] != [1, 2]:
        raise ValueError("Resumed ledger did not complete step 2")
    if resumed_rows[0] != rows[0]:
        raise ValueError("Resume changed the preserved step-1 ledger row")
    if not all((resumed / "checkpoints/global_step_2" / folder / ".metadata").is_file()
               for folder in ("model", "optimizer")):
        raise ValueError("Restored step-2 DCP is incomplete")
    from math import isclose
    for key in ("loss", "step_loss"):
        if key in rows[1] and key in resumed_rows[1]:
            if not isclose(float(rows[1][key]), float(resumed_rows[1][key]), rel_tol=1e-3, abs_tol=1e-4):
                raise ValueError("Restored step-2 loss drift exceeds fixed tolerance")
    save_steps = int(yaml.safe_load((run / "veomni_cli.yaml").read_text())["train"]["save_steps"])
    if save_steps not in (512, 1024):
        raise ValueError("Unexpected v4.2 save interval")
    estimate = max(times) * save_steps
    report = {"status": "passed", "optimizer_updates": 2,
              "step_time_seconds": times, "save_steps": save_steps,
              "conservative_save_window_seconds": estimate,
              "qualification_scope": "real_weight_forward_backward_dcp_save_and_step1_restore"}
    output = smoke / "SMOKE_PASSED.json"
    if output.exists():
        raise FileExistsError("Smoke success marker already exists")
    temporary = output.with_name("." + output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
