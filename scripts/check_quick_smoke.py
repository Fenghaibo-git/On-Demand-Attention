#!/usr/bin/env python3
"""Check the one-GPU real-weight diagnostic without claiming formal qualification."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    smoke = args.run.resolve(strict=True) / "qualification/quick_smoke"
    rows = [json.loads(line) for line in (smoke / "train_metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 1 or rows[0].get("global_step") != 1:
        raise ValueError("Quick smoke must complete exactly one optimizer update")
    row = rows[0]
    for key in ("loss", "grad_norm", "step_time_seconds"):
        value = float(row[key])
        if not math.isfinite(value) or (key == "step_time_seconds" and value <= 0):
            raise ValueError(f"Quick smoke has invalid {key}")
    tokens = float(row["router/optimization/step_global_eligible_tokens"])
    if not math.isfinite(tokens) or tokens <= 0:
        raise ValueError("Quick smoke processed no eligible Router tokens")
    if row.get("router/optimization/step_token_mean_enabled") != 1.0:
        raise ValueError("The v4.2 eligible-token reduction was not enabled")
    checkpoint = smoke / "checkpoints/global_step_1"
    if not all((checkpoint / section / ".metadata").is_file() for section in ("model", "optimizer")):
        raise ValueError("Quick smoke DCP model or optimizer metadata is incomplete")
    if len(list((checkpoint / "extra_state").glob("extra_state_rank_*.pt"))) != 1:
        raise ValueError("Quick smoke DCP rank state is incomplete")
    if not (smoke / "trainable_exports/global_step_1/trainable_state.pt").is_file():
        raise ValueError("Quick smoke Router export is missing")
    result = {
        "status": "passed", "scope": "single_gpu_real_weight_forward_backward_dcp_save",
        "optimizer_updates": 1, "eligible_tokens": int(tokens),
        "step_time_seconds": float(row["step_time_seconds"]),
        "formal_eight_gpu_qualification": False,
    }
    marker = smoke / "QUICK_SMOKE_PASSED.json"
    if marker.exists():
        raise FileExistsError("Quick smoke success marker already exists")
    temporary = marker.with_name("." + marker.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
