#!/usr/bin/env python3
"""Fail-closed online W&B and recovery admission before formal training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    root = Path(os.environ["ROUTER_RUN_ROOT"]).resolve(strict=True)
    if run.parent != root:
        raise ValueError("Formal run must be directly under ROUTER_RUN_ROOT")
    train = yaml.safe_load((run / "veomni_cli.yaml").read_text())["train"]
    scientific = json.loads((run / "configs/scientific_contract.json").read_text())
    cap = 1024 if scientific.get("variant") == "qwen35_2b_paper_reference_mb8" else 512
    if train.get("save_steps") != cap:
        raise ValueError("Save interval differs from the named v4.2 recipe")
    smoke = json.loads((run / "qualification/smoke/SMOKE_PASSED.json").read_text())
    if (smoke.get("status") != "passed" or smoke.get("save_steps") != cap
            or not smoke.get("conservative_save_window_seconds")):
        raise ValueError("Real-weight smoke timing is incomplete")
    if os.environ.get("WANDB_MODE") != "online":
        raise ValueError("Formal training requires WANDB_MODE=online")
    if os.environ.get("WANDB_RESUME") or os.environ.get("WANDB_RUN_ID"):
        raise ValueError("Fresh training must not inherit W&B resume identity")
    import wandb
    viewer = wandb.Api(timeout=20).viewer
    if not viewer:
        raise ValueError("W&B account could not be authenticated online")
    from sys import path as sys_path
    repo = Path(__file__).resolve().parents[1]
    sys_path.insert(0, str(repo / "src"))
    from tools.validate_resilient_longrun_config import validate

    report = validate(run / "veomni_cli.yaml", cap,
                      smoke["conservative_save_window_seconds"] / cap,
                      None)
    output = run / "PREFLIGHT_PASSED.json"
    if output.exists():
        raise FileExistsError("Preflight marker already exists")
    output.write_text(json.dumps({"status": "passed", "recovery": report,
                                  "replay_window_seconds": smoke["conservative_save_window_seconds"],
                                  "wandb_online_authenticated": True}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "replay_window_seconds": smoke["conservative_save_window_seconds"]}))


if __name__ == "__main__":
    main()
