#!/usr/bin/env python3
"""Prepare one resume from smoke step 1, preserving the original smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    smoke = args.run.resolve(strict=True) / "qualification/smoke"
    source = smoke / "train_metrics.jsonl"
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    if [json.loads(line)["global_step"] for line in lines] != [1, 2]:
        raise ValueError("Smoke must complete steps 1 and 2 before restore")
    checkpoint = smoke / "checkpoints/global_step_1"
    if not checkpoint.is_dir():
        raise ValueError("Step-1 DCP is missing")
    destination = smoke.parent / "restored_from_step_1"
    if destination.exists():
        raise FileExistsError("Preserve the previous restore attempt")
    destination.mkdir()
    config = yaml.safe_load((smoke / "smoke.yaml").read_text())
    config["train"]["output_dir"] = str(destination)
    config["train"]["load_checkpoint_path"] = str(checkpoint)
    (destination / "restore.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (destination / "train_metrics.jsonl").write_text(lines[0] + "\n")
    (destination / "metric_contract.json").write_bytes((smoke / "metric_contract.json").read_bytes())
    (destination / "README.md").write_text("# DCP restore qualification\n\nResume at step 1 and recompute step 2.\n")


if __name__ == "__main__":
    main()
