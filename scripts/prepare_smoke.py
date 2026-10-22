#!/usr/bin/env python3
"""Create an isolated two-update DCP qualification config inside a prepared run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    if not (run / "PREPARED.json").is_file():
        raise ValueError("Run is missing its prepared contract")
    config = yaml.safe_load((run / "veomni_cli.yaml").read_text())
    if Path(config["train"]["output_dir"]).resolve() != run:
        raise ValueError("Training config does not belong to this run")
    target = run / "qualification/smoke"
    if target.exists():
        raise FileExistsError("Preserve the previous smoke result")
    target.mkdir(parents=True)
    config["train"].update(output_dir=str(target), stop_after_global_step=2,
                            save_steps=1, use_wandb=False, model_probe_steps=0,
                            enable_profiling=False, load_checkpoint_path=None)
    config["train"]["wandb_entity"] = None
    config["train"]["wandb_project"] = None
    config["train"]["wandb_name"] = None
    path = target / "smoke.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True))
    (target / "README.md").write_text("# Two-update qualification\n\nThis is a smoke run; it is not part of the formal training ledger.\n")
    print(json.dumps({"status": "prepared", "config": str(path)}))


if __name__ == "__main__":
    main()
