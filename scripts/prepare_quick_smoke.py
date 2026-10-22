#!/usr/bin/env python3
"""Prepare a one-GPU, one-update v4.2 diagnostic from a real prepared run.

This diagnostic deliberately changes batch size, DP size and sequence length.
It does not qualify the eight-GPU formal training configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    prepared = json.loads((run / "PREPARED.json").read_text())
    scientific = json.loads((run / "configs/scientific_contract.json").read_text())
    if prepared.get("status") != "passed" or scientific.get("variant") != "qwen35_2b_mb1":
        raise ValueError("Quick smoke requires a prepared Qwen3.5-2B MB1 v4.2 run")
    config = yaml.safe_load((run / "veomni_cli.yaml").read_text())
    if Path(config["train"]["output_dir"]).resolve() != run:
        raise ValueError("Training config does not belong to this run")
    if config["train"]["data_parallel_shard_size"] != 8 or config["train"]["global_batch_size"] != 64:
        raise ValueError("Unexpected formal batch configuration")
    target = run / "qualification/quick_smoke"
    if target.exists():
        raise FileExistsError("Preserve the previous quick smoke attempt")

    changes = {
        "data_parallel_shard_size": 1,
        "global_batch_size": 1,
        "max_seq_len": 16384,
        "max_steps": 3072,
        "stop_after_global_step": 1,
        "save_steps": 1,
        "use_wandb": False,
    }
    contract = {
        "format": "router-v42-qwen35-2b-quick-smoke-v1",
        "formal_protocol_sha256": prepared["protocol_sha256"],
        "diagnostic_changes": changes,
        "scope": "single_gpu_real_weight_forward_backward_dcp_save",
    }
    probe_hash = digest_json(contract)
    config["data"]["max_seq_len"] = changes["max_seq_len"]
    config["data"]["num_workers"] = 1
    config["train"].update(
        output_dir=str(target),
        data_parallel_shard_size=1,
        global_batch_size=1,
        max_steps=3072,
        stop_after_global_step=1,
        save_steps=1,
        use_wandb=False,
        model_probe_steps=0,
        enable_profiling=False,
        load_checkpoint_path=None,
        wandb_entity=None,
        wandb_project=None,
        wandb_name=None,
        wandb_comparison_family="router-v42-qwen35-2b-quick-smoke",
        wandb_group="router-v42-qwen35-2b-quick-smoke",
        wandb_comparison_protocol_hash=probe_hash,
    )
    config["model"]["config_kwargs"]["router_optimization_protocol_sha256"] = probe_hash
    target.mkdir(parents=True)
    (target / "quick_smoke.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (target / "diagnostic_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (target / "README.md").write_text(
        "# Single-GPU quick smoke\n\n"
        "One real-weight optimizer update and DCP save at the formal 16K data boundary, DP1 and batch 1. "
        "See diagnostic_contract.json for every difference from the formal recipe. "
        "This result does not authorize formal training.\n"
    )
    print(json.dumps({"status": "prepared", "config": str(target / "quick_smoke.yaml"),
                      "diagnostic_protocol_sha256": probe_hash}))


if __name__ == "__main__":
    main()
