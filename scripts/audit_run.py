#!/usr/bin/env python3
"""Audit a completed formal v4.2 run without restarting training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dcp(root: Path, step: int, ranks: int) -> dict:
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = root / f"checkpoints/global_step_{step}"
    result = {}
    for name in ("model", "optimizer"):
        path = checkpoint / name
        metadata_file = path / ".metadata"
        if not metadata_file.is_file():
            raise ValueError(f"DCP {step} lacks {name} metadata")
        metadata = FileSystemReader(str(path)).read_metadata()
        if not metadata.state_dict_metadata or not metadata.storage_data:
            raise ValueError(f"DCP {step} {name} has no stored state")
        for storage in metadata.storage_data.values():
            shard = path / storage.relative_path
            if not shard.is_file() or shard.stat().st_size < storage.offset + storage.length:
                raise ValueError(f"DCP {step} {name} is missing a referenced shard")
        result[name] = {"metadata_sha256": sha256(metadata_file),
                        "state_keys": len(metadata.state_dict_metadata),
                        "storage_entries": len(metadata.storage_data)}
    states = sorted((checkpoint / "extra_state").glob("extra_state_rank_*.pt"))
    if len(states) != ranks or {path.stem for path in states} != {f"extra_state_rank_{rank}" for rank in range(ranks)}:
        raise ValueError(f"DCP {step} lacks a complete {ranks}-rank extra state")
    contract = None
    for path in states:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("global_step") != step:
            raise ValueError(f"DCP {step} extra-state step mismatch")
        for key in ("lr_scheduler", "train_dataloader", "torch_rng_state", "training_contract"):
            if key not in payload:
                raise ValueError(f"DCP {step} lacks {key}")
        if contract is None:
            contract = payload["training_contract"]
        elif payload["training_contract"] != contract:
            raise ValueError(f"DCP {step} rank contracts differ")
    result["training_contract"] = contract
    result["extra_state"] = {"ranks": ranks,
                             "sha256": {path.name: sha256(path) for path in states}}
    return result


def audit_export(root: Path, step: int, config: dict) -> dict:
    import torch
    from veomni.utils.tensor_state import tensor_state_sha256

    path = root / f"trainable_exports/global_step_{step}/trainable_state.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("global_step") != step:
        raise ValueError(f"Router export {step} has the wrong identity")
    metadata = payload.get("model_config", {})
    for key in ("router_schema_sha256", "router_robust_decision_margin_objective_sha256",
                "router_optimization_protocol_sha256"):
        if metadata.get(key) != config.get(key):
            raise ValueError(f"Router export {step} changed {key}")
    state = payload.get("state_dict", {})
    if not state or not all(name.startswith("router_head.") for name in state):
        raise ValueError(f"Router export {step} includes a non-Head tensor")
    if not all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError(f"Router export {step} must be finite FP32")
    tensor_hash = tensor_state_sha256(state)
    if tensor_hash != payload.get("tensor_state_sha256"):
        raise ValueError(f"Router export {step} tensor hash mismatch")
    return {"sha256": sha256(path), "tensor_state_sha256": tensor_hash,
            "tensors": len(state), "parameters": sum(value.numel() for value in state.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path(os.environ["ROUTER_RUN_ROOT"]).resolve(strict=True)
    run = args.run.resolve(strict=True)
    if run.parent != root:
        raise ValueError("Audit target is outside the canonical run root")
    if (run / "training_passed.json").exists():
        raise FileExistsError("Training audit already passed; preserve existing evidence")
    training = yaml.safe_load((run / "veomni_cli.yaml").read_text())
    train = training["train"]
    model = json.loads((run / "model_config/config.json").read_text())
    prepared = json.loads((run / "PREPARED.json").read_text())
    if prepared.get("protocol_sha256") != model.get("router_optimization_protocol_sha256"):
        raise ValueError("Prepared and model protocols differ")
    if train.get("use_wandb") is not True or train.get("ckpt_manager") != "dcp":
        raise ValueError("Formal run lacks online logging or DCP")
    wandb = json.loads((run / "wandb_run.json").read_text())
    if not wandb.get("id") or not wandb.get("url") or wandb.get("entity") != train.get("wandb_entity"):
        raise ValueError("W&B run identity is incomplete")
    expected_steps = list(range(1, 3073))
    ledger = run / "train_metrics.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    if [row.get("global_step") for row in rows] != expected_steps:
        raise ValueError("Formal local metrics must contain exactly steps 1..3072")
    for row in rows:
        if row.get("comparison/schema/exact_step_aggregation") != 1:
            raise ValueError("Formal comparison metrics were not exactly aggregated")
        if not math.isfinite(float(row.get("loss", float("nan")))):
            raise ValueError("Formal loss contains a nonfinite step")
    points = tuple(range(int(train["save_steps"]), 3073, int(train["save_steps"])))
    if points[-1] != 3072:
        raise ValueError("Final optimizer step has no save point")
    checkpoints = {str(step): audit_dcp(run, step, 8) for step in points}
    exports = {str(step): audit_export(run, step, model) for step in points}
    report = {"status": "passed", "format": "router-v42-training-audit-v1",
              "protocol_sha256": prepared["protocol_sha256"],
              "metrics_sha256": sha256(ledger), "metrics_steps": 3072,
              "metric_contract_sha256": sha256(run / "metric_contract.json"),
              "checkpoint_structure": checkpoints, "router_exports": exports,
              "wandb_run_id": wandb["id"], "wandb_url": wandb["url"],
              "limitation": "DCP metadata and referenced shard lengths checked; payload corruption requires a real restore"}
    output = run / "training_audit.json"
    temporary = output.with_name("." + output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    marker = run / "training_passed.json"
    temporary = marker.with_name("." + marker.name + ".tmp")
    temporary.write_text(json.dumps({"status": "passed", "audit_sha256": sha256(output),
                                     "protocol_sha256": prepared["protocol_sha256"]}, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    print(json.dumps({"status": "passed", "steps": 3072, "save_points": list(points)}))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
