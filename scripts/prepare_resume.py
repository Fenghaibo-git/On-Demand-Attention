#!/usr/bin/env python3
"""Select a verified DCP and archive any post-checkpoint ledger before resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from audit_run import audit_dcp


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path: Path, contents: bytes) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(contents)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint-step", type=int, choices=(512, 1024, 1536, 2048, 2560))
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    root = Path(os.environ["ROUTER_RUN_ROOT"]).resolve(strict=True)
    if run.parent != root or not (run / "PREPARED.json").is_file():
        raise ValueError("Resume target is not a prepared canonical run")
    if (run / "training_passed.json").exists():
        raise ValueError("Completed training must not be resumed")
    config = yaml.safe_load((run / "veomni_cli.yaml").read_text())
    train = config["train"]
    scientific = json.loads((run / "configs/scientific_contract.json").read_text())
    save_steps = 1024 if scientific.get("variant") == "qwen35_2b_paper_reference_mb8" else 512
    if (train.get("ckpt_manager") != "dcp" or train.get("use_wandb") is not True
            or train.get("save_steps") != save_steps or train.get("stop_after_global_step") != 3072):
        raise ValueError("Formal v4.2 recovery contract changed")
    wandb = json.loads((run / "wandb_run.json").read_text())
    if not wandb.get("id") or not wandb.get("url") or wandb.get("entity") != train["wandb_entity"]:
        raise ValueError("Original W&B run identity is missing")
    if not (run / "metric_contract.json").is_file():
        raise ValueError("Cannot resume without the original metric contract")
    ledger = run / "train_metrics.jsonl"
    original = ledger.read_bytes()
    lines = [line for line in original.splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    steps = [row.get("global_step") for row in rows]
    if steps != list(range(1, len(rows) + 1)) or len(rows) >= 3072:
        raise ValueError("Existing formal ledger must be a contiguous incomplete prefix")
    if any(row.get("comparison/schema/exact_step_aggregation") != 1 for row in rows):
        raise ValueError("Existing step comparison aggregation is incomplete")
    if args.checkpoint_step is not None and args.checkpoint_step % save_steps:
        raise ValueError("Checkpoint step is outside the configured save grid")
    candidates = ([args.checkpoint_step] if args.checkpoint_step is not None else
                  [step for step in range(3072 - save_steps, 0, -save_steps) if step <= len(rows)])
    safe = None
    diagnostics = {}
    for step in candidates:
        if step is None or step > len(rows):
            continue
        try:
            diagnostics = audit_dcp(run, step, 8)
            safe = step
            break
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError):
            continue
    if safe is None:
        raise ValueError("No verified DCP lies within the complete metrics prefix")
    checkpoint = run / f"checkpoints/global_step_{safe}"
    prepared = json.loads((run / "PREPARED.json").read_text())
    model = json.loads((run / "model_config/config.json").read_text())
    contract = diagnostics["training_contract"]
    expected = {
        "architectures": model["architectures"],
        "router_schema_sha256": model["router_schema_sha256"],
        "router_robust_decision_margin_objective_sha256": model["router_robust_decision_margin_objective_sha256"],
        "router_initial_tensor_sha256": prepared["initial_tensor_sha256"],
        "router_optimization_protocol_sha256": prepared["protocol_sha256"],
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("DCP training contract differs from the prepared v4.2 run")
    recovery = run / "recovery"
    attempt = next((recovery / f"attempt_{index:03d}" for index in range(1, 1000)
                    if not (recovery / f"attempt_{index:03d}").exists()), None)
    if attempt is None:
        raise ValueError("Recovery attempt counter exhausted")
    attempt.mkdir(parents=True)
    archive = attempt / "train_metrics_before_resume.jsonl"
    archive.write_bytes(original)
    prefix = b"\n".join(lines[:safe]) + b"\n"
    if len(rows) > safe:
        atomic_bytes(ledger, prefix)
    config["train"]["load_checkpoint_path"] = str(checkpoint)
    (attempt / "resume.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (attempt / "README.md").write_text(
        "# Router v4.2 recovery attempt\n\nThe original metrics prefix and selected checkpoint were audited. "
        "The pre-resume ledger is preserved here. A failed restart may be retried from "
        "another verified checkpoint without rerunning completed training.\n"
    )
    report = {"status": "prepared", "checkpoint_step": safe,
              "checkpoint": str(checkpoint), "dcp_audit": diagnostics,
              "archived_metrics_sha256": sha256(archive),
              "resumed_prefix_sha256": sha256(ledger),
              "metric_contract_sha256": sha256(run / "metric_contract.json"),
              "wandb_run_id": wandb["id"], "attempt": str(attempt)}
    (attempt / "RESUME_PREPARED.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (run / "README.md").open("a") as handle:
        handle.write(f"\n- Recovery attempt: {attempt.relative_to(run)} from DCP step {safe}.\n")
    print(json.dumps({"status": "prepared", "checkpoint_step": safe,
                      "attempt": str(attempt), "wandb_run_id": wandb["id"]}))


if __name__ == "__main__":
    main()
