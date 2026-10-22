#!/usr/bin/env python3
# Modified for the portable Router v4.2 release; see NOTICE.
"""Validate the minimum recoverability contract for a VeOmni long run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from veomni.utils.wandb_metrics import (
    ROUTER_WANDB_METRIC_SCHEMA_VERSION,
    validate_wandb_comparison_contract,
)


CANONICAL_RUN_ROOT = Path(os.environ.get("ROUTER_RUN_ROOT", "/path/to/runs")).resolve()
WANDB_METRIC_SCHEMA = (
    REPO_ROOT
    / "src/configs/metric_schemas/router_wandb_comparison_v1.json"
)
WANDB_FIELDS = (
    "wandb_entity",
    "wandb_project",
    "wandb_name",
    "wandb_group",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(
    config_path: Path,
    max_checkpoint_steps: int,
    measured_step_seconds: float | None,
    max_recovery_seconds: float | None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("train"), dict):
        raise ValueError("configuration lacks a train mapping")
    train = payload["train"]
    output_dir = Path(str(train.get("output_dir", ""))).resolve()
    try:
        output_dir.relative_to(CANONICAL_RUN_ROOT)
    except ValueError as error:
        raise ValueError(f"output_dir is outside {CANONICAL_RUN_ROOT}: {output_dir}") from error
    if output_dir == CANONICAL_RUN_ROOT:
        raise ValueError("output_dir must be a named run below ROUTER_RUN_ROOT")
    if train.get("ckpt_manager") != "dcp":
        raise ValueError("long-run recovery currently requires ckpt_manager: dcp")
    stop = train.get("stop_after_global_step")
    save = train.get("save_steps")
    if not isinstance(stop, int) or stop <= 0:
        raise ValueError("stop_after_global_step must be a positive integer")
    if not isinstance(save, int) or save <= 0:
        raise ValueError("save_steps must be a positive integer")
    if save > max_checkpoint_steps:
        raise ValueError(
            f"save_steps={save} exceeds the recovery cap {max_checkpoint_steps}"
        )
    if save > stop:
        raise ValueError("save_steps cannot exceed the formal stop step")
    if train.get("export_trainable_weights") is not True:
        raise ValueError("export_trainable_weights must be true")
    if train.get("use_wandb") is not True:
        raise ValueError("use_wandb must be true for a long run")
    missing_wandb = [name for name in WANDB_FIELDS if not str(train.get(name, "")).strip()]
    if missing_wandb:
        raise ValueError(f"missing W&B fields: {missing_wandb}")
    tags = train.get("wandb_tags")
    if not isinstance(tags, list) or not tags or any(not str(tag).strip() for tag in tags):
        raise ValueError("wandb_tags must be a nonempty list of nonempty values")
    wandb_comparison_contract = validate_wandb_comparison_contract(
        train, required=True
    )
    metric_schema = json.loads(WANDB_METRIC_SCHEMA.read_text(encoding="utf-8"))
    if metric_schema.get("schema_version") != ROUTER_WANDB_METRIC_SCHEMA_VERSION:
        raise ValueError(
            "frozen Router W&B metric schema is missing or has the wrong version"
        )
    worst_case_seconds = None
    if measured_step_seconds is not None:
        if measured_step_seconds <= 0:
            raise ValueError("measured_step_seconds must be positive")
        worst_case_seconds = save * measured_step_seconds
        if max_recovery_seconds is not None and worst_case_seconds > max_recovery_seconds:
            raise ValueError(
                f"checkpoint interval risks {worst_case_seconds:.1f}s of replay, "
                f"above the cap {max_recovery_seconds:.1f}s"
            )
    return {
        "format": "veomni-resilient-longrun-config-audit-v2",
        "status": "passed",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "output_dir": str(output_dir),
        "stop_after_global_step": stop,
        "checkpoint_interval_steps": save,
        "maximum_checkpoint_interval_steps": max_checkpoint_steps,
        "measured_step_seconds": measured_step_seconds,
        "worst_case_replay_seconds": worst_case_seconds,
        "maximum_replay_seconds": max_recovery_seconds,
        "checkpoint_manager": "dcp",
        "router_export_enabled": True,
        "wandb_online_metadata_complete": True,
        "wandb_comparison_contract": wandb_comparison_contract,
        "wandb_metric_schema": str(WANDB_METRIC_SCHEMA),
        "wandb_metric_schema_sha256": sha256_file(WANDB_METRIC_SCHEMA),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-checkpoint-steps", type=int, default=512)
    parser.add_argument("--measured-step-seconds", type=float)
    parser.add_argument("--max-recovery-seconds", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_checkpoint_steps <= 0 or (args.max_recovery_seconds is not None and args.max_recovery_seconds <= 0):
        raise SystemExit("recovery caps must be positive")
    report = validate(
        args.config,
        args.max_checkpoint_steps,
        args.measured_step_seconds,
        args.max_recovery_seconds,
    )
    if args.output:
        atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
