#!/usr/bin/env python3
"""Strictly admit an audited v4.2 Qwen3.5-2B head for pinned native vLLM.

Run in the pinned Transformers 5 training/evaluation environment. This is a
head-only validation; it does not claim that native inference passed quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path, help="Prepared hf_runtime directory")
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    run_root = Path(os.environ["ROUTER_RUN_ROOT"]).resolve(strict=True)
    if not runtime.is_relative_to(run_root) or runtime.name != "hf_runtime":
        raise ValueError("Admission target must be a canonical prepared runtime")
    marker = json.loads((runtime / "RUNTIME_PREPARED.json").read_text())
    config_path = runtime / "config.json"
    audit_path = runtime / "runtime_audit.json"
    from qwen3_5_router_runtime_experiments.portable_head import file_sha256

    if (marker.get("status") != "passed"
            or marker.get("config_sha256") != file_sha256(config_path)
            or marker.get("runtime_audit_sha256") != file_sha256(audit_path)):
        raise ValueError("Prepared runtime marker changed")
    output = runtime.parent / "vllm_head_admission.json"
    if output.exists():
        raise FileExistsError("Preserve the previous admission attempt")
    config_doc = json.loads(config_path.read_text())
    checkpoint = Path(config_doc["router_checkpoint"]).resolve(strict=True)
    base = Path(config_doc["router_base_model_path"]).resolve(strict=True)
    run = runtime.parents[2]
    # Both an individual export and a three-point merge must belong to this run.
    if not checkpoint.is_relative_to(run):
        raise ValueError("Router checkpoint belongs to another run")
    if base.name != "Qwen3.5-2B":
        raise ValueError("Native runtime only supports the selected Qwen3.5-2B base")
    if file_sha256(base / "config.json") != config_doc["router_base_config_sha256"]:
        raise ValueError("Base configuration changed")

    import torch
    from torch import nn
    from hf_qwen35.configuration_qwen3_5_selected_hybrid_cache_router import Qwen3_5SelectedHybridCacheRouterConfig
    from hf_qwen35.modeling_qwen3_5_selected_hybrid_cache_router import (
        Qwen3_5SelectedHybridCacheRouterForCausalLM as Runtime,
        Qwen3_5ThreeTowerInteractionRouterHead,
    )

    config = Qwen3_5SelectedHybridCacheRouterConfig(**config_doc)
    shim = nn.Module()
    shim.config = config
    shim.router_head = Qwen3_5ThreeTowerInteractionRouterHead(config).float()
    shim.lm_head = nn.Linear(1, 1, bias=False)
    shim._router_checkpoint_loaded = False
    shim._router_checkpoint_metadata = {}
    shim._checkpoint_path = types.MethodType(Runtime._checkpoint_path, shim)
    shim._audit_path = types.MethodType(Runtime._audit_path, shim)
    Runtime._ensure_router_checkpoint(shim)
    sources = sorted({
        str(Path(sys.modules[Runtime.__module__].__file__).resolve()),
        str(Path(sys.modules[Qwen3_5SelectedHybridCacheRouterConfig.__module__].__file__).resolve()),
        str(Path(__file__).resolve()),
        str((ROOT / "src/qwen3_5_router_runtime_experiments/portable_head.py").resolve()),
    })
    bound = sources + [str(checkpoint), str(audit_path), str(base / "config.json")]
    document = {
        "format": "q35-router-head-admission-v1", "status": "passed",
        "checkpoint": str(checkpoint), "audit": str(audit_path),
        "base_model_path": str(base), "base_config": str(base / "config.json"),
        "loader_sources": sources,
        "bindings": {name: file_sha256(name) for name in bound},
        "tensor_state_sha256": shim._router_checkpoint_metadata["tensor_state_sha256"],
        "metadata": shim._router_checkpoint_metadata,
        "head_parameters": sum(p.numel() for p in shim.router_head.parameters()),
        "head_tensors": len(shim.router_head.state_dict()),
        "runtime_quality_qualified": False,
    }
    temporary = output.with_name("." + output.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    with (run / "README.md").open("a") as handle:
        handle.write(f"\n- Native head admission: {output.relative_to(run)}.\n")
    print(json.dumps({"status": "passed", "admission": str(output)}))


if __name__ == "__main__":
    main()
