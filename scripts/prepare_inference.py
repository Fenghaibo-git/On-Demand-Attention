#!/usr/bin/env python3
"""Prepare a validated Router head for the separate vLLM environment."""
import argparse
import json
import os
from pathlib import Path

from router_checkpoint import ROOT, sha256, validate_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Keep generated artifacts outside the source checkout")
    if destination.exists():
        raise FileExistsError("Preserve the existing admission directory")
    payload, config, head, audit = validate_export(args.router, args.model)
    if audit["family"] == "gemma4":
        raise ValueError("Gemma vLLM routing is not implemented; use the Transformers entry")
    destination.mkdir(parents=True)
    audit_path = destination / "export_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    base = Path(audit["base_model_path"])
    from veomni.models.transformers.qwen3_router import modeling_qwen3_router
    sources = [str(Path(__file__).resolve()), str(ROOT / "scripts/router_checkpoint.py"),
               str(Path(modeling_qwen3_router.__file__).resolve())]
    paths = sources + [audit["checkpoint"], str(audit_path), str(base / "config.json")]
    document = {
        "format": "q35-router-head-admission-v1", "status": "passed",
        "family": audit["family"], "checkpoint": audit["checkpoint"], "audit": str(audit_path),
        "base_model_path": str(base), "base_config": str(base / "config.json"),
        "loader_sources": sources, "bindings": {p: sha256(p) for p in paths},
        "tensor_state_sha256": payload["tensor_state_sha256"], "metadata": audit,
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "head_tensors": len(head.state_dict()), "runtime_quality_qualified": False,
    }
    temporary = destination / ".admission.json.tmp"
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination / "admission.json")
    (destination / "README.md").write_text("# On Demand Attention inference admission\n\n"
        "External Router export and base configuration validated for compatibility. "
        "This is not evidence of training completion or benchmark quality.\n")
    print(destination / "admission.json")


if __name__ == "__main__":
    main()
