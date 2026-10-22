#!/usr/bin/env python3
"""Create the predeclared equal-weight 1024/2048/3072 A0 Router merge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


STEPS = (1024, 2048, 3072)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    marker = json.loads((run / "training_passed.json").read_text())
    audit_path = run / "training_audit.json"
    if marker.get("status") != "passed" or marker.get("audit_sha256") != sha256(audit_path):
        raise ValueError("Source training audit is missing or changed")
    audit = json.loads(audit_path.read_text())
    output_dir = run / "merges/uniform_1024_2048_3072"
    if output_dir.exists():
        raise FileExistsError("Preserve an existing merge attempt")
    import torch
    from veomni.utils.tensor_state import tensor_state_sha256

    payloads = []
    sources = []
    for step in STEPS:
        path = run / f"trainable_exports/global_step_{step}/trainable_state.pt"
        expected = audit["router_exports"][str(step)]
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Source export at step {step} changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload["state_dict"]
        if payload.get("global_step") != step or tensor_state_sha256(state) != expected["tensor_state_sha256"]:
            raise ValueError(f"Source export at step {step} has an invalid tensor state")
        if payloads and (payload["model_class"] != payloads[0]["model_class"]
                         or payload["model_config"] != payloads[0]["model_config"]
                         or set(state) != set(payloads[0]["state_dict"])):
            raise ValueError("All source Routers must have the same schema and contract")
        payloads.append(payload)
        sources.append({"step": step, "sha256": expected["sha256"],
                        "tensor_state_sha256": expected["tensor_state_sha256"]})
    result = {}
    for name in sorted(payloads[0]["state_dict"]):
        tensors = [payload["state_dict"][name] for payload in payloads]
        if any(t.dtype != torch.float32 or t.shape != tensors[0].shape
               or not bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError(f"Source tensor {name} is not finite aligned FP32")
        result[name] = (sum(t.to(torch.float64) for t in tensors) / 3).to(torch.float32).contiguous()
    result_hash = tensor_state_sha256(result)
    output_dir.mkdir(parents=True)
    path = output_dir / "trainable_state.pt"
    torch.save({"format_version": 1, "global_step": 3072,
                "model_class": payloads[0]["model_class"],
                "model_config": payloads[0]["model_config"],
                "state_dict": result, "tensor_state_sha256": result_hash}, path)
    report = {"status": "passed", "format": "router-v42-equal-weight-merge-v1",
              "sources": sources, "weights": [1 / 3] * 3,
              "tensor_state_sha256": result_hash, "export_sha256": sha256(path),
              "objective_sha256": payloads[0]["model_config"]["router_robust_decision_margin_objective_sha256"],
              "step_semantics": "maximum_source_step_for_runtime_compatibility_not_training_step"}
    audit_file = output_dir / "merge_audit.json"
    audit_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    marker_file = output_dir / "merge_passed.json"
    temporary = marker_file.with_name("." + marker_file.name + ".tmp")
    temporary.write_text(json.dumps({"status": "passed", "audit_sha256": sha256(audit_file)}, indent=2) + "\n")
    os.replace(temporary, marker_file)
    (output_dir / "README.md").write_text("# Equal-weight A0 merge\n\nInference-only mean of steps 1024, 2048 and 3072.\n")
    print(json.dumps({"status": "passed", "export": str(path), "tensor_state_sha256": result_hash}))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
