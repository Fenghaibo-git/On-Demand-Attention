#!/usr/bin/env python3
"""Link Qwen3.5-2B base assets to an audited v4.2 selected-cache runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--step", type=int, choices=(1024, 2048, 3072))
    group.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    root = Path(os.environ["ROUTER_RUN_ROOT"]).resolve(strict=True)
    if run.parent != root:
        raise ValueError("Run is outside the configured canonical root")
    marker = json.loads((run / "training_passed.json").read_text())
    audit_path = run / "training_audit.json"
    if marker.get("status") != "passed" or marker.get("audit_sha256") != sha256(audit_path):
        raise ValueError("Training audit is missing or changed")
    training_audit = json.loads(audit_path.read_text())
    model_config = json.loads((run / "model_config/config.json").read_text())
    if model_config.get("architectures") != ["Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM"]:
        raise ValueError("This runtime is for the Qwen3.5-2B v4.2 head")
    if model_config.get("hidden_size") != 2048:
        raise ValueError("The selected-cache adapter expects Qwen3.5-2B")
    if args.merge:
        source_dir = run / "merges/uniform_1024_2048_3072"
        merge_marker = json.loads((source_dir / "merge_passed.json").read_text())
        merge_audit_path = source_dir / "merge_audit.json"
        if merge_marker.get("audit_sha256") != sha256(merge_audit_path):
            raise ValueError("Merge audit is missing or changed")
        source_audit = json.loads(merge_audit_path.read_text())
        export = source_dir / "trainable_state.pt"
        tensor_hash = source_audit["tensor_state_sha256"]
        export_hash = source_audit["export_sha256"]
        name = "merge_1024_2048_3072"
    else:
        export = run / f"trainable_exports/global_step_{args.step}/trainable_state.pt"
        source_audit = training_audit["router_exports"][str(args.step)]
        tensor_hash = source_audit["tensor_state_sha256"]
        export_hash = source_audit["sha256"]
        name = f"step_{args.step}"
    if sha256(export) != export_hash:
        raise ValueError("Audited Router export changed")
    base = Path(model_config["router_base_model_path"]).resolve(strict=True)
    if sha256(base / "config.json") != model_config["router_base_config_sha256"]:
        raise ValueError("Base model config changed")
    output = run / "evaluations" / name / "hf_runtime"
    if output.exists():
        raise FileExistsError("Preserve the previous runtime preparation")
    output.mkdir(parents=True)
    for source in sorted(base.iterdir()):
        if source.name == "config.json" or source.is_dir() or source.name.startswith("."):
            continue
        if source.suffix not in {".json", ".jinja", ".txt", ".safetensors", ".model"}:
            continue
        (output / source.name).symlink_to(source)
    for name_code in ("configuration_qwen3_5_selected_hybrid_cache_router.py",
                      "modeling_qwen3_5_selected_hybrid_cache_router.py"):
        shutil.copy2(ROOT / "src/hf_qwen35" / name_code, output / name_code)
    config = json.loads((ROOT / "src/hf_qwen35/config.json").read_text())
    config.update(router_base_model_path=str(base),
                  router_base_config_sha256=model_config["router_base_config_sha256"],
                  router_base_weights_manifest_sha256=model_config["router_base_weights_manifest_sha256"],
                  router_expected_training_schema_sha256=model_config["router_schema_sha256"],
                  router_expected_objective_sha256=model_config["router_robust_decision_margin_objective_sha256"],
                  router_checkpoint=str(export), router_audit=str(output / "runtime_audit.json"),
                  router_action_threshold=0.0)
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    runtime_audit = {"status": "passed", "objective_sha256": config["router_expected_objective_sha256"],
                     "export_tensor_state_sha256": tensor_hash, "export_sha256": export_hash,
                     "source_training_audit_sha256": sha256(audit_path)}
    (output / "runtime_audit.json").write_text(json.dumps(runtime_audit, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text("# v4.2 selected-cache runtime\n\nThis directory links separately obtained base assets and one audited Router export.\n")
    marker_path = output / "RUNTIME_PREPARED.json"
    temporary = marker_path.with_name("." + marker_path.name + ".tmp")
    temporary.write_text(json.dumps({"status": "passed", "config_sha256": sha256(output / "config.json"),
                                     "runtime_audit_sha256": sha256(output / "runtime_audit.json")}, indent=2) + "\n")
    os.replace(temporary, marker_path)
    with (run / "README.md").open("a") as handle:
        handle.write(f"\n- Evaluator runtime: {output.relative_to(run)}.\n")
    print(json.dumps({"status": "passed", "runtime": str(output)}))


if __name__ == "__main__":
    main()
