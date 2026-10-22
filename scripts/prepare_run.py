#!/usr/bin/env python3
"""Materialize one fresh v4.2 run from verified external assets.

Run this in the pinned training environment. All generated assets go under
ROUTER_RUN_ROOT; model and dataset payloads remain at their supplied paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "qwen35_2b_mb1": "qwen35_2b_mb1",
    "qwen35_2b_paper_reference_mb8": "qwen35_2b_mb1",
    "qwen3_8b": "qwen3_8b",
    "qwen35_27b": "qwen35_27b",
    "gemma4_12b": "gemma4_12b",
}
BASE_NAMES = {
    "qwen35_2b_mb1": "Qwen3.5-2B",
    "qwen3_8b": "Qwen3-8B",
    "qwen35_27b": "Qwen3.5-27B",
    "gemma4_12b": "gemma-4-12B-it",
}
SOURCE_SUFFIXES = (".safetensors",)
A0_QWEN35_2B_SHA256 = "8a2225f29a78ff5f1f8d3ff5e79b980cd972714cd55ddbd5df100339382e4982"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_template(variant: str) -> tuple[dict, dict]:
    template = yaml.safe_load((ROOT / "configs/train" / (variant + ".yaml")).read_text())
    model_key = VARIANTS[variant]
    model = json.loads((ROOT / "configs/model" / (model_key + ".json")).read_text())
    train = template["train"]
    expected = {
        "global_batch_size": 64, "micro_batch_size": 8 if variant.endswith("paper_reference_mb8") else 1,
        "save_steps": 1024 if variant.endswith("paper_reference_mb8") else 512,
        "stop_after_global_step": 3072, "num_train_epochs": 1,
        "lr": 3e-4, "lr_min": 3e-5, "lr_start": 0.0,
        "lr_warmup_ratio": 102 / 3072, "lr_decay_start_ratio": 102 / 3072,
        "lr_decay_style": "cosine", "ckpt_manager": "dcp",
        "optimizer": "adamw", "weight_decay": 0.0, "max_grad_norm": 1.0,
        "data_parallel_shard_size": 8, "data_parallel_mode": "ddp",
        "rmpad": False, "dyn_bsz": False, "seed": 20260827,
    }
    for key, value in expected.items():
        if train.get(key) != value:
            raise ValueError(f"Template changed v4.2 training field: {key}")
    if template["data"].get("max_seq_len") != 16384:
        raise ValueError("Template changed the 16K data boundary")
    if template["model"]["config_kwargs"].get("router_optimizer_step_reduction") != "global_eligible_token_mean_v1":
        raise ValueError("Template changed whole-step eligible-token normalization")
    if train.get("use_wandb") is not True or train.get("wandb_metric_schema_version") != "router-comparison-v1":
        raise ValueError("Formal training requires online comparison metrics")
    if not all(train.get(key) is None for key in ("wandb_entity", "wandb_project", "wandb_comparison_protocol_hash")):
        raise ValueError("Submission templates must require runtime identity binding")
    if model.get("router_history_nll_threshold") != 0.0:
        raise ValueError("History Local rule must use raw g < 0")
    if model.get("router_robust_decision_margin_lambda") != 0.001:
        raise ValueError("The v4.2 decision margin shifted")
    if model.get("architectures") != template["model"]["config_kwargs"].get("architectures"):
        raise ValueError("Model and training templates select different architectures")
    return template, model


def build_model_manifest(model_path: Path) -> dict:
    files = sorted(path for path in model_path.rglob("*.safetensors") if path.is_file() and not path.is_symlink())
    if not files:
        raise ValueError("Base model must have real safetensors at the supplied path")
    return {
        "format": "router-v42-base-weights-v1",
        "files": [
            {"path": path.relative_to(model_path).as_posix(), "bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in files
        ],
    }


def count_parquet_rows(train_path: Path) -> int:
    import pyarrow.parquet as pq

    shards = sorted(train_path.rglob("*.parquet"))
    if not shards:
        raise ValueError("No Parquet training shards found")
    return sum(pq.ParquetFile(path).metadata.num_rows for path in shards)


def configure_model(model_dict: dict, variant: str, base_path: Path, weights_manifest_hash: str) -> tuple[dict, object]:
    from transformers import AutoConfig

    model_dict["router_base_model_path"] = str(base_path)
    model_dict["router_base_config_sha256"] = digest_file(base_path / "config.json")
    model_dict["router_base_weights_manifest_sha256"] = weights_manifest_hash
    model_dict["router_schema_sha256"] = None
    model_dict["router_robust_decision_margin_objective_sha256"] = None
    model_dict["router_optimization_protocol_sha256"] = None
    config = AutoConfig.for_model(model_dict.pop("model_type"), **model_dict)
    model_dict["model_type"] = config.model_type
    if variant.startswith("qwen35_2b"):
        from veomni.models.transformers.qwen3_5_three_stream_router.modeling_qwen3_5_three_stream_router import _set_qwen3_5_router_defaults
        from veomni.models.transformers.qwen3_5_three_stream_nll_selected_robust_decision_margin_router.modeling_qwen3_5_three_stream_nll_selected_robust_decision_margin_router import _ROBUST_MARGIN_DEFAULTS, robust_decision_margin_objective_schema_sha256

        for name, value in _ROBUST_MARGIN_DEFAULTS.items():
            setattr(config, name, value)
        # The structural schema predates the robust objective enum. The
        # model's own constructor applies the same temporary legacy value.
        config.router_objective = "weighted_bce"
        try:
            _set_qwen3_5_router_defaults(config)
        finally:
            config.router_objective = "robust_decision_margin_huber"
        config.router_robust_decision_margin_objective_sha256 = robust_decision_margin_objective_schema_sha256(config)
    elif variant == "qwen3_8b":
        from veomni.models.transformers.qwen3_three_stream_router.modeling_qwen3_three_stream_router import three_stream_schema_sha256
        from veomni.models.transformers.qwen3_large_robust_margin_router.objective import objective_sha256

        config.router_schema_sha256 = three_stream_schema_sha256(config)
        config.router_robust_decision_margin_objective_sha256 = objective_sha256(config)
    elif variant == "qwen35_27b":
        from veomni.models.transformers.qwen3_5_large_robust_margin_router.modeling_qwen3_5_large_robust_margin_router import set_scaled_hybrid_defaults
        from veomni.models.transformers.qwen3_large_robust_margin_router.objective import objective_sha256

        set_scaled_hybrid_defaults(config)
        config.router_robust_decision_margin_objective_sha256 = objective_sha256(config)
    elif variant == "gemma4_12b":
        from veomni.models.transformers.gemma4_unified_three_stream_router.modeling_gemma4_unified_three_stream_router import set_defaults
        from veomni.models.transformers.qwen3_large_robust_margin_router.objective import objective_sha256

        set_defaults(config)
        config.router_robust_decision_margin_objective_sha256 = objective_sha256(config)
    else:
        raise ValueError("Unsupported method variant")
    result = config.to_dict()
    for key in ("router_schema_sha256", "router_robust_decision_margin_objective_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(result.get(key, ""))):
            raise ValueError(f"Model contract lacks {key}")
    return result, config


def initialize_a0(config, destination: Path, expected_hash: str | None = None) -> str:
    import torch
    from torch import nn
    from veomni.models.transformers.qwen3_router.modeling_qwen3_router import Qwen3RMSNorm, _build_router_head
    from veomni.utils.tensor_state import tensor_state_sha256

    torch.manual_seed(20260827)
    head = _build_router_head(config).float()
    def initialize(module):
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                module.weight.normal_(0.0, float(config.initializer_range))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, Qwen3RMSNorm):
                module.weight.fill_(1.0)
    head.apply(initialize)
    state = {"router_head." + name: tensor.detach().cpu().contiguous() for name, tensor in head.state_dict().items()}
    if len(state) != int(config.router_expected_head_tensor_count):
        raise ValueError("A0 tensor count mismatch")
    if sum(value.numel() for value in state.values()) != int(config.router_expected_head_parameter_count):
        raise ValueError("A0 parameter count mismatch")
    fingerprint = tensor_state_sha256(state)
    if expected_hash is not None and fingerprint != expected_hash:
        raise ValueError("Seeded A0 differs from the verified Qwen3.5-2B reference")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    torch.save({"format_version": 1, "tensor_state_sha256": fingerprint, "state_dict": state,
                "metadata": {"method": "v4.2", "variant": "A0", "seed": 20260827}}, temporary)
    os.replace(temporary, destination)
    return fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--check-template", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    template, model_dict = read_template(args.variant)
    if args.check_template:
        print(json.dumps({"status": "passed", "variant": args.variant, "method": "v4.2"}))
        return 0
    root_value = os.environ.get("ROUTER_RUN_ROOT")
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not all((root_value, os.environ.get("ROUTER_MODEL_ROOT"), os.environ.get("ROUTER_DATA_ROOT"),
                entity, project, args.model_path, args.train_path, args.dataset_manifest, args.run_name)):
        parser.error("ROUTER_RUN_ROOT, ROUTER_MODEL_ROOT, ROUTER_DATA_ROOT, W&B identity and all asset/run arguments are required")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.run_name):
        parser.error("run name must be a simple directory name")
    root = Path(root_value).resolve(strict=True)
    run = root / args.run_name
    if run.exists() or run.is_symlink():
        raise FileExistsError("Fresh run directory already exists")
    model_path = args.model_path.resolve(strict=True)
    train_path = args.train_path.resolve(strict=True)
    dataset_manifest = args.dataset_manifest.resolve(strict=True)
    if not model_path.is_dir() or not train_path.is_dir() or not dataset_manifest.is_file():
        raise ValueError("Base model, train split, and dataset manifest are required")
    model_root = Path(os.environ["ROUTER_MODEL_ROOT"]).resolve(strict=True)
    data_root = Path(os.environ["ROUTER_DATA_ROOT"]).resolve(strict=True)
    if model_path.parent != model_root or not train_path.is_relative_to(data_root / "processed"):
        raise ValueError("Model and training data must be in their canonical roots")
    if dataset_manifest != train_path.parents[1] / "manifest.json":
        raise ValueError("Dataset manifest must bind the supplied train split")
    for catalog in (model_root / "README.md", model_path / "README.md",
                    data_root / "processed/README.md", dataset_manifest.parent / "README.md"):
        if not catalog.is_file():
            raise ValueError("Read the model and dataset catalogs before preparing a run")
        catalog.read_text()
    if model_path.name != BASE_NAMES[VARIANTS[args.variant]]:
        raise ValueError("Supplied base model does not match the selected variant")
    if count_parquet_rows(train_path) != 196608:
        raise ValueError("v4.2 requires exactly 196608 training rows")
    base_config_hash = digest_file(model_path / "config.json")
    if base_config_hash != model_dict.get("router_base_config_sha256"):
        raise ValueError("Base model config does not match the frozen model revision")
    weights_manifest = build_model_manifest(model_path)
    dataset_hash = digest_file(dataset_manifest)
    model_config, config = configure_model(model_dict, args.variant, model_path, digest_json(weights_manifest))
    run.mkdir(mode=0o775)
    (run / "model_config").mkdir()
    (run / "configs").mkdir()
    atomic_json(run / "model_weights_manifest.json", weights_manifest)
    initial_path = run / "initialization/router_initial_state.pt"
    initial_hash = initialize_a0(
        config, initial_path,
        A0_QWEN35_2B_SHA256 if args.variant.startswith("qwen35_2b") else None,
    )
    scientific = {
        "method": "v4.2", "variant": args.variant, "model_config_sha256": digest_json(model_config),
        "base_config_sha256": base_config_hash, "base_weights_manifest_sha256": digest_json(weights_manifest),
        "dataset_manifest_sha256": dataset_hash, "initial_tensor_sha256": initial_hash,
        "history_local_rule": "g<0", "runtime_full_rule": "score>0",
        "target": "signed_log1p(g-0.001)", "loss": "Huber(delta=1), global eligible-token mean",
        "seed": 20260827, "global_batch_size": 64,
        "micro_batch_size": template["train"]["micro_batch_size"], "dp": 8,
        "updates": 3072, "save_steps": template["train"]["save_steps"],
        "warmup_steps": 102, "peak_lr": 3e-4, "minimum_lr": 3e-5,
    }
    protocol_hash = digest_json(scientific)
    model_config["router_optimization_protocol_sha256"] = protocol_hash
    atomic_json(run / "model_config/config.json", model_config)
    atomic_json(run / "configs/scientific_contract.json", scientific)
    data = template["data"]
    data["train_path"] = str(train_path)
    model = template["model"]
    model["config_path"] = str(run / "model_config")
    model["model_path"] = str(model_path)
    model["tokenizer_path"] = str(model_path)
    model["config_kwargs"].update(
        router_optimization_protocol_sha256=protocol_hash,
        router_optimizer_step_reduction="global_eligible_token_mean_v1",
    )
    for key in ("router_schema_sha256", "router_robust_decision_margin_objective_sha256"):
        model["config_kwargs"][key] = model_config[key]
    train = template["train"]
    train["output_dir"] = str(run)
    train["initial_trainable_weights_path"] = str(initial_path)
    train["initial_trainable_weights_sha256"] = initial_hash
    train["wandb_entity"] = entity
    train["wandb_project"] = project
    train["wandb_name"] = args.run_name
    train["wandb_comparison_protocol_hash"] = protocol_hash
    train["wandb_group"] = train["wandb_comparison_family"]
    train["wandb_tags"].append(args.run_name)
    (run / "veomni_cli.yaml").write_text(yaml.safe_dump(template, sort_keys=True, allow_unicode=True))
    (run / "README.md").write_text("# Router v4.2 run\n\nPrepared from public templates. See configs/scientific_contract.json and veomni_cli.yaml. Training and evaluation have not started.\n")
    atomic_json(run / "PREPARED.json", {"status": "passed", "protocol_sha256": protocol_hash,
                                      "scientific_contract_sha256": digest_file(run / "configs/scientific_contract.json"),
                                      "dataset_manifest_sha256": dataset_hash, "initial_tensor_sha256": initial_hash})
    print(json.dumps({"status": "prepared", "run_dir": str(run), "protocol_sha256": protocol_hash}))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.exit(main())
