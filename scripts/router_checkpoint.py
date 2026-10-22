"""Validate portable v4.2 exports before loading either inference backend.

This checks inference compatibility, not training completion or model quality.
The formal training audit remains a separate stage.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARCHITECTURES = {
    "Qwen3LargeRobustMarginRouterForCausalLM": "qwen3",
    "Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM": "qwen35",
    "Qwen3_5LargeRobustMarginRouterForCausalLM": "qwen35",
    "Gemma4UnifiedThreeStreamRobustMarginRouterForCausalLM": "gemma4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(value, expected):
    if isinstance(expected, bool) or isinstance(value, bool):
        return type(value) is type(expected) and value == expected
    if isinstance(expected, float) and isinstance(value, (float, int)):
        return math.isfinite(value) and math.isclose(value, expected, rel_tol=0.004, abs_tol=1e-8)
    return value == expected


def validate_export(checkpoint, base):
    import torch
    from transformers import AutoConfig
    from veomni.models.transformers.qwen3_router.modeling_qwen3_router import _build_router_head
    from veomni.utils.tensor_state import tensor_state_sha256

    checkpoint, base = Path(checkpoint).resolve(strict=True), Path(base).resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Expected a VeOmni format_version=1 Router export")
    values, state = payload.get("model_config"), payload.get("state_dict")
    if not isinstance(values, dict) or not isinstance(state, dict) or not state:
        raise ValueError("Export is missing model_config or state_dict")
    architectures = values.get("architectures", [])
    if not isinstance(architectures, list) or len(architectures) != 1 or architectures[0] not in ARCHITECTURES:
        raise ValueError("This entry accepts v4.2 robust-margin exports only")
    family = ARCHITECTURES[architectures[0]]
    expected = {
        "router_objective": "robust_decision_margin_huber",
        "router_robust_decision_margin_lambda": 0.001,
        "router_robust_decision_margin_huber_delta": 1.0,
        "router_robust_decision_margin_loss_weighting": "eligible_token_equal",
        "router_history_nll_threshold": 0.0,
        "router_one_step_transform": "signed_log1p",
        "router_one_step_positive_weight": 1.0,
        "router_one_step_sign_aux_weight": 0.0,
        "router_optimizer_step_reduction": "global_eligible_token_mean_v1",
        "router_feature_schema": ["history_state", "current_embedding", "current_swa_hidden"],
        "router_head_architecture": "three_tower_interaction_swiglu",
        "router_runtime_history_mode": "full_cache_nll_selected" if family == "qwen3" else "selected_hybrid_cache",
        "router_sink_size": 4, "router_window_size": 2048,
    }
    for name, value in expected.items():
        if not matches(values.get(name), value):
            raise ValueError(f"Incompatible v4.2 field: {name}")
    if sha256(base / "config.json") != values.get("router_base_config_sha256"):
        raise ValueError("Base config differs from the exported backbone revision")
    raw = json.loads((base / "config.json").read_text())
    native = raw.get("text_config", raw)
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size"):
        if values.get(key) != native.get(key):
            raise ValueError(f"Backbone geometry differs: {key}")
    config_values = dict(values)
    config = AutoConfig.for_model(config_values.pop("model_type"), **config_values)
    if family == "qwen3":
        from veomni.models.transformers.qwen3_three_stream_router.modeling_qwen3_three_stream_router import three_stream_schema_sha256
        structural = three_stream_schema_sha256(config)
    elif family == "qwen35":
        from veomni.models.transformers.qwen3_5_three_stream_router.modeling_qwen3_5_three_stream_router import qwen3_5_three_stream_schema_sha256
        structural = qwen3_5_three_stream_schema_sha256(config)
    else:
        from veomni.models.transformers.gemma4_unified_three_stream_router.modeling_gemma4_unified_three_stream_router import set_defaults
        set_defaults(config)
        structural = config.router_schema_sha256
    if structural != values.get("router_schema_sha256"):
        raise ValueError("Structural schema hash differs from the exported contract")
    if architectures[0] == "Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM":
        from veomni.models.transformers.qwen3_5_three_stream_nll_selected_robust_decision_margin_router.modeling_qwen3_5_three_stream_nll_selected_robust_decision_margin_router import robust_decision_margin_objective_schema_sha256 as objective_hash
    else:
        from veomni.models.transformers.qwen3_large_robust_margin_router.objective import objective_sha256 as objective_hash
    if objective_hash(config) != values.get("router_robust_decision_margin_objective_sha256"):
        raise ValueError("Objective schema hash differs from the exported contract")
    if not all(isinstance(key, str) and key.startswith("router_head.") and isinstance(value, torch.Tensor)
               and value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for key, value in state.items()):
        raise ValueError("Export must contain only finite FP32 Router tensors")
    digest = tensor_state_sha256(state)
    if digest != payload.get("tensor_state_sha256"):
        raise ValueError("Router tensor-state checksum mismatch")
    head = _build_router_head(config).float()
    head.load_state_dict({key.removeprefix("router_head."): value for key, value in state.items()}, strict=True)
    if len(state) != values.get("router_expected_head_tensor_count") or sum(v.numel() for v in state.values()) != values.get("router_expected_head_parameter_count"):
        raise ValueError("Router tensor/parameter count differs")
    report = {
        "status": "passed", "scope": "inference_export_compatibility", "family": family,
        "checkpoint": str(checkpoint), "base_model_path": str(base),
        "export_sha256": sha256(checkpoint), "export_tensor_state_sha256": digest,
        "objective_sha256": values["router_robust_decision_margin_objective_sha256"],
        "training_completion_verified": False, "runtime_quality_qualified": False,
    }
    return payload, config, head.eval(), report


def load_hf(base, checkpoint, device="cuda:0"):
    import torch
    from transformers import AutoTokenizer, GenerationConfig

    payload, training_config, head, report = validate_export(checkpoint, base)
    family, values = report["family"], dict(payload["model_config"])
    values.update(router_action_threshold=0.0, use_cache=True)
    if family == "gemma4":
        from hf_gemma4.runtime import GemmaSelectedHybrid, audit_rope
        runtime = GemmaSelectedHybrid
        config_kw = {}
    elif family == "qwen3":
        from hf_qwen3.configuration_qwen3_selected_cache_router import Qwen3SelectedCacheRouterConfig
        from hf_qwen3.modeling_qwen3_selected_cache_router import Qwen3SelectedCacheRouterForCausalLM
        runtime = Qwen3SelectedCacheRouterForCausalLM
        config_kw = {"config": Qwen3SelectedCacheRouterConfig(**values)}
    else:
        from hf_qwen35.configuration_qwen3_5_selected_hybrid_cache_router import Qwen3_5SelectedHybridCacheRouterConfig
        from hf_qwen35.modeling_qwen3_5_selected_hybrid_cache_router import Qwen3_5SelectedHybridCacheRouterForCausalLM
        runtime = Qwen3_5SelectedHybridCacheRouterForCausalLM
        config_kw = {"config": Qwen3_5SelectedHybridCacheRouterConfig(**values)}
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model, loading = runtime.from_pretrained(str(base), dtype=dtype, device_map={"": device},
        attn_implementation="sdpa", local_files_only=True, output_loading_info=True, **config_kw)
    missing = [k for k in loading.get("missing_keys", []) if not k.startswith("router_head.")]
    unexpected = [k for k in loading.get("unexpected_keys", []) if not k.startswith(("mtp.", "model.visual."))]
    if missing or unexpected or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Backbone loading mismatch: missing={missing}, unexpected={unexpected}")
    if family == "gemma4":
        model.setup_router(head.to(device=device, dtype=torch.float32))
        audit_rope(model.trunk.rotary_emb)
    else:
        model.router_head.load_state_dict(head.state_dict(), strict=True)
        model.router_head.to(device=device, dtype=torch.float32).eval()
        # The portable validator above has checked the complete export contract.
        # Do not call the historical objective-specific loader for a new v4.2 run.
        model._router_checkpoint_loaded = True
        model._router_checkpoint_metadata = report
    model.generation_config = GenerationConfig.from_pretrained(str(base), local_files_only=True)
    model.generation_config.use_cache = True
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, padding_side="left")
    return model, tokenizer, report
