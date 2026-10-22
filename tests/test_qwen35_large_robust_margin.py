"""Hybrid scaling contracts; runs only in the pinned Qwen3.5 TF5.9 image."""

import copy

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
import transformers.models.qwen3_5.modeling_qwen3_5 as official

from veomni.models.transformers.qwen3_5_large_robust_margin_router.modeling_qwen3_5_large_robust_margin_router import Qwen3_5LargeRobustMarginRouterForCausalLM as Router
from veomni.models.transformers.qwen3_5_three_stream_nll_selected_robust_decision_margin_router.modeling_qwen3_5_three_stream_nll_selected_robust_decision_margin_router import Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM as Legacy
from veomni.models.transformers.qwen3_5_three_stream_router import modeling_qwen3_5_three_stream_router as baseline
from veomni.models.transformers.qwen3_router.modeling_qwen3_router import _build_router_head


def tiny_config(layers=4):
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_conv_kernel_dim=4, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
        attention_dropout=0.0, use_cache=False)
    config.router_feature_schema = ["history_state", "current_embedding", "current_swa_hidden"]
    config.router_head_architecture = "three_tower_interaction_swiglu"
    config.router_hidden_size = 16
    config.router_intermediate_size = 32
    config.router_num_blocks = 2
    config.router_sink_size = 1
    config.router_window_size = 3
    config.router_history_nll_threshold = 0.0
    config.router_objective = "robust_decision_margin_huber"
    head = _build_router_head(config)
    config.router_expected_head_parameter_count = sum(p.numel() for p in head.parameters())
    config.router_expected_head_tensor_count = len(head.state_dict())
    config._attn_implementation = "eager"
    return config


@pytest.fixture(autouse=True)
def disable_cuda_only_kernels(monkeypatch):
    for name in ("FusedRMSNormGated", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        monkeypatch.setattr(official, name, None)


def test_scaled_64_layer_layout_and_legacy_global_unchanged():
    before = list(baseline._EXPECTED_LAYER_TYPES)
    with torch.device("meta"):
        model = Router(tiny_config(64))
    assert len(model.model.layers) == 64
    assert model.config.layer_types.count("full_attention") == 16
    assert model.config.layer_types.count("linear_attention") == 48
    assert baseline._EXPECTED_LAYER_TYPES == before


def test_matches_qualified_legacy_numerics_at_same_dimensions():
    config = tiny_config(24)
    old = Legacy(copy.deepcopy(config))
    new = Router(copy.deepcopy(config))
    new.load_state_dict(old.state_dict(), strict=True)
    ids = torch.randint(3, 64, (1, 12))
    args = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
    a, b = old(**args), new(**args)
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)
    for key in ("router_raw_signed_nll_gap", "router_robust_decision_margin",
                "router_transformed_robust_decision_margin_target", "router_full_target",
                "history_local_selected_mask", "router_robust_decision_score"):
        torch.testing.assert_close(a.training_probe_tensors[key], b.training_probe_tensors[key], rtol=0, atol=0)
    b.loss.backward()
    assert all(p.grad is None for n, p in new.named_parameters() if not n.startswith("router_head."))


def test_rejects_wrong_layer_order():
    config = tiny_config()
    config.layer_types[0] = "full_attention"
    with pytest.raises(ValueError, match="layout"):
        Router(config)
