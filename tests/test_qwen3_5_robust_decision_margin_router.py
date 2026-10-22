"""Contracts for the Qwen3.5 robust decision-margin Huber ablation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
import transformers.models.qwen3_5.modeling_qwen3_5 as official_modeling
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from veomni.models.registry import ModelRegistry
from veomni.models.transformers.qwen3_5_three_stream_nll_selected_robust_decision_margin_router.modeling_qwen3_5_three_stream_nll_selected_robust_decision_margin_router import (
    Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM,
    robust_decision_margin_objective_schema_payload,
    robust_decision_margin_objective_schema_sha256,
)
from veomni.models.transformers.qwen3_5_three_stream_router.modeling_qwen3_5_three_stream_router import (
    _EXPECTED_LAYER_TYPES,
)
from veomni.models.transformers.qwen3_router.modeling_qwen3_router import (
    _build_router_head,
)


OBJECTIVE_SHA256 = "49e05162991dc1b6ac3c62ace1fd4574518ad4e5a1949a41e69e696ff2ff43ef"


def _objective_only_model():
    model = object.__new__(
        Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM
    )
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        router_robust_decision_margin_lambda=0.001,
        router_one_step_target="signed_nll",
        router_one_step_harm_threshold=0.0,
        router_one_step_positive_weight=1.0,
        router_one_step_huber_delta=1.0,
        router_one_step_sign_aux_weight=0.0,
        router_one_step_sign_aux_temperature=0.1,
        router_one_step_sign_near_zero_threshold=0.0001,
        router_history_nll_threshold=0.0,
    )
    model._last_robust_margin_metrics = {}
    return model


def _objective_config() -> SimpleNamespace:
    return SimpleNamespace(
        router_robust_decision_margin_objective_schema_version=1,
        router_schema_sha256=(
            "d17934038878169f856ae60dbf7503961358f2260a752d92c8321dcabf8f9789"
        ),
        router_robust_decision_margin_lambda=0.001,
        router_robust_decision_margin_objective=(
            "eligible_token_mean_huber_on_signed_log1p_shifted_margin"
        ),
        router_robust_decision_margin_huber_delta=1.0,
        router_robust_decision_margin_loss_weighting="eligible_token_equal",
        router_robust_decision_margin_output_space=(
            "signed_log1p_robust_decision_margin"
        ),
        router_robust_decision_margin_runtime_score_threshold=0.0,
        router_history_nll_threshold=0.0,
        router_robust_decision_margin_interpretation=(
            "huber_m_estimate_not_conditional_mean"
        ),
    )


def _tiny_model_config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=list(_EXPECTED_LAYER_TYPES),
        attention_dropout=0.0,
        use_cache=False,
    )
    config.architectures = [
        "Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM"
    ]
    config.router_sink_size = 1
    config.router_window_size = 16
    config.router_history_nll_threshold = 0.0
    config.router_feature_schema = [
        "history_state",
        "current_embedding",
        "current_swa_hidden",
    ]
    config.router_head_architecture = "three_tower_interaction_swiglu"
    config.router_hidden_size = 16
    config.router_num_blocks = 1
    config.router_intermediate_size = 32
    probe = _build_router_head(config)
    config.router_expected_head_parameter_count = sum(
        parameter.numel() for parameter in probe.parameters()
    )
    config.router_expected_head_tensor_count = len(probe.state_dict())
    config.router_objective = "robust_decision_margin_huber"
    config._attn_implementation = "eager"
    return config


def test_registry_discovers_independent_robust_margin_objective() -> None:
    assert ModelRegistry.get_model_cls_from_model_arch(
        "Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM"
    ) is Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM


def test_tiny_model_constructs_with_truthful_public_objective(monkeypatch) -> None:
    monkeypatch.setattr(official_modeling, "FusedRMSNormGated", None)
    monkeypatch.setattr(official_modeling, "chunk_gated_delta_rule", None)
    monkeypatch.setattr(official_modeling, "fused_recurrent_gated_delta_rule", None)
    monkeypatch.setattr(official_modeling, "causal_conv1d_fn", None)
    monkeypatch.setattr(official_modeling, "causal_conv1d_update", None)
    model = Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM(
        _tiny_model_config()
    )
    assert model.config.router_objective == "robust_decision_margin_huber"
    assert (
        model.config.router_robust_decision_margin_objective_sha256
        == robust_decision_margin_objective_schema_sha256(model.config)
    )
    assert all(parameter.requires_grad for parameter in model.router_head.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("router_head.")
    )


def test_target_is_exact_shifted_signed_log1p_margin() -> None:
    model = _objective_only_model()
    gap = torch.tensor([[-0.25, 0.0, 0.001, 0.002, 0.5]], dtype=torch.float32)
    margin = gap - 0.001
    expected = margin.sign() * torch.log1p(margin.abs())
    torch.testing.assert_close(model._decision_margin(gap), margin)
    torch.testing.assert_close(model._transform_target(gap), expected)
    torch.testing.assert_close(model._inverse_target(expected), margin)
    assert expected[0, 2].item() == 0.0


def test_loss_is_equal_token_mean_huber_without_margin_weighting() -> None:
    model = _objective_only_model()
    # The last target is far into Huber's linear regime.  Its gradient is
    # bounded at one token-unit rather than being multiplied by |d|.
    gap = torch.tensor([[0.101, math.expm1(3.0) + 0.001]], dtype=torch.float32)
    score = torch.zeros_like(gap, requires_grad=True)
    target = model._transform_target(gap)
    mask = torch.ones_like(gap, dtype=torch.bool)
    loss, metrics = model._compute_one_step_loss(score, gap, target, mask)
    expected = torch.nn.functional.huber_loss(
        score, target, reduction="mean", delta=1.0
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["huber_loss"], expected.detach())
    loss.backward()
    expected_gradient = torch.tensor(
        [[-target[0, 0].item() / 2.0, -0.5]], dtype=torch.float32
    )
    torch.testing.assert_close(score.grad, expected_gradient, atol=1.0e-7, rtol=1.0e-6)


def test_strict_decision_boundary_is_d_zero() -> None:
    model = _objective_only_model()
    gap = torch.tensor([[0.0009, 0.0010, 0.0011]])
    score = torch.zeros_like(gap, requires_grad=True)
    target = model._transform_target(gap)
    mask = torch.ones_like(gap, dtype=torch.bool)
    _, metrics = model._compute_one_step_loss(score, gap, target, mask)
    assert metrics["sign_target_rate"].item() == pytest.approx(1.0 / 3.0)
    assert metrics["sign_predicted_positive_rate"].item() == 0.0


def test_three_stream_history_input_selector_remains_control_threshold_zero() -> None:
    model = _objective_only_model()
    trunk = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    local = trunk + 100.0
    gap = torch.tensor([[-0.1, 0.0, 0.0005, 0.002]])
    mask = torch.ones_like(gap, dtype=torch.bool)
    source, selected_local = model._select_history(trunk, local, gap, mask)
    assert selected_local.tolist() == [[True, False, False, False]]
    torch.testing.assert_close(source[:, :1], local[:, :1])
    torch.testing.assert_close(source[:, 1:], trunk[:, 1:])


def test_objective_schema_is_honest_about_robust_estimand() -> None:
    config = _objective_config()
    payload = robust_decision_margin_objective_schema_payload(config)
    assert payload["decision_boundary_shift"] == 0.001
    assert payload["input_history_nll_threshold"] == 0.0
    assert payload["input_contract_relative_to_control"] == "unchanged"
    assert payload["magnitude_weight"] is False
    assert payload["is_strict_conditional_expectation_of_decision_margin"] is False
    assert robust_decision_margin_objective_schema_sha256(config) == OBJECTIVE_SHA256
