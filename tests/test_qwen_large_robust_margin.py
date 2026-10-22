"""Objective and Qwen3 contracts; runs in the pinned Qwen3 TF4.51 image."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3 import modeling_qwen3 as official
from veomni.models.transformers.qwen3_router import modeling_qwen3_router as base
from veomni.models.transformers.qwen3_three_stream_router import modeling_qwen3_three_stream_router as streams

from veomni.models.transformers.qwen3_large_robust_margin_router.modeling_qwen3_large_robust_margin_router import Qwen3LargeRobustMarginRouterForCausalLM as Router
from veomni.models.transformers.qwen3_large_robust_margin_router.objective import ROBUST_DEFAULTS, set_robust_defaults
from veomni.models.transformers.qwen3_router.modeling_qwen3_router import _build_router_head


@pytest.fixture(autouse=True)
def cpu_only_kernels(monkeypatch):
    for module in (base, streams):
        for name in ("Qwen3RMSNorm", "Qwen3MLP"):
            monkeypatch.setattr(module, name, getattr(official, name))
    monkeypatch.setattr(base, "apply_rotary_pos_emb", official.apply_rotary_pos_emb)
    monkeypatch.setattr(streams, "_get_three_stream_compiled_flex", lambda stream: streams.raw_flex_attention)


def objective_model():
    model = object.__new__(Router)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(**ROBUST_DEFAULTS,
        router_one_step_sign_aux_temperature=0.1,
        router_one_step_sign_near_zero_threshold=0.0001)
    model._last_robust_margin_metrics = {}
    return model


def tiny_config():
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, use_cache=False,
        use_sliding_window=False, sliding_window=None, attention_dropout=0.0,
        tie_word_embeddings=False)
    config.router_feature_schema = ["history_state", "current_embedding", "current_swa_hidden"]
    config.router_head_architecture = "three_tower_interaction_swiglu"
    config.router_hidden_size = 16
    config.router_intermediate_size = 32
    config.router_num_blocks = 2
    config.router_sink_size = 1
    config.router_window_size = 3
    head = _build_router_head(config)
    config.router_expected_head_parameter_count = sum(p.numel() for p in head.parameters())
    config.router_expected_head_tensor_count = len(head.state_dict())
    config._attn_implementation = "eager"
    return config


def test_target_and_loss_match_plain_huber_and_bounded_gradient():
    model = objective_model()
    gap = torch.tensor([-100.0, -0.001, 0.0, 0.001, 0.002, 100.0])
    mask = torch.tensor([True, True, False, True, True, True])
    score = torch.tensor([50., .2, 99., 0., -.2, -50.], requires_grad=True)
    target = model._transform_target(gap)
    d = gap.float() - 0.001
    torch.testing.assert_close(target, d.sign() * torch.log1p(d.abs()))
    loss, _ = model._compute_one_step_loss(score, gap, target, mask)
    expected = F.huber_loss(score[mask], target[mask], delta=1.)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert score.grad[2] == 0
    assert score.grad.abs().max() <= 1 / mask.sum() + 1e-7


@pytest.mark.parametrize("name,value", [
    ("router_history_nll_threshold", .001),
    ("router_robust_decision_margin_lambda", .002),
    ("router_one_step_positive_weight", 2.),
    ("router_one_step_sign_aux_weight", .1),
    ("router_robust_decision_margin_lambda", float("nan")),
    ("router_robust_decision_margin_lambda", float("inf")),
])
def test_rejects_semantic_drift(name, value):
    config = SimpleNamespace(**{**ROBUST_DEFAULTS, name: value})
    with pytest.raises(ValueError):
        set_robust_defaults(config)


def test_float32_transport_is_accepted():
    config = SimpleNamespace(**{**ROBUST_DEFAULTS,
        "router_robust_decision_margin_lambda": float(torch.tensor(.001))})
    set_robust_defaults(config)


def test_tiny_qwen3_forward_only_trains_head_and_preserves_history_tie():
    model = Router(tiny_config())
    ids = torch.randint(3, 64, (1, 16))
    outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
    assert torch.isfinite(outputs.loss)
    probes = outputs.training_probe_tensors
    valid = probes["valid_token_mask"].bool()
    gap = probes["router_raw_signed_nll_gap"]
    assert torch.equal(probes["history_local_selected_mask"], gap.lt(0) & valid)
    assert torch.equal(probes["router_full_target"], gap.gt(.001) & valid)
    assert "router_raw_target" not in probes
    expected = model._transform_target(gap)
    torch.testing.assert_close(probes["router_transformed_robust_decision_margin_target"][valid], expected[valid])
    outputs.loss.backward()
    assert any(p.grad is not None for p in model.router_head.parameters())
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("router_head."))


def test_zero_gap_uses_full_history_but_negative_decision_margin():
    model = objective_model()
    gap = torch.tensor([[0., -.1, .1]])
    full = torch.ones((1, 3, 2))
    local = -full
    mask = torch.ones_like(gap, dtype=torch.bool)
    history, chosen = model._select_threshold_history_source(full, local, gap, mask)
    assert chosen.tolist() == [[False, True, False]]
    assert torch.equal(history[:, 0], full[:, 0])
    assert model._transform_target(gap)[0, 0] < 0
