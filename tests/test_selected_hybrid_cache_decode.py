from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest
import torch
from torch import nn
import transformers.models.qwen3_5.modeling_qwen3_5 as official_modeling


ROOT = pathlib.Path(__file__).resolve().parents[1] / "src/hf_qwen35"


def load_local_modules():
    package_name = "qwen3_5_selected_hybrid_cache_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    loaded = {}
    for module_name in (
        "configuration_qwen3_5_selected_hybrid_cache_router",
        "modeling_qwen3_5_selected_hybrid_cache_router",
    ):
        qualified = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified, ROOT / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded


MODULES = load_local_modules()
CONFIGURATION = MODULES["configuration_qwen3_5_selected_hybrid_cache_router"]
MODELING = MODULES["modeling_qwen3_5_selected_hybrid_cache_router"]


class FixedScore(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))
        self.calls = 0

    def forward(self, history, embedding, local_hidden):
        del embedding, local_hidden
        self.calls += 1
        return self.values[: history.shape[0]]


class CapturingScore(FixedScore):
    def __init__(self, values):
        super().__init__(values)
        self.history = None
        self.embedding = None
        self.local_hidden = None

    def forward(self, history, embedding, local_hidden):
        self.history = history.detach().clone()
        self.embedding = embedding.detach().clone()
        self.local_hidden = local_hidden.detach().clone()
        return super().forward(history, embedding, local_hidden)


def force_cpu_fallbacks(monkeypatch) -> None:
    monkeypatch.setattr(official_modeling, "FusedRMSNormGated", None)
    monkeypatch.setattr(official_modeling, "chunk_gated_delta_rule", None)
    monkeypatch.setattr(official_modeling, "fused_recurrent_gated_delta_rule", None)
    monkeypatch.setattr(official_modeling, "causal_conv1d_fn", None)
    monkeypatch.setattr(official_modeling, "causal_conv1d_update", None)


def make_config(window_size=2):
    config = CONFIGURATION.Qwen3_5SelectedHybridCacheRouterConfig(
        vocab_size=127,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=True,
        eos_token_id=126,
        max_position_embeddings=64,
        router_hidden_size=16,
        router_intermediate_size=32,
        router_num_blocks=2,
        router_expected_head_parameter_count=7_137,
        router_expected_head_tensor_count=22,
        router_sink_size=1,
        router_window_size=window_size,
        router_action_threshold=0.0,
    )
    config._attn_implementation = "sdpa"
    return config


def make_pair(monkeypatch, seed=1, window_size=2):
    force_cpu_fallbacks(monkeypatch)
    torch.manual_seed(seed)
    config = make_config(window_size)
    baseline = official_modeling.Qwen3_5ForCausalLM(config).eval()
    router = MODELING.Qwen3_5SelectedHybridCacheRouterForCausalLM(config).eval()
    incompatible = router.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(name.startswith("router_head.") for name in incompatible.missing_keys)
    router._router_checkpoint_loaded = True
    router._router_checkpoint_metadata = {
        "global_step": 0,
        "training_history_nll_threshold": 0.0,
    }
    return baseline, router


def prefill(model, tokens, mask=None):
    if mask is None:
        mask = torch.ones_like(tokens)
    return model(input_ids=tokens, attention_mask=mask, use_cache=True)


def decode(model, token, cache, total_length, mask=None):
    if mask is None:
        mask = torch.ones(token.shape[0], total_length, dtype=torch.long)
    return model(
        input_ids=token,
        attention_mask=mask,
        past_key_values=cache,
        use_cache=True,
    )


def assert_hybrid_cache_matches_dynamic(selected, dynamic, length):
    for selected_layer, dynamic_layer in zip(selected.layers, dynamic.layers):
        if isinstance(selected_layer, MODELING.SelectedFullAttentionStaticLayer):
            torch.testing.assert_close(selected_layer.keys[:, :, :length], dynamic_layer.keys)
            torch.testing.assert_close(selected_layer.values[:, :, :length], dynamic_layer.values)
        else:
            torch.testing.assert_close(selected_layer.conv_states, dynamic_layer.conv_states)
            torch.testing.assert_close(
                selected_layer.recurrent_states, dynamic_layer.recurrent_states
            )






@torch.no_grad()
def test_full_prefill_and_forced_full_decode_match_official(monkeypatch):
    baseline, router = make_pair(monkeypatch, seed=11, window_size=2)
    prompt = torch.tensor([[4, 8, 15, 16, 23, 42]])
    baseline_prefill = prefill(baseline, prompt)
    router_prefill = prefill(router, prompt)
    torch.testing.assert_close(router_prefill.logits, baseline_prefill.logits)
    pointers = router_prefill.past_key_values.committed_data_ptrs()

    router.router_head = FixedScore([1.0])
    baseline_decode = decode(
        baseline, torch.tensor([[9]]), baseline_prefill.past_key_values, 7
    )
    router_decode = decode(
        router, torch.tensor([[9]]), router_prefill.past_key_values, 7
    )
    torch.testing.assert_close(router_decode.logits, baseline_decode.logits, atol=2e-5, rtol=2e-5)
    assert router._last_route_full.tolist() == [True]
    assert pointers == router_decode.past_key_values.committed_data_ptrs()
    assert_hybrid_cache_matches_dynamic(
        router_decode.past_key_values, baseline_decode.past_key_values, 7
    )


@torch.no_grad()
def test_full_equivalent_local_candidate_matches_official(monkeypatch):
    baseline, router = make_pair(monkeypatch, seed=12, window_size=2)
    prompt = torch.tensor([[3, 1]])
    baseline_prefill = prefill(baseline, prompt)
    router_prefill = prefill(router, prompt)
    router._active_mode = "local"
    baseline_decode = decode(
        baseline, torch.tensor([[4]]), baseline_prefill.past_key_values, 3
    )
    router_decode = decode(
        router, torch.tensor([[4]]), router_prefill.past_key_values, 3
    )
    torch.testing.assert_close(router_decode.logits, baseline_decode.logits, atol=2e-5, rtol=2e-5)
    assert router._last_route_full.tolist() == [False]
    assert_hybrid_cache_matches_dynamic(
        router_decode.past_key_values, baseline_decode.past_key_values, 3
    )


@torch.no_grad()
def test_candidate_staging_keeps_committed_cache_immutable(monkeypatch):
    _, router = make_pair(monkeypatch, seed=13, window_size=2)
    output = prefill(router, torch.tensor([[3, 1, 4, 1, 5]]))
    cache = output.past_key_values
    snapshots = []
    for layer in cache.layers:
        if isinstance(layer, MODELING.SelectedFullAttentionStaticLayer):
            snapshots.append((layer.keys[:, :, :5].clone(), layer.values[:, :, :5].clone()))
        else:
            snapshots.append((layer.conv_states.clone(), layer.recurrent_states.clone()))
    embedding = router.model.embed_tokens(torch.tensor([[9]]))
    _, rotary_positions = router._decode_position_ids(None, 1, 5, embedding.device)
    position_embeddings = router.model.rotary_emb(embedding, rotary_positions)
    cache.begin_stage()
    router._candidate(
        embedding,
        position_embeddings,
        cache,
        5,
        torch.tensor([True]),
        full=False,
    )
    for layer, expected in zip(cache.layers, snapshots):
        if isinstance(layer, MODELING.SelectedFullAttentionStaticLayer):
            assert torch.equal(layer.keys[:, :, :5], expected[0])
            assert torch.equal(layer.values[:, :, :5], expected[1])
        else:
            assert torch.equal(layer.conv_states, expected[0])
            assert torch.equal(layer.recurrent_states, expected[1])


@torch.no_grad()
def test_strict_zero_threshold_and_nonfinite_fail_safe(monkeypatch):
    _, router = make_pair(monkeypatch, seed=14, window_size=2)
    output = prefill(router, torch.tensor([[3, 1, 4, 1]]))
    router.router_head = FixedScore([0.0])
    decoded = decode(router, torch.tensor([[5]]), output.past_key_values, 5)
    assert router._last_route_full.tolist() == [False]

    router._finish_request()
    second = prefill(router, torch.tensor([[3, 1, 4, 1]]))
    router.router_head = FixedScore([float("nan")])
    decode(router, torch.tensor([[5]]), second.past_key_values, 5)
    assert router._last_route_full.tolist() == [True]








@torch.no_grad()
def test_mixed_full_rows_are_compacted_into_one_exact_subset(monkeypatch):
    baseline, router = make_pair(monkeypatch, seed=15, window_size=2)
    prompt = torch.tensor(
        [
            [0, 2, 7, 1, 8],
            [3, 1, 4, 1, 5],
            [0, 0, 9, 2, 6],
        ]
    )
    mask = torch.tensor(
        [
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ]
    )
    baseline_prefill = prefill(baseline, prompt, mask)
    router_prefill = prefill(router, prompt, mask)
    previous = router._previous_selected_hidden.clone()
    capture = CapturingScore([1.0, -1.0, 1.0])
    router.router_head = capture
    token = torch.tensor([[3], [5], [8]])
    decode_mask = torch.cat((mask, torch.ones(3, 1, dtype=torch.long)), dim=1)
    baseline_decode = decode(
        baseline, token, baseline_prefill.past_key_values, 6, decode_mask
    )
    router_decode = decode(
        router, token, router_prefill.past_key_values, 6, decode_mask
    )

    assert capture.calls == 1
    assert router._last_route_full.tolist() == [True, False, True]
    assert router._request_full_calls == 1
    assert router._request_full_subset_max == 2
    torch.testing.assert_close(capture.history, previous)
    torch.testing.assert_close(
        capture.embedding, router.model.embed_tokens(token)[:, 0]
    )
    torch.testing.assert_close(
        router_decode.logits[[0, 2]],
        baseline_decode.logits[[0, 2]],
        atol=2e-5,
        rtol=2e-5,
    )
    for selected_layer, dynamic_layer in zip(
        router_decode.past_key_values.layers,
        baseline_decode.past_key_values.layers,
    ):
        if isinstance(selected_layer, MODELING.SelectedFullAttentionStaticLayer):
            torch.testing.assert_close(
                selected_layer.keys[[0, 2], :, :6], dynamic_layer.keys[[0, 2]]
            )
            torch.testing.assert_close(
                selected_layer.values[[0, 2], :, :6], dynamic_layer.values[[0, 2]]
            )
        else:
            torch.testing.assert_close(
                selected_layer.conv_states[[0, 2]], dynamic_layer.conv_states[[0, 2]]
            )
            torch.testing.assert_close(
                selected_layer.recurrent_states[[0, 2]],
                dynamic_layer.recurrent_states[[0, 2]],
            )


@torch.no_grad()
def test_greedy_generate_uses_hybrid_cache(monkeypatch):
    _, router = make_pair(monkeypatch, seed=16, window_size=2)
    monkeypatch.setenv("QWEN35_SELECTED_CACHE_ROUTER_MODE", "local")
    output = router.generate(
        torch.tensor([[2, 7, 1]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        max_new_tokens=3,
        do_sample=False,
    )
    assert output.shape == (1, 6)
    summary = router.router_telemetry_summary()
    assert summary["routed_row_tokens"] == 2
    assert summary["full_row_tokens"] == 0
    assert summary["selected_hybrid_cache_invariants_passed"]
    assert summary["router_score_space"] == "signed_log1p_delta_nll"
    router._router_checkpoint_metadata["score_space"] = (
        "signed_log1p_robust_decision_margin"
    )
    assert (
        router.router_telemetry_summary()["router_score_space"]
        == "signed_log1p_robust_decision_margin"
    )
