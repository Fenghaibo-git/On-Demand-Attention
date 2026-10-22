import importlib.util
import os
import pathlib
import sys
import types
import unittest
from contextlib import contextmanager

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


ROOT = pathlib.Path(__file__).resolve().parents[1] / "src/hf_qwen3"


def load_local_modules():
    package_name = "qwen3_selected_cache_router_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    sys.modules[package_name] = package
    loaded = {}
    for module_name in (
        "configuration_qwen3_selected_cache_router",
        "modeling_qwen3_selected_cache_router",
    ):
        qualified = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified, ROOT / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded


MODULES = load_local_modules()
CONFIGURATION = MODULES["configuration_qwen3_selected_cache_router"]
MODELING = MODULES["modeling_qwen3_selected_cache_router"]








def make_config(window_size=16):
    config = CONFIGURATION.Qwen3SelectedCacheRouterConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        layer_types=["full_attention", "full_attention"],
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_sliding_window=False,
        sliding_window=None,
        eos_token_id=96,
        router_hidden_size=16,
        router_intermediate_size=32,
        router_num_blocks=2,
        router_expected_head_parameter_count=8_769,
        router_expected_head_tensor_count=22,
        router_sink_size=2,
        router_window_size=window_size,
        router_action_threshold=0.001,
    )
    config._attn_implementation = "sdpa"
    return config


def make_pair(seed=1, window_size=16):
    torch.manual_seed(seed)
    config = make_config(window_size)
    baseline = Qwen3ForCausalLM(config).eval()
    router = MODELING.Qwen3SelectedCacheRouterForCausalLM(config).eval()
    incompatible = router.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(name.startswith("router_head.") for name in incompatible.missing_keys)
    router._router_checkpoint_loaded = True
    router._router_checkpoint_metadata = {
        "global_step": 0,
        "training_history_nll_threshold": 0.001,
    }
    return baseline, router




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


class SelectedCacheDecodeTest(unittest.TestCase):
    def test_forced_local_mode_needs_no_checkpoint_or_router_call(self):
        _, router = make_pair(seed=10, window_size=2)
        router._router_checkpoint_loaded = False
        fixed = FixedScore([1.0])
        router.router_head = fixed
        previous = os.environ.get("QWEN3_SELECTED_CACHE_ROUTER_MODE")
        os.environ["QWEN3_SELECTED_CACHE_ROUTER_MODE"] = "local"
        try:
            output = prefill(router, torch.tensor([[3, 1, 4, 1, 5]]))
            decoded = decode(
                router, torch.tensor([[9]]), output.past_key_values, 6
            )
        finally:
            if previous is None:
                os.environ.pop("QWEN3_SELECTED_CACHE_ROUTER_MODE", None)
            else:
                os.environ["QWEN3_SELECTED_CACHE_ROUTER_MODE"] = previous

        self.assertEqual(fixed.calls, 0)
        self.assertEqual(router._last_route_full.tolist(), [False])
        self.assertEqual(decoded.past_key_values.get_seq_length(), 6)
        router._finish_request()
        summary = router.router_telemetry_summary()
        self.assertEqual(summary["router_mode"], "local")
        self.assertEqual(summary["full_row_tokens"], 0)
        self.assertTrue(summary["selected_cache_router_real_invariants_passed"])

    def test_prefill_uses_original_qwen_and_does_not_route(self):
        baseline, router = make_pair(seed=11)
        prompt = torch.tensor([[4, 8, 15, 16, 23]])
        baseline_output = prefill(baseline, prompt)
        router_output = prefill(router, prompt)
        torch.testing.assert_close(router_output.logits, baseline_output.logits)
        self.assertIsInstance(
            router_output.past_key_values, MODELING.SelectedMixedStaticCache
        )
        self.assertEqual(router_output.past_key_values.get_seq_length(), 5)
        self.assertIsNone(router._last_route_full)

    def test_threshold_is_strict_and_router_is_called_once(self):
        _, router = make_pair(seed=12, window_size=2)
        prompt = torch.tensor([[3, 1, 4, 1, 5]])
        output = prefill(router, prompt)
        fixed = FixedScore([0.001])
        router.router_head = fixed
        decoded = decode(router, torch.tensor([[9]]), output.past_key_values, 6)
        self.assertEqual(fixed.calls, 1)
        self.assertEqual(router._last_route_full.tolist(), [False])
        self.assertEqual(decoded.past_key_values.get_seq_length(), 6)

    def test_full_candidate_matches_plain_qwen_and_keeps_static_addresses(self):
        baseline, router = make_pair(seed=13, window_size=2)
        prompt = torch.tensor([[3, 1, 4, 1, 5]])
        baseline_prefill = prefill(baseline, prompt)
        router_prefill = prefill(router, prompt)
        pointers = router_prefill.past_key_values.committed_data_ptrs()
        router.router_head = FixedScore([0.002])
        baseline_decode = decode(
            baseline, torch.tensor([[9]]), baseline_prefill.past_key_values, 6
        )
        router_decode = decode(
            router, torch.tensor([[9]]), router_prefill.past_key_values, 6
        )
        torch.testing.assert_close(
            router_decode.logits, baseline_decode.logits, atol=2e-5, rtol=2e-5
        )
        self.assertEqual(router._last_route_full.tolist(), [True])
        self.assertEqual(
            pointers, router_decode.past_key_values.committed_data_ptrs()
        )
        for baseline_layer, selected_layer in zip(
            baseline_decode.past_key_values.layers,
            router_decode.past_key_values.layers,
        ):
            torch.testing.assert_close(
                selected_layer.keys[:, :, :6, :],
                baseline_layer.keys,
                atol=2e-5,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                selected_layer.values[:, :, :6, :],
                baseline_layer.values,
                atol=2e-5,
                rtol=2e-5,
            )
        router._finish_request()
        summary = router.router_telemetry_summary()
        self.assertEqual(summary["full_row_tokens"], 1)
        self.assertEqual(summary["full_rerun_row_tokens"], 1)
        self.assertEqual(summary["full_equivalent_reuse_row_tokens"], 0)
        self.assertEqual(summary["full_forward_calls"], 1)

    def test_full_equivalent_full_route_reuses_local_candidate(self):
        # sink=2 + window=2: the decoded sequence length of four is the
        # inclusive Full-equivalence boundary.
        baseline, router = make_pair(seed=18, window_size=2)
        prompt = torch.tensor([[3, 1, 4]])
        baseline_prefill = prefill(baseline, prompt)
        router_prefill = prefill(router, prompt)
        router.router_head = FixedScore([0.002])

        baseline_decode = decode(
            baseline, torch.tensor([[1]]), baseline_prefill.past_key_values, 4
        )
        router_decode = decode(
            router, torch.tensor([[1]]), router_prefill.past_key_values, 4
        )
        torch.testing.assert_close(
            router_decode.logits, baseline_decode.logits, atol=2e-5, rtol=2e-5
        )
        self.assertEqual(router._last_route_full.tolist(), [True])
        for baseline_layer, selected_layer in zip(
            baseline_decode.past_key_values.layers,
            router_decode.past_key_values.layers,
        ):
            torch.testing.assert_close(
                selected_layer.keys[:, :, :4, :],
                baseline_layer.keys,
                atol=2e-5,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                selected_layer.values[:, :, :4, :],
                baseline_layer.values,
                atol=2e-5,
                rtol=2e-5,
            )

        router._finish_request()
        summary = router.router_telemetry_summary()
        self.assertEqual(summary["full_row_tokens"], 1)
        self.assertEqual(summary["full_rerun_row_tokens"], 0)
        self.assertEqual(summary["full_equivalent_reuse_row_tokens"], 1)
        self.assertEqual(summary["full_forward_calls"], 0)
        self.assertEqual(summary["full_candidate_active_row_tokens"], 1)
        self.assertEqual(summary["selected_full_kv_row_tokens"], 1)
        self.assertTrue(summary["selected_cache_router_real_invariants_passed"])

    def test_mixed_batch_uses_selected_hidden_and_kv(self):
        _, router = make_pair(seed=14, window_size=2)
        prompt = torch.tensor([[0, 2, 7, 1], [3, 1, 4, 1]])
        mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
        prefilled = prefill(router, prompt, mask)
        previous = router._previous_selected_hidden.detach().clone()
        capturing = CapturingScore([-1.0, 1.0])
        router.router_head = capturing
        decoded = decode(
            router,
            torch.tensor([[8], [5]]),
            prefilled.past_key_values,
            5,
            torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        )
        self.assertEqual(capturing.calls, 1)
        self.assertEqual(router._last_route_full.tolist(), [False, True])
        torch.testing.assert_close(capturing.history, previous)
        torch.testing.assert_close(
            capturing.embedding,
            router.model.embed_tokens(torch.tensor([[8], [5]]))[:, 0, :],
        )
        self.assertEqual(decoded.past_key_values.get_seq_length(), 5)
        router._finish_request()
        summary = router.router_telemetry_summary()
        self.assertEqual(summary["routed_row_tokens"], 2)
        self.assertEqual(summary["full_row_tokens"], 1)
        self.assertEqual(summary["selected_full_kv_row_tokens"], 1)
        self.assertEqual(summary["selected_local_kv_row_tokens"], 1)
        self.assertTrue(summary["selected_cache_router_real_invariants_passed"])

    def test_nonfinite_score_fails_safe_to_full(self):
        _, router = make_pair(seed=15)
        prefilled = prefill(router, torch.tensor([[2, 7, 1]]))
        router.router_head = FixedScore([float("nan")])
        decode(router, torch.tensor([[8]]), prefilled.past_key_values, 4)
        self.assertEqual(router._last_route_full.tolist(), [True])


    def test_greedy_generate_uses_custom_cache_and_finishes_aggregate(self):
        _, router = make_pair(seed=16)
        fixed = FixedScore([-1.0])
        router.router_head = fixed
        output = router.generate(
            torch.tensor([[2, 7, 1]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            max_new_tokens=3,
            do_sample=False,
        )
        self.assertEqual(output.shape[1], 6)
        self.assertEqual(fixed.calls, 2)
        summary = router.router_telemetry_summary()
        self.assertEqual(summary["routed_row_tokens"], 2)
        self.assertEqual(summary["full_row_tokens"], 0)
        self.assertEqual(summary["request_count"], 1)


if __name__ == "__main__":
    unittest.main()
