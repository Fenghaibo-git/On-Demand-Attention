# Modified for the portable Router v4.2 release; see NOTICE.
"""Causal selected-hybrid-cache deployment for the Qwen3.5 Router.

Full prefill initializes all six attention K/V histories and all eighteen
DeltaNet convolution/recurrent states.  Each routed decode token first stages
a Local candidate from the immutable selected cache.  Only rows selected by
the Router are recomputed as Full.  The selected hidden, logits, attention K/V,
DeltaNet convolution state, and DeltaNet recurrent state are committed as one
transaction.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, CacheLayerMixin, LinearAttentionCacheLayerMixin
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .configuration_qwen3_5_selected_hybrid_cache_router import (
    Qwen3_5SelectedHybridCacheRouterConfig,
)


_FEATURE_SCHEMA = ["history_state", "current_embedding", "current_swa_hidden"]
_EXPECTED_FORMAL_LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
] * 6
_SUMMARY_FORMAT = "qwen3.5-selected-hybrid-cache-router-aggregate-v1"
_CONTROL_TRAINING_ARCHITECTURE = "Qwen3_5ThreeStreamNllSelectedRouterForCausalLM"
_ROBUST_MARGIN_TRAINING_ARCHITECTURE = (
    "Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM"
)
_ROBUST_MARGIN_OBJECTIVE_SHA256 = (
    "49e05162991dc1b6ac3c62ace1fd4574518ad4e5a1949a41e69e696ff2ff43ef"
)
_HISTORY_ALIGNED_ROBUST_MARGIN_TRAINING_ARCHITECTURE = (
    "Qwen3_5ThreeStreamNllSelectedHistoryAlignedRobustDecisionMarginRouterForCausalLM"
)
_HISTORY_ALIGNED_ROBUST_MARGIN_OBJECTIVE_SHA256 = (
    "13eaee02590bc4d3da3bc44612ba0faf6a0bd0496b489d12fd4082f76b7879dd"
)
_FOCAL_TRAINING_ARCHITECTURE = (
    "Qwen3_5ThreeStreamNllSelectedFocalRouterForCausalLM"
)
_FOCAL_OBJECTIVE_SHA256 = (
    "eaa2c68bfa23c5503404656bc46bd468ff6e094bdca3258cbc8e1fc718071226"
)


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value.strip()


def _finite_env_float(name: str, default: float) -> float:
    result = float(_env(name, str(default)))
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}.")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        header = {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _contiguous_runs(mask: torch.BoolTensor) -> list[tuple[int, int]]:
    values = mask.detach().to(device="cpu", dtype=torch.bool).tolist()
    runs: list[tuple[int, int]] = []
    start = None
    for index, enabled in enumerate(values + [False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            runs.append((start, index))
            start = None
    return runs


class SelectedFullAttentionStaticLayer(CacheLayerMixin):
    """Preallocated selected K/V plus separate uncommitted candidate buffers."""

    is_compileable = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = int(max_cache_len)
        self.max_batch_size = 0
        self.cumulative_length = 0
        self.device = torch.device("cpu")
        self.staged_keys: torch.Tensor | None = None
        self.staged_values: torch.Tensor | None = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("Selected attention cache expects aligned [B,H,T,D] K/V.")
        self.max_batch_size = int(key_states.shape[0])
        self.device = key_states.device
        shape = (
            self.max_batch_size,
            int(key_states.shape[1]),
            self.max_cache_len,
            int(key_states.shape[3]),
        )
        self.keys = torch.empty(shape, dtype=key_states.dtype, device=key_states.device)
        self.values = torch.empty(shape, dtype=value_states.dtype, device=value_states.device)
        staged_shape = (*shape[:2], 1, shape[3])
        self.staged_keys = torch.empty(staged_shape, dtype=key_states.dtype, device=key_states.device)
        self.staged_values = torch.empty(staged_shape, dtype=value_states.dtype, device=value_states.device)
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        del args, kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if int(key_states.shape[0]) != self.max_batch_size:
            raise ValueError("Selected attention-cache batch size cannot change.")
        length = int(key_states.shape[-2])
        end = self.cumulative_length + length
        if end > self.max_cache_len:
            raise ValueError(f"Selected cache capacity exceeded: {end} > {self.max_cache_len}.")
        self.keys[:, :, self.cumulative_length : end].copy_(key_states)
        self.values[:, :, self.cumulative_length : end].copy_(value_states)
        self.cumulative_length = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def begin_stage(self) -> None:
        if not self.is_initialized:
            raise RuntimeError("Full prefill must initialize every attention cache layer.")
        self.staged_keys.zero_()
        self.staged_values.zero_()

    def stage_rows(
        self,
        row_start: int,
        row_end: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        if key_states.shape[-2] != 1 or value_states.shape != key_states.shape:
            raise ValueError("Decode staging requires aligned one-token K/V.")
        if key_states.shape[0] != row_end - row_start:
            raise ValueError("Staged K/V rows do not match the selected slice.")
        self.staged_keys[row_start:row_end].copy_(key_states)
        self.staged_values[row_start:row_end].copy_(value_states)

    def stage_indices(
        self,
        row_indices: torch.LongTensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        if key_states.shape[-2] != 1 or value_states.shape != key_states.shape:
            raise ValueError("Decode staging requires aligned one-token K/V.")
        if key_states.shape[0] != row_indices.numel():
            raise ValueError("Staged K/V rows do not match compacted row indices.")
        self.staged_keys.index_copy_(0, row_indices, key_states)
        self.staged_values.index_copy_(0, row_indices, value_states)

    def commit_one(self) -> None:
        if self.cumulative_length >= self.max_cache_len:
            raise ValueError("Selected attention cache capacity exceeded during commit.")
        position = self.cumulative_length
        self.keys[:, :, position : position + 1].copy_(self.staged_keys)
        self.values[:, :, position : position + 1].copy_(self.staged_values)
        self.cumulative_length += 1

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cumulative_length + int(query_length), 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


class SelectedDeltaNetStaticLayer(LinearAttentionCacheLayerMixin):
    """Selected DeltaNet state with a separate transactional staging area."""

    is_compileable = False

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.staged_conv_states: torch.Tensor | None = None
        self.staged_recurrent_states: torch.Tensor | None = None

    def lazy_initialization(
        self,
        conv_states: torch.Tensor | None = None,
        recurrent_states: torch.Tensor | None = None,
    ) -> None:
        if conv_states is not None:
            self.dtype, self.device = conv_states.dtype, conv_states.device
            self.conv_states = torch.empty_like(conv_states)
            self.conv_states.copy_(conv_states)
            self.is_conv_states_initialized = True
        if recurrent_states is not None:
            self.dtype, self.device = recurrent_states.dtype, recurrent_states.device
            self.recurrent_states = torch.empty_like(recurrent_states)
            self.recurrent_states.copy_(recurrent_states)
            self.is_recurrent_states_initialized = True

    def update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        if not self.is_conv_states_initialized:
            self.lazy_initialization(conv_states=conv_states)
        else:
            self.conv_states.copy_(conv_states)
        self.has_previous_state = True
        return self.conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        if not self.is_recurrent_states_initialized:
            self.lazy_initialization(recurrent_states=recurrent_states)
        else:
            self.recurrent_states.copy_(recurrent_states)
        return self.recurrent_states

    def begin_stage(self) -> None:
        if not (
            self.has_previous_state
            and self.is_conv_states_initialized
            and self.is_recurrent_states_initialized
        ):
            raise RuntimeError("Full prefill must initialize every DeltaNet state.")
        if (
            self.staged_conv_states is None
            or self.staged_conv_states.shape != self.conv_states.shape
        ):
            self.staged_conv_states = torch.empty_like(self.conv_states)
        if (
            self.staged_recurrent_states is None
            or self.staged_recurrent_states.shape != self.recurrent_states.shape
        ):
            self.staged_recurrent_states = torch.empty_like(self.recurrent_states)
        self.staged_conv_states.copy_(self.conv_states)
        self.staged_recurrent_states.copy_(self.recurrent_states)

    def stage_rows(
        self,
        row_start: int,
        row_end: int,
        conv_states: torch.Tensor,
        recurrent_states: torch.Tensor,
    ) -> None:
        if conv_states.shape[0] != row_end - row_start:
            raise ValueError("Staged DeltaNet rows do not match the selected slice.")
        self.staged_conv_states[row_start:row_end].copy_(conv_states)
        self.staged_recurrent_states[row_start:row_end].copy_(recurrent_states)

    def stage_indices(
        self,
        row_indices: torch.LongTensor,
        conv_states: torch.Tensor,
        recurrent_states: torch.Tensor,
    ) -> None:
        if conv_states.shape[0] != row_indices.numel():
            raise ValueError("Staged DeltaNet rows do not match compacted row indices.")
        self.staged_conv_states.index_copy_(0, row_indices, conv_states)
        self.staged_recurrent_states.index_copy_(0, row_indices, recurrent_states)

    def commit_one(self) -> None:
        self.conv_states.copy_(self.staged_conv_states)
        self.recurrent_states.copy_(self.staged_recurrent_states)
        self.has_previous_state = True


class SelectedHybridStaticCache(Cache):
    """Full recoverable attention history plus selected DeltaNet state."""

    def __init__(self, config: Qwen3_5SelectedHybridCacheRouterConfig, max_cache_len: int):
        layers = []
        for layer_type in config.layer_types:
            if layer_type == "full_attention":
                layers.append(SelectedFullAttentionStaticLayer(max_cache_len))
            elif layer_type == "linear_attention":
                layers.append(SelectedDeltaNetStaticLayer())
            else:
                raise ValueError(f"Unsupported Qwen3.5 layer type {layer_type!r}.")
        super().__init__(layers=layers)
        self.capacity = int(max_cache_len)

    def assert_ready_and_aligned(self) -> int:
        attention_lengths = [
            layer.get_seq_length()
            for layer in self.layers
            if isinstance(layer, SelectedFullAttentionStaticLayer)
        ]
        if not attention_lengths or len(set(attention_lengths)) != 1:
            raise AssertionError(f"Selected attention-cache lengths diverged: {attention_lengths}.")
        for layer in self.layers:
            if isinstance(layer, SelectedDeltaNetStaticLayer) and not (
                layer.has_previous_state
                and layer.is_conv_states_initialized
                and layer.is_recurrent_states_initialized
            ):
                raise AssertionError("A selected DeltaNet layer was not initialized by prefill.")
        return attention_lengths[0]

    def begin_stage(self) -> None:
        self.assert_ready_and_aligned()
        for layer in self.layers:
            layer.begin_stage()

    def commit_one(self) -> None:
        self.assert_ready_and_aligned()
        for layer in self.layers:
            layer.commit_one()
        self.assert_ready_and_aligned()

    def committed_data_ptrs(self) -> tuple[tuple[int, ...], ...]:
        result = []
        for layer in self.layers:
            if isinstance(layer, SelectedFullAttentionStaticLayer):
                result.append((layer.keys.data_ptr(), layer.values.data_ptr()))
            else:
                result.append(
                    (layer.conv_states.data_ptr(), layer.recurrent_states.data_ptr())
                )
        return tuple(result)


class _DeltaCandidateCache:
    """A one-layer mutable clone used to stage one DeltaNet candidate."""

    def __init__(
        self,
        layer_idx: int,
        source: SelectedDeltaNetStaticLayer,
        row_indices: torch.LongTensor,
    ):
        self.layer_idx = int(layer_idx)
        layer = SelectedDeltaNetStaticLayer()
        layer.lazy_initialization(
            conv_states=source.conv_states.index_select(0, row_indices).clone()
        )
        layer.lazy_initialization(
            recurrent_states=source.recurrent_states.index_select(0, row_indices).clone()
        )
        layer.has_previous_state = True
        self.layers = [None] * (self.layer_idx + 1)
        self.layers[self.layer_idx] = layer

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        return layer_idx == self.layer_idx and self.layers[self.layer_idx].has_previous_state

    def update_conv_state(self, value: torch.Tensor, layer_idx: int, **kwargs):
        if layer_idx != self.layer_idx:
            raise ValueError("Candidate cache received the wrong DeltaNet layer index.")
        return self.layers[layer_idx].update_conv_state(value, **kwargs)

    def update_recurrent_state(self, value: torch.Tensor, layer_idx: int, **kwargs):
        if layer_idx != self.layer_idx:
            raise ValueError("Candidate cache received the wrong DeltaNet layer index.")
        return self.layers[layer_idx].update_recurrent_state(value, **kwargs)


class Qwen3RouterSwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float):
        super().__init__()
        self.norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.gate_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_projection = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states)
        return hidden_states + self.down_projection(
            F.silu(self.gate_projection(normalized)) * self.up_projection(normalized)
        )


class Qwen3_5ThreeTowerInteractionRouterHead(nn.Module):
    """Exact A0 28.325889M-parameter Router used by training."""

    def __init__(self, config: Qwen3_5SelectedHybridCacheRouterConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        router_size = int(config.router_hidden_size)
        eps = float(config.rms_norm_eps)
        self.history_norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.embedding_norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.swa_norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.history_projection = nn.Linear(hidden_size, router_size, bias=True)
        self.embedding_projection = nn.Linear(hidden_size, router_size, bias=True)
        self.swa_projection = nn.Linear(hidden_size, router_size, bias=True)
        self.fusion_projection = nn.Linear(9 * router_size, router_size, bias=True)
        self.blocks = nn.ModuleList(
            Qwen3RouterSwiGLUBlock(router_size, int(config.router_intermediate_size), eps)
            for _ in range(int(config.router_num_blocks))
        )
        self.output_norm = Qwen3RMSNorm(router_size, eps=eps)
        self.output_projection = nn.Linear(router_size, 1, bias=True)

    def forward(
        self,
        history_state: torch.Tensor,
        current_embedding: torch.Tensor,
        current_swa_hidden: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=history_state.device.type, enabled=False):
            history = F.silu(self.history_projection(self.history_norm(history_state.float())))
            embedding = F.silu(
                self.embedding_projection(self.embedding_norm(current_embedding.float()))
            )
            swa = F.silu(self.swa_projection(self.swa_norm(current_swa_hidden.float())))
            groups = [history, embedding, swa]
            for left, right in ((history, embedding), (history, swa), (embedding, swa)):
                groups.extend((left * right, torch.abs(left - right)))
            fused = F.silu(self.fusion_projection(torch.cat(groups, dim=-1)))
            hidden_states = (history + embedding + swa + fused) * 0.5
            for block in self.blocks:
                hidden_states = block(hidden_states)
            return self.output_projection(self.output_norm(hidden_states)).squeeze(-1)


def _validate_config(config: Qwen3_5SelectedHybridCacheRouterConfig) -> None:
    if list(config.router_feature_schema) != _FEATURE_SCHEMA:
        raise ValueError(f"Runtime requires exact feature schema {_FEATURE_SCHEMA}.")
    if config.router_head_architecture != "three_tower_interaction_swiglu":
        raise ValueError("Selected-hybrid-cache runtime accepts only the A0 head.")
    if int(config.router_sink_size) < 0 or int(config.router_window_size) < 1:
        raise ValueError("Invalid sink/window configuration.")
    if not math.isfinite(float(config.router_action_threshold)):
        raise ValueError("Router action threshold must be finite.")
    if any(value not in {"linear_attention", "full_attention"} for value in config.layer_types):
        raise ValueError("Qwen3.5 selected cache accepts only linear/full attention layers.")
    if int(config.num_hidden_layers) == 24 and list(config.layer_types) != _EXPECTED_FORMAL_LAYER_TYPES:
        raise ValueError("Formal Qwen3.5-2B layer layout drifted.")


class Qwen3_5SelectedHybridCacheRouterForCausalLM(Qwen3_5ForCausalLM):
    config_class = Qwen3_5SelectedHybridCacheRouterConfig
    _keys_to_ignore_on_load_missing = [r"^router_head\."]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\.", r"^model\.visual\."]
    _checkpoint_conversion_mapping = {r"^model\.language_model\.": "model."}

    def __init__(self, config: Qwen3_5SelectedHybridCacheRouterConfig):
        _validate_config(config)
        super().__init__(config)
        self.router_head = Qwen3_5ThreeTowerInteractionRouterHead(config)
        self.router_head.apply(self._init_weights)
        self.router_head.float()
        parameter_count = sum(value.numel() for value in self.router_head.parameters())
        tensor_count = len(self.router_head.state_dict())
        if parameter_count != int(config.router_expected_head_parameter_count):
            raise RuntimeError(
                f"Router head has {parameter_count} parameters; expected "
                f"{config.router_expected_head_parameter_count}."
            )
        if tensor_count != int(config.router_expected_head_tensor_count):
            raise RuntimeError(
                f"Router head has {tensor_count} tensors; expected "
                f"{config.router_expected_head_tensor_count}."
            )

        self._router_checkpoint_loaded = False
        self._router_checkpoint_metadata: dict[str, Any] = {}
        self._previous_selected_hidden: torch.Tensor | None = None
        self._prompt_starts: list[int] | None = None
        self._unfinished_mask: torch.BoolTensor | None = None
        self._request_route_counts: torch.Tensor | None = None
        self._request_full_counts: torch.Tensor | None = None
        self._request_open = False
        self._active_mode = "routed"
        self._active_threshold = float(config.router_action_threshold)
        self._last_router_score: torch.Tensor | None = None
        self._last_route_full: torch.Tensor | None = None
        self._last_router_active_mask: torch.Tensor | None = None
        self._request_full_calls = 0
        self._request_full_subset_max = 0
        self._stats = {
            "request_count": 0,
            "prefill_row_tokens": 0,
            "routed_row_tokens": 0,
            "full_row_tokens": 0,
            "local_row_tokens": 0,
            "full_forward_calls": 0,
            "full_rerun_row_tokens": 0,
            "full_equivalent_reuse_row_tokens": 0,
            "nonfinite_score_row_tokens": 0,
            "no_routed_samples": 0,
            "samples_with_routes": 0,
            "zero_full_samples_with_routes": 0,
            "full_subset_batch_size_max": 0,
        }
        atexit.register(self.flush_router_summary)

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse=recurse)
        if hasattr(self, "router_head"):
            self.router_head.float()
        return module

    @staticmethod
    def _router_mode_from_env() -> str:
        mode = _env("QWEN35_SELECTED_CACHE_ROUTER_MODE", "routed")
        if mode not in {"local", "routed"}:
            raise ValueError("QWEN35_SELECTED_CACHE_ROUTER_MODE must be local or routed.")
        return mode

    def _checkpoint_path(self) -> Path:
        value = _env(
            "QWEN35_SELECTED_CACHE_ROUTER_CHECKPOINT",
            str(self.config.router_checkpoint or ""),
        )
        path = Path(value).expanduser() if value else Path("")
        if value and path.is_dir():
            path = path / "trainable_state.pt"
        if not value or not path.is_file():
            raise FileNotFoundError("Qwen3.5 selected-cache runtime requires a Router checkpoint.")
        return path

    def _audit_path(self) -> Path:
        value = _env("QWEN35_SELECTED_CACHE_ROUTER_AUDIT", str(self.config.router_audit or ""))
        path = Path(value).expanduser() if value else Path("")
        if not value or not path.is_file():
            raise FileNotFoundError("Qwen3.5 selected-cache runtime requires a formal audit file.")
        return path

    def _ensure_router_checkpoint(self) -> None:
        if self._router_checkpoint_loaded:
            return
        checkpoint_path = self._checkpoint_path()
        audit_path = self._audit_path()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Router checkpoint must use VeOmni format_version=1.")
        model_config = payload.get("model_config")
        state_dict = payload.get("state_dict")
        if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
            raise ValueError("Router checkpoint lacks model_config/state_dict.")
        architectures = model_config.get("architectures")
        if architectures != [_ROBUST_MARGIN_TRAINING_ARCHITECTURE]:
            raise ValueError(
                "Router checkpoint has an unsupported training architecture: "
                f"{architectures!r}."
            )
        expected = {
            "router_schema_version": 8,
            "router_schema_sha256": str(self.config.router_expected_training_schema_sha256),
            "router_feature_schema": _FEATURE_SCHEMA,
            "router_head_architecture": "three_tower_interaction_swiglu",
            "router_hidden_size": int(self.config.router_hidden_size),
            "router_num_blocks": int(self.config.router_num_blocks),
            "router_intermediate_size": int(self.config.router_intermediate_size),
            "router_expected_head_parameter_count": int(self.config.router_expected_head_parameter_count),
            "router_expected_head_tensor_count": int(self.config.router_expected_head_tensor_count),
            "router_training_history_mode": "full",
            "router_runtime_history_mode": "selected_hybrid_cache",
            "router_history_selector": "local_when_signed_nll_lt_threshold",
            "router_history_nll_threshold": 0.0,
            "router_sink_size": int(self.config.router_sink_size),
            "router_window_size": int(self.config.router_window_size),
            "router_window_includes_current": True,
            "router_one_step_target": "signed_nll",
            "router_one_step_transform": "signed_log1p",
            "router_base_weights_manifest_sha256": str(self.config.router_base_weights_manifest_sha256),
            "router_base_config_sha256": str(self.config.router_base_config_sha256),
        }
        objective_sha256 = None
        score_space = "signed_log1p_delta_nll"
        training_objective = "signed_log1p_delta_nll_huber_delta1"
        if architectures in (
            [_ROBUST_MARGIN_TRAINING_ARCHITECTURE],
            [_HISTORY_ALIGNED_ROBUST_MARGIN_TRAINING_ARCHITECTURE],
        ):
            history_aligned = architectures == [
                _HISTORY_ALIGNED_ROBUST_MARGIN_TRAINING_ARCHITECTURE
            ]
            objective_sha256 = (
                _HISTORY_ALIGNED_ROBUST_MARGIN_OBJECTIVE_SHA256
                if history_aligned
                else str(getattr(self.config, "router_expected_objective_sha256", ""))
            )
            if not re.fullmatch(r"[0-9a-f]{64}", objective_sha256):
                raise ValueError("The runtime requires a bound v4.2 objective SHA-256")
            score_space = "signed_log1p_robust_decision_margin"
            training_objective = "robust_decision_margin_huber"
            expected.update(
                {
                    "router_history_nll_threshold": 0.001 if history_aligned else 0.0,
                    "router_objective": training_objective,
                    "router_robust_decision_margin_objective_schema_version": 1,
                    "router_robust_decision_margin_objective": (
                        "eligible_token_mean_huber_on_signed_log1p_shifted_margin"
                    ),
                    "router_robust_decision_margin_objective_sha256": (
                        objective_sha256
                    ),
                    "router_robust_decision_margin_lambda": 0.001,
                    "router_robust_decision_margin_output_space": score_space,
                    "router_robust_decision_margin_loss_weighting": (
                        "eligible_token_equal"
                    ),
                    "router_robust_decision_margin_huber_delta": 1.0,
                    "router_robust_decision_margin_runtime_score_threshold": 0.0,
                    "router_robust_decision_margin_interpretation": (
                        "huber_m_estimate_not_conditional_mean"
                    ),
                }
            )
        elif architectures == [_FOCAL_TRAINING_ARCHITECTURE]:
            objective_sha256 = _FOCAL_OBJECTIVE_SHA256
            score_space = "full_action_focal_logit_non_calibrated"
            training_objective = "standard_binary_sigmoid_focal"
            expected.update(
                {
                    "router_objective": "standard_focal",
                    "router_focal_objective_schema_version": 1,
                    "router_focal_objective": (
                        "binary_sigmoid_focal_loss_with_logits"
                    ),
                    "router_focal_objective_sha256": objective_sha256,
                    "router_focal_decision_nll_threshold": 0.0,
                    "router_focal_full_alpha": 0.25,
                    "router_focal_gamma": 2.0,
                    "router_focal_normalization": (
                        "global_eligible_token_mean_across_data_parallel_ranks"
                    ),
                    "router_focal_output_space": "full_action_focal_logit",
                    "router_focal_probability_semantics": (
                        "sigmoid_focal_score_not_calibrated_full_probability"
                    ),
                    "router_focal_reference_logit_threshold": 0.0,
                    "router_focal_history_source": (
                        "detached_huber_matched_signed_nll_threshold_selected_hidden_right_shifted_one"
                    ),
                    "router_focal_boundary_policy": (
                        "full_when_strictly_greater_local_on_equal"
                    ),
                }
            )
        mismatches = {
            name: {"observed": model_config.get(name), "expected": value}
            for name, value in expected.items()
            if model_config.get(name) != value
        }
        if mismatches:
            raise ValueError(f"Router checkpoint/schema mismatch: {mismatches}.")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "passed" or audit.get("export_tensor_state_sha256") != payload.get(
            "tensor_state_sha256"
        ):
            raise ValueError("Router formal audit does not authorize this checkpoint.")
        if objective_sha256 is not None and audit.get("objective_sha256") != objective_sha256:
            raise ValueError("Router formal audit does not certify the training objective.")
        prefix = "router_head."
        router_state = {
            name[len(prefix) :]: tensor
            for name, tensor in state_dict.items()
            if name.startswith(prefix)
        }
        if len(router_state) != len(state_dict) or set(router_state) != set(self.router_head.state_dict()):
            raise ValueError("Router export tensor schema does not match the A0 runtime head.")
        tensor_hash = _tensor_state_sha256(state_dict)
        if tensor_hash != payload.get("tensor_state_sha256"):
            raise ValueError("Router embedded tensor-state hash mismatch.")
        self.router_head.load_state_dict(router_state, strict=True)
        self.router_head.to(self.lm_head.weight.device, dtype=torch.float32).eval()
        self._router_checkpoint_loaded = True
        self._router_checkpoint_metadata = {
            "path": str(checkpoint_path.resolve()),
            "file_sha256": _file_sha256(checkpoint_path),
            "tensor_state_sha256": tensor_hash,
            "audit": str(audit_path.resolve()),
            "audit_sha256": _file_sha256(audit_path),
            "global_step": int(payload.get("global_step", -1)),
            "schema_version": 8,
            "schema_sha256": model_config["router_schema_sha256"],
            "training_history_nll_threshold": float(
                model_config["router_history_nll_threshold"]
            ),
            "training_objective": training_objective,
            "objective_sha256": objective_sha256,
            "score_space": score_space,
        }

    def _begin_request(self, batch_size: int, prefill_length: int, device: torch.device) -> None:
        self._finish_request()
        self._active_mode = self._router_mode_from_env()
        self._request_open = True
        self._stats["request_count"] += batch_size
        self._stats["prefill_row_tokens"] += batch_size * prefill_length
        self._request_route_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._request_full_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        self._previous_selected_hidden = None
        self._prompt_starts = None
        self._request_full_calls = 0
        self._request_full_subset_max = 0

    def _finish_request(self) -> None:
        if not self._request_open:
            return
        route_counts = self._request_route_counts.detach().cpu()
        full_counts = self._request_full_counts.detach().cpu()
        has_routes = route_counts.gt(0)
        self._stats["no_routed_samples"] += int((~has_routes).sum())
        self._stats["samples_with_routes"] += int(has_routes.sum())
        self._stats["zero_full_samples_with_routes"] += int((has_routes & full_counts.eq(0)).sum())
        self._stats["full_forward_calls"] += self._request_full_calls
        self._stats["full_subset_batch_size_max"] = max(
            self._stats["full_subset_batch_size_max"], self._request_full_subset_max
        )
        self._request_open = False
        self._request_route_counts = None
        self._request_full_counts = None
        self._unfinished_mask = None
        self._previous_selected_hidden = None
        self._prompt_starts = None

    def _summary_path(self) -> Path | None:
        value = _env("QWEN35_SELECTED_CACHE_ROUTER_SUMMARY_PATH", str(self.config.router_summary_path or ""))
        return Path(value.replace("{pid}", str(os.getpid()))).expanduser() if value else None

    def router_telemetry_summary(self) -> dict[str, Any]:
        routed = int(self._stats["routed_row_tokens"])
        full = int(self._stats["full_row_tokens"])
        local = int(self._stats["local_row_tokens"])
        return {
            "format": _SUMMARY_FORMAT,
            "runtime_history_mode": "selected_hybrid_cache",
            "runtime_contract": "local_first_selected_hidden_logits_attention_kv_delta_states",
            "router_mode": self._active_mode,
            "router_action_threshold": float(self._active_threshold),
            "router_score_space": self._router_checkpoint_metadata.get(
                "score_space", "signed_log1p_delta_nll"
            ),
            "router_action_comparison": ">",
            "nonfinite_action": "full",
            "checkpoint": dict(self._router_checkpoint_metadata),
            **{name: int(value) for name, value in self._stats.items()},
            "full_rate": full / routed if routed else None,
            "full_rate_definition": "full_row_tokens / routed_row_tokens; prefill excluded",
            "selected_hybrid_cache_invariants_passed": full + local == routed,
        }

    def flush_router_summary(self) -> None:
        path = self._summary_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            temporary.write_text(
                json.dumps(self.router_telemetry_summary(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except Exception:
            if not getattr(__import__("sys"), "is_finalizing")():
                raise

    @staticmethod
    def _prompt_padding_starts(
        attention_mask: torch.Tensor | None, batch_size: int, length: int
    ) -> list[int]:
        if attention_mask is None:
            return [0] * batch_size
        if attention_mask.shape != (batch_size, length):
            raise ValueError("Full prefill requires an aligned 2-D attention mask.")
        mask = attention_mask.detach().to(device="cpu", dtype=torch.bool)
        starts = []
        columns = torch.arange(length)
        for row in range(batch_size):
            valid_count = int(mask[row].sum())
            if valid_count < 1:
                raise ValueError("Every prompt row must contain at least one token.")
            start = length - valid_count
            if not torch.equal(mask[row], columns.ge(start)):
                raise ValueError("Selected-cache runtime accepts only monotonic left padding.")
            starts.append(start)
        return starts

    def _decode_position_ids(
        self,
        position_ids: torch.LongTensor | None,
        batch_size: int,
        past_length: int,
        device: torch.device,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        if position_ids is None:
            # Match Qwen3.5TextModel.forward exactly for direct cached calls.
            # Generation supplies its own attention-mask-derived position IDs
            # when left padding needs row-specific positions.
            text = torch.full(
                (batch_size, 1), past_length, dtype=torch.long, device=device
            )
            return text, text.unsqueeze(0).expand(3, -1, -1)
        position_ids = position_ids.to(device=device, dtype=torch.long)
        if position_ids.ndim == 2:
            if position_ids.shape[0] == 1 and batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)
            if position_ids.shape != (batch_size, 1):
                raise ValueError("Decode position_ids must identify one token per row.")
            return position_ids, position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim == 3 and position_ids.shape == (4, batch_size, 1):
            return position_ids[0], position_ids[1:]
        raise ValueError("Qwen3.5 decode position_ids must be [B,1] or [4,B,1].")

    @staticmethod
    def _project_attention_current(attention, hidden_states, position_embeddings):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query, gate = torch.chunk(
            attention.q_proj(hidden_states).view(*input_shape, -1, attention.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query = attention.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        return query, key, value, gate

    @staticmethod
    def _attention_output(attention, query, key, value, gate, attention_mask):
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attention.config._attn_implementation, eager_attention_forward
        )
        output, _ = interface(
            attention,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=attention.scaling,
        )
        output = output.reshape(*gate.shape).contiguous() * torch.sigmoid(gate)
        return attention.o_proj(output)

    @staticmethod
    def _delta_candidate(layer_idx, module, normalized, cache_layer, row_indices):
        temporary = _DeltaCandidateCache(layer_idx, cache_layer, row_indices)
        output = module(hidden_states=normalized, cache_params=temporary, attention_mask=None)
        temporary_layer = temporary.layers[layer_idx]
        cache_layer.stage_indices(
            row_indices,
            temporary_layer.conv_states,
            temporary_layer.recurrent_states,
        )
        return output

    def _local_past_indices(self, row_indices, past_length, device):
        sink_size = int(self.config.router_sink_size)
        recent_limit = max(int(self.config.router_window_size) - 1, 0)
        rows = []
        lengths = []
        rows_cpu = row_indices.detach().cpu().tolist()
        for row in rows_cpu:
            valid_start = int(self._prompt_starts[row])
            history_count = max(past_length - valid_start, 0)
            sink_count = min(sink_size, history_count)
            sink = torch.arange(valid_start, valid_start + sink_count, device=device)
            recent_start = max(valid_start, past_length - recent_limit)
            recent_start = max(recent_start, valid_start + sink_count)
            recent = torch.arange(recent_start, past_length, device=device)
            selected = torch.cat((sink, recent)).long()
            rows.append(selected)
            lengths.append(int(selected.numel()))
        max_past = max(lengths, default=0)
        indices = torch.zeros(len(rows_cpu), max_past, dtype=torch.long, device=device)
        valid = torch.zeros(len(rows_cpu), max_past + 1, dtype=torch.bool, device=device)
        for offset, selected in enumerate(rows):
            length = int(selected.numel())
            if length:
                indices[offset, :length] = selected
                valid[offset, :length] = True
            valid[offset, max_past] = True
        return indices, valid

    def _local_attention(
        self,
        attention,
        cache_layer,
        query,
        current_key,
        current_value,
        gate,
        row_indices,
        past_length,
    ):
        indices, valid = self._local_past_indices(
            row_indices, past_length, current_key.device
        )
        batch = int(row_indices.numel())
        max_past = indices.shape[1]
        key = torch.empty(
            batch,
            current_key.shape[1],
            max_past + 1,
            current_key.shape[-1],
            dtype=current_key.dtype,
            device=current_key.device,
        )
        value = torch.empty_like(key)
        if max_past:
            expanded = indices[:, None, :, None].expand(
                batch, current_key.shape[1], max_past, current_key.shape[-1]
            )
            torch.gather(
                cache_layer.keys.index_select(0, row_indices)[:, :, :past_length],
                dim=2,
                index=expanded,
                out=key[:, :, :max_past],
            )
            torch.gather(
                cache_layer.values.index_select(0, row_indices)[:, :, :past_length],
                dim=2,
                index=expanded,
                out=value[:, :, :max_past],
            )
        key[:, :, max_past:].copy_(current_key)
        value[:, :, max_past:].copy_(current_value)
        mask = None if bool(valid.all()) else valid[:, None, None, :]
        return self._attention_output(attention, query, key, value, gate, mask)

    def _full_attention(
        self,
        attention,
        cache_layer,
        query,
        current_key,
        current_value,
        gate,
        row_indices,
        past_length,
    ):
        key = torch.cat(
            (cache_layer.keys.index_select(0, row_indices)[:, :, :past_length], current_key),
            dim=2,
        )
        value = torch.cat(
            (cache_layer.values.index_select(0, row_indices)[:, :, :past_length], current_value),
            dim=2,
        )
        starts = [self._prompt_starts[row] for row in row_indices.detach().cpu().tolist()]
        if all(start == 0 for start in starts):
            mask = None
        else:
            columns = torch.arange(past_length + 1, device=current_key.device)
            starts_tensor = torch.tensor(starts, device=current_key.device)
            mask = columns[None].ge(starts_tensor[:, None])[:, None, None, :]
        return self._attention_output(attention, query, key, value, gate, mask)

    def _candidate(
        self,
        current_embedding,
        position_embeddings,
        cache,
        past_length,
        active_mask,
        full,
        selected_hidden=None,
    ):
        if selected_hidden is None:
            selected_hidden = self._previous_selected_hidden[:, None].clone()
        row_indices = active_mask.nonzero(as_tuple=False).flatten()
        if row_indices.numel() == 0:
            return selected_hidden
        cos, sin = position_embeddings
        if full:
            self._request_full_calls += 1
            self._request_full_subset_max = max(
                self._request_full_subset_max, int(row_indices.numel())
            )
        hidden_states = current_embedding.index_select(0, row_indices)
        run_position_embeddings = (
            cos.index_select(0, row_indices),
            sin.index_select(0, row_indices),
        )
        for layer_idx, layer in enumerate(self.model.layers):
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            cache_layer = cache.layers[layer_idx]
            if layer.layer_type == "linear_attention":
                update = self._delta_candidate(
                    layer_idx,
                    layer.linear_attn,
                    normalized,
                    cache_layer,
                    row_indices,
                )
            else:
                query, current_key, current_value, gate = self._project_attention_current(
                    layer.self_attn, normalized, run_position_embeddings
                )
                cache_layer.stage_indices(row_indices, current_key, current_value)
                function = self._full_attention if full else self._local_attention
                update = function(
                    layer.self_attn,
                    cache_layer,
                    query,
                    current_key,
                    current_value,
                    gate,
                    row_indices,
                    past_length,
                )
            hidden_states = residual + update
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )
        selected_hidden.index_copy_(0, row_indices, self.model.norm(hidden_states))
        return selected_hidden

    def _full_equivalent_mask(self, active_mask, past_length):
        limit = int(self.config.router_sink_size) + int(self.config.router_window_size)
        lengths = torch.tensor(
            [past_length - int(start) + 1 for start in self._prompt_starts],
            dtype=torch.long,
            device=active_mask.device,
        )
        return active_mask & lengths.le(limit)

    def _record_decode(self, active_mask, route_full, finite, full_rerun, equivalent):
        local = active_mask & ~route_full
        self._request_route_counts.add_(active_mask.long())
        self._request_full_counts.add_(route_full.long())
        self._stats["routed_row_tokens"] += int(active_mask.sum())
        self._stats["full_row_tokens"] += int(route_full.sum())
        self._stats["local_row_tokens"] += int(local.sum())
        self._stats["nonfinite_score_row_tokens"] += int((active_mask & ~finite).sum())
        self._stats["full_rerun_row_tokens"] += int(full_rerun.sum())
        self._stats["full_equivalent_reuse_row_tokens"] += int(equivalent.sum())

    def _prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs: dict,
        generation_mode,
        batch_size: int,
        max_cache_length: int,
    ) -> None:
        del generation_mode
        existing = model_kwargs.get("past_key_values")
        if existing is not None:
            if not isinstance(existing, SelectedHybridStaticCache):
                raise TypeError("Only SelectedHybridStaticCache may be passed explicitly.")
            return
        if generation_config.use_cache is False:
            raise ValueError("Selected-hybrid-cache generation requires use_cache=True.")
        if generation_config.cache_implementation is not None:
            raise ValueError("cache_implementation is fixed by selected-cache modeling.")
        if max_cache_length > int(self.config.max_position_embeddings):
            raise ValueError("Requested generation exceeds the Qwen3.5 position limit.")
        model_kwargs["past_key_values"] = SelectedHybridStaticCache(
            self.config, max_cache_length
        )

    def generate(self, *args, **kwargs):
        generation_config = kwargs.get("generation_config", self.generation_config)

        def resolved(name, default=None):
            value = kwargs.get(name, getattr(generation_config, name, default))
            return default if value is None else value

        if bool(resolved("do_sample", False)):
            raise ValueError("Selected-cache evaluation requires do_sample=False.")
        if int(resolved("num_beams", 1)) != 1 or int(resolved("num_return_sequences", 1)) != 1:
            raise ValueError("Selected-cache evaluation supports greedy single-return decoding only.")
        if kwargs.get("assistant_model") is not None:
            raise ValueError("Selected-cache evaluation rejects assisted decoding.")
        self._active_threshold = _finite_env_float(
            "QWEN35_SELECTED_CACHE_ROUTER_ACTION_THRESHOLD",
            float(self.config.router_action_threshold),
        )
        self._active_mode = self._router_mode_from_env()
        if self._active_mode == "routed":
            self._ensure_router_checkpoint()
        try:
            return super().generate(*args, **kwargs)
        finally:
            self._finish_request()
            self.flush_router_summary()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        router_active_mask: torch.BoolTensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if labels is not None:
            raise ValueError("Production selected-cache modeling does not score labels.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds.")
        if use_cache is False or output_attentions:
            raise ValueError("Selected-cache runtime requires cache and does not return attentions.")
        if self.config._attn_implementation != "sdpa":
            raise ValueError("Formal selected-cache runtime requires SDPA.")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        batch_size, sequence_length, _ = inputs_embeds.shape
        if past_key_values is None:
            past_key_values = SelectedHybridStaticCache(
                self.config, int(self.config.max_position_embeddings)
            )
        if not isinstance(past_key_values, SelectedHybridStaticCache):
            raise TypeError("Selected-cache modeling requires SelectedHybridStaticCache.")
        past_length = past_key_values.get_seq_length()

        if past_length == 0:
            self._begin_request(batch_size, sequence_length, inputs_embeds.device)
            self._prompt_starts = self._prompt_padding_starts(
                attention_mask, batch_size, sequence_length
            )
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state
            self._previous_selected_hidden = hidden_states[:, -1].detach()
            past_key_values.assert_ready_and_aligned()
        else:
            if sequence_length != 1:
                raise ValueError("Cached selected-cache decode consumes exactly one token.")
            if self._previous_selected_hidden is None or self._unfinished_mask is None:
                raise RuntimeError("Decode is missing Full-prefill Router state.")
            active_mask = self._unfinished_mask.clone()
            if router_active_mask is not None:
                if router_active_mask.shape != (batch_size,):
                    raise ValueError("router_active_mask must have shape [batch].")
                active_mask &= router_active_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            elif input_ids is not None:
                eos_ids = self.generation_config.eos_token_id or self.config.eos_token_id
                eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
                for eos_id in eos_ids or []:
                    active_mask &= input_ids[:, -1].ne(int(eos_id))
            self._unfinished_mask = active_mask.detach().clone()

            if _contiguous_runs(active_mask):
                _, rotary_positions = self._decode_position_ids(
                    position_ids, batch_size, past_length, inputs_embeds.device
                )
                position_embeddings = self.model.rotary_emb(inputs_embeds, rotary_positions)
                past_key_values.begin_stage()
                local_hidden = self._candidate(
                    inputs_embeds,
                    position_embeddings,
                    past_key_values,
                    past_length,
                    active_mask,
                    full=False,
                )
                if self._active_mode == "routed":
                    self._ensure_router_checkpoint()
                    score = self.router_head(
                        self._previous_selected_hidden,
                        inputs_embeds[:, 0],
                        local_hidden[:, 0],
                    )
                    finite = torch.isfinite(score)
                    route_full = active_mask & (
                        (finite & score.gt(self._active_threshold)) | ~finite
                    )
                else:
                    score = torch.zeros(batch_size, dtype=torch.float32, device=inputs_embeds.device)
                    finite = torch.ones_like(active_mask)
                    route_full = torch.zeros_like(active_mask)
                selected_hidden = local_hidden.clone()
                equivalent = route_full & self._full_equivalent_mask(active_mask, past_length)
                full_rerun = route_full & ~equivalent
                if _contiguous_runs(full_rerun):
                    selected_hidden = self._candidate(
                        inputs_embeds,
                        position_embeddings,
                        past_key_values,
                        past_length,
                        full_rerun,
                        full=True,
                        selected_hidden=selected_hidden,
                    )
                past_key_values.commit_one()
                self._previous_selected_hidden = torch.where(
                    active_mask[:, None],
                    selected_hidden[:, 0],
                    self._previous_selected_hidden,
                ).detach()
                self._record_decode(active_mask, route_full, finite, full_rerun, equivalent)
                self._last_router_score = score.detach()
                self._last_route_full = route_full.detach()
                self._last_router_active_mask = active_mask.detach()
                hidden_states = selected_hidden
            else:
                hidden_states = self._previous_selected_hidden[:, None]
                self._last_router_score = None
                self._last_route_full = None
                self._last_router_active_mask = active_mask.detach()

        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, indices])
        if return_dict is False:
            return logits, past_key_values
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=None,
        )


__all__ = [
    "Qwen3_5SelectedHybridCacheRouterForCausalLM",
    "Qwen3_5ThreeTowerInteractionRouterHead",
    "SelectedHybridStaticCache",
    "SelectedFullAttentionStaticLayer",
    "SelectedDeltaNetStaticLayer",
]


# Transformers 5.x moved checkpoint key conversion from a model-class
# attribute to a central registry.  Reuse the official qwen3_5_text transform
# so the multimodal checkpoint's `model.language_model.*` tensors populate this
# text-only runtime as `model.*`; vision and MTP remain explicitly excluded.
register_checkpoint_conversion_mapping(
    "Qwen3_5SelectedHybridCacheRouterForCausalLM",
    get_checkpoint_conversion_mapping("qwen3_5_text"),
    overwrite=True,
)
