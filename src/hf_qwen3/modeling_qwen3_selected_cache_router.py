"""Efficient causal selected-mixed-cache deployment for the Qwen3 Router.

The production contract is deliberately narrow: Full prefill, then one
Local-first Router action for each consumed generated token.  The action
selects the current hidden state, logits source, and every layer's current K/V.
No observed next token, NLL label, oracle branch, or training-history selector
is available in this runtime.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .configuration_qwen3_selected_cache_router import (
    Qwen3SelectedCacheRouterConfig,
)


_FEATURE_SCHEMA = ["history_state", "current_embedding", "current_swa_hidden"]
_SUMMARY_FORMAT = "qwen3-selected-cache-router-aggregate-v1"


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
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


class SelectedMixedStaticLayer(CacheLayerMixin):
    """Preallocated K/V storage with one uncommitted current-token slot."""

    is_compileable = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = int(max_cache_len)
        self.max_batch_size = 0
        self.cumulative_length = 0
        self.device = torch.device("cpu")

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("Selected cache expects aligned [B,H,T,D] K/V.")
        self.max_batch_size = int(key_states.shape[0])
        self.device = key_states.device
        shape = (
            self.max_batch_size,
            int(key_states.shape[1]),
            self.max_cache_len,
            int(key_states.shape[3]),
        )
        self.keys = torch.empty(shape, dtype=key_states.dtype, device=key_states.device)
        self.values = torch.empty(
            shape, dtype=value_states.dtype, device=value_states.device
        )
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if int(key_states.shape[0]) != self.max_batch_size:
            raise ValueError("Selected cache batch size cannot change inside a request.")
        length = int(key_states.shape[-2])
        start = self.cumulative_length
        end = start + length
        if end > self.max_cache_len:
            raise ValueError(
                f"Selected cache capacity exceeded: {end} > {self.max_cache_len}."
            )
        self.keys[:, :, start:end, :].copy_(key_states)
        self.values[:, :, start:end, :].copy_(value_states)
        self.cumulative_length = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def stage_rows(
        self,
        row_start: int,
        row_end: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        if not self.is_initialized:
            raise RuntimeError("Cannot stage into an uninitialized selected cache.")
        position = self.cumulative_length
        if position >= self.max_cache_len:
            raise ValueError("Selected cache has no free staging slot.")
        expected = row_end - row_start
        if key_states.shape[0] != expected or value_states.shape != key_states.shape:
            raise ValueError("Staged K/V rows do not match the selected cache slice.")
        self.keys[row_start:row_end, :, position : position + 1, :].copy_(
            key_states
        )
        self.values[row_start:row_end, :, position : position + 1, :].copy_(
            value_states
        )

    def commit_one(self) -> None:
        if self.cumulative_length >= self.max_cache_len:
            raise ValueError("Selected cache capacity exceeded during commit.")
        self.cumulative_length += 1

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cumulative_length + int(query_length), 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


class SelectedMixedStaticCache(Cache):
    """All-layer selected cache whose physical addresses never change."""

    def __init__(self, num_layers: int, max_cache_len: int):
        if num_layers < 1 or max_cache_len < 1:
            raise ValueError("Selected cache dimensions must be positive.")
        super().__init__(
            layers=[
                SelectedMixedStaticLayer(max_cache_len) for _ in range(num_layers)
            ]
        )
        self.capacity = int(max_cache_len)

    def assert_aligned(self) -> int:
        lengths = [layer.get_seq_length() for layer in self.layers]
        if len(set(lengths)) != 1:
            raise AssertionError(f"Selected cache layer lengths diverged: {lengths}.")
        return lengths[0]

    def commit_one(self) -> None:
        self.assert_aligned()
        for layer in self.layers:
            layer.commit_one()
        self.assert_aligned()

    def committed_data_ptrs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (layer.keys.data_ptr(), layer.values.data_ptr())
            for layer in self.layers
            if layer.is_initialized
        )


class Qwen3RouterSwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float):
        super().__init__()
        self.norm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.gate_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_projection = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states)
        update = self.down_projection(
            F.silu(self.gate_projection(normalized))
            * self.up_projection(normalized)
        )
        return hidden_states + update


class Qwen3ThreeTowerInteractionRouterHead(nn.Module):
    """Exact A0/schema-v4 28.325889M-parameter Router head."""

    def __init__(self, config: Qwen3SelectedCacheRouterConfig):
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
            Qwen3RouterSwiGLUBlock(
                router_size, int(config.router_intermediate_size), eps
            )
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
            history = F.silu(
                self.history_projection(self.history_norm(history_state.float()))
            )
            embedding = F.silu(
                self.embedding_projection(
                    self.embedding_norm(current_embedding.float())
                )
            )
            swa = F.silu(
                self.swa_projection(self.swa_norm(current_swa_hidden.float()))
            )
            groups = [history, embedding, swa]
            for left, right in (
                (history, embedding),
                (history, swa),
                (embedding, swa),
            ):
                groups.extend((left * right, torch.abs(left - right)))
            fused = F.silu(self.fusion_projection(torch.cat(groups, dim=-1)))
            hidden_states = (history + embedding + swa + fused) * 0.5
            for block in self.blocks:
                hidden_states = block(hidden_states)
            return self.output_projection(
                self.output_norm(hidden_states)
            ).squeeze(-1)


def _validate_config(config: Qwen3SelectedCacheRouterConfig) -> None:
    if list(config.router_feature_schema) != _FEATURE_SCHEMA:
        raise ValueError(f"Runtime requires exact feature schema {_FEATURE_SCHEMA}.")
    if config.router_head_architecture != "three_tower_interaction_swiglu":
        raise ValueError("Selected-cache deployment accepts only the A0 Router head.")
    if int(config.router_sink_size) < 0 or int(config.router_window_size) < 1:
        raise ValueError("Invalid sink/window configuration.")
    if bool(config.use_sliding_window) or config.sliding_window is not None:
        raise ValueError("Native sliding caches would destroy recoverable Full history.")
    if not math.isfinite(float(config.router_action_threshold)):
        raise ValueError("Router action threshold must be finite.")


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


class Qwen3SelectedCacheRouterForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3SelectedCacheRouterConfig
    # Base safetensors intentionally contain only frozen Qwen weights. Router
    # tensors are mandatory external weights validated by _ensure_router_checkpoint.
    _keys_to_ignore_on_load_missing = {r"^router_head\."}

    def __init__(self, config: Qwen3SelectedCacheRouterConfig):
        _validate_config(config)
        super().__init__(config)
        self.router_head = Qwen3ThreeTowerInteractionRouterHead(config)
        self.router_head.apply(self._init_weights)
        self.router_head.float()
        parameter_count = sum(p.numel() for p in self.router_head.parameters())
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
        self._request_counters: torch.Tensor | None = None
        self._request_full_calls = 0
        self._request_full_subset_max = 0
        self._request_open = False
        self._active_mode = "routed"
        self._active_threshold = float(config.router_action_threshold)
        self._last_router_score: torch.Tensor | None = None
        self._last_route_full: torch.Tensor | None = None
        self._last_router_active_mask: torch.Tensor | None = None
        self._local_key_workspace: torch.Tensor | None = None
        self._local_value_workspace: torch.Tensor | None = None
        self._stats: dict[str, int] = {
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
            "full_candidate_computed_row_tokens": 0,
            "local_candidate_computed_row_tokens": 0,
            "full_candidate_active_row_tokens": 0,
            "local_candidate_active_row_tokens": 0,
            "selected_full_kv_row_tokens": 0,
            "selected_local_kv_row_tokens": 0,
        }
        atexit.register(self.flush_router_summary)

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse=recurse)
        if hasattr(self, "router_head"):
            self.router_head.float()
        return module

    def _checkpoint_path(self) -> Path:
        value = _env(
            "QWEN3_SELECTED_CACHE_ROUTER_CHECKPOINT",
            str(self.config.router_checkpoint or ""),
        )
        path = Path(value).expanduser() if value else Path("")
        if value and path.is_dir():
            path = path / "trainable_state.pt"
        if not value or not path.is_file():
            raise FileNotFoundError(
                "Selected-cache deployment requires "
                "QWEN3_SELECTED_CACHE_ROUTER_CHECKPOINT."
            )
        return path

    def _audit_path(self) -> Path:
        value = _env(
            "QWEN3_SELECTED_CACHE_ROUTER_AUDIT",
            str(self.config.router_audit or ""),
        )
        path = Path(value).expanduser() if value else Path("")
        if not value or not path.is_file():
            raise FileNotFoundError(
                "Selected-cache deployment requires a formal _SUCCESS file via "
                "QWEN3_SELECTED_CACHE_ROUTER_AUDIT."
            )
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

        expected = {
            "architectures": ["Qwen3ThreeStreamNllSelectedRouterForCausalLM"],
            "router_schema_version": 4,
            "router_schema_sha256": str(
                self.config.router_expected_training_schema_sha256
            ),
            "router_feature_schema": _FEATURE_SCHEMA,
            "router_head_architecture": "three_tower_interaction_swiglu",
            "router_hidden_size": int(self.config.router_hidden_size),
            "router_num_blocks": int(self.config.router_num_blocks),
            "router_intermediate_size": int(self.config.router_intermediate_size),
            "router_expected_head_parameter_count": int(
                self.config.router_expected_head_parameter_count
            ),
            "router_expected_head_tensor_count": int(
                self.config.router_expected_head_tensor_count
            ),
            "router_training_history_mode": "full",
            "router_runtime_history_mode": "full_cache_nll_selected",
            "router_history_selector": "local_when_signed_nll_lt_threshold",
            "router_history_nll_threshold": 0.001,
            "router_sink_size": int(self.config.router_sink_size),
            "router_window_size": int(self.config.router_window_size),
            "router_swa_all_layers": True,
            "router_window_includes_current": True,
            "router_one_step_target": "signed_nll",
            "router_one_step_transform": "signed_log1p",
            "router_one_step_nll_scale": 1.0,
            "router_base_weights_manifest_sha256": str(
                self.config.router_base_weights_manifest_sha256
            ),
            "router_base_config_sha256": str(
                self.config.router_base_config_sha256
            ),
        }
        mismatches = {
            name: {"observed": model_config.get(name), "expected": value}
            for name, value in expected.items()
            if model_config.get(name) != value
        }
        if mismatches:
            raise ValueError(f"Router checkpoint/schema mismatch: {mismatches}.")

        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected_audit = {
            "status": "passed",
            "format": "qwen3-three-stream-nll-selected-success-v1",
            "history": "full",
            "runtime_history_mode": "full_cache_nll_selected",
            "history_nll_threshold": 0.001,
            "export_tensor_state_sha256": payload.get("tensor_state_sha256"),
        }
        audit_mismatches = {
            name: {"observed": audit.get(name), "expected": value}
            for name, value in expected_audit.items()
            if audit.get(name) != value
        }
        if audit_mismatches:
            raise ValueError(f"Router formal audit mismatch: {audit_mismatches}.")
        detailed_audit = Path(str(audit.get("audit", "")))
        if not detailed_audit.is_file() or _file_sha256(detailed_audit) != audit.get(
            "audit_sha256"
        ):
            raise ValueError("Router detailed audit file/hash mismatch.")

        prefix = "router_head."
        router_state = {
            name[len(prefix) :]: tensor
            for name, tensor in state_dict.items()
            if name.startswith(prefix)
        }
        if len(router_state) != len(state_dict):
            raise ValueError("Router export contains non-Router trainable tensors.")
        expected_keys = set(self.router_head.state_dict())
        if set(router_state) != expected_keys:
            raise ValueError(
                "Router tensor schema mismatch: "
                f"missing={sorted(expected_keys - set(router_state))}, "
                f"unexpected={sorted(set(router_state) - expected_keys)}."
            )
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
            "schema_version": 4,
            "schema_sha256": model_config["router_schema_sha256"],
            "training_history_mode": "full",
            "training_history_nll_threshold": 0.001,
        }

    def _summary_path(self) -> Path | None:
        value = _env(
            "QWEN3_SELECTED_CACHE_ROUTER_SUMMARY_PATH",
            str(self.config.router_summary_path or ""),
        )
        if not value:
            return None
        return Path(value.replace("{pid}", str(os.getpid()))).expanduser()

    def _begin_request(
        self, batch_size: int, prefill_length: int, device: torch.device
    ) -> None:
        self._finish_request()
        self._active_mode = self._router_mode_from_env()
        self._request_open = True
        self._stats["request_count"] += int(batch_size)
        self._stats["prefill_row_tokens"] += int(batch_size * prefill_length)
        self._request_route_counts = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        self._request_full_counts = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        # routed, full, local, nonfinite, local-candidate, full-candidate,
        # selected-local-kv, selected-full-kv, full-rerun,
        # full-equivalent-reuse
        self._request_counters = torch.zeros(10, dtype=torch.long, device=device)
        self._request_full_calls = 0
        self._request_full_subset_max = 0
        self._unfinished_mask = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )
        self._previous_selected_hidden = None
        self._prompt_starts = None

    @staticmethod
    def _router_mode_from_env() -> str:
        mode = _env("QWEN3_SELECTED_CACHE_ROUTER_MODE", "routed")
        if mode not in {"local", "routed"}:
            raise ValueError(
                "QWEN3_SELECTED_CACHE_ROUTER_MODE must be 'local' or 'routed'."
            )
        return mode

    def _finish_request(self) -> None:
        if not self._request_open:
            return
        counters = self._request_counters.detach().cpu().tolist()
        route_counts = self._request_route_counts.detach().cpu()
        full_counts = self._request_full_counts.detach().cpu()
        names = (
            "routed_row_tokens",
            "full_row_tokens",
            "local_row_tokens",
            "nonfinite_score_row_tokens",
            "local_candidate_active_row_tokens",
            "full_candidate_active_row_tokens",
            "selected_local_kv_row_tokens",
            "selected_full_kv_row_tokens",
            "full_rerun_row_tokens",
            "full_equivalent_reuse_row_tokens",
        )
        for name, value in zip(names, counters):
            self._stats[name] += int(value)
        self._stats["local_candidate_computed_row_tokens"] += int(counters[4])
        self._stats["full_candidate_computed_row_tokens"] += int(counters[5])
        self._stats["full_forward_calls"] += int(self._request_full_calls)
        self._stats["full_subset_batch_size_max"] = max(
            self._stats["full_subset_batch_size_max"],
            int(self._request_full_subset_max),
        )
        has_routes = route_counts.gt(0)
        self._stats["no_routed_samples"] += int((~has_routes).sum().item())
        self._stats["samples_with_routes"] += int(has_routes.sum().item())
        self._stats["zero_full_samples_with_routes"] += int(
            (has_routes & full_counts.eq(0)).sum().item()
        )
        self._request_open = False
        self._request_route_counts = None
        self._request_full_counts = None
        self._request_counters = None
        self._unfinished_mask = None
        self._previous_selected_hidden = None
        self._prompt_starts = None

    def router_telemetry_summary(self) -> dict[str, Any]:
        routed = int(self._stats["routed_row_tokens"])
        full = int(self._stats["full_row_tokens"])
        local = int(self._stats["local_row_tokens"])
        invariant = (
            self._stats["local_candidate_active_row_tokens"] == routed
            and self._stats["full_candidate_active_row_tokens"] == full
            and self._stats["selected_full_kv_row_tokens"] == full
            and self._stats["selected_local_kv_row_tokens"] == local
            and self._stats["full_rerun_row_tokens"]
            + self._stats["full_equivalent_reuse_row_tokens"]
            == full
            and self._stats["full_forward_calls"]
            <= self._stats["full_rerun_row_tokens"]
            and full + local == routed
        )
        return {
            "format": _SUMMARY_FORMAT,
            "runtime_history_mode": "selected_cache_router_real",
            "runtime_contract": (
                "local_first_one_router_overlap_full_reuse_"
                "selected_hidden_logits_kv"
            ),
            "router_mode": self._active_mode,
            "telemetry_granularity": "aggregate_only",
            "oracle_telemetry_enabled": False,
            "router_action_threshold": float(self._active_threshold),
            "router_score_space": "signed_log1p_delta_nll",
            "router_action_comparison": ">",
            "nonfinite_action": "full",
            "full_equivalent_max_sequence_length": int(
                self.config.router_sink_size
            )
            + int(self.config.router_window_size),
            "training_history_nll_threshold_used_at_runtime": False,
            "checkpoint_training_history_nll_threshold": (
                self._router_checkpoint_metadata.get(
                    "training_history_nll_threshold"
                )
            ),
            "checkpoint": dict(self._router_checkpoint_metadata),
            **{name: int(value) for name, value in self._stats.items()},
            "full_rate": full / routed if routed else None,
            "full_rate_definition": (
                "full_row_tokens / routed_row_tokens; prefill excluded"
            ),
            "selected_cache_router_real_invariants_passed": bool(invariant),
        }

    def flush_router_summary(self) -> None:
        path = self._summary_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            temporary.write_text(
                json.dumps(
                    self.router_telemetry_summary(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except Exception:
            # Never mask the original inference exception from an atexit hook.
            if not getattr(__import__("sys"), "is_finalizing")():
                raise

    @staticmethod
    def _project_current(
        attention,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(hidden_states).view(shape)).transpose(
            1, 2
        )
        key = attention.k_norm(attention.k_proj(hidden_states).view(shape)).transpose(
            1, 2
        )
        value = attention.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        return query, key, value

    @staticmethod
    def _attention_output(
        attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
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
            sliding_window=None,
        )
        return attention.o_proj(
            output.reshape(query.shape[0], 1, -1).contiguous()
        )

    def _position_embeddings(
        self, inputs_embeds: torch.Tensor, position_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.shape[0] == 1 and inputs_embeds.shape[0] > 1:
            position_ids = position_ids.expand(inputs_embeds.shape[0], -1)
        return self.model.rotary_emb(inputs_embeds, position_ids)

    def _ensure_local_workspace(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_keys = int(self.config.router_sink_size) + int(
            self.config.router_window_size
        )
        required = (batch_size, num_heads, max_keys, head_dim)
        existing = self._local_key_workspace
        if (
            existing is None
            or existing.shape[0] < batch_size
            or tuple(existing.shape[1:]) != required[1:]
            or existing.dtype != dtype
            or existing.device != device
        ):
            self._local_key_workspace = torch.empty(
                required, dtype=dtype, device=device
            )
            self._local_value_workspace = torch.empty(
                required, dtype=dtype, device=device
            )
        return self._local_key_workspace, self._local_value_workspace

    def _local_indices(
        self,
        row_start: int,
        row_end: int,
        past_length: int,
        device: torch.device,
    ) -> tuple[torch.LongTensor, torch.BoolTensor, bool]:
        if self._prompt_starts is None:
            raise RuntimeError("Prompt padding metadata is missing.")
        sink_size = int(self.config.router_sink_size)
        recent_limit = max(int(self.config.router_window_size) - 1, 0)
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        for row in range(row_start, row_end):
            valid_start = int(self._prompt_starts[row])
            history_count = max(past_length - valid_start, 0)
            sink_count = min(sink_size, history_count)
            sink = torch.arange(
                valid_start,
                valid_start + sink_count,
                dtype=torch.long,
                device=device,
            )
            recent_start = max(valid_start, past_length - recent_limit)
            recent_start = max(recent_start, valid_start + sink_count)
            recent = torch.arange(
                recent_start, past_length, dtype=torch.long, device=device
            )
            current = torch.tensor([past_length], dtype=torch.long, device=device)
            selected = torch.cat((sink, recent, current))
            rows.append(selected)
            lengths.append(int(selected.numel()))
        max_length = max(lengths)
        indices = torch.zeros(
            row_end - row_start, max_length, dtype=torch.long, device=device
        )
        valid = torch.zeros(
            row_end - row_start, max_length, dtype=torch.bool, device=device
        )
        for offset, selected in enumerate(rows):
            length = int(selected.numel())
            indices[offset, :length] = selected
            valid[offset, :length] = True
        return indices, valid, len(set(lengths)) == 1

    def _full_equivalent_mask(
        self, active_mask: torch.BoolTensor, past_length: int
    ) -> torch.BoolTensor:
        """Rows whose exact Local key union already equals Full history."""

        if self._prompt_starts is None:
            raise RuntimeError("Prompt padding metadata is missing.")
        full_equivalent_limit = int(self.config.router_sink_size) + int(
            self.config.router_window_size
        )
        sequence_lengths = torch.tensor(
            [past_length - int(start) + 1 for start in self._prompt_starts],
            dtype=torch.long,
            device=active_mask.device,
        )
        return active_mask & sequence_lengths.le(full_equivalent_limit)

    def _local_candidate(
        self,
        current_embedding: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: SelectedMixedStaticCache,
        past_length: int,
        active_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        if self._previous_selected_hidden is None:
            raise RuntimeError("A Full prefill must initialize Router history.")
        result = self._previous_selected_hidden[:, None, :].clone()
        cos, sin = position_embeddings
        for row_start, row_end in _contiguous_runs(active_mask):
            hidden_states = current_embedding[row_start:row_end]
            run_position_embeddings = (
                cos[row_start:row_end],
                sin[row_start:row_end],
            )
            indices, valid, uniform = self._local_indices(
                row_start,
                row_end,
                past_length,
                current_embedding.device,
            )
            for layer_idx, layer in enumerate(self.model.layers):
                residual = hidden_states
                normalized = layer.input_layernorm(hidden_states)
                query, current_key, current_value = self._project_current(
                    layer.self_attn, normalized, run_position_embeddings
                )
                cache_layer = cache.layers[layer_idx]
                cache_layer.stage_rows(
                    row_start, row_end, current_key, current_value
                )
                workspace_key, workspace_value = self._ensure_local_workspace(
                    row_end - row_start,
                    int(current_key.shape[1]),
                    int(current_key.shape[-1]),
                    current_key.dtype,
                    current_key.device,
                )
                key_out = workspace_key[
                    : row_end - row_start, :, : indices.shape[1], :
                ]
                value_out = workspace_value[
                    : row_end - row_start, :, : indices.shape[1], :
                ]
                expanded = indices[:, None, :, None].expand(
                    row_end - row_start,
                    current_key.shape[1],
                    indices.shape[1],
                    current_key.shape[-1],
                )
                torch.gather(
                    cache_layer.keys[row_start:row_end, :, : past_length + 1, :],
                    dim=2,
                    index=expanded,
                    out=key_out,
                )
                torch.gather(
                    cache_layer.values[
                        row_start:row_end, :, : past_length + 1, :
                    ],
                    dim=2,
                    index=expanded,
                    out=value_out,
                )
                local_mask = None if uniform else valid[:, None, None, :]
                attention_output = self._attention_output(
                    layer.self_attn, query, key_out, value_out, local_mask
                )
                hidden_states = residual + attention_output
                hidden_states = hidden_states + layer.mlp(
                    layer.post_attention_layernorm(hidden_states)
                )
            result[row_start:row_end] = self.model.norm(hidden_states)
        return result

    def _full_candidate(
        self,
        current_embedding: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: SelectedMixedStaticCache,
        past_length: int,
        full_mask: torch.BoolTensor,
        selected_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if self._prompt_starts is None:
            raise RuntimeError("Prompt padding metadata is missing.")
        cos, sin = position_embeddings
        for row_start, row_end in _contiguous_runs(full_mask):
            self._request_full_calls += 1
            self._request_full_subset_max = max(
                self._request_full_subset_max, row_end - row_start
            )
            hidden_states = current_embedding[row_start:row_end]
            run_position_embeddings = (
                cos[row_start:row_end],
                sin[row_start:row_end],
            )
            starts = self._prompt_starts[row_start:row_end]
            if all(start == 0 for start in starts):
                attention_mask = None
            else:
                positions = torch.arange(
                    past_length + 1,
                    dtype=torch.long,
                    device=current_embedding.device,
                )
                starts_tensor = torch.tensor(
                    starts, dtype=torch.long, device=current_embedding.device
                )
                attention_mask = (
                    positions[None, :] >= starts_tensor[:, None]
                )[:, None, None, :]
            for layer_idx, layer in enumerate(self.model.layers):
                residual = hidden_states
                normalized = layer.input_layernorm(hidden_states)
                query, current_key, current_value = self._project_current(
                    layer.self_attn, normalized, run_position_embeddings
                )
                cache_layer = cache.layers[layer_idx]
                # This overwrites only the uncommitted Local slot for Full rows.
                cache_layer.stage_rows(
                    row_start, row_end, current_key, current_value
                )
                key = cache_layer.keys[
                    row_start:row_end, :, : past_length + 1, :
                ]
                value = cache_layer.values[
                    row_start:row_end, :, : past_length + 1, :
                ]
                attention_output = self._attention_output(
                    layer.self_attn, query, key, value, attention_mask
                )
                hidden_states = residual + attention_output
                hidden_states = hidden_states + layer.mlp(
                    layer.post_attention_layernorm(hidden_states)
                )
            selected_hidden[row_start:row_end] = self.model.norm(hidden_states)
        return selected_hidden

    def _record_decode(
        self,
        active_mask: torch.BoolTensor,
        route_full: torch.BoolTensor,
        finite: torch.BoolTensor,
        full_rerun: torch.BoolTensor,
        full_equivalent_reuse: torch.BoolTensor,
    ) -> None:
        local = active_mask & ~route_full
        self._request_route_counts.add_(active_mask.long())
        self._request_full_counts.add_(route_full.long())
        updates = torch.stack(
            (
                active_mask.sum(),
                route_full.sum(),
                local.sum(),
                (active_mask & ~finite).sum(),
                active_mask.sum(),
                route_full.sum(),
                local.sum(),
                route_full.sum(),
                full_rerun.sum(),
                full_equivalent_reuse.sum(),
            )
        ).to(dtype=torch.long)
        self._request_counters.add_(updates)

    def _prompt_padding_starts(
        self, attention_mask: torch.Tensor | None, batch_size: int, length: int
    ) -> list[int]:
        if attention_mask is None:
            return [0] * batch_size
        if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, length):
            raise ValueError("Full prefill requires a 2-D aligned attention mask.")
        mask = attention_mask.detach().to(device="cpu", dtype=torch.bool)
        starts: list[int] = []
        positions = torch.arange(length)
        for row in range(batch_size):
            valid_count = int(mask[row].sum().item())
            if valid_count < 1:
                raise ValueError("Every prompt row must contain at least one token.")
            start = length - valid_count
            expected = positions >= start
            if not torch.equal(mask[row], expected):
                raise ValueError(
                    "Selected-cache runtime accepts only monotonic left padding."
                )
            starts.append(start)
        return starts

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
            if not isinstance(existing, SelectedMixedStaticCache):
                raise TypeError("Only SelectedMixedStaticCache may be passed explicitly.")
            return
        if generation_config.use_cache is False:
            raise ValueError("Selected-cache generation requires use_cache=True.")
        if generation_config.cache_implementation is not None:
            raise ValueError("cache_implementation is fixed by selected-cache modeling.")
        if max_cache_length > int(self.config.max_position_embeddings):
            raise ValueError(
                f"Requested cache length {max_cache_length} exceeds the model limit "
                f"{self.config.max_position_embeddings}; truncation is forbidden."
            )
        model_kwargs["past_key_values"] = SelectedMixedStaticCache(
            int(self.config.num_hidden_layers), int(max_cache_length)
        )

    def generate(self, *args, **kwargs):
        generation_config = kwargs.get("generation_config", self.generation_config)

        def resolved(name, default=None):
            value = kwargs.get(name, getattr(generation_config, name, default))
            return default if value is None else value

        if bool(resolved("do_sample", False)):
            raise ValueError("Selected-cache evaluation requires do_sample=False.")
        if int(resolved("num_beams", 1)) != 1:
            raise ValueError("Selected-cache evaluation rejects beam decoding.")
        if int(resolved("num_return_sequences", 1)) != 1:
            raise ValueError("Selected-cache evaluation requires one return sequence.")
        forbidden = {
            "assistant_model": kwargs.get("assistant_model"),
            "prompt_lookup_num_tokens": resolved("prompt_lookup_num_tokens"),
            "assistant_early_exit": resolved("assistant_early_exit"),
            "penalty_alpha": resolved("penalty_alpha"),
            "constraints": resolved("constraints"),
            "force_words_ids": resolved("force_words_ids"),
        }
        active = {
            name: value
            for name, value in forbidden.items()
            if value is not None and value is not False and value != 0
        }
        if active:
            raise ValueError(
                f"Selected-cache evaluation rejects assisted/contrastive decoding: {active}."
            )
        self._active_threshold = _finite_env_float(
            "QWEN3_SELECTED_CACHE_ROUTER_ACTION_THRESHOLD",
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
            raise ValueError("Production selected-cache modeling does not train or score labels.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds.")
        if use_cache is False:
            raise ValueError("Selected-cache modeling requires use_cache=True.")
        if output_attentions:
            raise ValueError("Selected-cache modeling does not return attentions.")
        if self.config._attn_implementation != "sdpa":
            raise ValueError("Formal selected-cache runtime requires SDPA.")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        batch_size, sequence_length, _ = inputs_embeds.shape
        if past_key_values is None:
            past_key_values = SelectedMixedStaticCache(
                int(self.config.num_hidden_layers),
                int(self.config.max_position_embeddings),
            )
        if not isinstance(past_key_values, SelectedMixedStaticCache):
            raise TypeError("Selected-cache modeling requires SelectedMixedStaticCache.")
        past_length = past_key_values.get_seq_length()
        is_decode = past_length > 0

        if not is_decode:
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
            self._previous_selected_hidden = hidden_states[:, -1, :].detach()
            past_key_values.assert_aligned()
        else:
            if sequence_length != 1:
                raise ValueError("Cached selected-cache decode consumes exactly one token.")
            if self._previous_selected_hidden is None or self._unfinished_mask is None:
                raise RuntimeError("Cached decode is missing Full-prefill request state.")
            if self._prompt_starts is None:
                raise RuntimeError("Cached decode is missing prompt padding metadata.")
            if self._active_mode == "routed":
                self._ensure_router_checkpoint()
            if position_ids is None:
                position_ids = torch.full(
                    (1, 1),
                    past_length,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            if position_ids.shape[0] == 1 and batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)
            active_mask = self._unfinished_mask.clone()
            if router_active_mask is not None:
                if router_active_mask.shape != (batch_size,):
                    raise ValueError("router_active_mask must have shape [batch].")
                active_mask &= router_active_mask.to(
                    device=inputs_embeds.device, dtype=torch.bool
                )
            elif input_ids is not None:
                eos_ids = self.generation_config.eos_token_id
                if eos_ids is None:
                    eos_ids = self.config.eos_token_id
                if eos_ids is not None:
                    eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
                    for eos_id in eos_ids:
                        active_mask &= input_ids[:, -1].ne(int(eos_id))
            self._unfinished_mask = active_mask.detach().clone()

            if _contiguous_runs(active_mask):
                position_embeddings = self._position_embeddings(
                    inputs_embeds, position_ids
                )
                local_hidden = self._local_candidate(
                    inputs_embeds,
                    position_embeddings,
                    past_key_values,
                    past_length,
                    active_mask,
                )
                if self._active_mode == "routed":
                    # Exactly one batched Router invocation for this decode step.
                    score = self.router_head(
                        self._previous_selected_hidden,
                        inputs_embeds[:, 0, :],
                        local_hidden[:, 0, :],
                    )
                    finite = torch.isfinite(score)
                    route_full = active_mask & (
                        (finite & score.gt(self._active_threshold)) | ~finite
                    )
                else:
                    # The checkpoint-independent Local control shares the exact
                    # selected-cache state machine while bypassing Router/Full.
                    score = torch.zeros(
                        batch_size, dtype=torch.float32, device=inputs_embeds.device
                    )
                    finite = torch.ones_like(active_mask)
                    route_full = torch.zeros_like(active_mask)
                selected_hidden = local_hidden.clone()
                # Local is mathematically Full whenever the sink+window union
                # covers the complete valid sequence.  Preserve the Router's
                # logical Full action, but reuse that already-computed exact
                # candidate instead of repeating all decoder layers.
                full_equivalent_reuse = route_full & self._full_equivalent_mask(
                    active_mask, past_length
                )
                full_rerun = route_full & ~full_equivalent_reuse
                full_runs = _contiguous_runs(full_rerun)
                if full_runs:
                    selected_hidden = self._full_candidate(
                        inputs_embeds,
                        position_embeddings,
                        past_key_values,
                        past_length,
                        full_rerun,
                        selected_hidden,
                    )
                past_key_values.commit_one()
                self._previous_selected_hidden = torch.where(
                    active_mask[:, None],
                    selected_hidden[:, 0, :],
                    self._previous_selected_hidden,
                ).detach()
                self._record_decode(
                    active_mask,
                    route_full,
                    finite,
                    full_rerun,
                    full_equivalent_reuse,
                )
                self._last_router_score = score.detach()
                self._last_route_full = route_full.detach()
                self._last_router_active_mask = active_mask.detach()
                hidden_states = selected_hidden
            else:
                hidden_states = self._previous_selected_hidden[:, None, :]
                self._last_router_score = None
                self._last_route_full = None
                self._last_router_active_mask = active_mask.detach()

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
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
    "Qwen3SelectedCacheRouterForCausalLM",
    "Qwen3ThreeTowerInteractionRouterHead",
    "SelectedMixedStaticCache",
    "SelectedMixedStaticLayer",
]
