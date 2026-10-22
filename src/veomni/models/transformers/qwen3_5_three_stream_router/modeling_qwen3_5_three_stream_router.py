"""Faithful Qwen3.5 hybrid three-stream signed-DeltaNLL Router training.

Only the six ordinary attention layers differ between Full and Local:
Full reads the complete same-segment history while Local reads sink tokens
plus a recent window.  The eighteen Gated DeltaNet layers retain their global
compressed history.  At every layer both candidates read the immutable Full
trunk past; their current-token state never becomes another token's training
history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ....distributed.parallel_state import get_parallel_state
from ..qwen3_one_step_router.modeling_qwen3_one_step_router import (
    Qwen3OneStepRouterForCausalLM,
    _set_one_step_config_defaults,
)
from ..qwen3_router.modeling_qwen3_router import (
    Qwen3RouterForCausalLM,
    _build_router_head,
    _check_router_tensor_is_finite,
    _set_router_config_defaults,
    build_router_eligible_mask,
    online_softmax_merge_local_self,
)
from ..qwen3_three_stream_router.modeling_qwen3_three_stream_router import (
    create_three_stream_masks,
)
from .delta_counterfactual import (
    _single_segment_per_row,
    isolated_causal_conv1d,
    three_stream_from_trunk_scan,
)


_FEATURE_SCHEMA = ["history_state", "current_embedding", "current_swa_hidden"]
_EXPECTED_LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
] * 6
_FLEX_GRAPHS: dict[str, object] = {}

_DEFAULTS = {
    "router_schema_version": 8,
    "router_training_history_mode": "full",
    "router_runtime_history_mode": "selected_hybrid_cache",
    "router_local_definition": "window_only_full_attention_layers",
    "router_linear_history_mode": "full_trunk_recurrent_state",
    "router_training_attention_backend": "flex_attention_and_fla_delta",
    "router_equivalent_stream_canonicalization": (
        "full_to_trunk_each_layer_and_local_to_trunk_before_first_windowed_"
        "attention_or_inside_structural_full_context"
    ),
    "router_full_trunk_execution": "official_qwen3_5_token_mixer_path",
    "router_training_sequence_layout": "padded_fast_path_or_segmentwise_packed_reset",
    "router_sink_size": 4,
    "router_window_size": 2048,
    "router_window_includes_current": True,
    "router_feature_schema": _FEATURE_SCHEMA,
    "router_head_architecture": "three_tower_interaction_swiglu",
    "router_hidden_size": 1024,
    "router_num_blocks": 2,
    "router_intermediate_size": 2048,
    "router_expected_head_parameter_count": 28_325_889,
    "router_expected_head_tensor_count": 22,
    "router_history_selector": "local_when_signed_nll_lt_threshold",
    "router_history_nll_threshold": 0.001,
    "router_control_threshold_status": "pending_qwen3_5_prior",
    "router_control_p999_nll_threshold": 0.0,
    "router_one_step_harm_threshold": 0.0,
    "router_one_step_target": "signed_nll",
    "router_one_step_transform": "signed_log1p",
    "router_one_step_nll_scale": 1.0,
    "router_one_step_huber_delta": 1.0,
    "router_one_step_positive_weight": 1.0,
    "router_one_step_sign_aux_weight": 0.0,
    "router_loss_weight": 1.0,
    "router_eligible_only": True,
    "router_return_hidden_states": False,
    "router_check_finite": True,
    "router_kl_chunk_size": 16,
    "router_group_rms_enabled": False,
    "router_learned_fusion_scales": False,
    "router_layer_scale_enabled": False,
}


def _compiled_flex(stream: str, device_type: str):
    if stream not in {"trunk", "full_candidate", "local_candidate"}:
        raise ValueError(f"Unknown Qwen3.5 FlexAttention stream {stream!r}.")
    # The unfused implementation is intentionally retained for CPU numerical
    # tests.  Formal training is CUDA-only and uses the compiled kernel below.
    if device_type != "cuda":
        return flex_attention
    if stream not in _FLEX_GRAPHS:
        _FLEX_GRAPHS[stream] = torch.compile(flex_attention, dynamic=True)
    return _FLEX_GRAPHS[stream]


def _set_qwen3_5_router_defaults(config: Qwen3_5TextConfig) -> None:
    for name, value in _DEFAULTS.items():
        if not hasattr(config, name):
            setattr(config, name, value)
    _set_one_step_config_defaults(config)
    _set_router_config_defaults(config)

    if int(config.router_schema_version) != 8:
        raise ValueError("Qwen3.5 hybrid Router requires schema version 8.")
    if list(config.layer_types) != _EXPECTED_LAYER_TYPES:
        raise ValueError(
            "The Qwen3.5-2B experiment requires the exact 18-DeltaNet/6-Full "
            "layer layout."
        )
    if config.router_training_history_mode != "full":
        raise ValueError("Qwen3.5 one-step supervision requires a Full trunk.")
    if config.router_runtime_history_mode != "selected_hybrid_cache":
        raise ValueError("Qwen3.5 runtime must commit the selected hybrid cache.")
    if config.router_local_definition != "window_only_full_attention_layers":
        raise ValueError("Unknown Qwen3.5 Local definition.")
    if config.router_linear_history_mode != "full_trunk_recurrent_state":
        raise ValueError("Candidate DeltaNet layers must read Full-trunk recurrent history.")
    if config.router_training_attention_backend != "flex_attention_and_fla_delta":
        raise ValueError("Formal Qwen3.5 training requires FlexAttention and FLA DeltaRule.")
    if config.router_equivalent_stream_canonicalization != (
        "full_to_trunk_each_layer_and_local_to_trunk_before_first_windowed_"
        "attention_or_inside_structural_full_context"
    ):
        raise ValueError("Unknown Qwen3.5 equivalent-stream canonicalization rule.")
    if config.router_full_trunk_execution != "official_qwen3_5_token_mixer_path":
        raise ValueError("Full trunk must execute the official Qwen3.5 token mixer path.")
    if config.router_training_sequence_layout != (
        "padded_fast_path_or_segmentwise_packed_reset"
    ):
        raise ValueError("Unknown Qwen3.5 training sequence-layout contract.")
    if not bool(config.router_window_includes_current):
        raise ValueError("The configured Local window must include the current token.")
    if config.router_head_architecture != "three_tower_interaction_swiglu":
        raise ValueError("The matched experiment requires the A0 three-tower head.")
    if list(config.router_feature_schema) != _FEATURE_SCHEMA:
        raise ValueError(f"Qwen3.5 Router features must be {_FEATURE_SCHEMA}.")
    if config.router_history_selector != "local_when_signed_nll_lt_threshold":
        raise ValueError("Unknown selected-history rule.")
    if not math.isfinite(float(config.router_history_nll_threshold)):
        raise ValueError("History threshold must be finite.")
    if config.router_one_step_target != "signed_nll":
        raise ValueError("The matched experiment predicts signed DeltaNLL.")
    if config.router_one_step_transform != "signed_log1p":
        raise ValueError("The matched experiment uses signed-log1p targets.")
    if any(
        (
            bool(config.router_group_rms_enabled),
            bool(config.router_learned_fusion_scales),
            bool(config.router_layer_scale_enabled),
            float(config.router_one_step_sign_aux_weight) != 0.0,
        )
    ):
        raise ValueError("The A0 matched experiment cannot enable head regularizers.")

    observed_hash = qwen3_5_three_stream_schema_sha256(config)
    configured_hash = getattr(config, "router_schema_sha256", None)
    if configured_hash not in {None, "", observed_hash}:
        raise ValueError(
            f"Qwen3.5 Router schema hash mismatch: {configured_hash} != {observed_hash}."
        )
    config.router_schema_sha256 = observed_hash


def qwen3_5_three_stream_schema_payload(config: Qwen3_5TextConfig) -> dict:
    full_layers = [
        index for index, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"
    ]
    linear_layers = [
        index for index, layer_type in enumerate(config.layer_types) if layer_type == "linear_attention"
    ]
    payload = {
        "format": "qwen3.5-hybrid-three-stream-nll-selected-router-v8",
        "schema_version": int(config.router_schema_version),
        "base_model": {
            "path": str(getattr(config, "router_base_model_path", "")),
            "config_sha256": str(getattr(config, "router_base_config_sha256", "")),
            "weights_manifest_sha256": str(
                getattr(config, "router_base_weights_manifest_sha256", "")
            ),
            "text_only": True,
            "vision_loaded": False,
            "mtp_loaded": False,
        },
        "backbone": {
            "hidden_size": int(config.hidden_size),
            "num_layers": int(config.num_hidden_layers),
            "linear_attention_layers": linear_layers,
            "full_attention_layers": full_layers,
            "linear_history": str(config.router_linear_history_mode),
        },
        "local": {
            "definition": str(config.router_local_definition),
            "sink_size": int(config.router_sink_size),
            "window_size_including_current": int(config.router_window_size),
            "window_includes_current": bool(config.router_window_includes_current),
            "full_attention_layers_only": full_layers,
            "deltanet_global_recurrent_memory_retained": True,
        },
        "streams": {
            "trunk": "always_full",
            "full_trunk_execution": str(config.router_full_trunk_execution),
            "full": "current_token_full_on_full_trunk_past",
            "local": "current_token_windowed_attention_on_full_trunk_past",
            "candidate_attention_kv_committed": False,
            "candidate_delta_conv_recurrent_state_committed": False,
            "delta_parallelization": "shifted_query_of_previous_trunk_state",
            "equivalent_stream_canonicalization": str(
                config.router_equivalent_stream_canonicalization
            ),
        },
        "features": {
            "schema": list(config.router_feature_schema),
            "history_selector": str(config.router_history_selector),
            "history_nll_threshold": float(config.router_history_nll_threshold),
            "right_shift": 1,
        },
        "head": {
            "architecture": str(config.router_head_architecture),
            "hidden_size": int(config.router_hidden_size),
            "num_blocks": int(config.router_num_blocks),
            "intermediate_size": int(config.router_intermediate_size),
            "expected_parameter_count": int(config.router_expected_head_parameter_count),
            "expected_tensor_count": int(config.router_expected_head_tensor_count),
            "fp32": True,
        },
        "target": {
            "name": "signed_nll",
            "definition": "nll_local_next_token_minus_nll_full_next_token",
            "transform": str(config.router_one_step_transform),
            "scale": float(config.router_one_step_nll_scale),
            "huber_delta": float(config.router_one_step_huber_delta),
            "action_threshold_transformed_score": 0.0,
            "control_threshold_status": str(config.router_control_threshold_status),
        },
        "alignment": {
            "next_token_offset": 1,
            "same_segment_required": True,
            "segment_first_history": "zero_and_excluded",
            "sequence_last_excluded": True,
            "structural_context_length_gt": int(config.router_sink_size)
            + int(config.router_window_size),
            "training_sequence_layout": str(config.router_training_sequence_layout),
            "packed_deltanet_reset": "official_trunk_executed_per_segment",
        },
    }
    # Optional input attribution must change the structural checkpoint contract.
    # The legacy payload remains byte-for-byte identical when this is disabled.
    from ..qwen3_router.recall_features import recall_schema, validate_recall_config
    if validate_recall_config(config) is not None:
        payload["recall_input_ablation"] = recall_schema(config)
        payload["recall_input_ablation"]["decision_timing"] = "after_current_local_forward"
    return payload


def qwen3_5_three_stream_schema_sha256(config: Qwen3_5TextConfig) -> str:
    encoded = json.dumps(
        qwen3_5_three_stream_schema_payload(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class Qwen3_5ThreeStreamRouterOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    router_loss: Optional[torch.FloatTensor] = None
    router_transformed_score: Optional[torch.FloatTensor] = None
    router_targets: Optional[torch.FloatTensor] = None
    router_raw_target: Optional[torch.FloatTensor] = None
    router_predicted_raw_target: Optional[torch.FloatTensor] = None
    full_token_nll: Optional[torch.FloatTensor] = None
    local_token_nll: Optional[torch.FloatTensor] = None
    history_local_selected_mask: Optional[torch.BoolTensor] = None
    training_metrics: Optional[dict] = None
    training_probe_tensors: Optional[dict] = None
    trunk_hidden_states: Optional[torch.FloatTensor] = None
    full_candidate_hidden_states: Optional[torch.FloatTensor] = None
    local_candidate_hidden_states: Optional[torch.FloatTensor] = None


class Qwen3_5ThreeStreamDeltaNet(Qwen3_5GatedDeltaNet):
    """One Full trunk and two isolated-current DeltaNet candidates."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        # Inherit the official module directly so pretrained names remain
        # `model.layers.N.linear_attn.*` with no wrapper-only `.delta` prefix.
        super().__init__(config, layer_idx)

    def _project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(
            *hidden_states.shape[:2], self.num_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        log_decay = -self.A_log.float().exp() * torch.nn.functional.softplus(
            a.float() + self.dt_bias
        )
        return raw_qkv, z, beta, log_decay

    def _split_qkv(
        self, convolved: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = torch.split(
            convolved,
            (self.key_dim, self.key_dim, self.value_dim),
            dim=-1,
        )
        shape = (*convolved.shape[:2], -1)
        query = query.reshape(*shape, self.head_k_dim)
        key = key.reshape(*shape, self.head_k_dim)
        value = value.reshape(*shape, self.head_v_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        return query, key, value

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        full_hidden: torch.Tensor,
        local_hidden: torch.Tensor,
        position_ids: torch.LongTensor,
        valid_mask: torch.BoolTensor,
        single_segment_per_row: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = trunk_hidden.shape
        # The Full trunk is the frozen upstream model, so execute its token
        # mixer through the official method.  Candidate scans below are
        # additional counterfactual work and must never redefine the baseline.
        if single_segment_per_row:
            trunk_output = super().forward(
                hidden_states=trunk_hidden,
                cache_params=None,
                attention_mask=valid_mask,
            )
        else:
            # The official Qwen3.5 fallback convolution does not consume
            # seq_idx, so a single packed call could leak convolution and
            # recurrent state across segment boundaries.  Packed layouts are
            # uncommon in the formal padded protocol; execute each segment
            # independently to preserve the official operation exactly.
            trunk_output = torch.zeros_like(trunk_hidden)
            positions_cpu = position_ids.detach().cpu()
            valid_cpu = valid_mask.detach().cpu()
            for row in range(batch_size):
                index = 0
                while index < sequence_length:
                    if not bool(valid_cpu[row, index]):
                        index += 1
                        continue
                    if int(positions_cpu[row, index]) != 0:
                        raise ValueError(
                            "Every packed Qwen3.5 segment must start at position zero."
                        )
                    end = index + 1
                    while (
                        end < sequence_length
                        and bool(valid_cpu[row, end])
                        and int(positions_cpu[row, end])
                        == int(positions_cpu[row, end - 1]) + 1
                    ):
                        end += 1
                    trunk_output[row : row + 1, index:end] = super().forward(
                        hidden_states=trunk_hidden[row : row + 1, index:end],
                        cache_params=None,
                        attention_mask=None,
                    )
                    index = end
        projected = [self._project(stream) for stream in (trunk_hidden, full_hidden, local_hidden)]
        trunk_raw, trunk_z, trunk_beta, trunk_decay = projected[0]
        full_raw, full_z, full_beta, full_decay = projected[1]
        local_raw, local_z, local_beta, local_decay = projected[2]

        trunk_conv, full_conv = isolated_causal_conv1d(
            trunk_raw,
            full_raw,
            self.conv1d.weight,
            position_ids,
            valid_mask,
            self.conv1d.bias,
            single_segment_per_row,
        )
        _, local_conv = isolated_causal_conv1d(
            trunk_raw,
            local_raw,
            self.conv1d.weight,
            position_ids,
            valid_mask,
            self.conv1d.bias,
            single_segment_per_row,
        )
        trunk_q, trunk_k, trunk_v = self._split_qkv(trunk_conv)
        full_q, full_k, full_v = self._split_qkv(full_conv)
        local_q, local_k, local_v = self._split_qkv(local_conv)

        _trunk_core, full_core, local_core = three_stream_from_trunk_scan(
            self.chunk_gated_delta_rule,
            trunk_q,
            trunk_k,
            trunk_v,
            trunk_decay,
            trunk_beta,
            full_q,
            full_k,
            full_v,
            full_decay,
            full_beta,
            local_q,
            local_k,
            local_v,
            local_decay,
            local_beta,
            position_ids,
            valid_mask,
            single_segment_per_row,
        )

        outputs = [trunk_output]
        for core, z in (
            (full_core, full_z),
            (local_core, local_z),
        ):
            core = self.norm(
                core.reshape(-1, self.head_v_dim),
                z.reshape(-1, self.head_v_dim),
            ).reshape(batch_size, sequence_length, self.value_dim)
            outputs.append(self.out_proj(core))
        return tuple(outputs)


class Qwen3_5ThreeStreamFullAttention(Qwen3_5Attention):
    """Full trunk plus Full/Local current-token attention candidates."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        # Preserve the official `self_attn.*` parameter namespace for exact
        # text-branch checkpoint loading.
        super().__init__(config, layer_idx)

    def _project(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                *input_shape, -1, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        return query, key, value, gate

    def _finish(self, output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        output = output.transpose(1, 2).reshape(gate.shape).contiguous()
        return self.o_proj(output * torch.sigmoid(gate))

    def _trunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        mask: BlockMask,
        valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        output = _compiled_flex("trunk", query.device.type)(
            query,
            key,
            value,
            block_mask=mask,
            scale=self.scaling,
            enable_gqa=self.num_key_value_groups > 1,
            kernel_options={"ROWS_GUARANTEED_SAFE": False},
        )
        output = output.masked_fill(~valid_mask[:, None, :, None], 0.0)
        return self._finish(output, gate)

    def _candidate(
        self,
        query: torch.Tensor,
        trunk_key: torch.Tensor,
        trunk_value: torch.Tensor,
        candidate_key: torch.Tensor,
        candidate_value: torch.Tensor,
        gate: torch.Tensor,
        past_mask: BlockMask,
        position_ids: torch.LongTensor,
        valid_mask: torch.BoolTensor,
        stream: str,
    ) -> torch.Tensor:
        past_output, past_lse = _compiled_flex(stream, query.device.type)(
            query,
            trunk_key,
            trunk_value,
            block_mask=past_mask,
            scale=self.scaling,
            enable_gqa=self.num_key_value_groups > 1,
            return_lse=True,
            kernel_options={"ROWS_GUARANTEED_SAFE": False},
        )
        candidate_key = repeat_kv(candidate_key, self.num_key_value_groups)
        candidate_value = repeat_kv(candidate_value, self.num_key_value_groups)
        self_score = (
            query.float() * candidate_key.float()
        ).sum(dim=-1) * self.scaling
        output, _ = online_softmax_merge_local_self(
            past_output,
            past_lse,
            self_score,
            candidate_value,
            empty_past_mask=~(valid_mask & position_ids.gt(0)),
            valid_token_mask=valid_mask,
        )
        return self._finish(output, gate)

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        full_hidden: torch.Tensor,
        local_hidden: torch.Tensor,
        trunk_mask: BlockMask,
        full_past_mask: BlockMask,
        local_past_mask: BlockMask,
        position_ids: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        valid_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trunk_q, trunk_k, trunk_v, trunk_gate = self._project(
            trunk_hidden, position_embeddings
        )
        full_q, full_k, full_v, full_gate = self._project(
            full_hidden, position_embeddings
        )
        local_q, local_k, local_v, local_gate = self._project(
            local_hidden, position_embeddings
        )
        if trunk_hidden.is_cuda:
            trunk_output, _ = super().forward(
                hidden_states=trunk_hidden,
                position_embeddings=position_embeddings,
                attention_mask=trunk_mask,
                past_key_values=None,
            )
        else:
            # CPU contract tests use FlexAttention directly because the
            # upstream capability declaration intentionally rejects Flex at
            # model construction time.
            trunk_output = self._trunk(
                trunk_q, trunk_k, trunk_v, trunk_gate, trunk_mask, valid_mask
            )
        full_output = self._candidate(
            full_q,
            trunk_k,
            trunk_v,
            full_k,
            full_v,
            full_gate,
            full_past_mask,
            position_ids,
            valid_mask,
            "full_candidate",
        )
        local_output = self._candidate(
            local_q,
            trunk_k,
            trunk_v,
            local_k,
            local_v,
            local_gate,
            local_past_mask,
            position_ids,
            valid_mask,
            "local_candidate",
        )
        return trunk_output, full_output, local_output


class Qwen3_5ThreeStreamDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5ThreeStreamDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen3_5ThreeStreamFullAttention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.structural_full_context = int(config.router_sink_size) + int(
            config.router_window_size
        )
        # Local cannot differ before the first layer that actually applies a
        # window.  The isolated-current DeltaRule is mathematically identical
        # to the trunk in those leading layers, but BF16 shifted-query and scan
        # reduction orders need not be bitwise identical.  Record this
        # structural fact explicitly so that numerical noise cannot become a
        # false signed-DeltaNLL label.
        self.local_action_has_occurred = any(
            layer_type == "full_attention"
            for layer_type in config.layer_types[: layer_idx + 1]
        )

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        full_hidden: torch.Tensor,
        local_hidden: torch.Tensor,
        trunk_mask: BlockMask,
        full_past_mask: BlockMask,
        local_past_mask: BlockMask,
        position_ids: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        valid_mask: torch.BoolTensor,
        single_segment_per_row: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residuals = (trunk_hidden, full_hidden, local_hidden)
        normalized = tuple(self.input_layernorm(value) for value in residuals)
        if self.layer_type == "linear_attention":
            mixed = self.linear_attn(
                *normalized,
                position_ids,
                valid_mask,
                single_segment_per_row,
            )
        else:
            mixed = self.self_attn(
                *normalized,
                trunk_mask,
                full_past_mask,
                local_past_mask,
                position_ids,
                position_embeddings,
                valid_mask,
            )
        outputs = []
        for residual, update in zip(residuals, mixed):
            hidden = residual + update
            hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
            outputs.append(hidden.masked_fill(~valid_mask[..., None], 0.0))
        trunk_output, _full_output, local_output = outputs

        # In exact arithmetic the Full counterfactual is the Full trunk: both
        # read the same immutable past and execute the same current action.
        # Likewise Local is identical while sink+window still covers the whole
        # segment.  The shifted-query DeltaRule and split past/self attention
        # use different BF16 reduction orders, and the resulting round-off can
        # otherwise be amplified by later recurrent layers into a spurious
        # training label.  Canonicalize only these mathematically identical
        # regions after every layer; the genuinely truncated Local tail remains
        # untouched and still follows its isolated counterfactual recursion.
        full_output = trunk_output
        if not self.local_action_has_occurred:
            local_output = trunk_output
        else:
            structural_equivalent = valid_mask & position_ids.add(1).le(
                self.structural_full_context
            )
            local_output = torch.where(
                structural_equivalent[..., None], trunk_output, local_output
            )
        return trunk_output, full_output, local_output


class Qwen3_5ThreeStreamModel(Qwen3_5PreTrainedModel):
    """Frozen text-only Qwen3.5 trunk and isolated current candidates."""

    config: Qwen3_5TextConfig
    _no_split_modules = ["Qwen3_5ThreeStreamDecoderLayer"]
    _supports_flex_attn = True

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            Qwen3_5ThreeStreamDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        del kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds.")
        if past_key_values is not None or use_cache:
            raise ValueError("Three-stream training does not use a decode cache.")
        if output_attentions or output_hidden_states:
            raise ValueError("Three-stream training does not retain per-layer outputs.")
        if get_parallel_state().sp_enabled:
            raise ValueError("Qwen3.5 three-stream training requires sequence parallel size 1.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, sequence_length, _ = inputs_embeds.shape
        if attention_mask is None:
            valid_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        else:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("Qwen3.5 three-stream attention mask must be [B,T].")
            valid_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        if cache_position is None:
            cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1 and batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)
        if position_ids.shape != (batch_size, sequence_length):
            raise ValueError("Qwen3.5 three-stream position_ids must be [B,T].")
        position_ids = position_ids.to(device=inputs_embeds.device, dtype=torch.long)
        single_segment_layout = _single_segment_per_row(position_ids, valid_mask)

        trunk_mask, full_past_mask, local_past_mask = create_three_stream_masks(
            position_ids,
            valid_mask,
            int(self.config.router_sink_size),
            int(self.config.router_window_size),
            "full",
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        trunk_hidden = inputs_embeds
        full_hidden = inputs_embeds
        local_hidden = inputs_embeds
        for index, layer in enumerate(self.layers):
            trunk_hidden, full_hidden, local_hidden = layer(
                trunk_hidden,
                full_hidden,
                local_hidden,
                trunk_mask,
                full_past_mask,
                local_past_mask,
                position_ids,
                position_embeddings,
                valid_mask,
                single_segment_layout,
            )
            if self.config.router_check_finite:
                for name, value in (
                    ("trunk", trunk_hidden),
                    ("full_candidate", full_hidden),
                    ("local_candidate", local_hidden),
                ):
                    _check_router_tensor_is_finite(f"layer_{index}.{name}", value)
        return (
            self.norm(trunk_hidden),
            self.norm(full_hidden),
            self.norm(local_hidden),
            inputs_embeds,
            position_ids,
            valid_mask,
        )


class Qwen3_5ThreeStreamNllSelectedRouterForCausalLM(Qwen3_5PreTrainedModel):
    """Text-only Qwen3.5 A0 Router trained with signed-log1p Huber."""

    config: Qwen3_5TextConfig
    _no_split_modules = ["Qwen3_5ThreeStreamDecoderLayer"]
    _supports_flex_attn = True
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _keys_to_ignore_on_load_unexpected = [r"^mtp\.", r"^model\.visual\."]
    _checkpoint_conversion_mapping = {r"^model\.language_model\.": "model."}

    # Reuse the already-qualified objective implementation byte-for-byte.
    _aligned_next_token_labels = staticmethod(
        Qwen3OneStepRouterForCausalLM._aligned_next_token_labels
    )
    _compute_token_nll_only = Qwen3OneStepRouterForCausalLM._compute_token_nll_only
    _target_scale = Qwen3OneStepRouterForCausalLM._target_scale
    _transform_target = Qwen3OneStepRouterForCausalLM._transform_target
    _inverse_target = Qwen3OneStepRouterForCausalLM._inverse_target
    _compute_one_step_loss = Qwen3OneStepRouterForCausalLM._compute_one_step_loss
    _distributed_denominator = staticmethod(Qwen3RouterForCausalLM._distributed_denominator)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        # The matched A0 head is intentionally reused byte-for-byte from the
        # Qwen3 experiment and therefore contains Qwen3RMSNorm (unit-centered),
        # while the Qwen3.5 backbone uses its distinct zero-centered RMSNorm.
        # VeOmni materializes missing Router parameters after loading the text
        # checkpoint, so this explicit branch is required to keep those norms
        # deterministic instead of leaving their empty-storage values undefined.
        if isinstance(module, Qwen3RMSNorm):
            module.weight.fill_(1.0)

    def __init__(self, config: Qwen3_5TextConfig):
        _set_qwen3_5_router_defaults(config)
        super().__init__(config)
        self.model = Qwen3_5ThreeStreamModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_head = _build_router_head(config)
        self.post_init()
        self.router_head.float()
        self.requires_grad_(False)
        self.router_head.requires_grad_(True)

        parameter_count = sum(value.numel() for value in self.router_head.parameters())
        tensor_count = len(self.router_head.state_dict())
        self.config.router_head_parameter_count = int(parameter_count)
        self.config.router_head_tensor_count = int(tensor_count)
        if parameter_count != int(config.router_expected_head_parameter_count):
            raise RuntimeError(
                f"Qwen3.5 A0 Router has {parameter_count} parameters; expected "
                f"{config.router_expected_head_parameter_count}."
            )
        if tensor_count != int(config.router_expected_head_tensor_count):
            raise RuntimeError(
                f"Qwen3.5 A0 Router has {tensor_count} tensors; expected "
                f"{config.router_expected_head_tensor_count}."
            )

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse=recurse)
        if hasattr(self, "router_head"):
            self.router_head.float()
        return module

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def gradient_checkpointing_enable(self, *args, **kwargs):
        del args, kwargs

    def _get_router_loss_mask(
        self,
        valid_mask: torch.BoolTensor,
        labels: torch.LongTensor,
        position_ids: torch.LongTensor,
    ) -> torch.BoolTensor:
        return build_router_eligible_mask(
            valid_mask,
            labels,
            position_ids,
            int(self.config.router_sink_size),
            int(self.config.router_window_size),
        )

    def _build_features(
        self,
        history_source: torch.Tensor,
        embeddings: torch.Tensor,
        local_hidden: torch.Tensor,
        position_ids: torch.LongTensor,
        valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        history = torch.zeros_like(history_source)
        history[:, 1:] = history_source[:, :-1]
        history = history.masked_fill(position_ids.eq(0)[..., None], 0.0)
        from ..qwen3_router.recall_features import (
            local_confidence_features, pack_recall_features, validate_recall_config,
        )
        variant = validate_recall_config(self.config)
        if variant is not None:
            confidence = None
            if variant in ("hec", "hesc"):
                # Only Local hidden and the frozen LM head participate; reference
                # labels and Full outputs never enter this confidence computation.
                # Both arms still execute the complete current Local candidate.
                flat = local_hidden.detach().reshape(-1, local_hidden.shape[-1])
                parts = []
                with torch.no_grad():
                    for chunk in flat.split(int(self.config.router_kl_chunk_size)):
                        logits = nn.functional.linear(chunk, self.lm_head.weight.detach())
                        parts.append(local_confidence_features(logits))
                confidence = torch.cat(parts).reshape(*local_hidden.shape[:-1], -1)
            return pack_recall_features(
                history, embeddings, local_hidden, confidence, variant,
            ).masked_fill(~valid_mask[..., None], 0.0)
        return torch.cat((history, embeddings, local_hidden), dim=-1).masked_fill(
            ~valid_mask[..., None], 0.0
        )

    def _select_history(
        self,
        trunk_hidden: torch.Tensor,
        local_hidden: torch.Tensor,
        raw_target: torch.Tensor,
        loss_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        selected_local = (
            loss_mask
            & torch.isfinite(raw_target)
            & raw_target.detach().lt(float(self.config.router_history_nll_threshold))
        )
        source = torch.where(
            selected_local[..., None], local_hidden.detach(), trunk_hidden.detach()
        )
        return source, selected_local

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Qwen3_5ThreeStreamRouterOutput:
        del logits_to_keep
        if labels is None:
            raise ValueError("Qwen3.5 three-stream supervision requires labels.")
        (
            trunk_hidden,
            full_hidden,
            local_hidden,
            embeddings,
            position_ids,
            valid_mask,
        ) = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        full_nll, local_nll = self._compute_token_nll_only(
            full_hidden, local_hidden, labels, valid_mask, position_ids
        )
        raw_target = local_nll - full_nll
        loss_mask = self._get_router_loss_mask(valid_mask, labels, position_ids)
        if not torch.isfinite(raw_target[loss_mask]).all():
            raise FloatingPointError("Qwen3.5 signed-DeltaNLL target is non-finite.")
        transformed_target = self._transform_target(raw_target)
        safe_target = torch.where(
            loss_mask, transformed_target, torch.zeros_like(transformed_target)
        )
        history_source, selected_local = self._select_history(
            trunk_hidden, local_hidden, raw_target, loss_mask
        )
        features = self._build_features(
            history_source, embeddings, local_hidden, position_ids, valid_mask
        )
        transformed_score = self.router_head(features)
        if transformed_score.dtype != torch.float32:
            raise TypeError("Qwen3.5 A0 Router output must remain FP32.")
        router_loss, metrics = self._compute_one_step_loss(
            transformed_score, raw_target, safe_target, loss_mask
        )
        loss = router_loss * float(self.config.router_loss_weight)
        predicted_raw = self._inverse_target(transformed_score)
        if return_dict is False:
            return loss, transformed_score, safe_target, raw_target

        selected_count = selected_local.float().sum()
        eligible_count = loss_mask.float().sum()
        selection_stats = torch.stack((selected_count, eligible_count))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(selection_stats)
        local_history_rate = selection_stats[0] / selection_stats[1].clamp_min(1.0)
        training_metrics = {
            "router/objective/total_loss": metrics["huber_loss"],
            "router/objective/huber_loss": metrics["huber_loss"],
            "router/regression/signed_nll_mean": metrics["raw_target_mean"],
            "router/regression/transformed_mae": metrics["transformed_mae"],
            "router/regression/transformed_rmse": metrics["transformed_rmse"],
            "router/sign0/target_full_rate": metrics["sign_target_rate"],
            "router/sign0/predicted_full_rate": metrics[
                "sign_predicted_positive_rate"
            ],
            "router/sign0/accuracy": metrics["sign_accuracy"],
            "router/sign0/precision": metrics["sign_precision"],
            "router/sign0/recall": metrics["sign_recall"],
            "router/sign0/f1": metrics["sign_f1"],
            "router/history/local_selected_rate": local_history_rate,
            "router/history/full_selected_rate": 1.0 - local_history_rate,
            "router/history/selection_threshold": loss.new_tensor(
                float(self.config.router_history_nll_threshold)
            ),
        }
        probes = {
            "router_transformed_score": transformed_score.detach(),
            "router_targets": safe_target.detach(),
            "router_raw_target": torch.where(
                loss_mask, raw_target, torch.zeros_like(raw_target)
            ).detach(),
            "router_predicted_raw_target": predicted_raw.detach(),
            "router_feature_l2": features.detach().float().norm(dim=-1),
            "full_token_nll": full_nll.detach(),
            "local_token_nll": local_nll.detach(),
            "position_ids": position_ids.detach(),
            "valid_token_mask": loss_mask.detach(),
            "history_local_selected_mask": selected_local.detach(),
            **({"input_ids": input_ids.detach()} if input_ids is not None else {}),
        }
        return Qwen3_5ThreeStreamRouterOutput(
            loss=loss,
            router_loss=router_loss,
            router_transformed_score=transformed_score,
            router_targets=safe_target,
            router_raw_target=raw_target,
            router_predicted_raw_target=predicted_raw,
            full_token_nll=full_nll,
            local_token_nll=local_nll,
            history_local_selected_mask=selected_local,
            training_metrics=training_metrics,
            training_probe_tensors=probes,
            trunk_hidden_states=(
                trunk_hidden if bool(self.config.router_return_hidden_states) else None
            ),
            full_candidate_hidden_states=(
                full_hidden if bool(self.config.router_return_hidden_states) else None
            ),
            local_candidate_hidden_states=(
                local_hidden if bool(self.config.router_return_hidden_states) else None
            ),
        )


ModelClass = Qwen3_5ThreeStreamNllSelectedRouterForCausalLM


__all__ = [
    "ModelClass",
    "Qwen3_5ThreeStreamNllSelectedRouterForCausalLM",
    "Qwen3_5ThreeStreamRouterOutput",
    "qwen3_5_three_stream_schema_payload",
    "qwen3_5_three_stream_schema_sha256",
]
