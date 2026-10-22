"""Three-stream Qwen3 supervision for a Local-first signed-NLL Router.

The training history is a fixed Always-Full or Always-SWA trunk.  At every
layer, two ephemeral current-token streams attend only to that trunk's past:
one with Full attention and one with sink-plus-window attention.  Candidate
K/V never become another token's history.  This makes every token's label a
one-action counterfactual under an explicitly selected training history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack

from ....distributed.parallel_state import get_parallel_state
from ..qwen3_one_step_router.modeling_qwen3_one_step_router import (
    Qwen3OneStepRouterForCausalLM,
    _set_one_step_config_defaults,
)
from ..qwen3_router.modeling_qwen3_router import (
    KwargsForCausalLM,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RouterAttention,
    Qwen3RouterPreTrainedModel,
    Qwen3RotaryEmbedding,
    _build_router_head,
    _check_router_tensor_is_finite,
    _set_router_config_defaults,
    online_softmax_merge_local_self,
    raw_flex_attention,
    repeat_kv,
)


_THREE_STREAM_DEFAULTS = {
    "router_schema_version": 2,
    "router_training_history_mode": "full",
    "router_runtime_history_mode": "selected",
    "router_training_attention_backend": "flex_attention",
    "router_swa_all_layers": True,
    "router_window_includes_current": True,
    "router_feature_schema": ["history_state", "current_embedding", "current_swa_hidden"],
    "router_head_architecture": "three_tower_interaction_swiglu",
    "router_hidden_size": 1024,
    "router_num_blocks": 2,
    "router_intermediate_size": 2048,
    "router_expected_head_parameter_count": 28_325_889,
    "router_expected_head_tensor_count": 22,
}

_THREE_STREAM_COMPILED_FLEX: dict[str, object] = {}


def _get_three_stream_compiled_flex(stream: str):
    """Keep distinct dynamic graphs for trunk, Full-past, and SWA-past calls."""

    if stream not in {"trunk", "full_candidate", "swa_candidate"}:
        raise ValueError(f"Unknown three-stream FlexAttention graph {stream!r}.")
    if raw_flex_attention is None:
        raise RuntimeError("Three-stream Qwen3 training requires PyTorch FlexAttention.")
    if stream not in _THREE_STREAM_COMPILED_FLEX:
        _THREE_STREAM_COMPILED_FLEX[stream] = torch.compile(
            raw_flex_attention, dynamic=True
        )
    return _THREE_STREAM_COMPILED_FLEX[stream]


def three_stream_schema_payload(config: Qwen3Config) -> dict:
    """Return the complete checkpoint/runtime compatibility contract."""

    payload = {
        "format": "qwen3-three-stream-router-schema-v2",
        "schema_version": int(config.router_schema_version),
        "base_model": {
            "path": str(getattr(config, "router_base_model_path", "")),
            "weights_manifest_sha256": str(
                getattr(config, "router_base_weights_manifest_sha256", "")
            ),
            "config_sha256": str(getattr(config, "router_base_config_sha256", "")),
        },
        "features": list(config.router_feature_schema),
        "head": {
            "architecture": str(config.router_head_architecture),
            "hidden_size": int(config.router_hidden_size),
            "num_blocks": int(config.router_num_blocks),
            "intermediate_size": int(config.router_intermediate_size),
            "expected_parameter_count": int(config.router_expected_head_parameter_count),
            "tower_projection_bias": True,
            "fusion_projection_bias": True,
            "block_linears_bias": False,
            "output_projection_bias": True,
            "pre_rmsnorm": True,
        },
        "history": {
            "training": str(config.router_training_history_mode),
            "runtime": str(config.router_runtime_history_mode),
        },
        "swa": {
            "sink_size": int(config.router_sink_size),
            "window_size_including_current": int(config.router_window_size),
            "all_layers": bool(config.router_swa_all_layers),
            "window_includes_current": bool(config.router_window_includes_current),
            "native_sliding_cache": False,
            "retain_complete_selected_kv": True,
            "training_attention_backend": str(config.router_training_attention_backend),
        },
        "target": {
            "name": str(config.router_one_step_target),
            "transform": str(config.router_one_step_transform),
            "scale": float(config.router_one_step_nll_scale),
            "huber_delta": float(config.router_one_step_huber_delta),
            "route_full_when_transformed_score_gt": 0.0,
        },
        "alignment": {
            "next_token_offset": 1,
            "same_packed_segment_required": True,
            "segment_first_history_reset": "zero_and_excluded",
            "sequence_last_excluded": True,
            "structural_context_length_gt": int(config.router_sink_size)
            + int(config.router_window_size),
        },
    }
    if int(config.router_schema_version) == 3:
        payload["format"] = "qwen3-three-stream-router-schema-v3"
        payload["head"].update(
            {
                "expected_tensor_count": int(
                    config.router_expected_head_tensor_count
                ),
                "group_rms_enabled": bool(config.router_group_rms_enabled),
                "group_rms_eps": float(config.router_group_rms_eps),
                "learned_fusion_scales": bool(
                    config.router_learned_fusion_scales
                ),
                "fusion_scale_init": float(config.router_fusion_scale_init),
                "layer_scale_enabled": bool(config.router_layer_scale_enabled),
                "layer_scale_init": float(config.router_layer_scale_init),
            }
        )
        payload["target"].update(
            {
                "sign_aux_weight": float(
                    config.router_one_step_sign_aux_weight
                ),
                "sign_aux_temperature": float(
                    config.router_one_step_sign_aux_temperature
                ),
                "sign_aux_zero_policy": str(
                    config.router_one_step_sign_aux_zero_policy
                ),
                "sign_near_zero_threshold": float(
                    config.router_one_step_sign_near_zero_threshold
                ),
            }
        )
    elif int(config.router_schema_version) == 4:
        payload["format"] = "qwen3-three-stream-nll-selected-router-schema-v4"
        payload["features"] = {
            "schema": list(config.router_feature_schema),
            "history_selector": str(config.router_history_selector),
            "history_nll_threshold": float(config.router_history_nll_threshold),
            "local_rule": "g_i < history_nll_threshold",
            "default_branch": "full_trunk",
            "tie_branch": "full_trunk",
            "right_shift": 1,
        }
        payload["head"].update(
            {
                "expected_tensor_count": int(
                    config.router_expected_head_tensor_count
                ),
                "group_rms_enabled": bool(config.router_group_rms_enabled),
                "learned_fusion_scales": bool(
                    config.router_learned_fusion_scales
                ),
                "layer_scale_enabled": bool(config.router_layer_scale_enabled),
                "fp32": True,
            }
        )
        payload["history"].update(
            {
                "runtime_kv_commit": "full",
                "training_trunk": "always_full",
                "selected_hidden_only_router_feature": True,
            }
        )
        payload["streams"] = {
            "trunk": "always_full",
            "full_counterfactual": "current_token_full_on_full_trunk_past",
            "swa_counterfactual": "current_token_swa_on_full_trunk_past",
            "counterfactual_kv_committed_to_history": False,
        }
        payload["target"].update(
            {
                "full_source": "full_counterfactual",
                "local_source": "swa_counterfactual",
                "sign_aux_weight": float(
                    config.router_one_step_sign_aux_weight
                ),
                "control_p999_nll_threshold": float(
                    config.router_control_p999_nll_threshold
                ),
            }
        )
    if getattr(config, "router_recall_input_version", None) is not None:
        from ..qwen3_router.recall_features import recall_schema
        payload["recall_input_ablation"] = recall_schema(config)
    if getattr(config, "router_v42_input_version", None) is not None:
        from ..qwen3_recall_v42_router.features import input_schema
        payload["v42_input_ablation"] = input_schema(config)
    return payload


def three_stream_schema_sha256(config: Qwen3Config) -> str:
    encoded = json.dumps(
        three_stream_schema_payload(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _set_three_stream_config_defaults(config: Qwen3Config) -> None:
    for name, value in _THREE_STREAM_DEFAULTS.items():
        if not hasattr(config, name):
            setattr(config, name, value)

    schema_version = int(config.router_schema_version)
    if config.router_training_history_mode not in {"full", "swa"}:
        raise ValueError("`router_training_history_mode` must be 'full' or 'swa'.")
    expected_runtime_history = (
        "full_cache_nll_selected" if schema_version == 4 else "selected"
    )
    if config.router_runtime_history_mode != expected_runtime_history:
        raise ValueError(
            "Three-stream checkpoint runtime history mismatch: expected "
            f"{expected_runtime_history!r}."
        )
    if schema_version == 4 and config.router_training_history_mode != "full":
        raise ValueError("Schema-v4 threshold-selected training requires Full history.")
    if config.router_training_attention_backend != "flex_attention":
        raise ValueError("Formal three-stream training requires FlexAttention for every stream.")
    if not bool(config.router_swa_all_layers):
        raise ValueError("Formal three-stream training applies SWA to every Qwen3 layer.")
    if not bool(config.router_window_includes_current):
        raise ValueError("`router_window_size` must include the current token.")
    if bool(getattr(config, "use_sliding_window", False)) or getattr(config, "sliding_window", None) is not None:
        raise ValueError("Native Qwen3 sliding-window attention/cache must remain disabled.")
    if list(config.router_feature_schema) != _THREE_STREAM_DEFAULTS["router_feature_schema"]:
        raise ValueError("Three-stream Router feature schema/order is immutable.")
    if config.router_head_architecture not in {
        "three_tower_interaction_swiglu",
        "three_tower_interaction_swiglu_stabilized",
    }:
        raise ValueError("Three-stream Router requires a supported three-tower head.")
    if config.router_one_step_target != "signed_nll":
        raise ValueError("The formal three-stream experiment trains signed NLL only.")
    if config.router_one_step_transform != "signed_log1p":
        raise ValueError("The formal three-stream experiment requires `signed_log1p`.")

    if config.router_head_architecture == "three_tower_interaction_swiglu":
        if schema_version not in {2, 4}:
            raise ValueError(
                "The A0 three-tower head requires schema version 2 or 4."
            )
        if any(
            (
                bool(config.router_group_rms_enabled),
                bool(config.router_learned_fusion_scales),
                bool(config.router_layer_scale_enabled),
                float(config.router_one_step_sign_aux_weight) != 0.0,
            )
        ):
            raise ValueError("Schema-v2 legacy Head cannot enable stabilized ablations.")
    else:
        if int(config.router_schema_version) != 3:
            raise ValueError("The stabilized three-tower head requires schema version 3.")
        if not bool(config.router_group_rms_enabled):
            raise ValueError("The stabilized ablation ladder starts with group RMS enabled.")
        if bool(config.router_learned_fusion_scales) and not bool(
            config.router_group_rms_enabled
        ):
            raise ValueError("Learned fusion scales require the group-RMS stage.")
        if bool(config.router_layer_scale_enabled) and not bool(
            config.router_learned_fusion_scales
        ):
            raise ValueError("LayerScale requires learned fusion scales in the ladder.")
        if float(config.router_one_step_sign_aux_weight) > 0.0 and not bool(
            config.router_layer_scale_enabled
        ):
            raise ValueError("Auxiliary sign loss requires the LayerScale stage.")

    actual_schema_hash = three_stream_schema_sha256(config)
    configured_schema_hash = getattr(config, "router_schema_sha256", None)
    if configured_schema_hash not in {None, "", actual_schema_hash}:
        raise ValueError(
            "Three-stream Router schema hash mismatch: configured "
            f"{configured_schema_hash}, computed {actual_schema_hash}."
        )
    config.router_schema_sha256 = actual_schema_hash


@dataclass
class Qwen3ThreeStreamRouterOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    router_loss: Optional[torch.FloatTensor] = None
    router_transformed_score: Optional[torch.FloatTensor] = None
    router_targets: Optional[torch.FloatTensor] = None
    router_raw_target: Optional[torch.FloatTensor] = None
    router_predicted_raw_target: Optional[torch.FloatTensor] = None
    full_token_nll: Optional[torch.FloatTensor] = None
    swa_token_nll: Optional[torch.FloatTensor] = None
    history_local_selected_mask: Optional[torch.BoolTensor] = None
    training_metrics: Optional[dict] = None
    training_probe_tensors: Optional[dict] = None
    trunk_hidden_states: Optional[torch.FloatTensor] = None
    full_candidate_hidden_states: Optional[torch.FloatTensor] = None
    swa_candidate_hidden_states: Optional[torch.FloatTensor] = None


def _segment_ids(position_ids: torch.LongTensor, valid_mask: torch.BoolTensor) -> torch.LongTensor:
    """Assign a segment id only at valid position-id resets."""

    return (position_ids.eq(0) & valid_mask).to(torch.int64).cumsum(dim=-1)


def dense_three_stream_mask_reference(
    position_ids: torch.LongTensor,
    valid_mask: torch.BoolTensor,
    sink_size: int,
    window_size: int,
    trunk_mode: str,
) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    """Return the semantic dense masks used to audit the FlexAttention masks.

    Unlike the block-mask implementation, this reference does not add dummy
    diagonal entries for empty rows.  It therefore describes exactly which
    real history tokens each stream may read and is intentionally kept simple
    enough for boundary, packing, and padding tests.
    """

    if position_ids.ndim != 2 or position_ids.shape != valid_mask.shape:
        raise ValueError("Dense three-stream masks require aligned 2-D tensors.")
    if trunk_mode not in {"full", "swa"}:
        raise ValueError("`trunk_mode` must be full or swa.")
    if sink_size < 0 or window_size < 1:
        raise ValueError("SWA requires sink_size >= 0 and window_size >= 1.")

    valid_mask = valid_mask.to(dtype=torch.bool, device=position_ids.device)
    position_ids = position_ids.to(dtype=torch.long)
    segment_ids = _segment_ids(position_ids, valid_mask)
    query_position = position_ids[:, :, None]
    key_position = position_ids[:, None, :]
    common = (
        valid_mask[:, :, None]
        & valid_mask[:, None, :]
        & segment_ids[:, :, None].eq(segment_ids[:, None, :])
    )
    causal = key_position <= query_position
    past_only = key_position < query_position
    local = (key_position < int(sink_size)) | (
        key_position >= query_position - int(window_size) + 1
    )

    trunk = common & causal
    if trunk_mode == "swa":
        trunk = trunk & local
    full_past = common & past_only
    swa_past = full_past & local
    return trunk, full_past, swa_past


def create_three_stream_masks(
    position_ids: torch.LongTensor,
    valid_mask: torch.BoolTensor,
    sink_size: int,
    window_size: int,
    trunk_mode: str,
) -> Tuple[BlockMask, BlockMask, BlockMask]:
    """Build trunk, Full-past, and SWA-past block masks.

    Candidate masks are past-only.  Their current-token self term is computed
    from candidate K/V and merged separately, so candidate K/V can never leak
    into another query row.
    """

    if position_ids.ndim != 2 or position_ids.shape != valid_mask.shape:
        raise ValueError("Three-stream masks require aligned 2-D position/valid tensors.")
    if trunk_mode not in {"full", "swa"}:
        raise ValueError("`trunk_mode` must be full or swa.")
    if sink_size < 0 or window_size < 1:
        raise ValueError("SWA requires sink_size >= 0 and window_size >= 1.")

    batch_size, sequence_length = position_ids.shape
    segment_ids = _segment_ids(position_ids, valid_mask)

    def common(b, q_idx, kv_idx):
        q_valid = valid_mask[b, q_idx]
        kv_valid = valid_mask[b, kv_idx]
        same_segment = segment_ids[b, q_idx] == segment_ids[b, kv_idx]
        return q_valid, kv_valid, same_segment

    def is_local(q_position, kv_position):
        return (kv_position < sink_size) | (kv_position >= q_position - window_size + 1)

    def trunk_mask(b, h, q_idx, kv_idx):
        del h
        q_valid, kv_valid, same_segment = common(b, q_idx, kv_idx)
        q_position = position_ids[b, q_idx]
        kv_position = position_ids[b, kv_idx]
        causal = kv_position <= q_position
        allowed = q_valid & kv_valid & same_segment & causal
        if trunk_mode == "swa":
            allowed = allowed & is_local(q_position, kv_position)
        # Invalid rows get a dummy diagonal that is zeroed after attention.
        return allowed | ((~q_valid) & (q_idx == kv_idx))

    def full_past_mask(b, h, q_idx, kv_idx):
        del h
        q_valid, kv_valid, same_segment = common(b, q_idx, kv_idx)
        q_position = position_ids[b, q_idx]
        kv_position = position_ids[b, kv_idx]
        allowed = q_valid & kv_valid & same_segment & (kv_position < q_position)
        has_past = q_valid & q_position.gt(0)
        return allowed | ((~has_past) & (q_idx == kv_idx))

    def swa_past_mask(b, h, q_idx, kv_idx):
        del h
        q_valid, kv_valid, same_segment = common(b, q_idx, kv_idx)
        q_position = position_ids[b, q_idx]
        kv_position = position_ids[b, kv_idx]
        allowed = (
            q_valid
            & kv_valid
            & same_segment
            & (kv_position < q_position)
            & is_local(q_position, kv_position)
        )
        has_past = q_valid & q_position.gt(0)
        return allowed | ((~has_past) & (q_idx == kv_idx))

    kwargs = dict(
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=position_ids.device,
    )
    return (
        create_block_mask(trunk_mask, **kwargs),
        create_block_mask(full_past_mask, **kwargs),
        create_block_mask(swa_past_mask, **kwargs),
    )


class Qwen3ThreeStreamDecoderLayer(nn.Module):
    """Advance a fixed trunk and two isolated current-action candidates."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3RouterAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _trunk_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: BlockMask,
        valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        run_flex_attention = _get_three_stream_compiled_flex("trunk")
        output = run_flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=self.self_attn.scaling,
            enable_gqa=self.self_attn.num_key_value_groups > 1,
            kernel_options={"ROWS_GUARANTEED_SAFE": False},
        )
        output = output.masked_fill(~valid_mask[:, None, :, None], 0.0)
        batch_size, _, sequence_length, _ = output.shape
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, -1).contiguous()
        return self.self_attn.o_proj(output)

    def _candidate_attention(
        self,
        candidate_query: torch.Tensor,
        trunk_key: torch.Tensor,
        trunk_value: torch.Tensor,
        candidate_key: torch.Tensor,
        candidate_value: torch.Tensor,
        past_mask: BlockMask,
        position_ids: torch.LongTensor,
        valid_mask: torch.BoolTensor,
        stream: str,
    ) -> torch.Tensor:
        run_flex_attention = _get_three_stream_compiled_flex(stream)
        past_output, past_lse = run_flex_attention(
            candidate_query,
            trunk_key,
            trunk_value,
            block_mask=past_mask,
            scale=self.self_attn.scaling,
            enable_gqa=self.self_attn.num_key_value_groups > 1,
            return_lse=True,
            kernel_options={"ROWS_GUARANTEED_SAFE": False},
        )
        candidate_key = repeat_kv(candidate_key, self.self_attn.num_key_value_groups)
        candidate_value = repeat_kv(candidate_value, self.self_attn.num_key_value_groups)
        self_score = (
            candidate_query.float() * candidate_key.float()
        ).sum(dim=-1) * self.self_attn.scaling
        merged_output, _ = online_softmax_merge_local_self(
            past_output,
            past_lse,
            self_score,
            candidate_value,
            empty_past_mask=~(valid_mask & position_ids.gt(0)),
            valid_token_mask=valid_mask,
        )
        batch_size, _, sequence_length, _ = merged_output.shape
        merged_output = merged_output.transpose(1, 2).reshape(
            batch_size, sequence_length, -1
        ).contiguous()
        return self.self_attn.o_proj(merged_output)

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        full_candidate_hidden: torch.Tensor,
        swa_candidate_hidden: torch.Tensor,
        trunk_mask: BlockMask,
        full_past_mask: BlockMask,
        swa_past_mask: BlockMask,
        position_ids: torch.LongTensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        valid_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trunk_residual = trunk_hidden
        full_residual = full_candidate_hidden
        swa_residual = swa_candidate_hidden

        trunk_normalized = self.input_layernorm(trunk_hidden)
        full_normalized = self.input_layernorm(full_candidate_hidden)
        swa_normalized = self.input_layernorm(swa_candidate_hidden)
        trunk_q, trunk_k, trunk_v = self.self_attn.project_qkv(
            trunk_normalized, position_embeddings
        )
        full_q, full_k, full_v = self.self_attn.project_qkv(
            full_normalized, position_embeddings
        )
        swa_q, swa_k, swa_v = self.self_attn.project_qkv(
            swa_normalized, position_embeddings
        )

        trunk_attention = self._trunk_attention(
            trunk_q, trunk_k, trunk_v, trunk_mask, valid_mask
        )
        full_attention = self._candidate_attention(
            full_q,
            trunk_k,
            trunk_v,
            full_k,
            full_v,
            full_past_mask,
            position_ids,
            valid_mask,
            "full_candidate",
        )
        swa_attention = self._candidate_attention(
            swa_q,
            trunk_k,
            trunk_v,
            swa_k,
            swa_v,
            swa_past_mask,
            position_ids,
            valid_mask,
            "swa_candidate",
        )

        trunk_hidden = trunk_residual + trunk_attention
        full_candidate_hidden = full_residual + full_attention
        swa_candidate_hidden = swa_residual + swa_attention
        trunk_hidden = trunk_hidden + self.mlp(self.post_attention_layernorm(trunk_hidden))
        full_candidate_hidden = full_candidate_hidden + self.mlp(
            self.post_attention_layernorm(full_candidate_hidden)
        )
        swa_candidate_hidden = swa_candidate_hidden + self.mlp(
            self.post_attention_layernorm(swa_candidate_hidden)
        )
        return trunk_hidden, full_candidate_hidden, swa_candidate_hidden


class Qwen3ThreeStreamModel(Qwen3RouterPreTrainedModel):
    """Frozen Qwen3 backbone with one history trunk and two action candidates."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            Qwen3ThreeStreamDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
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
            raise ValueError("Three-stream oracle training does not support KV-cache decoding.")
        if output_attentions or output_hidden_states:
            raise ValueError("Three-stream oracle training does not retain per-layer outputs.")
        if get_parallel_state().sp_enabled:
            raise ValueError("Three-stream Router training requires ulysses_parallel_size=1.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, sequence_length, _ = inputs_embeds.shape
        if attention_mask is None:
            valid_mask = torch.ones(
                (batch_size, sequence_length), dtype=torch.bool, device=inputs_embeds.device
            )
        else:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("Three-stream attention mask must be dense [batch, sequence].")
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
            raise ValueError("Three-stream position_ids must match the dense token matrix.")
        position_ids = position_ids.to(device=inputs_embeds.device, dtype=torch.long)

        trunk_mask, full_past_mask, swa_past_mask = create_three_stream_masks(
            position_ids,
            valid_mask,
            int(self.config.router_sink_size),
            int(self.config.router_window_size),
            self.config.router_training_history_mode,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        trunk_hidden = inputs_embeds
        full_candidate_hidden = inputs_embeds
        swa_candidate_hidden = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            trunk_hidden, full_candidate_hidden, swa_candidate_hidden = layer(
                trunk_hidden,
                full_candidate_hidden,
                swa_candidate_hidden,
                trunk_mask,
                full_past_mask,
                swa_past_mask,
                position_ids,
                position_embeddings,
                valid_mask,
            )
            if self.config.router_check_finite:
                _check_router_tensor_is_finite(f"layer_{layer_idx}.trunk_hidden", trunk_hidden)
                _check_router_tensor_is_finite(
                    f"layer_{layer_idx}.full_candidate_hidden", full_candidate_hidden
                )
                _check_router_tensor_is_finite(
                    f"layer_{layer_idx}.swa_candidate_hidden", swa_candidate_hidden
                )

        trunk_hidden = self.norm(trunk_hidden)
        full_candidate_hidden = self.norm(full_candidate_hidden)
        swa_candidate_hidden = self.norm(swa_candidate_hidden)
        return (
            trunk_hidden,
            full_candidate_hidden,
            swa_candidate_hidden,
            inputs_embeds,
            position_ids,
            valid_mask,
        )


class Qwen3ThreeStreamRouterForCausalLM(Qwen3OneStepRouterForCausalLM):
    """Train a three-input signed-NLL Router under a fixed Full/SWA history."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3Config):
        # Set the experiment-specific values before shared validation.
        for name, value in _THREE_STREAM_DEFAULTS.items():
            if not hasattr(config, name):
                setattr(config, name, value)
        _set_one_step_config_defaults(config)
        _set_router_config_defaults(config)
        _set_three_stream_config_defaults(config)
        Qwen3RouterPreTrainedModel.__init__(self, config)
        self.model = Qwen3ThreeStreamModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_head = _build_router_head(config)
        self.post_init()
        self.router_head.float()
        self.requires_grad_(False)
        self.router_head.requires_grad_(True)
        parameter_count = sum(parameter.numel() for parameter in self.router_head.parameters())
        tensor_count = len(self.router_head.state_dict())
        self.config.router_head_parameter_count = int(parameter_count)
        self.config.router_head_tensor_count = int(tensor_count)
        if (
            int(config.hidden_size) in {2048, 4096}
            and parameter_count != int(config.router_expected_head_parameter_count)
        ):
            raise RuntimeError(
                f"Three-tower Router parameter count is {parameter_count}, expected "
                f"{config.router_expected_head_parameter_count}."
            )
        if tensor_count != int(config.router_expected_head_tensor_count):
            raise RuntimeError(
                f"Three-tower Router tensor count is {tensor_count}, expected "
                f"{config.router_expected_head_tensor_count}."
            )

    def _build_three_stream_features(
        self,
        trunk_hidden: torch.Tensor,
        input_embeddings: torch.Tensor,
        swa_candidate_hidden: torch.Tensor,
        position_ids: torch.LongTensor,
        valid_mask: torch.BoolTensor,
        history_source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if history_source is None:
            history_source = trunk_hidden
        history = torch.zeros_like(trunk_hidden)
        history[:, 1:] = history_source[:, :-1]
        history = history.masked_fill(position_ids.eq(0)[..., None], 0.0)
        features = torch.cat((history, input_embeddings, swa_candidate_hidden), dim=-1)
        if getattr(self.config, "router_recall_input_version", None) is not None:
            from ..qwen3_router.recall_features import pack_recall_features
            features = pack_recall_features(
                history, input_embeddings, swa_candidate_hidden,
                getattr(self, "_recall_confidence", None), self.config.router_recall_input_variant,
            )
        return features.masked_fill(~valid_mask[..., None], 0.0)

    def _history_selection_metrics(
        self,
        local_selected_mask: torch.BoolTensor,
        selector_mask: torch.BoolTensor,
        raw_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        selected_count = local_selected_mask.float().sum()
        eligible_count = selector_mask.float().sum()
        selected_gain = torch.where(
            local_selected_mask,
            -raw_target.float(),
            torch.zeros_like(raw_target.float()),
        ).sum()
        statistics = torch.stack((selected_count, eligible_count, selected_gain))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                statistics, op=torch.distributed.ReduceOp.SUM
            )
        denominator = statistics[1].clamp_min(1.0)
        selected_denominator = statistics[0].clamp_min(1.0)
        local_rate = statistics[0] / denominator
        return {
            "router/history/local_selected_rate": local_rate,
            "router/history/full_selected_rate": 1.0 - local_rate,
            "router/history/local_selected_mean_nll_advantage": statistics[2]
            / selected_denominator,
            "router/history/selection_threshold": local_rate.new_tensor(
                float(self.config.router_history_nll_threshold)
            ),
        }

    def _select_threshold_history_source(
        self,
        trunk_hidden: torch.Tensor,
        swa_candidate_hidden: torch.Tensor,
        raw_target: torch.Tensor,
        selector_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        threshold = float(self.config.router_history_nll_threshold)
        local_selected_mask = (
            selector_mask
            & torch.isfinite(raw_target)
            & raw_target.detach().lt(threshold)
        )
        history_source = torch.where(
            local_selected_mask[..., None],
            swa_candidate_hidden.detach(),
            trunk_hidden.detach(),
        )
        return history_source, local_selected_mask

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
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Qwen3ThreeStreamRouterOutput:
        del logits_to_keep
        if labels is None:
            raise ValueError("Three-stream Router supervision requires causal-LM labels.")

        (
            trunk_hidden,
            full_candidate_hidden,
            swa_candidate_hidden,
            input_embeddings,
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

        full_nll, swa_nll = self._compute_token_nll_only(
            full_candidate_hidden,
            swa_candidate_hidden,
            labels,
            valid_mask,
            position_ids,
        )
        raw_target = swa_nll - full_nll
        loss_mask = self._get_router_loss_mask(valid_mask, labels, position_ids)
        if not torch.isfinite(raw_target[loss_mask]).all():
            raise FloatingPointError("Three-stream signed-NLL target is non-finite in the loss mask.")
        transformed_target = self._transform_target(raw_target)
        safe_target = torch.where(loss_mask, transformed_target, torch.zeros_like(transformed_target))
        history_source = None
        history_local_selected_mask = None
        if getattr(self.config, "router_history_selector", None) == (
            "local_when_signed_nll_lt_threshold"
        ):
            history_source, history_local_selected_mask = (
                self._select_threshold_history_source(
                    trunk_hidden,
                    swa_candidate_hidden,
                    raw_target,
                    loss_mask,
                )
            )
        router_features = self._build_three_stream_features(
            trunk_hidden,
            input_embeddings,
            swa_candidate_hidden,
            position_ids,
            valid_mask,
            history_source,
        )
        head_diagnostics = None
        if hasattr(self.router_head, "forward_with_diagnostics"):
            transformed_score, head_diagnostics = (
                self.router_head.forward_with_diagnostics(router_features)
            )
        else:
            transformed_score = self.router_head(router_features)
        if transformed_score.dtype != torch.float32:
            raise TypeError(
                f"Three-tower Router output must remain FP32, got {transformed_score.dtype}."
            )
        if self.config.router_check_finite:
            _check_router_tensor_is_finite("router_transformed_score", transformed_score)
            _check_router_tensor_is_finite("router_transformed_target", safe_target)

        router_loss, metrics = self._compute_one_step_loss(
            transformed_score,
            raw_target,
            safe_target,
            loss_mask,
        )
        loss = router_loss * float(self.config.router_loss_weight)
        predicted_raw = self._inverse_target(transformed_score)
        if return_dict is False:
            return loss, transformed_score, safe_target, raw_target

        probe_tensors = {
            "router_transformed_score": transformed_score.detach(),
            "router_targets": safe_target.detach(),
            "router_raw_target": torch.where(
                loss_mask, raw_target, torch.zeros_like(raw_target)
            ).detach(),
            "router_predicted_raw_target": predicted_raw.detach(),
            "router_feature_l2": router_features.detach().float().norm(dim=-1),
            "full_token_nll": full_nll.detach(),
            "swa_token_nll": swa_nll.detach(),
            "position_ids": position_ids.detach(),
            "valid_token_mask": loss_mask.detach(),
            **({"input_ids": input_ids.detach()} if input_ids is not None else {}),
            **(head_diagnostics or {}),
        }
        if history_local_selected_mask is not None:
            probe_tensors.update(
                {
                    "history_local_selected_mask": history_local_selected_mask.detach(),
                    "history_selected_vs_full_l2": (
                        history_source.float() - trunk_hidden.float()
                    )
                    .norm(dim=-1)
                    .detach(),
                }
            )
        training_metrics = {
            "router/objective/total_loss": (
                metrics["huber_loss"]
                if history_local_selected_mask is not None
                else loss.detach()
            ),
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
            "router/huber_loss": metrics["huber_loss"],
            "router/sign_aux_loss": metrics["sign_aux_loss"],
            "router/weighted_sign_aux_loss": metrics["weighted_sign_aux_loss"],
            "router/sign_aux_token_rate": metrics["sign_aux_token_rate"],
            "router/exact_zero_target_rate": metrics["exact_zero_target_rate"],
            "router/near_zero_target_rate": metrics["near_zero_target_rate"],
            "router/signed_nll_mean": metrics["raw_target_mean"],
            "router/transformed_mae": metrics["transformed_mae"],
            "router/transformed_rmse": metrics["transformed_rmse"],
            "router/sign_target_rate": metrics["sign_target_rate"],
            "router/sign_predicted_positive_rate": metrics["sign_predicted_positive_rate"],
            "router/sign_accuracy": metrics["sign_accuracy"],
            "router/sign_precision": metrics["sign_precision"],
            "router/sign_recall": metrics["sign_recall"],
            "router/sign_f1": metrics["sign_f1"],
        }
        if history_local_selected_mask is not None:
            training_metrics.update(
                self._history_selection_metrics(
                    history_local_selected_mask, loss_mask, raw_target
                )
            )
            training_metrics.update(
                {
                    "router/control_p999/target_full_rate": metrics[
                        "harm_target_rate"
                    ],
                    "router/control_p999/predicted_full_rate": metrics[
                        "harm_predicted_positive_rate"
                    ],
                    "router/control_p999/accuracy": metrics["harm_accuracy"],
                    "router/control_p999/precision": metrics["harm_precision"],
                    "router/control_p999/recall": metrics["harm_recall"],
                    "router/control_p999/f1": metrics["harm_f1"],
                }
            )
        fusion_scales = getattr(self.router_head, "fusion_scales", None)
        if fusion_scales is not None:
            for name, value in zip(("a", "b", "c", "f"), fusion_scales):
                training_metrics[f"router/fusion_scale_{name}"] = value.detach()
        for index, block in enumerate(getattr(self.router_head, "blocks", ())):
            layer_scale = getattr(block, "layer_scale", None)
            if layer_scale is not None:
                training_metrics[f"router/layer_scale_{index}"] = (
                    layer_scale.detach()
                )
        return Qwen3ThreeStreamRouterOutput(
            loss=loss,
            router_loss=router_loss,
            router_transformed_score=transformed_score,
            router_targets=safe_target,
            router_raw_target=raw_target,
            router_predicted_raw_target=predicted_raw,
            full_token_nll=full_nll,
            swa_token_nll=swa_nll,
            history_local_selected_mask=history_local_selected_mask,
            training_metrics=training_metrics,
            training_probe_tensors=probe_tensors,
            trunk_hidden_states=trunk_hidden if bool(self.config.router_return_hidden_states) else None,
            full_candidate_hidden_states=(
                full_candidate_hidden if bool(self.config.router_return_hidden_states) else None
            ),
            swa_candidate_hidden_states=(
                swa_candidate_hidden if bool(self.config.router_return_hidden_states) else None
            ),
        )


ModelClass = Qwen3ThreeStreamRouterForCausalLM
