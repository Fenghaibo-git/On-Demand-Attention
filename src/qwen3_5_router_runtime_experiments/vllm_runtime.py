# Modified for the portable Router v4.2 release; see NOTICE.
"""Native vLLM 0.18.0 Qwen3.5 dual-cache candidate implementation.

Experimental, eager, serial batch 1 only. Real-weight qualification is required;
this file is not installed into the existing vLLM 0.8.5 or HF queues.
Native paged Full KV and native GDN kernels are retained. Local attention reads
an independent compact selected-KV allocation via SDPA. No HF backbone is run.
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path
import types

import torch
from torch import nn
from torch.nn import functional as F
import vllm
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)

from .compact_cache import CompactSelectedKV
from .portable_head import PortableA0Head, load_admitted_head, file_sha256
from .serial_position_source import read_positions
from .vllm_contract import (
    SerialRequestClock,
    validate_vllm_contract,
    validate_text_config,
)


class AuditedHead(nn.Module):
    """Run the existing strict export loader without instantiating an HF LLM."""

    def __init__(self, text_config, lm_head, base_model_path):
        super().__init__()
        self.router_head = PortableA0Head(
            hidden_size=text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.base_model_path = base_model_path
        object.__setattr__(self, "lm_head", lm_head)
        self._router_checkpoint_loaded = False
        self._router_checkpoint_metadata = {}

    def admit(self):
        path = os.environ.get("Q35_DUAL_ROUTER_ADMISSION")
        if not path:
            raise ValueError(
                "Q35_DUAL_ROUTER_ADMISSION must name a strict HF-side admission"
            )
        self._router_checkpoint_metadata = load_admitted_head(
            self.router_head, path, self.base_model_path
        )
        self.router_head.to(device=self.lm_head.weight.device, dtype=torch.float32)
        self._router_checkpoint_loaded = True

    def forward(self, previous, embedding, local):
        if not self._router_checkpoint_loaded:
            raise RuntimeError(
                "An audited Router checkpoint must be admitted before decode"
            )
        return self.router_head(previous, embedding, local)


class Qwen35DualCacheForCausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    base_model_path = Path(os.environ.get("ROUTER_MODEL_ROOT", "/path/to/models")) / "Qwen3.5-2B"
    base_config_sha256 = "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
    expected_hidden_size = 2048
    expected_num_hidden_layers = 24
    full_equivalent_reuse = True

    get_mamba_state_dtype_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config
    )
    get_mamba_state_shape_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
    )
    get_mamba_state_copy_func = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func
    )

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # The native engine retains Qwen3.5's M-RoPE contract in text-only mode.
        # Text has identical T/H/W positions and a zero generation offset.
        if mm_features:
            raise ValueError("Multimodal positions are outside this text-only experiment")
        if not input_tokens:
            raise ValueError("A nonempty text prompt is required")
        return torch.arange(len(input_tokens), dtype=torch.long).expand(3, -1).clone(), 0

    def __init__(self, *, vllm_config, prefix=""):
        vc = vllm_config
        from .qwen35_state_contract import validate_qwen35_state_config
        validate_qwen35_state_config(vc)
        backend = vc.attention_config.backend
        backend_name = getattr(backend, "name", str(backend))
        self._contract = {
            "tensor_parallel_size": vc.parallel_config.tensor_parallel_size,
            "pipeline_parallel_size": vc.parallel_config.pipeline_parallel_size,
            "max_num_seqs": vc.scheduler_config.max_num_seqs,
            "enforce_eager": vc.model_config.enforce_eager,
            "enable_chunked_prefill": vc.scheduler_config.enable_chunked_prefill,
            "enable_prefix_caching": vc.cache_config.enable_prefix_caching,
            "speculative": vc.speculative_config is not None,
            "quantized": vc.quant_config is not None,
            "dtype": str(vc.model_config.dtype),
            "backend": backend_name,
            "kv_cache_dtype": vc.cache_config.cache_dtype,
            "sink_size": 4,
            "window_size": 2048,
            "policy": os.environ.get("Q35_DUAL_POLICY", "router"),
            "full_input_mode": os.environ.get("Q35_DUAL_FULL_INPUT", "embedding"),
            "threshold": float(os.environ.get("Q35_DUAL_THRESHOLD", "0")),
            "mamba_cache_dtype": vc.cache_config.mamba_cache_dtype,
            "mamba_ssm_cache_dtype": vc.cache_config.mamba_ssm_cache_dtype,
        }
        validate_vllm_contract(vllm.__version__, self._contract)
        base_model = Path(vc.model_config.model).resolve()
        if base_model != self.base_model_path:
            raise ValueError("Only the canonical selected Qwen3.5 assets are admitted")
        config_path = base_model / "config.json"
        if (
            file_sha256(config_path)
            != self.base_config_sha256
        ):
            raise ValueError("Canonical base configuration hash changed")
        reference = json.loads(config_path.read_text())["text_config"]
        validate_text_config(reference, vc.model_config.hf_text_config.to_dict())
        cfg = vc.model_config.hf_text_config
        long_rope = os.environ.get("Q35_DUAL_LONG_ROPE_512K") == "1"
        allowed_overrides = {"architectures", "text_config"} if long_rope else {"architectures"}
        if vc.model_config.hf_overrides and set(vc.model_config.hf_overrides) != allowed_overrides:
            raise ValueError("Only the gated architecture and position overrides are admitted")
        if long_rope:
            if self.base_model_path.name != "Qwen3.5-27B" or self._contract["tensor_parallel_size"] not in (2, 4, 8):
                raise ValueError("Extended RoPE is admitted only for the 27B multi-GPU speed curve")
            override = vc.model_config.hf_overrides["text_config"]
            expected_rope = dict(reference["rope_parameters"])
            expected_rope.update(rope_type="linear", factor=2.0)
            if (set(override) != {"max_position_embeddings", "rope_parameters"}
                    or override["max_position_embeddings"] != 524288
                    or override["rope_parameters"] != expected_rope
                    or cfg.max_position_embeddings != 524288
                    or cfg.rope_parameters.get("rope_type") != "linear"
                    or not math.isclose(float(cfg.rope_parameters.get("factor", float("nan"))),
                                         2.0, rel_tol=1e-6, abs_tol=1e-6)):
                raise ValueError("512K RoPE extension differs from the frozen linear factor-2 contract")
        elif cfg.max_position_embeddings != reference["max_position_embeddings"]:
            raise ValueError("Native-position run changed max_position_embeddings")
        super().__init__(vllm_config=vc, prefix=prefix)
        cfg = vc.model_config.hf_text_config
        if (
            cfg.hidden_size != self.expected_hidden_size
            or cfg.num_hidden_layers != self.expected_num_hidden_layers
            or cfg.model_type != "qwen3_5_text"
        ):
            raise ValueError("The selected dense Qwen3.5 architecture differs")
        self.admission = (
            AuditedHead(cfg, self.lm_head, vc.model_config.model)
            if self._contract["policy"] == "router"
            else None
        )
        self._clock = SerialRequestClock()
        self._phase = "profile"
        self._previous = None
        self._local = {}
        self._staged = {}
        self._attention_layers = {}
        self._gdn_layers = {}
        self._stats = {
            "prefill_tokens": 0,
            "routed_tokens": 0,
            "logical_full_tokens": 0,
            "full_reruns": 0,
            "nonfinite_scores": 0,
        }
        self._last_score = None
        self._last_full = None
        for index, layer in enumerate(self.model.layers):
            if layer.layer_type == "full_attention":
                attention = layer.self_attn.attn
                self._attention_layers[index] = attention
                original = attention.forward

                def redirected(
                    module,
                    query,
                    key,
                    value,
                    *args,
                    _index=index,
                    _original=original,
                    **kwargs,
                ):
                    return self._attention(
                        _index, module, _original, query, key, value, *args, **kwargs
                    )

                attention.forward = types.MethodType(redirected, attention)
            else:
                self._gdn_layers[index] = layer.linear_attn

    def load_weights(self, weights):
        def text_only():
            for name, weight in weights:
                if name.startswith(("model.visual.", "mtp.")):
                    continue
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model.") :]
                yield name, weight

        loaded = super().load_weights(text_only())
        if self.admission is not None:
            self.admission.admit()
            loaded |= {
                "admission." + name for name, _ in self.admission.named_parameters()
            }
        return loaded

    @staticmethod
    def _metadata(module):
        context = get_forward_context()
        all_metadata = context.attn_metadata
        if not isinstance(all_metadata, dict):
            raise TypeError("Pinned vLLM V1 requires per-layer metadata")
        metadata = all_metadata[
            module.layer_name if hasattr(module, "layer_name") else module.prefix
        ]
        return context, metadata

    def _attention(self, index, module, original, query, key, value, *args, **kwargs):
        if self._phase == "profile":
            return original(query, key, value, *args, **kwargs)
        context, meta = self._metadata(module)
        if (
            module.impl.__class__.__name__ != "FlashAttentionImpl"
            or meta.block_table.shape[0] != 1
        ):
            raise ValueError("Only single-request FlashAttention metadata is admitted")
        if bool(meta.use_cascade):
            raise ValueError("Cascade attention is outside this experiment")
        heads, kv_heads, dim = module.num_heads, module.num_kv_heads, module.head_size
        q = query.reshape(-1, heads, dim)
        k = key.reshape(-1, kv_heads, dim)
        v = value.reshape(-1, kv_heads, dim)
        if self._phase == "prefill":
            if meta.num_actual_tokens != q.shape[0]:
                raise ValueError(
                    "Padded or mixed prefill is outside this serial experiment"
                )
            output = original(query, key, value, *args, **kwargs)
            local = CompactSelectedKV(4, 2048)
            local.initialize(k.transpose(0, 1)[None], v.transpose(0, 1)[None], [0])
            self._local[index] = local
            return output
        if q.shape[0] != 1 or meta.num_actual_tokens != 1:
            raise ValueError("Decode attention must contain one actual token")
        self._staged[index] = (k, v)
        paged = module.kv_cache[context.virtual_engine]
        if self._phase == "full":
            module.impl.do_kv_cache_update(module, k, v, paged, meta.slot_mapping)
            output = torch.empty_like(q)
            module.impl.forward(module, q, k, v, paged, meta, output=output)
            return output.reshape(1, heads * dim)
        if self._phase != "local":
            raise RuntimeError("Invalid attention phase")
        local = self._local[index]
        if local.length != self._clock.length:
            raise RuntimeError("Compact KV is stale")
        compact_k, compact_v, valid = local.candidate(
            torch.zeros(1, device=q.device, dtype=torch.long),
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
        )
        if self._clock.length >= local.capacity:
            valid = None  # Serial unpadded history fills every compact slot.
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            compact_k,
            compact_v,
            attn_mask=valid,
            dropout_p=0.0,
            is_causal=False,
            scale=module.impl.scale,
            enable_gqa=True,
        )
        return output.reshape(1, heads * dim)

    def _snapshot_gdn(self):
        result = []
        for module in self._gdn_layers.values():
            context, meta = self._metadata(module)
            if (
                meta.num_actual_tokens != 1
                or meta.num_prefills != 0
                or meta.num_decodes != 1
                or meta.spec_sequence_masks is not None
            ):
                raise ValueError(
                    "Only non-speculative one-token GDN decode is admitted"
                )
            indices = meta.non_spec_state_indices_tensor[:1].flatten().long()
            tensors = module.kv_cache[context.virtual_engine]
            if len(tensors) != 2:
                raise ValueError("GDN must expose conv and recurrent states")
            for tensor in tensors:
                result.append(
                    (tensor, indices, tensor.index_select(0, indices).clone())
                )
        return result

    @staticmethod
    def _restore_gdn(snapshots):
        for tensor, indices, previous in snapshots:
            tensor.index_copy_(0, indices, previous)

    def _commit_attention(self):
        if set(self._staged) != set(self._attention_layers):
            raise RuntimeError(
                "Not every attention layer staged the selected candidate"
            )
        for index, module in self._attention_layers.items():
            context, meta = self._metadata(module)
            key, value = self._staged[index]
            if self._phase != "full":
                module.impl.do_kv_cache_update(
                    module,
                    key,
                    value,
                    module.kv_cache[context.virtual_engine],
                    meta.slot_mapping,
                )
            self._local[index].append_selected(
                key.transpose(0, 1)[None],
                value.transpose(0, 1)[None],
                self._clock.length,
            )
        self._staged.clear()

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        context = get_forward_context()
        if context.attn_metadata is None:
            self._phase = "profile"
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
        if intermediate_tensors is not None:
            raise ValueError("Pipeline-parallel intermediate state is not admitted")
        values = read_positions(self, positions)
        phase = self._clock.classify(values)
        if phase == "prefill":
            self._phase = "prefill"
            self._local.clear()
            self._staged.clear()
            self._previous = None
            try:
                hidden = super().forward(
                    input_ids, positions, None, inputs_embeds, **kwargs
                )
                if len(self._local) != len(self._attention_layers):
                    raise RuntimeError(
                        "Full prefill did not initialize every Local cache"
                    )
                self._previous = hidden[-1:].detach().clone()
                self._clock.commit("prefill", len(values))
                self._stats["prefill_tokens"] += len(values)
                return hidden
            except Exception:
                self._clock.failed = True
                raise
        if self._previous is None:
            raise RuntimeError("Missing selected hidden for current request")
        embedding = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        snapshots = self._snapshot_gdn()
        self._staged.clear()
        try:
            self._phase = "local"
            with torch.profiler.record_function("router.local_candidate"):
                local = self.model(None, positions, None, embedding)
            policy = self._contract["policy"]
            if policy == "router":
                with torch.profiler.record_function("router.head"):
                    score = self.admission(self._previous, embedding, local)
                finite = bool(torch.isfinite(score).all())
                full = not finite or bool(score.item() > self._contract["threshold"])
                self._last_score = float(score.item()) if finite else None
                self._stats["nonfinite_scores"] += int(not finite)
            else:
                full = policy == "always_full"
                self._last_score = None
            equivalent = (
                self.full_equivalent_reuse
                and
                self._clock.length + 1 <= 2052
                and self._contract["full_input_mode"] == "embedding"
            )
            hidden = local
            if full and not equivalent:
                self._restore_gdn(snapshots)
                self._phase = "full"
                full_input = (
                    local
                    if self._contract["full_input_mode"] == "local_final_hidden"
                    else embedding
                )
                with torch.profiler.record_function("router.full_rerun"):
                    hidden = self.model(None, positions, None, full_input)
                self._stats["full_reruns"] += 1
            with torch.profiler.record_function("router.commit"):
                self._commit_attention()
                self._previous = hidden.detach().clone()
                self._clock.commit("decode", 1)
            self._stats["routed_tokens"] += 1
            self._stats["logical_full_tokens"] += int(full)
            self._last_full = full
            return hidden
        except Exception:
            self._restore_gdn(snapshots)
            self._clock.failed = True
            raise

    def runtime_report(self):
        counts = dict(self._stats)
        counts["logical_local_tokens"] = (
            counts["routed_tokens"] - counts["logical_full_tokens"]
        )
        counts["full_equivalent_reuse_tokens"] = (
            counts["logical_full_tokens"] - counts["full_reruns"]
        )
        return {
            "format": "qwen35-dual-kv-native-vllm-experiment-v1",
            "contract": self._contract,
            "counts": counts,
            "local_storage_bytes": sum(c.storage_bytes() for c in self._local.values()),
            "quality_and_speed_qualified": False,
            "router": (
                self.admission._router_checkpoint_metadata
                if self.admission is not None
                else None
            ),
            "last_score": self._last_score,
            "last_full": self._last_full,
        }


class Qwen35DualCache27BForCausalLM(Qwen35DualCacheForCausalLM):
    """The audited v4.2 27B head over the same serial selected-cache engine."""

    base_model_path = Path(os.environ.get("ROUTER_MODEL_ROOT", "/path/to/models")) / "Qwen3.5-27B"
    base_config_sha256 = "f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c"
    expected_hidden_size = 5120
    expected_num_hidden_layers = 64
    full_equivalent_reuse = False
