# Modified for On Demand Attention; see NOTICE.
"""Opt-in serial Qwen3 adapter; Full, absolute Local, StreamingLLM, Router."""
import json
import math
import os
from pathlib import Path
import types
import torch
import vllm
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from qwen3_5_router_runtime_experiments.vllm_runtime import AuditedHead, Qwen35DualCacheForCausalLM
from qwen3_5_router_runtime_experiments.vllm_contract import SerialRequestClock, validate_text_config
from qwen3_5_router_runtime_experiments.serial_position_source import read_positions
from qwen3_5_router_runtime_experiments.portable_head import file_sha256
from .cache import RingKV, retained_positions

BASES = {
    'Qwen3-8B': ('f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30', 4096, 36, 0.0),
}


class Qwen3FastRouter(Qwen3ForCausalLM):
    streaming = False
    standalone = False
    native_full = False
    _metadata = staticmethod(Qwen35DualCacheForCausalLM._metadata)

    def __init__(self, *, vllm_config, prefix=''):
        vc = vllm_config
        base = Path(vc.model_config.model).resolve()
        if base.parent != Path(os.environ['ROUTER_MODEL_ROOT']).resolve() or base.name not in BASES:
            raise ValueError('Only selected canonical Qwen3 models are admitted')
        digest, hidden, layers, self._threshold = BASES[base.name]
        if file_sha256(base/'config.json') != digest:
            raise ValueError('Canonical model config changed')
        reference = json.loads((base/'config.json').read_text())
        cfg = vc.model_config.hf_config
        validate_text_config(reference, cfg.to_dict())
        if (vllm.__version__ != '0.18.0' or vc.parallel_config.tensor_parallel_size != 1
                or vc.parallel_config.pipeline_parallel_size != 1
                or vc.scheduler_config.max_num_seqs != 1 or vc.scheduler_config.async_scheduling
                or vc.scheduler_config.enable_chunked_prefill or vc.cache_config.enable_prefix_caching
                or not vc.model_config.enforce_eager or vc.quant_config or vc.speculative_config
                or vc.model_config.dtype != torch.bfloat16
                or getattr(vc.attention_config.backend, 'name', None) != 'FLASH_ATTN'):
            raise ValueError('Require pinned serial BF16 TP1 synchronous unchunked runtime')
        if set(vc.model_config.hf_overrides or {}) - {'architectures', 'max_position_embeddings', 'rope_scaling', 'rope_parameters'}:
            raise ValueError('Unexpected architectural override')
        if cfg.hidden_size != hidden or cfg.num_hidden_layers != layers:
            raise ValueError('Unexpected Qwen3 geometry')
        super().__init__(vllm_config=vc, prefix=prefix)
        self._contract = dict(tensor_parallel_size=1, pipeline_parallel_size=1, max_num_seqs=1,
                              enforce_eager=True, enable_chunked_prefill=False, enable_prefix_caching=False,
                              speculative=False, threshold=self._threshold,
                              base_model=str(base), max_model_len=vc.model_config.max_model_len,
                              rope_parameters=cfg.rope_parameters, streaming=self.streaming)
        self.admission = None if self.standalone or self.native_full else AuditedHead(cfg, self.lm_head, str(base))
        self._clock, self._phase, self._previous = SerialRequestClock(), 'profile', None
        self._attention_layers, self._gdn_layers, self._local, self._staged, self._raw_selected = {}, {}, {}, {}, {}
        self._graphs = {}
        self._use_graph = os.environ.get('Q3_STREAM_GRAPH', '1') == '1'
        self._oracle = os.environ.get('Q3_STREAM_ORACLE', '0') == '1'
        self._maximum = vc.model_config.max_model_len
        self._decode_position = torch.zeros((), dtype=torch.int64, device=self.lm_head.weight.device)
        self._stats = dict(prefill_tokens=0, routed_tokens=0, logical_full_tokens=0, full_reruns=0, nonfinite_scores=0)
        self._stream_stats = dict(graph_captures=0, decode_full_kv_writes=0)
        self._trace_enabled, self._route_trace = False, []
        self._last_score = self._last_full = None
        from .commit import SelectedCommit
        self._selected_commit = SelectedCommit(self)
        for index, layer in enumerate(self.model.layers):
            attention = layer.self_attn.attn
            self._attention_layers[index] = attention
            original = attention.forward
            def redirected(module, query, key, value, *args, _index=index, _original=original, **kwargs):
                return self._attention(_index, module, _original, query, key, value, *args, **kwargs)
            attention.forward = types.MethodType(redirected, attention)
            if self.standalone:
                attention.get_kv_cache_spec = types.MethodType(lambda module, config: None, attention)
            if self.streaming:
                layer.self_attn.forward = types.MethodType(self._stream_forward(index), layer.self_attn)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if self.admission is not None:
            self.admission.admit()
            loaded |= {'admission.' + name for name, _ in self.admission.named_parameters()}
        return loaded

    def _stream_forward(self, index):
        owner = self
        def forward(module, positions, hidden_states):
            qkv, _ = module.qkv_proj(hidden_states)
            q, k, v = qkv.split([module.q_size, module.kv_size, module.kv_size], dim=-1)
            q = module.q_norm(q.view(-1, module.num_heads, module.head_dim)).view(-1, module.q_size)
            k = module.k_norm(k.view(-1, module.num_kv_heads, module.head_dim)).view(-1, module.kv_size)
            if owner._phase == 'prefill':
                pos = retained_positions(k.shape[0], 4, 2047, k.device)
                owner._raw_selected[index] = k.view(-1, module.num_kv_heads, module.head_dim).index_select(0, pos)
            if owner._phase in ('prefill', 'profile'):
                q, k = module.rotary_emb(positions, q, k)
            output = module.attn(q, k, v)
            output, _ = module.o_proj(output)
            return output
        return forward

    def _attention(self, index, module, original, query, key, value, *args, **kwargs):
        if self._phase == 'profile' or self.native_full:
            return original(query, key, value, *args, **kwargs)
        context, meta = (None, None) if self.standalone else self._metadata(module)
        heads, hk, dim = module.num_heads, module.num_kv_heads, module.head_size
        q, k, v = query.reshape(-1, heads, dim), key.reshape(-1, hk, dim), value.reshape(-1, hk, dim)
        if self._phase == 'prefill':
            if self.standalone:
                from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func
                output = torch.empty_like(q)
                flash_attn_varlen_func(q=q, k=k, v=v, out=output,
                    cu_seqlens_q=self._prefill_cu, cu_seqlens_k=self._prefill_cu,
                    max_seqlen_q=q.shape[0], max_seqlen_k=q.shape[0],
                    softmax_scale=module.impl.scale, causal=True, window_size=(-1, -1), fa_version=2)
                output = output.reshape(query.shape)
            else:
                output = original(query, key, value, *args, **kwargs)
            if index not in self._local:
                ring = RingKV(heads, hk, dim, k.device, k.dtype, shift=self.streaming,
                              rotary_dim=dim, theta=1e6)
                if self.streaming:
                    ring.bind_native_rope(self.model.layers[index].self_attn.rotary_emb)
                ring.position = self._decode_position
                self._local[index] = ring
            self._local[index].seed(k, v, raw_selected_key=self._raw_selected.pop(index, None))
            return output
        if q.shape[0] != 1 or (meta is not None and (meta.num_actual_tokens != 1 or meta.use_cascade)):
            raise ValueError('Require one non-cascade decode token')
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        self._staged[index] = k, v
        if self._phase == 'full':
            paged = module.kv_cache[context.virtual_engine]
            module.impl.do_kv_cache_update(module, k, v, paged, meta.slot_mapping)
            output = torch.empty_like(q)
            module.impl.forward(module, q, k, v, paged, meta, output=output)
        else:
            ring = self._local[index]
            output = ring.oracle(q, k, v) if self._oracle else ring.attention(q, k, v)
        return output.reshape(1, heads*dim)

    def _run(self, phase, positions, embedding):
        self._phase = phase
        if not self._use_graph:
            return self.model(None, positions, None, embedding.clone())
        if phase == 'full':
            _, meta = self._metadata(next(iter(self._attention_layers.values())))
            maximum = min(self._maximum, ((meta.max_seq_len+63)//64)*64)
        else:
            maximum = self._maximum
        graph_key = f'{phase}:{maximum}'
        self._active_graph_key = graph_key
        if graph_key not in self._graphs:
            from .graph import BackboneGraph
            self._graphs[graph_key] = BackboneGraph(self, phase, positions, embedding, maximum)
            self._stream_stats['graph_captures'] += 1
        return self._graphs[graph_key](positions, embedding)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        source = getattr(self, '_cpu_position_source', None)
        if get_forward_context().attn_metadata is None and (source is None or source.active is None):
            self._phase = 'profile'
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if intermediate_tensors is not None or positions.ndim != 1:
            raise ValueError('Only serial text execution is admitted')
        values = read_positions(self, positions)
        phase = self._clock.classify(values)
        if phase == 'prefill':
            self._phase = 'prefill'
            self._staged.clear()
            if self.standalone:
                self._prefill_cu = torch.tensor([0,len(values)],device=positions.device,dtype=torch.int32)
            hidden = self.model(input_ids, positions, None, inputs_embeds)
            if not self.native_full and len(self._local) != len(self._attention_layers):
                raise RuntimeError('Incomplete ring initialization')
            if self.admission is not None:
                if self._previous is None:
                    self._previous = hidden[-1:].detach().clone()
                else:
                    self._previous.copy_(hidden[-1:])
            self._clock.commit('prefill', len(values))
            self._stats['prefill_tokens'] += len(values)
            return hidden
        embedding = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        if self.native_full:
            hidden = self._run('full', positions, embedding)
            self._clock.commit('decode', 1)
            return hidden
        try:
            hidden = self._run('local', positions, embedding)
            full = False
            if self.admission is not None:
                score_tensor = self._graphed_score if self._use_graph else self.admission(self._previous, embedding, hidden)
                score = float(score_tensor.item())
                full = not math.isfinite(score) or score > self._threshold
                self._last_score = score if math.isfinite(score) else None
                self._stats['nonfinite_scores'] += int(not math.isfinite(score))
                if full:
                    hidden = self._run('full', positions, embedding)
                    self._stats['full_reruns'] += 1
                self._previous.copy_(hidden)
            self._selected_commit()
            self._clock.commit('decode', 1)
            self._stats['routed_tokens'] += 1
            self._stats['logical_full_tokens'] += int(full)
            self._last_full = full
            if self._trace_enabled:
                self._route_trace.append(dict(position=self._clock.length-1, full=full, score=self._last_score))
            return hidden
        except BaseException:
            self._clock.failed = True
            raise

    def runtime_report(self):
        return dict(format='qwen3-shared-streaming-ring-v1', contract=self._contract,
                    counts=dict(self._stats), stats=dict(self._stream_stats),
                    native_full=self.native_full, standalone=self.standalone,
                    shared_kernel='qwen35_streaming_vllm.kernels.partials',
                    graph_replays={k:v.replays for k,v in self._graphs.items()},
                    local_storage_bytes=sum(r.storage_bytes() for r in self._local.values()),
                    graph_enabled=self._use_graph, oracle=self._oracle,
                    router=self.admission._router_checkpoint_metadata if self.admission else None,
                    route_trace=self._route_trace if self._trace_enabled else None)


class Qwen3FastLocal(Qwen3FastRouter):
    standalone = True


class Qwen3Streaming(Qwen3FastLocal):
    streaming = True


class Qwen3FastFull(Qwen3FastRouter):
    native_full = True
