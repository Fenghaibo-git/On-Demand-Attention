"""Opt-in Qwen3.5-27B StreamingLLM and selected-history Router, serial BF16."""
import math
import os
import types
import torch
from vllm.forward_context import get_forward_context
from qwen3_5_router_runtime_experiments.vllm_runtime import Qwen35DualCache27BForCausalLM
from qwen35_absolute_tp_vllm.serial_position_source import read_positions
from qwen3_5_router_runtime_experiments.prepared_gdn_state import PreparedGDNStates
from .cache import RingKV, retained_positions


class Qwen35FastRouter27B(Qwen35DualCache27BForCausalLM):
    """Absolute-position selected-history router; original step1536 semantics."""
    streaming = False
    standalone = False
    native_full = False

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config.parallel_config.tensor_parallel_size not in (1,2,4,8):
            raise ValueError('Require admitted TP1/2/4/8 batch-one execution')
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError('Use the verified synchronous CPU position producer')
        expected = 'always_full' if self.native_full else 'always_local' if self.standalone else 'router'
        if self._contract['policy'] != expected:
            raise ValueError('Architecture and runtime policy disagree')
        self._graphs = {}
        self._use_graph = os.environ.get('Q35_STREAM_GRAPH', '1') == '1'
        self._oracle = os.environ.get('Q35_STREAM_ORACLE', '0') == '1'
        self._maximum = vllm_config.model_config.max_model_len
        self._fused_states = PreparedGDNStates(self)
        self._raw_selected = {}
        self._decode_position = torch.zeros((), dtype=torch.int64, device=self.lm_head.weight.device)
        self._stream_stats = dict(snapshots=0, decode_full_kv_writes=0, graph_captures=0)
        self._trace_enabled = False
        self._route_trace = []
        from .commit import SelectedCommit
        self._selected_commit = SelectedCommit(self)
        if self.standalone:
            # RingKV owns the complete attention history. Returning no native
            # KV spec avoids reserving an entire prompt's unused paged blocks.
            for module in self._attention_layers.values():
                module.get_kv_cache_spec = types.MethodType(lambda module, config: None, module)
        if self.streaming:
            for index in self._attention_layers:
                module = self.model.layers[index].self_attn
                module.forward = types.MethodType(self._stream_forward(index), module)

    def _stream_forward(self, index):
        owner = self
        def forward(module, positions, output, hidden_states):
            # Native Qwen projections/norm/gate; only RoPE/cache handling differs.
            qkv, _ = module.qkv_proj(hidden_states)
            q_gate, k, v = qkv.split([module.q_size*2, module.kv_size, module.kv_size], dim=-1)
            shape = q_gate.shape[:-1]
            q, gate = torch.chunk(q_gate.view(*shape, module.num_heads, -1), 2, dim=-1)
            q, gate = q.reshape(*shape, -1), gate.reshape(*shape, -1)
            q = module.q_norm(q.view(-1, module.num_heads, module.head_dim)).view(-1, module.q_size)
            k = module.k_norm(k.view(-1, module.num_kv_heads, module.head_dim)).view(-1, module.kv_size)
            if owner._phase == 'prefill':
                pos = retained_positions(k.shape[0], 4, 2047, k.device)
                owner._raw_selected[index] = k.view(-1, module.num_kv_heads, module.head_dim).index_select(0, pos)
            if owner._phase in ('prefill', 'profile'):
                q, k = module.rotary_emb(positions, q, k)
            attention = module.attn(q, k, v)
            attention = attention * torch.sigmoid(gate)
            output[:], _ = module.o_proj(attention)
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
                flash_attn_varlen_func(
                    q=q, k=k, v=v, out=output,
                    cu_seqlens_q=self._prefill_cu, cu_seqlens_k=self._prefill_cu,
                    max_seqlen_q=q.shape[0], max_seqlen_k=q.shape[0],
                    softmax_scale=module.impl.scale, causal=True, window_size=(-1, -1),
                    fa_version=2)
                output = output.reshape(query.shape)
            else:
                output = original(query, key, value, *args, **kwargs)
            if index not in self._local:
                ring = RingKV(heads, hk, dim, k.device, k.dtype, shift=self.streaming)
                if self.streaming:
                    ring.bind_native_rope(self.model.layers[index].self_attn.rotary_emb)
                ring.position = self._decode_position
                self._local[index] = ring
            self._local[index].seed(k, v, raw_selected_key=self._raw_selected.pop(index, None))
            return output
        if q.shape[0] != 1 or (meta is not None and (meta.num_actual_tokens != 1 or meta.use_cascade)):
            raise ValueError('Require serial one-token non-cascade decode')
        k, v, q = k.contiguous(), v.contiguous(), q.contiguous()
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

    def _snapshot_gdn(self):
        self._stream_stats['snapshots'] += 1
        return self._fused_states.snapshot()

    def _restore_gdn(self, snapshots):
        self._fused_states.restore(snapshots)

    def _run(self, phase, positions, embedding):
        self._phase = phase
        if not self._use_graph:
            return self.model(None, positions, None, embedding)
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

    def _commit_attention(self):
        self._selected_commit()

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        context = get_forward_context()
        if context.attn_metadata is None:
            self._phase = 'profile'
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if intermediate_tensors is not None or kwargs:
            raise ValueError('Unsupported forward input')
        values = read_positions(self, positions)
        phase = self._clock.classify(values)
        if phase == 'prefill':
            self._phase = 'prefill'
            self._staged.clear()
            if self.standalone:
                self._prefill_cu = torch.tensor([0,len(values)],device=positions.device,dtype=torch.int32)
            hidden = self.model(input_ids, positions, None, inputs_embeds)
            if not self.native_full and len(self._local) != 16:
                raise RuntimeError('Incomplete ring initialization')
            if not self.standalone and not self.native_full:
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
        snapshots = None if self.standalone else self._snapshot_gdn()
        try:
            hidden = self._run('local', positions, embedding)
            full = False
            if not self.standalone:
                score_tensor = self._graphed_score if self._use_graph else self.admission(self._previous, embedding, hidden)
                score = float(score_tensor.item())
                full = not math.isfinite(score) or score > 0
                self._last_score = score if math.isfinite(score) else None
                self._stats['nonfinite_scores'] += int(not math.isfinite(score))
                if full:
                    self._restore_gdn(snapshots)
                    hidden = self._run('full', positions, embedding)
                    self._stats['full_reruns'] += 1
                self._previous.copy_(hidden)
            self._commit_attention()
            self._clock.commit('decode', 1)
            self._stats['routed_tokens'] += 1
            self._stats['logical_full_tokens'] += int(full)
            self._last_full = full
            if self._trace_enabled:
                self._route_trace.append(dict(position=self._clock.length-1, full=full, score=self._last_score))
            return hidden
        except BaseException:
            if snapshots is not None:
                self._restore_gdn(snapshots)
            self._clock.failed = True
            raise

    def runtime_report(self):
        result = super().runtime_report()
        result.update(format='qwen35-streaming-ring-v1', streaming_rope_shift=self.streaming,
                      standalone=self.standalone, native_full=self.native_full, stats=self._stream_stats,
                      graph_replays={k:v.replays for k,v in self._graphs.items()},
                      graph_enabled=self._use_graph, oracle=self._oracle,
                      route_trace=self._route_trace if self._trace_enabled else None)
        return result


class Qwen35FastLocal27B(Qwen35FastRouter27B):
    standalone = True


class Qwen35Streaming27B(Qwen35FastLocal27B):
    streaming = True


class Qwen35FastFull27B(Qwen35FastRouter27B):
    native_full = True
