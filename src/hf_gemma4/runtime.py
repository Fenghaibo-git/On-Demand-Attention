# Modified for On Demand Attention; see NOTICE.
"""Gemma native Full prefill and transactional selected-hybrid SDPA decode.

The official Gemma trunk executes BOTH candidates. Only its one-token global
attention mask differs for Local. Candidates never mutate the committed cache.
Native sliding caches keep their window1024; Full history of global layers is
retained. No Full-equivalent Local result reuse or new attention implementation.
"""
import ast
import contextlib
import copy
import os
import sys
import types
import torch
from torch import nn
from veomni.utils.tensor_state import tensor_state_sha256 as tensor_sha
from transformers import Gemma4UnifiedForConditionalGeneration, GenerationConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedCausalLMOutputWithPast

FLEX_KERNEL_OPTIONS = {'BLOCK_M': 32, 'BLOCK_N': 32}


@contextlib.contextmanager
def flex_prefill(model):
    """Use native FlexAttention only for prefill; keep audited SDPA decode."""
    configs = {}
    for module in model.modules():
        config = getattr(module, 'config', None)
        if config is not None and hasattr(config, '_attn_implementation'):
            configs[id(config)] = config
    saved = [(config, config._attn_implementation) for config in configs.values()]
    assert saved and all(value == 'sdpa' for _, value in saved), saved
    try:
        for config, _ in saved: config._attn_implementation = 'flex_attention'
        yield
    finally:
        for config, value in saved: config._attn_implementation = value


def canonicalize_rope(rope):
    """Same deterministic registration repair as the frozen ropefix constructor."""
    before = dict(rope._buffers)
    persistence = set(rope._non_persistent_buffers_set)
    rope._buffers = {name: rope._buffers[name] for name in sorted(rope._buffers)}
    assert all(rope._buffers[name] is value for name, value in before.items())
    assert rope._non_persistent_buffers_set == persistence


def audit_rope(rope):
    """Check installed native frequencies against a fresh CPU native reference."""
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextRotaryEmbedding
    reference = Gemma4UnifiedTextRotaryEmbedding(rope.config, device='cpu')
    canonicalize_rope(reference)
    assert list(rope._buffers) == sorted(reference._buffers)
    assert rope._non_persistent_buffers_set == reference._non_persistent_buffers_set
    records = []
    for name, value in rope._buffers.items():
        expected = reference._buffers[name]
        observed = value.detach().cpu()
        assert observed.dtype == expected.dtype == torch.float32
        assert torch.isfinite(observed).all()
        # Fixed FP32 frequency tolerance; no change to BF16 logit/state gates.
        torch.testing.assert_close(observed, expected, atol=1e-7, rtol=1e-6)
        records.append(dict(name=name, shape=list(observed.shape), dtype=str(observed.dtype),
                            max_abs=float((observed-expected).abs().max()),
                            sha256=tensor_sha({name: observed}), nonpersistent=True))
    return dict(status='passed', buffers=records, frequency_atol=1e-7, frequency_rtol=1e-6,
                native_max_position_embeddings=rope.config.max_position_embeddings,
                deterministic_buffer_order=True, distributed_inference=False)






def local_allowed(valid, sink=4, window=2048):
    valid = valid.bool(); ranks = valid.long().cumsum(-1); length = valid.sum(-1, keepdim=True)
    return valid & ((ranks <= sink) | (ranks > length - window))


def route_scores(scores, active, threshold=0.):
    finite = torch.isfinite(scores)
    return active & ((finite & (scores > threshold)) | ~finite), active & ~finite


class TransactionCache(Cache):
    """Read shared committed history; hold one candidate K/V per layer separately."""
    def __init__(self, base, indices):
        super().__init__(layers=base.layers)
        self.base = base; self.indices = indices; self.staged = {}

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        assert key_states.shape[-2] == 1 and layer_idx not in self.staged
        self.staged[layer_idx] = (key_states, value_states)
        previous = self.base.layers[layer_idx]
        indices = self.indices.to(key_states.device)
        keys = previous.keys.index_select(0, indices)
        values = previous.values.index_select(0, indices)
        return torch.cat((keys, key_states), -2), torch.cat((values, value_states), -2)


def commit(base, local, full, full_in_local, active_indices, batch_size):
    assert len(local.staged) == len(base.layers)
    if full is not None: assert len(full.staged) == len(base.layers)
    for i, layer in enumerate(base.layers):
        key, value = local.staged[i]
        if full is not None:
            key = key.clone(); value = value.clone()
            key.index_copy_(0, full_in_local.to(key.device), full.staged[i][0])
            value.index_copy_(0, full_in_local.to(value.device), full.staged[i][1])
        shape = (batch_size, key.shape[1], 1, key.shape[-1])
        selected_key = torch.zeros(shape, dtype=key.dtype, device=key.device)
        selected_value = torch.zeros(shape, dtype=value.dtype, device=value.device)
        selected_key.index_copy_(0, active_indices.to(key.device), key)
        selected_value.index_copy_(0, active_indices.to(value.device), value)
        # Inactive rows get masked, zero placeholders, never candidate state.
        base.update(selected_key, selected_value, i)


class TrackStopping(StoppingCriteria):
    def __init__(self, original, model): self.original = original; self.model = model
    def __call__(self, input_ids, scores, **kwargs):
        done = self.original(input_ids, scores, **kwargs)
        self.model._active_rows &= ~done.to(self.model._active_rows.device)
        return done


class GemmaSelectedHybrid(Gemma4UnifiedForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        canonicalize_rope(self.model.language_model.rotary_emb)

    def setup_router(self, head):
        self.router_head = head
        self.force_mode = None; self.trace = False; self.last_trace = None
        self.reset_router()

    def reset_router(self):
        self.previous_hidden = None; self._active_rows = None
        self.stats = dict(full_row_tokens=0, local_row_tokens=0, routed_row_tokens=0,
                          nonfinite_score_row_tokens=0, full_candidate_row_tokens=0,
                          local_candidate_row_tokens=0, prefill_row_tokens=0)

    @property
    def trunk(self): return self.model.language_model

    def _get_stopping_criteria(self, generation_config, stopping_criteria, tokenizer=None):
        original = super()._get_stopping_criteria(generation_config, stopping_criteria, tokenizer)
        return StoppingCriteriaList([TrackStopping(original, self)])

    def generate(self, *args, **kwargs):
        assert kwargs.get('do_sample', self.generation_config.do_sample) is False
        assert kwargs.get('num_beams', 1) == 1 and kwargs.get('num_return_sequences', 1) == 1
        self.reset_router()
        try: return super().generate(*args, **kwargs)
        finally: self.previous_hidden = None; self._active_rows = None

    def router_summary(self):
        s = dict(self.stats)
        assert s['full_row_tokens'] + s['local_row_tokens'] == s['routed_row_tokens']
        assert s['full_candidate_row_tokens'] == s['full_row_tokens']
        s.update(router_action_threshold=0., excludes_prefill=True,
                 full_fraction=s['full_row_tokens']/s['routed_row_tokens'] if s['routed_row_tokens'] else None)
        return s

    def candidate(self, embedding, positions, mask, base, indices, local):
        transaction = TransactionCache(base, indices)
        emb = embedding.index_select(0, indices); pos = positions.index_select(0, indices)
        valid = mask.index_select(0, indices)
        attention = valid
        if local:
            sliding = create_sliding_window_causal_mask(config=self.trunk.config, inputs_embeds=emb,
                         attention_mask=valid, past_key_values=transaction, position_ids=pos)
            attention = dict(full_attention=local_allowed(valid)[:, None, None, :], sliding_attention=sliding)
        output = self.trunk(inputs_embeds=emb, attention_mask=attention, position_ids=pos,
                            past_key_values=transaction, use_cache=True, return_dict=True)
        return output.last_hidden_state, transaction

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=True, logits_to_keep=1,
                router_active_mask=None, **kwargs):
        assert labels is None and use_cache is not False
        assert input_ids is not None and inputs_embeds is None, 'Frozen text token IDs required'
        assert attention_mask is not None and attention_mask.ndim == 2
        b, length = input_ids.shape
        if position_ids is None:
            position_ids = (attention_mask.long().cumsum(-1)-1).clamp_min(0)[:, -length:]
        if position_ids.shape[0] == 1: position_ids = position_ids.expand(b, -1)
        past = 0 if past_key_values is None else past_key_values.get_seq_length()
        if past == 0:
            self.reset_router(); self._active_rows = torch.ones(b, dtype=torch.bool, device=input_ids.device)
            last = torch.arange(length, device=input_ids.device).expand(b, -1).masked_fill(~attention_mask.bool(), -1).max(-1).values
            assert bool((last >= 0).all())
            def capture(module, args, hidden):
                self.previous_hidden = hidden[torch.arange(b, device=hidden.device), last.to(hidden.device)].detach().clone()
            hook = self.trunk.norm.register_forward_hook(capture)
            try:
                with flex_prefill(self):
                    result = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_values=past_key_values, use_cache=True, logits_to_keep=logits_to_keep,
                        kernel_options=FLEX_KERNEL_OPTIONS, **kwargs)
            finally: hook.remove()
            self.stats['prefill_row_tokens'] = int(attention_mask.sum())
            return result
        assert length == 1 and isinstance(past_key_values, DynamicCache)
        assert self.previous_hidden is not None
        active = self._active_rows.clone()
        if router_active_mask is not None: active &= router_active_mask.to(active.device)
        active &= attention_mask[:, -1].bool()
        self._active_rows = active
        mask = attention_mask.clone(); mask[:, -1] = active.to(mask.dtype)
        indices = active.nonzero(as_tuple=False).flatten()
        assert indices.numel() > 0, 'No active decode rows'
        embedding = self.trunk.embed_tokens(input_ids)
        local_hidden, local_cache = self.candidate(embedding, position_ids, mask, past_key_values, indices, True)
        features = torch.cat((self.previous_hidden.index_select(0, indices),
                              embedding.index_select(0, indices)[:, 0], local_hidden[:, 0]), -1)
        scores = self.router_head(features.float())
        ones = torch.ones_like(scores, dtype=torch.bool)
        selected_full, nonfinite = route_scores(scores, ones)
        if self.force_mode == 'full': selected_full = ones
        elif self.force_mode == 'local': selected_full = ~ones
        elif isinstance(self.force_mode, torch.Tensor): selected_full = self.force_mode.to(scores.device).index_select(0, indices)
        full_in_local = selected_full.nonzero(as_tuple=False).flatten()
        full_cache = None; selected = local_hidden.clone()
        if full_in_local.numel():
            full_indices = indices.index_select(0, full_in_local)
            full_hidden, full_cache = self.candidate(embedding, position_ids, mask, past_key_values, full_indices, False)
            selected.index_copy_(0, full_in_local, full_hidden)
        if self.trace:
            self.last_trace = dict(scores=scores.detach().cpu(), full=selected_full.detach().cpu(),
                active=active.detach().cpu(), embedding=embedding.detach().cpu(), features=features.detach().cpu(),
                local_hidden=local_hidden.detach().cpu(), selected_hidden=selected.detach().cpu(),
                local_kv={i: tuple(v.detach().cpu() for v in pair) for i, pair in local_cache.staged.items()},
                full_kv={} if full_cache is None else {i: tuple(v.detach().cpu() for v in pair) for i,pair in full_cache.staged.items()})
        commit(past_key_values, local_cache, full_cache, full_in_local, indices, b)
        self.previous_hidden = self.previous_hidden.clone()
        self.previous_hidden.index_copy_(0, indices, selected[:, 0])
        count = indices.numel(); full_count = int(selected_full.sum())
        self.stats['full_row_tokens'] += full_count; self.stats['local_row_tokens'] += count-full_count
        self.stats['routed_row_tokens'] += count; self.stats['full_candidate_row_tokens'] += full_count
        self.stats['local_candidate_row_tokens'] += count; self.stats['nonfinite_score_row_tokens'] += int(nonfinite.sum())
        logits = self.lm_head(self.previous_hidden[:, None])
        cap = self.trunk.config.final_logit_softcapping
        if cap is not None: logits = torch.tanh(logits / cap) * cap
        if bool(nonfinite.any()) and self.force_mode is None:
            raise FloatingPointError('Nonfinite Router score: fail-safe Full committed; formal batch rejected')
        return Gemma4UnifiedCausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
