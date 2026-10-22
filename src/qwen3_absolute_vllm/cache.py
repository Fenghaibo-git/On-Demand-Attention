# Modified for On Demand Attention; see NOTICE.
"""Qwen3 full-dimension RoPE binding; shared StreamingLLM ring kernel."""
import math
import torch
from qwen35_2b_absolute_vllm.cache import RingKV as SharedRingKV, retained_positions


class RingKV(SharedRingKV):
    def bind_native_rope(self, rope):
        if (rope.rotary_dim != self.rotary_dim or rope.head_size != self.dim
                or not rope.is_neox_style):
            raise ValueError('Expected Qwen3 full-dimension NeoX RoPE')
        cache = rope.cos_sin_cache[:self.capacity+1].to(self.keys.device, self.keys.dtype)
        cos, sin = cache.chunk(2, dim=-1)
        self.cos.copy_(cos.float())
        self.sin.copy_(sin.float())
        self.rope_reference = rope

    def oracle(self, query, key, value):
        """Independent materialized PyTorch attention + native Qwen3 RoPE."""
        pos = retained_positions(self.length, self.sink, self.recent, self.keys.device)
        slot = torch.where(pos < self.sink, pos, self.sink + pos % self.recent)
        k = torch.cat((self.keys[:, slot].transpose(0, 1), key), 0)
        v = torch.cat((self.values[:, slot].transpose(0, 1), value), 0)
        q = query
        if self.shift:
            positions = torch.arange(k.shape[0], device=k.device)
            dummy = torch.zeros((k.shape[0], self.kv_heads * self.dim), device=k.device, dtype=k.dtype)
            _, rk = self.rope_reference.forward_cuda(positions, dummy, k.reshape(k.shape[0], -1).clone())
            qp = positions[-1:]
            rq, _ = self.rope_reference.forward_cuda(qp, q.reshape(1,-1).clone(), key.reshape(1,-1).clone())
            k, q = rk.reshape_as(k), rq.reshape_as(q)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        logits = torch.einsum('bhd,thd->bht', q.float(), k.float()) / math.sqrt(self.dim)
        return torch.einsum('bht,thd->bhd', logits.softmax(-1), v.float()).to(query.dtype)
