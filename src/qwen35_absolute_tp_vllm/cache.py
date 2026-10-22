"""Persistent bounded KV storage and an independent PyTorch oracle."""
import math
import torch
import triton


def retained_positions(length, sink, recent, device=None):
    initial = torch.arange(min(length, sink), device=device)
    tail = torch.arange(min(length, max(sink, length - recent)), length, device=device)
    return torch.cat((initial, tail))


def rotate(x, positions, rotary_dim=64, theta=1e7):
    """Independent FP32 NeoX rotation, with explicit activation-dtype output."""
    freq = theta ** (-torch.arange(0, rotary_dim, 2, device=x.device).float() / rotary_dim)
    angle = positions.float()[:, None] * freq[None]
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    a, b = x[..., :rotary_dim // 2].float(), x[..., rotary_dim // 2:rotary_dim].float()
    out = x.clone()
    out[..., :rotary_dim // 2] = (a * cos - b * sin).to(x.dtype)
    out[..., rotary_dim // 2:rotary_dim] = (b * cos + a * sin).to(x.dtype)
    return out


class RingKV:
    def __init__(self, heads, kv_heads, head_dim, device, dtype, *, sink=4,
                 window=2048, shift=False, rotary_dim=64, theta=1e7):
        if window < 2 or sink < 0 or heads % kv_heads:
            raise ValueError("Invalid ring geometry")
        self.heads, self.kv_heads, self.dim = heads, kv_heads, head_dim
        self.sink, self.recent = sink, window - 1
        self.capacity = sink + self.recent
        self.shift, self.rotary_dim, self.theta = shift, rotary_dim, theta
        self.keys = torch.empty((kv_heads, self.capacity, head_dim), device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)
        self.position = torch.zeros((), device=device, dtype=torch.int64)
        freq = theta ** (-torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim)
        angle = torch.arange(self.capacity + 1, device=device).float()[:, None] * freq[None]
        self.cos, self.sin = angle.cos().contiguous(), angle.sin().contiguous()
        self.splits = triton.cdiv(self.capacity + 1, 16)
        self.partial = torch.empty((heads, self.splits, head_dim), dtype=torch.float32, device=device)
        self.maxima = torch.empty((heads, self.splits), dtype=torch.float32, device=device)
        self.sums = torch.empty_like(self.maxima)
        self.output = torch.empty((1, heads, head_dim), device=device, dtype=dtype)
        self.length = 0
        self.rope_reference = None

    def bind_native_rope(self, rope):
        """Use the model's quantized trigonometric cache, with unchanged layout."""
        if (rope.rotary_dim != self.rotary_dim or rope.head_size != self.dim
                or not rope.is_neox_style or not rope.mrope_interleaved):
            raise ValueError('Unsupported native partial text RoPE')
        cache = rope.cos_sin_cache[:self.capacity+1].to(self.keys.device, self.keys.dtype)
        cos, sin = cache.chunk(2, dim=-1)
        self.cos.copy_(cos.float())
        self.sin.copy_(sin.float())
        self.rope_reference = rope

    def seed(self, key, value, *, raw_selected_key=None):
        """Full initial prompt, then evict for one coming decode token.

        Shapes are [tokens, kv_heads, head_dim]. Selected raw keys must have
        been copied before the backbone's in-place positional encoding.
        """
        if key.shape != value.shape or key.ndim != 3 or key.shape[1:] != (self.kv_heads, self.dim):
            raise ValueError("Invalid prefill KV")
        length = key.shape[0]
        pos = retained_positions(length, self.sink, self.recent, key.device)
        slot = torch.where(pos < self.sink, pos, self.sink + pos % self.recent)
        selected = key.index_select(0, pos) if raw_selected_key is None else raw_selected_key
        if selected.shape != (pos.numel(), self.kv_heads, self.dim):
            raise ValueError("Raw-key selection mismatch")
        self.keys[:, slot] = selected.transpose(0, 1)
        self.values[:, slot] = value.index_select(0, pos).transpose(0, 1)
        self.length = length
        self.position.fill_(length)

    def attention(self, query, key, value):
        from .kernels import partials
        from qwen3_5_router_runtime_experiments.optimized_kernels import ring_reduce
        if query.shape != (1, self.heads, self.dim) or key.shape != (1, self.kv_heads, self.dim) or value.shape != key.shape:
            raise ValueError("Ring attention requires one decode token")
        if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
            raise ValueError("Ring attention requires contiguous one-token tensors")
        partials[(self.heads, self.splits)](
            query, self.keys, self.values, key, value, self.cos, self.sin, self.position,
            self.partial, self.maxima, self.sums, H=self.heads, HK=self.kv_heads, D=self.dim,
            SINK=self.sink, RECENT=self.recent, SPLITS=self.splits, SCALE=self.dim ** -0.5,
            SHIFT=self.shift, ROT=self.rotary_dim, TILE=16, BD=triton.next_power_of_2(self.dim),
            num_warps=8 if self.dim > 128 else 4, enable_fp_fusion=False)
        ring_reduce[(self.heads, triton.cdiv(self.dim, 32))](
            self.partial, self.maxima, self.sums, self.output, HEAD_DIM=self.dim,
            SPLITS=self.splits, SPLIT_BLOCK=triton.next_power_of_2(self.splits), DIM=32,
            num_warps=4)
        return self.output

    def commit(self, key, value):
        from .kernels import commit
        commit[(triton.cdiv(self.kv_heads * self.dim, 256),)](
            self.keys, self.values, key, value, self.position,
            D=self.dim, HK=self.kv_heads, SINK=self.sink, RECENT=self.recent, B=256)
        self.position.add_(1)
        self.length += 1

    def oracle(self, query, key, value):
        """Direct FP32 attention over materialized chronology; never timed."""
        pos = retained_positions(self.length, self.sink, self.recent, self.keys.device)
        slot = torch.where(pos < self.sink, pos, self.sink + pos % self.recent)
        k = torch.cat((self.keys[:, slot].transpose(0, 1), key), 0)
        v = torch.cat((self.values[:, slot].transpose(0, 1), value), 0)
        q = query
        if self.shift:
            if self.rope_reference is None:
                k = rotate(k, torch.arange(k.shape[0], device=k.device), self.rotary_dim, self.theta)
                q = rotate(q, torch.tensor([k.shape[0] - 1], device=k.device), self.rotary_dim, self.theta)
            else:
                positions = torch.arange(k.shape[0], device=k.device).expand(3,-1).contiguous()
                dummy = torch.zeros((k.shape[0], self.dim), device=k.device, dtype=k.dtype)
                _, rk = self.rope_reference.forward_cuda(positions, dummy, k.reshape(k.shape[0], -1).clone())
                qp = torch.full((3,1), k.shape[0]-1, device=k.device, dtype=torch.int64)
                rq, _ = self.rope_reference.forward_cuda(qp, q.reshape(1,-1).clone(), key.reshape(1,-1).clone())
                k, q = rk.reshape_as(k), rq.reshape_as(q)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        logits = torch.einsum('bhd,thd->bht', q.float(), k.float()) / math.sqrt(self.dim)
        return torch.einsum('bht,thd->bhd', logits.softmax(-1), v.float()).to(query.dtype)

    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.keys, self.values, self.position, self.cos, self.sin,
                    self.partial, self.maxima, self.sums, self.output))
