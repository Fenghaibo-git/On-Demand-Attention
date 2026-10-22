"""Qwen3.5 geometry for the original bit-preserving selected-write kernel."""
import torch

from .graphed_kv_commit import tensor_signature


class Qwen35SelectedCommit:
    def __init__(self, owner, *, _device_type='cuda'):
        self.owner, self.device_type = owner, _device_type
        self.tables = {}
        self.builds = 0

    def _build(self, phase, context, meta):
        owner = self.owner
        if tuple(owner._attention_layers) != (3, 7, 11, 15, 19, 23):
            raise ValueError('Require the six Qwen3.5-2B ordinary attention layers')
        if phase not in ('local', 'full') or set(owner._staged) != set(owner._attention_layers):
            raise ValueError('Require one complete selected candidate')
        rows, references = [], []
        block_size = None
        for index, module in owner._attention_layers.items():
            key, value = owner._staged[index]
            local = owner._local[index]
            paged = module.kv_cache[context.virtual_engine]
            if (module.num_kv_heads, module.head_size) != (2, 256):
                raise ValueError('Require Qwen3.5-2B two KV heads of size 256')
            if key.shape != (1, 2, 256) or value.shape != key.shape:
                raise ValueError('Require one staged Qwen3.5 KV token')
            if any(not x.is_contiguous() for x in (key, value)):
                raise ValueError('Require contiguous staged KV')
            if (local.capacity, local.sink_size, local.recent_size) != (2051, 4, 2047):
                raise ValueError('Unexpected Local ring capacity')
            if any(x.shape != (2, 2051, 256) or x.stride() != (2051*256, 256, 1)
                   for x in (local.keys, local.values)):
                raise ValueError('Require the original direct-ring layout')
            tensors = (key, value, local.keys, local.values, paged)
            if (key.device.type != self.device_type or any(x.device != key.device
                    or x.dtype != torch.bfloat16 for x in tensors)):
                raise ValueError('Require unquantized BF16 KV on the configured device')
            if paged.ndim != 5 or paged.shape[0] != 2 or tuple(paged.shape[-2:]) != (2, 256):
                raise ValueError('Require FlashAttention paged KV layout')
            if block_size is None:
                block_size = paged.shape[2]
            elif paged.shape[2] != block_size:
                raise ValueError('All layers must share the paged block size')
            rows.append([x.data_ptr() for x in tensors] + list(paged.stride()))
            references.append((index, tensors, local, tuple(map(tensor_signature, tensors))))
        self.tables[phase] = dict(
            table=torch.tensor(rows, dtype=torch.int64, device=references[0][1][0].device),
            references=references, block_size=block_size, ring_pitch=2051,
            slot_pointer=meta.slot_mapping.data_ptr())
        self.builds += 1
