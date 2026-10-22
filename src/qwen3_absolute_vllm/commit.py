"""One selected-token write launch for all Qwen3 attention layers."""
import torch
from types import SimpleNamespace
from qwen3_5_router_runtime_experiments.graphed_kv_commit_kernels import selected_commit_all


class SelectedCommit:
    def __init__(self, owner):
        self.owner, self.tables = owner, {}
        self.calls = 0

    def __call__(self):
        owner = self.owner
        phase = owner._phase
        if owner.standalone:
            context, meta = None, SimpleNamespace(slot_mapping=owner._decode_position)
        else:
            context, meta = owner._metadata(next(iter(owner._attention_layers.values())))
        tensors, rows = [], []
        block_size = None
        for index, module in owner._attention_layers.items():
            key, value = owner._staged[index]
            ring = owner._local[index]
            paged = ring.keys if owner.standalone else module.kv_cache[context.virtual_engine]
            if key.shape != (1, ring.kv_heads, ring.dim) or value.shape != key.shape or not key.is_contiguous() or not value.is_contiguous():
                raise ValueError('Require one contiguous BF16 token per layer')
            if not owner.standalone and (paged.ndim != 5 or paged.shape[0] != 2 or paged.shape[-2:] != (ring.kv_heads, ring.dim)):
                raise ValueError('Unexpected native FlashAttention cache layout')
            if ring.length != owner._clock.length:
                raise ValueError('Stale selected KV chronology')
            values = (key, value, ring.keys, ring.values, paged)
            if any(x.dtype != torch.bfloat16 or x.device != key.device for x in values):
                raise ValueError('All selected KV must be BF16 on the same device')
            tensors.extend(values)
            rows.append(tuple(x.data_ptr() for x in values) + ((0,0,0,0,0) if owner.standalone else tuple(paged.stride())))
            current_block = 1 if owner.standalone else paged.shape[2]
            if block_size is None:
                block_size = current_block
            elif block_size != current_block:
                raise ValueError('Mixed KV block geometry')
        signature = tuple(rows)
        table_key = owner._active_graph_key if owner._use_graph else phase
        if table_key not in self.tables or self.tables[table_key][0] != signature:
            if table_key in self.tables and owner._use_graph:
                raise ValueError('Captured selected-output pointers changed')
            self.tables[table_key] = (signature, torch.tensor(rows, dtype=torch.int64, device=key.device), tuple(tensors))
        table = self.tables[table_key][1]
        write_full = not owner.standalone and phase == 'local'
        selected_commit_all[(len(owner._attention_layers), (ring.kv_heads*ring.dim+255)//256)](
            table, meta.slot_mapping, owner._clock.length, WRITE_FULL=write_full,
            BLOCK_SIZE=block_size, HEAD_DIM=ring.dim, ELEMENTS=ring.kv_heads*ring.dim, CAPACITY=2051,
            SINKS=4, RECENT=2047, BLOCK=256)
        owner._decode_position.add_(1)
        for ring in owner._local.values():
            ring.length += 1
        owner._stream_stats['decode_full_kv_writes'] += int(write_full)*len(owner._attention_layers)
        self.calls += 1
