"""One exact selected-history write kernel for all Qwen3 attention layers.

Pointer tables are created during excluded warmup and retain references to
the stable CUDA-graph outputs and persistent Local allocations. Arithmetic
and attention order are unchanged. Full execution already wrote its paged
KV, whereas Local execution writes both the paged history and Local ring.
"""
import torch


def tensor_signature(value):
    return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


class AllLayerSelectedCommit:
    def __init__(self, owner):
        self.owner = owner
        self.tables = {}
        self.builds = self.calls = 0

    def _build(self, phase, context, meta):
        model = self.owner
        if len(model._attention_layers) != 28:
            raise ValueError('Require the admitted 28-layer Qwen3-1.7B')
        rows, references = [], []
        block_size = ring_pitch = None
        for index, module in model._attention_layers.items():
            key, value = model._staged[index]
            local = model._local[index]
            paged = module.kv_cache[context.virtual_engine]
            expected = (1, module.num_kv_heads, module.head_size)
            if expected != (1, 8, 128):
                raise ValueError('Require Qwen3-1.7B 8 KV heads of size 128')
            if key.shape != expected or value.shape != expected:
                raise ValueError('Require one staged Qwen3 KV token')
            if any(not x.is_contiguous() for x in (key, value)):
                raise ValueError('Require contiguous staged KV')
            stride = local.keys.stride()
            if (stride != local.values.stride() or stride[1:] != (128, 1)
                    or stride[0] not in (2051 * 128, 2052 * 128)):
                raise ValueError('Require a physical ring with at most one candidate tail slot')
            if ring_pitch is None:
                ring_pitch = stride[0] // 128
            elif stride[0] != ring_pitch * 128:
                raise ValueError('All layers must share the Local physical pitch')
            tensors = (key, value, local.keys, local.values, paged)
            if any(x.device != key.device or x.dtype != torch.bfloat16 for x in tensors):
                raise ValueError('Require unchanged unquantized BF16 KV on one device')
            if key.device.type != 'cuda':
                raise ValueError('Fused all-layer commit is a CUDA implementation')
            if paged.ndim != 5 or paged.shape[0] != 2 or tuple(paged.shape[-2:]) != expected[1:]:
                raise ValueError('Require FlashAttention [2,blocks,slots,heads,dim] cache')
            if block_size is None:
                block_size = paged.shape[2]
            elif paged.shape[2] != block_size:
                raise ValueError('All layers must share one paged block size')
            if (local.capacity, local.sink_size, local.recent_size) != (2051, 4, 2047):
                raise ValueError('Unexpected Local ring capacity')
            rows.append([x.data_ptr() for x in tensors] + list(paged.stride()))
            references.append((index, tensors, local, tuple(tensor_signature(x) for x in tensors)))
        table = torch.tensor(rows, dtype=torch.int64, device=references[0][1][0].device)
        self.tables[phase] = dict(table=table, references=references,
                                  block_size=block_size, ring_pitch=ring_pitch,
                                  slot_pointer=meta.slot_mapping.data_ptr())
        self.builds += 1

    def _validate(self, state, context, meta):
        model = self.owner
        if meta.slot_mapping.numel() != 1 or meta.slot_mapping.data_ptr() != state['slot_pointer']:
            raise ValueError('Serial slot metadata allocation changed after warmup')
        for index, tensors, local, storage in state['references']:
            key, value = model._staged[index]
            if key is not tensors[0] or value is not tensors[1] or model._local[index] is not local:
                raise ValueError('Graph KV outputs or persistent Local cache changed')
            current = (key, value, local.keys, local.values,
                       model._attention_layers[index].kv_cache[context.virtual_engine])
            if tuple(tensor_signature(x) for x in current) != storage:
                raise ValueError('Selected KV or Full/Local cache storage changed after warmup')
            if local.length != model._clock.length:
                raise ValueError('Selected-history length is stale')

    def __call__(self):
        model = self.owner
        phase = model._phase
        if phase not in ('local', 'full') or set(model._staged) != set(model._attention_layers):
            raise RuntimeError('Require one complete selected candidate')
        context, meta = model._metadata(next(iter(model._attention_layers.values())))
        if phase not in self.tables:
            self._build(phase, context, meta)
        state = self.tables[phase]
        self._validate(state, context, meta)
        from .graphed_kv_commit_kernels import selected_commit_all
        selected_commit_all[(len(state['references']), 4)](
            state['table'], meta.slot_mapping, model._clock.length,
            WRITE_FULL=phase == 'local', BLOCK_SIZE=state['block_size'],
            HEAD_DIM=128, ELEMENTS=1024, CAPACITY=state['ring_pitch'], SINKS=4, RECENT=2047, BLOCK=256,
        )
        for _, _, local, _ in state['references']:
            local.length += 1
        self.calls += 1
        model._staged.clear()

    def report(self):
        return dict(table_builds=self.builds, kernel_calls=self.calls,
                    kernels_per_decode=1, full_paged_write_only_on_local=True)
