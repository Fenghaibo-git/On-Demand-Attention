"""Batch-one backbone CUDA graphs with setup rollback and explicit metadata."""
import copy
import torch
from vllm.forward_context import get_forward_context
from qwen3_5_router_runtime_experiments.native_qwen35_graph import NativeHybridCaptureRollback


class BackboneGraph:
    def __init__(self, owner, phase, positions, embedding, maximum):
        self.owner, self.phase, self.maximum = owner, phase, maximum
        self.positions, self.embedding = torch.empty_like(positions), torch.empty_like(embedding)
        self.positions.copy_(positions)
        self.embedding.copy_(embedding)
        current = get_forward_context()
        self.metadata, self.groups = {}, []
        seen = {}
        for name, meta in current.attn_metadata.items():
            if id(meta) not in seen:
                frozen = copy.copy(meta)
                tensors, scalars = {}, {}
                for key, value in vars(meta).items():
                    if isinstance(value, torch.Tensor):
                        tensor = value.clone()
                        setattr(frozen, key, tensor)
                        tensors[key] = tensor
                    else:
                        scalars[key] = value
                if hasattr(frozen, 'max_seq_len'):
                    frozen.max_seq_len = maximum
                seen[id(meta)] = frozen
                self.groups.append((name, frozen, tensors, scalars))
            self.metadata[name] = seen[id(meta)]
        self._refresh(current)
        rollback = NativeHybridCaptureRollback(
            owner._attention_layers if phase == 'full' else {}, owner._gdn_layers, current)
        original = current.attn_metadata
        current.attn_metadata = self.metadata
        current_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(current_stream)
        self.score = None
        def execute():
            output = owner.model(None, self.positions, None, self.embedding)
            score = owner.admission(owner._previous, self.embedding, output) if phase == 'local' and not owner.standalone else None
            return output, score
        try:
            with torch.cuda.stream(stream):
                for _ in range(2):
                    owner._phase = phase
                    execute()
                    rollback.restore()
            current_stream.wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                owner._phase = phase
                self.output, self.score = execute()
            current_stream.wait_stream(stream)
            rollback.restore()
            self.staged = dict(owner._staged)
        finally:
            current.attn_metadata = original
            rollback.restore()
        self.replays = 0

    def _refresh(self, context):
        for name, frozen, tensors, scalars in self.groups:
            live = context.attn_metadata[name]
            if type(live) is not type(frozen):
                raise ValueError('Captured attention metadata kind changed')
            for key, target in tensors.items():
                source = getattr(live, key)
                if source.shape != target.shape or source.dtype != target.dtype or source.device != target.device:
                    raise ValueError(f'Captured attention metadata layout changed: {key}')
                target.copy_(source)
            for key, value in scalars.items():
                actual = getattr(live, key)
                if key == 'max_seq_len':
                    if actual > self.maximum:
                        raise ValueError('Decode exceeds captured sequence bound')
                elif actual != value:
                    raise ValueError(f'Captured attention metadata mode changed: {key}')

    def __call__(self, positions, embedding):
        if positions.shape != self.positions.shape or embedding.shape != self.embedding.shape:
            raise ValueError('Captured input shape changed')
        self._refresh(get_forward_context())
        self.positions.copy_(positions)
        self.embedding.copy_(embedding)
        self.owner._phase = self.phase
        self.graph.replay()
        self.owner._staged = dict(self.staged)
        if self.score is not None:
            self.owner._graphed_score = self.score
        self.replays += 1
        return self.output
