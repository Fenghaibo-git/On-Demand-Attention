"""Opt-in CPU-only bulk layout checks; compile separately before model warmup."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

from .graphed_kv_commit import AllLayerSelectedCommit

_loaded = None


def native_layout_class():
    global _loaded
    if _loaded is not None:
        return _loaded.FrozenLayouts
    directory = os.environ.get('QWEN_LAYOUT_NATIVE_DIR')
    if not directory:
        raise ValueError('QWEN_LAYOUT_NATIVE_DIR must name a separately compiled, ABI-matched layout extension')
    root = Path(directory).resolve()
    metadata = json.loads((root / 'build.json').read_text())
    if metadata['torch_version'] != torch.__version__ or metadata['python_version'] != list(sys.version_info[:2]):
        raise ValueError('Layout extension Python/PyTorch ABI differs from the current process')
    source = Path(__file__).with_name('native_tensor_layouts.cpp')
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata['source_sha256']:
        raise ValueError('Layout extension source changed; rebuild before warmup')
    extension = Path(metadata['extension_path']).resolve()
    if extension.parent != root or hashlib.sha256(extension.read_bytes()).hexdigest() != metadata['extension_sha256']:
        raise ValueError('Layout extension artifact or path differs from its build metadata')
    spec = importlib.util.spec_from_file_location('qwen_tensor_layouts_native', extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _loaded = module
    return module.FrozenLayouts


class NativeCacheSignatures:
    """Take immutable snapshots of the same ordered tensor layouts as the baseline."""
    def __init__(self, owner, router):
        self.owner, self.router = owner, router
        self.layout_class = native_layout_class()

    def __call__(self, context):
        owner = self.owner
        layers = (owner._attention_layers.items() if self.router else
                  ((index, layer.self_attn.attn) for index, layer in enumerate(owner.model.layers)))
        indices, tensors = [], []
        for index, module in layers:
            indices.append(index)
            tensors.append(module.kv_cache[context.virtual_engine])
            if self.router:
                local = owner._local[index]
                tensors.extend((local.keys, local.values))
        return tuple(indices), self.layout_class(tuple(tensors))


class CompiledAllLayerSelectedCommit(AllLayerSelectedCommit):
    """Keep the inherited build/commit kernel and bulk-check layouts on the CPU."""
    def __init__(self, owner):
        super().__init__(owner)
        self.layout_class = native_layout_class()

    def _build(self, phase, context, meta):
        super()._build(phase, context, meta)
        state = self.tables[phase]
        state['compiled_layouts'] = self.layout_class(tuple(
            tensor for _, tensors, _, _ in state['references'] for tensor in tensors))

    def _validate(self, state, context, meta):
        model = self.owner
        if meta.slot_mapping.numel() != 1 or meta.slot_mapping.data_ptr() != state['slot_pointer']:
            raise ValueError('Serial slot metadata allocation changed after warmup')
        current = []
        for index, tensors, local, _ in state['references']:
            key, value = model._staged[index]
            if key is not tensors[0] or value is not tensors[1] or model._local[index] is not local:
                raise ValueError('Graph KV outputs or persistent Local cache changed')
            current.extend((key, value, local.keys, local.values,
                            model._attention_layers[index].kv_cache[context.virtual_engine]))
            if local.length != model._clock.length:
                raise ValueError('Selected-history length is stale')
        if not state['compiled_layouts'].matches(tuple(current)):
            raise ValueError('Selected KV or Full/Local cache storage changed after warmup')

    def report(self):
        result = super().report()
        result['layout_checks'] = 'native_cpu_bulk_pointer_shape_stride_dtype_device'
        return result
