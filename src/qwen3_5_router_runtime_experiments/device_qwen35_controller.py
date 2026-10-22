"""Conditional Local/Full graphs for the pinned Qwen3.5-2B text backbone.

Production capture uses CUDA only when explicitly invoked. The injected CPU
backend validates state transitions; it is not a CUDA or model-speed result.
"""
import os
from types import SimpleNamespace

import torch

from .device_capture_state import DecodeCaptureRollback
from .device_qwen3_controller import Qwen3DeviceController, TorchCaptureBackend, current_metadata, metadata_layout
from .device_qwen35_commit import Qwen35SelectedCommit
from .graphed_kv_commit import tensor_signature
from .prepared_gdn_state import PreparedGDNStates


def layer_name(module):
    return module.layer_name if hasattr(module, 'layer_name') else module.prefix


def metadata_value(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', tensor_signature(value))
    if value is None or type(value) in (int, bool, float, str):
        return (type(value).__name__, value)
    if isinstance(value, dict):
        return ('dict', tuple((key, metadata_value(item)) for key, item in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(map(metadata_value, value)))
    raise ValueError('Unsupported captured GDN metadata field type')


def metadata_items(metadata, names, *, serial=False):
    """Serialize shared metadata once per call, retaining every layer binding.

    vLLM shares metadata within an attention group. Do not cache across decode
    steps: both scalar modes and tensor layouts may change in the same object.
    Distinct objects are checked independently even when their fields alias.
    """
    signatures, result = {}, []
    for name in names:
        item = metadata[name]
        key = id(item)
        value = signatures.get(key)
        if value is None:
            if serial and (item.num_actual_tokens != 1 or item.num_decodes != 1
                           or item.num_prefills != 0 or item.spec_sequence_masks is not None):
                raise ValueError('Require one native non-speculative GDN decode token')
            value = metadata_value(vars(item))
            signatures[key] = value
        result.append((name, value))
    return tuple(result)


class Qwen35CaptureBackend(TorchCaptureBackend):
    def commit(self, state, slots, position, write_full):
        from .device_selected_commit import selected_commit_all_device_position
        selected_commit_all_device_position[(len(state['references']), 2)](
            state['table'], slots, position, WRITE_FULL=write_full,
            BLOCK_SIZE=state['block_size'], HEAD_DIM=256, ELEMENTS=512,
            CAPACITY=2051, SINKS=4, RECENT=2047, BLOCK=256)


class Qwen35DeviceController(Qwen3DeviceController):
    def __init__(self, owner, maximum, directory, *, _backend=None, _commit_factory=None):
        super().__init__(owner, maximum, directory, _backend=_backend or Qwen35CaptureBackend(),
                         _commit_factory=_commit_factory or Qwen35SelectedCommit)
        self.replay_started = False

    def _flash_context(self, context):
        if not isinstance(context.attn_metadata, dict):
            raise ValueError('Require per-layer hybrid metadata')
        return SimpleNamespace(virtual_engine=context.virtual_engine, attn_metadata={
            layer_name(module): context.attn_metadata[layer_name(module)]
            for module in self.owner._attention_layers.values()})

    def _gdn_metadata(self, context):
        if (tuple(self.owner._gdn_layers) != tuple(index for index, _ in self.state.layers)
                or any(self.owner._gdn_layers[index] is not module for index, module in self.state.layers)):
            raise ValueError('Captured GDN registry changed')
        if self.deduplicate_gdn_metadata:
            return metadata_items(context.attn_metadata, self.state.names)
        return tuple((name, metadata_value(vars(context.attn_metadata[name]))) for name in self.state.names)

    def _cache_tensors(self, context):
        result = []
        for index, module in self.owner._attention_layers.items():
            local = self.owner._local[index]
            result.extend((module.kv_cache[context.virtual_engine], local.keys, local.values))
            result.extend(local._workspace or ())
            result.append(local._device_position)
        for module in self.owner._gdn_layers.values():
            result.extend(module.kv_cache[context.virtual_engine])
        result.extend(backup for _, _, backup in self.state.entries)
        return tuple(result)

    def _bind_modules(self):
        owner = self.owner
        self.modules = tuple(owner.model.modules()) + tuple(owner.admission.modules())
        self.model_owner, self.admission_owner = owner.model, owner.admission
        self._check_hooks()
        self.parameters = tuple((module, name, p, None if p.is_inference() else p._version, tensor_signature(p))
            for module in self.modules for name, p in module._parameters.items() if p is not None)
        self._parameter_layout = None
        if os.environ.get('Q35_COMPILED_MODULE_CHECKS', '0') == '1':
            from .compiled_device_modules import native_module_class, native_module_binding, vllm_parameter_types, VLLM_PARAMETER_SHA256
            kinds = vllm_parameter_types()
            self._module_check = native_module_class()(self.modules, kinds)
            self._module_check_binding = native_module_binding()
            self._module_check_binding.update(parameter_types=[f'{kind.__module__}.{kind.__name__}' for kind in kinds],
                vllm_parameter_source_sha256=VLLM_PARAMETER_SHA256 if kinds else None)
            self._check_hooks()

    def _capture(self, ids, positions, context):
        if self.frozen:
            raise RuntimeError('Device route requires excluded warmup before frozen execution')
        owner = self.owner
        meta = current_metadata(self._flash_context(context), self.maximum)
        if ids.device.type != self.backend.device_type or torch.is_grad_enabled():
            raise ValueError('Require inference on the configured device')
        self._bind_modules()
        if any(p.dtype != torch.float32 for p in owner.admission.router_head.parameters()):
            raise ValueError('The real Router must remain FP32')
        if self.bound_inputs:
            self.ids, self.positions = ids, positions
        else:
            self.ids = torch.empty_like(ids)
            self.positions = torch.empty(positions.shape, dtype=positions.dtype, device=positions.device)
            self.ids.copy_(ids)
            self.positions.copy_(positions)
        # A producer MRoPE view is strided across its three rows. Take the
        # first row directly; flattening it could allocate a stale copy.
        self.decode_position = self.positions[0, :1] if self.positions.ndim == 2 else self.positions[:1]
        self.previous = owner._previous.detach().clone()
        self.counters = torch.zeros(4, dtype=torch.int64, device=ids.device)
        self.device, self.threshold = ids.device, owner._contract['threshold']
        self.virtual_engine, self.metadata = context.virtual_engine, metadata_layout(meta)
        self.local_owners = tuple(owner._local.items())
        self.prefix_reuse = owner._matrix_reuse
        self.state = PreparedGDNStates(owner, backend='torch' if self.backend.device_type == 'cpu' else 'triton',
            compiled_layouts=os.environ.get('Q35_COMPILED_STATE_LAYOUTS', '0') == '1')
        self.state.snapshot()
        self.state.freeze()
        owner._prepared_gdn_store = owner._direct_state_store = self.state
        # Select the optimization during excluded setup. Engines with no
        # shared metadata retain the original direct per-layer path.
        self.deduplicate_gdn_metadata = len({id(context.attn_metadata[name])
            for name in self.state.names}) < len(self.state.names)
        self.gdn_metadata = self._gdn_metadata(context)
        for local in owner._local.values():
            local.bind_position(self.decode_position)
        # These row reads occur only in excluded setup. Graph replay reads indices
        # on device; setup backups are independent of the graph's live snapshots.
        state_views = []
        for source, index, backup in self.state.entries:
            row = int(index.item())
            if not 0 <= row < source.shape[0]:
                raise ValueError('Capture GDN state row is out of bounds')
            state_views.extend((source[row:row+1], backup))
        rollback = DecodeCaptureRollback(owner, context, meta, (self.previous, self.counters, *state_views))
        self.rollback_bytes = rollback.saved_bytes
        old_max, old_prefix = meta.max_seq_len, owner._matrix_prefix
        old_skips = getattr(owner, '_matrix_prefix_skips', 0)
        self.stage_graphs = []
        try:
            with rollback:
                meta.max_seq_len = self.maximum

                def prelude():
                    self.state.full_plan.copy(False)
                    owner._phase = 'local'
                    owner._staged.clear()
                    owner._matrix_prefix = None
                    embedding = owner.embed_input_ids(self.ids)
                    hidden = owner.model(None, self.positions, None, embedding)
                    score = owner.admission.router_head(self.previous, embedding, hidden)
                    return embedding, hidden, score, dict(owner._staged), owner._matrix_prefix

                graph, result = self.backend.capture(prelude)
                self.embedding, self.local_hidden, self.score, staged, self.prefix = result
                self.stage_graphs.append(graph)
                if self.score.shape != (1,) or self.score.dtype != torch.float32:
                    raise ValueError('Real Router must produce one FP32 score')
                if self.prefix_reuse and (not isinstance(self.prefix, tuple) or len(self.prefix) != 2):
                    raise ValueError('Missing current-token shared prefix')
                owner._staged = dict(staged)
                self.commit._build('local', context, meta)

                def full():
                    plan = self.state.prefix_plan if self.prefix_reuse else self.state.full_plan
                    plan.copy(True)
                    owner._phase = 'full'
                    owner._staged.clear()
                    owner._matrix_prefix = self.prefix
                    skips = getattr(owner, '_matrix_prefix_skips', 0)
                    try:
                        hidden = owner.model(None, self.positions, None, self.embedding)
                        return hidden, dict(owner._staged)
                    finally:
                        # Graph replay does not run Python layer wrappers. Account
                        # for skipped prefix layers once via post-timing counters.
                        owner._matrix_prefix_skips = skips

                full_graph, (self.full_hidden, staged) = self.backend.capture(full)
                self.stage_graphs.append(full_graph)
                owner._staged = dict(staged)
                self.commit._build('full', context, meta)

                def selected(phase, hidden):
                    self.backend.commit(self.commit.tables[phase], meta.slot_mapping,
                                        self.decode_position, phase == 'local')
                    self.previous.copy_(hidden)

                local_commit, _ = self.backend.capture(lambda: selected('local', self.local_hidden))
                full_commit, _ = self.backend.capture(lambda: selected('full', self.full_hidden))
                self.stage_graphs.extend((local_commit, full_commit))
                stages = (graph, (full_graph, full_commit), local_commit)
                for policy in self.POLICIES:
                    self.native[policy] = self.backend.native(self.directory, stages, self.score,
                        self.counters, self.threshold, policy, self)
                    rollback.restore()
                    for _ in range(3 if policy == 2 else 1):
                        self.backend.replay(self.native[policy])
                self.captures = 4
        except BaseException:
            self.close()
            raise
        finally:
            meta.max_seq_len = old_max
            owner._matrix_prefix, owner._matrix_prefix_skips = old_prefix, old_skips
        self.cache_tensors = self._cache_tensors(context)
        if os.environ.get('Q35_COMPILED_STATE_LAYOUTS', '0') == '1':
            from .compiled_layout_checks import native_layout_class
            self._bulk_layout = native_layout_class()(self.cache_tensors)
        else:
            self.cache_layout = tuple(map(tensor_signature, self.cache_tensors))
        self.input_layout = (tensor_signature(self.ids), tensor_signature(self.positions))
        self.buffer_layout = tuple(map(tensor_signature, (self.previous, self.counters, self.score,
                                                         self.embedding, self.local_hidden, self.full_hidden)))
        self.prefix_layout = tuple(map(tensor_signature, self.prefix or ()))
        self.position_layout = tensor_signature(self.decode_position)
        self.ready = True
        owner._previous = self.previous
        owner._staged.clear()

    def _guard(self, ids, positions, context):
        if self.owner._matrix_reuse != self.prefix_reuse:
            raise ValueError('Prefix mode needs a separately captured engine')
        if self._gdn_metadata(context) != self.gdn_metadata:
            raise ValueError('Captured GDN metadata allocation or mode changed')
        if (tuple(map(tensor_signature, self.prefix or ())) != self.prefix_layout
                or tensor_signature(self.decode_position) != self.position_layout):
            raise ValueError('Captured prefix or device position storage changed')
        super()._guard(ids, positions, self._flash_context(context))

    def replay(self, ids, positions, context, policy):
        self.replay_started = False
        if policy not in self.POLICIES:
            raise ValueError('Policy has no independently warmed native graph')
        if not self.ready:
            self._capture(ids, positions, context)
        self._guard(ids, positions, context)
        if not self.bound_inputs:
            self.ids.copy_(ids)
            self.positions.copy_(positions)
        self.replay_started = True
        self.backend.replay(self.native[policy])
        self.replays += 1
        self.owner._phase = 'device'
        self.owner._staged.clear()
        return self.previous

    def restore_failed_gdn(self):
        if self.ready and self.replay_started:
            self.state.full_plan.copy(True)

    def report(self):
        result = super().report()
        result.update(architecture='Qwen3.5-2B', prefix_reuse=getattr(self, 'prefix_reuse', None),
            local_attention='original_chronological_ring_kernel_with_device_position',
            gdn_snapshot='one_original_copy_kernel_inside_prelude',
            gdn_restore='one_original_copy_kernel_on_Full',
            snapshot_kernels_per_decode=1, restore_kernels_per_Full=1,
            gdn_metadata_checks=('once_per_distinct_object_per_decode'
                if getattr(self, 'deduplicate_gdn_metadata', False) else 'original_per_layer'),
            prepared_state_entries=len(self.state.entries) if hasattr(self, 'state') else 0)
        return result
