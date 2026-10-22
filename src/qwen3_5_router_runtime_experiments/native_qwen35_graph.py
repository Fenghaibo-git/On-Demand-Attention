"""Capture the native hybrid backbone with setup-only recurrent rollback."""
import os
import types

import torch

from .device_qwen3_controller import TorchCaptureBackend, current_metadata, metadata_layout
from .device_qwen35_controller import layer_name, metadata_items, metadata_value
from .graphed_kv_commit import tensor_signature


class NativeHybridCaptureRollback:
    """Back up only rows/slots touched by excluded one-token graph setup."""
    def __init__(self, attention, recurrent, context):
        views = []
        for module in attention.values():
            meta = context.attn_metadata[layer_name(module)]
            if meta.slot_mapping.numel() != 1:
                raise ValueError('Native graph setup requires one current KV slot')
            slot = int(meta.slot_mapping.item())
            paged = module.kv_cache[context.virtual_engine]
            if paged.ndim != 5 or paged.shape[0] != 2:
                raise ValueError('Require native FlashAttention paged KV layout')
            block, offset = divmod(slot, paged.shape[2])
            if slot < 0 or block >= paged.shape[1]:
                raise ValueError('Native graph setup has no valid paged slot')
            views.append(paged[:, block, offset])
        for module in recurrent.values():
            meta = context.attn_metadata[layer_name(module)]
            index = meta.non_spec_state_indices_tensor
            if index.shape != (1,) or index.dtype not in (torch.int32, torch.int64):
                raise ValueError('Require one non-speculative GDN state index')
            row = int(index.item())
            banks = module.kv_cache[context.virtual_engine]
            if len(banks) != 2:
                raise ValueError('Require conv and recurrent state banks')
            for bank in banks:
                if bank.ndim < 2 or not 0 <= row < bank.shape[0]:
                    raise ValueError('Native graph setup has no valid GDN row')
                views.append(bank[row:row+1])
        self.copies = tuple((view, view.clone()) for view in views)

    def restore(self):
        with torch.no_grad():
            for target, saved in self.copies:
                target.copy_(saved)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.restore()

    @property
    def saved_bytes(self):
        return sum(saved.numel()*saved.element_size() for _, saved in self.copies)


class NativeQwen35Graph:
    """Retain native forward and sample outside the captured backbone."""
    def __init__(self, owner, model, maximum, get_context, *, _backend=None):
        from .graph_input_mode import bound_graph_inputs_enabled
        self.bound_inputs = bound_graph_inputs_enabled()
        self.owner, self.model = owner, model
        self.original = model.forward
        self.maximum, self.get_context = maximum, get_context
        self.backend = _backend or TorchCaptureBackend()
        self.attention, self.recurrent = {}, {}
        for index, layer in enumerate(model.layers):
            if layer.layer_type == 'full_attention':
                self.attention[index] = layer.self_attn.attn
            elif layer.layer_type == 'linear_attention':
                self.recurrent[index] = layer.linear_attn
            else:
                raise ValueError('Unexpected native hybrid layer type')
        if (tuple(self.attention) != (3, 7, 11, 15, 19, 23)
                or tuple(self.recurrent) != tuple(i for i in range(24) if i % 4 != 3)):
            raise ValueError('Require the native Qwen3.5-2B hybrid architecture')
        self.graph = self.output = None
        self.captures = self.replays = 0
        self.frozen = self.failed = False
        self._bulk_layout = self._module_check = None
        self.deduplicate_gdn_metadata = None

        def redirected(module, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
            return self(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

        self.redirected = types.MethodType(redirected, model)
        model.forward = self.redirected

    def _metadata(self, context):
        if not isinstance(context.attn_metadata, dict):
            raise ValueError('Require native per-layer hybrid metadata')
        flash = {layer_name(module): context.attn_metadata[layer_name(module)] for module in self.attention.values()}
        meta = current_metadata(types.SimpleNamespace(attn_metadata=flash), self.maximum)
        if self.deduplicate_gdn_metadata is None:
            self.deduplicate_gdn_metadata = len({id(context.attn_metadata[layer_name(module)])
                for module in self.recurrent.values()}) < len(self.recurrent)
        if self.deduplicate_gdn_metadata:
            names = tuple(layer_name(module) for module in self.recurrent.values())
            recurrent = metadata_items(context.attn_metadata, names, serial=True)
            return meta, (metadata_layout(meta), recurrent)
        # No sharing at capture: retain the previous direct path without
        # allocating a per-decode deduplication dictionary.
        recurrent = []
        for module in self.recurrent.values():
            name = layer_name(module)
            item = context.attn_metadata[name]
            if (item.num_actual_tokens != 1 or item.num_decodes != 1 or item.num_prefills != 0
                    or item.spec_sequence_masks is not None):
                raise ValueError('Require one native non-speculative GDN decode token')
            recurrent.append((name, metadata_value(vars(item))))
        return meta, (metadata_layout(meta), tuple(recurrent))

    def _cache_tensors(self, context):
        result = [module.kv_cache[context.virtual_engine] for module in self.attention.values()]
        for module in self.recurrent.values():
            result.extend(module.kv_cache[context.virtual_engine])
        return tuple(result)

    def _check_hooks(self):
        global_hooks = torch.nn.modules.module
        if global_hooks._global_forward_hooks or global_hooks._global_forward_pre_hooks:
            raise ValueError('Native graph requires inference without Python callbacks')
        if self._module_check is not None:
            self._module_check.check(None, None)
            return
        for module in self.modules:
            if module.training or module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
                raise ValueError('Native graph requires inference without Python callbacks')

    def _bind_modules(self):
        self.modules = tuple(self.model.modules())
        self.layer_registry = tuple(self.model.layers)
        self._check_hooks()
        self.parameters = tuple((module, name, p, None if p.is_inference() else p._version, tensor_signature(p))
            for module in self.modules for name, p in module._parameters.items() if p is not None)
        if os.environ.get('Q35_COMPILED_MODULE_CHECKS', '0') == '1':
            from .compiled_device_modules import native_module_class, native_module_binding, vllm_parameter_types, VLLM_PARAMETER_SHA256
            kinds = vllm_parameter_types()
            self._module_check = native_module_class()(self.modules, kinds)
            self._module_check_binding = native_module_binding()
            self._module_check_binding.update(parameter_types=[f'{kind.__module__}.{kind.__name__}' for kind in kinds],
                vllm_parameter_source_sha256=VLLM_PARAMETER_SHA256 if kinds else None)
            self._check_hooks()

    def _capture(self, input_ids, positions, inputs_embeds, context):
        from .qwen35_state_contract import validate_qwen35_state_banks
        validate_qwen35_state_banks(self.recurrent, context)
        if self.frozen:
            raise RuntimeError('Native graph needs excluded warmup before frozen execution')
        if torch.is_grad_enabled() or positions.device.type != self.backend.device_type:
            raise ValueError('Require native inference on the configured device')
        meta, self.metadata = self._metadata(context)
        self.virtual_engine = context.virtual_engine
        self._bind_modules()
        if self.bound_inputs:
            self.tensors = (input_ids, positions, inputs_embeds)
        else:
            self.tensors = tuple(None if value is None else torch.empty(value.shape, dtype=value.dtype, device=value.device)
                                 for value in (input_ids, positions, inputs_embeds))
            for target, source in zip(self.tensors, (input_ids, positions, inputs_embeds)):
                if target is not None:
                    target.copy_(source)
        rollback = NativeHybridCaptureRollback(self.attention, self.recurrent, context)
        self.capture_rollback_bytes = rollback.saved_bytes
        old_max = meta.max_seq_len
        try:
            with rollback:
                meta.max_seq_len = self.maximum
                self.graph, self.output = self.backend.capture(
                    lambda: self.original(self.tensors[0], self.tensors[1], None, self.tensors[2]))
        except BaseException:
            self.graph = self.output = None
            raise
        finally:
            meta.max_seq_len = old_max
        self.cache_tensors = self._cache_tensors(context)
        if os.environ.get('Q35_COMPILED_STATE_LAYOUTS', '0') == '1':
            from .compiled_layout_checks import native_layout_class
            self._bulk_layout = native_layout_class()(self.cache_tensors)
        else:
            self.cache_layout = tuple(map(tensor_signature, self.cache_tensors))
        self.input_layout = tuple(None if value is None else tensor_signature(value) for value in self.tensors)
        self.output_layout = tensor_signature(self.output)
        self.captures += 1

    def _guard(self, input_ids, positions, inputs_embeds, context):
        if torch.is_grad_enabled() or self.owner.language_model.model is not self.model:
            raise ValueError('Native graph model or inference mode changed')
        if self.model.forward is not self.redirected or tuple(self.model.layers) != self.layer_registry:
            raise ValueError('Captured native layer registry or forward changed')
        _, signature = self._metadata(context)
        if context.virtual_engine != self.virtual_engine or signature != self.metadata:
            raise ValueError('Native graph metadata allocation or mode changed')
        for index, module in self.attention.items():
            if self.model.layers[index].self_attn.attn is not module:
                raise ValueError('Captured native attention object changed')
        for index, module in self.recurrent.items():
            if self.model.layers[index].linear_attn is not module:
                raise ValueError('Captured native GDN object changed')
        current = self._cache_tensors(context)
        valid = self._bulk_layout.matches(current) if self._bulk_layout is not None else tuple(map(tensor_signature, current)) == self.cache_layout
        if not valid:
            raise ValueError('Native graph cache allocation changed')
        for target, source, saved in zip(self.tensors, (input_ids, positions, inputs_embeds), self.input_layout):
            if (target is None) != (source is None):
                raise ValueError('Native graph input mode changed')
            if target is not None and (tensor_signature(target) != saved or target.shape != source.shape
                    or target.dtype != source.dtype or target.device != source.device):
                raise ValueError('Native graph input storage/shape/dtype/device changed')
            if self.bound_inputs and source is not None and tensor_signature(source) != saved:
                raise ValueError('Bound native input allocation or layout changed')
        if tensor_signature(self.output) != self.output_layout:
            raise ValueError('Native graph output storage changed')
        self._check_hooks()
        if self._module_check is None:
            for module, name, p, version, layout in self.parameters:
                if (module._parameters.get(name) is not p or (version is not None and p._version != version)
                        or tensor_signature(p) != layout):
                    raise ValueError('Captured native parameter changed')

    def __call__(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        context = self.get_context()
        token_count = positions.shape[-1]
        if context.attn_metadata is None or token_count != 1:
            return self.original(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        # A one-token prompt is still prefill; metadata distinguishes it from decode.
        first_gdn = context.attn_metadata[layer_name(next(iter(self.recurrent.values())))]
        if first_gdn.num_prefills:
            return self.original(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if self.failed:
            raise RuntimeError('Failed native graph engine cannot continue')
        if self.bound_inputs and (input_ids is None or inputs_embeds is not None):
            raise ValueError('Bound native Graph inputs require token IDs')
        if (intermediate_tensors is not None or kwargs or positions.ndim not in (1, 2)
                or (positions.ndim == 2 and positions.shape[0] != 3)
                or (input_ids is None and inputs_embeds is None)):
            raise ValueError('Require one serial native text-ID or embedding input')
        if input_ids is not None and (input_ids.shape != (1,) or input_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Require one native int32 or int64 token ID')
        if positions.dtype != torch.int64 or any(value is not None and value.device != positions.device
                                                 for value in (input_ids, inputs_embeds)):
            raise ValueError('Native graph inputs must share one device with int64 positions')
        if inputs_embeds is not None and (inputs_embeds.ndim != 2 or inputs_embeds.shape[0] != 1
                                         or inputs_embeds.dtype != torch.bfloat16):
            raise ValueError('Require one BF16 native embedding row')
        try:
            if self.graph is None:
                self._capture(input_ids, positions, inputs_embeds, context)
            self._guard(input_ids, positions, inputs_embeds, context)
            if not self.bound_inputs:
                for target, source in zip(self.tensors, (input_ids, positions, inputs_embeds)):
                    if target is not None:
                        target.copy_(source)
            self.graph.replay()
            self.replays += 1
            return self.output
        except BaseException:
            self.failed = True
            raise

    def freeze(self):
        if self.graph is None:
            raise RuntimeError('Native graph needs excluded warmup')
        self.frozen = True

    def close(self):
        self.model.forward = self.original
        self.graph = self.output = None

    def report(self):
        return dict(captures=self.captures, replays=self.replays, frozen=self.frozen,
            input_mode='bound' if self.bound_inputs else 'copy',
            extra_input_copies_per_decode=0 if self.bound_inputs else sum(value is not None for value in getattr(self, 'tensors', ())),
            failed=self.failed, phases=['full'] if self.graph is not None else [],
            capture_rollback_bytes=getattr(self, 'capture_rollback_bytes', 0),
            graph_launches_per_decode=1, router_head_executions=0,
            gdn_snapshot_restore_kernels_per_decode=0, local_kv_allocations=0,
            original_native_backbone_and_output=True,
            module_checks='native_cpu' if self._module_check is not None else 'python',
            module_check_build=getattr(self, '_module_check_binding', None),
            compiled_cpu_layout_checks=self._bulk_layout is not None,
            gdn_metadata_checks=('once_per_distinct_object_per_decode'
                if self.deduplicate_gdn_metadata else 'original_per_layer'),
            real_gpu_numeric_and_tps_verified=False)
