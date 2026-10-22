"""Opt-in CPU position source for the pinned serial vLLM 0.18 text path.

The producer already copies these CPU buffers to the model's GPU position
view. Bind that producer transaction instead of copying positions back. An
installed source never falls back to device reads. Import has no CUDA work.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import threading
import types

import numpy as np
import torch


RUNNER_SHA256 = 'e164efbb988bc23ddff83842d27930a8cc904104d3c74d28c8c9369535b08f01'
BUFFER_SHA256 = '8c970d2862d40b0789a0c2eebefbb76a80018e8f8885e0fe8a91d1d8c5a4a6d0'
MROPE_SHA256 = 'd5fc5ca640bf47fdfc27447bcc6b67b2294c9ac840bbc736fe75cc6098e2a51c'


def _tensor_signature(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError('Position buffer must be a torch Tensor')
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.dtype, tensor.device)


def _method_identity(method):
    return getattr(method, '__self__', None), getattr(method, '__func__', method)


def _source_hash(value):
    path = inspect.getsourcefile(value)
    if path is None:
        raise ValueError('Cannot identify the pinned position producer source')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_function_source(function, name, expected_hash):
    if (not inspect.isfunction(function) or function.__code__.co_name != name
            or _source_hash(function) != expected_hash):
        raise ValueError('Pinned position producer callable source changed')


def _attest_runner(runner):
    kind = type(runner)
    if (kind.__module__, kind.__name__) != ('vllm.v1.worker.gpu_model_runner', 'GPUModelRunner'):
        raise ValueError('CPU positions require the exact pinned GPUModelRunner')
    if importlib.metadata.version('vllm') != '0.18.0' or _source_hash(kind) != RUNNER_SHA256:
        raise ValueError('Pinned vLLM position producer source changed')
    buffer_type = type(runner.positions)
    if ((buffer_type.__module__, buffer_type.__name__) != ('vllm.v1.utils', 'CpuGpuBuffer')
            or _source_hash(buffer_type) != BUFFER_SHA256):
        raise ValueError('Pinned CPU/GPU buffer source changed')
    for name in ('_prepare_inputs', '_preprocess', '_model_forward', '_calc_mrope_positions'):
        _check_function_source(getattr(kind, name), name, RUNNER_SHA256)
        if _method_identity(getattr(runner, name)) != (runner, getattr(kind, name)):
            raise ValueError('Position producer methods must be unmodified before install')
    _check_function_source(buffer_type.copy_to_gpu, 'copy_to_gpu', BUFFER_SHA256)
    for buffer in (runner.positions, getattr(runner, 'mrope_positions', None)):
        if buffer is not None and (type(buffer) is not buffer_type or
                _method_identity(buffer.copy_to_gpu) != (buffer, buffer_type.copy_to_gpu)):
            raise ValueError('Position buffer copy method was overridden')
    if runner.uses_mrope:
        mrope_type = kind._calc_mrope_positions.__globals__['MRotaryEmbedding']
        if _source_hash(mrope_type) != MROPE_SHA256:
            raise ValueError('Pinned MRoPE CPU producer source changed')
        _check_function_source(mrope_type.get_next_input_positions_tensor,
                               'get_next_input_positions_tensor', MROPE_SHA256)
    return dict(vllm_version='0.18.0', gpu_model_runner_sha256=RUNNER_SHA256,
                cpu_gpu_buffer_sha256=BUFFER_SHA256,
                mrope_sha256=MROPE_SHA256 if runner.uses_mrope else None)


class _SerialPositionSource:
    def __init__(self, runner, source_proof):
        self.runner, self.model = runner, runner.model
        self.clock = getattr(self.model, '_clock', None)
        if self.clock is None or not callable(getattr(self.clock, 'classify', None)):
            raise ValueError('Install only on a custom runtime with SerialRequestClock')
        self.source_proof = dict(source_proof)
        self.mrope = bool(runner.uses_mrope)
        self.buffer_names = ('positions', 'mrope_positions') if self.mrope else ('positions',)
        self.buffers = {}
        self._check_mode()
        for name in self.buffer_names:
            buffer = getattr(runner, name)
            cpu, gpu, array = buffer.cpu, buffer.gpu, buffer.np
            expected_shape = ((3, runner.max_num_tokens + 1) if name == 'mrope_positions'
                              else (runner.max_num_tokens,))
            if (cpu.device.type != 'cpu' or gpu.device.type != 'cuda'
                    or cpu.dtype != torch.int64 or gpu.dtype != torch.int64
                    or tuple(cpu.shape) != expected_shape or tuple(gpu.shape) != expected_shape
                    or not cpu.is_contiguous() or not gpu.is_contiguous()):
                raise ValueError('Unexpected pinned position buffer dtype/device/layout')
            if (not isinstance(array, np.ndarray) or array.dtype != np.int64
                    or array.shape != expected_shape or not array.flags.c_contiguous
                    or array.__array_interface__['data'][0] != cpu.data_ptr()
                    or array.strides != tuple(x * cpu.element_size() for x in cpu.stride())):
                raise ValueError('Position numpy array must alias its CPU Tensor exactly')
            self.buffers[name] = (buffer, cpu, gpu, array, _tensor_signature(cpu),
                                  _tensor_signature(gpu), _method_identity(buffer.copy_to_gpu))
        self.methods = {name: _method_identity(getattr(runner, name))
                        for name in ('_prepare_inputs', '_preprocess', '_model_forward', '_calc_mrope_positions')}
        self.original_prepare = runner._prepare_inputs
        self.original_forward = runner._model_forward
        self.pending = self.active = None
        self.last_request = None
        self.failed = False
        self.counts = dict(producer_batches=0, forward_calls=0, completed_forwards=0,
                           reads=0, prefill_reads=0, decode_reads=0, position_tokens=0,
                           rejected=0, gpu_position_copies=0, fallback_reads=0)

        def prepare(module, scheduler_output, num_scheduled_tokens):
            return self._prepare(scheduler_output, num_scheduled_tokens)

        def forward(module, input_ids=None, positions=None, intermediate_tensors=None,
                    inputs_embeds=None, **kwargs):
            return self._forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

        self.prepare_hook = types.MethodType(prepare, runner)
        self.forward_hook = types.MethodType(forward, runner)
        runner._prepare_inputs = self.prepare_hook
        runner._model_forward = self.forward_hook
        self.model._cpu_position_source = self

    def _check_mode(self):
        r = self.runner
        tp = r.parallel_config.tensor_parallel_size
        if tp not in (1,2,4,8):
            raise ValueError('Unsupported tensor parallel size')
        required = dict(tensor_parallel_size=tp, pipeline_parallel_size=1, max_num_seqs=1,
                        enforce_eager=True, enable_chunked_prefill=False,
                        enable_prefix_caching=False, speculative=False)
        contract = getattr(self.model, '_contract', {})
        if any(contract.get(key) != value for key, value in required.items()):
            raise ValueError('Require the existing eager serial unchunked runtime contract')
        if (r.uses_xdrope_dim != 0 or bool(r.uses_mrope) != self.mrope
                or r.supports_mm_inputs or r.is_multimodal_pruning_enabled
                or r.enable_prompt_embeds or r.model_config.is_encoder_decoder
                or r.use_async_scheduling or r.speculative_config is not None
                or r.parallel_config.use_ubatching or r.parallel_config.data_parallel_size != 1
                or r.dcp_world_size != 1 or r.cache_config.kv_sharing_fast_prefill):
            raise ValueError('Alternate position overwrite, async or padded execution is not admitted')

    def _check_bindings(self):
        if self.failed:
            raise ValueError('CPU position source is invalid after a rejected transaction')
        if self.runner.model is not self.model or self.model._clock is not self.clock:
            raise ValueError('Position model/request clock identity changed')
        self._check_mode()
        for name, (buffer, cpu, gpu, array, cpu_sig, gpu_sig, copy_method) in self.buffers.items():
            if (getattr(self.runner, name) is not buffer or buffer.cpu is not cpu
                    or buffer.gpu is not gpu or buffer.np is not array
                    or _tensor_signature(cpu) != cpu_sig or _tensor_signature(gpu) != gpu_sig
                    or array.__array_interface__['data'][0] != cpu.data_ptr()
                    or array.shape != tuple(cpu.shape)
                    or array.strides != tuple(x * cpu.element_size() for x in cpu.stride())
                    or array.dtype != np.int64
                    or _method_identity(buffer.copy_to_gpu) != copy_method):
                raise ValueError('Position CPU/GPU buffer identity, storage or alias changed')
        if (self.runner._prepare_inputs is not self.prepare_hook
                or self.runner._model_forward is not self.forward_hook):
            raise ValueError('Position transaction hook was replaced')
        for name in ('_preprocess', '_calc_mrope_positions'):
            if _method_identity(getattr(self.runner, name)) != self.methods[name]:
                raise ValueError('Position producer dependency was replaced')

    def _scheduler_identity(self, scheduler, counts):
        r = self.runner
        batch = r.input_batch
        ids = tuple(batch.req_ids)
        if batch.num_reqs != 1 or len(ids) != 1:
            raise ValueError('Require exactly one active serial request')
        req_id = ids[0]
        scheduled = scheduler.num_scheduled_tokens
        n = scheduler.total_num_scheduled_tokens
        if (type(n) is not int or n < 1 or n > r.max_num_tokens
                or set(scheduled) != {req_id} or scheduled[req_id] != n
                or not isinstance(counts, np.ndarray) or counts.shape != (1,)
                or counts.dtype != np.int32 or int(counts[0]) != n
                or scheduler.scheduled_spec_decode_tokens or scheduler.scheduled_encoder_inputs
                or scheduler.preempted_req_ids):
            raise ValueError('Invalid serial producer token schedule')
        start = int(batch.num_computed_tokens_cpu[0])
        request = r.requests[req_id]
        if request.mm_features or request.prompt_embeds is not None:
            raise ValueError('CPU position source supports text token IDs only')
        if self.mrope and request.mrope_position_delta != 0:
            raise ValueError('Text-only MRoPE requires zero position delta')
        fresh = tuple(item.req_id for item in scheduler.scheduled_new_reqs)
        if start == 0:
            if (fresh != (req_id,) or request is self.last_request
                    or n != len(request.prompt_token_ids)):
                raise ValueError('Prefill requires a fresh full serial request')
            phase = 'prefill'
        else:
            if (fresh or n != 1 or request is not self.last_request
                    or self.clock.failed or start != self.clock.length):
                raise ValueError('Producer does not match the selected-history decode clock')
            phase = 'decode'
        return dict(request_id=req_id, request=request, batch=batch, scheduler=scheduler,
                    start=start, count=n, phase=phase, thread=threading.get_ident(), read=False)

    def _host_values(self, ticket):
        start, n = ticket['start'], ticket['count']
        if n == 1:
            value = int(self.runner.positions.np[0])
            if value != start:
                raise ValueError('CPU position contents disagree with producer/request clock')
            if self.mrope:
                rows = self.runner.mrope_positions.np
                if any(int(rows[index, 0]) != value for index in range(3)):
                    raise ValueError('Text MRoPE rows differ or disagree with producer/request clock')
            return (value,)
        expected = np.arange(start, start + n, dtype=np.int64)
        ordinary = self.runner.positions.np[:n]
        if not np.array_equal(ordinary, expected):
            raise ValueError('CPU position contents disagree with producer/request clock')
        if self.mrope and not np.all(self.runner.mrope_positions.np[:, :n] == expected[None]):
            raise ValueError('Text MRoPE rows differ or disagree with producer/request clock')
        return tuple(int(value) for value in ordinary)

    def _check_ticket(self, ticket, positions):
        self._check_bindings()
        r = self.runner
        n = ticket['count']
        if (ticket['thread'] != threading.get_ident() or r.input_batch is not ticket['batch']
                or tuple(r.input_batch.req_ids) != (ticket['request_id'],)
                or r.requests.get(ticket['request_id']) is not ticket['request']
                or int(r.input_batch.num_computed_tokens_cpu[0]) != ticket['start']
                or ticket['scheduler'].num_scheduled_tokens != {ticket['request_id']: n}
                or ticket['scheduler'].total_num_scheduled_tokens != n):
            raise ValueError('Position producer/request lifetime changed')
        name = 'mrope_positions' if self.mrope else 'positions'
        base = self.buffers[name][5]
        expected = (base[0], (3, n) if self.mrope else (n,), *base[2:])
        if _tensor_signature(positions) != expected:
            raise ValueError('Model positions must be the exact unpadded producer GPU slice')
        if self._host_values(ticket) != ticket['values']:
            raise ValueError('CPU position contents changed after producer copy')

    def _reject(self):
        self.failed = True
        self.counts['rejected'] += 1

    def _prepare(self, scheduler, counts):
        try:
            self._check_bindings()
            if self.pending is not None or self.active is not None:
                raise ValueError('Previous position transaction was not consumed')
            ticket = self._scheduler_identity(scheduler, counts)
            self.pending = ticket
            result = self.original_prepare(scheduler, counts)
            ticket['values'] = self._host_values(ticket)
            self.counts['producer_batches'] += 1
            return result
        except BaseException:
            self._reject()
            raise

    def _forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        try:
            ticket = self.pending
            if ticket is None or 'values' not in ticket or self.active is not None:
                raise ValueError('No completed producer transaction for model positions')
            if intermediate_tensors is not None or inputs_embeds is not None or kwargs:
                raise ValueError('Only the pinned unpadded text-ID forward is supported')
            ticket['positions'] = positions
            self.active = ticket
            self.counts['forward_calls'] += 1
            result = self.original_forward(input_ids=input_ids, positions=positions,
                                           intermediate_tensors=None, inputs_embeds=None)
            if (not ticket['read'] or self.clock.failed
                    or self.clock.length != ticket['start'] + ticket['count']):
                raise ValueError('Runtime did not read and commit exactly one position transaction')
            self.last_request = ticket['request']
            self.counts['completed_forwards'] += 1
            return result
        except BaseException:
            self._reject()
            raise
        finally:
            self.active = self.pending = None

    def read(self, model, positions):
        ticket = self.active
        try:
            if (model is not self.model or ticket is None or ticket['read']
                    or positions is not ticket['positions']):
                raise ValueError('CPU positions require one read inside their producer forward')
            self._check_ticket(ticket, positions)
            values = list(ticket['values'])
            if self.clock.classify(values) != ticket['phase']:
                raise ValueError('Serial clock disagrees with the position producer phase')
            ticket['read'] = True
            self.counts['reads'] += 1
            self.counts[ticket['phase'] + '_reads'] += 1
            self.counts['position_tokens'] += len(values)
            return values
        except BaseException:
            self._reject()
            raise

    def report(self):
        return dict(format='serial-cpu-position-source-v1', installed=True,
                    layout='mrope_text' if self.mrope else 'ordinary',
                    **self.counts, failed=self.failed, pending=self.pending is not None,
                    source_proof=self.source_proof, gpu_numerical_or_tps_qualified=False)


def install_cpu_position_source(worker):
    """Explicit worker RPC. Native models have no redundant position read."""
    runner = worker.model_runner
    if getattr(runner.model, '_cpu_position_source', None) is not None:
        raise ValueError('A CPU position source is already installed')
    return _SerialPositionSource(runner, _attest_runner(runner)).report()


def position_source_report(worker):
    source = getattr(worker.model_runner.model, '_cpu_position_source', None)
    return source.report() if source is not None else dict(installed=False)


def read_positions(model, positions):
    """Retain the old path unless explicitly installed; no installed fallback."""
    source = getattr(model, '_cpu_position_source', None)
    if source is not None:
        return source.read(model, positions)
    if positions.ndim == 2:
        if not bool((positions == positions[0:1]).all()):
            raise ValueError('Text-only positions required; multimodal input is not admitted')
        positions = positions[0]
    elif positions.ndim != 1:
        raise ValueError('Invalid position layout')
    return positions.detach().cpu().tolist()


install = install_cpu_position_source
