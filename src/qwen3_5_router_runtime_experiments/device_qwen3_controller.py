"""Qwen3 stage integration for the opt-in native conditional graph.

An injectable CPU backend tests transactions; production uses only the CUDA
backend. No device discovery, allocation or native load occurs on import.
"""
import os
import torch

from .device_capture_state import DecodeCaptureRollback
from .graphed_kv_commit import AllLayerSelectedCommit, tensor_signature


class TorchCaptureBackend:
    device_type='cuda'

    def capture(self, function):
        current=torch.cuda.current_stream()
        stream=torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                function()
        current.wait_stream(stream)
        graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):
            result=function()
        current.wait_stream(stream)
        graph.instantiate()
        graph.replay()  # Initialize the actual captured outputs before table build.
        return graph,result

    def commit(self, state, slots, position, write_full):
        from .device_selected_commit import selected_commit_all_device_position
        selected_commit_all_device_position[(len(state['references']),4)](
            state['table'],slots,position,WRITE_FULL=write_full,BLOCK_SIZE=state['block_size'],
            HEAD_DIM=128,ELEMENTS=1024,CAPACITY=state['ring_pitch'],SINKS=4,RECENT=2047,BLOCK=256)

    def native(self, directory, stages, score, counters, threshold, forced, keepalive):
        from .device_route_binding import NativeRouteGraph
        return NativeRouteGraph(directory,stages,score,counters,threshold,forced,keepalive=keepalive)

    def replay(self, graph):
        graph.replay(torch.cuda.current_stream())


def metadata_layout(meta):
    return tuple((name,tensor_signature(getattr(meta,name))) for name in
                 ('block_table','slot_mapping','seq_lens','query_start_loc'))


def current_metadata(context,maximum):
    all_meta=context.attn_metadata
    if not isinstance(all_meta,dict) or not all_meta:
        raise ValueError('Require per-layer attention metadata')
    meta=next(iter(all_meta.values()))
    if any(other is not meta for other in all_meta.values()):
        raise ValueError('Every layer must share the same attention metadata object')
    if (meta.num_actual_tokens!=1 or meta.block_table.shape[0]!=1 or meta.use_cascade
            or meta.max_seq_len>maximum or meta.slot_mapping.numel()!=1):
        raise ValueError('Require serial, bounded, non-cascade decode')
    return meta


class Qwen3DeviceController:
    POLICIES=(-1,2,16)  # natural, controlled 12.5%, controlled Full

    def __init__(self,owner,maximum,directory,*,_backend=None,_commit_factory=None):
        from .graph_input_mode import bound_graph_inputs_enabled
        self.bound_inputs = bound_graph_inputs_enabled()
        self.owner,self.maximum,self.directory=owner,maximum,directory
        self.backend=_backend or TorchCaptureBackend()
        factory=_commit_factory
        if factory is None:
            if os.environ.get('Q3_COMPILED_LAYOUT_CHECKS','0')=='1':
                from .compiled_layout_checks import CompiledAllLayerSelectedCommit
                factory=CompiledAllLayerSelectedCommit
            else:
                factory=AllLayerSelectedCommit
        self.commit=factory(owner)
        self.ready=False
        self.frozen=False
        self.replays=0
        self.captures=0
        self.native={}
        self.previous=self.counters=None
        self.flushed=(0,0,0)
        self._bulk_layout=None
        self._module_check=None

    def _cache_tensors(self,context):
        tensors=[]
        for index,module in self.owner._attention_layers.items():
            local=self.owner._local[index]
            tensors.extend((module.kv_cache[context.virtual_engine],local.keys,local.values,
                            local._attention_keys,local._attention_values))
        return tuple(tensors)

    def _check_hooks(self):
        # Device graphs cannot invoke arbitrary Python callbacks once per step.
        # The measurement hook has an explicitly ported device equivalent.
        global_hooks=torch.nn.modules.module
        if global_hooks._global_forward_hooks or global_hooks._global_forward_pre_hooks:
            raise ValueError('Global Python hooks are not supported by device routing')
        allowed=getattr(self.owner,'_device_allowed_score_hook',None)
        if self._module_check is not None:
            self._module_check.check(self.owner.admission,allowed)
            return
        for module in self.modules:
            hooks=module._forward_hooks
            known=(module is self.owner.admission and allowed is not None
                   and len(hooks)==1 and hooks.get(allowed[0]) is allowed[1]
                   and allowed[1].__code__ is allowed[2])
            if ((hooks and not known) or module._forward_pre_hooks or module._backward_hooks
                    or module.training):
                raise ValueError('Device routing requires unchanged inference modules without Python hooks')

    def _capture(self,ids,positions,context):
        if self.frozen:
            raise RuntimeError('Device route requires excluded warmup before frozen execution')
        owner=self.owner
        meta=current_metadata(context,self.maximum)
        if ids.device.type!=self.backend.device_type or torch.is_grad_enabled():
            raise ValueError('Device route requires inference on the configured device')
        self.modules=tuple(owner.model.modules())+tuple(owner.admission.modules())
        self.model_owner,self.admission_owner=owner.model,owner.admission
        self._check_hooks()
        if any(p.dtype!=torch.float32 for p in owner.admission.router_head.parameters()):
            raise ValueError('The admitted real head must remain FP32')
        if self.bound_inputs:
            self.ids, self.positions = ids, positions
        else:
            self.ids=torch.empty_like(ids)
            self.positions=torch.empty_like(positions)
            self.ids.copy_(ids);self.positions.copy_(positions)
        self.previous=owner._previous.detach().clone()
        self.counters=torch.zeros(4,dtype=torch.int64,device=ids.device)
        self.device=ids.device
        self.threshold=owner._contract['threshold']
        self.virtual_engine=context.virtual_engine
        self.metadata=metadata_layout(meta)
        self.local_owners=tuple(owner._local.items())
        self.cache_tensors=self._cache_tensors(context)
        if os.environ.get('Q3_COMPILED_LAYOUT_CHECKS','0')=='1':
            from .compiled_layout_checks import native_layout_class
            self._bulk_layout=native_layout_class()(self.cache_tensors)
        else:
            self.cache_layout=tuple(map(tensor_signature,self.cache_tensors))
        # Checking the module registry catches replaced parameters without
        # walking the entire module tree again on each decode.
        self.parameters=tuple((module,name,p,None if p.is_inference() else p._version,tensor_signature(p))
            for module in self.modules for name,p in module._parameters.items() if p is not None)
        compiled_modules=os.environ.get('Q3_COMPILED_MODULE_CHECKS','0')=='1'
        ordinary_parameters=all(type(p) in (torch.Tensor,torch.nn.Parameter) for _,_,p,_,_ in self.parameters)
        self._parameter_layout=(native_layout_class()(tuple(p for _,_,p,_,_ in self.parameters))
                                if self._bulk_layout is not None and ordinary_parameters and not compiled_modules else None)
        if compiled_modules:
            from .compiled_device_modules import native_module_class,native_module_binding,vllm_parameter_types,VLLM_PARAMETER_SHA256
            parameter_types=vllm_parameter_types()
            self._module_check=native_module_class()(self.modules,parameter_types)
            self._module_check_binding=native_module_binding()
            self._module_check_binding.update(parameter_types=[f'{kind.__module__}.{kind.__name__}' for kind in parameter_types],
                vllm_parameter_source_sha256=VLLM_PARAMETER_SHA256 if parameter_types else None)
            self._check_hooks()
        old_max=meta.max_seq_len
        self.stage_graphs=[]
        rollback=DecodeCaptureRollback(owner,context,meta,(self.previous,self.counters))
        self.rollback_bytes=rollback.saved_bytes
        try:
            with rollback:
                meta.max_seq_len=self.maximum

                def prelude():
                    owner._phase='local';owner._staged.clear()
                    embedding=owner.embed_input_ids(self.ids)
                    local=owner.model(None,self.positions,None,embedding.clone())
                    score=owner.admission.router_head(self.previous,embedding,local)
                    return embedding,local,score,dict(owner._staged)

                local_graph,(self.embedding,self.local_hidden,self.score,local_staged)=self.backend.capture(prelude)
                self.stage_graphs.append(local_graph)
                if self.score.shape!=(1,) or self.score.dtype!=torch.float32:
                    raise ValueError('Real Router must produce one FP32 score')
                owner._staged=dict(local_staged)
                self.commit._build('local',context,meta)

                def full():
                    owner._phase='full';owner._staged.clear()
                    hidden=owner.model(None,self.positions,None,self.embedding.clone())
                    return hidden,dict(owner._staged)

                full_graph,(self.full_hidden,full_staged)=self.backend.capture(full)
                self.stage_graphs.append(full_graph)
                owner._staged=dict(full_staged)
                self.commit._build('full',context,meta)

                def selected(phase,hidden):
                    self.backend.commit(self.commit.tables[phase],meta.slot_mapping,
                                        self.positions,phase=='local')
                    self.previous.copy_(hidden)

                local_commit,_=self.backend.capture(lambda:selected('local',self.local_hidden))
                full_commit,_=self.backend.capture(lambda:selected('full',self.full_hidden))
                self.stage_graphs.extend((local_commit,full_commit))
                stages=(local_graph,(full_graph,full_commit),local_commit)
                for policy in self.POLICIES:
                    self.native[policy]=self.backend.native(self.directory,stages,self.score,
                        self.counters,owner._contract['threshold'],policy,self)
                    rollback.restore()
                    # Exercise Full and Local outcomes for the mixed graph;
                    # all work is excluded setup and is rolled back afterwards.
                    for _ in range(3 if policy==2 else 1):
                        self.backend.replay(self.native[policy])
                self.captures=4
        except BaseException:
            self.close()
            raise
        finally:
            meta.max_seq_len=old_max
        self.input_layout=(tensor_signature(self.ids),tensor_signature(self.positions))
        self.buffer_layout=tuple(map(tensor_signature,(self.previous,self.counters,self.score,
                                                       self.embedding,self.local_hidden,self.full_hidden)))
        self.ready=True
        owner._previous=self.previous
        owner._staged.clear()

    def _guard(self,ids,positions,context):
        owner=self.owner
        if torch.is_grad_enabled() or owner._contract['threshold']!=self.threshold:
            raise ValueError('Captured inference mode or routing threshold changed')
        if owner.model is not self.model_owner or owner.admission is not self.admission_owner:
            raise ValueError('Captured model or admission object changed')
        meta=current_metadata(context,self.maximum)
        if context.virtual_engine!=self.virtual_engine or metadata_layout(meta)!=self.metadata:
            raise ValueError('Captured metadata allocation changed')
        if (ids.shape!=self.ids.shape or ids.dtype!=self.ids.dtype or ids.device!=self.device
                or positions.shape!=self.positions.shape or positions.dtype!=self.positions.dtype
                or positions.device!=self.device):
            raise ValueError('Device graph input layout changed')
        if (tuple(owner._local)!=tuple(index for index,_ in self.local_owners)
                or any(owner._local[index] is not local for index,local in self.local_owners)):
            raise ValueError('Persistent Local objects changed')
        if any(local.length!=owner._clock.length for _,local in self.local_owners):
            raise ValueError('Local selected history is stale')
        current=self._cache_tensors(context)
        valid=(self._bulk_layout.matches(current) if self._bulk_layout is not None
               else tuple(map(tensor_signature,current))==self.cache_layout)
        if not valid:
            raise ValueError('Captured Full/Local tensor storage changed')
        if (owner._previous is not self.previous or tuple(map(tensor_signature,(
                self.previous,self.counters,self.score,self.embedding,self.local_hidden,self.full_hidden)))
                !=self.buffer_layout):
            raise ValueError('Captured route state storage changed')
        if (tensor_signature(self.ids),tensor_signature(self.positions))!=self.input_layout:
            raise ValueError('Captured input storage changed')
        if self.bound_inputs and (tensor_signature(ids), tensor_signature(positions)) != self.input_layout:
            raise ValueError('Bound graph input allocation or layout changed')
        self._check_hooks()
        if self._module_check is None:
            current_parameters=[]
            for module,name,p,version,layout in self.parameters:
                if (module._parameters.get(name) is not p or (version is not None and p._version!=version)
                        or (self._parameter_layout is None and tensor_signature(p)!=layout)):
                    raise ValueError('Captured inference parameter changed')
                current_parameters.append(p)
            if self._parameter_layout is not None and not self._parameter_layout.matches(tuple(current_parameters)):
                raise ValueError('Captured inference parameter storage changed')

    def replay(self,ids,positions,context,policy):
        if policy not in self.POLICIES:
            raise ValueError('Policy has no independently warmed native graph')
        if not self.ready:
            self._capture(ids,positions,context)
        self._guard(ids,positions,context)
        if not self.bound_inputs:
            self.ids.copy_(ids);self.positions.copy_(positions)
        self.backend.replay(self.native[policy])
        self.replays+=1
        self.owner._phase='device'
        self.owner._staged.clear()
        return self.previous

    def reset_request(self,previous):
        if self.ready:
            self.previous.copy_(previous)
            self.counters.zero_()
            self.owner._previous=self.previous
        self.flushed=(0,0,0)

    def flush_statistics(self,expected_steps):
        if not self.ready:
            if expected_steps:
                raise ValueError('No device graph was executed')
            return (0,0,0,None)
        values=tuple(int(v) for v in self.counters.detach().cpu().tolist())
        if (values[0]!=expected_steps or not 0<=values[2]<=values[1]<=values[0]
                or values[3] not in (0,1)):
            raise ValueError('Device decision counts disagree with submitted decode steps')
        delta=tuple(values[i]-self.flushed[i] for i in range(3))
        if any(v<0 for v in delta):
            raise ValueError('Device counters regressed')
        self.flushed=values[:3]
        return (*delta,bool(values[3]))

    def freeze(self):
        if not self.ready:
            raise RuntimeError('Device graph needs excluded warmup')
        self.frozen=True

    def close(self):
        for graph in list(self.native.values()):
            graph.close()
        self.native.clear()
        self.ready=False

    def report(self):
        return dict(captures=self.captures,replays=self.replays,frozen=self.frozen,
                    input_mode='bound' if self.bound_inputs else 'copy',
                    extra_input_copies_per_decode=0 if self.bound_inputs else 2,
                    module_checks='native_cpu' if self._module_check is not None else 'python',
                    module_check_build=getattr(self,'_module_check_binding',None),
                    warmed_policies=sorted(self.native),graph_launches_per_decode=1,
                    host_score_reads_per_decode=0,output_projection_outside_route_graph=True,
                    capture_rollback_bytes=getattr(self,'rollback_bytes',0),
                    commit=dict(table_builds=self.commit.builds,kernels_per_decode=1),
                    real_gpu_numeric_and_tps_verified=False)
