"""Prepare serial GDN copy descriptors once; keep the original copy kernel."""
import math
import os

import torch

from .graphed_kv_commit import tensor_signature


class CopyPlan:
    def __init__(self, entries, backend):
        self.entries=tuple(entries)
        self.backend=backend
        self.sources=tuple(source for source,_,_ in self.entries)
        self.indices=tuple(index for _,index,_ in self.entries)
        self.backups=tuple(backup for _,_,backup in self.entries)
        self.shapes=tuple(tuple(source.shape) for source in self.sources)
        self.strides=tuple(tuple(source.stride()) for source in self.sources)
        starts=[0]
        for shape in self.shapes:
            starts.append(starts[-1]+(math.prod(shape[1:])+1023)//1024)
        self.starts=tuple(starts)
        self.launch=None
        if backend=='triton' and self.entries:
            if self.sources[0].device.type!='cuda' and os.environ.get('TRITON_INTERPRET')!='1':
                raise RuntimeError('CPU kernel execution requires TRITON_INTERPRET=1')
            from .optimized_kernels import state_copy
            self.launch=state_copy[(starts[-1],)]

    @torch.no_grad()
    def copy(self, restore):
        if not self.entries:return
        if self.backend=='torch':
            for source,index,backup in self.entries:
                if source.device.type!='cpu':raise RuntimeError('Reference state copies are CPU-only')
                row=int(index.item())
                if not 0<=row<source.shape[0]:raise ValueError('State index out of bounds')
                target,value=(source[row],backup[0]) if restore else (backup[0],source[row])
                target.copy_(value)
        else:
            self.launch(self.sources,self.indices,self.backups,SHAPES=self.shapes,
                STRIDES=self.strides,BLOCK_STARTS=self.starts,RESTORE=restore,BLOCK=1024,debug=True)


class PreparedGDNStates:
    """Preserve one independent backup per layer/state entry, including aliases.

    State-index *values* may change every step. Layout checks ensure that the
    stored views still alias the current metadata; no index value is read on
    the host in the Triton path. Rebinding is allowed only before freeze().
    """
    def __init__(self, owner, *, backend='triton', compiled_layouts=False):
        if backend not in ('torch','triton'):raise ValueError('Unknown state backend')
        self.owner,self.backend=owner,backend
        self.layers=tuple(owner._gdn_layers.items())
        if not self.layers:raise ValueError('Prepared GDN state requires recurrent layers')
        self.names=tuple(module.layer_name if hasattr(module,'layer_name') else module.prefix
                         for _,module in self.layers)
        self.layout_class=None
        if compiled_layouts:
            from .compiled_layout_checks import native_layout_class
            self.layout_class=native_layout_class()
        self.entries=None
        self.frozen=False
        self.builds=self.snapshots=self.restores=self.prefix_restores=0
        self._buffers={}

    def _resolve(self):
        if (tuple(self.owner._gdn_layers)!=tuple(index for index,_ in self.layers)
                or any(self.owner._gdn_layers[index] is not module for index,module in self.layers)):
            raise ValueError('GDN layer registry changed')
        context,_=self.owner._metadata(self.layers[0][1])
        if not isinstance(context.attn_metadata,dict):raise ValueError('Require per-layer GDN metadata')
        tensors=[]
        for (_,module),name in zip(self.layers,self.names):
            meta=context.attn_metadata[name]
            if (meta.num_actual_tokens!=1 or meta.num_prefills!=0 or meta.num_decodes!=1
                    or meta.spec_sequence_masks is not None):
                raise ValueError('Only non-speculative serial GDN decode is admitted')
            index=meta.non_spec_state_indices_tensor
            banks=module.kv_cache[context.virtual_engine]
            if (len(banks)!=2 or index.ndim!=1 or index.numel()!=1
                    or index.dtype not in (torch.int32,torch.int64)):
                raise ValueError('Require conv/recurrent banks and one integer state index')
            for bank in banks:
                if bank.ndim<2 or bank.device!=index.device:
                    raise ValueError('State bank/index layout or device is invalid')
            tensors.extend((*banks,index))
        return context.virtual_engine,tuple(tensors)

    def _matches(self, virtual_engine, tensors):
        if virtual_engine!=self.virtual_engine:return False
        if self.layout_class is not None:return self.layouts.matches(tensors)
        return tuple(map(tensor_signature,tensors))==self.layouts

    def _prepare(self, virtual_engine, tensors):
        if self.frozen:raise ValueError('GDN state binding changed after warmup')
        entries=[]
        active={}
        states=tuple((bank,tensors[offset+2]) for offset in range(0,len(tensors),3)
                     for bank in tensors[offset:offset+2])
        for ordinal,(source,index) in enumerate(states):
            key=(ordinal,tensor_signature(source))
            backup=self._buffers.get(key)
            if backup is None:
                backup=torch.empty((1,*source.shape[1:]),dtype=source.dtype,device=source.device)
            active[key]=backup
            entries.append((source,index,backup))
        self._buffers=active
        self.entries=tuple(entries)
        self.full_plan=CopyPlan(self.entries,self.backend)
        self.prefix_plan=CopyPlan(self.entries[6:],self.backend)
        self.virtual_engine=virtual_engine
        self.layouts=(self.layout_class(tensors) if self.layout_class is not None
                      else tuple(map(tensor_signature,tensors)))
        self.builds+=1

    def snapshot(self):
        engine,tensors=self._resolve()
        if self.entries is None or not self._matches(engine,tensors):
            self._prepare(engine,tensors)
        self.full_plan.copy(False)
        self.snapshots+=1
        return self.entries

    def restore(self, entries):
        if self.entries is None:raise ValueError('No prepared GDN transaction to restore')
        if entries is self.entries:
            self.full_plan.copy(True)
            self.restores+=1
        elif (len(entries)==len(self.prefix_plan.entries)
              and all(got is expected for got,expected in zip(entries,self.prefix_plan.entries))):
            self.prefix_plan.copy(True)
            self.restores+=1
            self.prefix_restores+=1
        else:raise ValueError('Restore must use the current complete or shared-prefix transaction')

    def freeze(self):
        if self.entries is None:raise ValueError('GDN state plans require excluded warmup')
        self.frozen=True

    def report(self):
        return dict(enabled=True,plan_builds=self.builds,frozen=self.frozen,
            snapshot_calls=self.snapshots,restore_calls=self.restores,prefix_restore_calls=self.prefix_restores,
            state_entries=len(self.entries or ()),
            backup_bytes=sum(backup.numel()*backup.element_size() for _,_,backup in self.entries or ()),
            compiled_layouts=self.layout_class is not None,
            host_state_index_reads=0 if self.backend=='triton' else None,
            kernel='unchanged_optimized_kernels.state_copy',gpu_tps_verified=False)
