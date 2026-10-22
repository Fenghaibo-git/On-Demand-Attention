"""Save only slots touched by one-token Qwen3 graph warmup; CPU-testable."""
import torch


class DecodeCaptureRollback:
    def __init__(self, owner, context, meta, buffers=()):
        if meta.slot_mapping.numel()!=1:
            raise ValueError('Capture rollback requires a single decode slot')
        self.owner=owner
        self.phase=owner._phase
        self.staged=dict(owner._staged)
        self.length=owner._clock.length
        self.failed=owner._clock.failed
        # This read occurs only during excluded graph setup, never on replay.
        slot=int(meta.slot_mapping.item())
        if slot < 0:
            raise ValueError('Real decode graph requires an assigned paged slot')
        self.locals=tuple((index,local,local.length) for index,local in owner._local.items())
        views=list(buffers)
        for index,module in owner._attention_layers.items():
            local=owner._local[index]
            if local.length != self.length or self.length < local.capacity:
                raise ValueError('Capture requires full, current Local history')
            paged=module.kv_cache[context.virtual_engine]
            block,offset=divmod(slot,paged.shape[2])
            if block>=paged.shape[1]:
                raise ValueError('Capture slot exceeds paged storage')
            views.append(paged[:,block,offset])
            ring=local.sink_size+self.length%local.recent_size
            views.extend((local.keys[:,ring],local.values[:,ring]))
            if hasattr(local,'_attention_keys'):
                views.extend((local._attention_keys[:,local.capacity],
                              local._attention_values[:,local.capacity]))
        self.copies=tuple((view,view.clone()) for view in views)
        self.restored=False

    def restore(self):
        with torch.no_grad():
            for destination,saved in self.copies:
                destination.copy_(saved)
        self.owner._phase=self.phase
        self.owner._staged=dict(self.staged)
        self.owner._clock.length=self.length
        self.owner._clock.failed=self.failed
        for index,local,length in self.locals:
            if self.owner._local[index] is not local:
                raise RuntimeError('Capture replaced a persistent Local allocation')
            local.length=length
        self.restored=True

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.restore()

    @property
    def saved_bytes(self):
        return sum(saved.numel()*saved.element_size() for _,saved in self.copies)
