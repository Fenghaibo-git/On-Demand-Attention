# Modified for On Demand Attention; see NOTICE.
"""Explicit native binding. Import performs no CUDA discovery or library load."""
import ctypes
import hashlib
import json
import math
from pathlib import Path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeRouteGraph:
    def __init__(self, directory, stages, score, counters, threshold, forced, *, keepalive):
        # Called only by the explicitly registered CUDA runtime after capture.
        import torch
        if (score.dtype != torch.float32 or score.numel() != 1 or not score.is_contiguous()
                or counters.dtype != torch.int64 or counters.shape != (4,)
                or not counters.is_contiguous() or score.device != counters.device
                or score.device.type != 'cuda'):
            raise ValueError('Native route requires one CUDA FP32 score and four int64 counters')
        if (not math.isfinite(threshold) or type(forced) is not int or not -1 <= forced <= 16
                or len(stages) != 3 or len(stages[1]) != 2):
            raise ValueError('Invalid route policy or stage composition')
        root = Path(directory).resolve()
        metadata = json.loads((root/'BUILD.json').read_text())
        for name, expected in metadata['sources'].items():
            if _sha(Path(__file__).with_name(Path(name).name)) != expected:
                raise ValueError('Native route source differs from offline build')
        path = root/'libqwen_device_route.so'
        if _sha(path) != metadata['artifacts'][path.name]:
            raise ValueError('Native route binary differs from offline build')
        self.library = lib = ctypes.CDLL(str(path))
        u64 = ctypes.c_uint64
        pointer = ctypes.POINTER(u64)
        lib.qwen_route_graph_last_error.restype = ctypes.c_char_p
        lib.qwen_route_graph_create.argtypes = [pointer,u64,u64,ctypes.c_double,ctypes.c_int,pointer]
        lib.qwen_route_graph_replay.argtypes = [u64,u64]
        lib.qwen_route_graph_destroy.argtypes = [u64]
        lib.qwen_route_stage_create.argtypes = [pointer,u64,pointer]
        lib.qwen_route_stage_destroy.argtypes = [u64]
        self.handle = 0
        self.composites = []
        self.keepalive = (stages,score,counters,keepalive)
        self.device = score.device
        try:
            with torch.cuda.device(self.device):
                full = self._compose(stages[1])
                tail = self._compose(())
                handles = (u64*4)(stages[0].raw_cuda_graph(), full,
                                   stages[2].raw_cuda_graph(), tail)
                result = u64()
                self._check(lib.qwen_route_graph_create(handles,score.data_ptr(),counters.data_ptr(),
                                                        threshold,forced,ctypes.byref(result)))
                self.handle = result.value
        except BaseException:
            self.close()
            raise

    def _check(self, result):
        if result:
            message = self.library.qwen_route_graph_last_error()
            raise RuntimeError(message.decode() if message else 'Native route graph failed')

    def _compose(self, graphs):
        u64=ctypes.c_uint64
        handles=(u64*len(graphs))(*(g.raw_cuda_graph() for g in graphs))
        result=u64()
        self._check(self.library.qwen_route_stage_create(handles,len(graphs),ctypes.byref(result)))
        self.composites.append(result.value)
        return result.value

    def replay(self, stream):
        if not self.handle:
            raise RuntimeError('Device route graph is closed')
        self._check(self.library.qwen_route_graph_replay(self.handle,stream.cuda_stream))

    def close(self):
        if not getattr(self,'library',None):
            return
        import torch
        with torch.cuda.device(self.device):
            if self.handle:
                self._check(self.library.qwen_route_graph_destroy(self.handle))
                self.handle=0
            while self.composites:
                self._check(self.library.qwen_route_stage_destroy(self.composites[-1]))
                self.composites.pop()
        self.keepalive=None
