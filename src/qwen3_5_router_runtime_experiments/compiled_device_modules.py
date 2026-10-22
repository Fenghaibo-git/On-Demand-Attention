"""Load the opt-in CPU module checker before excluded graph warmup."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

_loaded=None
_loaded_info=None
VLLM_PARAMETER_SHA256='13d4d581a96ce71018e23c7499c81b8bdd212728022492f8aa36fb814e6c8ef1'


def native_module_class():
    global _loaded,_loaded_info
    if _loaded is not None:
        return _loaded.FrozenModules
    directory=os.environ.get('QWEN_MODULE_NATIVE_DIR')
    if not directory:
        raise ValueError('QWEN_MODULE_NATIVE_DIR must name a separately compiled module-check extension')
    root=Path(directory).resolve()
    info=json.loads((root/'build.json').read_text())
    if info['torch_version']!=torch.__version__ or info['python_version']!=list(sys.version_info[:2]):
        raise ValueError('Module extension Python/PyTorch ABI differs from the current process')
    source=Path(__file__).with_name('native_device_modules.cpp')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=info['source_sha256']:
        raise ValueError('Module extension source changed; rebuild before warmup')
    extension=Path(info['extension_path']).resolve()
    if extension.parent!=root or hashlib.sha256(extension.read_bytes()).hexdigest()!=info['extension_sha256']:
        raise ValueError('Module extension artifact or path differs from its build metadata')
    spec=importlib.util.spec_from_file_location('qwen_device_modules_native',extension)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _loaded=module
    _loaded_info=dict(info,build_path=str(root/'build.json'),
                      build_sha256=hashlib.sha256((root/'build.json').read_bytes()).hexdigest())
    return module.FrozenModules


def native_module_binding():
    native_module_class()
    return dict(_loaded_info)


def vllm_parameter_types():
    # The real Qwen/vLLM model has already imported this module. Do not import
    # an engine or discover a platform just to check ordinary CPU fixtures.
    module=sys.modules.get('vllm.model_executor.parameter')
    if module is None:
        return ()
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=VLLM_PARAMETER_SHA256:
        raise ValueError('Pinned vLLM parameter implementation changed')
    # Non-quantized Qwen linear layers use this exact loader-aware subclass.
    return (module.ModelWeightParameter,)
