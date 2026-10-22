"""Qwen3 uses the identical qualified Qwen3.5 StreamingLLM Triton kernel."""

def register():
    from vllm import ModelRegistry
    for name in ('Qwen3FastFull', 'Qwen3FastLocal', 'Qwen3Streaming', 'Qwen3FastRouter'):
        ModelRegistry.register_model(name, f'qwen3_absolute_vllm.runtime:{name}')
