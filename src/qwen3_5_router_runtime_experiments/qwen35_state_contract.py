"""Preserve native Qwen3.5 recurrent precision under architecture aliases.

vLLM 0.18.0 applies its Qwen3.5 default updater by architecture *name*.
Custom ModelRegistry aliases do not inherit that updater from their Python
base class. Set these engine options before cache planning, not in forward.
"""


def native_qwen35_state_options():
    # vLLM 0.18 CacheConfig has no explicit bfloat16 convolution enum.
    # With the validated BF16 model, auto resolves convolution state to BF16.
    return dict(mamba_cache_dtype='auto', mamba_ssm_cache_dtype='float32')


def validate_qwen35_state_config(config):
    expected = getattr(config.model_config.hf_text_config, 'mamba_ssm_dtype', None)
    cache = config.cache_config
    if (expected != 'float32' or cache.mamba_ssm_cache_dtype != expected
            or cache.mamba_cache_dtype != 'auto'
            or str(config.model_config.dtype) != 'torch.bfloat16'):
        raise ValueError(
            'Qwen3.5 requires native BF16 convolution state and FP32 recurrent '
            'state. Pass mamba_cache_dtype="auto" with a BF16 model, '
            'mamba_ssm_cache_dtype="float32" to LLM before cache planning; '
            'a custom architecture alias bypasses the vLLM native default updater.')


def validate_qwen35_state_banks(recurrent, context):
    """Check actual allocated banks as well as the requested configuration."""
    for module in recurrent.values():
        banks = module.kv_cache[context.virtual_engine]
        if len(banks) != 2 or tuple(str(bank.dtype) for bank in banks) != (
                'torch.bfloat16', 'torch.float32'):
            raise ValueError('Allocated Qwen3.5 state differs from native BF16/FP32 precision')
