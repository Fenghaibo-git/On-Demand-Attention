"""Opt-in Router runtime experiments; never installed into frozen baselines."""


def register_vllm_plugin():
    """Explicit registration only; importing this package does not alter vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen35DualCacheForCausalLM",
        "qwen3_5_router_runtime_experiments.vllm_runtime:Qwen35DualCacheForCausalLM",
    )
    ModelRegistry.register_model(
        "Qwen35DualCache27BForCausalLM",
        "qwen3_5_router_runtime_experiments.vllm_runtime:Qwen35DualCache27BForCausalLM",
    )
