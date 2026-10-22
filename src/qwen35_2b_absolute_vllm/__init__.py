"""Opt-in Qwen3.5-2B StreamingLLM and selected-history Router runtime."""


def register():
    from vllm import ModelRegistry
    for name in ("Qwen35Streaming2B", "Qwen35FastLocal2B", "Qwen35FastRouter2B", "Qwen35FastFull2B"):
        ModelRegistry.register_model(name, f"qwen35_2b_absolute_vllm.runtime:{name}")

register()
