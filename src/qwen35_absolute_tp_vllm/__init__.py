"""Opt-in Qwen3.5-27B StreamingLLM and selected-history Router runtime."""


def register():
    from vllm import ModelRegistry
    for name in ("Qwen35Streaming27B", "Qwen35FastLocal27B", "Qwen35FastRouter27B", "Qwen35FastFull27B"):
        ModelRegistry.register_model(name, f"qwen35_absolute_tp_vllm.runtime:{name}")

register()
