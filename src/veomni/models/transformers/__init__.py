"""Customized Transformer model packages with lazy optional dependencies."""

from __future__ import annotations

import importlib


__all__ = [
    "deepseek_v3",
    "flux",
    "janus",
    "llama",
    "movqgan",
    "qwen2",
    "qwen2_5_omni",
    "qwen2_5vl",
    "qwen2_vl",
    "qwen3",
    "qwen3_moe",
    "qwen3_5_three_stream_router",
    "seed_oss",
    "wan",
]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
