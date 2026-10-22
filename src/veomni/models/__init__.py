"""VeOmni model entry points with dependency-isolated lazy imports."""

from __future__ import annotations

import importlib


_AUTO_EXPORTS = {
    "build_foundation_model",
    "build_processor",
    "build_tokenizer",
}
_MODULE_UTIL_EXPORTS = {
    "init_empty_weights",
    "load_model_weights",
    "save_model_assets",
    "save_model_weights",
}
_SUBPACKAGES = {"transformers", "seed_omni", "custom"}

__all__ = sorted(_AUTO_EXPORTS | _MODULE_UTIL_EXPORTS | _SUBPACKAGES)


def __getattr__(name: str):
    if name in _AUTO_EXPORTS:
        return getattr(importlib.import_module(".auto", __name__), name)
    if name in _MODULE_UTIL_EXPORTS:
        return getattr(importlib.import_module(".module_utils", __name__), name)
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
