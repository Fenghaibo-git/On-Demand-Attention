"""Deterministic hashes for small trainable-state contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch


def tensor_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash names, dtypes, shapes, and raw tensor bytes in canonical order."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("Tensor-state hashes require a string-to-Tensor mapping.")
        value = tensor.detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
        # Flatten first so scalar parameters (for example learned fusion and
        # LayerScale coefficients) can be reinterpreted as bytes as well.
        # This preserves the byte stream, and therefore existing hashes, for
        # every non-scalar contiguous tensor.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()
