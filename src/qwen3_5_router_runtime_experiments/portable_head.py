"""A0 Router head with no Transformers/Qwen3.5 dependency.

The tensor names and FP32 operations match the frozen HF Router head. A
separate HF-side admission bridge runs the existing strict checkpoint loader;
native vLLM verifies every bound input before consuming its result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        return self.weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps))


class SwiGLUBlock(nn.Module):
    def __init__(self, size, intermediate, eps):
        super().__init__()
        self.norm = RMSNorm(size, eps)
        self.gate_projection = nn.Linear(size, intermediate, bias=False, dtype=torch.float32)
        self.up_projection = nn.Linear(size, intermediate, bias=False, dtype=torch.float32)
        self.down_projection = nn.Linear(intermediate, size, bias=False, dtype=torch.float32)

    def forward(self, x):
        normalized = self.norm(x)
        return x + self.down_projection(
            F.silu(self.gate_projection(normalized)) * self.up_projection(normalized)
        )


class PortableA0Head(nn.Module):
    def __init__(self, hidden_size=2048, router_size=1024, intermediate=2048, blocks=2, eps=1e-6):
        super().__init__()
        self.history_norm = RMSNorm(hidden_size, eps)
        self.embedding_norm = RMSNorm(hidden_size, eps)
        self.swa_norm = RMSNorm(hidden_size, eps)
        self.history_projection = nn.Linear(hidden_size, router_size, dtype=torch.float32)
        self.embedding_projection = nn.Linear(hidden_size, router_size, dtype=torch.float32)
        self.swa_projection = nn.Linear(hidden_size, router_size, dtype=torch.float32)
        self.fusion_projection = nn.Linear(9 * router_size, router_size, dtype=torch.float32)
        self.blocks = nn.ModuleList(
            SwiGLUBlock(router_size, intermediate, eps) for _ in range(blocks)
        )
        self.output_norm = RMSNorm(router_size, eps)
        self.output_projection = nn.Linear(router_size, 1, dtype=torch.float32)

    def forward(self, previous, embedding, local):
        with torch.autocast(device_type=previous.device.type, enabled=False):
            h = F.silu(self.history_projection(self.history_norm(previous.float())))
            e = F.silu(self.embedding_projection(self.embedding_norm(embedding.float())))
            l = F.silu(self.swa_projection(self.swa_norm(local.float())))
            groups = [h, e, l]
            for left, right in ((h, e), (h, l), (e, l)):
                groups.extend((left * right, torch.abs(left - right)))
            fused = F.silu(self.fusion_projection(torch.cat(groups, dim=-1)))
            x = (h + e + l + fused) * 0.5
            for block in self.blocks:
                x = block(x)
            return self.output_projection(self.output_norm(x)).squeeze(-1)


def load_admitted_head(head, admission_path, base_model_path):
    path = Path(admission_path)
    document = json.loads(path.read_text())
    if (
        document.get("format") != "q35-router-head-admission-v1"
        or document.get("status") != "passed"
    ):
        raise ValueError("Require a successful HF-side Router-head admission")
    if Path(base_model_path).resolve() != Path(document["base_model_path"]).resolve():
        raise ValueError("Native model differs from the admitted base model")
    bindings = document.get("bindings", {})
    required = {document.get("checkpoint"), document.get("audit"), document.get("base_config")}
    if None in required or not required.issubset(bindings) or not document.get("loader_sources"):
        raise ValueError("Incomplete admission bindings")
    for source in document["loader_sources"]:
        if source not in bindings:
            raise ValueError("Loader source is not hash-bound")
    for filename, expected in bindings.items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or file_sha256(filename) != expected
        ):
            raise ValueError(f"Admission binding changed: {filename}")
    payload = torch.load(document["checkpoint"], map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    digest = tensor_state_sha256(state)
    if digest != payload.get("tensor_state_sha256") or digest != document["tensor_state_sha256"]:
        raise ValueError("Router tensor-state hash mismatch")
    if not all(key.startswith("router_head.") for key in state):
        raise ValueError("Export includes non-Router tensors")
    stripped = {key[len("router_head.") :]: tensor for key, tensor in state.items()}
    if set(stripped) != set(head.state_dict()):
        raise ValueError("Portable A0 tensor names differ from the admitted head")
    head.load_state_dict(stripped, strict=True)
    head.float().eval()
    return document
