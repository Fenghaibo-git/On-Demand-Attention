"""Causal, label-free recall inputs shared by training and cached decoding."""
from __future__ import annotations

import math

import torch

VERSION = "recall-input-ablation-v1"
VARIANTS = ("he", "hes", "hec", "hesc", "he_zero")
CONFIDENCE_NAMES = (
    "top1_probability", "top1_minus_top2_probability",
    "log1p_top1_top2_log_probability_ratio", "normalized_entropy",
    "top5_probability_mass", "top20_probability_mass",
)


def validate_recall_config(config):
    version = getattr(config, "router_recall_input_version", None)
    if version is None:
        return None
    if version != VERSION:
        raise ValueError(f"Unknown recall input version: {version!r}")
    variant = getattr(config, "router_recall_input_variant", None)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown recall input variant: {variant!r}")
    return variant


@torch.no_grad()
def local_confidence_features(logits: torch.Tensor) -> torch.Tensor:
    """Temperature-one statistics, invariant to additive logit constants.

    Only current Local LM-head output enters: no true next-token index, Full
    logits, token identities, or learned vocabulary projection. The log
    probability ratio is the logit gap, avoiding probability underflow.
    """
    if logits.ndim < 2 or logits.shape[-1] < 2:
        raise ValueError("confidence requires a vocabulary of at least two")
    z = logits.detach().float()
    if not torch.isfinite(z).all():
        raise FloatingPointError("nonfinite Local logits")
    logp = z - torch.logsumexp(z, dim=-1, keepdim=True)
    p = logp.exp()
    top = torch.topk(logp, min(20, z.shape[-1]), dim=-1).values
    top_p = top.exp()
    entropy = -(p * logp).sum(-1) / math.log(z.shape[-1])
    return torch.stack((
        top_p[..., 0], top_p[..., 0] - top_p[..., 1],
        torch.log1p((top[..., 0] - top[..., 1]).clamp_min(0)),
        entropy, top_p[..., :5].sum(-1), top_p.sum(-1),
    ), dim=-1)


def pack_recall_features(history, embedding, swa_hidden, confidence, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown recall input variant: {variant!r}")
    if history.shape != embedding.shape or history.shape != swa_hidden.shape:
        raise ValueError("history, embedding and Local hidden must align")
    # Duplicate H keeps the third tower active without supplying new info.
    # A separate zero arm measures the effect of dormant projection weights.
    third = swa_hidden if variant in ("hes", "hesc") else history
    if variant == "he_zero":
        third = torch.zeros_like(history)
    if variant in ("hec", "hesc"):
        if confidence is None or confidence.shape != (*history.shape[:-1], len(CONFIDENCE_NAMES)):
            raise ValueError("this arm requires aligned current Local confidence")
        extra = confidence.detach().float()
    else:
        extra = torch.zeros((*history.shape[:-1], len(CONFIDENCE_NAMES)), device=history.device)
    return torch.cat((history.float(), embedding.float(), third.float(), extra), dim=-1)


def recall_schema(config):
    variant = validate_recall_config(config)
    return {
        "version": VERSION, "variant": variant,
        "confidence_names": list(CONFIDENCE_NAMES), "temperature": 1.0,
        "missing_swa": "zeros" if variant == "he_zero" else "history_duplicate",
        "confidence_injection": "bias_free_fp32_projection_before_residual_blocks",
        "feature_sources": "past_selected_hidden,current_embedding,current_local_hidden,current_local_logits",
        "label_or_full_logits_in_confidence": False,
    }
