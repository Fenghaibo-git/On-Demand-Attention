"""One-step Full/Local Router supervision for Qwen3.

This module deliberately keeps the backbone, causal Router features, and
Full/Local counterfactual implementation identical to ``qwen3_router``.  It
changes only the supervision contract:

* ``signed_nll`` predicts ``NLL_local - NLL_full`` for the aligned next token.
* ``forward_kl`` predicts ``KL(P_full || P_local)`` as a matched control.

Both targets are training-only oracles.  The Router head consumes only causal
features that are available before the current Full/Local action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack

from ..qwen3_router.modeling_qwen3_router import (
    KwargsForCausalLM,
    Qwen3RouterForCausalLM,
    _check_router_tensor_is_finite,
)


_ONE_STEP_CONFIG_DEFAULTS = {
    "router_one_step_target": "signed_nll",
    "router_one_step_transform": "auto",
    "router_one_step_nll_scale": 1.0,
    "router_one_step_kl_scale": 0.1,
    "router_one_step_huber_delta": 1.0,
    "router_one_step_harm_threshold": 0.0,
    "router_one_step_positive_weight": 1.0,
    "router_one_step_sign_aux_weight": 0.0,
    "router_one_step_sign_aux_temperature": 0.1,
    "router_one_step_sign_aux_zero_policy": "ignore_exact_zero",
    "router_one_step_sign_near_zero_threshold": 1.0e-4,
}


def _set_one_step_config_defaults(config: Qwen3Config) -> None:
    for name, value in _ONE_STEP_CONFIG_DEFAULTS.items():
        if not hasattr(config, name):
            setattr(config, name, value)

    if config.router_one_step_target not in {"signed_nll", "forward_kl"}:
        raise ValueError("`router_one_step_target` must be 'signed_nll' or 'forward_kl'.")
    if config.router_one_step_transform == "auto":
        config.router_one_step_transform = (
            "signed_log1p" if config.router_one_step_target == "signed_nll" else "log1p"
        )
    valid_transforms = {
        "signed_nll": {"identity", "signed_log1p"},
        "forward_kl": {"identity", "log1p"},
    }
    if config.router_one_step_transform not in valid_transforms[config.router_one_step_target]:
        raise ValueError(
            f"Transform {config.router_one_step_transform!r} is invalid for "
            f"target {config.router_one_step_target!r}."
        )
    if float(config.router_one_step_nll_scale) <= 0:
        raise ValueError("`router_one_step_nll_scale` must be positive.")
    if float(config.router_one_step_kl_scale) <= 0:
        raise ValueError("`router_one_step_kl_scale` must be positive.")
    if float(config.router_one_step_huber_delta) <= 0:
        raise ValueError("`router_one_step_huber_delta` must be positive.")
    if float(config.router_one_step_harm_threshold) < 0:
        raise ValueError("`router_one_step_harm_threshold` must be non-negative.")
    if float(config.router_one_step_positive_weight) <= 0:
        raise ValueError("`router_one_step_positive_weight` must be positive.")
    if float(config.router_one_step_sign_aux_weight) < 0:
        raise ValueError("`router_one_step_sign_aux_weight` must be non-negative.")
    if float(config.router_one_step_sign_aux_temperature) <= 0:
        raise ValueError("`router_one_step_sign_aux_temperature` must be positive.")
    if config.router_one_step_sign_aux_zero_policy != "ignore_exact_zero":
        raise ValueError(
            "`router_one_step_sign_aux_zero_policy` must be 'ignore_exact_zero'."
        )
    if float(config.router_one_step_sign_near_zero_threshold) < 0:
        raise ValueError(
            "`router_one_step_sign_near_zero_threshold` must be non-negative."
        )


@dataclass
class Qwen3OneStepRouterCausalLMOutput(ModelOutput):
    """Training output for one-step signed-NLL or KL regression."""

    loss: Optional[torch.FloatTensor] = None
    router_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[torch.FloatTensor] = None
    router_targets: Optional[torch.FloatTensor] = None
    router_raw_target: Optional[torch.FloatTensor] = None
    router_predicted_raw_target: Optional[torch.FloatTensor] = None
    router_divergence: Optional[torch.FloatTensor] = None
    full_token_nll: Optional[torch.FloatTensor] = None
    local_token_nll: Optional[torch.FloatTensor] = None
    training_metrics: Optional[dict] = None
    training_probe_tensors: Optional[dict] = None
    full_hidden_states: Optional[torch.FloatTensor] = None
    local_hidden_states: Optional[torch.FloatTensor] = None


class Qwen3OneStepRouterForCausalLM(Qwen3RouterForCausalLM):
    """Train the same causal Router head against a one-step continuous oracle."""

    def __init__(self, config: Qwen3Config):
        _set_one_step_config_defaults(config)
        super().__init__(config)

    @staticmethod
    def _aligned_next_token_labels(
        labels: torch.LongTensor,
        valid_token_mask: torch.BoolTensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        if labels.shape != valid_token_mask.shape or labels.shape != position_ids.shape:
            raise ValueError("Aligned token NLL inputs must have identical [batch, sequence] shapes.")
        aligned_labels = torch.full_like(labels, -100)
        aligned_mask = torch.zeros_like(valid_token_mask)
        if labels.shape[1] > 1:
            aligned_labels[:, :-1] = labels[:, 1:]
            aligned_mask[:, :-1] = (
                valid_token_mask[:, :-1]
                & valid_token_mask[:, 1:]
                & labels[:, 1:].ne(-100)
                & position_ids[:, 1:].eq(position_ids[:, :-1] + 1)
            )
        return aligned_labels, aligned_mask

    @torch.no_grad()
    def _compute_token_nll_only(
        self,
        full_hidden_states: torch.Tensor,
        local_hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        valid_token_mask: torch.BoolTensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute aligned Full/Local NLL without materializing KL probabilities."""

        aligned_labels, aligned_mask = self._aligned_next_token_labels(
            labels, valid_token_mask, position_ids
        )
        chunk_size = int(self.config.router_kl_chunk_size)
        output_weight = self.lm_head.weight.detach()
        full_chunks = []
        local_chunks = []
        confidence_chunks = []
        self._recall_confidence = None
        collect_confidence = (
            getattr(self.config, "router_recall_input_version", None) is not None
            and getattr(self.config, "router_recall_input_variant", None) in {"hec", "hesc"}
        )
        for start in range(0, full_hidden_states.shape[1], chunk_size):
            end = min(start + chunk_size, full_hidden_states.shape[1])
            full_logits = nn.functional.linear(full_hidden_states[:, start:end], output_weight).float()
            local_logits = nn.functional.linear(local_hidden_states[:, start:end], output_weight).float()
            if collect_confidence:
                from ..qwen3_router.recall_features import local_confidence_features
                confidence_chunks.append(local_confidence_features(local_logits))
            target = aligned_labels[:, start:end]
            target_mask = aligned_mask[:, start:end]
            safe_target = target.clamp_min(0).unsqueeze(-1)
            full_nll = torch.logsumexp(full_logits, dim=-1) - full_logits.gather(
                -1, safe_target
            ).squeeze(-1)
            local_nll = torch.logsumexp(local_logits, dim=-1) - local_logits.gather(
                -1, safe_target
            ).squeeze(-1)
            nan = torch.full_like(full_nll, float("nan"))
            full_chunks.append(torch.where(target_mask, full_nll, nan))
            local_chunks.append(torch.where(target_mask, local_nll, nan))
        if confidence_chunks:
            self._recall_confidence = torch.cat(confidence_chunks, dim=1)
        return torch.cat(full_chunks, dim=1), torch.cat(local_chunks, dim=1)

    def _target_scale(self) -> float:
        if self.config.router_one_step_target == "signed_nll":
            return float(self.config.router_one_step_nll_scale)
        return float(self.config.router_one_step_kl_scale)

    def _transform_target(self, raw_target: torch.Tensor) -> torch.Tensor:
        transform = self.config.router_one_step_transform
        scale = self._target_scale()
        raw_target = raw_target.float()
        if transform == "identity":
            return raw_target / scale
        if transform == "signed_log1p":
            return raw_target.sign() * torch.log1p(raw_target.abs() / scale)
        if transform == "log1p":
            return torch.log1p(raw_target.clamp_min(0.0) / scale)
        raise AssertionError(f"Unhandled one-step transform {transform!r}.")

    def _inverse_target(self, transformed_target: torch.Tensor) -> torch.Tensor:
        transform = self.config.router_one_step_transform
        scale = self._target_scale()
        transformed_target = transformed_target.float()
        if transform == "identity":
            return transformed_target * scale
        if transform == "signed_log1p":
            return transformed_target.sign() * torch.expm1(transformed_target.abs()) * scale
        if transform == "log1p":
            return torch.expm1(transformed_target).clamp_min(0.0) * scale
        raise AssertionError(f"Unhandled one-step transform {transform!r}.")

    def _compute_one_step_loss(
        self,
        router_logits: torch.Tensor,
        raw_target: torch.Tensor,
        transformed_target: torch.Tensor,
        loss_mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, dict]:
        classification_groups = (
            (
                ("sign", 0.0),
                ("harm", float(self.config.router_one_step_harm_threshold)),
            )
            if self.config.router_one_step_target == "signed_nll"
            else (("kl", float(self.config.router_kl_threshold)),)
        )
        if not bool(loss_mask.any()) and getattr(self.config, "router_optimizer_step_reduction", None) != "global_eligible_token_mean_v1":
            zero = router_logits.float().sum() * 0.0
            metrics = {
                name: zero.detach()
                for name in (
                    "raw_target_mean",
                    "transformed_mae",
                    "transformed_rmse",
                    "huber_loss",
                    "sign_aux_loss",
                    "weighted_sign_aux_loss",
                    "sign_aux_token_rate",
                    "exact_zero_target_rate",
                    "near_zero_target_rate",
                )
            }
            for prefix, _ in classification_groups:
                metrics.update(
                    {
                        f"{prefix}_target_rate": zero.detach(),
                        f"{prefix}_predicted_positive_rate": zero.detach(),
                        f"{prefix}_accuracy": zero.detach(),
                        f"{prefix}_precision": zero.detach(),
                        f"{prefix}_recall": zero.detach(),
                        f"{prefix}_f1": zero.detach(),
                    }
                )
            legacy_prefix = "harm" if self.config.router_one_step_target == "signed_nll" else "kl"
            metrics.update(
                {
                    "harm_rate": metrics[f"{legacy_prefix}_target_rate"],
                    "predicted_harm_rate": metrics[f"{legacy_prefix}_predicted_positive_rate"],
                    "harm_accuracy": metrics[f"{legacy_prefix}_accuracy"],
                    "harm_precision": metrics[f"{legacy_prefix}_precision"],
                    "harm_recall": metrics[f"{legacy_prefix}_recall"],
                    "harm_f1": metrics[f"{legacy_prefix}_f1"],
                }
            )
            return zero, metrics

        safe_target = torch.where(loss_mask, transformed_target, torch.zeros_like(transformed_target))
        per_token_loss = nn.functional.huber_loss(
            router_logits.float(),
            safe_target.float(),
            reduction="none",
            delta=float(self.config.router_one_step_huber_delta),
        )
        weights = loss_mask.float()
        if self.config.router_one_step_target == "signed_nll":
            positive = raw_target.gt(float(self.config.router_one_step_harm_threshold)) & loss_mask
            weights = weights * torch.where(
                positive,
                weights.new_tensor(float(self.config.router_one_step_positive_weight)),
                weights.new_tensor(1.0),
            )

        global_weight, world_size = self._distributed_denominator(weights.sum())
        if not bool(global_weight > 0):
            huber_loss = router_logits.float().sum() * 0.0
        else:
            huber_loss = (per_token_loss * weights).sum() * (
                float(world_size) / global_weight
            )

        sign_aux_weight = float(self.config.router_one_step_sign_aux_weight)
        sign_mask = (
            loss_mask & raw_target.ne(0.0)
            if self.config.router_one_step_target == "signed_nll"
            else torch.zeros_like(loss_mask)
        )
        sign_temperature = float(self.config.router_one_step_sign_aux_temperature)
        safe_raw_sign = torch.where(
            sign_mask, raw_target.sign(), torch.zeros_like(raw_target)
        )
        per_token_sign_loss = nn.functional.softplus(
            -safe_raw_sign.float() * router_logits.float() / sign_temperature
        )
        global_sign_count, sign_world_size = self._distributed_denominator(
            sign_mask.float().sum()
        )
        if sign_aux_weight > 0.0 and bool(global_sign_count > 0):
            sign_aux_loss = (
                per_token_sign_loss.mul(sign_mask.float()).sum()
                * (float(sign_world_size) / global_sign_count)
            )
        else:
            sign_aux_loss = router_logits.float().sum() * 0.0
        loss = huber_loss + sign_aux_weight * sign_aux_loss

        predicted_raw = self._inverse_target(router_logits.detach())
        error = (router_logits.detach().float() - safe_target.float()) * loss_mask.float()
        statistics = [
            loss_mask.float().sum(),
            (raw_target.nan_to_num() * loss_mask.float()).sum(),
            error.abs().sum(),
            error.square().sum(),
            (per_token_loss.detach() * weights).sum(),
            weights.sum(),
            (per_token_sign_loss.detach() * sign_mask.float()).sum(),
            sign_mask.float().sum(),
            (loss_mask & raw_target.eq(0.0)).float().sum(),
            (
                loss_mask
                & raw_target.abs().le(
                    float(self.config.router_one_step_sign_near_zero_threshold)
                )
            ).float().sum(),
        ]
        for _, threshold in classification_groups:
            observed_positive = raw_target.gt(threshold) & loss_mask
            predicted_positive = predicted_raw.gt(threshold) & loss_mask
            statistics.extend(
                (
                    (observed_positive & predicted_positive).float().sum(),
                    ((~observed_positive) & predicted_positive & loss_mask).float().sum(),
                    (observed_positive & (~predicted_positive)).float().sum(),
                    ((~observed_positive) & (~predicted_positive) & loss_mask).float().sum(),
                )
            )
        local_statistics = torch.stack(statistics)
        global_statistics = local_statistics.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_statistics, op=torch.distributed.ReduceOp.SUM)
        token_count = global_statistics[0].clamp_min(1.0)
        metrics = {
            "raw_target_mean": global_statistics[1] / token_count,
            "transformed_mae": global_statistics[2] / token_count,
            "transformed_rmse": torch.sqrt(global_statistics[3] / token_count),
            "huber_loss": global_statistics[4]
            / global_statistics[5].clamp_min(1.0),
            "sign_aux_loss": global_statistics[6]
            / global_statistics[7].clamp_min(1.0),
            "weighted_sign_aux_loss": sign_aux_weight
            * global_statistics[6]
            / global_statistics[7].clamp_min(1.0),
            "sign_aux_token_rate": global_statistics[7] / token_count,
            "exact_zero_target_rate": global_statistics[8] / token_count,
            "near_zero_target_rate": global_statistics[9] / token_count,
        }
        offset = 10
        for prefix, _ in classification_groups:
            true_positive, false_positive, false_negative, true_negative = global_statistics[
                offset : offset + 4
            ]
            predicted_count = true_positive + false_positive
            target_count = true_positive + false_negative
            precision = true_positive / predicted_count.clamp_min(1.0)
            recall = true_positive / target_count.clamp_min(1.0)
            metrics.update(
                {
                    f"{prefix}_target_rate": target_count / token_count,
                    f"{prefix}_predicted_positive_rate": predicted_count / token_count,
                    f"{prefix}_accuracy": (true_positive + true_negative) / token_count,
                    f"{prefix}_precision": precision,
                    f"{prefix}_recall": recall,
                    f"{prefix}_f1": 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12),
                }
            )
            offset += 4

        # Preserve the historical generic keys for dashboards while making
        # their threshold explicit through the target-specific metrics below.
        # In particular, forward-KL now uses router_kl_threshold (0.05 in the
        # registered Qwen3 runs), rather than the old accidental KL > 0 event.
        legacy_prefix = "harm" if self.config.router_one_step_target == "signed_nll" else "kl"
        metrics.update(
            {
                "harm_rate": metrics[f"{legacy_prefix}_target_rate"],
                "predicted_harm_rate": metrics[f"{legacy_prefix}_predicted_positive_rate"],
                "harm_accuracy": metrics[f"{legacy_prefix}_accuracy"],
                "harm_precision": metrics[f"{legacy_prefix}_precision"],
                "harm_recall": metrics[f"{legacy_prefix}_recall"],
                "harm_f1": metrics[f"{legacy_prefix}_f1"],
            }
        )
        return loss, metrics

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Qwen3OneStepRouterCausalLMOutput:
        del logits_to_keep
        if labels is None:
            raise ValueError("One-step Router supervision requires causal-LM labels.")

        full_hidden, local_hidden, input_embeddings, position_ids, valid_mask = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        target_name = self.config.router_one_step_target
        return_comparison_oracles = bool(self.config.router_return_token_nll)
        divergence = None
        full_nll = None
        local_nll = None
        if target_name == "signed_nll" and not return_comparison_oracles:
            full_nll, local_nll = self._compute_token_nll_only(
                full_hidden, local_hidden, labels, valid_mask, position_ids
            )
        elif target_name == "forward_kl" and not return_comparison_oracles:
            divergence = self._compute_router_divergence(full_hidden, local_hidden)
        else:
            divergence, full_nll, local_nll = self._compute_router_oracles(
                full_hidden,
                local_hidden,
                labels=labels,
                valid_token_mask=valid_mask,
                position_ids=position_ids,
            )

        if target_name == "signed_nll":
            if full_nll is None or local_nll is None:
                raise AssertionError("signed_nll target requires aligned Full/Local NLL.")
            raw_target = local_nll - full_nll
        else:
            if divergence is None:
                raise AssertionError("forward_kl target requires Full-to-Local KL.")
            raw_target = divergence

        loss_mask = self._get_router_loss_mask(valid_mask, labels, position_ids)
        if not torch.isfinite(raw_target[loss_mask]).all():
            raise FloatingPointError("One-step oracle target is non-finite inside the Router loss mask.")
        transformed_target = self._transform_target(raw_target)
        safe_transformed_target = torch.where(
            loss_mask, transformed_target, torch.zeros_like(transformed_target)
        )

        router_features = self._build_router_features(
            input_embeddings, full_hidden, position_ids, valid_mask
        )
        router_logits = self.router_head(router_features)
        if self.config.router_check_finite:
            _check_router_tensor_is_finite("router_logits", router_logits)
            _check_router_tensor_is_finite(
                "router_one_step_target", safe_transformed_target
            )

        router_loss, metrics = self._compute_one_step_loss(
            router_logits,
            raw_target,
            safe_transformed_target,
            loss_mask,
        )
        loss = router_loss * float(self.config.router_loss_weight)
        predicted_raw = self._inverse_target(router_logits)

        if return_dict is False:
            return loss, router_logits, safe_transformed_target, raw_target

        target_prefix = "signed_nll" if target_name == "signed_nll" else "forward_kl"
        probe_tensors = {
            "router_logits": router_logits.detach(),
            "router_targets": safe_transformed_target.detach(),
            "router_raw_target": torch.where(
                loss_mask, raw_target, torch.zeros_like(raw_target)
            ).detach(),
            "router_predicted_raw_target": predicted_raw.detach(),
            "router_feature_l2": router_features.detach().float().norm(dim=-1),
            "position_ids": position_ids.detach(),
            "valid_token_mask": loss_mask.detach(),
            **({"input_ids": input_ids.detach()} if input_ids is not None else {}),
        }
        if divergence is not None:
            probe_tensors["router_divergence"] = divergence.detach()
        if full_nll is not None:
            probe_tensors["full_token_nll"] = full_nll.detach()
            probe_tensors["local_token_nll"] = local_nll.detach()

        training_metrics = {
            f"router/{target_prefix}_mean": metrics["raw_target_mean"],
            "router/transformed_mae": metrics["transformed_mae"],
            "router/transformed_rmse": metrics["transformed_rmse"],
            "router/harm_rate": metrics["harm_rate"],
            "router/predicted_harm_rate": metrics["predicted_harm_rate"],
            "router/harm_accuracy": metrics["harm_accuracy"],
            "router/harm_precision": metrics["harm_precision"],
            "router/harm_recall": metrics["harm_recall"],
            "router/harm_f1": metrics["harm_f1"],
        }
        if target_name == "signed_nll":
            for metric_name in (
                "target_rate",
                "predicted_positive_rate",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ):
                training_metrics[f"router/sign_{metric_name}"] = metrics[f"sign_{metric_name}"]
                training_metrics[f"router/control_harm_{metric_name}"] = metrics[
                    f"harm_{metric_name}"
                ]
        else:
            for metric_name in (
                "target_rate",
                "predicted_positive_rate",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ):
                training_metrics[f"router/kl_threshold_{metric_name}"] = metrics[
                    f"kl_{metric_name}"
                ]

        return Qwen3OneStepRouterCausalLMOutput(
            loss=loss,
            router_loss=router_loss,
            router_logits=router_logits,
            router_targets=safe_transformed_target,
            router_raw_target=raw_target,
            router_predicted_raw_target=predicted_raw,
            router_divergence=divergence,
            full_token_nll=full_nll,
            local_token_nll=local_nll,
            training_metrics=training_metrics,
            training_probe_tensors=probe_tensors,
            full_hidden_states=full_hidden if bool(self.config.router_return_hidden_states) else None,
            local_hidden_states=local_hidden if bool(self.config.router_return_hidden_states) else None,
        )


ModelClass = Qwen3OneStepRouterForCausalLM
