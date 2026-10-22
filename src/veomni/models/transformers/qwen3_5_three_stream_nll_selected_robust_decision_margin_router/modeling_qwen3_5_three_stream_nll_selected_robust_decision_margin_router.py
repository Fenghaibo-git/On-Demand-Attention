"""Robust decision-margin regression for the Qwen3.5 three-stream Router.

This is an objective-only ablation of the completed Qwen3.5-2B signed-DeltaNLL
Huber control.  For every eligible token it defines

``g = NLL_local - NLL_full``
``d = g - 0.001``
``y = sign(d) * log1p(abs(d))``

and minimizes the equally weighted eligible-token mean Huber loss between the
unbounded Router score and ``y`` with delta 1.  The data, frozen backbone, A0
head, initialization, optimizer, learning-rate trajectory, three compute
streams, and three Router inputs remain unchanged.  In particular, the input
history selector remains the paired control's strict ``g < 0`` rule; it is not
moved to 0.001 as a hidden second intervention.

The learned score is a robust decision-margin statistic: under conditional
noise it targets the Huber M-estimator in transformed-margin space.  It is not
strictly ``E[d | z]`` and must not be reported as such.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Tuple

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from ..qwen3_5_three_stream_router.modeling_qwen3_5_three_stream_router import (
    Qwen3_5ThreeStreamNllSelectedRouterForCausalLM,
)


_PUBLIC_OBJECTIVE = "robust_decision_margin_huber"
_ROBUST_MARGIN_DEFAULTS = {
    "router_robust_decision_margin_objective_schema_version": 1,
    "router_robust_decision_margin_objective": (
        "eligible_token_mean_huber_on_signed_log1p_shifted_margin"
    ),
    "router_robust_decision_margin_lambda": 0.001,
    "router_robust_decision_margin_output_space": (
        "signed_log1p_robust_decision_margin"
    ),
    "router_robust_decision_margin_loss_weighting": "eligible_token_equal",
    "router_robust_decision_margin_huber_delta": 1.0,
    "router_robust_decision_margin_runtime_score_threshold": 0.0,
    "router_robust_decision_margin_interpretation": (
        "huber_m_estimate_not_conditional_mean"
    ),
}


def robust_decision_margin_objective_schema_payload(
    config: Qwen3_5TextConfig,
) -> dict:
    """Return the immutable objective/checkpoint compatibility contract."""

    return {
        "format": "qwen3.5-hybrid-three-stream-robust-decision-margin-objective-v1",
        "schema_version": int(
            config.router_robust_decision_margin_objective_schema_version
        ),
        "model_family": "Qwen3.5-2B-text",
        "structural_router_schema_sha256": str(config.router_schema_sha256),
        "paired_control_objective": "signed_log1p_delta_nll_huber_delta1",
        "raw_quality_gap": "nll_local_minus_nll_full",
        "decision_boundary_shift": float(
            config.router_robust_decision_margin_lambda
        ),
        "decision_margin": "raw_quality_gap_minus_decision_boundary_shift",
        "target_transform": "sign(d)*log1p(abs(d))",
        "loss": str(config.router_robust_decision_margin_objective),
        "huber_delta": float(config.router_robust_decision_margin_huber_delta),
        "weighting": str(config.router_robust_decision_margin_loss_weighting),
        "normalization": "global_eligible_token_mean_across_data_parallel_ranks",
        "output_space": str(config.router_robust_decision_margin_output_space),
        "runtime_full_rule": "score_strictly_greater_than_zero",
        "runtime_score_threshold": float(
            config.router_robust_decision_margin_runtime_score_threshold
        ),
        "input_history_local_rule": "raw_quality_gap_strictly_less_than_zero",
        "input_history_nll_threshold": float(config.router_history_nll_threshold),
        "input_contract_relative_to_control": "unchanged",
        "interpretation": str(
            config.router_robust_decision_margin_interpretation
        ),
        "conditional_estimand": (
            "Huber_M_estimator_of_signed_log1p_decision_margin_given_causal_features"
        ),
        "is_strict_conditional_expectation_of_decision_margin": False,
        "bce": False,
        "sigmoid": False,
        "class_balance": False,
        "magnitude_weight": False,
        "noise_dead_zone": False,
        "target_clip": False,
        "sign_auxiliary_loss": False,
    }


def robust_decision_margin_objective_schema_sha256(
    config: Qwen3_5TextConfig,
) -> str:
    encoded = json.dumps(
        robust_decision_margin_objective_schema_payload(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM(
    Qwen3_5ThreeStreamNllSelectedRouterForCausalLM
):
    """A0 Router trained as a robust shifted decision-margin regressor."""

    def __init__(self, config: Qwen3_5TextConfig):
        for name, value in _ROBUST_MARGIN_DEFAULTS.items():
            if not hasattr(config, name):
                setattr(config, name, value)

        public_objective = getattr(config, "router_objective", _PUBLIC_OBJECTIVE)
        if public_objective != _PUBLIC_OBJECTIVE:
            raise ValueError(
                "Qwen3.5 robust-margin Router requires "
                f"router_objective={_PUBLIC_OBJECTIVE!r}."
            )

        # The common structural defaults predate independent objective names.
        # Satisfy that legacy enum guard only during parent construction and
        # restore the truthful public/exported objective immediately after it.
        config.router_objective = "weighted_bce"
        try:
            super().__init__(config)
        finally:
            config.router_objective = public_objective

        if int(config.router_robust_decision_margin_objective_schema_version) != 1:
            raise ValueError("Only robust decision-margin objective schema v1 is supported.")
        if config.router_robust_decision_margin_objective != (
            "eligible_token_mean_huber_on_signed_log1p_shifted_margin"
        ):
            raise ValueError("Unknown robust decision-margin objective.")
        if config.router_objective != _PUBLIC_OBJECTIVE:
            raise ValueError("The public robust-margin objective was not restored.")

        margin_lambda = float(config.router_robust_decision_margin_lambda)
        if not math.isfinite(margin_lambda) or margin_lambda != 0.001:
            raise ValueError("The Qwen3.5 robust decision boundary shift is exactly 0.001.")
        if float(config.router_history_nll_threshold) != 0.0:
            raise ValueError(
                "The objective-only ablation must retain the control's g < 0 "
                "input-history selector."
            )
        if config.router_one_step_transform != "signed_log1p":
            raise ValueError("Robust decision-margin supervision requires signed-log1p.")
        if float(config.router_one_step_nll_scale) != 1.0:
            raise ValueError("Robust decision-margin supervision fixes target scale to 1.")
        if float(config.router_one_step_huber_delta) != 1.0:
            raise ValueError("The inherited Huber delta must be exactly 1.")
        if float(config.router_robust_decision_margin_huber_delta) != 1.0:
            raise ValueError("The objective contract fixes Huber delta to 1.")
        if float(config.router_one_step_positive_weight) != 1.0:
            raise ValueError("Robust decision-margin supervision forbids class weighting.")
        if float(config.router_one_step_sign_aux_weight) != 0.0:
            raise ValueError("Robust decision-margin supervision forbids sign auxiliaries.")
        if float(config.router_one_step_harm_threshold) != 0.0:
            raise ValueError("Decision diagnostics must use the exact d = 0 boundary.")
        if config.router_robust_decision_margin_loss_weighting != (
            "eligible_token_equal"
        ):
            raise ValueError("Every eligible token must have equal objective weight.")
        if config.router_robust_decision_margin_output_space != (
            "signed_log1p_robust_decision_margin"
        ):
            raise ValueError("Unexpected robust decision-margin output space.")
        if float(config.router_robust_decision_margin_runtime_score_threshold) != 0.0:
            raise ValueError("The runtime score boundary is exactly zero.")
        if config.router_robust_decision_margin_interpretation != (
            "huber_m_estimate_not_conditional_mean"
        ):
            raise ValueError("The robust-margin interpretation must remain explicit.")

        actual_hash = robust_decision_margin_objective_schema_sha256(config)
        configured_hash = getattr(
            config, "router_robust_decision_margin_objective_sha256", None
        )
        if configured_hash not in {None, "", actual_hash}:
            raise ValueError(
                "Robust decision-margin objective hash mismatch: configured "
                f"{configured_hash}, computed {actual_hash}."
            )
        config.router_robust_decision_margin_objective_sha256 = actual_hash
        self._last_robust_margin_metrics: dict[str, torch.Tensor] = {}

    def _decision_margin(self, raw_signed_nll: torch.Tensor) -> torch.Tensor:
        return raw_signed_nll.float() - float(
            self.config.router_robust_decision_margin_lambda
        )

    def _transform_target(self, raw_signed_nll: torch.Tensor) -> torch.Tensor:
        margin = self._decision_margin(raw_signed_nll)
        return margin.sign() * torch.log1p(margin.abs())

    def _inverse_target(self, transformed_margin: torch.Tensor) -> torch.Tensor:
        transformed_margin = transformed_margin.float()
        return transformed_margin.sign() * torch.expm1(transformed_margin.abs())

    def _compute_one_step_loss(
        self,
        router_scores: torch.Tensor,
        raw_signed_nll: torch.Tensor,
        transformed_margin: torch.Tensor,
        loss_mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, dict]:
        margin = self._decision_margin(raw_signed_nll)
        loss, metrics = super()._compute_one_step_loss(
            router_scores,
            margin,
            transformed_margin,
            loss_mask,
        )
        self._last_robust_margin_metrics = {
            name: value.detach() for name, value in metrics.items()
        }
        return loss, metrics

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        if not hasattr(outputs, "training_metrics"):
            return outputs

        metrics = self._last_robust_margin_metrics
        inherited = dict(outputs.training_metrics or {})
        margin_lambda = float(self.config.router_robust_decision_margin_lambda)
        public_metrics = {
            "router/objective/total_loss": metrics["huber_loss"],
            "router/objective/token_mean_huber": metrics["huber_loss"],
            "router/objective/decision_boundary_shift": outputs.loss.detach().new_tensor(
                margin_lambda
            ),
            "router/labels/raw_signed_nll_mean": metrics["raw_target_mean"]
            + margin_lambda,
            "router/labels/robust_decision_margin_mean": metrics[
                "raw_target_mean"
            ],
            "router/labels/exact_zero_robust_decision_margin_rate": metrics[
                "exact_zero_target_rate"
            ],
            "router/labels/near_zero_robust_decision_margin_rate": metrics[
                "near_zero_target_rate"
            ],
            "router/regression/transformed_mae": metrics["transformed_mae"],
            "router/regression/transformed_rmse": metrics["transformed_rmse"],
            "router/decision/target_full_rate": metrics["sign_target_rate"],
            "router/decision/predicted_full_rate": metrics[
                "sign_predicted_positive_rate"
            ],
            "router/decision/accuracy": metrics["sign_accuracy"],
            "router/decision/precision": metrics["sign_precision"],
            "router/decision/recall": metrics["sign_recall"],
            "router/decision/f1": metrics["sign_f1"],
        }
        for name, value in inherited.items():
            if name.startswith("router/history/"):
                public_metrics[name] = value
        outputs.training_metrics = public_metrics

        raw_gap = outputs.router_raw_target.float()
        probes = dict(outputs.training_probe_tensors or {})
        loss_mask = probes.get("valid_token_mask")
        if loss_mask is None:
            loss_mask = torch.isfinite(raw_gap)
        loss_mask = loss_mask.bool()
        safe_gap = torch.where(loss_mask, raw_gap, torch.zeros_like(raw_gap))
        margin = torch.where(
            loss_mask,
            self._decision_margin(raw_gap),
            torch.zeros_like(raw_gap),
        )
        predicted_margin = self._inverse_target(outputs.router_transformed_score)
        outputs.router_raw_target = margin
        outputs.router_predicted_raw_target = predicted_margin
        for stale_name in (
            "router_targets",
            "router_raw_target",
            "router_predicted_raw_target",
        ):
            probes.pop(stale_name, None)
        probes.update(
            {
                "router_raw_signed_nll_gap": safe_gap.detach(),
                "router_robust_decision_margin": margin.detach(),
                "router_transformed_robust_decision_margin_target": (
                    outputs.router_targets.detach()
                ),
                "router_predicted_robust_decision_margin": (
                    predicted_margin.detach()
                ),
                "router_robust_decision_score": (
                    outputs.router_transformed_score.detach()
                ),
                "router_full_target": (margin.gt(0.0) & loss_mask).detach(),
            }
        )
        outputs.training_probe_tensors = probes
        return outputs


ModelClass = Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM


__all__ = [
    "Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM",
    "robust_decision_margin_objective_schema_payload",
    "robust_decision_margin_objective_schema_sha256",
]
