"""Robust signed-log margin objective shared by the 32B and 27B experiments."""

import hashlib
import json
import math

import torch


ROBUST_DEFAULTS = {
    "router_eligible_only": True,
    "router_objective": "robust_decision_margin_huber",
    "router_robust_decision_margin_objective_schema_version": 1,
    "router_robust_decision_margin_objective": "eligible_token_mean_huber_on_signed_log1p_shifted_margin",
    "router_robust_decision_margin_lambda": 0.001,
    "router_robust_decision_margin_output_space": "signed_log1p_robust_decision_margin",
    "router_robust_decision_margin_loss_weighting": "eligible_token_equal",
    "router_robust_decision_margin_huber_delta": 1.0,
    "router_robust_decision_margin_runtime_score_threshold": 0.0,
    "router_robust_decision_margin_interpretation": "huber_m_estimate_not_conditional_mean",
    "router_history_nll_threshold": 0.0,
    "router_one_step_harm_threshold": 0.0,
    "router_one_step_huber_delta": 1.0,
    "router_one_step_nll_scale": 1.0,
    "router_one_step_positive_weight": 1.0,
    "router_one_step_sign_aux_weight": 0.0,
    "router_one_step_target": "signed_nll",
    "router_one_step_transform": "signed_log1p",
    "router_loss_weight": 1.0,
}


def close(observed, expected, label):
    if not math.isfinite(float(observed)) or not math.isclose(
        float(observed), float(expected), rel_tol=1e-6, abs_tol=2e-7
    ):
        raise ValueError(f"{label} drifted: {observed!r} versus {expected!r}")


def set_robust_defaults(config):
    for name, expected in ROBUST_DEFAULTS.items():
        if not hasattr(config, name):
            setattr(config, name, expected)
        observed = getattr(config, name)
        if isinstance(expected, float):
            close(observed, expected, name)
        elif observed != expected:
            raise ValueError(f"{name} drifted: {observed!r}")


def objective_payload(config):
    reduction = getattr(config, "router_optimizer_step_reduction", None)
    if reduction not in (None, "global_eligible_token_mean_v1"):
        raise ValueError(f"unsupported large-model optimizer reduction: {reduction!r}")
    return {
        "format": "qwen-large-three-stream-robust-margin-objective-v1",
        "model_type": config.model_type,
        "base_model": str(getattr(config, "router_base_model_path", "")),
        "structural_router_schema_sha256": config.router_schema_sha256,
        "raw_gap": "nll_local_minus_nll_full",
        "decision_margin": "g_minus_0.001",
        "target": "sign(d)*log1p(abs(d))",
        "loss": "eligible_token_mean_huber",
        "delta": 1.0,
        "weighting": "eligible_token_equal",
        "microbatch_reduction": "global_eligible_token_mean_across_DP",
        "optimizer_reduction": (
            "global_eligible_token_mean_v1" if reduction is not None
            else "mean_of_global_microbatch_means_as_matched_control"
        ),
        "history_local_rule": "g_strictly_less_than_zero",
        "history_equal_action": "full",
        "runtime_full_rule": "score_strictly_greater_than_zero",
        "runtime_equal_action": "local",
        "interpretation": "Huber_M_location_of_signed_log1p_margin_not_E[d|z]",
        "class_or_magnitude_weight": False,
        "near_zero_filter": False,
    }


def objective_sha256(config):
    return hashlib.sha256(json.dumps(
        objective_payload(config), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def finish_contract(model):
    config = model.config
    set_robust_defaults(config)
    observed = objective_sha256(config)
    if getattr(config, "router_robust_decision_margin_objective_sha256", None) not in (None, "", observed):
        raise ValueError("robust-margin objective hash mismatch")
    config.router_robust_decision_margin_objective_sha256 = observed
    count = sum(p.numel() for p in model.router_head.parameters())
    if count != int(config.router_expected_head_parameter_count):
        raise ValueError(f"head parameter count mismatch: {count}")
    if len(model.router_head.state_dict()) != int(config.router_expected_head_tensor_count):
        raise ValueError("head tensor count mismatch")
    if any(p.requires_grad and not n.startswith("router_head.") for n, p in model.named_parameters()):
        raise ValueError("backbone must be frozen")
    model._last_robust_margin_metrics = {}


class RobustMarginObjectiveMixin:
    """Same numerical target/loss/probes as the qualified history-zero 2B arm."""

    def _decision_margin(self, gap):
        return gap.float() - float(self.config.router_robust_decision_margin_lambda)

    def _transform_target(self, gap):
        margin = self._decision_margin(gap)
        return margin.sign() * torch.log1p(margin.abs())

    def _inverse_target(self, score):
        score = score.float()
        return score.sign() * torch.expm1(score.abs())

    def _compute_one_step_loss(self, scores, raw_gap, transformed, mask):
        loss, metrics = super()._compute_one_step_loss(
            scores, self._decision_margin(raw_gap), transformed, mask
        )
        self._last_robust_margin_metrics = {key: value.detach() for key, value in metrics.items()}
        return loss, metrics

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        if not hasattr(outputs, "training_metrics"):
            return outputs
        metrics = self._last_robust_margin_metrics
        public = {
            "router/objective/total_loss": metrics["huber_loss"],
            "router/objective/token_mean_huber": metrics["huber_loss"],
            "router/objective/decision_boundary_shift": outputs.loss.detach().new_tensor(0.001),
            "router/labels/raw_signed_nll_mean": metrics["raw_target_mean"] + 0.001,
            "router/labels/robust_decision_margin_mean": metrics["raw_target_mean"],
            "router/labels/exact_zero_robust_decision_margin_rate": metrics["exact_zero_target_rate"],
            "router/labels/near_zero_robust_decision_margin_rate": metrics["near_zero_target_rate"],
            "router/regression/transformed_mae": metrics["transformed_mae"],
            "router/regression/transformed_rmse": metrics["transformed_rmse"],
            "router/decision/target_full_rate": metrics["sign_target_rate"],
            "router/decision/predicted_full_rate": metrics["sign_predicted_positive_rate"],
            **{f"router/decision/{key}": metrics[f"sign_{key}"]
               for key in ("accuracy", "precision", "recall", "f1")},
        }
        public.update({k: v for k, v in (outputs.training_metrics or {}).items()
                       if k.startswith("router/history/")})
        outputs.training_metrics = public
        gap = outputs.router_raw_target.float()
        probes = dict(outputs.training_probe_tensors or {})
        mask = probes["valid_token_mask"].bool()
        safe_gap = torch.where(mask, gap, torch.zeros_like(gap))
        margin = torch.where(mask, self._decision_margin(gap), torch.zeros_like(gap))
        predicted = self._inverse_target(outputs.router_transformed_score)
        outputs.router_raw_target = margin
        outputs.router_predicted_raw_target = predicted
        for stale in ("router_targets", "router_raw_target", "router_predicted_raw_target"):
            probes.pop(stale, None)
        probes.update({
            "router_raw_signed_nll_gap": safe_gap.detach(),
            "router_robust_decision_margin": margin.detach(),
            "router_transformed_robust_decision_margin_target": outputs.router_targets.detach(),
            "router_predicted_robust_decision_margin": predicted.detach(),
            "router_robust_decision_score": outputs.router_transformed_score.detach(),
            "router_full_target": (margin.gt(0) & mask).detach(),
        })
        outputs.training_probe_tensors = probes
        return outputs
