"""Versioned, stable comparison metrics for Router training runs.

Objective-specific metrics remain untouched. The helpers here derive a small
comparison namespace, validate its semantics, and aggregate Router rates over
all eligible tokens in an optimizer step instead of averaging microbatch
ratios.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROUTER_WANDB_METRIC_SCHEMA_VERSION = "router-comparison-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RATE_TOLERANCE = 1.0e-6
_CONSISTENCY_TOLERANCE = 2.0e-5

ROUTER_SUFFICIENT_STATISTIC_NAMES = frozenset(
    {
        "eligible",
        "raw_gap_sum",
        "history_full",
        "history_local",
        "target_full",
        "predicted_full",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    }
)

# Logical fields and all historical spellings. Multiple aliases may coexist;
# in that case they must agree before they enter the canonical namespace.
ROUTER_CORE_ALIASES: dict[str, tuple[str, ...]] = {
    "objective_loss": (
        "router/objective/total_loss",
        "router/huber_loss",
    ),
    "raw_quality_gap_mean": (
        "router/labels/raw_signed_nll_mean",
        "router/regression/signed_nll_mean",
        "router/regret/raw_signed_nll_mean",
        "router/signed_nll_mean",
    ),
    "target_full_rate": (
        "router/decision/target_full_rate",
        "router/sign0/target_full_rate",
        "router/sign_target_rate",
    ),
    "predicted_full_rate": (
        "router/decision/predicted_full_rate",
        "router/decision/reference_predicted_full_rate",
        "router/sign0/predicted_full_rate",
        "router/sign_predicted_positive_rate",
    ),
    "accuracy": (
        "router/decision/accuracy",
        "router/decision/reference_accuracy",
        "router/sign0/accuracy",
        "router/sign_accuracy",
    ),
    "precision": (
        "router/decision/precision",
        "router/decision/reference_precision",
        "router/sign0/precision",
        "router/sign_precision",
    ),
    "recall": (
        "router/decision/recall",
        "router/decision/reference_recall",
        "router/sign0/recall",
        "router/sign_recall",
    ),
    "f1": (
        "router/decision/f1",
        "router/decision/reference_f1",
        "router/sign0/f1",
        "router/sign_f1",
    ),
    "history_full_selected_rate": ("router/history/full_selected_rate",),
    "history_local_selected_rate": ("router/history/local_selected_rate",),
    "history_selection_threshold": ("router/history/selection_threshold",),
}

_RATE_FIELDS = {
    "target_full_rate",
    "predicted_full_rate",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "history_full_selected_rate",
    "history_local_selected_rate",
}

ROUTER_CORE_OUTPUT_NAMES = {
    "objective_loss": "comparison/optimization/router_objective_loss",
    "raw_quality_gap_mean": "comparison/router/label/raw_quality_gap_mean_nats",
    "target_full_rate": "comparison/router/reference/target_full_rate",
    "predicted_full_rate": "comparison/router/reference/predicted_full_rate",
    "accuracy": "comparison/router/reference/accuracy",
    "precision": "comparison/router/reference/precision",
    "recall": "comparison/router/reference/recall",
    "f1": "comparison/router/reference/f1",
    "history_full_selected_rate": "comparison/router/history/input_full_rate",
    "history_local_selected_rate": "comparison/router/history/input_local_rate",
    "history_selection_threshold": "comparison/router/history/selection_threshold",
}

ROUTER_COUNT_OUTPUT_NAMES = {
    "eligible": "comparison/router/count/eligible_tokens",
    "target_full": "comparison/router/count/target_full_tokens",
    "predicted_full": "comparison/router/count/predicted_full_tokens",
    "true_positive": "comparison/router/count/true_positive_tokens",
    "false_positive": "comparison/router/count/false_positive_tokens",
    "false_negative": "comparison/router/count/false_negative_tokens",
    "true_negative": "comparison/router/count/true_negative_tokens",
}

_SYSTEM_METRICS = {
    "step_time_seconds": "comparison/system/step_time_seconds",
    "tokens_per_second(M)": "comparison/system/input_tokens_per_second_millions",
    "max_memory_allocated(GB)": "comparison/system/max_memory_allocated_gib",
    "max_memory_reserved(GB)": "comparison/system/max_memory_reserved_gib",
    "num_alloc_retries": "comparison/system/cuda_allocation_retries",
    "consume_tokens(B)": "comparison/progress/cumulative_input_tokens_billions",
}

BASE_OUTPUT_NAMES = {
    "loss": "comparison/optimization/train_loss",
    "grad_norm": "comparison/optimization/pre_clip_grad_norm",
    "learning_rate": "comparison/optimization/post_step_learning_rate",
}

REQUIRED_ROUTER_STEP_METRICS = frozenset(
    {*ROUTER_CORE_OUTPUT_NAMES.values(), *ROUTER_COUNT_OUTPUT_NAMES.values()}
)

RULER16K_SUMMARY_OUTPUT_NAMES = {
    "score": "comparison/eval/ruler16k/score_points",
    "full_rate": "comparison/eval/ruler16k/full_rate",
    "gap_to_full": "comparison/eval/ruler16k/score_gap_to_original_full_points",
    "full_local_gap_recovery": "comparison/eval/ruler16k/full_local_gap_recovery",
    "runtime_score_threshold": "comparison/eval/ruler16k/runtime_score_threshold",
    "budget_calibrated": "comparison/eval/ruler16k/budget_calibrated",
}


def load_router_wandb_metric_schema_identity() -> dict[str, str]:
    """Load and hash the machine schema shipped beside the trainer source."""

    path = (
        Path(__file__).resolve().parents[2]
        / "configs/metric_schemas/router_wandb_comparison_v1.json"
    )
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    if payload.get("schema_version") != ROUTER_WANDB_METRIC_SCHEMA_VERSION:
        raise ValueError(f"Router W&B metric schema version mismatch: {path}")
    return {
        "schema_version": ROUTER_WANDB_METRIC_SCHEMA_VERSION,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _finite_float(value: Any, label: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric, observed {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, observed {result!r}")
    return result


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _resolve_alias(record: Mapping[str, Any], name: str) -> float | None:
    observed = [
        (key, _finite_float(record[key], key))
        for key in ROUTER_CORE_ALIASES[name]
        if key in record
    ]
    if not observed:
        return None
    reference_key, reference_value = observed[0]
    for key, value in observed[1:]:
        if not math.isclose(
            value, reference_value, rel_tol=1.0e-5, abs_tol=1.0e-6
        ):
            raise ValueError(
                f"W&B comparison aliases disagree for {name}: "
                f"{reference_key}={reference_value}, {key}={value}"
            )
    return reference_value


def resolve_router_core_values(
    record: Mapping[str, Any], *, required: bool
) -> dict[str, float]:
    resolved: dict[str, float] = {}
    missing: list[str] = []
    for name in ROUTER_CORE_ALIASES:
        value = _resolve_alias(record, name)
        if value is None:
            missing.append(name)
            continue
        if name in _RATE_FIELDS:
            if not -_RATE_TOLERANCE <= value <= 1.0 + _RATE_TOLERANCE:
                raise ValueError(f"{name} is outside [0, 1]: {value}")
            value = min(1.0, max(0.0, value))
        resolved[name] = value
    if required and missing:
        raise ValueError(
            "contracted Router run lacks comparison core metrics: "
            + ", ".join(missing)
        )
    if {
        "history_full_selected_rate",
        "history_local_selected_rate",
    }.issubset(resolved):
        total = (
            resolved["history_full_selected_rate"]
            + resolved["history_local_selected_rate"]
        )
        if not math.isclose(
            total, 1.0, rel_tol=0.0, abs_tol=_CONSISTENCY_TOLERANCE
        ):
            raise ValueError(
                "history Full/Local rates must sum to one for comparison; "
                f"observed {total}"
            )
    return resolved


def validate_wandb_comparison_contract(
    train_config: Mapping[str, Any], *, required: bool
) -> dict[str, Any]:
    """Validate run-level fields that make W&B runs meaningfully groupable."""

    fields = {
        "metric_schema_version": train_config.get("wandb_metric_schema_version"),
        "comparison_family": train_config.get("wandb_comparison_family"),
        "comparison_protocol_hash": train_config.get(
            "wandb_comparison_protocol_hash"
        ),
    }
    present = {
        name: value for name, value in fields.items() if value not in (None, "")
    }
    if not present and not required:
        return {
            "enabled": False,
            "metric_schema_version": ROUTER_WANDB_METRIC_SCHEMA_VERSION,
            "comparison_family": None,
            "comparison_protocol_hash": None,
        }
    missing = [name for name, value in fields.items() if not str(value or "").strip()]
    if missing:
        raise ValueError(f"missing W&B comparison contract fields: {missing}")
    if fields["metric_schema_version"] != ROUTER_WANDB_METRIC_SCHEMA_VERSION:
        raise ValueError(
            "unsupported W&B metric schema: "
            f"{fields['metric_schema_version']!r}; expected "
            f"{ROUTER_WANDB_METRIC_SCHEMA_VERSION!r}"
        )
    family = str(fields["comparison_family"])
    group = str(train_config.get("wandb_group") or "")
    if group != family:
        raise ValueError(
            "wandb_group must equal wandb_comparison_family for a contracted run: "
            f"group={group!r}, family={family!r}"
        )
    protocol_hash = str(fields["comparison_protocol_hash"])
    if not _SHA256.fullmatch(protocol_hash):
        raise ValueError("wandb_comparison_protocol_hash must be a lowercase SHA-256")
    return {"enabled": True, **fields}


def build_router_metric_semantics_config(model_config: Any) -> dict[str, Any]:
    """Resolve static label, score, and history semantics into W&B config."""

    boundary_policy = "full_when_strictly_greater_local_on_equal"
    if _config_value(
        model_config, "router_robust_decision_margin_objective_schema_version"
    ) is not None:
        objective_name = _config_value(
            model_config,
            "router_robust_decision_margin_objective",
            "robust_decision_margin_huber",
        )
        objective_hash = _config_value(
            model_config, "router_robust_decision_margin_objective_sha256"
        )
        decision_boundary = float(
            _config_value(model_config, "router_robust_decision_margin_lambda")
        )
        score_threshold = float(
            _config_value(
                model_config,
                "router_robust_decision_margin_runtime_score_threshold",
            )
        )
        score_space = _config_value(
            model_config, "router_robust_decision_margin_output_space"
        )
    elif _config_value(model_config, "router_focal_objective_schema_version") is not None:
        objective_name = _config_value(
            model_config, "router_focal_objective", "standard_focal"
        )
        objective_hash = _config_value(
            model_config, "router_focal_objective_sha256"
        )
        decision_boundary = float(
            _config_value(model_config, "router_focal_decision_nll_threshold")
        )
        score_threshold = float(
            _config_value(model_config, "router_focal_reference_logit_threshold")
        )
        score_space = _config_value(model_config, "router_focal_output_space")
        boundary_policy = str(
            _config_value(
                model_config,
                "router_focal_boundary_policy",
                "full_when_greater_than_or_equal",
            )
        )
    elif _config_value(
        model_config, "router_decision_regret_objective_schema_version"
    ) is not None:
        objective_name = _config_value(
            model_config,
            "router_decision_regret_objective",
            "decision_regret_weighted_bce",
        )
        objective_hash = _config_value(
            model_config, "router_decision_regret_objective_sha256"
        )
        decision_boundary = float(
            _config_value(model_config, "router_decision_regret_compute_cost_lambda")
        )
        score_threshold = float(
            _config_value(
                model_config, "router_decision_regret_reference_logit_threshold", 0.0
            )
        )
        score_space = _config_value(
            model_config, "router_decision_regret_output_space"
        )
    elif _config_value(model_config, "router_margin_objective_schema_version") is not None:
        objective_name = _config_value(
            model_config, "router_margin_objective", "decision_margin_huber"
        )
        objective_hash = _config_value(
            model_config, "router_margin_objective_sha256"
        )
        decision_boundary = float(
            _config_value(model_config, "router_compute_cost_lambda")
        )
        score_threshold = float(
            _config_value(model_config, "router_margin_runtime_score_threshold")
        )
        score_space = _config_value(model_config, "router_margin_output_space")
    elif _config_value(model_config, "router_regret_objective_schema_version") is not None:
        objective_name = _config_value(
            model_config, "router_regret_objective", "regret_weighted_bce"
        )
        objective_hash = _config_value(
            model_config, "router_regret_objective_sha256"
        ) or _config_value(model_config, "router_schema_sha256")
        regret_schema_version = int(
            _config_value(model_config, "router_regret_objective_schema_version")
        )
        if regret_schema_version >= 3:
            decision_boundary = float(
                _config_value(model_config, "router_regret_compute_cost_lambda")
            )
        else:
            decision_boundary = float(
                _config_value(model_config, "router_regret_decision_nll_threshold")
            )
            boundary_policy = "full_when_greater_than_or_equal"
        score_threshold = float(
            _config_value(model_config, "router_regret_runtime_logit_threshold")
        )
        score_space = _config_value(model_config, "router_regret_output_space")
    else:
        target = _config_value(model_config, "router_one_step_target", "unknown")
        transform = _config_value(
            model_config, "router_one_step_transform", "unknown"
        )
        objective_name = f"{target}_{transform}_huber"
        objective_hash = _config_value(model_config, "router_schema_sha256")
        if _config_value(model_config, "router_regularization_schema_version") is not None:
            objective_name += "_regularized"
            objective_hash = _config_value(
                model_config, "router_regularization_sha256"
            ) or objective_hash
        decision_boundary = 0.0
        score_threshold = 0.0
        score_space = f"{transform}_regression_score"

    objective_hash = str(objective_hash) if objective_hash else None

    history_selector = str(
        _config_value(model_config, "router_history_selector", "") or ""
    )
    if history_selector == "router_action_logit_strict_gt_zero":
        history_selection_space = "router_score"
        history_selection_threshold = float(
            _config_value(model_config, "router_regret_runtime_logit_threshold", 0.0)
        )
        history_full_rule = "router_score_strictly_greater_than_history_threshold"
        history_local_rule = "router_score_less_than_or_equal_to_history_threshold"
    else:
        history_selection_space = "raw_quality_gap_nats"
        history_selection_threshold = float(
            _config_value(model_config, "router_history_nll_threshold", 0.0)
        )
        history_full_rule = "raw_quality_gap_greater_than_or_equal_to_history_threshold"
        history_local_rule = "raw_quality_gap_strictly_less_than_history_threshold"

    boundary_definitions = {
        "full_when_strictly_greater_local_on_equal": (
            "full_when_raw_quality_gap_strictly_greater_than_boundary",
            "local",
            "gt",
        ),
        "full_when_greater_than_or_equal": (
            "full_when_raw_quality_gap_greater_than_or_equal_to_boundary",
            "full",
            "ge",
        ),
    }
    if boundary_policy not in boundary_definitions:
        raise ValueError(
            f"unsupported Router decision boundary policy: {boundary_policy!r}"
        )
    target_definition, tie_action, boundary_operator = boundary_definitions[
        boundary_policy
    ]

    return {
        "metric_schema_version": ROUTER_WANDB_METRIC_SCHEMA_VERSION,
        "objective_name": str(objective_name),
        "objective_sha256": objective_hash,
        "raw_quality_gap_definition": "next_token_nll_local_minus_next_token_nll_full",
        "raw_quality_gap_unit": "natural_log_nats_per_eligible_token",
        "decision_target_definition": target_definition,
        "decision_boundary_nats": decision_boundary,
        "decision_boundary_operator": boundary_operator,
        "boundary_tie_action": tie_action,
        "reference_score_threshold": score_threshold,
        "reference_score_operator": "gt",
        "score_space": str(score_space),
        "history_selection_space": history_selection_space,
        "history_selection_threshold": history_selection_threshold,
        "rate_unit": "fraction_0_to_1",
        "decision_metric_aggregation": "global_eligible_token_sufficient_statistics_across_dp_and_microbatches",
        "objective_loss_aggregation": (
            "global_eligible_token_mean_across_dp_and_microbatches"
            if _config_value(model_config, "router_optimizer_step_reduction") == "global_eligible_token_mean_v1"
            else "arithmetic_mean_of_global_microbatch_objective_means_matching_gradient_accumulation"
        ),
        "history_full_rule": history_full_rule,
        "history_local_rule": history_local_rule,
    }


def validate_router_metric_semantics_config(
    semantics: Mapping[str, Any], *, required: bool
) -> dict[str, Any]:
    """Validate the static fields that give comparison curves their meaning."""

    required_strings = (
        "metric_schema_version",
        "objective_name",
        "raw_quality_gap_definition",
        "raw_quality_gap_unit",
        "decision_target_definition",
        "decision_boundary_operator",
        "boundary_tie_action",
        "reference_score_operator",
        "score_space",
        "rate_unit",
        "decision_metric_aggregation",
        "objective_loss_aggregation",
        "history_selection_space",
        "history_full_rule",
        "history_local_rule",
    )
    missing = [
        name for name in required_strings if not str(semantics.get(name) or "").strip()
    ]
    if missing:
        raise ValueError(f"Router metric semantics lack fields: {missing}")
    if semantics["metric_schema_version"] != ROUTER_WANDB_METRIC_SCHEMA_VERSION:
        raise ValueError("Router metric semantics use an unsupported schema version")
    if semantics["decision_boundary_operator"] not in {"gt", "ge"}:
        raise ValueError("Router decision boundary operator must be gt or ge")
    if semantics["reference_score_operator"] != "gt":
        raise ValueError("Router reference score operator must be gt")
    if semantics["boundary_tie_action"] not in {"full", "local"}:
        raise ValueError("Router boundary tie action must be Full or Local")
    for name in (
        "decision_boundary_nats",
        "reference_score_threshold",
        "history_selection_threshold",
    ):
        _finite_float(semantics.get(name), name)
    objective_hash = semantics.get("objective_sha256")
    if required and not _SHA256.fullmatch(str(objective_hash or "")):
        raise ValueError(
            "contracted Router metric semantics require an objective SHA-256"
        )
    if required and str(semantics["score_space"]).lower() in {"none", "unknown"}:
        raise ValueError("contracted Router metric semantics require a score space")
    return dict(semantics)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


class RouterCoreMetricAccumulator:
    """Aggregate exact optimizer-step Router metrics from microbatch summaries."""

    def __init__(self, *, strict: bool, token_weighted_objective: bool = False) -> None:
        self.strict = strict
        self.token_weighted_objective = token_weighted_objective
        self.microbatches = 0
        self.objective_loss_sum = 0.0
        self.eligible = 0.0
        self.raw_gap_sum = 0.0
        self.history_full = 0.0
        self.history_local = 0.0
        self.target_full = 0.0
        self.predicted_full = 0.0
        self.true_positive = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0
        self.true_negative = 0.0
        self.history_threshold: float | None = None

    def update(
        self,
        record: Mapping[str, Any],
        *,
        eligible_token_count: int | float | None = None,
        sufficient_statistics: Mapping[str, Any] | None = None,
    ) -> None:
        has_router = any(key.startswith("router/") for key in record)
        if not has_router:
            if self.strict:
                raise ValueError("contracted Router microbatch has no router metrics")
            return
        values = resolve_router_core_values(record, required=self.strict)
        if len(values) != len(ROUTER_CORE_ALIASES):
            return
        if sufficient_statistics is not None:
            missing_statistics = ROUTER_SUFFICIENT_STATISTIC_NAMES.difference(
                sufficient_statistics
            )
            if missing_statistics:
                raise ValueError(
                    "Router sufficient statistics are incomplete: "
                    + ", ".join(sorted(missing_statistics))
                )
            statistics = {
                name: _finite_float(sufficient_statistics[name], name)
                for name in ROUTER_SUFFICIENT_STATISTIC_NAMES
            }
            count = statistics["eligible"]
            if eligible_token_count is not None and not math.isclose(
                count,
                _finite_float(eligible_token_count, "eligible_token_count"),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError(
                    "eligible-token count disagrees with sufficient statistics"
                )
        else:
            if self.strict:
                raise ValueError(
                    "contracted Router microbatch lacks probe-derived sufficient statistics"
                )
            if eligible_token_count is None:
                return
            count = _finite_float(eligible_token_count, "eligible_token_count")
        if count <= 0.0:
            raise ValueError(
                f"eligible_token_count must be positive, observed {count}"
            )

        if sufficient_statistics is not None:
            target = statistics["target_full"]
            predicted = statistics["predicted_full"]
            true_positive = statistics["true_positive"]
            false_positive = statistics["false_positive"]
            false_negative = statistics["false_negative"]
            true_negative = statistics["true_negative"]
            raw_gap_sum = statistics["raw_gap_sum"]
            history_full = statistics["history_full"]
            history_local = statistics["history_local"]
        else:
            target = values["target_full_rate"] * count
            predicted = values["predicted_full_rate"] * count
            true_positive = (
                values["accuracy"] * count - count + target + predicted
            ) / 2.0
            lower = max(0.0, target + predicted - count)
            upper = min(target, predicted)
            slack = max(1.0e-3, count * 2.0e-5)
            if true_positive < lower - slack or true_positive > upper + slack:
                raise ValueError(
                    "inconsistent Router decision rates cannot form a confusion matrix: "
                    f"count={count}, target={target}, predicted={predicted}, "
                    f"accuracy={values['accuracy']}"
                )
            true_positive = min(upper, max(lower, true_positive))
            false_positive = predicted - true_positive
            false_negative = target - true_positive
            true_negative = count - true_positive - false_positive - false_negative
            raw_gap_sum = values["raw_quality_gap_mean"] * count
            history_full = values["history_full_selected_rate"] * count
            history_local = values["history_local_selected_rate"] * count

        count_tolerance = max(1.0e-6, count * 1.0e-10)
        count_identities = (
            (count, true_positive + false_positive + false_negative + true_negative),
            (target, true_positive + false_negative),
            (predicted, true_positive + false_positive),
            (count, history_full + history_local),
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=count_tolerance)
            for left, right in count_identities
        ):
            raise ValueError("Router sufficient-statistic count identities failed")
        if any(
            value < -count_tolerance
            for value in (
                target,
                predicted,
                true_positive,
                false_positive,
                false_negative,
                true_negative,
                history_full,
                history_local,
            )
        ):
            raise ValueError("Router sufficient statistics contain a negative count")
        if sufficient_statistics is not None and any(
            not math.isclose(value, round(value), rel_tol=0.0, abs_tol=1.0e-6)
            for value in (
                count,
                target,
                predicted,
                true_positive,
                false_positive,
                false_negative,
                true_negative,
                history_full,
                history_local,
            )
        ):
            raise ValueError("probe-derived Router counts must be integral")

        expected_precision = _safe_ratio(true_positive, predicted)
        expected_recall = _safe_ratio(true_positive, target)
        expected_f1 = _safe_ratio(
            2.0 * expected_precision * expected_recall,
            expected_precision + expected_recall,
        )
        expected_values = (
            ("raw_quality_gap_mean", raw_gap_sum / count),
            ("target_full_rate", target / count),
            ("predicted_full_rate", predicted / count),
            ("accuracy", (true_positive + true_negative) / count),
            ("precision", expected_precision),
            ("recall", expected_recall),
            ("f1", expected_f1),
            ("history_full_selected_rate", history_full / count),
            ("history_local_selected_rate", history_local / count),
        )
        for name, expected in expected_values:
            if not math.isclose(
                values[name], expected, rel_tol=1.0e-4, abs_tol=3.0e-5
            ):
                raise ValueError(
                    f"inconsistent Router {name}: reported={values[name]}, "
                    f"reconstructed={expected}"
                )

        threshold = values["history_selection_threshold"]
        if self.history_threshold is None:
            self.history_threshold = threshold
        elif not math.isclose(
            threshold,
            self.history_threshold,
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        ):
            raise ValueError(
                "history selection threshold changed within one optimizer step: "
                f"{self.history_threshold} -> {threshold}"
            )

        self.microbatches += 1
        self.objective_loss_sum += values["objective_loss"] * (count if self.token_weighted_objective else 1.0)
        self.eligible += count
        self.raw_gap_sum += raw_gap_sum
        self.history_full += history_full
        self.history_local += history_local
        self.target_full += target
        self.predicted_full += predicted
        self.true_positive += true_positive
        self.false_positive += false_positive
        self.false_negative += false_negative
        self.true_negative += true_negative

    def finalize(self) -> dict[str, float]:
        if self.microbatches == 0:
            if self.strict:
                raise ValueError(
                    "contracted Router optimizer step has no aggregatable microbatch"
                )
            return {}
        denominator = self.eligible
        precision = _safe_ratio(
            self.true_positive, self.true_positive + self.false_positive
        )
        recall = _safe_ratio(
            self.true_positive, self.true_positive + self.false_negative
        )
        values = {
            "objective_loss": self.objective_loss_sum / (denominator if self.token_weighted_objective else self.microbatches),
            "raw_quality_gap_mean": self.raw_gap_sum / denominator,
            "target_full_rate": self.target_full / denominator,
            "predicted_full_rate": self.predicted_full / denominator,
            "accuracy": (self.true_positive + self.true_negative) / denominator,
            "precision": precision,
            "recall": recall,
            "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
            "history_full_selected_rate": self.history_full / denominator,
            "history_local_selected_rate": self.history_local / denominator,
            "history_selection_threshold": float(self.history_threshold),
        }
        result = {
            ROUTER_CORE_OUTPUT_NAMES[name]: value for name, value in values.items()
        }
        counts = {
            "eligible": self.eligible,
            "target_full": self.target_full,
            "predicted_full": self.predicted_full,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
        }
        result.update(
            {ROUTER_COUNT_OUTPUT_NAMES[name]: value for name, value in counts.items()}
        )
        result["comparison/schema/router_core_coverage"] = 1.0
        result["comparison/schema/router_core_missing_count"] = 0.0
        result["comparison/schema/exact_step_aggregation"] = 1.0
        return result


def _canonical_router_metrics_from_record(
    record: Mapping[str, Any],
) -> dict[str, float] | None:
    if not REQUIRED_ROUTER_STEP_METRICS.issubset(record):
        return None
    result = {
        name: _finite_float(record[name], name)
        for name in REQUIRED_ROUTER_STEP_METRICS
    }
    result["comparison/schema/router_core_coverage"] = 1.0
    result["comparison/schema/router_core_missing_count"] = 0.0
    result["comparison/schema/exact_step_aggregation"] = 1.0
    return result


def build_wandb_comparison_metrics(
    record: Mapping[str, Any],
    *,
    trajectory_steps: int | None = None,
    strict_router: bool = False,
    router_step_metrics: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Build one canonical local/W&B payload from an optimizer-step record."""

    if "global_step" not in record:
        raise ValueError("comparison metrics require global_step")
    step = int(_finite_float(record["global_step"], "global_step"))
    if step <= 0:
        raise ValueError(f"global_step must be positive, observed {step}")

    result: dict[str, float] = {"comparison/progress/global_step": float(step)}
    if trajectory_steps is not None:
        if trajectory_steps <= 0:
            raise ValueError("trajectory_steps must be positive")
        result["comparison/progress/trajectory_fraction"] = step / float(
            trajectory_steps
        )

    for source, target in BASE_OUTPUT_NAMES.items():
        if source in record:
            result[target] = _finite_float(record[source], source)
    for source, target in _SYSTEM_METRICS.items():
        if source in record:
            result[target] = _finite_float(record[source], source)

    result["comparison/schema/contract_enabled"] = float(strict_router)

    exact_metrics = None
    if router_step_metrics is not None:
        missing = REQUIRED_ROUTER_STEP_METRICS.difference(router_step_metrics)
        if missing:
            raise ValueError(
                "exact Router step metrics are incomplete: "
                + ", ".join(sorted(missing))
            )
        exact_metrics = {
            name: _finite_float(router_step_metrics[name], name)
            for name in REQUIRED_ROUTER_STEP_METRICS
        }
        exact_metrics.update(
            {
                "comparison/schema/router_core_coverage": 1.0,
                "comparison/schema/router_core_missing_count": 0.0,
                "comparison/schema/exact_step_aggregation": 1.0,
            }
        )
    else:
        exact_metrics = _canonical_router_metrics_from_record(record)
    if exact_metrics is not None:
        result.update(exact_metrics)
        return result

    has_router_metrics = any(key.startswith("router/") for key in record)
    if strict_router and has_router_metrics:
        raise ValueError(
            "contracted Router run requires exact optimizer-step aggregation"
        )
    if has_router_metrics:
        resolved = resolve_router_core_values(record, required=False)
        result.update(
            {
                ROUTER_CORE_OUTPUT_NAMES[name]: value
                for name, value in resolved.items()
            }
        )
        missing_count = len(ROUTER_CORE_ALIASES) - len(resolved)
        result["comparison/schema/router_core_coverage"] = len(resolved) / float(
            len(ROUTER_CORE_ALIASES)
        )
        result["comparison/schema/router_core_missing_count"] = float(missing_count)
        result["comparison/schema/exact_step_aggregation"] = 0.0
    elif strict_router:
        raise ValueError("contracted Router run has no Router metrics")
    return result


def build_ruler16k_comparison_summary(
    analysis_summary: Mapping[str, Any],
    *,
    protocol_snapshot: Mapping[str, Any] | None = None,
    budget_calibrated: bool | None = None,
) -> dict[str, float]:
    """Derive the frozen RULER payload from either supported analysis format.

    Qwen3 analyses expose ``quality`` and ``natural_route_contract`` directly.
    Qwen3.5 analyses expose a ``cases`` table and bind a separate protocol
    snapshot.  The latter must carry the runtime threshold, while calibration
    provenance is explicit rather than guessed from the numeric threshold.
    """

    if analysis_summary.get("benchmark") != "ruler16k":
        raise ValueError("RULER comparison summary requires benchmark=ruler16k")
    quality = analysis_summary.get("quality")
    route_contract = analysis_summary.get("natural_route_contract")
    if isinstance(quality, Mapping) and isinstance(route_contract, Mapping):
        router = quality.get("selected_cache_router_real")
        original = quality.get("original_full")
        if not isinstance(router, Mapping) or not isinstance(original, Mapping):
            raise ValueError("RULER analysis lacks Router or Original Full quality")
        score = _finite_float(router.get("score"), "RULER Router score")
        original_score = _finite_float(
            original.get("score"), "RULER Original Full score"
        )
        full_rate = _finite_float(router.get("full_rate"), "RULER Full rate")
        gap = _finite_float(router.get("gap_to_full"), "RULER gap to Full")
        recovery = _finite_float(
            router.get("full_local_gap_recovery"), "RULER Full-Local recovery"
        )
        threshold = _finite_float(
            route_contract.get("transformed_score"),
            "RULER runtime score threshold",
        )
        calibrated = route_contract.get("budget_calibration")
        if not isinstance(calibrated, bool):
            raise ValueError("RULER budget_calibration must be boolean")
        if budget_calibrated is not None and budget_calibrated is not calibrated:
            raise ValueError(
                "explicit RULER calibration provenance disagrees with analysis"
            )
    else:
        rows = analysis_summary.get("cases")
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                "RULER analysis lacks either quality/natural_route_contract or cases"
            )
        cases: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not str(row.get("case") or ""):
                raise ValueError("RULER cases contain a malformed row")
            name = str(row["case"])
            if name in cases:
                raise ValueError(f"RULER analysis contains duplicate case {name!r}")
            cases[name] = row
        router_names = sorted(name for name in cases if name.startswith("router_"))
        if len(router_names) != 1:
            raise ValueError(
                "RULER cases analysis must contain exactly one router_* case; "
                f"observed {router_names}"
            )
        original = cases.get("original_full")
        local = cases.get("always_local")
        router = cases[router_names[0]]
        if not isinstance(original, Mapping) or not isinstance(local, Mapping):
            raise ValueError("RULER cases analysis lacks fixed Full/Local controls")
        score = _finite_float(router.get("score"), "RULER Router score")
        original_score = _finite_float(
            original.get("score"), "RULER Original Full score"
        )
        local_score = _finite_float(local.get("score"), "RULER Always-Local score")
        full_rate = _finite_float(router.get("full_rate"), "RULER Full rate")
        gap = original_score - score
        full_local_gap = original_score - local_score
        if math.isclose(full_local_gap, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("RULER Full-Local score gap is zero")
        recovery = (score - local_score) / full_local_gap

        if not isinstance(protocol_snapshot, Mapping):
            raise ValueError(
                "RULER cases analysis requires its hash-bound protocol snapshot"
            )
        runtime = protocol_snapshot.get("runtime")
        if not isinstance(runtime, Mapping):
            raise ValueError("RULER protocol snapshot lacks runtime semantics")
        if runtime.get("router_action_comparison") != ">":
            raise ValueError("RULER runtime score comparison must be strict >")
        threshold = _finite_float(
            runtime.get("router_action_threshold"),
            "RULER runtime score threshold",
        )
        if budget_calibrated is None:
            recorded_calibration = runtime.get("budget_calibration")
            if not isinstance(recorded_calibration, bool):
                raise ValueError(
                    "RULER cases analysis requires explicit budget_calibrated "
                    "provenance; it cannot be inferred from the threshold value"
                )
            calibrated = recorded_calibration
        else:
            calibrated = budget_calibrated

    if not 0.0 <= score <= 100.0 or not 0.0 <= original_score <= 100.0:
        raise ValueError("RULER scores must be benchmark points in [0, 100]")
    if not 0.0 <= full_rate <= 1.0:
        raise ValueError("RULER Full rate must be a fraction in [0, 1]")
    if not math.isclose(
        gap, original_score - score, rel_tol=1.0e-8, abs_tol=1.0e-6
    ):
        raise ValueError(
            "RULER gap_to_full disagrees with Original Full minus Router score"
        )
    values = {
        "score": score,
        "full_rate": full_rate,
        "gap_to_full": gap,
        "full_local_gap_recovery": recovery,
        "runtime_score_threshold": threshold,
        "budget_calibrated": float(calibrated),
    }
    return {
        RULER16K_SUMMARY_OUTPUT_NAMES[name]: value
        for name, value in values.items()
    }


def build_proxy_comparison_summary(
    *, ppl_ratio: Any, full_rate: Any
) -> dict[str, float]:
    """Build the two deployment-proxy summary metrics with fixed units."""

    ratio = _finite_float(ppl_ratio, "proxy PPL ratio")
    rate = _finite_float(full_rate, "proxy Full rate")
    if ratio <= 0.0:
        raise ValueError("proxy PPL ratio must be positive")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("proxy Full rate must be a fraction in [0, 1]")
    return {
        "comparison/eval/proxy/ppl_ratio": ratio,
        "comparison/eval/proxy/full_rate": rate,
    }
