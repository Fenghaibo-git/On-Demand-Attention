import math

import pytest

from veomni.utils.wandb_metrics import (
    ROUTER_WANDB_METRIC_SCHEMA_VERSION,
    RouterCoreMetricAccumulator,
    build_proxy_comparison_summary,
    build_ruler16k_comparison_summary,
    build_router_metric_semantics_config,
    build_wandb_comparison_metrics,
    validate_wandb_comparison_contract,
    validate_router_metric_semantics_config,
)


def v4_record() -> dict:
    return {
        "global_step": 32,
        "step_time_seconds": 2.0,
        "loss": 0.2,
        "grad_norm": 0.7,
        "learning_rate": 3.0e-4,
        "consume_tokens(B)": 0.01,
        "router/objective/total_loss": 0.2,
        "router/objective/huber_loss": 0.2,
        "router/regression/signed_nll_mean": 0.03,
        "router/signed_nll_mean": 0.03,
        "router/sign0/target_full_rate": 0.3,
        "router/sign_target_rate": 0.3,
        "router/sign0/predicted_full_rate": 0.2,
        "router/sign_predicted_positive_rate": 0.2,
        "router/sign0/accuracy": 0.8,
        "router/sign_accuracy": 0.8,
        "router/sign0/precision": 0.6,
        "router/sign_precision": 0.6,
        "router/sign0/recall": 0.4,
        "router/sign_recall": 0.4,
        "router/sign0/f1": 0.48,
        "router/sign_f1": 0.48,
        "router/history/full_selected_rate": 0.55,
        "router/history/local_selected_rate": 0.45,
        "router/history/selection_threshold": 0.001,
    }


def decision_record(
    *,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    objective_loss: float,
    raw_gap_mean: float,
    history_full_rate: float,
) -> tuple[dict, int]:
    count = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        {
            "router/objective/total_loss": objective_loss,
            "router/labels/raw_signed_nll_mean": raw_gap_mean,
            "router/decision/target_full_rate": (tp + fn) / count,
            "router/decision/predicted_full_rate": (tp + fp) / count,
            "router/decision/accuracy": (tp + tn) / count,
            "router/decision/precision": precision,
            "router/decision/recall": recall,
            "router/decision/f1": f1,
            "router/history/full_selected_rate": history_full_rate,
            "router/history/local_selected_rate": 1.0 - history_full_rate,
            "router/history/selection_threshold": 0.0,
        },
        count,
    )


def sufficient_statistics(
    *,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    raw_gap_mean: float,
    history_full_rate: float,
) -> dict:
    count = tp + fp + fn + tn
    return {
        "eligible": count,
        "raw_gap_sum": raw_gap_mean * count,
        "history_full": history_full_rate * count,
        "history_local": (1.0 - history_full_rate) * count,
        "target_full": tp + fn,
        "predicted_full": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def test_legacy_aliases_map_but_are_marked_inexact():
    metrics = build_wandb_comparison_metrics(
        v4_record(), trajectory_steps=128, strict_router=False
    )
    assert metrics["comparison/schema/router_core_coverage"] == 1.0
    assert metrics["comparison/schema/exact_step_aggregation"] == 0.0
    assert metrics["comparison/schema/contract_enabled"] == 0.0
    assert metrics["comparison/router/reference/target_full_rate"] == 0.3
    assert metrics["comparison/router/reference/predicted_full_rate"] == 0.2
    assert metrics["comparison/router/label/raw_quality_gap_mean_nats"] == 0.03
    assert metrics["comparison/progress/trajectory_fraction"] == 0.25
    assert metrics["comparison/progress/cumulative_input_tokens_billions"] == 0.01


def test_new_decision_namespace_maps_to_same_stable_keys():
    record = v4_record()
    for key in list(record):
        if key.startswith("router/sign") or key == "router/regression/signed_nll_mean":
            del record[key]
    record["router/labels/raw_signed_nll_mean"] = 0.03
    for name, value in {
        "target_full_rate": 0.3,
        "predicted_full_rate": 0.2,
        "accuracy": 0.8,
        "precision": 0.6,
        "recall": 0.4,
        "f1": 0.48,
    }.items():
        record[f"router/decision/{name}"] = value
    metrics = build_wandb_comparison_metrics(record, strict_router=False)
    assert metrics["comparison/schema/router_core_missing_count"] == 0.0
    assert metrics["comparison/router/reference/f1"] == 0.48


def test_exact_accumulator_uses_eligible_token_sufficient_statistics():
    first, _ = decision_record(
        tp=10,
        fp=10,
        fn=20,
        tn=60,
        objective_loss=0.2,
        raw_gap_mean=0.01,
        history_full_rate=0.6,
    )
    second, _ = decision_record(
        tp=120,
        fp=30,
        fn=30,
        tn=120,
        objective_loss=0.4,
        raw_gap_mean=0.05,
        history_full_rate=0.2,
    )
    accumulator = RouterCoreMetricAccumulator(strict=True)
    accumulator.update(
        first,
        sufficient_statistics=sufficient_statistics(
            tp=10,
            fp=10,
            fn=20,
            tn=60,
            raw_gap_mean=0.01,
            history_full_rate=0.6,
        ),
    )
    accumulator.update(
        second,
        sufficient_statistics=sufficient_statistics(
            tp=120,
            fp=30,
            fn=30,
            tn=120,
            raw_gap_mean=0.05,
            history_full_rate=0.2,
        ),
    )
    exact = accumulator.finalize()

    assert exact["comparison/schema/exact_step_aggregation"] == 1.0
    assert exact["comparison/router/count/eligible_tokens"] == 400
    assert math.isclose(
        exact["comparison/router/label/raw_quality_gap_mean_nats"], 0.04
    )
    assert math.isclose(
        exact["comparison/router/reference/target_full_rate"], 180 / 400
    )
    assert math.isclose(
        exact["comparison/router/reference/predicted_full_rate"], 170 / 400
    )
    assert math.isclose(
        exact["comparison/router/reference/accuracy"], 310 / 400
    )
    assert math.isclose(
        exact["comparison/router/reference/precision"], 130 / 170
    )
    assert math.isclose(
        exact["comparison/router/reference/recall"], 130 / 180
    )
    assert math.isclose(
        exact["comparison/router/history/input_full_rate"], 120 / 400
    )
    # Objective loss follows equal microbatch gradient accumulation; token
    # statistics use global eligible-token weighting.
    assert math.isclose(
        exact["comparison/optimization/router_objective_loss"], 0.3
    )

    metrics = build_wandb_comparison_metrics(
        {"global_step": 1, "loss": 0.3},
        trajectory_steps=10,
        strict_router=True,
        router_step_metrics=exact,
    )
    assert metrics["comparison/schema/contract_enabled"] == 1.0
    assert metrics["comparison/schema/exact_step_aggregation"] == 1.0
    assert metrics["comparison/router/count/true_positive_tokens"] == 130


def test_contracted_legacy_microbatch_average_is_rejected():
    with pytest.raises(ValueError, match="requires exact optimizer-step aggregation"):
        build_wandb_comparison_metrics(v4_record(), strict_router=True)


def test_disagreeing_aliases_are_rejected():
    record = v4_record()
    record["router/sign_accuracy"] = 0.7
    with pytest.raises(ValueError, match="aliases disagree"):
        build_wandb_comparison_metrics(record)


def test_strict_accumulator_rejects_missing_core_metric():
    record, count = decision_record(
        tp=10,
        fp=10,
        fn=20,
        tn=60,
        objective_loss=0.2,
        raw_gap_mean=0.01,
        history_full_rate=0.6,
    )
    del record["router/decision/f1"]
    with pytest.raises(ValueError, match="lacks comparison core metrics: f1"):
        RouterCoreMetricAccumulator(strict=True).update(
            record, eligible_token_count=count
        )


def test_semantics_config_preserves_boundary_and_score_contract():
    greater_equal = build_router_metric_semantics_config(
        {
            "router_objective": "standard_focal",
            "router_focal_objective_schema_version": 1,
            "router_focal_decision_nll_threshold": 0.001,
            "router_focal_reference_logit_threshold": 0.0,
            "router_focal_output_space": "full_action_focal_logit",
            "router_focal_boundary_policy": "full_when_greater_than_or_equal",
            "router_history_nll_threshold": 0.001,
        }
    )
    assert greater_equal["decision_boundary_operator"] == "ge"
    assert greater_equal["boundary_tie_action"] == "full"
    assert greater_equal["reference_score_operator"] == "gt"

    strict = build_router_metric_semantics_config(
        {
            "router_objective": "robust_decision_margin_huber",
            "router_robust_decision_margin_objective_schema_version": 1,
            "router_robust_decision_margin_lambda": 0.001,
            "router_robust_decision_margin_runtime_score_threshold": 0.0,
            "router_robust_decision_margin_output_space": "signed_log1p_margin",
            "router_history_nll_threshold": 0.0,
        }
    )
    assert strict["decision_boundary_operator"] == "gt"
    assert strict["boundary_tie_action"] == "local"
    strict["objective_sha256"] = "c" * 64
    assert validate_router_metric_semantics_config(strict, required=True) == strict

    strict["objective_sha256"] = None
    with pytest.raises(ValueError, match="require an objective SHA-256"):
        validate_router_metric_semantics_config(strict, required=True)


def test_semantics_config_handles_regret_and_margin_objective_families():
    regret = build_router_metric_semantics_config(
        {
            "router_regret_objective_schema_version": 2,
            "router_regret_objective": "class_balanced_weighted_bce",
            "router_regret_objective_sha256": "d" * 64,
            "router_regret_decision_nll_threshold": 0.001,
            "router_regret_runtime_logit_threshold": 0.0,
            "router_regret_output_space": "decision_logit",
            "router_history_nll_threshold": 0.001,
        }
    )
    assert regret["objective_name"] == "class_balanced_weighted_bce"
    assert regret["decision_boundary_operator"] == "ge"
    assert regret["decision_boundary_nats"] == 0.001

    margin = build_router_metric_semantics_config(
        {
            "router_margin_objective_schema_version": 1,
            "router_margin_objective": "huber_on_signed_log1p_margin",
            "router_margin_objective_sha256": "e" * 64,
            "router_compute_cost_lambda": 0.01,
            "router_margin_runtime_score_threshold": 0.0,
            "router_margin_output_space": "signed_log1p_decision_margin",
            "router_history_nll_threshold": 0.01,
        }
    )
    assert margin["decision_boundary_operator"] == "gt"
    assert margin["score_space"] == "signed_log1p_decision_margin"
    assert margin["history_selection_space"] == "raw_quality_gap_nats"


def test_comparison_contract_is_all_or_nothing():
    contract = validate_wandb_comparison_contract(
        {
            "wandb_metric_schema_version": ROUTER_WANDB_METRIC_SCHEMA_VERSION,
            "wandb_comparison_family": "matched-router-study",
            "wandb_comparison_protocol_hash": "b" * 64,
            "wandb_group": "matched-router-study",
        },
        required=True,
    )
    assert contract["enabled"] is True
    with pytest.raises(ValueError, match="comparison contract fields"):
        validate_wandb_comparison_contract(
            {
                "wandb_metric_schema_version": ROUTER_WANDB_METRIC_SCHEMA_VERSION,
                "wandb_group": "matched-router-study",
            },
            required=True,
        )


def test_ruler_and_proxy_summary_names_have_fixed_units():
    ruler = build_ruler16k_comparison_summary(
        {
            "benchmark": "ruler16k",
            "natural_route_contract": {
                "budget_calibration": True,
                "transformed_score": 0.49,
            },
            "quality": {
                "original_full": {"score": 81.94},
                "selected_cache_router_real": {
                    "score": 78.16,
                    "full_rate": 0.5105,
                    "gap_to_full": 3.78,
                    "full_local_gap_recovery": 0.9397,
                },
            },
        }
    )
    assert ruler["comparison/eval/ruler16k/score_points"] == 78.16
    assert ruler["comparison/eval/ruler16k/full_rate"] == 0.5105
    assert ruler["comparison/eval/ruler16k/budget_calibrated"] == 1.0

    proxy = build_proxy_comparison_summary(ppl_ratio=1.004, full_rate=0.2934)
    assert proxy == {
        "comparison/eval/proxy/ppl_ratio": 1.004,
        "comparison/eval/proxy/full_rate": 0.2934,
    }


def test_qwen35_cases_ruler_summary_uses_protocol_and_explicit_provenance():
    ruler = build_ruler16k_comparison_summary(
        {
            "benchmark": "ruler16k",
            "cases": [
                {"case": "original_full", "score": 94.35, "full_rate": 1.0},
                {"case": "always_local", "score": 28.21, "full_rate": 0.0},
                {
                    "case": "router_step_1024",
                    "score": 92.08,
                    "full_rate": 0.2560162609206441,
                },
            ],
        },
        protocol_snapshot={
            "runtime": {
                "router_action_comparison": ">",
                "router_action_threshold": 0.0,
            }
        },
        budget_calibrated=False,
    )
    assert ruler["comparison/eval/ruler16k/score_points"] == 92.08
    assert math.isclose(
        ruler["comparison/eval/ruler16k/full_rate"], 0.2560162609206441
    )
    assert math.isclose(
        ruler["comparison/eval/ruler16k/score_gap_to_original_full_points"],
        2.27,
    )
    assert math.isclose(
        ruler["comparison/eval/ruler16k/full_local_gap_recovery"],
        (92.08 - 28.21) / (94.35 - 28.21),
    )
    assert ruler["comparison/eval/ruler16k/budget_calibrated"] == 0.0


def test_qwen35_cases_ruler_summary_never_guesses_calibration_provenance():
    with pytest.raises(ValueError, match="explicit budget_calibrated provenance"):
        build_ruler16k_comparison_summary(
            {
                "benchmark": "ruler16k",
                "cases": [
                    {"case": "original_full", "score": 94.35},
                    {"case": "always_local", "score": 28.21},
                    {
                        "case": "router_step_1024",
                        "score": 92.08,
                        "full_rate": 0.256,
                    },
                ],
            },
            protocol_snapshot={
                "runtime": {
                    "router_action_comparison": ">",
                    "router_action_threshold": 0.0,
                }
            },
        )


def test_ruler_summary_rejects_ambiguous_gap_sign():
    with pytest.raises(ValueError, match="gap_to_full disagrees"):
        build_ruler16k_comparison_summary(
            {
                "benchmark": "ruler16k",
                "natural_route_contract": {
                    "budget_calibration": False,
                    "transformed_score": 0.0,
                },
                "quality": {
                    "original_full": {"score": 80.0},
                    "selected_cache_router_real": {
                        "score": 75.0,
                        "full_rate": 0.2,
                        "gap_to_full": -5.0,
                        "full_local_gap_recovery": 0.9,
                    },
                },
            }
        )
