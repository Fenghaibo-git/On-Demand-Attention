"""Actual Huber/Adam accumulation equivalence, including CPU DDP ranks."""

from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from tasks.train_torch import (
    _collect_router_probe_sufficient_statistics,
    _model_training_contract,
    _validate_resume_training_contract,
)
from veomni.models.transformers.qwen3_5_three_stream_nll_selected_robust_decision_margin_router.modeling_qwen3_5_three_stream_nll_selected_robust_decision_margin_router import (
    Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM,
)
from veomni.models.transformers.qwen3_router.modeling_qwen3_router import build_router_eligible_mask
from veomni.optim.lr_scheduler import build_lr_scheduler
from veomni.utils.router_step_token_mean import (
    EXECUTION_FIELD, PROTOCOL_FIELD, REDUCTION_FIELD, STEP_TOKEN_MEAN,
    RouterStepTokenMean, eligible_mask_from_batch, step_token_mean_enabled,
    step_token_mean_execution_contract, validate_step_token_mean_config,
)
from veomni.utils.wandb_metrics import RouterCoreMetricAccumulator, build_router_metric_semantics_config


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/train/qwen35_2b_mb1.yaml"


def objective_config(enabled=True):
    config = SimpleNamespace(
        architectures=["Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM"],
        router_objective="robust_decision_margin_huber",
        router_robust_decision_margin_lambda=0.001,
        router_robust_decision_margin_loss_weighting="eligible_token_equal",
        router_one_step_target="signed_nll", router_one_step_transform="signed_log1p",
        router_one_step_harm_threshold=0.0, router_one_step_positive_weight=1.0,
        router_one_step_huber_delta=1.0, router_one_step_nll_scale=1.0,
        router_one_step_sign_aux_weight=0.0, router_one_step_sign_aux_temperature=0.1,
        router_one_step_sign_near_zero_threshold=0.0001,
        router_history_nll_threshold=0.0, router_loss_weight=1.0,
    )
    if enabled:
        setattr(config, REDUCTION_FIELD, STEP_TOKEN_MEAN)
        setattr(config, PROTOCOL_FIELD, hashlib.sha256(CONFIG.read_bytes()).hexdigest())
    return config


class LossHead(torch.nn.Module):
    """Use the production objective without allocating the frozen backbone."""
    def __init__(self, enabled=True):
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(5, 12), torch.nn.SiLU(), torch.nn.Linear(12, 1))
        objective = object.__new__(Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM)
        torch.nn.Module.__init__(objective)
        objective.config = objective_config(enabled)
        objective._last_robust_margin_metrics = {}
        self.objective = objective

    def forward(self, features, gap, mask):
        score = self.head(features).squeeze(-1)
        target = self.objective._transform_target(gap)
        loss, _ = self.objective._compute_one_step_loss(score, gap, target, mask)
        return loss


def fixture_tensors(world=1):
    generator = torch.Generator().manual_seed(20260908)
    features = torch.randn(64, 7, 5, generator=generator)
    gap = torch.randn(64, 7, generator=generator) * 2.0
    mask = torch.rand(64, 7, generator=generator) > 0.35
    mask[0] = False  # An empty local rank/microbatch must still join collectives.
    for rank in range(world):
        mask[(rank + 1) * (64 // world) - 1] = False  # One globally empty microbatch for micro=1.
    return features, gap, mask


def oracle_backward(model, features, gap, mask):
    score = model.head(features).squeeze(-1)
    target = (gap - 0.001).sign() * torch.log1p((gap - 0.001).abs())
    loss = torch.nn.functional.huber_loss(score[mask], target[mask], delta=1.0)
    loss.backward()
    return loss.detach()


def assert_parameters_close(left, right):
    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)


def assert_optimizer_close(left, right):
    for a, b in zip(left.state.values(), right.state.values()):
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("micro", [1, 2, 8, 64])
@pytest.mark.parametrize("clip", [0.05, 1.0])
def test_repartition_matches_whole_token_loss_gradient_clip_and_adam(micro, clip):
    torch.manual_seed(17)
    reference = LossHead()
    candidate = copy.deepcopy(reference)
    features, gap, mask = fixture_tensors()
    expected_loss = oracle_backward(reference, features, gap, mask)
    chunks = [slice(start, start + micro) for start in range(0, 64, micro)]
    plan = RouterStepTokenMean.from_counts([int(mask[c].sum()) for c in chunks], device="cpu")
    observed_loss = 0.0
    for index, chunk in enumerate(chunks):
        if plan.global_counts[index] == 0:
            continue
        loss = candidate(features[chunk], gap[chunk], mask[chunk])
        normalized = plan.scale_loss(loss, mask[chunk], index)
        observed_loss += normalized.detach()
        normalized.backward()
    torch.testing.assert_close(observed_loss, expected_loss, rtol=2e-6, atol=2e-7)
    for a, b in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)
    ga = torch.nn.utils.clip_grad_norm_(reference.parameters(), clip)
    gb = torch.nn.utils.clip_grad_norm_(candidate.parameters(), clip)
    torch.testing.assert_close(ga, gb, rtol=2e-5, atol=2e-7)
    oa = torch.optim.AdamW(reference.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
    ob = torch.optim.AdamW(candidate.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
    oa.step()
    ob.step()
    assert_parameters_close(reference, candidate)
    assert_optimizer_close(oa, ob)


def _ddp_worker(rank, world, store_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(19)
        initial = LossHead().state_dict()
        features, gap, mask = fixture_tensors(world)
        reference = LossHead()
        reference.load_state_dict(initial)
        expected_loss = oracle_backward(reference, features, gap, mask)
        ref_grads = [p.grad.detach().clone() for p in reference.parameters()]
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.05)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
        ref_optimizer.step()
        local_n = 64 // world
        partition = slice(rank * local_n, (rank + 1) * local_n)
        local_features, local_gap, local_mask = features[partition], gap[partition], mask[partition]
        for micro in [local_n, 2, 1]:
            net = LossHead()
            net.load_state_dict(initial)
            ddp = torch.nn.parallel.DistributedDataParallel(net)
            chunks = [slice(start, start + micro) for start in range(0, local_n, micro)]
            plan = RouterStepTokenMean.from_counts([int(local_mask[c].sum()) for c in chunks], device="cpu")
            assert plan.total_tokens == int(mask.sum())
            total = torch.zeros(())
            for index, chunk in enumerate(chunks):
                if plan.global_counts[index] == 0:
                    continue
                loss = ddp(local_features[chunk], local_gap[chunk], local_mask[chunk])
                normalized = plan.scale_loss(loss, local_mask[chunk], index)
                normalized.backward()
                total += normalized.detach()
            dist.all_reduce(total)
            total /= world
            torch.testing.assert_close(total, expected_loss, rtol=2e-6, atol=2e-7)
            for parameter, expected in zip(net.parameters(), ref_grads):
                torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-7)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.05)
            optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
            optimizer.step()
            assert_parameters_close(reference, net)
            assert_optimizer_close(ref_optimizer, optimizer)
        # A mismatch on just one rank must reject on every rank, not deadlock.
        plan = RouterStepTokenMean.from_counts([1], device="cpu")
        with pytest.raises(ValueError, match="counts disagree"):
            plan.scale_loss(torch.zeros((), requires_grad=True), torch.ones(2 if rank == 0 else 1, dtype=torch.bool), 0)
        with pytest.raises(ValueError, match="counts disagree"):
            plan.scale_loss(torch.zeros((), requires_grad=True), None if rank == 0 else torch.ones(1), 0)
        with pytest.raises(ValueError):
            RouterStepTokenMean.from_counts([] if rank == 0 else [1], device="cpu")
        with pytest.raises(ValueError, match="number of microbatches"):
            RouterStepTokenMean.from_counts([1, 1] if rank == 0 else [1], device="cpu")
        with pytest.raises(ValueError, match="zero global"):
            RouterStepTokenMean.from_counts([0], device="cpu")
        with pytest.raises(ValueError):
            RouterStepTokenMean.prepare(
                [{"labels": torch.ones(1 if rank == 0 else (1, 2))}],
                lambda valid, labels, positions: valid,
                device="cpu",
            )

        # Exercise the production public-metric collector on a rank with no
        # eligible tokens; every rank must still enter its global reduction.
        import tasks.train_torch as trainer

        original_parallel_state = trainer.get_parallel_state
        trainer.get_parallel_state = lambda: SimpleNamespace(fsdp_group=dist.group.WORLD)
        try:
            local_valid = torch.tensor([rank != 0])
            stats = _collect_router_probe_sufficient_statistics(
                {
                    "valid_token_mask": local_valid,
                    "router_raw_signed_nll_gap": torch.tensor([0.5]),
                    "router_robust_decision_score": torch.tensor([0.2]),
                    "history_local_selected_mask": torch.tensor([False]),
                },
                {
                    "decision_boundary_nats": 0.001,
                    "decision_boundary_operator": "gt",
                    "reference_score_operator": "gt",
                    "reference_score_threshold": 0.0,
                },
                allow_empty_local=True,
            )
            assert stats["eligible"] == world - 1
            assert stats["true_positive"] == world - 1
        finally:
            trainer.get_parallel_state = original_parallel_state
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 8])
def test_actual_ddp_global64_equivalence(tmp_path, world):
    mp.spawn(_ddp_worker, args=(world, str(tmp_path / "gloo")), nprocs=world, join=True)


def test_mask_precount_matches_production_predicate_with_padding_and_resets():
    labels = torch.tensor([[1, 2, 3, -100, 5, 6, 7], [1, 2, 3, 4, -100, -100, -100]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2], [0, 1, 2, 3, 0, 0, 0]])
    builder = lambda v, l, p: build_router_eligible_mask(v, l, p, 1, 1)
    batch = {"labels": labels, "attention_mask": valid, "position_ids": positions}
    expected = builder(valid, labels, positions)
    torch.testing.assert_close(eligible_mask_from_batch(batch, builder), expected)
    plan = RouterStepTokenMean.prepare([batch], builder, device="cpu")
    assert plan.total_tokens == int(expected.sum())


def test_zero_step_bad_count_and_legacy_optout():
    with pytest.raises(ValueError, match="zero global"):
        RouterStepTokenMean.from_counts([0, 0], device="cpu")
    with pytest.raises(ValueError, match="nonnegative"):
        RouterStepTokenMean.from_counts([-1], device="cpu")
    plan = RouterStepTokenMean.from_counts([2], device="cpu")
    with pytest.raises(ValueError, match="counts disagree"):
        plan.scale_loss(torch.zeros((), requires_grad=True), torch.ones(1, dtype=torch.bool), 0)
    assert not step_token_mean_enabled(objective_config(False))
    bad = objective_config()
    setattr(bad, REDUCTION_FIELD, "typo")
    with pytest.raises(ValueError, match="Unsupported"):
        step_token_mean_enabled(bad)




def test_resume_contract_rejects_old_reduction_and_changed_schedule():
    train = SimpleNamespace(**yaml.safe_load(CONFIG.read_text())["train"])
    train.data_parallel_size = 8
    train.train_steps = 3072
    config = objective_config()
    setattr(config, EXECUTION_FIELD, step_token_mean_execution_contract(train))
    current = _model_training_contract(config)
    _validate_resume_training_contract(copy.deepcopy(current), current)
    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_resume_training_contract(_model_training_contract(objective_config(False)), current)
    train.lr_decay_style = "constant"
    setattr(config, EXECUTION_FIELD, step_token_mean_execution_contract(train))
    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_resume_training_contract(current, _model_training_contract(config))


def scheduler_from_config(optimizer):
    train = yaml.safe_load(CONFIG.read_text())["train"]
    return build_lr_scheduler(optimizer, train_steps=3072, **{k: train[k] for k in (
        "lr", "lr_min", "lr_start", "lr_warmup_ratio", "lr_decay_style", "lr_decay_start_ratio", "lr_decay_ratio"
    )})


def test_actual_scheduler_endpoints_and_resume():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = scheduler_from_config(optimizer)
    expected = {102: 3e-4, 512: 2.8750199763549826e-4, 1024: 2.407276454369119e-4, 2048: 1.0174745907863902e-4, 3072: 3e-5}
    previous = None
    for step in range(1, 3073):
        applied = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        observed = optimizer.param_groups[0]["lr"]
        if step in expected:
            assert observed == pytest.approx(expected[step], rel=1e-12, abs=1e-15)
        if step > 102:
            assert observed < previous
        previous = observed
        if step == 1024:
            opt_state, sched_state = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
        if step == 3072:
            assert abs(applied - 3e-5) <= 1e-10
    restored = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=3e-4)
    restored_scheduler = scheduler_from_config(restored)
    restored.load_state_dict(opt_state)
    restored_scheduler.load_state_dict(sched_state)
    for _ in range(1024, 3072):
        restored.step()
        restored_scheduler.step()
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_public_objective_metric_uses_matching_token_reduction():
    exact = RouterCoreMetricAccumulator(strict=True, token_weighted_objective=True)
    legacy = RouterCoreMetricAccumulator(strict=True)
    for count, loss in [(1, 0.5), (3, 2.0)]:
        record = {
            "router/objective/total_loss": loss,
            "router/labels/raw_signed_nll_mean": 0.1,
            "router/decision/target_full_rate": 1.0,
            "router/decision/predicted_full_rate": 1.0,
            "router/decision/accuracy": 1.0,
            "router/decision/precision": 1.0,
            "router/decision/recall": 1.0,
            "router/decision/f1": 1.0,
            "router/history/full_selected_rate": 1.0,
            "router/history/local_selected_rate": 0.0,
            "router/history/selection_threshold": 0.0,
        }
        stats = dict(eligible=count, raw_gap_sum=count * 0.1, history_full=count, history_local=0,
                     target_full=count, predicted_full=count, true_positive=count,
                     false_positive=0, false_negative=0, true_negative=0)
        exact.update(record, sufficient_statistics=stats)
        legacy.update(record, sufficient_statistics=stats)
    field = "comparison/optimization/router_objective_loss"
    assert exact.finalize()[field] == pytest.approx(1.625)
    assert legacy.finalize()[field] == pytest.approx(1.25)
    config = SimpleNamespace(**yaml.safe_load(CONFIG.read_text())["model"]["config_kwargs"])
    assert build_router_metric_semantics_config(config)["objective_loss_aggregation"] == "global_eligible_token_mean_across_dp_and_microbatches"
    delattr(config, REDUCTION_FIELD)
    assert build_router_metric_semantics_config(config)["objective_loss_aggregation"] == "arithmetic_mean_of_global_microbatch_objective_means_matching_gradient_accumulation"
