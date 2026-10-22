"""Opt-in optimizer-step token mean for qualified equal-token Huber models.

The model's legacy scalar is W * local_loss_sum / global_micro_count.
Multiplying it by global_micro_count / global_step_count before backward
therefore gives the desired gradient after DDP's averaging. There is no
additional division by the number of microbatches. Legacy runs do not opt in.
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.distributed as dist


STEP_TOKEN_MEAN = "global_eligible_token_mean_v1"
REDUCTION_FIELD = "router_optimizer_step_reduction"
PROTOCOL_FIELD = "router_optimization_protocol_sha256"
EXECUTION_FIELD = "router_optimizer_step_execution_contract"
QUALIFIED_STEP_TOKEN_MEAN_ARCHITECTURES = frozenset({
    "Qwen3RecallV42RouterForCausalLM",
    "Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM",
    "Qwen3LargeRobustMarginRouterForCausalLM",
    "Qwen3_5LargeRobustMarginRouterForCausalLM",
    "Gemma4UnifiedThreeStreamRobustMarginRouterForCausalLM",
})


def step_token_mean_enabled(config) -> bool:
    mode = getattr(config, REDUCTION_FIELD, None)
    if mode is None:
        return False
    if mode != STEP_TOKEN_MEAN:
        raise ValueError(f"Unsupported optimizer-step Router reduction: {mode!r}")
    return True


def validate_step_token_mean_config(config, train_args) -> bool:
    """Do not silently apply count weighting to a weighted/auxiliary objective."""
    if not step_token_mean_enabled(config):
        return False
    architectures = set(getattr(config, "architectures", []) or [])
    if len(architectures) != 1 or not architectures.issubset(QUALIFIED_STEP_TOKEN_MEAN_ARCHITECTURES):
        raise ValueError("Step token mean v1 requires a qualified history-zero robust Huber architecture")
    expected_strings = {
        "router_objective": "robust_decision_margin_huber",
        "router_one_step_target": "signed_nll",
        "router_one_step_transform": "signed_log1p",
        "router_robust_decision_margin_loss_weighting": "eligible_token_equal",
    }
    for name, expected in expected_strings.items():
        if getattr(config, name, None) != expected:
            raise ValueError(f"Step token mean requires {name}={expected!r}")
    expected_numbers = {
        "router_history_nll_threshold": 0.0,
        "router_one_step_positive_weight": 1.0,
        "router_one_step_sign_aux_weight": 0.0,
        "router_one_step_huber_delta": 1.0,
        "router_one_step_nll_scale": 1.0,
        "router_loss_weight": 1.0,
        "router_robust_decision_margin_lambda": 0.001,
    }
    for name, expected in expected_numbers.items():
        observed = float(getattr(config, name, float("nan")))
        if not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-10):
            raise ValueError(f"Step token mean requires {name}={expected}")
    if getattr(train_args, "data_parallel_mode", None) != "ddp" or int(train_args.ulysses_parallel_size) != 1:
        raise ValueError("Step token mean v1 requires DDP without sequence parallelism")
    protocol = getattr(config, PROTOCOL_FIELD, "")
    if not re.fullmatch(r"[0-9a-f]{64}", str(protocol)):
        raise ValueError("Step token mean requires an optimization protocol SHA-256")
    if protocol != getattr(train_args, "wandb_comparison_protocol_hash", None):
        raise ValueError("Step token mean and W&B optimization protocol hashes differ")
    return True


def step_token_mean_execution_contract(train_args) -> dict:
    """Bind the actual optimizer/scheduler arguments, not just a claimed hash."""
    fields = (
        "optimizer", "global_batch_size", "micro_batch_size", "data_parallel_size",
        "num_train_epochs", "lr", "lr_start", "lr_min", "lr_warmup_ratio",
        "lr_decay_style", "lr_decay_start_ratio", "lr_decay_ratio", "weight_decay",
        "max_grad_norm", "enable_mixed_precision", "seed",
    )
    result = {name: getattr(train_args, name) for name in fields}
    result["scheduled_optimizer_updates"] = int(train_args.train_steps) * int(train_args.num_train_epochs)
    return result


def eligible_mask_from_batch(batch: dict, mask_builder: Callable) -> torch.Tensor:
    """Use the model's exact eligibility predicate before its expensive forward."""
    labels = batch.get("labels")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("Step token mean requires padded [batch, sequence] labels")
    valid = batch.get("attention_mask")
    valid = torch.ones_like(labels, dtype=torch.bool) if valid is None else valid.bool()
    positions = batch.get("position_ids")
    if positions is None:
        positions = torch.arange(labels.shape[1], device=labels.device).expand_as(labels)
    elif positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if positions.shape[0] == 1:
        positions = positions.expand_as(labels)
    if valid.shape != labels.shape or positions.shape != labels.shape:
        raise ValueError("Step token mean requires aligned labels, attention mask and positions")
    return mask_builder(valid, labels, positions)


@dataclass(frozen=True)
class RouterStepTokenMean:
    local_counts: tuple[int, ...]
    global_counts: tuple[int, ...]

    @classmethod
    def prepare(cls, micro_batches: Sequence[dict], mask_builder: Callable, *, device) -> "RouterStepTokenMean":
        local_counts = ()
        error = None
        try:
            local_counts = tuple(
                int(eligible_mask_from_batch(batch, mask_builder).sum().item()) for batch in micro_batches
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        return cls.from_counts(local_counts, device=device, _precount_error=error)

    @classmethod
    def from_counts(cls, local_counts: Sequence[int], *, device, _precount_error=None) -> "RouterStepTokenMean":
        error = _precount_error
        try:
            local_counts = tuple(operator.index(count) for count in local_counts)
            if not local_counts or any(count < 0 for count in local_counts):
                error = error or "Step token counts must be nonnegative and nonempty"
        except (TypeError, ValueError) as exc:
            local_counts = ()
            error = error or f"Step token counts must be integers: {exc}"
        size = torch.tensor(
            [len(local_counts), -len(local_counts), int(error is not None)], dtype=torch.int64, device=device
        )
        if dist.is_available() and dist.is_initialized():
            # All ranks reject local validation errors together, before any
            # differently sized collective or expensive forward can begin.
            dist.all_reduce(size, op=dist.ReduceOp.MAX)
        if int(size[2]):
            raise ValueError(error or "Another rank rejected eligible-token precount inputs")
        if int(size[0]) != -int(size[1]):
            raise ValueError("Ranks disagree on the number of microbatches")
        counts = torch.tensor(local_counts, dtype=torch.int64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        global_counts = tuple(int(count) for count in counts.cpu().tolist())
        if sum(global_counts) <= 0:
            raise ValueError("Optimizer step has zero global eligible tokens; no update is allowed")
        return cls(local_counts, global_counts)

    @property
    def total_tokens(self) -> int:
        return sum(self.global_counts)

    def weight(self, micro_index: int) -> float:
        return self.global_counts[micro_index] / self.total_tokens

    def scale_loss(self, loss: torch.Tensor, actual_mask: torch.Tensor, micro_index: int) -> torch.Tensor:
        actual_count = int(actual_mask.bool().sum().item()) if isinstance(actual_mask, torch.Tensor) else -1
        mismatch = torch.tensor(
            int(actual_count != self.local_counts[micro_index]),
            dtype=torch.int64, device=loss.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        if int(mismatch):
            raise ValueError("Precomputed and actual eligible-token counts disagree")
        return loss.mean() * self.weight(micro_index)
