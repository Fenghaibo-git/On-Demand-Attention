# Modified for the portable Router v4.2 release; see NOTICE.
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import wandb
import numpy as np
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_energon_dataset,
    build_interleave_dataset,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.data.data_transform import process_pretrain_example, process_sft_example, process_tokenized_example
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.tensor_state import tensor_state_sha256
from veomni.utils.ce_indexer_training import CEIndexerStepMean, ce_indexer_contract, validate_ce_indexer_training
from veomni.utils.router_step_token_mean import (
    EXECUTION_FIELD,
    PROTOCOL_FIELD,
    REDUCTION_FIELD,
    RouterStepTokenMean,
    step_token_mean_execution_contract,
    validate_step_token_mean_config,
)
from veomni.utils.wandb_metrics import (
    RouterCoreMetricAccumulator,
    build_router_metric_semantics_config,
    build_wandb_comparison_metrics,
    load_router_wandb_metric_schema_identity,
    validate_router_metric_semantics_config,
    validate_wandb_comparison_contract,
)


logger = helper.create_logger(__name__)


def _validate_resume_metrics(path, checkpoint_step):
    """Require the local metrics ledger to end exactly at the resumed DCP."""

    if path is None or not os.path.isfile(path):
        raise FileNotFoundError("A resumed local run requires train_metrics.jsonl.")
    steps = []
    with open(path, encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record["global_step"])
            if step <= 0:
                raise ValueError(f"Invalid global_step at {path}:{line_number}: {step}")
            steps.append(step)
    expected = list(range(1, int(checkpoint_step) + 1))
    if steps != expected:
        raise ValueError(
            "Resumed train_metrics.jsonl must contain exactly contiguous steps "
            f"1..{checkpoint_step}; observed {len(steps)} rows ending at "
            f"{steps[-1] if steps else None}."
        )


def _capture_extra_state(
    global_step,
    lr_scheduler,
    train_dataloader,
    environ_meter,
    training_contract=None,
):
    state = {
        "global_step": global_step,
        "lr_scheduler": lr_scheduler.state_dict(),
        "train_dataloader": train_dataloader.state_dict(),
        "environ_meter": environ_meter.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }
    if training_contract is not None:
        state["training_contract"] = training_contract
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(extra_state):
    torch.set_rng_state(extra_state["torch_rng_state"])
    if "python_rng_state" in extra_state:
        random.setstate(extra_state["python_rng_state"])
    if "numpy_rng_state" in extra_state:
        np.random.set_state(extra_state["numpy_rng_state"])
    if "torch_cuda_rng_state_all" in extra_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(extra_state["torch_cuda_rng_state_all"])


def _checkpoint_due(global_step, save_steps, stop_after_global_step):
    periodic = bool(save_steps and global_step % save_steps == 0)
    segment_boundary = stop_after_global_step == global_step
    return periodic or segment_boundary, segment_boundary


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _model_training_contract(model_config):
    """Small immutable identity persisted in DCP extra state."""

    contract = {
        "architectures": list(getattr(model_config, "architectures", []) or []),
        "router_schema_version": getattr(model_config, "router_schema_version", None),
        "router_schema_sha256": getattr(model_config, "router_schema_sha256", None),
        "router_training_history_mode": getattr(
            model_config, "router_training_history_mode", None
        ),
        "router_runtime_history_mode": getattr(
            model_config, "router_runtime_history_mode", None
        ),
        "router_initial_tensor_sha256": getattr(
            model_config, "router_initial_tensor_sha256", None
        ),
    }
    if hasattr(model_config, "ce_indexer_schema_version"):
        contract["ce_indexer"] = ce_indexer_contract(model_config)
    if hasattr(model_config, "router_regularization_schema_version"):
        contract.update(
            {
                "router_regularization_schema_version": getattr(
                    model_config, "router_regularization_schema_version"
                ),
                "router_regularization_sha256": getattr(
                    model_config, "router_regularization_sha256", None
                ),
                "router_head_dropout": getattr(
                    model_config, "router_head_dropout", None
                ),
                "router_weight_decay": getattr(
                    model_config, "router_weight_decay", None
                ),
                "router_weight_decay_exclusions": list(
                    getattr(model_config, "router_weight_decay_exclusions", [])
                ),
            }
        )
    if hasattr(model_config, "router_decision_regret_objective_schema_version"):
        contract.update(
            {
                "router_objective": getattr(
                    model_config, "router_objective", None
                ),
                "router_decision_regret_objective_schema_version": getattr(
                    model_config,
                    "router_decision_regret_objective_schema_version",
                ),
                "router_decision_regret_objective": getattr(
                    model_config, "router_decision_regret_objective", None
                ),
                "router_decision_regret_objective_sha256": getattr(
                    model_config,
                    "router_decision_regret_objective_sha256",
                    None,
                ),
                "router_decision_regret_compute_cost_lambda": getattr(
                    model_config,
                    "router_decision_regret_compute_cost_lambda",
                    None,
                ),
                "router_decision_regret_weight_definition": getattr(
                    model_config,
                    "router_decision_regret_weight_definition",
                    None,
                ),
                "router_decision_regret_normalization": getattr(
                    model_config,
                    "router_decision_regret_normalization",
                    None,
                ),
                "router_decision_regret_output_space": getattr(
                    model_config,
                    "router_decision_regret_output_space",
                    None,
                ),
            }
        )
    if hasattr(
        model_config, "router_robust_decision_margin_objective_schema_version"
    ):
        contract.update(
            {
                "router_objective": getattr(
                    model_config, "router_objective", None
                ),
                "router_robust_decision_margin_objective_schema_version": getattr(
                    model_config,
                    "router_robust_decision_margin_objective_schema_version",
                ),
                "router_robust_decision_margin_objective": getattr(
                    model_config,
                    "router_robust_decision_margin_objective",
                    None,
                ),
                "router_robust_decision_margin_objective_sha256": getattr(
                    model_config,
                    "router_robust_decision_margin_objective_sha256",
                    None,
                ),
                "router_robust_decision_margin_lambda": getattr(
                    model_config,
                    "router_robust_decision_margin_lambda",
                    None,
                ),
                "router_robust_decision_margin_loss_weighting": getattr(
                    model_config,
                    "router_robust_decision_margin_loss_weighting",
                    None,
                ),
                "router_robust_decision_margin_output_space": getattr(
                    model_config,
                    "router_robust_decision_margin_output_space",
                    None,
                ),
            }
        )
    if hasattr(model_config, "router_focal_objective_schema_version"):
        contract.update(
            {
                "router_objective": getattr(
                    model_config, "router_objective", None
                ),
                "router_focal_objective_schema_version": getattr(
                    model_config, "router_focal_objective_schema_version"
                ),
                "router_focal_objective": getattr(
                    model_config, "router_focal_objective", None
                ),
                "router_focal_objective_sha256": getattr(
                    model_config, "router_focal_objective_sha256", None
                ),
                "router_focal_decision_nll_threshold": getattr(
                    model_config, "router_focal_decision_nll_threshold", None
                ),
                "router_focal_full_alpha": getattr(
                    model_config, "router_focal_full_alpha", None
                ),
                "router_focal_gamma": getattr(
                    model_config, "router_focal_gamma", None
                ),
                "router_focal_ignore_abs_gap_le": getattr(
                    model_config, "router_focal_ignore_abs_gap_le", None
                ),
                "router_focal_normalization": getattr(
                    model_config, "router_focal_normalization", None
                ),
                "router_focal_output_space": getattr(
                    model_config, "router_focal_output_space", None
                ),
                "router_focal_history_source": getattr(
                    model_config, "router_focal_history_source", None
                ),
                "router_focal_boundary_policy": getattr(
                    model_config, "router_focal_boundary_policy", None
                ),
            }
        )
    if getattr(model_config, REDUCTION_FIELD, None) is not None:
        contract[REDUCTION_FIELD] = getattr(model_config, REDUCTION_FIELD)
        contract[PROTOCOL_FIELD] = getattr(model_config, PROTOCOL_FIELD, None)
        contract[EXECUTION_FIELD] = getattr(model_config, EXECUTION_FIELD, None)
    return contract


def _validate_model_optimizer_contract(model_config, train_args):
    """Bind an opt-in modeling regularization contract to trainer arguments."""

    if not hasattr(model_config, "router_regularization_schema_version"):
        return
    expected_weight_decay = float(model_config.router_weight_decay)
    actual_weight_decay = float(train_args.weight_decay)
    if actual_weight_decay != expected_weight_decay:
        raise ValueError(
            "Regularized Router weight decay mismatch: model contract expects "
            f"{expected_weight_decay}, trainer configured {actual_weight_decay}."
        )
    if str(train_args.optimizer) != "adamw":
        raise ValueError("Regularized Router v1 requires the AdamW optimizer.")
    expected_exclusions = list(model_config.router_weight_decay_exclusions)
    actual_exclusions = list(train_args.no_decay_params)
    if actual_exclusions != expected_exclusions:
        raise ValueError(
            "Regularized Router no-decay mismatch: model contract expects "
            f"{expected_exclusions}, trainer configured {actual_exclusions}."
        )


def _collect_router_scalar_gradient_metrics(model):
    """Collect the tiny stabilized-Head gradients without affecting training."""

    unwrapped_model = _unwrap_model(model)
    metrics = {}
    fusion_names = ("a", "b", "c", "f")
    for name, parameter in unwrapped_model.named_parameters():
        if parameter.grad is None:
            continue
        if name == "router_head.fusion_scales":
            gradient = parameter.grad.detach().float().reshape(-1)
            if gradient.numel() != len(fusion_names):
                raise ValueError("Router fusion_scales gradient must contain four values.")
            for label, value in zip(fusion_names, gradient):
                metrics[f"router/fusion_scale_{label}_grad"] = float(value.item())
        elif name.startswith("router_head.blocks.") and name.endswith(".layer_scale"):
            block_index = name.split(".")[2]
            metrics[f"router/layer_scale_{block_index}_grad"] = float(
                parameter.grad.detach().float().item()
            )
    return metrics


def _collect_router_probe_sufficient_statistics(probe_tensors, semantics, *, allow_empty_local=False):
    """Compute exact global Router counts from the tensors used by the loss."""

    valid_token_mask = probe_tensors.get("valid_token_mask")
    if not isinstance(valid_token_mask, torch.Tensor):
        raise ValueError(
            "contracted Router forward must expose a Tensor valid_token_mask"
        )
    mask = valid_token_mask.detach().reshape(-1).bool()

    def select_tensor(names, label):
        observed = [
            (name, probe_tensors[name].detach().reshape(-1))
            for name in names
            if isinstance(probe_tensors.get(name), torch.Tensor)
        ]
        if not observed:
            raise ValueError(
                f"contracted Router probes lack a {label} tensor; expected one of {names}"
            )
        name, selected = observed[0]
        if selected.numel() != mask.numel():
            raise ValueError(
                f"Router probe {name} has {selected.numel()} values, expected {mask.numel()}"
            )
        for alias, value in observed[1:]:
            if value.numel() != mask.numel() or not torch.allclose(
                selected[mask].float(),
                value[mask].float(),
                rtol=1.0e-6,
                atol=1.0e-7,
            ):
                raise ValueError(
                    f"Router probe aliases disagree for {label}: {name}, {alias}"
                )
        return selected

    raw_gap = select_tensor(
        ("router_raw_signed_nll_gap", "router_raw_target"),
        "raw quality gap",
    ).float()
    score = select_tensor(
        (
            "router_robust_decision_score",
            "router_focal_logit",
            "router_decision_evidence_logit",
            "router_decision_logit",
            "router_transformed_score",
        ),
        "reference score",
    ).float()
    history_local = select_tensor(
        (
            "history_local_selected_mask",
            "router_history_oracle_binary_target_local",
        ),
        "history Local-selection",
    ).bool()
    if bool(history_local[~mask].any()):
        raise ValueError("history Local-selection mask contains ineligible tokens")
    if not allow_empty_local and not bool(mask.any()):
        raise ValueError("contracted Router microbatch has zero eligible tokens")
    if not bool(torch.isfinite(raw_gap[mask]).all()):
        raise ValueError("Router raw quality gap contains a non-finite eligible value")
    if not bool(torch.isfinite(score[mask]).all()):
        raise ValueError("Router reference score contains a non-finite eligible value")

    boundary = float(semantics["decision_boundary_nats"])
    if semantics["decision_boundary_operator"] == "gt":
        target_full = raw_gap.gt(boundary) & mask
    elif semantics["decision_boundary_operator"] == "ge":
        target_full = raw_gap.ge(boundary) & mask
    else:
        raise ValueError("unsupported Router decision boundary operator")
    if semantics["reference_score_operator"] != "gt":
        raise ValueError("unsupported Router reference score operator")
    predicted_full = score.gt(float(semantics["reference_score_threshold"])) & mask
    history_full = mask & (~history_local)

    statistics = torch.stack(
        (
            mask.double().sum(),
            raw_gap[mask].double().sum(),
            history_full.double().sum(),
            history_local.double().sum(),
            target_full.double().sum(),
            predicted_full.double().sum(),
            (target_full & predicted_full).double().sum(),
            ((~target_full) & predicted_full & mask).double().sum(),
            (target_full & (~predicted_full) & mask).double().sum(),
            ((~target_full) & (~predicted_full) & mask).double().sum(),
        )
    )
    dist.all_reduce(
        statistics,
        op=dist.ReduceOp.SUM,
        group=get_parallel_state().fsdp_group,
    )
    names = (
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
    )
    return dict(zip(names, statistics.tolist()))


def _validate_resume_training_contract(saved_contract, current_contract):
    protected_router_architectures = {
        "Qwen3_5ThreeStreamNllSelectedRobustDecisionMarginRouterForCausalLM",
        "Qwen3LargeRobustMarginRouterForCausalLM",
        "Qwen3_5LargeRobustMarginRouterForCausalLM",
        "Gemma4UnifiedThreeStreamRobustMarginRouterForCausalLM",
    }
    is_contract_protected_router = bool(protected_router_architectures.intersection(
        current_contract.get("architectures", [])))
    if saved_contract is None:
        if is_contract_protected_router:
            raise ValueError(
                "History-conditioned Router resume checkpoint has no training "
                "contract; cross-schema resume cannot be ruled out."
            )
        return
    if saved_contract != current_contract:
        raise ValueError(
            "Resume training contract mismatch; a checkpoint cannot cross model, "
            "schema, Router initialization, or F/S training-history modes. "
            f"saved={saved_contract}, current={current_contract}"
        )


def _load_initial_trainable_weights(model, path, expected_sha256, output_dir, global_rank):
    """Load an exact audited state into every and only trainable parameter."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or int(payload.get("format_version", -1)) != 1:
        raise ValueError("Initial trainable weights must use format_version=1.")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Initial trainable payload has no non-empty state_dict.")

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    state_names = set(state_dict)
    if state_names != trainable_names:
        raise ValueError(
            "Initial trainable state must cover exactly every requires-grad parameter; "
            f"missing={sorted(trainable_names - state_names)}, "
            f"unexpected={sorted(state_names - trainable_names)}."
        )
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("Every initial trainable state value must be a Tensor.")

    actual_sha256 = tensor_state_sha256(state_dict)
    embedded_sha256 = payload.get("tensor_state_sha256")
    if embedded_sha256 != actual_sha256:
        raise ValueError(
            "Initial trainable payload tensor hash is invalid: "
            f"embedded={embedded_sha256}, actual={actual_sha256}."
        )
    if expected_sha256 != actual_sha256:
        raise ValueError(
            "Initial trainable tensor hash differs from the configured contract: "
            f"expected={expected_sha256}, actual={actual_sha256}."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected initial trainable keys: {incompatible.unexpected_keys}."
        )
    model.config.router_initial_tensor_sha256 = actual_sha256

    if global_rank == 0:
        metadata = {
            "format_version": 1,
            "source_path": os.path.abspath(path),
            "tensor_state_sha256": actual_sha256,
            "state_key_count": len(state_dict),
            "tensor_count": int(sum(value.numel() for value in state_dict.values())),
            "payload_metadata": payload.get("metadata", {}),
        }
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, "initial_trainable_weights.json")
        temporary_path = metadata_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        os.replace(temporary_path, metadata_path)
    return actual_sha256


def _sample_model_probe(probe_tensors, max_tokens, seed):
    cpu_tensors = {
        name: value.detach().reshape(-1).cpu()
        for name, value in probe_tensors.items()
        if isinstance(value, torch.Tensor)
    }
    if "valid_token_mask" not in cpu_tensors:
        raise ValueError("Model probe tensors must include `valid_token_mask`.")

    valid_indices = torch.nonzero(cpu_tensors["valid_token_mask"].bool(), as_tuple=False).flatten()
    if max_tokens > 0 and valid_indices.numel() > max_tokens:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selection = torch.randperm(valid_indices.numel(), generator=generator)[:max_tokens]
        valid_indices = valid_indices[selection].sort().values
    return {name: value[valid_indices] for name, value in cpu_tensors.items()}


def _write_model_probe(output_dir, global_step, global_rank, probe_batches, max_tokens, seed):
    if not probe_batches:
        return
    names = tuple(probe_batches[0])
    if any(tuple(batch) != names for batch in probe_batches):
        raise ValueError("Every model probe batch must contain the same tensor names in the same order.")
    combined = {name: torch.cat([batch[name].detach().reshape(-1).cpu() for batch in probe_batches]) for name in names}
    sampling_seed = seed + global_step * 1_000_003 + global_rank * 10_007
    sampled = _sample_model_probe(combined, max_tokens, sampling_seed)
    payload = {
        "format_version": 1,
        "global_step": global_step,
        "global_rank": global_rank,
        "sampling_seed": sampling_seed,
        "tensors": sampled,
    }
    rank_dir = os.path.join(output_dir, "router_probes", f"rank_{global_rank:05d}")
    os.makedirs(rank_dir, exist_ok=True)
    output_path = os.path.join(rank_dir, f"step_{global_step:08d}.pt")
    temporary_path = output_path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)


def _write_wandb_run_metadata(run, output_dir):
    """Persist a secret-free pointer to the online run beside local metrics."""
    payload = {
        "format_version": 1,
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "name": getattr(run, "name", None),
        "group": getattr(run, "group", None),
        "id": getattr(run, "id", None),
        "url": getattr(run, "url", None),
        "local_wandb_dir": os.path.join(output_dir, "wandb"),
    }
    output_path = os.path.join(output_dir, "wandb_run.json")
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")
    os.replace(temporary_path, output_path)


def _write_or_validate_metric_contract(
    output_dir,
    comparison_contract,
    metric_semantics,
    metric_schema_identity,
    *,
    resuming=False,
):
    """Persist one immutable local copy of the W&B comparison semantics."""

    payload = {
        "format_version": 1,
        "comparison_contract": comparison_contract,
        "metric_semantics": metric_semantics,
        "metric_schema_identity": metric_schema_identity,
    }
    output_path = os.path.join(output_dir, "metric_contract.json")
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as contract_file:
            observed = json.load(contract_file)
        if observed != payload:
            raise ValueError(
                "Local metric contract differs from the resumed run: "
                f"{output_path}"
            )
        return
    if resuming and comparison_contract.get("enabled"):
        raise ValueError(
            "Cannot enable a contracted metric schema in the middle of a legacy run; "
            "keep the resumed run legacy and create a shadow comparison view instead."
        )
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as contract_file:
        json.dump(payload, contract_file, indent=2, sort_keys=True)
        contract_file.write("\n")
    os.replace(temporary_path, output_path)


def _export_trainable_weights(model, model_config, output_dir, global_step):
    unwrapped_model = _unwrap_model(model)
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in unwrapped_model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable_state:
        logger.warning_rank0("No trainable parameters found; skip trainable-weight export.")
        return

    export_dir = os.path.join(output_dir, "trainable_exports", f"global_step_{global_step}")
    os.makedirs(export_dir, exist_ok=True)
    output_path = os.path.join(export_dir, "trainable_state.pt")
    temporary_path = output_path + ".tmp"
    torch.save(
        {
            "format_version": 1,
            "global_step": global_step,
            "model_class": unwrapped_model.__class__.__name__,
            "model_config": model_config.to_dict(),
            "tensor_state_sha256": tensor_state_sha256(trainable_state),
            "state_dict": trainable_state,
        },
        temporary_path,
    )
    os.replace(temporary_path, output_path)
    return output_path


def _save_checkpoint_with_trainable_export(
    checkpointer,
    save_checkpoint_path,
    state,
    *,
    global_step,
    model,
    model_config,
    output_dir,
    export_trainable_weights,
    global_rank,
):
    """Persist the small trainable state before entering distributed checkpoint I/O.

    Router training freezes the base model, so this export is the inference
    checkpoint that matters.  Writing it first prevents a later distributed
    checkpoint/NCCL failure from discarding an otherwise completed epoch.
    """
    if global_rank == 0 and export_trainable_weights:
        export_path = _export_trainable_weights(model, model_config, output_dir, global_step)
        if export_path is not None:
            logger.info_rank0(f"Trainable weights saved at {export_path} before distributed checkpoint.")

    # Non-zero ranks must not enter the distributed save while rank zero is
    # still copying trainable parameters to CPU.
    dist.barrier()
    checkpointer.save(save_checkpoint_path, state, global_steps=global_step)
    dist.barrier()


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def main():
    dist.init_process_group(backend=get_nccl_backend())
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    train_metrics_path = None
    if args.train.global_rank == 0 and not args.train.output_dir.startswith("hdfs://"):
        train_metrics_path = os.path.join(args.train.output_dir, "train_metrics.jsonl")
        metrics_file_mode = "a" if args.train.load_checkpoint_path and os.path.exists(train_metrics_path) else "w"
        with open(train_metrics_path, metrics_file_mode, encoding="utf-8"):
            pass

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "tokenized":
        transform = partial(process_tokenized_example, max_seq_len=args.data.max_seq_len)
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.enable_multisource:
            logger.info_rank0("Start building interleave dataset")
            train_dataset = build_interleave_dataset(
                args.data.train_path, args.data.datasets_type, transform=transform, seed=args.train.seed
            )
        elif args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "energon":
            logger.info_rank0("Start building Megatron-Energon native dataset")
            train_dataset = build_energon_dataset(
                args.data.train_path,
                transform=transform,
                max_samples_per_sequence=args.data.max_samples_per_sequence
                if hasattr(args.data, "max_samples_per_sequence")
                else None,
                virtual_epoch_length=args.data.virtual_epoch_length
                if hasattr(args.data, "virtual_epoch_length")
                else None,
                shuffle_buffer_size=args.data.shuffle_buffer_size
                if hasattr(args.data, "shuffle_buffer_size")
                else None,
                num_workers=args.data.num_workers,
            )
        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        config_kwargs=args.model.config_kwargs,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    if args.train.initial_trainable_weights_path:
        initial_sha256 = _load_initial_trainable_weights(
            model,
            args.train.initial_trainable_weights_path,
            args.train.initial_trainable_weights_sha256,
            args.train.output_dir,
            args.train.global_rank,
        )
        logger.info_rank0(
            "Loaded audited initial trainable state with tensor SHA-256 "
            f"{initial_sha256}."
        )
    model_config = model.config
    use_ce_indexer_step_mean = validate_ce_indexer_training(model_config, args.train)
    _validate_model_optimizer_contract(model_config, args.train)
    use_step_token_mean = validate_step_token_mean_config(model_config, args.train)
    if use_step_token_mean:
        setattr(model_config, EXECUTION_FIELD, step_token_mean_execution_contract(args.train))
    training_contract = _model_training_contract(model_config)
    wandb_comparison_contract = validate_wandb_comparison_contract(
        vars(args.train), required=False
    )
    router_metric_semantics = build_router_metric_semantics_config(model_config)
    validate_router_metric_semantics_config(
        router_metric_semantics,
        required=wandb_comparison_contract["enabled"],
    )
    router_metric_schema_identity = (
        load_router_wandb_metric_schema_identity()
        if wandb_comparison_contract["enabled"]
        else {
            "schema_version": wandb_comparison_contract["metric_schema_version"],
            "sha256": None,
        }
    )
    if args.train.global_rank == 0 and not args.train.output_dir.startswith("hdfs://"):
        _write_or_validate_metric_contract(
            args.train.output_dir,
            wandb_comparison_contract,
            router_metric_semantics,
            router_metric_schema_identity,
            resuming=bool(args.train.load_checkpoint_path),
        )
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    # Transformers 5.x may normalize the instance-level no-split collection
    # to a set even when a custom model declares a list at class scope.  The
    # VeOmni configuration remains a list, so normalize both public contracts
    # before combining them rather than relying on a concrete container type.
    model_no_split_modules = list(getattr(model, "_no_split_modules", ()) or ())
    configured_basic_modules = list(args.model.basic_modules or ())
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model_no_split_modules + configured_basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        no_decay_modules=args.train.no_decay_modules,
        no_decay_params=args.train.no_decay_params,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_decay_start_ratio=args.train.lr_decay_start_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb_init_kwargs = {}
            wandb_run_id = os.getenv("WANDB_RUN_ID")
            wandb_resume = os.getenv("WANDB_RESUME")
            if wandb_resume and not wandb_run_id:
                raise ValueError("WANDB_RESUME requires a non-empty WANDB_RUN_ID.")
            if wandb_run_id:
                wandb_init_kwargs["id"] = wandb_run_id
                wandb_init_kwargs["resume"] = wandb_resume or "allow"
            wandb_config = {**vars(args.model), **vars(args.data), **vars(args.train)}
            wandb_config["wandb_comparison_contract"] = wandb_comparison_contract
            wandb_config["wandb_metric_semantics"] = router_metric_semantics
            wandb_config["wandb_metric_schema_identity"] = (
                router_metric_schema_identity
            )
            wandb_run = wandb.init(
                entity=os.getenv("WANDB_ENTITY") or args.train.wandb_entity,
                project=os.getenv("WANDB_PROJECT") or args.train.wandb_project,
                name=args.train.wandb_name,
                group=args.train.wandb_group,
                tags=args.train.wandb_tags or None,
                dir=args.train.output_dir,
                config=wandb_config,  # Keep legacy flattened fields and add a namespaced contract.
                **wandb_init_kwargs,
            )
            _write_wandb_run_metadata(wandb_run, args.train.output_dir)

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        # save model_assets before training
        model_assets = [
            model_config,
            chat_template if args.data.data_type == "conversation" else tokenizer,
        ]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        _validate_resume_training_contract(
            state["extra_state"].get("training_contract"), training_contract
        )
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        _restore_rng_state(state["extra_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        metrics_error = [None]
        if args.train.global_rank == 0:
            try:
                _validate_resume_metrics(train_metrics_path, global_step)
            except Exception as error:
                metrics_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(metrics_error, src=0)
        if metrics_error[0] is not None:
            raise RuntimeError(metrics_error[0])

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    total_global_steps = args.train.train_steps * args.train.num_train_epochs
    stop_after_global_step = args.train.stop_after_global_step
    if stop_after_global_step is not None:
        if stop_after_global_step > total_global_steps:
            raise ValueError(
                f"stop_after_global_step={stop_after_global_step} exceeds the configured "
                f"trajectory length {total_global_steps}."
            )
        if stop_after_global_step <= global_step:
            raise ValueError(
                f"stop_after_global_step={stop_after_global_step} must be greater than "
                f"the resumed global_step={global_step}."
            )

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    validate_runtime = getattr(_unwrap_model(model), "validate_distributed_runtime_state", None)
    if validate_runtime is not None:
        validate_runtime()
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    stop_training = False
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            model_metrics: Dict[str, float] = {}
            model_probe_batches = []
            router_core_accumulator = (
                RouterCoreMetricAccumulator(strict=True, token_weighted_objective=use_step_token_mean)
                if wandb_comparison_contract["enabled"]
                else None
            )
            synchronize()
            start_time = time.time()
            ce_indexer_step_mean = (
                CEIndexerStepMean.prepare(micro_batches, device=get_device_type())
                if use_ce_indexer_step_mean else None
            )
            step_token_mean = (
                RouterStepTokenMean.prepare(
                    micro_batches, _unwrap_model(model)._get_router_loss_mask,
                    device=get_device_type(),
                )
                if use_step_token_mean else None
            )
            for micro_index, micro_batch in enumerate(micro_batches):
                environ_meter.add(micro_batch)
                if step_token_mean is not None and step_token_mean.global_counts[micro_index] == 0:
                    # All ranks agree: this microbatch contributes no objective.
                    continue
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    model_outputs = model(**micro_batch, use_cache=False)
                    if ce_indexer_step_mean is not None:
                        loss = ce_indexer_step_mean.scale_loss(model_outputs, micro_index)
                    elif step_token_mean is None:
                        loss: "torch.Tensor" = model_outputs.loss.mean() / len(micro_batches)
                    else:
                        loss = step_token_mean.scale_loss(
                            model_outputs.loss,
                            model_outputs.training_probe_tensors.get("valid_token_mask"),
                            micro_index,
                        )

                micro_model_metrics = {}
                for name, value in (getattr(model_outputs, "training_metrics", None) or {}).items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().float().item()
                    micro_model_metrics[name] = float(value)
                    micro_weight = (
                        step_token_mean.weight(micro_index)
                        if step_token_mean is not None else 1.0 / len(micro_batches)
                    )
                    model_metrics[name] = model_metrics.get(name, 0.0) + float(value) * micro_weight

                probe_tensors = getattr(model_outputs, "training_probe_tensors", None) or {}
                if router_core_accumulator is not None:
                    sufficient_statistics = (
                        _collect_router_probe_sufficient_statistics(
                            probe_tensors,
                            router_metric_semantics,
                            allow_empty_local=use_step_token_mean,
                        )
                    )
                    router_core_accumulator.update(
                        micro_model_metrics,
                        sufficient_statistics=sufficient_statistics,
                    )

                if args.train.model_probe_steps and global_step % args.train.model_probe_steps == 0:
                    if probe_tensors:
                        model_probe_batches.append(probe_tensors)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            # Prefer model-provided clip_grad_norm_ (now both FSDP1 and FSDP2 registers custom grad norm clipping)
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                logger.info_rank0(
                    "Can NOT find regitsered clip_grad_norm_ method in the model, using PyTorch default implementation.."
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            model_metrics.update(_collect_router_scalar_gradient_metrics(model))
            if step_token_mean is not None:
                model_metrics["router/optimization/step_global_eligible_tokens"] = float(step_token_mean.total_tokens)
                model_metrics["router/optimization/step_token_mean_enabled"] = 1.0
                model_metrics["router/optimization/applied_learning_rate"] = max(group["lr"] for group in optimizer.param_groups)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            if ce_indexer_step_mean is not None:
                model_metrics.update({
                    "ce_indexer/ce": float(total_loss),
                    "ce_indexer/eligible_tokens": sum(ce_indexer_step_mean.global_counts),
                    "ce_indexer/exact_step_aggregation": 1,
                })
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)
            router_step_metrics = (
                router_core_accumulator.finalize()
                if router_core_accumulator is not None
                else None
            )

            if model_probe_batches:
                _write_model_probe(
                    args.train.output_dir,
                    global_step,
                    args.train.global_rank,
                    model_probe_batches,
                    args.train.model_probe_max_tokens,
                    args.train.model_probe_seed,
                )

            postfix = f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}"
            if model_metrics:
                postfix += ", " + ", ".join(
                    f"{name.replace('router/', '', 1)}: {value:.3f}" for name, value in model_metrics.items()
                )
            data_loader_tqdm.set_postfix_str(postfix)
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                metric_record = {
                    "global_step": global_step,
                    "step_time_seconds": float(delta_time),
                    "loss": float(total_loss),
                    "grad_norm": float(grad_norm),
                    "learning_rate": float(lr),
                    **train_metrics,
                    **model_metrics,
                }
                comparison_metrics = build_wandb_comparison_metrics(
                    metric_record,
                    trajectory_steps=total_global_steps,
                    strict_router=wandb_comparison_contract["enabled"],
                    router_step_metrics=router_step_metrics,
                )
                metric_record.update(comparison_metrics)
                if train_metrics_path is not None:
                    with open(train_metrics_path, "a", encoding="utf-8") as metrics_file:
                        metrics_file.write(json.dumps(metric_record, sort_keys=True) + "\n")

                if args.train.use_wandb:
                    wandb_metrics = dict(train_metrics)
                    wandb_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb_metrics.update({f"training/{name}": value for name, value in model_metrics.items()})
                    wandb_metrics.update(comparison_metrics)
                    wandb.log(wandb_metrics, step=global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()

            checkpoint_due, stop_at_this_step = _checkpoint_due(
                global_step,
                args.train.save_steps,
                stop_after_global_step,
            )
            if checkpoint_due:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": _capture_extra_state(
                        global_step,
                        lr_scheduler,
                        train_dataloader,
                        environ_meter,
                        training_contract,
                    ),
                }
                _save_checkpoint_with_trainable_export(
                    Checkpointer,
                    args.train.save_checkpoint_path,
                    state,
                    global_step=global_step,
                    model=model,
                    model_config=model_config,
                    output_dir=args.train.output_dir,
                    export_trainable_weights=args.train.export_trainable_weights,
                    global_rank=args.train.global_rank,
                )
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
            if stop_at_this_step:
                if save_checkpoint_path is None:
                    raise AssertionError("Segment stop was reached without a completed checkpoint save.")
                logger.info_rank0(
                    f"Reached stop_after_global_step={global_step} after checkpoint completion; "
                    "ending this training segment."
                )
                stop_training = True
                break

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if not stop_training and args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": _capture_extra_state(
                    global_step,
                    lr_scheduler,
                    train_dataloader,
                    environ_meter,
                    training_contract,
                ),
            }
            _save_checkpoint_with_trainable_export(
                Checkpointer,
                args.train.save_checkpoint_path,
                state,
                global_step=global_step,
                model=model,
                model_config=model_config,
                output_dir=args.train.output_dir,
                export_trainable_weights=args.train.export_trainable_weights,
                global_rank=args.train.global_rank,
            )
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
        if stop_training:
            break

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    if args.train.global_rank == 0 and args.train.use_wandb:
        wandb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
