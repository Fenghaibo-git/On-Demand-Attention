"""Exact CE normalization for frozen-backbone indexer training with DDP."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


def ce_indexer_contract(config):
    """Semantic identity for the existing DCP extra-state resume checks."""
    fields = (
        "model_type", "hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps",
        "rope_theta", "rope_scaling", "tie_word_embeddings", "ce_indexer_schema_version",
        "ce_indexer_swa_window", "ce_indexer_sink_size", "ce_indexer_loss_normalization",
        "ce_indexer_initialization",
    )
    return {
        "format": "qwen3-ce-indexer-v1",
        "parameters": {name: getattr(config, name, None) for name in fields},
        "features": "swa_final_pre_norm_hidden",
        "values": "last_swa_layer_actual_v",
        "readout": "reuse_last_layer_qk_norm_rope_o_norm_ffn_then_final_norm_lm_head",
        "labels": "unshifted_next_token_valid_adjacent_positions_ignore_minus100",
        "packing": False,
        "trainable": ["indexer.q_proj.weight", "indexer.k_proj.weight"],
    }


def validate_ce_indexer_training(config, train_args):
    if not hasattr(config, "ce_indexer_schema_version"):
        return False
    if getattr(train_args, "data_parallel_mode", None) != "ddp":
        raise ValueError("CE indexer currently supports DDP; reused FFN/V access is not FSDP-qualified")
    for name in ("ulysses_parallel_size", "context_parallel_size", "tensor_parallel_size", "pipeline_parallel_size", "expert_parallel_size"):
        if getattr(train_args, name, 1) != 1:
            raise ValueError("CE indexer sequence/tensor/pipeline parallelism is not implemented")
    if getattr(train_args, "rmpad", False) or getattr(train_args, "rmpad_with_pos_ids", False):
        raise ValueError("CE indexer requires padded, unpacked sequences")
    if getattr(train_args, "enable_gradient_checkpointing", False):
        raise ValueError("Disable trunk gradient checkpointing; the frozen trunk uses no_grad and CE has its own checkpointing")
    if config.ce_indexer_loss_normalization != "global_step_token_mean_v1":
        raise ValueError("CE indexer requires exact optimizer-step token normalization")
    return True


@dataclass
class CEIndexerStepMean:
    local_counts: tuple
    global_counts: tuple
    world_size: int

    @classmethod
    def prepare(cls, micro_batches, device):
        from ..models.transformers.qwen3_ce_indexer.modeling_qwen3_ce_indexer import next_token_mask

        counts, error = (), None
        try:
            counts = tuple(int(next_token_mask(batch).sum().item()) for batch in micro_batches)
            if not counts:
                raise ValueError("Optimizer step has no microbatches")
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
            error = str(exc)
        distributed = dist.is_available() and dist.is_initialized()
        shape = torch.tensor([len(counts), -len(counts), int(error is not None)], device=device)
        if distributed:
            dist.all_reduce(shape, op=dist.ReduceOp.MAX)
        if shape[2].item() or shape[0].item() != -shape[1].item():
            raise ValueError(error or "Ranks disagree on CE microbatch layout or validity")
        global_counts = torch.tensor(counts, device=device, dtype=torch.long)
        if distributed:
            dist.all_reduce(global_counts)
        if global_counts.sum().item() == 0:
            raise ValueError("Optimizer step has zero global CE targets; no update is allowed")
        return cls(counts, tuple(global_counts.tolist()), dist.get_world_size() if distributed else 1)

    def scale_loss(self, output, micro_index):
        mismatch = torch.tensor(
            int(output.num_loss_tokens.item() != self.local_counts[micro_index]), device=output.loss_sum.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        if mismatch.item():
            raise ValueError("Precomputed and actual CE target counts disagree")
        # DDP averages rank gradients. Accumulation needs no second division.
        return output.loss_sum * (self.world_size / sum(self.global_counts))

    def weight(self, micro_index):
        return self.global_counts[micro_index] / sum(self.global_counts)
