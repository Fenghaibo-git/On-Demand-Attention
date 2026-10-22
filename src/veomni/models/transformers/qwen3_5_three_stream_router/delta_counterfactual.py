# Modified for On Demand Attention; see NOTICE.
"""Exact one-token counterfactual helpers for Qwen3.5 Gated DeltaNet.

Qwen3.5 interleaves recurrent Gated DeltaNet layers with ordinary causal
attention.  A one-action Router label must let every candidate at position
``i`` read the immutable trunk state after position ``i - 1`` without letting
the candidate update become position ``i + 1`` history.

The helpers in this module implement that contract in two forms:

* :func:`recurrent_three_stream_reference` is an intentionally simple token
  loop used as a numerical oracle in tests.
* :func:`three_stream_from_trunk_scan` reuses a normal trunk DeltaRule scan.
  Candidate queries are shifted onto the previous trunk transition, so no
  per-token recurrent matrices need to be materialized.

The backbone is frozen, therefore these routines are forward-only and do not
need a custom backward implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
from torch.nn import functional as F

_fla_l2norm_fwd = None


DeltaRule = Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]]


def l2_normalize_qk(value: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Match the Qwen3.5/FLA QK normalization definition."""

    if value.is_cuda:
        global _fla_l2norm_fwd
        if _fla_l2norm_fwd is None:
            from fla.modules.l2norm import l2norm_fwd
            _fla_l2norm_fwd = l2norm_fwd
        # FLA's public DeltaRule wrapper applies its input-contiguity guard
        # before entering l2norm_fwd; call the primitive with the same layout.
        normalized, _ = _fla_l2norm_fwd(value.contiguous(), eps=eps)
        return normalized
    return value * torch.rsqrt(value.float().square().sum(dim=-1, keepdim=True) + eps).to(
        value.dtype
    )


def _validate_delta_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("DeltaRule query/key must be aligned [B,T,H,K] tensors.")
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("DeltaRule value must be [B,T,H,V].")
    if log_decay.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError("DeltaRule decay/beta must be aligned [B,T,H] tensors.")


def isolated_causal_conv1d(
    trunk_raw: torch.Tensor,
    candidate_raw: torch.Tensor,
    weight: torch.Tensor,
    position_ids: torch.LongTensor,
    valid_mask: torch.BoolTensor,
    bias: Optional[torch.Tensor] = None,
    single_segment_per_row: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute trunk and isolated-current candidate causal convolutions.

    Historical convolution inputs always come from ``trunk_raw``.  At every
    position only the current input is replaced by ``candidate_raw``.  Segment
    resets and padding are respected explicitly, which is required because a
    dense depth-wise convolution would otherwise cross a packed boundary.
    Returned tensors include the Qwen3.5 SiLU activation.
    """

    if trunk_raw.shape != candidate_raw.shape or trunk_raw.ndim != 3:
        raise ValueError("Causal-conv streams must be aligned [B,T,C] tensors.")
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight[:, 0, :]
    if weight.ndim != 2 or weight.shape[0] != trunk_raw.shape[-1]:
        raise ValueError("Depth-wise causal-conv weight must be [C,K] or [C,1,K].")
    if position_ids.shape != trunk_raw.shape[:2] or valid_mask.shape != position_ids.shape:
        raise ValueError("Causal-conv position/valid tensors must be [B,T].")

    batch_size, sequence_length, channels = trunk_raw.shape
    kernel_size = int(weight.shape[-1])
    valid_mask = valid_mask.to(device=trunk_raw.device, dtype=torch.bool)
    position_ids = position_ids.to(device=trunk_raw.device, dtype=torch.long)

    if bias is not None and bias.shape != (channels,):
        raise ValueError("Depth-wise causal-conv bias must be [C].")

    if single_segment_per_row is None:
        single_segment_per_row = _single_segment_per_row(position_ids, valid_mask)
    if single_segment_per_row:
        # Use the same grouped-convolution operation as the official Qwen3.5
        # trunk.  Besides being faster, this prevents a different BF16 summation
        # order from becoming recurrent-state drift in later DeltaNet layers.
        preactivation = F.conv1d(
            trunk_raw.transpose(1, 2),
            weight.unsqueeze(1),
            bias,
            padding=kernel_size - 1,
            groups=channels,
        )[:, :, :sequence_length].transpose(1, 2)
    else:
        preactivation = torch.zeros_like(trunk_raw)
        # Packed segments need an explicit boundary-aware fallback because an
        # ordinary convolution would cross from one segment into the next.
        for lag in range(kernel_size):
            coefficient = weight[:, kernel_size - 1 - lag].reshape(1, 1, channels)
            if lag == 0:
                source = trunk_raw
                allowed = valid_mask
            else:
                source = F.pad(trunk_raw[:, :-lag, :], (0, 0, lag, 0))
                allowed = torch.zeros_like(valid_mask)
                allowed[:, lag:] = (
                    valid_mask[:, lag:]
                    & valid_mask[:, :-lag]
                    & position_ids[:, lag:].eq(position_ids[:, :-lag] + lag)
                )
            preactivation = preactivation + source * coefficient * allowed[..., None]
        if bias is not None:
            preactivation = preactivation + bias.reshape(1, 1, channels)

    current_coefficient = weight[:, -1].reshape(1, 1, channels)
    candidate_preactivation = preactivation + (
        candidate_raw - trunk_raw
    ) * current_coefficient
    trunk_output = F.silu(preactivation)
    candidate_output = F.silu(candidate_preactivation)
    trunk_output = trunk_output.masked_fill(~valid_mask[..., None], 0.0)
    candidate_output = candidate_output.masked_fill(~valid_mask[..., None], 0.0)
    return trunk_output, candidate_output


def recurrent_three_stream_reference(
    trunk_query: torch.Tensor,
    trunk_key: torch.Tensor,
    trunk_value: torch.Tensor,
    trunk_log_decay: torch.Tensor,
    trunk_beta: torch.Tensor,
    full_query: torch.Tensor,
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    full_log_decay: torch.Tensor,
    full_beta: torch.Tensor,
    local_query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    local_log_decay: torch.Tensor,
    local_beta: torch.Tensor,
    position_ids: torch.LongTensor,
    valid_mask: torch.BoolTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slow FP32 definition of the fixed-trunk three-stream recurrence."""

    groups = (
        (trunk_query, trunk_key, trunk_value, trunk_log_decay, trunk_beta),
        (full_query, full_key, full_value, full_log_decay, full_beta),
        (local_query, local_key, local_value, local_log_decay, local_beta),
    )
    for values in groups:
        _validate_delta_inputs(*values)
    if any(values[0].shape != trunk_query.shape for values in groups[1:]):
        raise ValueError("All DeltaRule Q/K streams must have identical shapes.")
    if any(values[2].shape != trunk_value.shape for values in groups[1:]):
        raise ValueError("All DeltaRule value streams must have identical shapes.")
    if position_ids.shape != trunk_query.shape[:2] or valid_mask.shape != position_ids.shape:
        raise ValueError("DeltaRule position/valid tensors must be [B,T].")

    batch_size, sequence_length, num_heads, key_dim = trunk_query.shape
    value_dim = int(trunk_value.shape[-1])
    scale = key_dim**-0.5
    outputs = [
        torch.zeros(
            batch_size,
            sequence_length,
            num_heads,
            value_dim,
            dtype=torch.float32,
            device=trunk_query.device,
        )
        for _ in range(3)
    ]
    states = torch.zeros(
        batch_size,
        num_heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=trunk_query.device,
    )
    valid_mask = valid_mask.to(dtype=torch.bool, device=trunk_query.device)

    normalized = []
    for query, key, value, log_decay, beta in groups:
        normalized.append(
            (
                l2_normalize_qk(query).float(),
                l2_normalize_qk(key).float(),
                value.float(),
                log_decay.float(),
                beta.float(),
            )
        )

    for index in range(sequence_length):
        reset = valid_mask[:, index] & position_ids[:, index].eq(0)
        states = torch.where(reset[:, None, None, None], torch.zeros_like(states), states)
        active = valid_mask[:, index]
        trunk_previous = states
        candidate_states = []
        for query, key, value, log_decay, beta in normalized:
            q_i = query[:, index]
            k_i = key[:, index]
            v_i = value[:, index]
            decay_i = log_decay[:, index].exp()[..., None, None]
            beta_i = beta[:, index][..., None]
            decayed = trunk_previous * decay_i
            recalled = (decayed * k_i[..., None]).sum(dim=-2)
            delta = (v_i - recalled) * beta_i
            candidate_state = decayed + k_i[..., None] * delta[..., None, :]
            candidate_states.append(candidate_state)
            stream_index = len(candidate_states) - 1
            result = (candidate_state * q_i[..., None]).sum(dim=-2) * scale
            outputs[stream_index][:, index] = torch.where(
                active[:, None, None], result, torch.zeros_like(result)
            )
        states = torch.where(
            active[:, None, None, None], candidate_states[0], states
        )

    return tuple(output.to(trunk_query.dtype) for output in outputs)


def _single_segment_per_row(
    position_ids: torch.LongTensor, valid_mask: torch.BoolTensor
) -> bool:
    """Return whether each row is one zero-based segment plus right padding."""

    if position_ids.shape != valid_mask.shape or position_ids.ndim != 2:
        raise ValueError("Position IDs and valid mask must be aligned [B,T] tensors.")
    sequence_length = position_ids.shape[1]
    expected = torch.arange(sequence_length, device=position_ids.device).unsqueeze(0)
    prefix_mask = expected < valid_mask.long().sum(dim=1, keepdim=True)
    layout_is_single = valid_mask.eq(prefix_mask).all() & (
        (~valid_mask) | position_ids.eq(expected)
    ).all()
    # One scalar synchronization per model forward is intentional.  Callers
    # pass the result through every DeltaNet layer so the 16K formal path does
    # not copy full position tensors to CPU dozens of times.
    return bool(layout_is_single.item())


def _run_delta_rule(
    rule: DeltaRule,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    position_ids: torch.LongTensor,
    valid_mask: torch.BoolTensor,
    single_segment_per_row: Optional[bool] = None,
) -> torch.Tensor:
    """Run a DeltaRule scan, resetting state at every packed segment."""

    def call(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = rule(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g=g.contiguous(),
            beta=b.contiguous(),
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )
        return output

    if single_segment_per_row is None:
        single_segment_per_row = _single_segment_per_row(position_ids, valid_mask)
    if single_segment_per_row:
        return call(query, key, value, log_decay, beta).masked_fill(
            ~valid_mask[..., None, None], 0.0
        )

    # Packed segments are uncommon in the formal padded training protocol but
    # remain part of the correctness contract.  The slow path keeps their
    # recurrent states strictly isolated instead of silently mixing segments.
    output = torch.zeros(
        (*query.shape[:3], value.shape[-1]), dtype=query.dtype, device=query.device
    )
    for row in range(query.shape[0]):
        index = 0
        while index < query.shape[1]:
            if not bool(valid_mask[row, index]):
                index += 1
                continue
            if int(position_ids[row, index]) != 0:
                raise ValueError("Every packed DeltaNet segment must start at position zero.")
            end = index + 1
            while (
                end < query.shape[1]
                and bool(valid_mask[row, end])
                and int(position_ids[row, end]) == int(position_ids[row, end - 1]) + 1
            ):
                end += 1
            output[row : row + 1, index:end] = call(
                query[row : row + 1, index:end],
                key[row : row + 1, index:end],
                value[row : row + 1, index:end],
                log_decay[row : row + 1, index:end],
                beta[row : row + 1, index:end],
            )
            index = end
    return output


def _candidate_core_from_previous_trunk_state(
    rule: DeltaRule,
    trunk_key: torch.Tensor,
    trunk_value: torch.Tensor,
    trunk_log_decay: torch.Tensor,
    trunk_beta: torch.Tensor,
    candidate_query: torch.Tensor,
    candidate_key: torch.Tensor,
    candidate_value: torch.Tensor,
    candidate_log_decay: torch.Tensor,
    candidate_beta: torch.Tensor,
    position_ids: torch.LongTensor,
    valid_mask: torch.BoolTensor,
    single_segment_per_row: Optional[bool] = None,
) -> torch.Tensor:
    """Query S_trunk(i-1), then add the isolated candidate self update."""

    key_dim = int(candidate_query.shape[-1])
    scale = key_dim**-0.5
    dot_qk = (candidate_query.float() * candidate_key.float()).sum(
        dim=-1, keepdim=True
    )
    decay = candidate_log_decay.float().exp()[..., None]
    beta = candidate_beta.float()[..., None]

    # If S is the trunk state after i-1 and the candidate update is
    # S_c = a_c (I - beta_c k_c k_c^T) S + beta_c k_c v_c^T,
    # the first term can be queried with
    # r_c = a_c (q_c - beta_c k_c (q_c^T k_c)).
    previous_state_query = decay * (
        candidate_query.float() - beta * candidate_key.float() * dot_qk
    )
    previous_state_query = previous_state_query.to(candidate_query.dtype)

    # A normal DeltaRule output at index j queries S_trunk(j).  Put r_i at
    # index i-1, run the trunk transition scan, then shift its output right.
    shifted_query = torch.zeros_like(previous_state_query)
    continuation = torch.zeros_like(valid_mask)
    if candidate_query.shape[1] > 1:
        continuation[:, 1:] = (
            valid_mask[:, 1:]
            & valid_mask[:, :-1]
            & position_ids[:, 1:].eq(position_ids[:, :-1] + 1)
        )
        shifted_query[:, :-1] = torch.where(
            continuation[:, 1:, None, None],
            previous_state_query[:, 1:],
            torch.zeros_like(previous_state_query[:, 1:]),
        )
    queried = _run_delta_rule(
        rule,
        shifted_query,
        trunk_key,
        trunk_value,
        trunk_log_decay,
        trunk_beta,
        position_ids,
        valid_mask,
        single_segment_per_row,
    )
    previous_term = torch.zeros_like(queried)
    if candidate_query.shape[1] > 1:
        previous_term[:, 1:] = torch.where(
            continuation[:, 1:, None, None],
            queried[:, :-1],
            torch.zeros_like(queried[:, :-1]),
        )

    self_term = (
        beta * dot_qk * candidate_value.float() * scale
    ).to(previous_term.dtype)
    return (previous_term + self_term).masked_fill(
        ~valid_mask[..., None, None], 0.0
    )


def three_stream_from_trunk_scan(
    rule: DeltaRule,
    trunk_query: torch.Tensor,
    trunk_key: torch.Tensor,
    trunk_value: torch.Tensor,
    trunk_log_decay: torch.Tensor,
    trunk_beta: torch.Tensor,
    full_query: torch.Tensor,
    full_key: torch.Tensor,
    full_value: torch.Tensor,
    full_log_decay: torch.Tensor,
    full_beta: torch.Tensor,
    local_query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    local_log_decay: torch.Tensor,
    local_beta: torch.Tensor,
    position_ids: torch.LongTensor,
    valid_mask: torch.BoolTensor,
    single_segment_per_row: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact parallel three-stream outputs using only ordinary trunk scans."""

    groups = (
        (trunk_query, trunk_key, trunk_value, trunk_log_decay, trunk_beta),
        (full_query, full_key, full_value, full_log_decay, full_beta),
        (local_query, local_key, local_value, local_log_decay, local_beta),
    )
    for values in groups:
        _validate_delta_inputs(*values)
    if position_ids.shape != trunk_query.shape[:2] or valid_mask.shape != position_ids.shape:
        raise ValueError("DeltaRule position/valid tensors must be [B,T].")

    normalized = []
    for query, key, value, log_decay, beta in groups:
        normalized.append(
            (
                l2_normalize_qk(query),
                l2_normalize_qk(key),
                value,
                log_decay,
                beta,
            )
        )
    trunk_query_n, trunk_key_n, trunk_value, trunk_log_decay, trunk_beta = normalized[0]
    trunk_output = _run_delta_rule(
        rule,
        trunk_query_n,
        trunk_key_n,
        trunk_value,
        trunk_log_decay,
        trunk_beta,
        position_ids,
        valid_mask,
        single_segment_per_row,
    )
    full_output = _candidate_core_from_previous_trunk_state(
        rule,
        trunk_key_n,
        trunk_value,
        trunk_log_decay,
        trunk_beta,
        *normalized[1],
        position_ids,
        valid_mask,
        single_segment_per_row,
    )
    local_output = _candidate_core_from_previous_trunk_state(
        rule,
        trunk_key_n,
        trunk_value,
        trunk_log_decay,
        trunk_beta,
        *normalized[2],
        position_ids,
        valid_mask,
        single_segment_per_row,
    )
    return trunk_output, full_output, local_output


__all__ = [
    "_single_segment_per_row",
    "isolated_causal_conv1d",
    "l2_normalize_qk",
    "recurrent_three_stream_reference",
    "three_stream_from_trunk_scan",
]
