"""Bounded selected-history KV, independent of the Full KV allocation.

Keys are already position-encoded. Ring order is immaterial to one-token
attention, provided keys, values and validity are permuted together.
The ring retains W-1 past tokens: the current candidate is supplied separately.
"""

from __future__ import annotations

import torch


class CompactSelectedKV:
    def __init__(self, sink_size: int, window_size: int):
        if sink_size < 0 or window_size < 1:
            raise ValueError("Require sink_size >= 0 and window_size >= 1")
        self.sink_size = int(sink_size)
        self.recent_size = int(window_size) - 1
        self.capacity = self.sink_size + self.recent_size
        self.length = 0
        self.keys = self.values = self.positions = self.starts = None

    @torch.no_grad()
    def initialize(self, keys, values, starts):
        if keys.ndim != 4 or keys.shape != values.shape or keys.shape[2] < 1:
            raise ValueError("Prefill must provide aligned nonempty [B,H,T,D] KV")
        if keys.dtype != values.dtype or keys.device != values.device:
            raise ValueError("K/V dtype and device must agree")
        batch, heads, length, dim = keys.shape
        self.starts = torch.as_tensor(starts, dtype=torch.long, device=keys.device)
        if self.starts.shape != (batch,) or bool(
            ((self.starts < 0) | (self.starts >= length)).any()
        ):
            raise ValueError("Each left-padded request must contain a valid token")
        shape = (batch, heads, self.capacity, dim)
        self.keys = keys.new_zeros(shape)
        self.values = values.new_zeros(shape)
        self.positions = torch.full(
            (batch, self.capacity), -1, device=keys.device, dtype=torch.long
        )
        slots = torch.arange(self.sink_size, device=keys.device)
        sink_positions = self.starts[:, None] + slots
        valid_sink = sink_positions < length
        self._seed_slots(keys, values, slots, sink_positions, valid_sink)
        if self.recent_size:
            absolute = torch.arange(max(0, length - self.recent_size), length, device=keys.device)
            ring_slots = self.sink_size + absolute.remainder(self.recent_size)
            positions = absolute[None].expand(batch, -1)
            valid = positions >= (self.starts[:, None] + self.sink_size)
            self._seed_slots(keys, values, ring_slots, positions, valid)
        self.length = int(length)

    def _seed_slots(self, keys, values, slots, positions, valid):
        if slots.numel() == 0:
            return
        # Advanced indexing touches only selected positions, not the complete
        # Full allocation. Initialization runs once after Full prefill.
        rows = torch.arange(keys.shape[0], device=keys.device)[:, None]
        safe = positions.clamp(0, keys.shape[2] - 1)
        k = keys[rows, :, safe].permute(0, 2, 1, 3)
        v = values[rows, :, safe].permute(0, 2, 1, 3)
        self.keys[:, :, slots] = torch.where(valid[:, None, :, None], k, 0)
        self.values[:, :, slots] = torch.where(valid[:, None, :, None], v, 0)
        self.positions[:, slots] = torch.where(valid, positions, -1)

    @torch.no_grad()
    def append_selected(self, keys, values, position: int):
        if self.keys is None or int(position) != self.length:
            raise ValueError("Only the next committed selected token may enter Local KV")
        expected = (*self.keys.shape[:2], 1, self.keys.shape[-1])
        if keys.shape != expected or values.shape != expected:
            raise ValueError("Committed KV must contain exactly one token for every row")
        if (
            keys.dtype != self.keys.dtype
            or values.dtype != self.values.dtype
            or keys.device != self.keys.device
            or values.device != self.values.device
        ):
            raise ValueError("Committed KV dtype/device changed")
        relative = int(position) - self.starts
        sink_rows = ((relative >= 0) & (relative < self.sink_size)).nonzero().flatten()
        if sink_rows.numel():
            slots = relative.index_select(0, sink_rows)
            self.keys[sink_rows, :, slots] = keys[sink_rows, :, 0]
            self.values[sink_rows, :, slots] = values[sink_rows, :, 0]
            self.positions[sink_rows, slots] = int(position)
        if self.recent_size:
            slot = self.sink_size + int(position) % self.recent_size
            valid = relative >= self.sink_size
            self.keys[:, :, slot] = torch.where(valid[:, None, None], keys[:, :, 0], 0)
            self.values[:, :, slot] = torch.where(valid[:, None, None], values[:, :, 0], 0)
            self.positions[:, slot] = torch.where(valid, int(position), -1)
        self.length += 1

    def candidate(self, row_indices, current_key, current_value, chronological=False):
        if self.keys is None:
            raise RuntimeError("Full prefill must initialize the Local cache")
        if row_indices.ndim != 1 or row_indices.dtype != torch.long:
            raise ValueError("Require one-dimensional long row indices")
        expected = (row_indices.numel(), self.keys.shape[1], 1, self.keys.shape[-1])
        if current_key.shape != expected or current_value.shape != expected:
            raise ValueError("Candidate must contain one aligned KV per selected row")
        # No Full KV is read here. Persistent storage is never mutated by a
        # Local or Full candidate; only append_selected publishes a token.
        past_key = self.keys.index_select(0, row_indices)
        past_value = self.values.index_select(0, row_indices)
        valid = self.positions.index_select(0, row_indices).ge(0)
        if chronological and self.recent_size:
            # Rotate only the bounded ring view. This preserves the baseline
            # reduction order without ever gathering from Full KV.
            start = max(0, self.length - self.recent_size) % self.recent_size
            pivot = self.sink_size + start
            key = torch.cat((past_key[:, :, :self.sink_size], past_key[:, :, pivot:],
                             past_key[:, :, self.sink_size:pivot], current_key), dim=2)
            value = torch.cat((past_value[:, :, :self.sink_size], past_value[:, :, pivot:],
                               past_value[:, :, self.sink_size:pivot], current_value), dim=2)
            valid = torch.cat((valid[:, :self.sink_size], valid[:, pivot:],
                               valid[:, self.sink_size:pivot]), dim=1)
        else:
            key = torch.cat((past_key, current_key), dim=2)
            value = torch.cat((past_value, current_value), dim=2)
        valid = torch.cat((valid, valid.new_ones((valid.shape[0], 1))), dim=1)
        return key, value, valid[:, None, None, :]

    def storage_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.keys, self.values, self.positions, self.starts)
            if t is not None
        )
