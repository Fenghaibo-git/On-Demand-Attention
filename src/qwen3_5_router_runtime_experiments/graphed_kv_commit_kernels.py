"""Bit-preserving BF16 slot writes with per-layer paged-cache strides."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['position'])
def selected_commit_all(TABLE, SLOTS, position,
                        WRITE_FULL: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                        HEAD_DIM: tl.constexpr, ELEMENTS: tl.constexpr,
                        CAPACITY: tl.constexpr, SINKS: tl.constexpr, RECENT: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = TABLE + tl.program_id(0) * 10
    key = tl.load(row + 0).to(tl.pointer_type(tl.bfloat16))
    value = tl.load(row + 1).to(tl.pointer_type(tl.bfloat16))
    local_key = tl.load(row + 2).to(tl.pointer_type(tl.bfloat16))
    local_value = tl.load(row + 3).to(tl.pointer_type(tl.bfloat16))
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = i < ELEMENTS
    k = tl.load(key + i, valid, other=0)
    v = tl.load(value + i, valid, other=0)
    ring_slot = tl.where(position < SINKS, position, SINKS + position % RECENT)
    ring_offset = (i // HEAD_DIM * CAPACITY + ring_slot) * HEAD_DIM + i % HEAD_DIM
    tl.store(local_key + ring_offset, k, valid)
    tl.store(local_value + ring_offset, v, valid)
    if WRITE_FULL:
        paged = tl.load(row + 4).to(tl.pointer_type(tl.bfloat16))
        stride_kv = tl.load(row + 5)
        stride_block = tl.load(row + 6)
        stride_token = tl.load(row + 7)
        stride_head = tl.load(row + 8)
        stride_dim = tl.load(row + 9)
        slot = tl.load(SLOTS)
        offset = (slot // BLOCK_SIZE * stride_block + slot % BLOCK_SIZE * stride_token
                  + i // HEAD_DIM * stride_head + i % HEAD_DIM * stride_dim)
        tl.store(paged + offset, k, valid & (slot >= 0))
        tl.store(paged + stride_kv + offset, v, valid & (slot >= 0))
