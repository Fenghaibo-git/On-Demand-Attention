"""Direct-ring single-token attention and fused transactional state copies.

No GPU work occurs on import. TRITON_INTERPRET=1 executes these same kernels
on CPU for logic validation. Device performance and BF16 model qualification
require separate GPU gates. This module does not replace frozen experiments.
"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["length"])
def ring_partials(
    query, keys, values, current_key, current_value, partial, maxima, sums,
    length, HEAD_DIM: tl.constexpr, KV_HEADS: tl.constexpr,
    QUERY_HEADS: tl.constexpr, SINK: tl.constexpr, RECENT: tl.constexpr,
    SPLITS: tl.constexpr, SCALE: tl.constexpr,
    TILE: tl.constexpr, DIM: tl.constexpr,
):
    head = tl.program_id(0)
    split = tl.program_id(1)
    kv_head = head // (QUERY_HEADS // KV_HEADS)
    t = split * TILE + tl.arange(0, TILE)
    d = tl.arange(0, DIM)
    sinks = tl.minimum(length, SINK)
    recent = tl.minimum(tl.maximum(length - SINK, 0), RECENT)
    past = sinks + recent
    if RECENT > 0:
        absolute = tl.maximum(SINK, length - RECENT) + t - sinks
        slot = tl.where(t < sinks, t, SINK + absolute % RECENT)
    else:
        slot = t
    offset = (kv_head * (SINK + RECENT) + slot[:, None]) * HEAD_DIM + d[None, :]
    valid_past = (t[:, None] < past) & (d[None, :] < HEAD_DIM)
    k = tl.load(keys + offset, valid_past, other=0).to(tl.float32)
    ck = tl.load(current_key + kv_head * HEAD_DIM + d, d < HEAD_DIM, other=0).to(tl.float32)
    k = tl.where(t[:, None] == past, ck[None, :], k)
    q = tl.load(query + head * HEAD_DIM + d, d < HEAD_DIM, other=0).to(tl.float32)
    logits = tl.sum(k * q[None, :], 1) * SCALE
    logits = tl.where(t <= past, logits, -float("inf"))
    maximum = tl.max(logits, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(logits - safe_maximum)
    normalizer = tl.sum(weights, 0)
    # K is dead before V becomes live. Keeping both FP32 tiles live spills
    # registers on sm80, even though the program compiles and is numerically OK.
    v = tl.load(values + offset, valid_past, other=0).to(tl.float32)
    cv = tl.load(current_value + kv_head * HEAD_DIM + d, d < HEAD_DIM, other=0).to(tl.float32)
    v = tl.where(t[:, None] == past, cv[None, :], v)
    numerator = tl.sum(weights[:, None] * v, 0)
    tl.store(partial + (head * SPLITS + split) * HEAD_DIM + d, numerator, d < HEAD_DIM)
    tl.store(maxima + head * SPLITS + split, maximum)
    tl.store(sums + head * SPLITS + split, normalizer)


@triton.jit
def ring_reduce(
    partial, maxima, sums, output, HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr, SPLIT_BLOCK: tl.constexpr, DIM: tl.constexpr,
):
    head = tl.program_id(0)
    s = tl.arange(0, SPLIT_BLOCK)
    # Split output dimensions as well: a 129-way split must not materialize
    # the entire [256, head_dim] partial matrix in one CTA's registers.
    d = tl.program_id(1) * DIM + tl.arange(0, DIM)
    m = tl.load(maxima + head * SPLITS + s, s < SPLITS, other=-float("inf"))
    l = tl.load(sums + head * SPLITS + s, s < SPLITS, other=0.0)
    m_all = tl.max(m, 0)
    weights = tl.exp(m - m_all)
    z = tl.load(
        partial + (head * SPLITS + s[:, None]) * HEAD_DIM + d[None, :],
        (s[:, None] < SPLITS) & (d[None, :] < HEAD_DIM), other=0.0,
    )
    result = tl.sum(z * weights[:, None], 0) / tl.sum(l * weights, 0)
    tl.store(output + head * HEAD_DIM + d, result, d < HEAD_DIM)


@triton.jit(do_not_specialize=["slot"])
def ring_commit(keys, values, current_key, current_value, slot,
                CAPACITY: tl.constexpr, HEAD_DIM: tl.constexpr,
                ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k = tl.load(current_key + i, i < ELEMENTS, other=0)
    v = tl.load(current_value + i, i < ELEMENTS, other=0)
    offset = (i // HEAD_DIM * CAPACITY + slot) * HEAD_DIM + i % HEAD_DIM
    tl.store(keys + offset, k, i < ELEMENTS)
    tl.store(values + offset, v, i < ELEMENTS)


@triton.jit
def state_copy(
    sources, indices, backups, SHAPES: tl.constexpr, STRIDES: tl.constexpr,
    BLOCK_STARTS: tl.constexpr, RESTORE: tl.constexpr, BLOCK: tl.constexpr,
):
    """One launch for all conv/recurrent tensors, including strided rows.

    A tuple of direct pointers avoids a device pointer table. The state index
    remains on device. No index_select allocation, extra clone, or host read.
    """
    program = tl.program_id(0)
    for i in tl.static_range(len(sources)):
        if (program >= BLOCK_STARTS[i]) & (program < BLOCK_STARTS[i + 1]):
            x = (program - BLOCK_STARTS[i]) * BLOCK + tl.arange(0, BLOCK)
            elements: tl.constexpr = 1
            for dim in tl.static_range(1, len(SHAPES[i])):
                elements *= SHAPES[i][dim]
            state = tl.load(indices[i]).to(tl.int64)
            tl.device_assert((state >= 0) & (state < SHAPES[i][0]), "GDN state index out of bounds")
            offset = state * STRIDES[i][0] + tl.full((BLOCK,), 0, tl.int64)
            trailing: tl.constexpr = 1
            for dim in tl.static_range(len(SHAPES[i]) - 1, 0, -1):
                coordinate = (x // trailing) % SHAPES[i][dim]
                offset += coordinate * STRIDES[i][dim]
                trailing *= SHAPES[i][dim]
            valid = (x < elements) & (state >= 0) & (state < SHAPES[i][0])
            if RESTORE:
                value = tl.load(backups[i] + x, valid, other=0)
                tl.store(sources[i] + offset, value, valid)
            else:
                value = tl.load(sources[i] + offset, valid, other=0)
                tl.store(backups[i] + x, value, valid)
